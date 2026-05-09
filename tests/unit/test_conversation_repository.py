"""Tests for the InMemoryConversationRepository.

Covers session lifecycle, turn ordering, vector search, and feedback.
The Neo4j implementation is exercised indirectly via integration tests
once a live driver is available; here we focus on the in-memory store
that backs unit tests and dev runs.
"""

from __future__ import annotations

import os

import pytest

# Make sure config validation passes in test runs.
os.environ.setdefault("LLM_API_KEY", "not-configured")
os.environ.setdefault("NEO4J_URI", "bolt://localhost:7687")
os.environ.setdefault("NEO4J_USER", "neo4j")
os.environ.setdefault("NEO4J_PASSWORD", "password")

from graphbuilder.domain.models.conversation_models import (  # noqa: E402
    ConversationSession,
    ConversationTurn,
)
from graphbuilder.infrastructure.config.settings import GraphBuilderConfig  # noqa: E402
from graphbuilder.infrastructure.repositories.conversation_repository import (  # noqa: E402
    InMemoryConversationRepository,
    create_conversation_repository,
)


@pytest.fixture
def repo() -> InMemoryConversationRepository:
    cfg = GraphBuilderConfig()
    return InMemoryConversationRepository(cfg)


def _turn(session_id: str, idx: int, query: str, *, embedding: list[float] | None = None) -> ConversationTurn:
    return ConversationTurn(
        session_id=session_id,
        idx=idx,
        user_query=query,
        llm_answer=f"answer to {query}",
        query_embedding=embedding,
    )


# ---------------------------------------------------------------- sessions


async def test_create_session_persists(repo):
    s = ConversationSession(user_id="alice", title="t1")
    saved = await repo.create_session(s)
    assert saved.id == s.id
    fetched = await repo.get_session(s.id)
    assert fetched is not None
    assert fetched.user_id == "alice"


async def test_list_sessions_orders_by_recent_activity(repo):
    a = await repo.create_session(ConversationSession(user_id="u1", title="A"))
    b = await repo.create_session(ConversationSession(user_id="u1", title="B"))
    # Bump B's last_active_at by appending a turn
    await repo.append_turn(_turn(b.id, 0, "hi"))
    sessions = await repo.list_sessions(user_id="u1", limit=10)
    assert [s.id for s in sessions][0] == b.id
    assert {s.id for s in sessions} == {a.id, b.id}


async def test_list_sessions_filters_by_user(repo):
    s1 = await repo.create_session(ConversationSession(user_id="u1"))
    s2 = await repo.create_session(ConversationSession(user_id="u2"))
    anon = await repo.create_session(ConversationSession(user_id=None))

    only_u1 = await repo.list_sessions(user_id="u1")
    assert {s.id for s in only_u1} == {s1.id}

    only_anon = await repo.list_sessions(user_id=None)
    assert {s.id for s in only_anon} == {anon.id}


async def test_delete_session_removes_session_and_turns(repo):
    s = await repo.create_session(ConversationSession())
    await repo.append_turn(_turn(s.id, 0, "q1"))
    await repo.append_turn(_turn(s.id, 1, "q2"))

    assert await repo.delete_session(s.id) is True
    assert await repo.get_session(s.id) is None
    assert await repo.get_turns_by_session(s.id) == []


async def test_delete_session_unknown_id_returns_false(repo):
    assert await repo.delete_session("nope") is False


async def test_update_session_summary_and_title(repo):
    s = await repo.create_session(ConversationSession())
    await repo.update_session_summary(s.id, "rolling summary v1")
    await repo.update_session_title(s.id, "renamed")
    fetched = await repo.get_session(s.id)
    assert fetched.summary == "rolling summary v1"
    assert fetched.title == "renamed"


# ---------------------------------------------------------------- turns


async def test_append_turn_bumps_session_counter_and_activity(repo):
    s = await repo.create_session(ConversationSession())
    initial_active = s.last_active_at
    await repo.append_turn(_turn(s.id, 0, "first"))
    await repo.append_turn(_turn(s.id, 1, "second"))

    fetched = await repo.get_session(s.id)
    assert fetched.turn_count == 2
    assert fetched.last_active_at >= initial_active


async def test_get_turns_by_session_returns_oldest_first(repo):
    s = await repo.create_session(ConversationSession())
    # Insert out-of-order to confirm ordering is by idx, not insertion time
    await repo.append_turn(_turn(s.id, 2, "third"))
    await repo.append_turn(_turn(s.id, 0, "first"))
    await repo.append_turn(_turn(s.id, 1, "second"))

    turns = await repo.get_turns_by_session(s.id)
    assert [t.user_query for t in turns] == ["first", "second", "third"]


async def test_get_turn_round_trip(repo):
    s = await repo.create_session(ConversationSession())
    t = await repo.append_turn(_turn(s.id, 0, "hello"))
    fetched = await repo.get_turn(t.id)
    assert fetched is not None
    assert fetched.user_query == "hello"


# ---------------------------------------------------------------- vector recall


async def test_vector_search_returns_most_similar_turn_in_session(repo):
    s = await repo.create_session(ConversationSession())
    # Three turns with deliberately different vectors.
    await repo.append_turn(_turn(s.id, 0, "about kinases", embedding=[1.0, 0.0, 0.0]))
    await repo.append_turn(_turn(s.id, 1, "about diseases", embedding=[0.0, 1.0, 0.0]))
    await repo.append_turn(_turn(s.id, 2, "about pathways", embedding=[0.0, 0.0, 1.0]))

    # Query matches the first turn closely.
    hits = await repo.vector_search_turns(
        query_embedding=[0.95, 0.05, 0.0],
        top_k=2,
        min_score=0.5,
        session_id=s.id,
    )
    assert hits, "expected at least one hit"
    top_turn, top_score = hits[0]
    assert top_turn.user_query == "about kinases"
    assert top_score > 0.9


async def test_vector_search_respects_session_filter(repo):
    s1 = await repo.create_session(ConversationSession())
    s2 = await repo.create_session(ConversationSession())
    await repo.append_turn(_turn(s1.id, 0, "in s1", embedding=[1.0, 0.0]))
    await repo.append_turn(_turn(s2.id, 0, "in s2", embedding=[1.0, 0.0]))

    only_s1 = await repo.vector_search_turns(
        query_embedding=[1.0, 0.0], top_k=5, min_score=0.0, session_id=s1.id,
    )
    assert {t.user_query for t, _ in only_s1} == {"in s1"}

    cross = await repo.vector_search_turns(
        query_embedding=[1.0, 0.0], top_k=5, min_score=0.0, session_id=None,
    )
    assert {t.user_query for t, _ in cross} == {"in s1", "in s2"}


async def test_vector_search_min_score_filters_dissimilar(repo):
    s = await repo.create_session(ConversationSession())
    await repo.append_turn(_turn(s.id, 0, "orthogonal", embedding=[1.0, 0.0]))
    hits = await repo.vector_search_turns(
        query_embedding=[0.0, 1.0], top_k=5, min_score=0.5, session_id=s.id,
    )
    assert hits == []


async def test_vector_search_empty_query_returns_empty(repo):
    s = await repo.create_session(ConversationSession())
    await repo.append_turn(_turn(s.id, 0, "anything", embedding=[1.0]))
    assert await repo.vector_search_turns(query_embedding=[], top_k=5) == []


# ---------------------------------------------------------------- feedback


async def test_record_feedback_updates_turn(repo):
    s = await repo.create_session(ConversationSession())
    t = await repo.append_turn(_turn(s.id, 0, "q"))
    assert await repo.record_feedback(t.id, rating=1, comment="nice") is True
    fetched = await repo.get_turn(t.id)
    assert fetched.feedback_rating == 1
    assert fetched.feedback_comment == "nice"


async def test_record_feedback_unknown_turn_returns_false(repo):
    assert await repo.record_feedback("missing", rating=-1) is False


# ---------------------------------------------------------------- factory


def test_factory_returns_in_memory_for_non_neo4j_provider(monkeypatch):
    # Default provider is whatever the env says; force it off.
    cfg = GraphBuilderConfig()
    monkeypatch.setattr(cfg.database, "provider", "memory")
    repo = create_conversation_repository(cfg, neo4j_driver=None)
    assert isinstance(repo, InMemoryConversationRepository)


def test_factory_returns_in_memory_when_no_driver_even_for_neo4j_provider(monkeypatch):
    cfg = GraphBuilderConfig()
    monkeypatch.setattr(cfg.database, "provider", "neo4j")
    repo = create_conversation_repository(cfg, neo4j_driver=None)
    assert isinstance(repo, InMemoryConversationRepository)


# ---------------------------------------------------------------- model serialisation


def test_session_round_trips_through_dict():
    s = ConversationSession(user_id="alice", title="t", summary="s", turn_count=3)
    s2 = ConversationSession.from_dict(s.to_dict())
    assert s2.id == s.id
    assert s2.user_id == "alice"
    assert s2.title == "t"
    assert s2.summary == "s"
    assert s2.turn_count == 3


def test_turn_round_trips_through_dict():
    t = ConversationTurn(
        session_id="s_1",
        idx=4,
        user_query="q",
        llm_answer="a",
        cited_entity_ids=["e1", "e2"],
        cited_chunk_ids=["c1"],
        prompt_tokens=100,
        completion_tokens=50,
        latency_ms=1234,
    )
    t2 = ConversationTurn.from_dict(t.to_dict())
    assert t2.session_id == "s_1"
    assert t2.idx == 4
    assert t2.cited_entity_ids == ["e1", "e2"]
    assert t2.cited_chunk_ids == ["c1"]
    assert t2.prompt_tokens == 100
    assert t2.latency_ms == 1234
