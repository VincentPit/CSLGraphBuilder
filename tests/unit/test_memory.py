"""Tests for the conversation memory layers (P6+P7 of docs/RAG_QA_PLAN.md).

Covers:
- Working memory window (last N turns rendered into the prompt)
- Rolling summary regeneration + freshness marker
- LLM-driven vs deterministic-fallback summary
- Episodic recall via vector_search_turns
- End-to-end QAService.ask() with memory + persisted query embeddings
- Empty / single-turn / no-LLM corner cases
"""

from __future__ import annotations

import asyncio
import os
from typing import List, Optional

import pytest

os.environ.setdefault("LLM_API_KEY", "not-configured")

from graphbuilder.core.retrieval.memory import (  # noqa: E402
    MemoryConfig,
    MemoryContext,
    MemoryService,
    _HEADER_EPISODIC,
    _HEADER_RECENT,
    _HEADER_SUMMARY,
)
from graphbuilder.core.retrieval.models import (  # noqa: E402
    Channel,
    ChannelResult,
    ItemKind,
    RetrievalTrace,
    RetrievedItem,
)
from graphbuilder.core.retrieval.qa_service import QAService  # noqa: E402
from graphbuilder.domain.models.conversation_models import (  # noqa: E402
    ConversationSession,
    ConversationTurn,
)
from graphbuilder.infrastructure.config.settings import GraphBuilderConfig  # noqa: E402
from graphbuilder.infrastructure.repositories.conversation_repository import (  # noqa: E402
    InMemoryConversationRepository,
)


# ---------------------------------------------------------------- helpers


def _turn(session_id: str, idx: int, q: str, a: str, *, embedding: Optional[List[float]] = None) -> ConversationTurn:
    return ConversationTurn(
        session_id=session_id, idx=idx, user_query=q, llm_answer=a,
        query_embedding=embedding,
    )


@pytest.fixture
def repo() -> InMemoryConversationRepository:
    return InMemoryConversationRepository(GraphBuilderConfig())


class _FakeLLM:
    """Stub the summariser. Records calls so tests can assert it was used."""

    def __init__(self, response: str = "summary text"):
        self._response = response
        self.calls: list[dict] = []

    async def generate_text(self, *, prompt, system_prompt, temperature, max_tokens):
        self.calls.append({"prompt": prompt, "system_prompt": system_prompt})
        return self._response


# ---------------------------------------------------------------- working window


async def test_build_returns_empty_for_unknown_session(repo):
    svc = MemoryService(conversation_repo=repo, llm_service=None)
    ctx = await svc.build(session_id=None, query="q", query_embedding=None)
    assert ctx.working_turns == []
    assert ctx.rendered_block == ""


async def test_build_returns_empty_for_session_with_no_turns(repo):
    s = await repo.create_session(ConversationSession())
    svc = MemoryService(conversation_repo=repo, llm_service=None)
    ctx = await svc.build(session_id=s.id, query="q", query_embedding=None)
    assert ctx.working_turns == []
    assert ctx.rendered_block == ""


async def test_working_window_keeps_last_n_turns_when_below_threshold(repo):
    s = await repo.create_session(ConversationSession())
    for i, q in enumerate(["q1", "q2"]):
        await repo.append_turn(_turn(s.id, i, q, f"a{i + 1}"))

    svc = MemoryService(
        conversation_repo=repo, llm_service=None,
        config=MemoryConfig(working_memory_turns=3),
    )
    ctx = await svc.build(session_id=s.id, query="q3", query_embedding=None)
    assert [t.user_query for t in ctx.working_turns] == ["q1", "q2"]
    assert _HEADER_RECENT in ctx.rendered_block
    assert _HEADER_SUMMARY not in ctx.rendered_block


async def test_working_window_only_keeps_n_most_recent_turns(repo):
    s = await repo.create_session(ConversationSession())
    for i, q in enumerate(["q1", "q2", "q3", "q4", "q5"]):
        await repo.append_turn(_turn(s.id, i, q, f"a{i + 1}"))

    svc = MemoryService(
        conversation_repo=repo, llm_service=None,
        config=MemoryConfig(working_memory_turns=2),
    )
    ctx = await svc.build(session_id=s.id, query="q6", query_embedding=None)
    assert [t.user_query for t in ctx.working_turns] == ["q4", "q5"]
    assert "q5" in ctx.rendered_block
    assert "q4" in ctx.rendered_block
    # Older turns should NOT appear verbatim — they live in the summary.
    assert "USER: q1" not in ctx.rendered_block


async def test_working_window_drops_oldest_when_byte_budget_exceeded(repo):
    s = await repo.create_session(ConversationSession())
    big = "X" * 2000
    await repo.append_turn(_turn(s.id, 0, big, big))
    await repo.append_turn(_turn(s.id, 1, "q2", "short"))

    svc = MemoryService(
        conversation_repo=repo, llm_service=None,
        config=MemoryConfig(working_memory_turns=2, max_working_chars=500, enable_summary=False),
    )
    ctx = await svc.build(session_id=s.id, query="next", query_embedding=None)
    assert "q2" in ctx.rendered_block
    # The big turn was trimmed away to fit the byte budget.
    assert big not in ctx.rendered_block


# ---------------------------------------------------------------- rolling summary


async def test_summary_regenerated_when_older_turns_exist(repo):
    s = await repo.create_session(ConversationSession())
    for i, q in enumerate(["a", "b", "c", "d", "e"]):
        await repo.append_turn(_turn(s.id, i, q, f"ans-{q}"))

    llm = _FakeLLM(response="They asked about a, b, c.")
    svc = MemoryService(
        conversation_repo=repo, llm_service=llm,
        config=MemoryConfig(working_memory_turns=2, background_summary_refresh=False),
    )
    ctx = await svc.build(session_id=s.id, query="f", query_embedding=None)
    assert ctx.summary_regenerated is True
    assert _HEADER_SUMMARY in ctx.rendered_block
    assert "[summary covers 3 turns]" in ctx.rolling_summary
    assert "They asked about a, b, c." in ctx.rolling_summary
    # Summary persisted onto the session for next-turn reuse.
    persisted = await repo.get_session(s.id)
    assert persisted.summary == ctx.rolling_summary


async def test_summary_cached_when_older_turn_count_unchanged(repo):
    s = await repo.create_session(ConversationSession())
    for i, q in enumerate(["a", "b", "c", "d"]):
        await repo.append_turn(_turn(s.id, i, q, f"ans-{q}"))

    llm = _FakeLLM(response="cached")
    svc = MemoryService(
        conversation_repo=repo, llm_service=llm,
        config=MemoryConfig(working_memory_turns=2, background_summary_refresh=False),
    )
    # First call: regenerates.
    ctx1 = await svc.build(session_id=s.id, query="x", query_embedding=None)
    assert ctx1.summary_regenerated is True
    first_call_count = len(llm.calls)

    # Second call WITHOUT adding new turns: should reuse the cache.
    ctx2 = await svc.build(session_id=s.id, query="y", query_embedding=None)
    assert ctx2.summary_regenerated is False
    assert ctx2.rolling_summary == ctx1.rolling_summary
    assert len(llm.calls) == first_call_count


async def test_summary_regenerated_when_more_turns_added(repo):
    s = await repo.create_session(ConversationSession())
    for i, q in enumerate(["a", "b", "c", "d"]):
        await repo.append_turn(_turn(s.id, i, q, f"ans-{q}"))

    llm = _FakeLLM(response="v1")
    svc = MemoryService(
        conversation_repo=repo, llm_service=llm,
        config=MemoryConfig(working_memory_turns=2, background_summary_refresh=False),
    )
    await svc.build(session_id=s.id, query="x", query_embedding=None)
    # New turn added → older window grows from 2 to 3 turns.
    await repo.append_turn(_turn(s.id, 4, "e", "ans-e"))
    llm._response = "v2"
    ctx = await svc.build(session_id=s.id, query="y", query_embedding=None)
    assert ctx.summary_regenerated is True
    assert "[summary covers 3 turns]" in ctx.rolling_summary


async def test_summary_uses_deterministic_fallback_without_llm(repo):
    s = await repo.create_session(ConversationSession())
    for i in range(4):
        await repo.append_turn(_turn(s.id, i, f"q{i}", f"a{i}"))

    svc = MemoryService(
        conversation_repo=repo, llm_service=None,
        config=MemoryConfig(working_memory_turns=2, background_summary_refresh=False),
    )
    ctx = await svc.build(session_id=s.id, query="next", query_embedding=None)
    # Fallback concatenates Q/A pairs — older turns end up in the summary.
    assert "Q: q0" in ctx.rolling_summary
    assert "Q: q1" in ctx.rolling_summary


async def test_summary_can_be_disabled(repo):
    s = await repo.create_session(ConversationSession())
    for i in range(4):
        await repo.append_turn(_turn(s.id, i, f"q{i}", f"a{i}"))

    svc = MemoryService(
        conversation_repo=repo, llm_service=None,
        config=MemoryConfig(working_memory_turns=2, enable_summary=False),
    )
    ctx = await svc.build(session_id=s.id, query="next", query_embedding=None)
    assert ctx.rolling_summary == ""
    assert _HEADER_SUMMARY not in ctx.rendered_block


async def test_summary_falls_back_when_llm_raises(repo):
    s = await repo.create_session(ConversationSession())
    for i in range(4):
        await repo.append_turn(_turn(s.id, i, f"q{i}", f"a{i}"))

    class _BoomLLM:
        async def generate_text(self, **kwargs):
            raise RuntimeError("provider down")

    svc = MemoryService(
        conversation_repo=repo, llm_service=_BoomLLM(),
        config=MemoryConfig(working_memory_turns=2, background_summary_refresh=False),
    )
    ctx = await svc.build(session_id=s.id, query="next", query_embedding=None)
    # Fallback path produced a Q/A summary, didn't raise.
    assert ctx.summary_regenerated is True
    assert "Q: q0" in ctx.rolling_summary


# --------------------------------------------- background rolling-summary refresh


async def _drain_tasks(rounds: int = 20) -> None:
    """Let detached asyncio tasks (a backgrounded summary regen) run to
    completion. Each ``sleep(0)`` advances ready callbacks one round; the
    regen's awaits all resolve immediately in these hermetic stubs, so
    one round suffices — the extras are cheap insurance."""
    for _ in range(rounds):
        await asyncio.sleep(0)


async def test_summary_backgrounded_serves_stopgap_then_refreshes(repo):
    s = await repo.create_session(ConversationSession())
    for i, q in enumerate(["a", "b", "c", "d", "e"]):
        await repo.append_turn(_turn(s.id, i, q, f"ans-{q}"))

    llm = _FakeLLM(response="LLM summary of a/b/c.")
    svc = MemoryService(
        conversation_repo=repo, llm_service=llm,
        config=MemoryConfig(working_memory_turns=2),  # background is the default
    )

    ctx = await svc.build(session_id=s.id, query="f", query_embedding=None)
    # This turn gets a deterministic stopgap, not the LLM summary, and the
    # trace says we didn't pay the regen cost on the answer's path.
    assert ctx.summary_regenerated is False
    assert "[summary covers 3 turns]" in ctx.rolling_summary
    assert "Q: a" in ctx.rolling_summary
    assert llm.calls == []  # summariser hasn't been touched yet

    await _drain_tasks()
    # The detached task has now produced + persisted the real summary.
    assert len(llm.calls) == 1
    persisted = await repo.get_session(s.id)
    assert persisted.summary.startswith("[summary covers 3 turns]")
    assert "LLM summary of a/b/c." in persisted.summary

    # Next turn (no new turns added) reuses the cached LLM summary; still
    # no synchronous regen, no extra LLM call.
    ctx2 = await svc.build(session_id=s.id, query="g", query_embedding=None)
    assert ctx2.summary_regenerated is False
    assert ctx2.rolling_summary == persisted.summary
    assert len(llm.calls) == 1


async def test_summary_backgrounded_serves_stale_cache_while_refreshing(repo):
    s = await repo.create_session(ConversationSession())
    for i, q in enumerate(["a", "b", "c", "d", "e"]):
        await repo.append_turn(_turn(s.id, i, q, f"ans-{q}"))

    llm = _FakeLLM(response="v1")
    svc = MemoryService(
        conversation_repo=repo, llm_service=llm,
        config=MemoryConfig(working_memory_turns=2),
    )
    # Seed: first build schedules the regen; drain so the cache holds v1.
    await svc.build(session_id=s.id, query="x", query_embedding=None)
    await _drain_tasks()
    seeded = (await repo.get_session(s.id)).summary
    assert "v1" in seeded and seeded.startswith("[summary covers 3 turns]")

    # A new turn → older window grows 3→4, so the cached marker is stale.
    await repo.append_turn(_turn(s.id, 5, "f", "ans-f"))
    llm._response = "v2"
    ctx = await svc.build(session_id=s.id, query="y", query_embedding=None)
    # Served the *stale* v1 summary — not v2, and not synchronously.
    assert ctx.summary_regenerated is False
    assert ctx.rolling_summary == seeded
    assert "[summary covers 3 turns]" in ctx.rolling_summary

    await _drain_tasks()
    refreshed = (await repo.get_session(s.id)).summary
    assert refreshed.startswith("[summary covers 4 turns]")
    assert "v2" in refreshed


async def test_summary_falls_back_to_sync_when_no_event_loop(repo, monkeypatch):
    """When ``asyncio.create_task`` can't schedule (no running loop), the
    regen runs inline so the turn still gets a fresh summary."""
    s = await repo.create_session(ConversationSession())
    for i in range(4):
        await repo.append_turn(_turn(s.id, i, f"q{i}", f"a{i}"))

    def _no_loop(*_a, **_k):
        raise RuntimeError("no running event loop")

    monkeypatch.setattr(asyncio, "create_task", _no_loop)

    llm = _FakeLLM(response="sync summary")
    svc = MemoryService(
        conversation_repo=repo, llm_service=llm,
        config=MemoryConfig(working_memory_turns=2),  # background on, but can't schedule
    )
    ctx = await svc.build(session_id=s.id, query="next", query_embedding=None)
    assert ctx.summary_regenerated is True
    assert "sync summary" in ctx.rolling_summary
    assert len(llm.calls) == 1


async def test_qa_ask_backgrounds_rolling_summary(repo, monkeypatch):
    """QAService uses the default MemoryConfig → a stale rolling summary
    is regenerated off the answer's critical path."""
    from graphbuilder.infrastructure.services import embedding_factory

    async def fake_embed_async(text):
        return [1.0, 0.0]

    monkeypatch.setattr(embedding_factory, "embed_async", fake_embed_async)

    llm = _RecordingLLM()
    svc = QAService(
        orchestrator=_FakeOrchestrator(), conversation_repo=repo, llm_service=llm,
    )

    sid = None
    last = None
    for i in range(5):
        last = await svc.ask(query=f"q{i}", session_id=sid)
        sid = last.session_id
    # 5th turn: working window = 3 (default) → 4 prior turns → older = 1.
    assert last.memory_trace["summary_regenerated"] is False  # backgrounded
    assert last.memory_trace["summary_chars"] > 0             # stopgap rendered in

    await _drain_tasks()
    assert len(llm.summary_prompts) >= 1                      # summariser ran, after the answer
    persisted = await repo.get_session(sid)
    assert persisted.summary.startswith("[summary covers 1 turn]")


# ---------------------------------------------------------------- episodic recall


async def test_episodic_recall_finds_relevant_older_turn(repo):
    s = await repo.create_session(ConversationSession())
    # Two prior turns embedded along orthogonal axes; the new query
    # matches the FIRST one.
    await repo.append_turn(_turn(s.id, 0, "what does Imatinib target?", "BCR-ABL", embedding=[1.0, 0.0]))
    await repo.append_turn(_turn(s.id, 1, "tell me about p53", "tumour suppressor", embedding=[0.0, 1.0]))

    svc = MemoryService(
        conversation_repo=repo, llm_service=None,
        config=MemoryConfig(working_memory_turns=0, episodic_min_score=0.5),
    )
    ctx = await svc.build(
        session_id=s.id,
        query="what about its side effects?",
        query_embedding=[0.95, 0.05],
    )
    assert ctx.episodic_hit is not None
    turn, score = ctx.episodic_hit
    assert turn.user_query == "what does Imatinib target?"
    assert score > 0.5
    assert _HEADER_EPISODIC in ctx.rendered_block
    assert "Imatinib" in ctx.rendered_block


async def test_episodic_recall_accepts_awaitable_embedding(repo):
    """``build`` may be handed an in-flight embed task instead of a
    vector — episodic recall resolves it lazily and finds the same hit."""
    s = await repo.create_session(ConversationSession())
    await repo.append_turn(_turn(s.id, 0, "what does Imatinib target?", "BCR-ABL", embedding=[1.0, 0.0]))
    await repo.append_turn(_turn(s.id, 1, "tell me about p53", "tumour suppressor", embedding=[0.0, 1.0]))

    svc = MemoryService(
        conversation_repo=repo, llm_service=None,
        config=MemoryConfig(working_memory_turns=0, episodic_min_score=0.5),
    )

    async def _embed():
        return [0.95, 0.05]

    ctx = await svc.build(
        session_id=s.id,
        query="what about its side effects?",
        query_embedding=asyncio.create_task(_embed()),
    )
    assert ctx.episodic_hit is not None
    assert ctx.episodic_hit[0].user_query == "what does Imatinib target?"


async def test_episodic_recall_excludes_working_memory_turns(repo):
    s = await repo.create_session(ConversationSession())
    await repo.append_turn(_turn(s.id, 0, "earlier-Q", "a0", embedding=[1.0, 0.0]))
    await repo.append_turn(_turn(s.id, 1, "recent-Q", "a1", embedding=[1.0, 0.0]))

    svc = MemoryService(
        conversation_repo=repo, llm_service=None,
        config=MemoryConfig(working_memory_turns=1, episodic_min_score=0.0),
    )
    ctx = await svc.build(
        session_id=s.id, query="anything",
        query_embedding=[1.0, 0.0],
    )
    # working memory holds the most-recent turn; episodic should not
    # double-up on that turn — should find the earlier one.
    assert ctx.episodic_hit is not None
    assert ctx.episodic_hit[0].user_query == "earlier-Q"


async def test_episodic_recall_skipped_when_disabled(repo):
    s = await repo.create_session(ConversationSession())
    await repo.append_turn(_turn(s.id, 0, "q", "a", embedding=[1.0]))

    svc = MemoryService(
        conversation_repo=repo, llm_service=None,
        config=MemoryConfig(enable_episodic_recall=False, working_memory_turns=0),
    )
    ctx = await svc.build(session_id=s.id, query="q", query_embedding=[1.0])
    assert ctx.episodic_hit is None
    assert _HEADER_EPISODIC not in ctx.rendered_block


async def test_episodic_recall_skipped_without_query_embedding(repo):
    s = await repo.create_session(ConversationSession())
    await repo.append_turn(_turn(s.id, 0, "q", "a", embedding=[1.0]))

    svc = MemoryService(conversation_repo=repo, llm_service=None)
    ctx = await svc.build(session_id=s.id, query="q", query_embedding=None)
    assert ctx.episodic_hit is None


async def test_episodic_recall_filters_below_min_score(repo):
    s = await repo.create_session(ConversationSession())
    await repo.append_turn(_turn(s.id, 0, "irrelevant", "a", embedding=[1.0, 0.0]))
    svc = MemoryService(
        conversation_repo=repo, llm_service=None,
        config=MemoryConfig(working_memory_turns=0, episodic_min_score=0.9),
    )
    ctx = await svc.build(
        session_id=s.id, query="orthogonal",
        query_embedding=[0.0, 1.0],
    )
    assert ctx.episodic_hit is None


# ---------------------------------------------------------------- QAService end-to-end


class _FakeOrchestrator:
    """Returns canned items + trace; ignores the query_embedding."""

    async def retrieve(self, query, *, top_k=None, query_embedding=None, config_override=None):
        item = RetrievedItem(
            kind=ItemKind.ENTITY, id="e1", label="Imatinib",
            score_vector=0.9, score_rrf=0.5,
            contributing_channels=[Channel.VECTOR_ENTITY],
        )
        trace = RetrievalTrace(
            query=query, channels=[ChannelResult(channel=Channel.VECTOR_ENTITY)],
            rrf_top_n=1, final_top_k=1, hydrated_chunks=0, total_latency_ms=1,
        )
        return [item], trace


class _RecordingLLM:
    """Captures the prompts so tests can assert memory blocks made it in."""

    def __init__(self, summary_response="summary", answer_response="answer [1]"):
        self._summary = summary_response
        self._answer = answer_response
        self.summary_prompts: list[str] = []
        self.answer_prompts: list[str] = []

    async def generate_text(self, *, prompt, system_prompt, temperature, max_tokens):
        # Crude split: the summariser's system prompt mentions
        # "compress chat history".
        if system_prompt and "compress chat history" in system_prompt:
            self.summary_prompts.append(prompt)
            return self._summary
        self.answer_prompts.append(prompt)
        return self._answer


async def test_qa_ask_persists_query_embedding_for_episodic_recall(repo, monkeypatch):
    """Each turn must save its query_embedding so the *next* turn's
    episodic recall can find it. Verifies the round-trip from
    QAService.ask through to the persisted ConversationTurn."""
    from graphbuilder.infrastructure.services import embedding_factory

    captured: dict = {"vec": [0.5, 0.5]}

    async def fake_embed_async(text):
        return list(captured["vec"])

    monkeypatch.setattr(embedding_factory, "embed_async", fake_embed_async)

    svc = QAService(
        orchestrator=_FakeOrchestrator(),
        conversation_repo=repo,
        llm_service=_RecordingLLM(),
    )

    result = await svc.ask(query="first")
    turns = await repo.get_turns_by_session(result.session_id)
    assert len(turns) == 1
    assert turns[0].query_embedding == [0.5, 0.5]


async def test_qa_ask_threads_memory_block_into_prompt(repo, monkeypatch):
    """After a couple of turns, the third turn's prompt should contain
    a RECENT TURNS block — proves the memory layer is wired in."""
    from graphbuilder.infrastructure.services import embedding_factory

    async def fake_embed_async(text):
        return [1.0, 0.0]

    monkeypatch.setattr(embedding_factory, "embed_async", fake_embed_async)

    llm = _RecordingLLM()
    svc = QAService(
        orchestrator=_FakeOrchestrator(),
        conversation_repo=repo,
        llm_service=llm,
    )

    r1 = await svc.ask(query="hello")
    r2 = await svc.ask(query="follow up", session_id=r1.session_id)
    r3 = await svc.ask(query="another", session_id=r2.session_id)

    assert len(llm.answer_prompts) == 3
    last_prompt = llm.answer_prompts[-1]
    assert _HEADER_RECENT in last_prompt
    assert "USER: hello" in last_prompt
    assert "USER: follow up" in last_prompt
    # And the memory_trace surfaces what we used.
    assert r3.memory_trace["working_turns"] == 2


async def test_qa_ask_first_turn_has_no_memory_block(repo, monkeypatch):
    from graphbuilder.infrastructure.services import embedding_factory

    async def fake_embed_async(text):
        return [1.0, 0.0]

    monkeypatch.setattr(embedding_factory, "embed_async", fake_embed_async)

    llm = _RecordingLLM()
    svc = QAService(
        orchestrator=_FakeOrchestrator(),
        conversation_repo=repo,
        llm_service=llm,
    )
    result = await svc.ask(query="alone")
    prompt = llm.answer_prompts[-1]
    assert _HEADER_RECENT not in prompt
    assert _HEADER_SUMMARY not in prompt
    assert _HEADER_EPISODIC not in prompt
    assert result.memory_trace["working_turns"] == 0
