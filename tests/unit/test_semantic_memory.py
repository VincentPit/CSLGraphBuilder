"""Tests for the cross-session semantic memory layer (P14 of docs/RAG_QA_PLAN.md).

Covers:
- ``SemanticMemoryService.load_persona`` returns the cached string from
  ``User.metadata`` (or ``""`` for anonymous / missing / disabled).
- ``refresh_persona`` summarises recent sessions, embeds the result, and
  writes both back to ``User.metadata`` (LLM and fallback paths).
- Freshness gate (``min_new_sessions_to_refresh``) and ``force=True``.
- Sessions with too few turns are skipped.
- ``QAService.ask`` splices the persona into the LLM ``system_prompt``.
- New-session creation schedules a background refresh.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any, List, Optional

import pytest

os.environ.setdefault("LLM_API_KEY", "not-configured")

from graphbuilder.core.retrieval.models import (  # noqa: E402
    Channel,
    ChannelResult,
    ItemKind,
    RetrievalTrace,
    RetrievedItem,
)
from graphbuilder.core.retrieval.qa_service import QAService  # noqa: E402
from graphbuilder.core.retrieval.semantic_memory import (  # noqa: E402
    META_COVERS,
    META_EMBEDDING,
    META_SUMMARY,
    META_UPDATED_AT,
    SemanticMemoryConfig,
    SemanticMemoryService,
)
from graphbuilder.domain.models.conversation_models import (  # noqa: E402
    ConversationSession,
    ConversationTurn,
)
from graphbuilder.domain.models.user_models import User  # noqa: E402
from graphbuilder.infrastructure.config.settings import GraphBuilderConfig  # noqa: E402
from graphbuilder.infrastructure.repositories.conversation_repository import (  # noqa: E402
    InMemoryConversationRepository,
)
from graphbuilder.infrastructure.repositories.user_repository import (  # noqa: E402
    InMemoryUserRepository,
)


# ---------------------------------------------------------------- helpers


def _turn(session_id: str, idx: int, q: str, a: str) -> ConversationTurn:
    return ConversationTurn(
        session_id=session_id, idx=idx, user_query=q, llm_answer=a,
    )


@pytest.fixture
def cfg() -> GraphBuilderConfig:
    return GraphBuilderConfig()


@pytest.fixture
def conv_repo(cfg) -> InMemoryConversationRepository:
    return InMemoryConversationRepository(cfg)


@pytest.fixture
def user_repo(cfg) -> InMemoryUserRepository:
    return InMemoryUserRepository(cfg)


class _FakeLLM:
    def __init__(self, response: str = "Persona: clinical pharmacologist focused on kinase inhibitors."):
        self._response = response
        self.calls: list[dict] = []

    async def generate_text(self, *, prompt, system_prompt, temperature, max_tokens):
        self.calls.append({"prompt": prompt, "system_prompt": system_prompt})
        return self._response


# ---------------------------------------------------------------- load_persona


async def test_load_persona_returns_empty_for_anonymous(user_repo, conv_repo):
    svc = SemanticMemoryService(
        user_repo=user_repo, conversation_repo=conv_repo, llm_service=None,
    )
    assert await svc.load_persona(None) == ""


async def test_load_persona_returns_empty_when_disabled(user_repo, conv_repo):
    u = User(display_name="x", metadata={META_SUMMARY: "old persona"})
    await user_repo.create_user(u)
    svc = SemanticMemoryService(
        user_repo=user_repo, conversation_repo=conv_repo, llm_service=None,
        config=SemanticMemoryConfig(enable=False),
    )
    assert await svc.load_persona(u.id) == ""


async def test_load_persona_returns_empty_for_unknown_user(user_repo, conv_repo):
    svc = SemanticMemoryService(
        user_repo=user_repo, conversation_repo=conv_repo, llm_service=None,
    )
    assert await svc.load_persona("user_nope") == ""


async def test_load_persona_returns_cached_text(user_repo, conv_repo):
    u = User(display_name="x", metadata={META_SUMMARY: "  clinical pharmacologist  "})
    await user_repo.create_user(u)
    svc = SemanticMemoryService(
        user_repo=user_repo, conversation_repo=conv_repo, llm_service=None,
    )
    assert await svc.load_persona(u.id) == "clinical pharmacologist"


# ---------------------------------------------------------------- refresh_persona


async def test_refresh_persona_writes_summary_and_embedding(user_repo, conv_repo):
    u = User(display_name="x")
    await user_repo.create_user(u)
    s = await conv_repo.create_session(ConversationSession(user_id=u.id))
    await conv_repo.append_turn(_turn(s.id, 0, "what does Imatinib target?", "BCR-ABL"))
    await conv_repo.append_turn(_turn(s.id, 1, "and side effects?", "nausea, oedema"))

    llm = _FakeLLM(response="Asks about kinase inhibitors at the mechanism level.")
    svc = SemanticMemoryService(
        user_repo=user_repo, conversation_repo=conv_repo, llm_service=llm,
    )
    new = await svc.refresh_persona(u.id)
    assert new is not None
    assert "kinase inhibitors" in new
    reloaded = await user_repo.get_user(u.id)
    assert reloaded.metadata[META_SUMMARY] == new
    assert reloaded.metadata[META_COVERS] == 1
    assert META_UPDATED_AT in reloaded.metadata
    # Embedding is opportunistic — may be empty in the hermetic env when
    # the embedding factory can't load a model. Just confirm the key exists.
    assert META_EMBEDDING in reloaded.metadata


async def test_refresh_persona_fallback_when_no_llm(user_repo, conv_repo):
    u = User(display_name="x", metadata={META_SUMMARY: "previous persona"})
    await user_repo.create_user(u)
    s = await conv_repo.create_session(ConversationSession(user_id=u.id))
    for i in range(2):
        await conv_repo.append_turn(_turn(s.id, i, f"q{i}", f"a{i}"))

    svc = SemanticMemoryService(
        user_repo=user_repo, conversation_repo=conv_repo, llm_service=None,
    )
    new = await svc.refresh_persona(u.id)
    assert new is not None
    # Deterministic fallback retains the previous persona text.
    assert "previous persona" in new


async def test_refresh_persona_freshness_gate(user_repo, conv_repo):
    # Cache says it covers 1 session and we still only have 1 → skip.
    u = User(display_name="x", metadata={META_SUMMARY: "p", META_COVERS: 1})
    await user_repo.create_user(u)
    s = await conv_repo.create_session(ConversationSession(user_id=u.id))
    for i in range(2):
        await conv_repo.append_turn(_turn(s.id, i, f"q{i}", f"a{i}"))

    svc = SemanticMemoryService(
        user_repo=user_repo, conversation_repo=conv_repo, llm_service=None,
    )
    assert await svc.refresh_persona(u.id) is None  # no refresh ran


async def test_refresh_persona_force_overrides_freshness_gate(user_repo, conv_repo):
    u = User(display_name="x", metadata={META_SUMMARY: "p", META_COVERS: 1})
    await user_repo.create_user(u)
    s = await conv_repo.create_session(ConversationSession(user_id=u.id))
    for i in range(2):
        await conv_repo.append_turn(_turn(s.id, i, f"q{i}", f"a{i}"))

    svc = SemanticMemoryService(
        user_repo=user_repo, conversation_repo=conv_repo, llm_service=None,
    )
    new = await svc.refresh_persona(u.id, force=True)
    assert new is not None


async def test_refresh_persona_skips_one_turn_sessions(user_repo, conv_repo):
    u = User(display_name="x")
    await user_repo.create_user(u)
    s = await conv_repo.create_session(ConversationSession(user_id=u.id))
    await conv_repo.append_turn(_turn(s.id, 0, "only-q", "only-a"))

    svc = SemanticMemoryService(
        user_repo=user_repo, conversation_repo=conv_repo, llm_service=None,
        config=SemanticMemoryConfig(min_turns_to_summarise=2),
    )
    assert await svc.refresh_persona(u.id) is None


# ---------------------------------------------------------------- QAService splice


class _FakeOrchestrator:
    def __init__(self, items, trace):
        self._items = items
        self._trace = trace

    async def retrieve(self, query, *, top_k=None, query_embedding=None, config_override=None):
        return list(self._items), self._trace


class _SpyLLM:
    def __init__(self, response="answer [1]"):
        self._response = response
        self.calls: list[dict] = []

    async def generate_text(self, *, prompt, system_prompt, temperature, max_tokens):
        self.calls.append({"prompt": prompt, "system_prompt": system_prompt})
        return self._response


def _item(eid="e1", label="Imatinib") -> RetrievedItem:
    return RetrievedItem(
        kind=ItemKind.ENTITY, id=eid, label=label,
        score_vector=0.9, score_rrf=0.5,
        chunk_preview=f"text about {label}",
        source_chunk_id="c1", source_doc_id="doc_1",
        contributing_channels=[Channel.VECTOR_ENTITY],
        reasoning="vector hit",
    )


def _trace() -> RetrievalTrace:
    return RetrievalTrace(
        query="q", extracted_terms=["q"],
        channels=[ChannelResult(channel=Channel.VECTOR_ENTITY, latency_ms=1)],
        rrf_top_n=1, final_top_k=1, hydrated_chunks=1, total_latency_ms=2,
    )


async def test_ask_splices_persona_into_system_prompt(user_repo, conv_repo):
    u = User(display_name="x", metadata={META_SUMMARY: "clinical pharmacologist"})
    await user_repo.create_user(u)

    llm = _SpyLLM(response="The drug targets BCR-ABL [1].")
    sem = SemanticMemoryService(
        user_repo=user_repo, conversation_repo=conv_repo, llm_service=None,
    )
    svc = QAService(
        orchestrator=_FakeOrchestrator([_item()], _trace()),
        conversation_repo=conv_repo,
        llm_service=llm,
        semantic_memory=sem,
    )
    await svc.ask(query="what does Imatinib target?", user_id=u.id)

    assert llm.calls, "LLM should have been called"
    sys_prompt = llm.calls[-1]["system_prompt"]
    assert "USER SEMANTIC SUMMARY" in sys_prompt
    assert "clinical pharmacologist" in sys_prompt


async def test_ask_omits_persona_block_when_user_has_no_summary(user_repo, conv_repo):
    u = User(display_name="x")  # no metadata
    await user_repo.create_user(u)

    llm = _SpyLLM()
    sem = SemanticMemoryService(
        user_repo=user_repo, conversation_repo=conv_repo, llm_service=None,
    )
    svc = QAService(
        orchestrator=_FakeOrchestrator([_item()], _trace()),
        conversation_repo=conv_repo, llm_service=llm,
        semantic_memory=sem,
    )
    await svc.ask(query="q", user_id=u.id)
    sys_prompt = llm.calls[-1]["system_prompt"]
    assert "USER SEMANTIC SUMMARY" not in sys_prompt


async def test_ask_omits_persona_block_for_anonymous(conv_repo, user_repo):
    llm = _SpyLLM()
    sem = SemanticMemoryService(
        user_repo=user_repo, conversation_repo=conv_repo, llm_service=None,
    )
    svc = QAService(
        orchestrator=_FakeOrchestrator([_item()], _trace()),
        conversation_repo=conv_repo, llm_service=llm,
        semantic_memory=sem,
    )
    await svc.ask(query="q", user_id=None)
    assert "USER SEMANTIC SUMMARY" not in llm.calls[-1]["system_prompt"]


async def test_new_session_schedules_background_persona_refresh(user_repo, conv_repo):
    """On a brand-new session, the service kicks off a background
    refresh. We swap in a SemanticMemoryService stub so the test stays
    deterministic; the QAService just has to *invoke* refresh_persona."""

    class _SemStub:
        def __init__(self):
            self.refresh_calls: list[str] = []
            self.load_calls: list[Optional[str]] = []

        async def load_persona(self, user_id):
            self.load_calls.append(user_id)
            return ""

        async def refresh_persona(self, user_id, *, force=False, include_session_ids=None):
            self.refresh_calls.append(user_id)
            return None

    u = User(display_name="x")
    await user_repo.create_user(u)
    stub = _SemStub()
    svc = QAService(
        orchestrator=_FakeOrchestrator([_item()], _trace()),
        conversation_repo=conv_repo, llm_service=_SpyLLM(),
        semantic_memory=stub,
    )
    await svc.ask(query="hello", user_id=u.id)

    # Background task was scheduled — yield once so it can run.
    await asyncio.sleep(0)
    assert u.id in stub.refresh_calls


async def test_set_semantic_memory_post_construction(conv_repo, user_repo):
    """The setter mirrors set_mutation_dispatcher so the router can
    construct the service first and attach the persona surface later."""
    u = User(display_name="x", metadata={META_SUMMARY: "later-attached"})
    await user_repo.create_user(u)
    llm = _SpyLLM()
    svc = QAService(
        orchestrator=_FakeOrchestrator([_item()], _trace()),
        conversation_repo=conv_repo, llm_service=llm,
    )
    # Without semantic memory: no persona block.
    await svc.ask(query="q", user_id=u.id)
    assert "USER SEMANTIC SUMMARY" not in llm.calls[-1]["system_prompt"]

    svc.set_semantic_memory(SemanticMemoryService(
        user_repo=user_repo, conversation_repo=conv_repo, llm_service=None,
    ))
    await svc.ask(query="q2", user_id=u.id)
    assert "later-attached" in llm.calls[-1]["system_prompt"]
