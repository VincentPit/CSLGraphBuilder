"""Tests for QAService and the /qa/ask flow (P5 of docs/RAG_QA_PLAN.md).

Uses fakes for the orchestrator and LLM service so the test stays
hermetic — the orchestrator already has its own unit suite in
test_retrieval.py, and the LLM service is exercised elsewhere.
"""

from __future__ import annotations

import os
from typing import Any, List, Optional

import pytest

os.environ.setdefault("LLM_API_KEY", "not-configured")

from graphbuilder.core.retrieval.models import (  # noqa: E402
    Channel,
    ChannelResult,
    ItemKind,
    RetrievalConfig,
    RetrievalTrace,
    RetrievedItem,
)
from graphbuilder.core.retrieval.qa_service import (  # noqa: E402
    ANONYMOUS_USER_ID,
    AskResult,
    QAService,
    _extract_cited_indices,
    _render_sources,
    _render_user_prompt,
)
from graphbuilder.domain.models.conversation_models import (  # noqa: E402
    ConversationSession,
)
from graphbuilder.infrastructure.config.settings import GraphBuilderConfig  # noqa: E402
from graphbuilder.infrastructure.repositories.conversation_repository import (  # noqa: E402
    InMemoryConversationRepository,
)


# ---------------------------------------------------------------- fakes


class FakeOrchestrator:
    """Returns canned items + trace; ignores the query text."""

    def __init__(self, items: List[RetrievedItem], trace: RetrievalTrace):
        self._items = items
        self._trace = trace
        self.calls: list[tuple[str, Optional[int]]] = []
        # Spy for the per-call ``config_override`` value, in arrival
        # order. Kept separate from ``calls`` so existing assertions
        # against the (query, top_k) tuple shape stay valid.
        self.config_overrides: list[Optional[Any]] = []

    async def retrieve(
        self,
        query: str,
        *,
        top_k: Optional[int] = None,
        query_embedding: Optional[List[float]] = None,  # P6+P7: optional pre-embedded
        config_override: Optional[Any] = None,           # P13 ablation hook
    ):
        self.calls.append((query, top_k))
        self.config_overrides.append(config_override)
        return list(self._items), self._trace


class FakeLLM:
    def __init__(self, response: str):
        self._response = response
        self.calls: list[dict] = []

    async def generate_text(
        self,
        *,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> str:
        self.calls.append(
            {"prompt": prompt, "system_prompt": system_prompt, "temperature": temperature}
        )
        return self._response


def _item(eid: str, label: str, *, kind: ItemKind = ItemKind.ENTITY) -> RetrievedItem:
    return RetrievedItem(
        kind=kind, id=eid, label=label,
        score_vector=0.9, score_rrf=0.5,
        chunk_preview=f"some text mentioning {label}",
        source_chunk_id="c1", source_doc_id="doc_1",
        contributing_channels=[Channel.VECTOR_ENTITY],
        reasoning="vector hit",
    )


def _trace() -> RetrievalTrace:
    return RetrievalTrace(
        query="q",
        extracted_terms=["q"],
        channels=[ChannelResult(channel=Channel.VECTOR_ENTITY, latency_ms=5)],
        rrf_top_n=2,
        final_top_k=2,
        hydrated_chunks=2,
        total_latency_ms=12,
    )


@pytest.fixture
def conv_repo() -> InMemoryConversationRepository:
    return InMemoryConversationRepository(GraphBuilderConfig())


@pytest.fixture
def items() -> List[RetrievedItem]:
    return [
        _item("e1", "Imatinib"),
        _item("e2", "BCR-ABL"),
    ]


# ---------------------------------------------------------------- helpers


def test_extract_cited_indices_dedupes_and_preserves_order():
    assert _extract_cited_indices("foo [2] bar [1] baz [2] qux [3]") == [2, 1, 3]


def test_extract_cited_indices_ignores_non_numeric():
    assert _extract_cited_indices("see [n] and [foo]") == []


def test_extract_cited_indices_empty():
    assert _extract_cited_indices("") == []


def test_render_sources_numbers_from_one(items):
    rendered = _render_sources(items)
    assert "[1]" in rendered
    assert "[2]" in rendered
    assert "Imatinib" in rendered
    assert "BCR-ABL" in rendered


def test_render_sources_handles_empty():
    assert "no relevant" in _render_sources([])


def test_render_user_prompt_contains_question_and_sources(items):
    prompt = _render_user_prompt("does Imatinib inhibit BCR-ABL?", items)
    assert "QUESTION" in prompt
    assert "SOURCES" in prompt
    assert "does Imatinib inhibit BCR-ABL?" in prompt


# ---------------------------------------------------------------- service


async def test_ask_creates_anonymous_session_when_none_given(items, conv_repo):
    orch = FakeOrchestrator(items, _trace())
    llm = FakeLLM("Imatinib targets BCR-ABL [1] and other kinases.")
    svc = QAService(orchestrator=orch, conversation_repo=conv_repo, llm_service=llm)

    result = await svc.ask(query="does Imatinib do anything?")

    assert isinstance(result, AskResult)
    assert result.session_id.startswith("session_")
    sess = await conv_repo.get_session(result.session_id)
    assert sess is not None
    assert sess.user_id == ANONYMOUS_USER_ID
    assert sess.turn_count == 1


async def test_ask_appends_to_existing_session(items, conv_repo):
    existing = await conv_repo.create_session(ConversationSession())
    orch = FakeOrchestrator(items, _trace())
    llm = FakeLLM("Answer [1].")
    svc = QAService(orchestrator=orch, conversation_repo=conv_repo, llm_service=llm)

    r1 = await svc.ask(query="q1", session_id=existing.id)
    r2 = await svc.ask(query="q2", session_id=existing.id)

    assert r1.session_id == existing.id == r2.session_id
    sess = await conv_repo.get_session(existing.id)
    assert sess.turn_count == 2
    turns = await conv_repo.get_turns_by_session(existing.id)
    # Turns must be saved in order.
    assert [t.user_query for t in turns] == ["q1", "q2"]
    assert [t.idx for t in turns] == [0, 1]


async def test_ask_unknown_session_raises_lookup(items, conv_repo):
    orch = FakeOrchestrator(items, _trace())
    llm = FakeLLM("x")
    svc = QAService(orchestrator=orch, conversation_repo=conv_repo, llm_service=llm)

    with pytest.raises(LookupError):
        await svc.ask(query="q", session_id="session_does_not_exist")


async def test_ask_empty_query_raises_value_error(items, conv_repo):
    orch = FakeOrchestrator(items, _trace())
    llm = FakeLLM("x")
    svc = QAService(orchestrator=orch, conversation_repo=conv_repo, llm_service=llm)

    with pytest.raises(ValueError):
        await svc.ask(query="   ")


async def test_ask_records_cited_ids_on_turn(items, conv_repo):
    orch = FakeOrchestrator(items, _trace())
    # Cite [1] (entity Imatinib) — [2] is mentioned but [99] is out of range.
    llm = FakeLLM("Imatinib [1] is famous. Also [99] should be ignored.")
    svc = QAService(orchestrator=orch, conversation_repo=conv_repo, llm_service=llm)

    result = await svc.ask(query="tell me about Imatinib")
    assert result.cited_source_indices == [1]

    turns = await conv_repo.get_turns_by_session(result.session_id)
    saved = turns[0]
    assert saved.cited_entity_ids == ["e1"]
    # The cited item also has source_chunk_id=c1 → recorded on the turn.
    assert "c1" in saved.cited_chunk_ids


async def test_ask_relationship_citation_records_in_relationship_list(conv_repo):
    rel_item = RetrievedItem(
        kind=ItemKind.RELATIONSHIP,
        id="r_42",
        label="A --INHIBITS--> B",
        score_rrf=0.5,
        contributing_channels=[Channel.VECTOR_RELATIONSHIP],
    )
    orch = FakeOrchestrator([rel_item], _trace())
    llm = FakeLLM("There is an inhibition link [1].")
    svc = QAService(orchestrator=orch, conversation_repo=conv_repo, llm_service=llm)

    result = await svc.ask(query="any inhibition?")
    turns = await conv_repo.get_turns_by_session(result.session_id)
    saved = turns[0]
    assert saved.cited_relationship_ids == ["r_42"]
    assert saved.cited_entity_ids == []


async def test_ask_without_llm_returns_degraded_message(items, conv_repo):
    orch = FakeOrchestrator(items, _trace())
    svc = QAService(orchestrator=orch, conversation_repo=conv_repo, llm_service=None)

    result = await svc.ask(query="anything")
    assert "no LLM configured" in result.answer
    # Even without LLM we still return retrieved sources to the user.
    assert len(result.sources) == 2


async def test_ask_passes_top_k_through_to_orchestrator(items, conv_repo):
    orch = FakeOrchestrator(items, _trace())
    llm = FakeLLM("x")
    svc = QAService(orchestrator=orch, conversation_repo=conv_repo, llm_service=llm)

    await svc.ask(query="q", top_k=3)
    assert orch.calls == [("q", 3)]


async def test_ask_handles_llm_exception_gracefully(items, conv_repo):
    class BoomLLM:
        async def generate_text(self, **kwargs):
            raise RuntimeError("provider down")

    orch = FakeOrchestrator(items, _trace())
    svc = QAService(orchestrator=orch, conversation_repo=conv_repo, llm_service=BoomLLM())

    result = await svc.ask(query="q")
    assert "LLM call failed" in result.answer
    # The turn is still persisted so we have a record of the failed attempt.
    turns = await conv_repo.get_turns_by_session(result.session_id)
    assert len(turns) == 1


async def test_ask_propagates_request_id_onto_turn(items, conv_repo, monkeypatch):
    from graphbuilder.infrastructure.services import qa_observability

    orch = FakeOrchestrator(items, _trace())
    llm = FakeLLM("ok [1]")
    svc = QAService(orchestrator=orch, conversation_repo=conv_repo, llm_service=llm)

    token = qa_observability.set_request_id("req_test_abc")
    try:
        result = await svc.ask(query="q")
    finally:
        qa_observability.reset_request_id(token)

    assert result.request_id == "req_test_abc"
    turns = await conv_repo.get_turns_by_session(result.session_id)
    assert turns[0].request_id == "req_test_abc"


# ---------------------------------------------------------------- intent routing


async def test_ask_classifies_query_and_applies_intent_profile(items, conv_repo):
    """A "What is associated with X?" query is relational. The
    orchestrator must receive a ``config_override`` whose values match
    the relational profile (final_top_k=16, cypher_top_k=20)."""
    from graphbuilder.core.retrieval.intent import INTENT_PROFILES

    orch = FakeOrchestrator(items, _trace())
    llm = FakeLLM("ok [1]")
    svc = QAService(orchestrator=orch, conversation_repo=conv_repo, llm_service=llm)

    await svc.ask(query="What is associated with brca1?")
    [override] = orch.config_overrides
    assert override is not None
    relational = INTENT_PROFILES["relational"]
    assert override.final_top_k == relational.final_top_k
    assert override.cypher_top_k == relational.cypher_top_k


async def test_ask_lookup_query_disables_vector_relationship(items, conv_repo):
    """Bare-term lookups must produce a config_override with
    ``enable_vector_relationship=False`` — the latency win for this
    intent depends on it."""
    orch = FakeOrchestrator(items, _trace())
    llm = FakeLLM("ok [1]")
    svc = QAService(orchestrator=orch, conversation_repo=conv_repo, llm_service=llm)

    await svc.ask(query="olaparib")
    [override] = orch.config_overrides
    assert override is not None
    assert override.enable_vector_relationship is False


async def test_ask_records_intent_on_trace(items, conv_repo):
    """``RetrievalTrace.intent`` must reflect the chosen intent so the
    eval harness and debug pane can see which profile drove the turn."""
    orch = FakeOrchestrator(items, _trace())
    llm = FakeLLM("ok [1]")
    svc = QAService(orchestrator=orch, conversation_repo=conv_repo, llm_service=llm)

    result = await svc.ask(query="What is brca1?")
    assert result.retrieval_trace is not None
    assert result.retrieval_trace.intent == "definitional"


async def test_ask_explicit_retrieval_override_bypasses_routing(items, conv_repo):
    """The P13 ablation hook must stay honest: passing
    ``retrieval_override`` skips classification entirely. The
    orchestrator receives the caller's exact config and the trace's
    intent stays None so the eval doesn't pretend a profile ran."""
    custom = RetrievalConfig(final_top_k=4, cypher_top_k=3)
    orch = FakeOrchestrator(items, _trace())
    llm = FakeLLM("ok [1]")
    svc = QAService(orchestrator=orch, conversation_repo=conv_repo, llm_service=llm)

    result = await svc.ask(
        query="What is associated with brca1?",  # would otherwise route to relational
        retrieval_override=custom,
    )
    [override] = orch.config_overrides
    assert override is custom  # passed straight through
    assert result.retrieval_trace.intent is None


async def test_ask_intent_override_does_not_mutate_service_base_config(items, conv_repo):
    """``apply_profile`` is pure; running multiple turns must not
    accumulate profile changes onto ``self._cfg``. Otherwise the
    second turn would see the first turn's profile baked in."""
    orch = FakeOrchestrator(items, _trace())
    llm = FakeLLM("ok [1]")
    base = RetrievalConfig(final_top_k=8, cypher_top_k=10)
    svc = QAService(
        orchestrator=orch, conversation_repo=conv_repo,
        llm_service=llm, config=base,
    )

    await svc.ask(query="What is associated with brca1?")  # relational profile
    await svc.ask(query="olaparib")                         # lookup profile

    # Service's stored base config must be unchanged.
    assert svc._cfg is base
    assert base.final_top_k == 8
    assert base.cypher_top_k == 10
