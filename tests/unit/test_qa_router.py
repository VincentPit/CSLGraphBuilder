"""Smoke tests for the /qa router (P5 of docs/RAG_QA_PLAN.md).

Verifies wiring + request/response shape using FastAPI's TestClient
with dependency overrides. The QAService internals are covered by
test_qa_service.py — here we only confirm the router translates
correctly between Pydantic and the dataclasses.
"""

from __future__ import annotations

import os
from typing import Any, List

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("LLM_API_KEY", "not-configured")
os.environ.setdefault("NEO4J_URI", "bolt://localhost:7687")
os.environ.setdefault("NEO4J_USER", "neo4j")
os.environ.setdefault("NEO4J_PASSWORD", "password")

# Ensure the API singleton picks up our overrides — we reset between tests.
import api.routers.qa as qa_router  # noqa: E402
from api.dependencies import (  # noqa: E402
    get_app_config,
    get_conversation_repo,
    get_document_repo,
    get_graph_repo,
    get_llm,
)
from api.main import app  # noqa: E402

from graphbuilder.core.retrieval.models import (  # noqa: E402
    Channel,
    ChannelResult,
    ItemKind,
    RetrievalTrace,
    RetrievedItem,
)
from graphbuilder.core.retrieval.qa_service import AskResult, QAService  # noqa: E402
from graphbuilder.domain.models.conversation_models import (  # noqa: E402
    ConversationSession,
    ConversationTurn,
)
from graphbuilder.infrastructure.config.settings import GraphBuilderConfig  # noqa: E402
from graphbuilder.infrastructure.repositories.conversation_repository import (  # noqa: E402
    InMemoryConversationRepository,
)


# ---------------------------------------------------------------- fakes


class _FakeRetrieval:
    async def retrieve(self, query: str, *, top_k=None, query_embedding=None, config_override=None):
        items = [
            RetrievedItem(
                kind=ItemKind.ENTITY,
                id="e1",
                label="Imatinib",
                score_vector=0.9,
                score_rrf=0.5,
                final_confidence=0.92,
                source_chunk_id="c1",
                source_doc_id="doc_1",
                chunk_preview="…Imatinib inhibits BCR-ABL…",
                contributing_channels=[Channel.VECTOR_ENTITY],
                reasoning="vector hit",
            )
        ]
        trace = RetrievalTrace(
            query=query,
            extracted_terms=["Imatinib"],
            channels=[
                ChannelResult(channel=Channel.VECTOR_ENTITY, latency_ms=4),
            ],
            rrf_top_n=1,
            final_top_k=1,
            hydrated_chunks=1,
            total_latency_ms=8,
        )
        return items, trace


class _FakeLLM:
    async def generate_text(self, **kwargs) -> str:
        return "Imatinib targets BCR-ABL [1]."


# ---------------------------------------------------------------- fixtures


@pytest.fixture
def conv_repo():
    return InMemoryConversationRepository(GraphBuilderConfig())


@pytest.fixture
def client(conv_repo):
    """Build a TestClient with dependencies replaced by in-memory fakes.

    We also reset the QAService module singleton in the qa router so each
    test sees a fresh service instance bound to *this* fixture's repo.
    """
    qa_router._qa_service_singleton = None  # reset singleton between tests

    def _override_conv_repo():
        return conv_repo

    def _override_graph_repo():
        return object()  # opaque — fake retrieval ignores it

    def _override_doc_repo():
        return None

    async def _override_llm():
        return _FakeLLM()

    def _override_config():
        return GraphBuilderConfig()

    app.dependency_overrides[get_conversation_repo] = _override_conv_repo
    app.dependency_overrides[get_graph_repo] = _override_graph_repo
    app.dependency_overrides[get_document_repo] = _override_doc_repo
    app.dependency_overrides[get_llm] = _override_llm
    app.dependency_overrides[get_app_config] = _override_config

    # Build the QAService ourselves with the fake retrieval and assign
    # to the singleton so the router's lazy factory picks it up.
    qa_router._qa_service_singleton = QAService(
        orchestrator=_FakeRetrieval(),  # type: ignore[arg-type]
        conversation_repo=conv_repo,
        llm_service=_FakeLLM(),
    )

    with TestClient(app) as c:
        yield c

    app.dependency_overrides.clear()
    qa_router._qa_service_singleton = None


# ---------------------------------------------------------------- tests


def test_post_ask_returns_well_formed_response(client):
    resp = client.post("/qa/ask", json={"query": "tell me about Imatinib"})
    assert resp.status_code == 200, resp.text
    body = resp.json()

    assert body["answer"] == "Imatinib targets BCR-ABL [1]."
    assert body["session_id"].startswith("session_")
    assert body["turn_id"].startswith("turn_")
    assert body["sources"][0]["id"] == "e1"
    assert body["sources"][0]["kind"] == "entity"
    assert body["cited_source_indices"] == [1]
    assert body["retrieval_trace"]["channels"][0]["channel"] == "vector_entity"
    # Request id middleware echoes a header.
    assert resp.headers.get("x-request-id")
    # Body request_id is the same value.
    assert body["request_id"] == resp.headers["x-request-id"]


def test_post_ask_rejects_empty_query(client):
    resp = client.post("/qa/ask", json={"query": "   "})
    assert resp.status_code == 400
    assert "empty" in resp.json()["detail"]


def test_post_ask_unknown_session_returns_404(client):
    resp = client.post(
        "/qa/ask", json={"query": "q", "session_id": "session_missing"}
    )
    assert resp.status_code == 404


def test_get_session_returns_session_and_turns(client, conv_repo):
    # First, create a session via /ask so we have something to fetch.
    ask = client.post("/qa/ask", json={"query": "q1"})
    session_id = ask.json()["session_id"]

    resp = client.get(f"/qa/sessions/{session_id}")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["session"]["id"] == session_id
    assert len(body["turns"]) == 1
    assert body["turns"][0]["user_query"] == "q1"


def test_get_session_404_for_unknown(client):
    resp = client.get("/qa/sessions/session_does_not_exist")
    assert resp.status_code == 404


def test_list_sessions_returns_anonymous_sessions(client):
    client.post("/qa/ask", json={"query": "first"})
    client.post("/qa/ask", json={"query": "second"})
    resp = client.get("/qa/sessions")
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["sessions"]) == 2


def test_delete_session_204(client):
    ask = client.post("/qa/ask", json={"query": "q"})
    session_id = ask.json()["session_id"]
    resp = client.delete(f"/qa/sessions/{session_id}")
    assert resp.status_code == 204
    # Subsequent GET should now 404.
    assert client.get(f"/qa/sessions/{session_id}").status_code == 404


def test_delete_unknown_session_404(client):
    assert client.delete("/qa/sessions/none").status_code == 404


def test_post_feedback_records_rating(client, conv_repo):
    ask = client.post("/qa/ask", json={"query": "q"})
    turn_id = ask.json()["turn_id"]

    resp = client.post(
        f"/qa/turns/{turn_id}/feedback", json={"rating": 1, "comment": "great"},
    )
    assert resp.status_code == 200
    assert resp.json() == {"turn_id": turn_id, "accepted": True}


def test_post_feedback_unknown_turn_404(client):
    resp = client.post(
        "/qa/turns/missing/feedback", json={"rating": 1},
    )
    assert resp.status_code == 404


# ---------------------------------------------------------------- P11 streaming


class _StreamingFakeLLM:
    """LLM with both streaming and non-streaming methods.

    The router test only cares about the SSE frame shape, so we keep
    chunks small and predictable.
    """

    async def generate_text_stream(self, *, prompt, system_prompt=None,
                                   temperature: float = 0.0, max_tokens: int = 1024):
        for piece in ("Imatinib ", "targets ", "BCR-ABL ", "[1]."):
            yield piece

    async def generate_text(self, **kwargs):
        return "Imatinib targets BCR-ABL [1]."


def _parse_sse(body: str):
    """Parse the raw SSE response body into ``[{event, data}]`` dicts.

    ``sse_starlette`` writes frames with CRLF line terminators
    (``event: <name>\\r\\ndata: <json>\\r\\n\\r\\n``). We normalise to
    LF before splitting so the blank-line delimiter is unambiguous,
    then pull the ``event:`` + ``data:`` fields out of each frame.
    """
    import json as _json

    normalised = body.replace("\r\n", "\n").strip()
    out = []
    for frame in normalised.split("\n\n"):
        if not frame.strip():
            continue
        ev_name = None
        data_lines: list[str] = []
        for line in frame.split("\n"):
            if line.startswith("event:"):
                ev_name = line[len("event:"):].strip()
            elif line.startswith("data:"):
                data_lines.append(line[len("data:"):].strip())
        if ev_name is None or not data_lines:
            continue
        try:
            payload = _json.loads("\n".join(data_lines))
        except _json.JSONDecodeError:
            payload = {"_raw": "\n".join(data_lines)}
        out.append({"event": ev_name, "data": payload})
    return out


def test_post_ask_stream_emits_complete_event_sequence(client, conv_repo):
    """The streaming endpoint must emit phase → retrieval → phase →
    delta(s) → done in order."""
    # Replace the singleton's LLM with the streaming fake — the
    # fixture-built service used the non-streaming _FakeLLM.
    qa_router._qa_service_singleton._llm = _StreamingFakeLLM()
    qa_router._qa_service_singleton._faithfulness._llm = _StreamingFakeLLM()

    resp = client.post("/qa/ask/stream", json={"query": "tell me about Imatinib"})
    assert resp.status_code == 200, resp.text
    # sse_starlette sets content-type text/event-stream
    assert resp.headers["content-type"].startswith("text/event-stream")

    events = _parse_sse(resp.text)
    kinds = [e["event"] for e in events]
    assert kinds[0] == "phase"
    assert events[0]["data"]["phase"] == "retrieving"
    assert "retrieval" in kinds
    # second phase event = generating
    assert any(
        e["event"] == "phase" and e["data"]["phase"] == "generating"
        for e in events
    )
    assert kinds[-1] == "done"
    deltas = [e["data"]["text"] for e in events if e["event"] == "delta"]
    assert "".join(deltas) == "Imatinib targets BCR-ABL [1]."

    done = events[-1]["data"]
    assert done["session_id"].startswith("session_")
    assert done["turn_id"].startswith("turn_")
    assert done["cited_source_indices"] == [1]
    assert done["faithfulness"] is not None


def test_post_ask_stream_rejects_empty_query(client):
    resp = client.post("/qa/ask/stream", json={"query": "   "})
    assert resp.status_code == 400


def test_post_ask_stream_unknown_session_yields_error_event(client):
    qa_router._qa_service_singleton._llm = _StreamingFakeLLM()
    resp = client.post(
        "/qa/ask/stream",
        json={"query": "q", "session_id": "session_missing"},
    )
    # We don't 404 here because the SSE stream is already open by the
    # time we try to resolve the session — surface the error inside the
    # stream instead so the client can render it the same way as a
    # mid-stream LLM failure.
    assert resp.status_code == 200
    events = _parse_sse(resp.text)
    assert events[-1]["event"] == "error"
    assert events[-1]["data"]["kind"] == "session_not_found"


# ---------------------------------------------------------------- P9 tool-use


class _ToolingFakeLLM:
    """LLM that requests one tool call then produces a final answer.

    Drives the agentic loop end-to-end through the router so the
    test exercises Pydantic translation of ToolCallModel as well.
    """

    def __init__(self):
        self._step = 0

    async def generate_text(self, **kwargs):
        return "non-streaming fallback [1]"

    async def generate_with_tools(self, *, messages, tools, temperature=0.0,
                                  max_tokens=1024):
        self._step += 1
        if self._step == 1:
            return {
                "content": None,
                "tool_calls": [{
                    "id": "call_a",
                    "name": "search_graph",
                    "arguments": {"query": "Imatinib", "top_k": 3},
                }],
                "finish_reason": "tool_calls",
            }
        return {
            "content": "Imatinib targets BCR-ABL [1].",
            "tool_calls": [],
            "finish_reason": "stop",
        }


def test_post_ask_with_enable_tools_records_tool_calls(client):
    """Router-level: enable_tools=true should run the agentic loop and
    surface tool_calls on the response."""
    from graphbuilder.core.retrieval.tools import ToolCallRecord

    # The router fixture builds the singleton without a dispatcher (see
    # the @pytest.fixture above). Inject a dispatcher fake here so the
    # agentic loop actually runs — calling the real ToolDispatcher
    # would hit the orchestrator + graph_repo, both of which are fakes
    # in the fixture, and we want to assert the wiring not the contents.
    class _RouterDispatcher:
        def __init__(self):
            self.dispatched: list[tuple[str, dict]] = []

        def openai_tool_schemas(self):
            return [{
                "type": "function",
                "function": {"name": "search_graph",
                             "parameters": {"type": "object"}},
            }]

        async def execute(self, name, args, *, tool_call_id=None):
            self.dispatched.append((name, args))
            return ToolCallRecord(
                tool=name, args=args,
                result={"items": [{"id": "e1", "label": "Imatinib"}]},
                latency_ms=2, tool_call_id=tool_call_id,
            )

    qa_router._qa_service_singleton._llm = _ToolingFakeLLM()
    qa_router._qa_service_singleton._tools = _RouterDispatcher()
    # Also point faithfulness's LLM at the tooling fake — it doesn't
    # use streaming/tools but the FaithfulnessChecker's ``_llm`` slot
    # holds a reference.
    qa_router._qa_service_singleton._faithfulness._llm = _ToolingFakeLLM()

    resp = client.post(
        "/qa/ask",
        json={"query": "tell me about Imatinib", "enable_tools": True},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()

    assert body["answer"] == "Imatinib targets BCR-ABL [1]."
    assert len(body["tool_calls"]) == 1
    tc = body["tool_calls"][0]
    assert tc["tool"] == "search_graph"
    assert tc["args"] == {"query": "Imatinib", "top_k": 3}
    assert tc["error"] is None


def test_post_ask_default_enable_tools_false_no_tool_calls(client):
    """Default request body must NOT trigger the agentic loop — tool_calls
    arrives empty."""
    resp = client.post("/qa/ask", json={"query": "q"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["tool_calls"] == []


# ---------------------------------------------------------------- P10 proposals


@pytest.fixture(autouse=True)
def _reset_proposal_store():
    """Keep the process-scoped proposal queue clean between tests."""
    from api import proposed_mutation_store
    proposed_mutation_store.reset_for_tests()
    yield
    proposed_mutation_store.reset_for_tests()


def _seed_proposal(**overrides):
    """Insert one proposal directly via the store (skipping the LLM)."""
    from api.proposed_mutation_store import add_proposal
    return add_proposal(**{
        "tool": "propose_entity",
        "args": {"name": "BRCA1", "entity_type": "GENE",
                 "description": "DNA repair gene", "aliases": [],
                 "external_ids": {}},
        "summary": "Propose entity: BRCA1 (GENE)",
        **overrides,
    })


def test_get_proposals_lists_pending(client):
    p = _seed_proposal()
    resp = client.get("/qa/proposals")
    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 1
    assert body["items"][0]["proposal_id"] == p.proposal_id
    assert body["items"][0]["status"] == "pending"


def test_get_proposals_filters_by_status(client):
    a = _seed_proposal()
    b = _seed_proposal(args={"name": "BRCA2", "entity_type": "GENE",
                             "description": "", "aliases": [],
                             "external_ids": {}})
    # Decide one — it should fall out of the pending filter.
    from api.proposed_mutation_store import mark_decided
    mark_decided(a.proposal_id, "approved")

    resp = client.get("/qa/proposals?status=pending")
    ids = {it["proposal_id"] for it in resp.json()["items"]}
    assert ids == {b.proposal_id}

    resp_all = client.get("/qa/proposals?status=all")
    ids = {it["proposal_id"] for it in resp_all.json()["items"]}
    assert ids == {a.proposal_id, b.proposal_id}


def test_apply_proposal_runs_handler_and_pins_target_id(client, monkeypatch):
    """Curator approves → MutationApplier runs against the graph repo."""
    p = _seed_proposal()

    # Wire a recording graph repo through the dependency override.
    from api.dependencies import get_graph_repo as get_graph_repo_dep

    class _Repo:
        async def save_entity(self, entity, **_):
            entity.id = "ent_BRCA1_id"
            return entity
        async def save_relationship(self, rel, **_):
            return rel
        async def get_entity_by_id(self, _):
            return None
        async def merge_entities(self, **_):
            return None

    app.dependency_overrides[get_graph_repo_dep] = lambda: _Repo()
    try:
        resp = client.post(f"/qa/proposals/{p.proposal_id}/apply", json={"notes": "lgtm"})
    finally:
        # Restore the fixture's override so other tests aren't disturbed.
        app.dependency_overrides[get_graph_repo_dep] = lambda: object()

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["applied_target_id"] == "ent_BRCA1_id"
    assert body["proposal"]["status"] == "approved"
    assert body["proposal"]["applied_target_id"] == "ent_BRCA1_id"
    assert body["proposal"]["decision_notes"] == "lgtm"


def test_apply_proposal_records_error_on_handler_failure(client):
    """A LookupError from the applier surfaces as 400 + apply_error pinned."""
    p = _seed_proposal(
        tool="update_entity",
        args={"entity_id": "missing", "description": "x", "add_aliases": [],
              "add_external_ids": {}},
        summary="Update entity missing",
    )

    from api.dependencies import get_graph_repo as get_graph_repo_dep

    class _MissingRepo:
        async def save_entity(self, e, **_): return e
        async def save_relationship(self, r, **_): return r
        async def get_entity_by_id(self, _): return None  # → LookupError

    app.dependency_overrides[get_graph_repo_dep] = lambda: _MissingRepo()
    try:
        resp = client.post(f"/qa/proposals/{p.proposal_id}/apply")
    finally:
        app.dependency_overrides[get_graph_repo_dep] = lambda: object()

    assert resp.status_code == 400
    # The decision was already recorded as approved, but the apply
    # error is pinned on the row so the curator UI can show retry.
    from api.proposed_mutation_store import get_proposal
    row = get_proposal(p.proposal_id)
    assert row.status == "approved"
    assert "missing" in (row.apply_error or "")


def test_reject_proposal_marks_rejected(client):
    p = _seed_proposal()
    resp = client.post(f"/qa/proposals/{p.proposal_id}/reject",
                       json={"notes": "duplicate of existing"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "rejected"
    assert body["decision_notes"] == "duplicate of existing"


def test_apply_unknown_proposal_404(client):
    resp = client.post("/qa/proposals/prop_doesnotexist/apply")
    assert resp.status_code == 404


def test_apply_already_decided_proposal_returns_409(client):
    p = _seed_proposal()
    from api.proposed_mutation_store import mark_decided
    mark_decided(p.proposal_id, "rejected")

    resp = client.post(f"/qa/proposals/{p.proposal_id}/apply")
    assert resp.status_code == 409


class _MutatingFakeLLM:
    """LLM that asks for one mutating tool call then produces an answer.

    Verifies the agentic loop routes the tool to the mutation
    dispatcher (which queues the proposal) rather than the read
    dispatcher (which would error on an unknown tool name).
    """

    def __init__(self):
        self._step = 0

    async def generate_text(self, **kwargs):
        return "fallback [1]"

    async def generate_with_tools(self, *, messages, tools, **kwargs):
        self._step += 1
        if self._step == 1:
            return {
                "content": None,
                "tool_calls": [{
                    "id": "call_1",
                    "name": "propose_entity",
                    "arguments": {
                        "name": "BRCA1", "entity_type": "GENE",
                        "description": "DNA repair gene",
                    },
                }],
                "finish_reason": "tool_calls",
            }
        return {
            "content": "Proposal queued for review [1].",
            "tool_calls": [],
            "finish_reason": "stop",
        }


def test_post_ask_with_enable_mutations_queues_proposal(client):
    """End-to-end: enable_mutations=true + LLM asks for propose_entity →
    a row lands in the proposal queue and the response surfaces it."""
    from api.proposed_mutation_store import add_proposal
    from graphbuilder.core.retrieval.mutation_tools import MutationToolDispatcher

    # The conftest builds the singleton without a mutation dispatcher;
    # also without a read-only dispatcher (the agentic loop needs at
    # least one of them active to fire).
    qa_router._qa_service_singleton._llm = _MutatingFakeLLM()
    qa_router._qa_service_singleton._faithfulness._llm = _MutatingFakeLLM()
    qa_router._qa_service_singleton.set_mutation_dispatcher(
        MutationToolDispatcher(enqueue_fn=add_proposal),
    )

    resp = client.post("/qa/ask", json={
        "query": "Add BRCA1 as a gene entity",
        "enable_tools": True,
        "enable_mutations": True,
    })
    assert resp.status_code == 200, resp.text
    body = resp.json()

    assert len(body["tool_calls"]) == 1
    tc = body["tool_calls"][0]
    assert tc["tool"] == "propose_entity"
    assert tc["error"] is None
    assert tc["result"]["status"] == "queued"
    assert tc["result"]["proposal_id"].startswith("prop_")

    # The proposal queue actually has it.
    from api.proposed_mutation_store import get_proposal
    row = get_proposal(tc["result"]["proposal_id"])
    assert row is not None
    assert row.tool == "propose_entity"
    assert row.status == "pending"


def test_post_ask_with_mutations_disabled_blocks_call_with_error_record(client):
    """If the LLM tries a mutating tool but enable_mutations=false, the
    dispatcher returns an error record instead of queueing — protects
    against a model that ignores the system prompt.

    Note: enable_tools=true must still be set so the agentic loop runs
    in the first place. Without any active dispatcher the loop never
    starts and the LLM can't get a chance to misbehave.
    """
    from graphbuilder.core.retrieval.tools import ToolCallRecord

    class _RouterDispatcher:
        def openai_tool_schemas(self):
            return [{"type": "function",
                     "function": {"name": "search_graph",
                                  "parameters": {"type": "object"}}}]

        async def execute(self, name, args, *, tool_call_id=None):
            return ToolCallRecord(
                tool=name, args=args, result={"items": []},
                tool_call_id=tool_call_id,
            )

    qa_router._qa_service_singleton._llm = _MutatingFakeLLM()
    qa_router._qa_service_singleton._tools = _RouterDispatcher()
    # No mutation dispatcher set — the routing should refuse the call.
    qa_router._qa_service_singleton.set_mutation_dispatcher(None)

    resp = client.post("/qa/ask", json={
        "query": "Add BRCA1 as a gene entity",
        "enable_tools": True,
        "enable_mutations": False,
    })
    assert resp.status_code == 200
    body = resp.json()

    # The mutating call landed in tool_calls with an error, and nothing
    # was queued.
    mut_calls = [c for c in body["tool_calls"] if c["tool"] == "propose_entity"]
    assert len(mut_calls) == 1
    assert mut_calls[0]["error"]
    assert "mutating tools are not enabled" in mut_calls[0]["error"]

    from api.proposed_mutation_store import list_proposals
    assert list_proposals() == []
