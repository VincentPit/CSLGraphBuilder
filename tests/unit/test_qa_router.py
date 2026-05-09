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
    async def retrieve(self, query: str, *, top_k=None, query_embedding=None):
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
