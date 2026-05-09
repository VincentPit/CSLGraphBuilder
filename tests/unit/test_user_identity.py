"""Tests for the lightweight chatbot identity flow (§14.1, revised
2026-05-09 of docs/RAG_QA_PLAN.md).

Three layers:
1. ``InMemoryUserRepository`` — round-trip, update, touch, list.
2. ``/users`` router — register, fetch, rename via TestClient.
3. ``/qa/*`` routes with ``X-User-Id`` — sessions partitioned by user,
   401 on unknown id, ownership enforced on get/delete, and the
   anonymous fallback still works.
"""

from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("LLM_API_KEY", "not-configured")
os.environ.setdefault("NEO4J_URI", "bolt://localhost:7687")
os.environ.setdefault("NEO4J_USER", "neo4j")
os.environ.setdefault("NEO4J_PASSWORD", "password")

import api.routers.qa as qa_router  # noqa: E402
from api.dependencies import (  # noqa: E402
    get_app_config,
    get_conversation_repo,
    get_document_repo,
    get_graph_repo,
    get_llm,
    get_user_repo,
)
from api.main import app  # noqa: E402

from graphbuilder.core.retrieval.models import (  # noqa: E402
    Channel,
    ChannelResult,
    ItemKind,
    RetrievalTrace,
    RetrievedItem,
)
from graphbuilder.core.retrieval.qa_service import QAService  # noqa: E402
from graphbuilder.domain.models.user_models import User  # noqa: E402
from graphbuilder.infrastructure.config.settings import GraphBuilderConfig  # noqa: E402
from graphbuilder.infrastructure.repositories.conversation_repository import (  # noqa: E402
    InMemoryConversationRepository,
)
from graphbuilder.infrastructure.repositories.user_repository import (  # noqa: E402
    InMemoryUserRepository,
    create_user_repository,
)


# ---------------------------------------------------------------- repository


@pytest.fixture
def user_repo() -> InMemoryUserRepository:
    return InMemoryUserRepository(GraphBuilderConfig())


async def test_create_and_get_round_trip(user_repo):
    u = User(display_name="Alice")
    saved = await user_repo.create_user(u)
    assert saved.id == u.id
    fetched = await user_repo.get_user(u.id)
    assert fetched is not None
    assert fetched.display_name == "Alice"


async def test_get_unknown_user_returns_none(user_repo):
    assert await user_repo.get_user("nope") is None
    # Empty / whitespace ids are also rejected cleanly.
    assert await user_repo.get_user("") is None


async def test_update_user_changes_display_name(user_repo):
    u = await user_repo.create_user(User(display_name="Alice"))
    updated = await user_repo.update_user(u.id, display_name="Alicia")
    assert updated is not None
    assert updated.display_name == "Alicia"
    re_read = await user_repo.get_user(u.id)
    assert re_read.display_name == "Alicia"


async def test_update_user_merges_metadata(user_repo):
    u = User(display_name="Alice", metadata={"theme": "dark"})
    await user_repo.create_user(u)
    await user_repo.update_user(u.id, metadata={"locale": "en-GB"})
    re_read = await user_repo.get_user(u.id)
    # Merge keeps both keys.
    assert re_read.metadata == {"theme": "dark", "locale": "en-GB"}


async def test_update_user_unknown_id_returns_none(user_repo):
    assert await user_repo.update_user("missing", display_name="ghost") is None


async def test_touch_user_bumps_last_seen(user_repo):
    u = await user_repo.create_user(User(display_name="Alice"))
    before = u.last_seen_at
    touched = await user_repo.touch_user(u.id)
    assert touched is not None
    assert touched.last_seen_at >= before


async def test_touch_user_unknown_id_returns_none(user_repo):
    assert await user_repo.touch_user("missing") is None


async def test_list_users_orders_by_recent_activity(user_repo):
    a = await user_repo.create_user(User(display_name="A"))
    b = await user_repo.create_user(User(display_name="B"))
    await user_repo.touch_user(a.id)  # bumps A's last_seen
    listed = await user_repo.list_users(limit=10)
    assert [u.id for u in listed][0] == a.id
    assert {u.id for u in listed} == {a.id, b.id}


def test_factory_returns_in_memory_for_non_neo4j_provider(monkeypatch):
    cfg = GraphBuilderConfig()
    monkeypatch.setattr(cfg.database, "provider", "memory")
    repo = create_user_repository(cfg, neo4j_driver=None)
    assert isinstance(repo, InMemoryUserRepository)


# ---------------------------------------------------------------- /users router


@pytest.fixture
def users_client(user_repo):
    """TestClient with the user repo dep overridden — keeps the rest of
    the app wired so /users + /qa share the same in-memory user store."""
    qa_router._qa_service_singleton = None
    app.dependency_overrides[get_user_repo] = lambda: user_repo
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


def test_post_users_registers_and_returns_id(users_client):
    resp = users_client.post("/users", json={"display_name": "Stephen"})
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["id"].startswith("user_")
    assert body["display_name"] == "Stephen"


def test_post_users_rejects_empty_name(users_client):
    resp = users_client.post("/users", json={"display_name": ""})
    assert resp.status_code == 422


def test_get_user_404_when_unknown(users_client):
    assert users_client.get("/users/missing").status_code == 404


def test_patch_user_updates_display_name(users_client):
    created = users_client.post("/users", json={"display_name": "Old"}).json()
    resp = users_client.patch(
        f"/users/{created['id']}", json={"display_name": "New"},
    )
    assert resp.status_code == 200
    assert resp.json()["display_name"] == "New"


def test_patch_user_rejects_empty_body(users_client):
    created = users_client.post("/users", json={"display_name": "X"}).json()
    resp = users_client.patch(f"/users/{created['id']}", json={})
    assert resp.status_code == 400


# ---------------------------------------------------------------- /qa/* with X-User-Id


class _FakeRetrieval:
    """Tiny fake — only exists so the qa router has something to call."""

    async def retrieve(self, query, *, top_k=None, query_embedding=None):
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


class _FakeLLM:
    async def generate_text(self, **_):
        return "answer [1]"


@pytest.fixture
def qa_client(user_repo):
    """Full /qa stack with the user repo wired in."""
    qa_router._qa_service_singleton = None
    cfg = GraphBuilderConfig()
    conv_repo = InMemoryConversationRepository(cfg)

    app.dependency_overrides[get_user_repo] = lambda: user_repo
    app.dependency_overrides[get_conversation_repo] = lambda: conv_repo
    app.dependency_overrides[get_graph_repo] = lambda: object()
    app.dependency_overrides[get_document_repo] = lambda: None
    app.dependency_overrides[get_llm] = lambda: _FakeLLM()
    app.dependency_overrides[get_app_config] = lambda: cfg

    qa_router._qa_service_singleton = QAService(
        orchestrator=_FakeRetrieval(),  # type: ignore[arg-type]
        conversation_repo=conv_repo,
        llm_service=_FakeLLM(),
    )

    with TestClient(app) as c:
        yield c, user_repo, conv_repo

    app.dependency_overrides.clear()
    qa_router._qa_service_singleton = None


def test_ask_without_x_user_id_falls_through_to_anonymous(qa_client):
    """Backwards compatibility: clients that don't send X-User-Id keep
    landing in the anonymous bucket, just like the pre-identity flow."""
    client, _, conv_repo = qa_client
    resp = client.post("/qa/ask", json={"query": "hello"})
    assert resp.status_code == 200
    body = resp.json()
    # session was created with user_id=None (anonymous).
    import asyncio
    sess = asyncio.run(conv_repo.get_session(body["session_id"]))
    assert sess.user_id is None


def test_ask_with_x_user_id_attaches_user_to_session(qa_client):
    client, user_repo, conv_repo = qa_client
    import asyncio
    user = asyncio.run(user_repo.create_user(User(display_name="Stephen")))
    resp = client.post(
        "/qa/ask",
        json={"query": "hi"},
        headers={"X-User-Id": user.id},
    )
    assert resp.status_code == 200
    body = resp.json()
    sess = asyncio.run(conv_repo.get_session(body["session_id"]))
    assert sess.user_id == user.id


def test_ask_with_unknown_x_user_id_401s(qa_client):
    client, _, _ = qa_client
    resp = client.post(
        "/qa/ask",
        json={"query": "hi"},
        headers={"X-User-Id": "user_does_not_exist"},
    )
    assert resp.status_code == 401
    assert "clear localStorage" in resp.json()["detail"]


def test_list_sessions_filters_by_x_user_id(qa_client):
    client, user_repo, conv_repo = qa_client
    import asyncio
    alice = asyncio.run(user_repo.create_user(User(display_name="Alice")))
    bob = asyncio.run(user_repo.create_user(User(display_name="Bob")))

    client.post("/qa/ask", json={"query": "alice-q"}, headers={"X-User-Id": alice.id})
    client.post("/qa/ask", json={"query": "bob-q"}, headers={"X-User-Id": bob.id})

    alice_view = client.get("/qa/sessions", headers={"X-User-Id": alice.id}).json()
    bob_view = client.get("/qa/sessions", headers={"X-User-Id": bob.id}).json()

    alice_users = {s["user_id"] for s in alice_view["sessions"]}
    bob_users = {s["user_id"] for s in bob_view["sessions"]}
    assert alice_users == {alice.id}
    assert bob_users == {bob.id}


def test_get_session_404s_for_other_user(qa_client):
    """Alice creates a session; Bob trying to fetch it should see a 404
    (treat ownership mismatch as 404, not 403, so we don't leak existence)."""
    client, user_repo, _ = qa_client
    import asyncio
    alice = asyncio.run(user_repo.create_user(User(display_name="Alice")))
    bob = asyncio.run(user_repo.create_user(User(display_name="Bob")))

    ask = client.post(
        "/qa/ask", json={"query": "secret"}, headers={"X-User-Id": alice.id},
    )
    sid = ask.json()["session_id"]

    # Alice can read it.
    assert client.get(
        f"/qa/sessions/{sid}", headers={"X-User-Id": alice.id},
    ).status_code == 200

    # Bob gets a 404 — same code as nonexistent.
    assert client.get(
        f"/qa/sessions/{sid}", headers={"X-User-Id": bob.id},
    ).status_code == 404


def test_delete_session_blocked_for_other_user(qa_client):
    client, user_repo, _ = qa_client
    import asyncio
    alice = asyncio.run(user_repo.create_user(User(display_name="Alice")))
    bob = asyncio.run(user_repo.create_user(User(display_name="Bob")))

    ask = client.post(
        "/qa/ask", json={"query": "x"}, headers={"X-User-Id": alice.id},
    )
    sid = ask.json()["session_id"]

    # Bob tries to delete — denied without revealing existence.
    resp = client.delete(f"/qa/sessions/{sid}", headers={"X-User-Id": bob.id})
    assert resp.status_code == 404
    # Alice's session still exists.
    assert client.get(
        f"/qa/sessions/{sid}", headers={"X-User-Id": alice.id},
    ).status_code == 200


def test_anonymous_session_is_not_owned_by_anyone(qa_client):
    """Sessions created without X-User-Id stay readable by everyone, so
    pre-identity flows keep working. The auth rule only kicks in for
    user-scoped sessions."""
    client, user_repo, _ = qa_client
    import asyncio
    alice = asyncio.run(user_repo.create_user(User(display_name="Alice")))

    ask = client.post("/qa/ask", json={"query": "anon"})  # no header
    sid = ask.json()["session_id"]

    # An identified user can still read it.
    assert client.get(
        f"/qa/sessions/{sid}", headers={"X-User-Id": alice.id},
    ).status_code == 200
