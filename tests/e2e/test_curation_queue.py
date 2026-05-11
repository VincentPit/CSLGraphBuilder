"""E2E tests for the curation queue endpoints.

Covers:
* The `verified` status surfaces in the queue (the bulk-approve power
  action depends on this — verified items are auto-approved by the
  pipeline but still need to appear in the curator's view so they can be
  human-confirmed in batch).
* Status filter narrows correctly.
* Counts endpoint includes the `verified` bucket.
* The in-memory fallback path triggers when the repo can't run Cypher.
"""

from __future__ import annotations

import asyncio
import os

import pytest
from fastapi.testclient import TestClient

from graphbuilder.domain.models.graph_models import (
    EntityType,
    GraphEntity,
    GraphRelationship,
    RelationshipType,
)


def _annotate(obj, status: str, *, curated: bool = False) -> None:
    obj.metadata.annotations["verification_status"] = status
    if curated:
        obj.metadata.annotations["curated"] = True


@pytest.fixture(scope="module")
def queue_repo():
    """Repo with one entity per status bucket so queue tests can probe each."""
    from graphbuilder.infrastructure.config.settings import get_config
    from graphbuilder.infrastructure.repositories.graph_repository import (
        InMemoryGraphRepository,
    )

    os.environ.setdefault("LLM_API_KEY", "not-configured")
    os.environ.setdefault("NEO4J_URI", "bolt://localhost:7687")
    os.environ.setdefault("NEO4J_USER", "neo4j")
    os.environ.setdefault("NEO4J_PASSWORD", "password")

    repo = InMemoryGraphRepository(get_config())
    loop = asyncio.new_event_loop()

    def _e(eid: str, name: str, status: str, *, curated: bool = False) -> GraphEntity:
        e = GraphEntity(name=name, entity_type=EntityType.CONCEPT, description=name)
        e.id = eid
        _annotate(e, status, curated=curated)
        return e

    # One entity per status — plus one curated item that should NOT appear.
    for ent in (
        _e("e-rejected", "RejectedConcept", "rejected"),
        _e("e-flagged", "FlaggedConcept", "flagged"),
        _e("e-unverified", "UnverifiedConcept", "unverified"),
        _e("e-verified", "VerifiedConcept", "verified"),
        _e("e-curated", "CuratedConcept", "verified", curated=True),  # excluded
    ):
        loop.run_until_complete(repo.save_entity(ent))

    # And one verified relationship to test the rel branch.
    src = GraphEntity(name="A", entity_type=EntityType.CONCEPT)
    src.id = "x-1"
    tgt = GraphEntity(name="B", entity_type=EntityType.CONCEPT)
    tgt.id = "x-2"
    loop.run_until_complete(repo.save_entity(src))
    loop.run_until_complete(repo.save_entity(tgt))

    rel = GraphRelationship(
        source_entity_id="x-1",
        target_entity_id="x-2",
        relationship_type=RelationshipType.RELATED_TO,
    )
    rel.id = "r-verified"
    _annotate(rel, "verified")
    loop.run_until_complete(repo.save_relationship(rel))

    loop.close()
    return repo


@pytest.fixture(scope="module")
def client(queue_repo):
    from api.dependencies import get_graph_repo
    from api.main import create_app

    app = create_app()
    app.dependency_overrides[get_graph_repo] = lambda: queue_repo
    with TestClient(app) as tc:
        yield tc
    app.dependency_overrides.clear()


class TestQueueVerifiedStatus:
    def test_queue_includes_verified_status(self, client: TestClient):
        """The `Approve N verified` button on the curation page filters
        items by verification_status === 'verified'. Before this fix the
        backend never emitted that status — the bulk-action button was
        dead code. This test locks in the contract.
        """
        r = client.get("/curation/queue")
        assert r.status_code == 200
        statuses = {item["verification_status"] for item in r.json()["items"]}
        assert "verified" in statuses

    def test_curated_items_are_excluded(self, client: TestClient):
        """Even with verification_status=verified, items already curated
        (curated=true) must not appear in the queue."""
        r = client.get("/curation/queue")
        ids = {item["id"] for item in r.json()["items"]}
        assert "e-curated" not in ids

    def test_status_filter_narrows_to_verified_only(self, client: TestClient):
        r = client.get("/curation/queue", params={"status": "verified"})
        assert r.status_code == 200
        items = r.json()["items"]
        assert items, "should return verified items"
        for item in items:
            assert item["verification_status"] == "verified"

    def test_verified_sorts_after_other_statuses(self, client: TestClient):
        """Verified is the auto-approved bucket — it must rank below
        rejected/flagged/unverified so reviewers see urgent items first."""
        r = client.get("/curation/queue")
        items = r.json()["items"]
        # Find the position of the first verified vs first non-verified item.
        first_verified = next(
            (i for i, it in enumerate(items) if it["verification_status"] == "verified"),
            None,
        )
        first_nonverified = next(
            (i for i, it in enumerate(items) if it["verification_status"] != "verified"),
            None,
        )
        if first_verified is not None and first_nonverified is not None:
            assert first_nonverified < first_verified


class TestQueueCounts:
    def test_counts_include_verified_bucket(self, client: TestClient):
        r = client.get("/curation/queue/counts")
        assert r.status_code == 200
        body = r.json()
        # Bucket exists in the response shape (frontend filter-chips depend on it).
        assert "verified" in body
        assert body["verified"] >= 1

    def test_total_sums_all_buckets(self, client: TestClient):
        body = client.get("/curation/queue/counts").json()
        assert body["total"] == (
            body["rejected"] + body["flagged"] + body["unverified"] + body["verified"]
        )


class TestInMemoryFallback:
    """The queue/counts endpoints push filtering into Cypher when Neo4j is
    available, but fall back to in-memory iteration otherwise. The test
    fixture uses InMemoryGraphRepository, so a 200 response on either
    endpoint means the fallback path is exercised."""

    def test_queue_works_without_cypher(self, client: TestClient):
        assert client.get("/curation/queue").status_code == 200

    def test_counts_work_without_cypher(self, client: TestClient):
        assert client.get("/curation/queue/counts").status_code == 200
