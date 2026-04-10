"""
E2E test — Real Open Targets + PubMed ingestion → graph query → export.

These tests hit the live Open Targets and PubMed APIs, so they require
network access.  They are marked with ``@pytest.mark.live`` so you can
skip them in CI with ``pytest -m "not live"``.

Run only these:  pytest tests/e2e/test_live_ingest.py -v
"""

import json
import time

import pytest
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Marker for tests that hit external services
# ---------------------------------------------------------------------------
pytestmark = pytest.mark.live


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def app_and_repo():
    """Create a fresh app + in-memory repo for the live tests."""
    import os
    os.environ.setdefault("LLM_API_KEY", "not-configured")
    os.environ.setdefault("NEO4J_URI", "bolt://localhost:7687")
    os.environ.setdefault("NEO4J_USER", "neo4j")
    os.environ.setdefault("NEO4J_PASSWORD", "password")

    from graphbuilder.infrastructure.config.settings import get_config
    from graphbuilder.infrastructure.repositories.graph_repository import InMemoryGraphRepository
    from api.main import create_app
    from api.dependencies import get_graph_repo

    config = get_config()
    repo = InMemoryGraphRepository(config)
    app = create_app()
    app.dependency_overrides[get_graph_repo] = lambda: repo
    yield app, repo
    app.dependency_overrides.clear()


@pytest.fixture(scope="module")
def client(app_and_repo):
    app, _ = app_and_repo
    with TestClient(app) as tc:
        yield tc


@pytest.fixture(scope="module")
def repo(app_and_repo):
    _, repo = app_and_repo
    return repo


# ---------------------------------------------------------------------------
# Open Targets — diabetes (EFO_0000400), small fetch
# ---------------------------------------------------------------------------

class TestOpenTargetsIngest:
    """Ingest a small slice of Open Targets disease-target data."""

    def test_ingest_and_query(self, client: TestClient, repo):
        # 1. Trigger ingest (background task runs synchronously in TestClient)
        r = client.post("/ingest/open-targets", json={
            "disease_id": "EFO_0000400",      # diabetes mellitus
            "max_associations": 5,
            "min_association_score": 0.0,
            "tag": "e2e-test",
        })
        assert r.status_code == 202
        data = r.json()
        assert data["source"] == "open-targets"
        job_id = data["job_id"]
        assert job_id

        # 2. Poll job status — should complete quickly
        status = self._poll_job(client, job_id, timeout=60)
        assert status in ("completed", "failed"), f"Job stuck in {status}"
        if status == "failed":
            pytest.skip("Open Targets API may be unavailable")

        # 3. Verify entities were created
        stats = client.get("/graph/stats").json()
        assert stats["total_entities"] >= 2, "Expected at least disease + 1 target"
        assert stats["total_relationships"] >= 1

        # 4. List entities and check the disease is present
        entities = client.get("/graph/entities").json()
        names = {e["name"].lower() for e in entities["items"]}
        assert any("diabet" in n for n in names), f"Expected diabetes entity, got: {names}"

        # 5. Export as JSON and verify it's consistent
        export = client.get("/export", params={"format": "json"}).json()
        assert len(export["entities"]) == stats["total_entities"]
        assert len(export["relationships"]) == stats["total_relationships"]

        # 6. Export as Cytoscape
        cyto = client.get("/export", params={"format": "cytoscape"}).json()
        nodes = [e for e in cyto["elements"] if e["group"] == "nodes"]
        assert len(nodes) == stats["total_entities"]

    @staticmethod
    def _poll_job(client, job_id, timeout=60):
        deadline = time.time() + timeout
        while time.time() < deadline:
            r = client.get(f"/documents/jobs/{job_id}")
            if r.status_code == 200:
                s = r.json().get("status", "")
                if s in ("completed", "failed"):
                    return s
            time.sleep(0.5)
        return "timeout"


# ---------------------------------------------------------------------------
# PubMed — small search
# ---------------------------------------------------------------------------

class TestPubMedIngest:
    """Ingest a small set of PubMed articles."""

    def test_ingest_and_query(self, client: TestClient, repo):
        # Clear prior data from the Open Targets test so counts are clean
        initial_entities = len(repo.entities)
        initial_rels = len(repo.relationships)

        # 1. Trigger ingest
        r = client.post("/ingest/pubmed", json={
            "query": "CRISPR gene editing",
            "max_articles": 3,
            "email": "e2e-test@graphbuilder.dev",
            "include_mesh": True,
            "include_keywords": True,
            "tag": "e2e-pubmed",
        })
        assert r.status_code == 202
        data = r.json()
        assert data["source"] == "pubmed"
        job_id = data["job_id"]

        # 2. Poll
        status = self._poll_job(client, job_id, timeout=90)
        assert status in ("completed", "failed"), f"Job stuck in {status}"
        if status == "failed":
            pytest.skip("PubMed API may be unavailable")

        # 3. Entities should have grown
        new_entities = len(repo.entities) - initial_entities
        new_rels = len(repo.relationships) - initial_rels
        assert new_entities >= 3, f"Expected >=3 new entities (articles), got {new_entities}"
        assert new_rels >= 1, f"Expected >=1 new relationships, got {new_rels}"

        # 4. Stats endpoint reflects combined data
        stats = client.get("/graph/stats").json()
        assert stats["total_entities"] == len(repo.entities)

        # 5. Verify document-type entities exist
        entities = client.get("/graph/entities").json()
        types = {e["entity_type"] for e in entities["items"]}
        # PubMed creates Document, Person, and Concept entities
        assert len(types) >= 2, f"Expected multiple entity types, got: {types}"

        # 6. Full export as GraphML
        r = client.get("/export", params={"format": "graphml"})
        assert r.status_code == 200
        assert "graphml" in r.text

    @staticmethod
    def _poll_job(client, job_id, timeout=90):
        deadline = time.time() + timeout
        while time.time() < deadline:
            r = client.get(f"/documents/jobs/{job_id}")
            if r.status_code == 200:
                s = r.json().get("status", "")
                if s in ("completed", "failed"):
                    return s
            time.sleep(0.5)
        return "timeout"


# ---------------------------------------------------------------------------
# Combined — verify both sources coexist
# ---------------------------------------------------------------------------

class TestCombinedGraph:
    """After both ingestions, verify the full graph is coherent."""

    def test_export_all_formats(self, client: TestClient):
        for fmt in ("json", "cytoscape", "graphml", "html"):
            r = client.get("/export", params={"format": fmt})
            assert r.status_code == 200, f"Export {fmt} returned {r.status_code}"

    def test_graph_stats_nonzero(self, client: TestClient, repo):
        if len(repo.entities) == 0:
            pytest.skip("No data ingested (prior ingest tests may have been skipped)")
        stats = client.get("/graph/stats").json()
        assert stats["total_entities"] > 0
        assert stats["total_relationships"] > 0

    def test_entities_have_metadata(self, client: TestClient):
        entities = client.get("/graph/entities", params={"limit": 5}).json()
        for e in entities["items"]:
            assert "id" in e
            assert "name" in e
            assert "entity_type" in e
            assert "created_at" in e
