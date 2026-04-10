"""
E2E test — Full pipeline: health → seed graph → query → curation queue → export.

Uses FastAPI's TestClient with an in-memory graph repository seeded with
realistic entities and relationships. No external services required.
"""

import json
import xml.etree.ElementTree as ET

import pytest
from fastapi.testclient import TestClient

from graphbuilder.domain.models.graph_models import (
    EntityType,
    GraphEntity,
    GraphRelationship,
    KnowledgeGraph,
    RelationshipType,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def seeded_repo():
    """Create an InMemoryGraphRepository pre-loaded with test data."""
    import asyncio
    from graphbuilder.infrastructure.config.settings import get_config
    from graphbuilder.infrastructure.repositories.graph_repository import InMemoryGraphRepository

    import os
    os.environ.setdefault("LLM_API_KEY", "not-configured")
    os.environ.setdefault("NEO4J_URI", "bolt://localhost:7687")
    os.environ.setdefault("NEO4J_USER", "neo4j")
    os.environ.setdefault("NEO4J_PASSWORD", "password")

    config = get_config()
    repo = InMemoryGraphRepository(config)

    # --- entities ---
    def _entity(eid, name, etype, desc):
        e = GraphEntity(name=name, entity_type=etype, description=desc)
        e.id = eid
        return e

    e1 = _entity("e-1", "Aspirin", EntityType.PRODUCT, "Non-steroidal anti-inflammatory drug")
    e2 = _entity("e-2", "Bayer", EntityType.ORGANIZATION, "German pharmaceutical company")
    e3 = _entity("e-3", "Headache", EntityType.CONCEPT, "Common medical symptom")
    e4 = _entity("e-4", "Germany", EntityType.LOCATION, "European country")

    for e in (e1, e2, e3, e4):
        asyncio.get_event_loop().run_until_complete(repo.save_entity(e))

    # --- relationships ---
    def _rel(rid, src, tgt, rtype, strength):
        r = GraphRelationship(source_entity_id=src, target_entity_id=tgt,
                              relationship_type=rtype, strength=strength)
        r.id = rid
        return r

    r1 = _rel("r-1", "e-2", "e-1", RelationshipType.MANUFACTURES, 0.95)
    r2 = _rel("r-2", "e-1", "e-3", RelationshipType.RELATED_TO, 0.8)
    r3 = _rel("r-3", "e-2", "e-4", RelationshipType.LOCATED_IN, 1.0)

    for r in (r1, r2, r3):
        asyncio.get_event_loop().run_until_complete(repo.save_relationship(r))

    # Mark one relationship as flagged so the curation queue has something
    r2.metadata.annotations["verification_status"] = "flagged"
    r2.metadata.annotations["verification_notes"] = "Low confidence"

    return repo


@pytest.fixture(scope="module")
def client(seeded_repo):
    """FastAPI TestClient with graph repo overridden to the seeded repo."""
    from api.main import create_app
    from api.dependencies import get_graph_repo

    app = create_app()
    app.dependency_overrides[get_graph_repo] = lambda: seeded_repo

    with TestClient(app) as tc:
        yield tc

    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# 1. Health
# ---------------------------------------------------------------------------

class TestHealth:
    def test_health_ok(self, client: TestClient):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"


# ---------------------------------------------------------------------------
# 2. Graph queries
# ---------------------------------------------------------------------------

class TestGraphQueries:
    def test_stats(self, client: TestClient):
        r = client.get("/graph/stats")
        assert r.status_code == 200
        data = r.json()
        assert data["total_entities"] == 4
        assert data["total_relationships"] == 3
        assert "Product" in data["entity_type_counts"]

    def test_list_entities(self, client: TestClient):
        r = client.get("/graph/entities")
        assert r.status_code == 200
        data = r.json()
        assert data["total"] == 4
        assert len(data["items"]) == 4
        names = {e["name"] for e in data["items"]}
        assert "Aspirin" in names
        assert "Bayer" in names

    def test_list_entities_filter_type(self, client: TestClient):
        r = client.get("/graph/entities", params={"entity_type": "Organization"})
        assert r.status_code == 200
        data = r.json()
        assert data["total"] == 1
        assert data["items"][0]["name"] == "Bayer"

    def test_list_entities_pagination(self, client: TestClient):
        r = client.get("/graph/entities", params={"limit": 2, "offset": 0})
        assert r.status_code == 200
        assert len(r.json()["items"]) == 2

        r2 = client.get("/graph/entities", params={"limit": 2, "offset": 2})
        assert r2.status_code == 200
        assert len(r2.json()["items"]) == 2

    def test_get_entity_by_id(self, client: TestClient):
        r = client.get("/graph/entities/e-1")
        assert r.status_code == 200
        assert r.json()["name"] == "Aspirin"

    def test_get_entity_not_found(self, client: TestClient):
        r = client.get("/graph/entities/nonexistent")
        assert r.status_code == 404

    def test_list_relationships(self, client: TestClient):
        r = client.get("/graph/relationships")
        assert r.status_code == 200
        data = r.json()
        assert data["total"] == 3

    def test_relationships_filter_by_type(self, client: TestClient):
        r = client.get("/graph/relationships", params={"relationship_type": "MANUFACTURES"})
        assert r.status_code == 200
        data = r.json()
        assert data["total"] == 1
        assert data["items"][0]["source_entity_id"] == "e-2"

    def test_relationships_filter_by_source(self, client: TestClient):
        r = client.get("/graph/relationships", params={"source_entity_id": "e-2"})
        assert r.status_code == 200
        assert r.json()["total"] == 2  # MANUFACTURES + LOCATED_IN


# ---------------------------------------------------------------------------
# 3. Curation queue
# ---------------------------------------------------------------------------

class TestCurationQueue:
    def test_queue_returns_flagged(self, client: TestClient):
        r = client.get("/curation/queue")
        assert r.status_code == 200
        data = r.json()
        assert data["total"] >= 1
        flagged = [i for i in data["items"] if i["id"] == "r-2"]
        assert len(flagged) == 1
        assert flagged[0]["verification_status"] == "flagged"

    def test_queue_filter_status(self, client: TestClient):
        r = client.get("/curation/queue", params={"status": "flagged"})
        assert r.status_code == 200
        assert r.json()["total"] >= 1

        r2 = client.get("/curation/queue", params={"status": "rejected"})
        assert r2.status_code == 200
        assert r2.json()["total"] == 0


# ---------------------------------------------------------------------------
# 4. Export — all formats
# ---------------------------------------------------------------------------

class TestExport:
    def test_export_json(self, client: TestClient):
        r = client.get("/export", params={"format": "json"})
        assert r.status_code == 200
        assert "application/json" in r.headers["content-type"]
        data = r.json()
        assert len(data["entities"]) == 4
        assert len(data["relationships"]) == 3
        assert data["statistics"]["total_entities"] == 4

    def test_export_cytoscape(self, client: TestClient):
        r = client.get("/export", params={"format": "cytoscape"})
        assert r.status_code == 200
        assert "application/json" in r.headers["content-type"]
        data = r.json()
        nodes = [e for e in data["elements"] if e["group"] == "nodes"]
        edges = [e for e in data["elements"] if e["group"] == "edges"]
        assert len(nodes) == 4
        assert len(edges) == 3

    def test_export_graphml(self, client: TestClient):
        r = client.get("/export", params={"format": "graphml"})
        assert r.status_code == 200
        assert "xml" in r.headers["content-type"]
        root = ET.fromstring(r.text)
        ns = {"g": "http://graphml.graphdrawing.org/graphml"}
        nodes = root.findall(".//g:graph/g:node", ns)
        edges = root.findall(".//g:graph/g:edge", ns)
        assert len(nodes) == 4
        assert len(edges) == 3

    def test_export_html(self, client: TestClient):
        r = client.get("/export", params={"format": "html"})
        assert r.status_code == 200
        assert "text/html" in r.headers["content-type"]
        assert "<!DOCTYPE html>" in r.text
        assert "vis-network" in r.text

    def test_export_download_header(self, client: TestClient):
        r = client.get("/export", params={"format": "json"})
        assert "attachment" in r.headers.get("content-disposition", "")
        assert "graph_export.json" in r.headers["content-disposition"]

    def test_export_unknown_format(self, client: TestClient):
        r = client.get("/export", params={"format": "pdf"})
        assert r.status_code == 400


# ---------------------------------------------------------------------------
# 5. Full pipeline flow (sequential)
# ---------------------------------------------------------------------------

class TestFullPipelineFlow:
    """Verify the entire read path in order: health → stats → entities → export."""

    def test_pipeline(self, client: TestClient):
        # Step 1: health
        assert client.get("/health").status_code == 200

        # Step 2: stats
        stats = client.get("/graph/stats").json()
        assert stats["total_entities"] > 0
        assert stats["total_relationships"] > 0

        # Step 3: fetch all entities
        entities = client.get("/graph/entities").json()
        entity_ids = {e["id"] for e in entities["items"]}
        assert entity_ids == {"e-1", "e-2", "e-3", "e-4"}

        # Step 4: fetch relationships for a specific entity
        rels = client.get("/graph/relationships",
                          params={"source_entity_id": "e-1"}).json()
        assert rels["total"] >= 1

        # Step 5: check curation queue
        queue = client.get("/curation/queue").json()
        assert queue["total"] >= 1

        # Step 6: export full graph as JSON and validate counts match stats
        export = client.get("/export", params={"format": "json"}).json()
        assert len(export["entities"]) == stats["total_entities"]
        assert len(export["relationships"]) == stats["total_relationships"]
