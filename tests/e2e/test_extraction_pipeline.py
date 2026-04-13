"""
E2E test — LLM dedup services + graph API with vector search stubs.

Uses FastAPI TestClient with an InMemoryGraphRepository.  The LLM dedup
methods are exercised directly (mocked LLM client), and resulting graph
state is queried through the API.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from graphbuilder.domain.models.graph_models import (
    EntityType,
    GraphEntity,
    GraphRelationship,
    RelationshipType,
)
from graphbuilder.infrastructure.services.llm_service import AdvancedLLMService
from graphbuilder.infrastructure.config.settings import LLMProvider


# ---------------------------------------------------------------------------
# LLM service helpers (same as unit/test_llm_dedup.py)
# ---------------------------------------------------------------------------


def _make_config_llm() -> MagicMock:
    config = MagicMock()
    config.llm.provider = LLMProvider.OPENAI
    config.llm.api_key = "sk-test"
    config.llm.base_url = "https://api.openai.com/v1"
    config.llm.api_version = "2024-02-01"
    config.llm.timeout = 30.0
    config.llm.model_name = "gpt-4o-mini"
    return config


def _make_mock_client() -> MagicMock:
    client = MagicMock()
    client.chat = MagicMock()
    client.chat.completions = MagicMock()
    client.chat.completions.create = AsyncMock()
    return client


def _make_api_response(content: str) -> MagicMock:
    choice = MagicMock()
    choice.message.content = content
    choice.finish_reason = "stop"
    usage = MagicMock()
    usage.prompt_tokens = 100
    usage.completion_tokens = 50
    usage.total_tokens = 150
    resp = MagicMock()
    resp.choices = [choice]
    resp.usage = usage
    resp.model = "gpt-4o-mini"
    return resp


def _make_llm_service() -> AdvancedLLMService:
    config = _make_config_llm()
    with patch.object(AdvancedLLMService, "_initialize_client", return_value=_make_mock_client()):
        service = AdvancedLLMService(config)
    return service


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def seeded_repo():
    """InMemoryGraphRepository with entities suitable for dedup testing."""
    import asyncio
    import os
    from graphbuilder.infrastructure.config.settings import get_config
    from graphbuilder.infrastructure.repositories.graph_repository import InMemoryGraphRepository

    os.environ.setdefault("LLM_API_KEY", "not-configured")
    os.environ.setdefault("NEO4J_URI", "bolt://localhost:7687")
    os.environ.setdefault("NEO4J_USER", "neo4j")
    os.environ.setdefault("NEO4J_PASSWORD", "password")

    config = get_config()
    repo = InMemoryGraphRepository(config)

    def _ent(eid, name, etype, desc=""):
        e = GraphEntity(name=name, entity_type=etype, description=desc)
        e.id = eid
        e.source_document_ids = ["doc-1"]
        return e

    entities = [
        _ent("e-1", "Tumor Necrosis Factor Alpha", EntityType.CONCEPT, "A cytokine"),
        _ent("e-2", "Aspirin", EntityType.PRODUCT, "NSAID"),
        _ent("e-3", "Headache", EntityType.CONCEPT, "Symptom"),
    ]
    for e in entities:
        asyncio.get_event_loop().run_until_complete(repo.save_entity(e))

    r1 = GraphRelationship(
        source_entity_id="e-2",
        target_entity_id="e-3",
        relationship_type=RelationshipType.RELATED_TO,
        description="Aspirin treats headaches",
        strength=0.9,
    )
    r1.id = "r-1"
    asyncio.get_event_loop().run_until_complete(repo.save_relationship(r1))

    return repo


@pytest.fixture(scope="module")
def client(seeded_repo):
    from api.main import create_app
    from api.dependencies import get_graph_repo

    app = create_app()
    app.dependency_overrides[get_graph_repo] = lambda: seeded_repo

    with TestClient(app) as tc:
        yield tc

    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# 1. Graph state after seeding (sanity)
# ---------------------------------------------------------------------------


class TestGraphState:
    def test_entities_present(self, client: TestClient):
        r = client.get("/graph/entities")
        assert r.status_code == 200
        names = {e["name"] for e in r.json()["items"]}
        assert "Tumor Necrosis Factor Alpha" in names
        assert "Aspirin" in names

    def test_relationships_present(self, client: TestClient):
        r = client.get("/graph/stats")
        assert r.status_code == 200
        assert r.json()["total_relationships"] >= 1


# ---------------------------------------------------------------------------
# 2. LLM entity dedup service (integration through mocked LLM)
# ---------------------------------------------------------------------------


class TestLLMEntityDedup:
    """Exercise resolve_entity_duplicates end-to-end with mocked LLM."""

    async def test_abbreviation_resolved(self):
        svc = _make_llm_service()
        payload = {
            "matches": [
                {
                    "new_name": "TNF-alpha",
                    "existing_name": "Tumor Necrosis Factor Alpha",
                    "confidence": 0.96,
                    "reasoning": "abbreviation",
                }
            ]
        }
        svc.client.chat.completions.create.return_value = _make_api_response(
            json.dumps(payload)
        )

        result = await svc.resolve_entity_duplicates(
            [{"name": "TNF-alpha", "type": "CONCEPT", "description": "cytokine"}],
            [{"name": "Tumor Necrosis Factor Alpha", "type": "CONCEPT", "description": "A cytokine"}],
        )

        assert result.success is True
        assert len(result.data["matches"]) == 1
        assert result.data["matches"][0]["confidence"] > 0.9

    async def test_no_false_positive_on_unrelated(self):
        svc = _make_llm_service()
        payload = {"matches": []}
        svc.client.chat.completions.create.return_value = _make_api_response(
            json.dumps(payload)
        )

        result = await svc.resolve_entity_duplicates(
            [{"name": "Ibuprofen", "type": "PRODUCT", "description": "NSAID"}],
            [{"name": "Aspirin", "type": "PRODUCT", "description": "NSAID"}],
        )

        assert result.success is True
        assert result.data["matches"] == []


# ---------------------------------------------------------------------------
# 3. LLM relationship dedup service
# ---------------------------------------------------------------------------


class TestLLMRelationshipDedup:
    async def test_semantic_duplicate_detected(self):
        svc = _make_llm_service()
        payload = {"duplicate_of": 0, "confidence": 0.88, "reasoning": "same meaning"}
        svc.client.chat.completions.create.return_value = _make_api_response(
            json.dumps(payload)
        )

        result = await svc.check_relationship_duplicates(
            {"source": "A", "target": "B", "type": "ACTIVATES", "description": "A activates B"},
            [{"source": "A", "target": "B", "type": "STIMULATES", "description": "A stimulates B"}],
        )

        assert result.success is True
        assert result.data["duplicate_of"] == 0

    async def test_genuinely_different_relationship_not_merged(self):
        svc = _make_llm_service()
        payload = {"duplicate_of": None, "confidence": 0, "reasoning": "different"}
        svc.client.chat.completions.create.return_value = _make_api_response(
            json.dumps(payload)
        )

        result = await svc.check_relationship_duplicates(
            {"source": "A", "target": "B", "type": "INHIBITS", "description": "A inhibits B"},
            [{"source": "A", "target": "B", "type": "ACTIVATES", "description": "A activates B"}],
        )

        assert result.success is True
        assert result.data["duplicate_of"] is None


# ---------------------------------------------------------------------------
# 4. Graph query after manual dedup-style merge
# ---------------------------------------------------------------------------


class TestDedupMergeViaAPI:
    """Simulate what the extraction pipeline does: merge an entity, then
    verify the graph state through the API."""

    def test_merged_entity_visible(self, client: TestClient, seeded_repo):
        """After merging source_document_ids on an existing entity, the
        entity count should stay the same."""
        import asyncio

        existing = asyncio.get_event_loop().run_until_complete(
            seeded_repo.get_entity_by_id("e-1")
        )
        assert existing is not None
        original_docs = list(existing.source_document_ids)
        existing.source_document_ids = list(dict.fromkeys(
            existing.source_document_ids + ["doc-2"]
        ))
        asyncio.get_event_loop().run_until_complete(seeded_repo.save_entity(existing))

        r = client.get("/graph/entities")
        assert r.status_code == 200
        data = r.json()
        assert data["total"] == 3  # still 3, not 4

        # Clean up
        existing.source_document_ids = original_docs
        asyncio.get_event_loop().run_until_complete(seeded_repo.save_entity(existing))
