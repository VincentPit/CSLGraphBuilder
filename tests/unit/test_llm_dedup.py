"""
Unit tests for LLM-powered deduplication methods.

Covers:
  - AdvancedLLMService.resolve_entity_duplicates
  - AdvancedLLMService.check_relationship_duplicates
  - ProcessDocumentUseCase embedding helpers (_get_embedding_model,
    _embed_entity_text, _embed_text)
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from graphbuilder.infrastructure.services.llm_service import AdvancedLLMService
from graphbuilder.infrastructure.config.settings import LLMProvider
from graphbuilder.domain.models.graph_models import (
    GraphEntity,
    EntityType,
)


# ---------------------------------------------------------------------------
# Helpers (same pattern as test_llm_service.py)
# ---------------------------------------------------------------------------


def _make_config(provider: LLMProvider = LLMProvider.OPENAI) -> MagicMock:
    config = MagicMock()
    config.llm.provider = provider
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
    response = MagicMock()
    response.choices = [choice]
    response.usage = usage
    response.model = "gpt-4o-mini"
    return response


def _make_service() -> AdvancedLLMService:
    config = _make_config()
    with patch.object(AdvancedLLMService, "_initialize_client", return_value=_make_mock_client()):
        service = AdvancedLLMService(config)
    return service


# ---------------------------------------------------------------------------
# resolve_entity_duplicates
# ---------------------------------------------------------------------------


class TestResolveEntityDuplicates:

    async def test_returns_matches_on_success(self):
        service = _make_service()
        payload = {
            "matches": [
                {
                    "new_name": "TNF-alpha",
                    "existing_name": "Tumor Necrosis Factor Alpha",
                    "confidence": 0.95,
                    "reasoning": "abbreviation",
                }
            ]
        }
        service.client.chat.completions.create.return_value = _make_api_response(
            json.dumps(payload)
        )

        result = await service.resolve_entity_duplicates(
            [{"name": "TNF-alpha", "type": "PROTEIN", "description": ""}],
            [{"name": "Tumor Necrosis Factor Alpha", "type": "PROTEIN", "description": ""}],
        )

        assert result.success is True
        assert len(result.data["matches"]) == 1
        assert result.data["matches"][0]["new_name"] == "TNF-alpha"

    async def test_empty_new_entities_short_circuits(self):
        service = _make_service()
        result = await service.resolve_entity_duplicates(
            [],
            [{"name": "X", "type": "CONCEPT", "description": ""}],
        )
        assert result.success is True
        assert result.data["matches"] == []
        service.client.chat.completions.create.assert_not_called()

    async def test_empty_existing_entities_short_circuits(self):
        service = _make_service()
        result = await service.resolve_entity_duplicates(
            [{"name": "X", "type": "CONCEPT", "description": ""}],
            [],
        )
        assert result.success is True
        assert result.data["matches"] == []

    async def test_filters_matches_without_required_fields(self):
        service = _make_service()
        payload = {
            "matches": [
                {"new_name": "A", "existing_name": "B", "confidence": 0.9},
                {"new_name": "", "existing_name": "C", "confidence": 0.8},
                {"new_name": "D", "existing_name": "", "confidence": 0.7},
            ]
        }
        service.client.chat.completions.create.return_value = _make_api_response(
            json.dumps(payload)
        )

        result = await service.resolve_entity_duplicates(
            [{"name": "A", "type": "CONCEPT", "description": ""}],
            [{"name": "B", "type": "CONCEPT", "description": ""}],
        )

        assert result.success is True
        assert len(result.data["matches"]) == 1
        assert result.data["matches"][0]["new_name"] == "A"

    async def test_llm_error_returns_failure(self):
        service = _make_service()
        service.client.chat.completions.create.side_effect = Exception("API down")

        result = await service.resolve_entity_duplicates(
            [{"name": "A", "type": "CONCEPT", "description": ""}],
            [{"name": "B", "type": "CONCEPT", "description": ""}],
        )

        assert result.success is False
        assert result.data["matches"] == []

    async def test_no_matches_returns_empty_list(self):
        service = _make_service()
        payload = {"matches": []}
        service.client.chat.completions.create.return_value = _make_api_response(
            json.dumps(payload)
        )

        result = await service.resolve_entity_duplicates(
            [{"name": "Apple", "type": "PRODUCT", "description": "fruit"}],
            [{"name": "Car", "type": "PRODUCT", "description": "vehicle"}],
        )

        assert result.success is True
        assert result.data["matches"] == []


# ---------------------------------------------------------------------------
# check_relationship_duplicates
# ---------------------------------------------------------------------------


class TestCheckRelationshipDuplicates:

    async def test_duplicate_found(self):
        service = _make_service()
        payload = {"duplicate_of": 0, "confidence": 0.92, "reasoning": "same fact"}
        service.client.chat.completions.create.return_value = _make_api_response(
            json.dumps(payload)
        )

        result = await service.check_relationship_duplicates(
            {"source": "A", "target": "B", "type": "ACTIVATES", "description": "A activates B"},
            [{"source": "A", "target": "B", "type": "STIMULATES", "description": "A stimulates B"}],
        )

        assert result.success is True
        assert result.data["duplicate_of"] == 0
        assert result.data["confidence"] >= 0.9

    async def test_no_duplicate(self):
        service = _make_service()
        payload = {"duplicate_of": None, "confidence": 0, "reasoning": "different"}
        service.client.chat.completions.create.return_value = _make_api_response(
            json.dumps(payload)
        )

        result = await service.check_relationship_duplicates(
            {"source": "A", "target": "B", "type": "INHIBITS", "description": "A inhibits B"},
            [{"source": "A", "target": "B", "type": "ACTIVATES", "description": "A activates B"}],
        )

        assert result.success is True
        assert result.data["duplicate_of"] is None

    async def test_empty_existing_short_circuits(self):
        service = _make_service()
        result = await service.check_relationship_duplicates(
            {"source": "A", "target": "B", "type": "ACTIVATES", "description": ""},
            [],
        )
        assert result.success is True
        assert result.data["duplicate_of"] is None
        service.client.chat.completions.create.assert_not_called()

    async def test_out_of_range_index_returns_none(self):
        service = _make_service()
        payload = {"duplicate_of": 5, "confidence": 0.9, "reasoning": "bad index"}
        service.client.chat.completions.create.return_value = _make_api_response(
            json.dumps(payload)
        )

        result = await service.check_relationship_duplicates(
            {"source": "A", "target": "B", "type": "X", "description": ""},
            [{"source": "A", "target": "B", "type": "Y", "description": ""}],
        )

        assert result.success is True
        assert result.data["duplicate_of"] is None

    async def test_negative_index_returns_none(self):
        service = _make_service()
        payload = {"duplicate_of": -1, "confidence": 0.9, "reasoning": ""}
        service.client.chat.completions.create.return_value = _make_api_response(
            json.dumps(payload)
        )

        result = await service.check_relationship_duplicates(
            {"source": "A", "target": "B", "type": "X", "description": ""},
            [{"source": "A", "target": "B", "type": "Y", "description": ""}],
        )

        assert result.success is True
        assert result.data["duplicate_of"] is None

    async def test_api_error_returns_failure(self):
        service = _make_service()
        service.client.chat.completions.create.side_effect = RuntimeError("timeout")

        result = await service.check_relationship_duplicates(
            {"source": "A", "target": "B", "type": "X", "description": ""},
            [{"source": "A", "target": "B", "type": "Y", "description": ""}],
        )

        assert result.success is False
        assert result.data["duplicate_of"] is None


# ---------------------------------------------------------------------------
# Embedding helpers (on ProcessDocumentUseCase)
#
# These helpers delegate to ``embedding_factory.embed``, which the use case
# imports inside the function body — so we patch the factory module rather
# than the use-case attribute. ``embedding_factory.embed`` itself returns
# ``None`` when sentence-transformers is unavailable, when text is empty,
# or when the encode call raises; the helpers just forward that contract.
# ---------------------------------------------------------------------------


class TestEmbeddingHelpers:

    def _make_use_case(self):
        from graphbuilder.application.use_cases.document_processing import ProcessDocumentUseCase
        config = MagicMock()
        config.processing.chunk_size = 512
        config.processing.overlap_size = 50
        uc = ProcessDocumentUseCase(
            config=config,
            document_repo=AsyncMock(),
            graph_repo=AsyncMock(),
            llm_service=AsyncMock(),
            content_extractor=AsyncMock(),
        )
        return uc

    def test_embed_returns_none_when_sentence_transformers_missing(self, monkeypatch):
        """When the factory cannot load a model it returns None — the
        use-case helpers must propagate that without raising."""
        from graphbuilder.infrastructure.services import embedding_factory

        monkeypatch.setattr(embedding_factory, "get_model", lambda: None)

        uc = self._make_use_case()
        entity = GraphEntity(name="Test", entity_type=EntityType.CONCEPT)
        assert uc._embed_entity_text(entity) is None
        assert uc._embed_text("anything") is None

    def test_embed_entity_text_uses_name_and_description(self, monkeypatch):
        from graphbuilder.infrastructure.services import embedding_factory

        captured: dict = {}

        def fake_embed(text: str):
            captured["text"] = text
            return [0.1, 0.2, 0.3]

        monkeypatch.setattr(embedding_factory, "embed", fake_embed)

        uc = self._make_use_case()
        entity = GraphEntity(
            name="Aspirin",
            entity_type=EntityType.PRODUCT,
            description="pain reliever",
        )
        assert uc._embed_entity_text(entity) == [0.1, 0.2, 0.3]
        assert "Aspirin" in captured["text"]
        assert "pain reliever" in captured["text"]

    def test_embed_text_returns_none_for_empty_string(self, monkeypatch):
        """``embedding_factory.embed`` short-circuits on empty input —
        ``_embed_text`` inherits that without any extra check of its own."""
        from graphbuilder.infrastructure.services import embedding_factory

        # Real factory.embed already returns None for "", but for an isolated
        # unit test we don't want to depend on a loaded model — patch it.
        called: dict = {"hits": 0}

        def fake_embed(text: str):
            called["hits"] += 1
            return [0.1] if (text or "").strip() else None

        monkeypatch.setattr(embedding_factory, "embed", fake_embed)

        uc = self._make_use_case()
        assert uc._embed_text("") is None

    def test_embed_text_returns_list(self, monkeypatch):
        from graphbuilder.infrastructure.services import embedding_factory

        monkeypatch.setattr(embedding_factory, "embed", lambda text: [0.5, 0.6])

        uc = self._make_use_case()
        assert uc._embed_text("some text") == [0.5, 0.6]
