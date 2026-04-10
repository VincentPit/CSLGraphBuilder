"""
Integration tests for infrastructure/services/llm_service.py

Patches _initialize_client so no real OpenAI credentials are needed.
All API calls are handled via AsyncMock on the client.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from graphbuilder.infrastructure.services.llm_service import AdvancedLLMService
from graphbuilder.infrastructure.config.settings import LLMProvider


# ---------------------------------------------------------------------------
# Helpers
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
    """Return a client whose chat.completions.create is an AsyncMock."""
    client = MagicMock()
    client.chat = MagicMock()
    client.chat.completions = MagicMock()
    client.chat.completions.create = AsyncMock()
    return client


def _make_api_response(content: str) -> MagicMock:
    """Build a minimal OpenAI-style chat completion response mock."""
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


def _make_service(provider: LLMProvider = LLMProvider.OPENAI) -> AdvancedLLMService:
    """Instantiate AdvancedLLMService with a fully mocked client."""
    config = _make_config(provider)
    with patch.object(AdvancedLLMService, "_initialize_client", return_value=_make_mock_client()):
        service = AdvancedLLMService(config)
    return service


# ---------------------------------------------------------------------------
# extract_entities
# ---------------------------------------------------------------------------


class TestExtractEntities:

    async def test_success_returns_entities(self):
        service = _make_service()
        payload = {
            "entities": [
                {
                    "name": "Alice",
                    "type": "PERSON",
                    "description": "A researcher",
                    "properties": {},
                    "confidence": 0.95,
                    "mentions": ["Alice"],
                }
            ],
            "metadata": {"total_entities": 1, "processing_notes": ""},
        }
        service.client.chat.completions.create.return_value = _make_api_response(
            json.dumps(payload)
        )

        result = await service.extract_entities("Alice is a researcher.")

        assert result.success is True
        assert len(result.data["entities"]) == 1
        assert result.data["entities"][0]["name"] == "Alice"

    async def test_success_records_metrics(self):
        service = _make_service()
        payload = {"entities": [], "metadata": {}}
        service.client.chat.completions.create.return_value = _make_api_response(
            json.dumps(payload)
        )

        result = await service.extract_entities("Some text.")

        assert "entities_extracted" in result.metrics
        assert "tokens_used" in result.metrics
        assert "processing_time" in result.metrics

    async def test_malformed_json_returns_failure(self):
        service = _make_service()
        service.client.chat.completions.create.return_value = _make_api_response(
            "not valid json {{{"
        )

        result = await service.extract_entities("Some text.")

        assert result.success is False
        assert len(result.errors) > 0

    async def test_missing_entities_field_returns_failure(self):
        service = _make_service()
        # Valid JSON but missing the "entities" key
        service.client.chat.completions.create.return_value = _make_api_response(
            json.dumps({"wrong_key": []})
        )

        result = await service.extract_entities("Some text.")

        assert result.success is False

    async def test_api_error_returns_failure(self):
        service = _make_service()
        service.client.chat.completions.create.side_effect = ConnectionError("timeout")

        result = await service.extract_entities("Some text.")

        assert result.success is False
        assert "Entity extraction failed" in result.message

    async def test_markdown_wrapped_json_is_parsed(self):
        """LLM sometimes returns JSON inside ```json ... ``` blocks."""
        service = _make_service()
        payload = {"entities": [], "metadata": {}}
        wrapped = f"```json\n{json.dumps(payload)}\n```"
        service.client.chat.completions.create.return_value = _make_api_response(wrapped)

        result = await service.extract_entities("Some text.")

        assert result.success is True

    async def test_config_temperature_forwarded(self):
        service = _make_service()
        payload = {"entities": [], "metadata": {}}
        service.client.chat.completions.create.return_value = _make_api_response(
            json.dumps(payload)
        )

        await service.extract_entities("Text.", config={"temperature": 0.7, "max_tokens": 500})

        call_kwargs = service.client.chat.completions.create.call_args.kwargs
        assert call_kwargs["temperature"] == 0.7
        assert call_kwargs["max_tokens"] == 500


# ---------------------------------------------------------------------------
# extract_relationships
# ---------------------------------------------------------------------------


class TestExtractRelationships:

    async def test_success_returns_relationships(self):
        service = _make_service()
        payload = {
            "relationships": [
                {
                    "source_entity": "Alice",
                    "target_entity": "ACME Corp",
                    "relationship_type": "WORKS_FOR",
                    "description": "Employment",
                    "confidence": 0.9,
                    "evidence": "Alice works at ACME",
                    "properties": {},
                }
            ],
            "metadata": {"total_relationships": 1, "processing_notes": ""},
        }
        service.client.chat.completions.create.return_value = _make_api_response(
            json.dumps(payload)
        )

        entities = [{"name": "Alice", "type": "PERSON", "description": ""}]
        result = await service.extract_relationships("Alice works at ACME.", entities)

        assert result.success is True
        assert len(result.data["relationships"]) == 1
        assert result.data["relationships"][0]["relationship_type"] == "WORKS_FOR"

    async def test_malformed_json_returns_failure(self):
        service = _make_service()
        service.client.chat.completions.create.return_value = _make_api_response("bad json")

        result = await service.extract_relationships("text.", [])

        assert result.success is False

    async def test_missing_relationships_field_returns_failure(self):
        service = _make_service()
        service.client.chat.completions.create.return_value = _make_api_response(
            json.dumps({"data": []})
        )

        result = await service.extract_relationships("text.", [])

        assert result.success is False

    async def test_metrics_recorded_on_success(self):
        service = _make_service()
        payload = {"relationships": [], "metadata": {}}
        service.client.chat.completions.create.return_value = _make_api_response(
            json.dumps(payload)
        )

        result = await service.extract_relationships("text.", [])

        assert result.metrics["relationships_extracted"] == 0


# ---------------------------------------------------------------------------
# summarize_content
# ---------------------------------------------------------------------------


class TestSummarizeContent:

    async def test_success_returns_summary(self):
        service = _make_service()
        payload = {
            "summary": "A brief overview.",
            "key_points": ["Point 1"],
            "entities_mentioned": [],
            "themes": ["research"],
            "word_count": 4,
            "metadata": {"original_length": 100, "compression_ratio": 0.5},
        }
        service.client.chat.completions.create.return_value = _make_api_response(
            json.dumps(payload)
        )

        result = await service.summarize_content("Long document text here.")

        assert result.success is True
        assert result.data["summary"] == "A brief overview."
        assert result.data["key_points"] == ["Point 1"]

    async def test_missing_summary_field_returns_failure(self):
        service = _make_service()
        service.client.chat.completions.create.return_value = _make_api_response(
            json.dumps({"wrong": "data"})
        )

        result = await service.summarize_content("text.")

        assert result.success is False


# ---------------------------------------------------------------------------
# _parse_json_response
# ---------------------------------------------------------------------------


class TestParseJsonResponse:
    """Direct unit tests for the JSON parser helper."""

    async def test_plain_json_is_parsed(self):
        service = _make_service()
        data = await service._parse_json_response('{"key": "value"}')
        assert data == {"key": "value"}

    async def test_markdown_wrapped_json_is_parsed(self):
        service = _make_service()
        data = await service._parse_json_response('```json\n{"key": "value"}\n```')
        assert data == {"key": "value"}

    async def test_invalid_json_raises_value_error(self):
        service = _make_service()
        with pytest.raises(ValueError, match="Invalid JSON"):
            await service._parse_json_response("not json")
