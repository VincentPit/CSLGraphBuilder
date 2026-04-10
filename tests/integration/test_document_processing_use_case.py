"""
Integration tests for application/use_cases/document_processing.py

Uses AsyncMock for all external dependencies (repos, LLM service, content extractor)
so no live Neo4j or OpenAI connection is required.
"""

from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from graphbuilder.application.use_cases.document_processing import ProcessDocumentUseCase
from graphbuilder.domain.models.graph_models import ProcessingStatus, SourceDocument
from graphbuilder.domain.models.processing_models import ProcessingResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config():
    config = MagicMock()
    config.processing.chunk_size = 512
    config.processing.overlap_size = 50
    return config


def _make_source_document(doc_id: str = "doc-1") -> SourceDocument:
    doc = MagicMock(spec=SourceDocument)
    doc.id = doc_id
    doc.title = "Test Document"
    doc.content_length = 1000
    doc.processed_chunks = 5
    doc.extracted_entities = 10
    doc.extracted_relationships = 5
    doc.processing_status = ProcessingStatus.PENDING
    return doc


def _make_use_case(
    document=None,
    pipeline_success=True,
):
    """
    Build a fully-mocked ProcessDocumentUseCase.

    Parameters
    ----------
    document:
        The SourceDocument returned by document_repo.get_by_id.
        Pass None to simulate "document not found".
    pipeline_success:
        Controls whether _execute_pipeline returns a successful result.
    """
    config = _make_config()
    document_repo = AsyncMock()
    graph_repo = AsyncMock()
    llm_service = AsyncMock()
    content_extractor = AsyncMock()

    document_repo.get_by_id.return_value = document
    document_repo.update.return_value = document

    use_case = ProcessDocumentUseCase(
        config=config,
        document_repo=document_repo,
        graph_repo=graph_repo,
        llm_service=llm_service,
        content_extractor=content_extractor,
    )

    # Mock the internal pipeline methods so we don't need a Neo4j connection
    mock_pipeline = MagicMock()
    mock_pipeline.id = "pipeline-1"
    mock_pipeline.get_execution_summary.return_value = {}

    pipeline_result = ProcessingResult(
        success=pipeline_success,
        message="Pipeline complete" if pipeline_success else "Pipeline failed",
    )

    use_case._create_processing_pipeline = AsyncMock(return_value=mock_pipeline)
    use_case._execute_pipeline = AsyncMock(return_value=pipeline_result)

    return use_case, document_repo


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestProcessDocumentUseCase:

    async def test_document_not_found_returns_failure(self):
        use_case, _ = _make_use_case(document=None)
        result = await use_case.execute("missing-id")

        assert result.success is False
        assert any("not found" in e.lower() for e in result.errors)

    async def test_document_found_returns_success(self):
        doc = _make_source_document()
        use_case, document_repo = _make_use_case(document=doc)

        result = await use_case.execute("doc-1")

        assert result.success is True
        assert result.message == "Document processing completed successfully"

    async def test_document_status_updated_to_in_progress(self):
        doc = _make_source_document()
        use_case, document_repo = _make_use_case(document=doc)

        await use_case.execute("doc-1")

        doc.update_processing_status.assert_any_call(ProcessingStatus.IN_PROGRESS)

    async def test_document_status_updated_to_completed_on_success(self):
        doc = _make_source_document()
        use_case, _ = _make_use_case(document=doc, pipeline_success=True)

        await use_case.execute("doc-1")

        doc.update_processing_status.assert_called_with(ProcessingStatus.COMPLETED)

    async def test_document_status_updated_to_failed_on_pipeline_failure(self):
        doc = _make_source_document()
        use_case, _ = _make_use_case(document=doc, pipeline_success=False)

        result = await use_case.execute("doc-1")

        assert result.success is False
        doc.update_processing_status.assert_called_with(
            ProcessingStatus.FAILED, "Pipeline failed"
        )

    async def test_document_repo_update_called_after_processing(self):
        doc = _make_source_document()
        use_case, document_repo = _make_use_case(document=doc)

        await use_case.execute("doc-1")

        assert document_repo.update.await_count >= 1

    async def test_result_data_contains_document_id(self):
        doc = _make_source_document("doc-42")
        use_case, _ = _make_use_case(document=doc)

        result = await use_case.execute("doc-42")

        assert result.data["document_id"] == "doc-42"

    async def test_processing_time_metric_recorded(self):
        doc = _make_source_document()
        use_case, _ = _make_use_case(document=doc)

        result = await use_case.execute("doc-1")

        assert "processing_time_seconds" in result.metrics
        assert result.metrics["processing_time_seconds"] >= 0

    async def test_unexpected_exception_returns_failure(self):
        doc = _make_source_document()
        use_case, document_repo = _make_use_case(document=doc)

        # Simulate an unexpected error inside _execute_pipeline
        use_case._execute_pipeline.side_effect = RuntimeError("Unexpected boom")

        result = await use_case.execute("doc-1")

        assert result.success is False
        assert any("Unexpected" in e for e in result.errors)

    async def test_create_pipeline_called_with_document(self):
        doc = _make_source_document()
        use_case, _ = _make_use_case(document=doc)

        await use_case.execute("doc-1", extraction_config={"key": "value"})

        use_case._create_processing_pipeline.assert_awaited_once_with(
            doc, {"key": "value"}
        )
