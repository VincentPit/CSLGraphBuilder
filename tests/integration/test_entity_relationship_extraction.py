"""
Integration tests for entity extraction with LLM-powered dedup
and relationship extraction with LLM entity resolution + relationship dedup.

All external dependencies (repos, LLM service) are fully mocked.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from graphbuilder.application.use_cases.document_processing import ProcessDocumentUseCase
from graphbuilder.domain.models.graph_models import (
    DocumentChunk,
    EntityType,
    GraphEntity,
    GraphRelationship,
    RelationshipType,
)
from graphbuilder.domain.models.processing_models import (
    ProcessingResult,
    ProcessingTask,
    TaskType,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config() -> MagicMock:
    config = MagicMock()
    config.processing.chunk_size = 512
    config.processing.overlap_size = 50
    return config


def _make_chunk(
    chunk_id: str = "chunk-1",
    document_id: str = "doc-1",
    content: str = "TNF-alpha inhibits cell growth.",
) -> DocumentChunk:
    c = DocumentChunk(
        content=content,
        document_id=document_id,
        chunk_index=0,
        token_count=10,
        character_count=len(content),
    )
    c.id = chunk_id
    return c


def _make_entity(
    eid: str,
    name: str,
    etype: EntityType = EntityType.CONCEPT,
    doc_id: str = "doc-1",
    description: str = "",
) -> GraphEntity:
    e = GraphEntity(name=name, entity_type=etype, description=description)
    e.id = eid
    e.source_document_ids = [doc_id]
    e.source_chunk_ids = ["chunk-1"]
    return e


def _make_task(document_id: str = "doc-1") -> ProcessingTask:
    task = ProcessingTask(
        task_type=TaskType.ENTITY_EXTRACTION,
        name="test-task",
        input_data={"document_id": document_id},
    )
    return task


def _make_use_case():
    """Build ProcessDocumentUseCase with fully mocked dependencies."""
    config = _make_config()
    document_repo = AsyncMock()
    graph_repo = AsyncMock()
    llm_service = AsyncMock()
    content_extractor = AsyncMock()

    uc = ProcessDocumentUseCase(
        config=config,
        document_repo=document_repo,
        graph_repo=graph_repo,
        llm_service=llm_service,
        content_extractor=content_extractor,
    )
    # Disable real embedding model
    uc._get_embedding_model = MagicMock(return_value=None)
    return uc, document_repo, graph_repo, llm_service


# ---------------------------------------------------------------------------
# Entity extraction with LLM dedup
# ---------------------------------------------------------------------------


class TestEntityExtractionWithLLMDedup:

    async def test_entities_saved_when_no_vector_candidates(self):
        """With no embedding model, no vector candidates → all entities saved as new."""
        uc, doc_repo, graph_repo, llm_service = _make_use_case()

        chunk = _make_chunk()
        doc_repo.get_chunks_by_document_id.return_value = [chunk]

        llm_service.extract_entities.return_value = ProcessingResult(
            success=True,
            message="ok",
            data={
                "entities": [
                    {"name": "TNF-alpha", "type": "PROTEIN", "description": "cytokine"},
                ]
            },
        )

        task = _make_task()
        result = await uc._execute_entity_extraction(task)

        assert result.success is True
        assert result.data["entities_extracted"] == 1
        assert result.data["merged_entities"] == 0
        graph_repo.save_entity.assert_called_once()

    async def test_entities_merged_when_llm_confirms_duplicate(self):
        """When vector pre-filter finds candidates and LLM confirms a match,
        the entity is merged instead of saved as new."""
        uc, doc_repo, graph_repo, llm_service = _make_use_case()

        chunk = _make_chunk()
        doc_repo.get_chunks_by_document_id.return_value = [chunk]

        existing_entity = _make_entity("e-existing", "Tumor Necrosis Factor Alpha", EntityType.PROTEIN)

        llm_service.extract_entities.return_value = ProcessingResult(
            success=True,
            message="ok",
            data={
                "entities": [
                    {"name": "TNF-alpha", "type": "PROTEIN", "description": "cytokine"},
                ]
            },
        )

        # Enable embedding model + vector search
        import numpy as np
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([0.1, 0.2, 0.3])
        uc._get_embedding_model = MagicMock(return_value=mock_model)

        graph_repo.vector_search_entities.return_value = [
            (existing_entity, 0.5),
        ]

        llm_service.resolve_entity_duplicates.return_value = ProcessingResult(
            success=True,
            message="1 match",
            data={
                "matches": [
                    {
                        "new_name": "TNF-alpha",
                        "existing_name": "Tumor Necrosis Factor Alpha",
                        "confidence": 0.95,
                        "reasoning": "abbreviation",
                    }
                ]
            },
        )

        task = _make_task()
        result = await uc._execute_entity_extraction(task)

        assert result.success is True
        assert result.data["merged_entities"] == 1
        # The existing entity should be saved (merged)
        graph_repo.save_entity.assert_called_once()
        saved = graph_repo.save_entity.call_args[0][0]
        assert saved.id == "e-existing"

    async def test_low_confidence_match_not_merged(self):
        """LLM match with confidence < 0.7 is not merged."""
        uc, doc_repo, graph_repo, llm_service = _make_use_case()

        chunk = _make_chunk()
        doc_repo.get_chunks_by_document_id.return_value = [chunk]

        existing = _make_entity("e-1", "SomeProtein", EntityType.PROTEIN)

        llm_service.extract_entities.return_value = ProcessingResult(
            success=True,
            message="ok",
            data={"entities": [{"name": "OtherProtein", "type": "PROTEIN", "description": ""}]},
        )

        import numpy as np
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([0.1])
        uc._get_embedding_model = MagicMock(return_value=mock_model)
        graph_repo.vector_search_entities.return_value = [(existing, 0.45)]

        llm_service.resolve_entity_duplicates.return_value = ProcessingResult(
            success=True,
            message="1 match",
            data={
                "matches": [
                    {
                        "new_name": "OtherProtein",
                        "existing_name": "SomeProtein",
                        "confidence": 0.5,
                        "reasoning": "weak",
                    }
                ]
            },
        )

        task = _make_task()
        result = await uc._execute_entity_extraction(task)

        assert result.data["merged_entities"] == 0
        assert result.data["entities_extracted"] == 1

    async def test_type_mismatch_prevents_merge(self):
        """Even with high-confidence LLM match, different entity types are not merged."""
        uc, doc_repo, graph_repo, llm_service = _make_use_case()

        chunk = _make_chunk()
        doc_repo.get_chunks_by_document_id.return_value = [chunk]

        # Existing is ORGANIZATION, extracted is PROTEIN → should not merge
        existing = _make_entity("e-1", "ABC", EntityType.ORGANIZATION)

        llm_service.extract_entities.return_value = ProcessingResult(
            success=True,
            message="ok",
            data={"entities": [{"name": "ABC", "type": "PROTEIN", "description": ""}]},
        )

        import numpy as np
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([0.1])
        uc._get_embedding_model = MagicMock(return_value=mock_model)
        graph_repo.vector_search_entities.return_value = [(existing, 0.9)]

        llm_service.resolve_entity_duplicates.return_value = ProcessingResult(
            success=True,
            message="1 match",
            data={
                "matches": [
                    {"new_name": "ABC", "existing_name": "ABC", "confidence": 0.99}
                ]
            },
        )

        task = _make_task()
        result = await uc._execute_entity_extraction(task)

        assert result.data["merged_entities"] == 0

    async def test_no_chunks_returns_failure(self):
        uc, doc_repo, graph_repo, llm_service = _make_use_case()
        doc_repo.get_chunks_by_document_id.return_value = []

        task = _make_task()
        result = await uc._execute_entity_extraction(task)

        assert result.success is False

    async def test_llm_extraction_failure_skips_chunk(self):
        """If LLM extraction fails for a chunk, processing continues."""
        uc, doc_repo, graph_repo, llm_service = _make_use_case()

        doc_repo.get_chunks_by_document_id.return_value = [
            _make_chunk("c1", content="chunk 1"),
            _make_chunk("c2", content="chunk 2"),
        ]

        llm_service.extract_entities.side_effect = [
            ProcessingResult(success=False, message="fail"),
            ProcessingResult(
                success=True,
                message="ok",
                data={"entities": [{"name": "X", "type": "Concept", "description": ""}]},
            ),
        ]

        task = _make_task()
        result = await uc._execute_entity_extraction(task)

        assert result.success is True
        assert result.data["entities_extracted"] == 1


# ---------------------------------------------------------------------------
# Relationship extraction with LLM entity resolution + relationship dedup
# ---------------------------------------------------------------------------


class TestRelationshipExtractionWithLLMDedup:

    async def _setup(self, *, relationships_data=None, existing_rels=None):
        """Common setup: single chunk, two entities, LLM returns relationships."""
        uc, doc_repo, graph_repo, llm_service = _make_use_case()

        chunk = _make_chunk()
        doc_repo.get_chunks_by_document_id.return_value = [chunk]

        e1 = _make_entity("e-1", "Aspirin", EntityType.PRODUCT)
        e2 = _make_entity("e-2", "Headache", EntityType.CONCEPT)

        graph_repo.get_all_entities.return_value = {
            "e-1": e1,
            "e-2": e2,
        }

        if relationships_data is None:
            relationships_data = [
                {
                    "source_entity": "Aspirin",
                    "target_entity": "Headache",
                    "relationship_type": "RELATED_TO",
                    "description": "Aspirin treats headaches",
                    "confidence": 0.9,
                }
            ]

        llm_service.extract_relationships.return_value = ProcessingResult(
            success=True,
            message="ok",
            data={"relationships": relationships_data},
        )

        graph_repo.get_entity_relationships.return_value = existing_rels or []

        return uc, doc_repo, graph_repo, llm_service

    async def test_basic_relationship_saved(self):
        uc, _, graph_repo, _ = await self._setup()

        task = _make_task()
        task.task_type = TaskType.RELATIONSHIP_EXTRACTION
        result = await uc._execute_relationship_extraction(task)

        assert result.success is True
        assert result.data["relationships_extracted"] == 1
        graph_repo.save_relationship.assert_called_once()

    async def test_unresolved_entity_resolved_by_llm(self):
        """If a relationship references an entity name not in the name→ID map,
        LLM resolution resolves it so the relationship can be saved."""
        uc, doc_repo, graph_repo, llm_service = _make_use_case()

        chunk = _make_chunk()
        doc_repo.get_chunks_by_document_id.return_value = [chunk]

        e1 = _make_entity("e-1", "Aspirin", EntityType.PRODUCT)
        e2 = _make_entity("e-2", "Headache", EntityType.CONCEPT)
        graph_repo.get_all_entities.return_value = {"e-1": e1, "e-2": e2}

        # LLM returns relationship with unresolved name "ASA"
        llm_service.extract_relationships.return_value = ProcessingResult(
            success=True,
            message="ok",
            data={
                "relationships": [
                    {
                        "source_entity": "ASA",
                        "target_entity": "Headache",
                        "relationship_type": "RELATED_TO",
                        "description": "ASA treats headache",
                        "confidence": 0.9,
                    }
                ]
            },
        )

        # Vector search returns Aspirin as candidate
        import numpy as np
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([0.1])
        uc._get_embedding_model = MagicMock(return_value=mock_model)

        graph_repo.vector_search_entities.return_value = [(e1, 0.5)]

        llm_service.resolve_entity_duplicates.return_value = ProcessingResult(
            success=True,
            message="resolved",
            data={
                "matches": [
                    {
                        "new_name": "ASA",
                        "existing_name": "Aspirin",
                        "confidence": 0.9,
                        "reasoning": "abbreviation",
                    }
                ]
            },
        )

        graph_repo.get_entity_relationships.return_value = []

        task = _make_task()
        result = await uc._execute_relationship_extraction(task)

        assert result.success is True
        assert result.data["relationships_extracted"] == 1
        graph_repo.save_relationship.assert_called_once()

    async def test_semantic_duplicate_relationship_merged(self):
        """When LLM says ACTIVATES ≈ STIMULATES, the existing relationship is
        updated instead of creating a new one."""
        existing_rel = GraphRelationship(
            source_entity_id="e-1",
            target_entity_id="e-2",
            relationship_type=RelationshipType.PART_OF,
            description="Aspirin stimulates...",
            source_chunk_ids=["old-chunk"],
            source_document_ids=["doc-0"],
        )
        existing_rel.id = "r-existing"

        uc, _, graph_repo, llm_service = await self._setup(
            relationships_data=[
                {
                    "source_entity": "Aspirin",
                    "target_entity": "Headache",
                    "relationship_type": "INHIBITS",
                    "description": "Aspirin inhibits headache",
                    "confidence": 0.8,
                }
            ],
            existing_rels=[existing_rel],
        )

        llm_service.check_relationship_duplicates.return_value = ProcessingResult(
            success=True,
            message="dup found",
            data={"duplicate_of": 0, "confidence": 0.85, "reasoning": "same fact"},
        )

        task = _make_task()
        result = await uc._execute_relationship_extraction(task)

        assert result.success is True
        assert result.data["merged_relationships"] == 1
        # The existing rel should be saved (merged), not a new one
        saved = graph_repo.save_relationship.call_args[0][0]
        assert saved.id == "r-existing"

    async def test_exact_type_match_skips_llm_dedup(self):
        """If an existing relationship has the exact same type, LLM dedup is not called."""
        existing_rel = GraphRelationship(
            source_entity_id="e-1",
            target_entity_id="e-2",
            relationship_type=RelationshipType.RELATED_TO,
            source_chunk_ids=["old-chunk"],
            source_document_ids=["doc-0"],
        )

        uc, _, graph_repo, llm_service = await self._setup(existing_rels=[existing_rel])

        task = _make_task()
        result = await uc._execute_relationship_extraction(task)

        llm_service.check_relationship_duplicates.assert_not_called()
        assert result.success is True

    async def test_no_entities_for_doc_returns_early(self):
        uc, doc_repo, graph_repo, llm_service = _make_use_case()

        doc_repo.get_chunks_by_document_id.return_value = [_make_chunk()]
        graph_repo.get_all_entities.return_value = {}

        task = _make_task()
        result = await uc._execute_relationship_extraction(task)

        assert result.success is True
        assert result.data["relationships_extracted"] == 0

    async def test_missing_document_id_returns_failure(self):
        uc, _, _, _ = _make_use_case()
        task = ProcessingTask(
            task_type=TaskType.RELATIONSHIP_EXTRACTION,
            name="test",
            input_data={},
        )
        result = await uc._execute_relationship_extraction(task)
        assert result.success is False

    async def test_unresolved_entities_skipped_gracefully(self):
        """Relationships with entity names that can't be resolved are skipped."""
        uc, doc_repo, graph_repo, llm_service = _make_use_case()

        chunk = _make_chunk()
        doc_repo.get_chunks_by_document_id.return_value = [chunk]

        e1 = _make_entity("e-1", "Aspirin", EntityType.PRODUCT)
        graph_repo.get_all_entities.return_value = {"e-1": e1}

        llm_service.extract_relationships.return_value = ProcessingResult(
            success=True,
            message="ok",
            data={
                "relationships": [
                    {
                        "source_entity": "Aspirin",
                        "target_entity": "UnknownEntity",
                        "relationship_type": "RELATED_TO",
                        "description": "...",
                        "confidence": 0.9,
                    }
                ]
            },
        )

        graph_repo.get_entity_relationships.return_value = []

        task = _make_task()
        result = await uc._execute_relationship_extraction(task)

        assert result.success is True
        assert result.data["relationships_extracted"] == 0
        graph_repo.save_relationship.assert_not_called()
