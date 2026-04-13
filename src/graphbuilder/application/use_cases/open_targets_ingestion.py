"""
Open Targets Ingestion Use Case.

Orchestrates the full pipeline for pulling disease–target association data
from the Open Targets Platform and persisting it as a knowledge graph:

    1. Fetch disease metadata + paginated target associations from the API.
    2. Map disease → GraphEntity (EntityType.CONCEPT)
       Map each target → GraphEntity (EntityType.CONCEPT)
       Map each association → GraphRelationship (RelationshipType.RELATED_TO)
    3. Save all entities and relationships via GraphRepositoryInterface.
    4. Return a ProcessingResult with rich metrics.

No live API or database call is made in this module; both are injected as
interfaces so the use case is fully unit-testable.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from ...domain.models.graph_models import (
    EntityType,
    GraphEntity,
    GraphRelationship,
    RelationshipType,
)
from ...domain.models.processing_models import ProcessingResult
from ...infrastructure.config.settings import GraphBuilderConfig
from ...infrastructure.external.open_targets_client import (
    IngestResult,
    OpenTargetsClient,
)
from ...infrastructure.repositories.graph_repository import GraphRepositoryInterface


@dataclass
class IngestionConfig:
    """Runtime parameters for a single Open Targets ingestion run."""

    disease_id: str
    max_associations: int = 500
    min_association_score: float = 0.0  # inclusive lower bound, 0 = keep all
    tag: str = "open-targets"


class OpenTargetsIngestionUseCase:
    """
    Ingest disease–target associations from Open Targets into the graph store.

    Parameters
    ----------
    config:
        Application-level configuration (used for API/client settings).
    graph_repo:
        Repository for persisting entities and relationships.
    client:
        Optional pre-constructed ``OpenTargetsClient``.  When ``None`` a new
        client is created from ``config`` settings on each ``execute`` call.
    """

    def __init__(
        self,
        config: GraphBuilderConfig,
        graph_repo: GraphRepositoryInterface,
        client: Optional[OpenTargetsClient] = None,
    ) -> None:
        self.config = config
        self.graph_repo = graph_repo
        self._client = client
        self.logger = logging.getLogger(self.__class__.__name__)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def execute(self, ingestion_config: IngestionConfig) -> ProcessingResult:
        """
        Run the full ingestion pipeline.

        Returns a ``ProcessingResult`` whose ``data`` dict contains:

        * ``disease_id`` — the requested EFO identifier
        * ``disease_name`` — human-readable disease name
        * ``entities_created`` — int
        * ``relationships_created`` — int
        * ``associations_fetched`` — int
        * ``total_associations_available`` — int from the API
        """
        start = datetime.now(timezone.utc)

        try:
            # ── 1. Fetch from Open Targets ──────────────────────────────
            ingest_result = await self._fetch(ingestion_config)

            if not ingest_result.success:
                return ProcessingResult(
                    success=False,
                    message=f"Open Targets fetch failed: {'; '.join(ingest_result.errors)}",
                    errors=ingest_result.errors,
                )

            # ── 2. Filter by score ──────────────────────────────────────
            associations = [
                a
                for a in ingest_result.associations
                if a.association_score >= ingestion_config.min_association_score
            ]

            # ── 3. Build domain objects ─────────────────────────────────
            disease_entity = self._build_disease_entity(
                ingest_result, ingestion_config
            )
            target_entities = [
                self._build_target_entity(assoc, ingestion_config)
                for assoc in associations
            ]
            relationships = [
                self._build_relationship(disease_entity, te, assoc)
                for te, assoc in zip(target_entities, associations)
            ]

            # ── 4. Persist ──────────────────────────────────────────────
            await self.graph_repo.save_entity(disease_entity)

            entities_saved = 1
            rels_saved = 0

            for entity in target_entities:
                await self.graph_repo.save_entity(entity)
                entities_saved += 1

            for rel in relationships:
                await self.graph_repo.save_relationship(rel)
                rels_saved += 1

            # ── 5. Build result ─────────────────────────────────────────
            elapsed = (datetime.now(timezone.utc) - start).total_seconds()

            result = ProcessingResult(
                success=True,
                message=(
                    f"Ingested {entities_saved} entities and {rels_saved} "
                    f"relationships for disease {ingestion_config.disease_id}"
                ),
                data={
                    "disease_id": ingestion_config.disease_id,
                    "disease_name": ingest_result.disease.name,
                    "entities_created": entities_saved,
                    "relationships_created": rels_saved,
                    "associations_fetched": len(associations),
                    "total_associations_available": ingest_result.total_associations,
                },
                processing_time=elapsed,
            )
            result.add_metric("entities_created", entities_saved)
            result.add_metric("relationships_created", rels_saved)
            result.add_metric("associations_fetched", len(associations))
            result.add_metric("processing_time", elapsed)
            return result

        except Exception as exc:
            self.logger.error(
                "Open Targets ingestion error: %s", exc, exc_info=True
            )
            return ProcessingResult(
                success=False,
                message=f"Ingestion failed: {exc}",
                errors=[str(exc)],
            )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _fetch(self, cfg: IngestionConfig) -> IngestResult:
        """Fetch from Open Targets, managing the client lifecycle."""
        if self._client is not None:
            return await self._client.fetch_disease(
                cfg.disease_id, max_associations=cfg.max_associations
            )

        # Create a short-lived client if none was injected.
        async with OpenTargetsClient() as client:
            return await client.fetch_disease(
                cfg.disease_id, max_associations=cfg.max_associations
            )

    @staticmethod
    def _build_disease_entity(
        ingest_result: IngestResult, cfg: IngestionConfig
    ) -> GraphEntity:
        disease = ingest_result.disease
        entity = GraphEntity(
            name=disease.name,
            entity_type=EntityType.CONCEPT,
            description=disease.description or None,
        )
        entity.add_external_id("open_targets", disease.id)
        entity.metadata.add_tag(cfg.tag)
        entity.metadata.add_tag("disease")
        entity.metadata.source_trust = "reviewed"
        entity.metadata.source_system = "open_targets"
        entity.metadata.add_annotation(
            "therapeutic_areas",
            [ta.get("name") for ta in disease.therapeutic_areas],
        )
        for syn in disease.synonyms:
            entity.add_alias(syn)
        return entity

    @staticmethod
    def _build_target_entity(
        assoc: Any, cfg: IngestionConfig
    ) -> GraphEntity:
        entity = GraphEntity(
            name=assoc.target_symbol or assoc.target_name,
            entity_type=EntityType.CONCEPT,
            description=(
                assoc.function_descriptions[0]
                if assoc.function_descriptions
                else None
            ),
        )
        entity.add_external_id("ensembl", assoc.target_id)
        entity.metadata.add_tag(cfg.tag)
        entity.metadata.add_tag("target")
        entity.metadata.source_trust = "reviewed"
        entity.metadata.source_system = "open_targets"
        entity.metadata.add_tag(assoc.biotype)
        entity.properties["biotype"] = assoc.biotype
        entity.properties["approved_name"] = assoc.target_name
        if assoc.target_symbol != assoc.target_name:
            entity.add_alias(assoc.target_name)
        return entity

    @staticmethod
    def _build_relationship(
        disease_entity: GraphEntity,
        target_entity: GraphEntity,
        assoc: Any,
    ) -> GraphRelationship:
        rel = GraphRelationship(
            source_entity_id=disease_entity.id,
            target_entity_id=target_entity.id,
            relationship_type=RelationshipType.RELATED_TO,
            description=f"Association score: {assoc.association_score:.3f}",
            strength=min(max(assoc.association_score, 0.0), 1.0),
        )
        rel.properties["association_score"] = assoc.association_score
        rel.properties["datatype_scores"] = assoc.datatype_scores
        rel.properties["source"] = "open_targets"
        rel.metadata.source_trust = "reviewed"
        rel.metadata.source_system = "open_targets"
        return rel
