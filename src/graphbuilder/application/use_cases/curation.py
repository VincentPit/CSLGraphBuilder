"""
Manual Curation Use Case — human-in-the-loop feedback loop.

Allows domain experts to approve, reject, or correct entities and
relationships produced by the automated extraction pipeline.  Every
curation action is recorded as an immutable audit event so the graph
can be replayed or audited later.

Supported actions
-----------------
approve_entity       Mark an entity as human-verified.
reject_entity        Soft-delete an entity (marks it rejected, not removed).
correct_entity       Update entity name / description / properties.
approve_relationship Mark a relationship as human-verified.
reject_relationship  Soft-delete a relationship.
correct_relationship Update relationship type / description / strength.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from ...domain.models.graph_models import (
    EntityType,
    GraphEntity,
    GraphRelationship,
    RelationshipType,
)
from ...domain.models.processing_models import ProcessingResult
from ...infrastructure.config.settings import GraphBuilderConfig
from ...infrastructure.repositories.graph_repository import GraphRepositoryInterface


class CurationAction(Enum):
    APPROVE_ENTITY        = "approve_entity"
    REJECT_ENTITY         = "reject_entity"
    CORRECT_ENTITY        = "correct_entity"
    APPROVE_RELATIONSHIP  = "approve_relationship"
    REJECT_RELATIONSHIP   = "reject_relationship"
    CORRECT_RELATIONSHIP  = "correct_relationship"


@dataclass
class CurationEvent:
    """Immutable record of a single curation action."""

    action: CurationAction
    target_id: str           # entity or relationship ID
    curator: str             # username / identifier of the human curator
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    reason: str = ""
    corrections: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action.value,
            "target_id": self.target_id,
            "curator": self.curator,
            "timestamp": self.timestamp.isoformat(),
            "reason": self.reason,
            "corrections": self.corrections,
        }


@dataclass
class CurationRequest:
    """Input for a single curation session."""

    curator: str
    events: List[CurationEvent] = field(default_factory=list)

    def approve_entity(self, entity_id: str, reason: str = "") -> "CurationRequest":
        self.events.append(CurationEvent(
            action=CurationAction.APPROVE_ENTITY,
            target_id=entity_id,
            curator=self.curator,
            reason=reason,
        ))
        return self

    def reject_entity(self, entity_id: str, reason: str = "") -> "CurationRequest":
        self.events.append(CurationEvent(
            action=CurationAction.REJECT_ENTITY,
            target_id=entity_id,
            curator=self.curator,
            reason=reason,
        ))
        return self

    def correct_entity(
        self,
        entity_id: str,
        corrections: Dict[str, Any],
        reason: str = "",
    ) -> "CurationRequest":
        """
        *corrections* may contain any subset of: name, description, properties.
        """
        self.events.append(CurationEvent(
            action=CurationAction.CORRECT_ENTITY,
            target_id=entity_id,
            curator=self.curator,
            reason=reason,
            corrections=corrections,
        ))
        return self

    def approve_relationship(self, rel_id: str, reason: str = "") -> "CurationRequest":
        self.events.append(CurationEvent(
            action=CurationAction.APPROVE_RELATIONSHIP,
            target_id=rel_id,
            curator=self.curator,
            reason=reason,
        ))
        return self

    def reject_relationship(self, rel_id: str, reason: str = "") -> "CurationRequest":
        self.events.append(CurationEvent(
            action=CurationAction.REJECT_RELATIONSHIP,
            target_id=rel_id,
            curator=self.curator,
            reason=reason,
        ))
        return self

    def correct_relationship(
        self,
        rel_id: str,
        corrections: Dict[str, Any],
        reason: str = "",
    ) -> "CurationRequest":
        """
        *corrections* may contain any subset of:
        relationship_type (str), description, strength (float).
        """
        self.events.append(CurationEvent(
            action=CurationAction.CORRECT_RELATIONSHIP,
            target_id=rel_id,
            curator=self.curator,
            reason=reason,
            corrections=corrections,
        ))
        return self


@dataclass
class CurationSummary:
    """Outcome summary of a ``CurationUseCase.execute`` call."""

    total_events: int = 0
    approved_entities: int = 0
    rejected_entities: int = 0
    corrected_entities: int = 0
    approved_relationships: int = 0
    rejected_relationships: int = 0
    corrected_relationships: int = 0
    skipped: int = 0
    errors: List[str] = field(default_factory=list)
    audit_log: List[Dict[str, Any]] = field(default_factory=list)


class CurationUseCase:
    """
    Apply a batch of human curation events to the knowledge graph.

    Parameters
    ----------
    config:
        Application-level configuration.
    graph_repo:
        Repository for reading and updating entities and relationships.
    """

    # Annotation key used to record curation state inside entity metadata
    _CURATED_KEY  = "curated"
    _REJECTED_KEY = "rejected"
    _CURATOR_KEY  = "curated_by"

    def __init__(
        self,
        config: GraphBuilderConfig,
        graph_repo: GraphRepositoryInterface,
    ) -> None:
        self.config = config
        self.graph_repo = graph_repo
        self.logger = logging.getLogger(self.__class__.__name__)

    async def execute(self, request: CurationRequest) -> ProcessingResult:
        """
        Process all events in *request* and return a ``ProcessingResult``.

        ``result.data["summary"]`` contains a ``CurationSummary`` dict.
        ``result.data["audit_log"]`` contains every processed event as a list
        of dicts suitable for writing to a file or external audit store.
        """
        start = datetime.now(timezone.utc)
        summary = CurationSummary(total_events=len(request.events))

        for event in request.events:
            try:
                await self._apply(event, summary)
                summary.audit_log.append(event.to_dict())
            except Exception as exc:
                msg = f"Failed to apply event {event.action.value} on {event.target_id}: {exc}"
                self.logger.error(msg, exc_info=True)
                summary.errors.append(msg)
                summary.skipped += 1

        elapsed = (datetime.now(timezone.utc) - start).total_seconds()
        success = len(summary.errors) == 0

        result = ProcessingResult(
            success=success,
            message=(
                f"Curation complete: {summary.total_events - summary.skipped} applied, "
                f"{summary.skipped} skipped"
            ),
            data={
                "summary": {
                    "total_events": summary.total_events,
                    "approved_entities": summary.approved_entities,
                    "rejected_entities": summary.rejected_entities,
                    "corrected_entities": summary.corrected_entities,
                    "approved_relationships": summary.approved_relationships,
                    "rejected_relationships": summary.rejected_relationships,
                    "corrected_relationships": summary.corrected_relationships,
                    "skipped": summary.skipped,
                    "errors": summary.errors,
                },
                "audit_log": summary.audit_log,
                "curator": request.curator,
            },
            errors=summary.errors,
            processing_time=elapsed,
        )
        result.add_metric("events_applied", summary.total_events - summary.skipped)
        result.add_metric("events_skipped", summary.skipped)
        return result

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _apply(
        self, event: CurationEvent, summary: CurationSummary
    ) -> None:
        action = event.action

        if action == CurationAction.APPROVE_ENTITY:
            await self._annotate_entity(event, curated=True, rejected=False)
            summary.approved_entities += 1

        elif action == CurationAction.REJECT_ENTITY:
            await self._annotate_entity(event, curated=True, rejected=True)
            summary.rejected_entities += 1

        elif action == CurationAction.CORRECT_ENTITY:
            await self._correct_entity(event)
            summary.corrected_entities += 1

        elif action == CurationAction.APPROVE_RELATIONSHIP:
            await self._annotate_relationship(event, curated=True, rejected=False)
            summary.approved_relationships += 1

        elif action == CurationAction.REJECT_RELATIONSHIP:
            await self._annotate_relationship(event, curated=True, rejected=True)
            summary.rejected_relationships += 1

        elif action == CurationAction.CORRECT_RELATIONSHIP:
            await self._correct_relationship(event)
            summary.corrected_relationships += 1

    async def _annotate_entity(
        self,
        event: CurationEvent,
        curated: bool,
        rejected: bool,
    ) -> None:
        entity = await self.graph_repo.get_entity_by_id(event.target_id)
        if entity is None:
            raise ValueError(f"Entity '{event.target_id}' not found")
        entity.metadata.add_annotation(self._CURATED_KEY, curated)
        entity.metadata.add_annotation(self._REJECTED_KEY, rejected)
        entity.metadata.add_annotation(self._CURATOR_KEY, event.curator)
        if event.reason:
            entity.metadata.add_annotation("curation_reason", event.reason)
        entity.metadata.update(updated_by=event.curator)
        await self.graph_repo.save_entity(entity)

    async def _correct_entity(self, event: CurationEvent) -> None:
        entity = await self.graph_repo.get_entity_by_id(event.target_id)
        if entity is None:
            raise ValueError(f"Entity '{event.target_id}' not found")

        corrections = event.corrections
        if "name" in corrections:
            entity.name = str(corrections["name"])
        if "description" in corrections:
            entity.description = str(corrections["description"])
        if "properties" in corrections and isinstance(corrections["properties"], dict):
            entity.properties.update(corrections["properties"])

        entity.metadata.add_annotation(self._CURATED_KEY, True)
        entity.metadata.add_annotation(self._CURATOR_KEY, event.curator)
        if event.reason:
            entity.metadata.add_annotation("curation_reason", event.reason)
        entity.metadata.update(updated_by=event.curator)
        await self.graph_repo.save_entity(entity)

    async def _annotate_relationship(
        self,
        event: CurationEvent,
        curated: bool,
        rejected: bool,
    ) -> None:
        rel = await self.graph_repo.get_relationship_by_id(event.target_id)
        if rel is None:
            raise ValueError(f"Relationship '{event.target_id}' not found")
        rel.metadata.add_annotation(self._CURATED_KEY, curated)
        rel.metadata.add_annotation(self._REJECTED_KEY, rejected)
        rel.metadata.add_annotation(self._CURATOR_KEY, event.curator)
        if event.reason:
            rel.metadata.add_annotation("curation_reason", event.reason)
        rel.metadata.update(updated_by=event.curator)
        await self.graph_repo.save_relationship(rel)

    async def _correct_relationship(self, event: CurationEvent) -> None:
        rel = await self.graph_repo.get_relationship_by_id(event.target_id)
        if rel is None:
            raise ValueError(f"Relationship '{event.target_id}' not found")

        corrections = event.corrections
        if "relationship_type" in corrections:
            rel.relationship_type = RelationshipType(corrections["relationship_type"])
        if "description" in corrections:
            rel.description = str(corrections["description"])
        if "strength" in corrections:
            rel.strength = float(corrections["strength"])

        rel.metadata.add_annotation(self._CURATED_KEY, True)
        rel.metadata.add_annotation(self._CURATOR_KEY, event.curator)
        if event.reason:
            rel.metadata.add_annotation("curation_reason", event.reason)
        rel.metadata.update(updated_by=event.curator)
        await self.graph_repo.save_relationship(rel)
