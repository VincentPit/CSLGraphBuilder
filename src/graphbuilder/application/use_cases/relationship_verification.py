"""
Relationship Verification Use Case.

Runs the ``CascadingVerifier`` against each relationship in the knowledge
graph, pairing it with the source document chunk that produced it (when
available).  Relationships that fail verification are flagged in their
``metadata.annotations`` — they are never deleted so curation can
review them.

Results are returned as a ``ProcessingResult`` with the full per-relationship
report in ``result.data["report"]``.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ...core.verification.cascading import CascadingVerifier, CascadingVerifierConfig
from ...core.verification.models import VerificationStatus
from ...domain.models.graph_models import GraphRelationship, KnowledgeGraph
from ...domain.models.processing_models import ProcessingResult

logger = logging.getLogger(__name__)


@dataclass
class VerificationConfig:
    """Configuration for a single verification run."""

    cascading: CascadingVerifierConfig = field(default_factory=CascadingVerifierConfig)
    """Settings forwarded to CascadingVerifier; controls which stages run."""

    context_map: Dict[str, str] = field(default_factory=dict)
    """
    Mapping of ``relationship_id → context_string``.
    When a relationship's ID is absent the description field is used as context
    (or an empty string if description is also absent).
    """

    entity_name_map: Dict[str, str] = field(default_factory=dict)
    """Mapping of ``entity_id → human-readable name`` for better prompts."""

    fail_annotation: str = "verification_failed"
    """Annotation key written to failing relationship metadata."""

    pass_annotation: str = "verification_passed"
    """Annotation key written to passing relationship metadata."""


class RelationshipVerificationUseCase:
    """
    Verify all relationships in a ``KnowledgeGraph``.

    Parameters
    ----------
    graph:
        A populated ``KnowledgeGraph`` instance.
    llm_service:
        Optional LLM service injected into ``CascadingVerifier`` for Stage 3.
    """

    def __init__(
        self,
        graph: KnowledgeGraph,
        llm_service: Optional[Any] = None,
    ) -> None:
        self._graph = graph
        self._llm_service = llm_service

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def execute(self, config: VerificationConfig) -> ProcessingResult:
        verifier = CascadingVerifier(
            config=config.cascading,
            llm_service=self._llm_service,
        )

        report: List[Dict[str, Any]] = []
        passed_count = 0
        failed_count = 0
        skipped_count = 0

        for rel in self._graph.relationships.values():
            context = config.context_map.get(rel.id) or rel.description or ""
            source_name = config.entity_name_map.get(rel.source_entity_id)
            target_name = config.entity_name_map.get(rel.target_entity_id)

            result = verifier.verify(
                relationship=rel,
                context=context,
                source_name=source_name,
                target_name=target_name,
            )

            # Annotate the relationship in-place
            if result.passed:
                rel.metadata.annotations[config.pass_annotation] = True
                rel.metadata.annotations[config.fail_annotation] = False
                passed_count += 1
            elif result.status == VerificationStatus.SKIPPED:
                skipped_count += 1
            else:
                rel.metadata.annotations[config.fail_annotation] = True
                rel.metadata.annotations[config.pass_annotation] = False
                failed_count += 1

            rel.metadata.annotations["verification_confidence"] = result.confidence
            rel.metadata.annotations["verification_reasoning"]  = result.reasoning

            report.append({
                "relationship_id":   rel.id,
                "source_entity_id":  rel.source_entity_id,
                "target_entity_id":  rel.target_entity_id,
                "relationship_type": rel.relationship_type.value,
                "status":            result.status.value,
                "confidence":        result.confidence,
                "reasoning":         result.reasoning,
                "stage_results": [
                    {
                        "stage":      sr.stage.value,
                        "status":     sr.status.value,
                        "confidence": sr.confidence,
                        "reasoning":  sr.reasoning,
                    }
                    for sr in result.stage_results
                ],
            })

        total = len(self._graph.relationships)
        logger.info(
            "Verification complete: %d total, %d passed, %d failed, %d skipped",
            total, passed_count, failed_count, skipped_count,
        )

        return ProcessingResult(
            success=True,
            message=(
                f"Verified {total} relationships: "
                f"{passed_count} passed, {failed_count} failed, {skipped_count} skipped."
            ),
            data={
                "total":     total,
                "passed":    passed_count,
                "failed":    failed_count,
                "skipped":   skipped_count,
                "report":    report,
            },
        )
