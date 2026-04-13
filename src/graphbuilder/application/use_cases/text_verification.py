"""
Text Verification Use Case.

Verifies whether a free-text description (claim) is supported by entities
and relationships already present in the Neo4j knowledge graph.

Uses the three-stage ``CascadingVerifier`` pipeline:
  1. TextMatch  — lexical term overlap
  2. Embedding  — semantic similarity
  3. LLM        — reasoning-based judgement

The use case searches the graph for candidate relationships whose entity
names or descriptions match terms in the input text, then runs verification
on each candidate with the user's text as supporting context.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ...core.verification.cascading import CascadingVerifier, CascadingVerifierConfig
from ...core.verification.models import VerificationResult, VerificationStatus
from ...domain.models.graph_models import GraphEntity, GraphRelationship, KnowledgeGraph

logger = logging.getLogger(__name__)


@dataclass
class TextVerificationConfig:
    """Configuration for a text verification run."""

    cascading: CascadingVerifierConfig = field(default_factory=CascadingVerifierConfig)
    max_candidates: int = 20
    """Maximum number of candidate relationships to verify."""


@dataclass
class TextVerificationEntry:
    """Result for a single candidate relationship."""

    relationship_id: str
    source_entity_id: str
    target_entity_id: str
    source_entity_name: str
    target_entity_name: str
    relationship_type: str
    relationship_description: str
    status: str
    confidence: float
    reasoning: str
    stage_results: List[Dict[str, Any]]


@dataclass
class TextVerificationReport:
    """Aggregate report returned by the use case."""

    query_text: str
    total_candidates: int
    verified: int
    not_verified: int
    skipped: int
    best_confidence: float
    entries: List[TextVerificationEntry]


class TextVerificationUseCase:
    """
    Verify a text description against a knowledge graph.

    Parameters
    ----------
    graph:
        A ``KnowledgeGraph`` instance populated with candidate entities
        and relationships.
    llm_service:
        Optional LLM service injected into ``CascadingVerifier`` for Stage 3.
    """

    def __init__(
        self,
        graph: KnowledgeGraph,
        llm_service: Optional[Any] = None,
        graph_repo: Optional[Any] = None,
    ) -> None:
        self._graph = graph
        self._llm_service = llm_service
        self._graph_repo = graph_repo

    def execute(
        self,
        query_text: str,
        config: TextVerificationConfig,
    ) -> TextVerificationReport:
        verifier = CascadingVerifier(
            config=config.cascading,
            llm_service=self._llm_service,
            graph_repo=self._graph_repo,
        )

        # Build entity name lookup
        entity_names: Dict[str, str] = {
            eid: ent.name for eid, ent in self._graph.entities.items()
        }

        entries: List[TextVerificationEntry] = []
        verified_count = 0
        not_verified_count = 0
        skipped_count = 0
        best_confidence = 0.0

        # Limit candidates
        candidates = list(self._graph.relationships.values())[: config.max_candidates]

        for rel in candidates:
            source_name = entity_names.get(rel.source_entity_id, rel.source_entity_id)
            target_name = entity_names.get(rel.target_entity_id, rel.target_entity_id)

            result = verifier.verify(
                relationship=rel,
                context=query_text,
                source_name=source_name,
                target_name=target_name,
            )

            stage_results = [
                {
                    "stage": sr.stage.value,
                    "status": sr.status.value,
                    "confidence": round(sr.confidence, 4),
                    "reasoning": sr.reasoning,
                    "metadata": sr.metadata if sr.metadata else None,
                }
                for sr in result.stage_results
            ]

            status_str = result.status.value
            if result.status == VerificationStatus.PASSED:
                verified_count += 1
            elif result.status == VerificationStatus.SKIPPED:
                skipped_count += 1
            else:
                not_verified_count += 1

            if result.confidence > best_confidence:
                best_confidence = result.confidence

            entries.append(
                TextVerificationEntry(
                    relationship_id=rel.id,
                    source_entity_id=rel.source_entity_id,
                    target_entity_id=rel.target_entity_id,
                    source_entity_name=source_name,
                    target_entity_name=target_name,
                    relationship_type=rel.relationship_type.value if hasattr(rel.relationship_type, "value") else str(rel.relationship_type),
                    relationship_description=rel.description or "",
                    status=status_str,
                    confidence=round(result.confidence, 4),
                    reasoning=result.reasoning,
                    stage_results=stage_results,
                )
            )

        # Sort by confidence descending
        entries.sort(key=lambda e: e.confidence, reverse=True)

        return TextVerificationReport(
            query_text=query_text,
            total_candidates=len(entries),
            verified=verified_count,
            not_verified=not_verified_count,
            skipped=skipped_count,
            best_confidence=round(best_confidence, 4),
            entries=entries,
        )


def extract_search_terms(text: str) -> List[str]:
    """
    Extract meaningful search terms from a free-text description.

    Strips common stop words and returns unique terms of length >= 3.
    """
    stop_words = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "shall",
        "should", "may", "might", "must", "can", "could", "and", "but", "or",
        "nor", "not", "so", "yet", "both", "either", "neither", "each",
        "every", "all", "any", "few", "more", "most", "other", "some", "such",
        "no", "only", "own", "same", "than", "too", "very", "just", "because",
        "as", "until", "while", "of", "at", "by", "for", "with", "about",
        "against", "between", "through", "during", "before", "after", "above",
        "below", "to", "from", "up", "down", "in", "out", "on", "off", "over",
        "under", "again", "further", "then", "once", "here", "there", "when",
        "where", "why", "how", "what", "which", "who", "whom", "this", "that",
        "these", "those", "its", "it", "he", "she", "they", "them",
    }
    words = re.findall(r"[A-Za-z0-9_-]+", text)
    seen = set()
    terms = []
    for w in words:
        lower = w.lower()
        if lower not in stop_words and len(w) >= 3 and lower not in seen:
            seen.add(lower)
            terms.append(w)
    return terms
