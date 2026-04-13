"""
Knowledge Conflict Detection.

Detects conflicts between new incoming knowledge and existing knowledge
graph content.  A conflict occurs when two sources make contradictory
claims about the same entity pair — for example, one chunk says
"Drug A treats Disease B" while another says "Drug A has no effect on
Disease B".

The detector uses a two-stage approach:
  1. **Structural check** — find existing relationships between the same
     entity pair that have a *different* relationship type or contradictory
     description.
  2. **LLM judgement** (optional) — asks the LLM whether two descriptions
     are contradictory, complementary, or redundant.

Results are returned as a list of ``ConflictReport`` entries so the caller
can surface them in the curation UI.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from ...domain.models.graph_models import GraphEntity, GraphRelationship, KnowledgeGraph

logger = logging.getLogger(__name__)


class ConflictType(Enum):
    """Types of knowledge conflicts."""
    CONTRADICTORY = "contradictory"      # Claims directly oppose each other
    INCONSISTENT = "inconsistent"        # Different relationship types between same pair
    REDUNDANT = "redundant"              # Same claim from different sources
    COMPLEMENTARY = "complementary"      # Non-conflicting, additive information


class ConflictSeverity(Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class ConflictEntry:
    """A single detected conflict between two pieces of knowledge."""

    conflict_type: str
    severity: str
    existing_relationship_id: str
    existing_relationship_type: str
    existing_description: str
    existing_source_chunk_ids: List[str]
    new_relationship_type: str
    new_description: str
    new_source_chunk_ids: List[str]
    source_entity_name: str
    target_entity_name: str
    reasoning: str


@dataclass
class ConflictReport:
    """Aggregate conflict detection report."""

    total_checked: int
    conflicts_found: int
    conflicts: List[ConflictEntry]


class KnowledgeConflictDetector:
    """
    Detect conflicts between new relationships and existing graph knowledge.

    Parameters
    ----------
    graph:
        The existing ``KnowledgeGraph`` to check against.
    llm_service:
        Optional LLM service for semantic conflict analysis.
    """

    def __init__(
        self,
        graph: KnowledgeGraph,
        llm_service: Optional[Any] = None,
    ) -> None:
        self._graph = graph
        self._llm_service = llm_service

    def check_conflicts(
        self,
        new_relationships: List[GraphRelationship],
        entity_names: Optional[Dict[str, str]] = None,
        use_llm: bool = False,
    ) -> ConflictReport:
        """
        Check a batch of new relationships against existing graph knowledge.

        Parameters
        ----------
        new_relationships:
            Relationships to check (not yet committed to the graph).
        entity_names:
            Map of entity_id → human-readable name for reporting.
        use_llm:
            Whether to use LLM for semantic conflict analysis.
        """
        if entity_names is None:
            entity_names = {
                eid: ent.name for eid, ent in self._graph.entities.items()
            }

        # Index existing relationships by (source, target) pair
        existing_by_pair: Dict[tuple, List[GraphRelationship]] = {}
        for rel in self._graph.relationships.values():
            key = (rel.source_entity_id, rel.target_entity_id)
            existing_by_pair.setdefault(key, []).append(rel)
            # Also check reverse direction
            rev_key = (rel.target_entity_id, rel.source_entity_id)
            existing_by_pair.setdefault(rev_key, []).append(rel)

        conflicts: List[ConflictEntry] = []

        for new_rel in new_relationships:
            key = (new_rel.source_entity_id, new_rel.target_entity_id)
            existing_rels = existing_by_pair.get(key, [])

            for existing_rel in existing_rels:
                conflict = self._compare_relationships(
                    existing_rel, new_rel, entity_names, use_llm
                )
                if conflict is not None:
                    conflicts.append(conflict)

        return ConflictReport(
            total_checked=len(new_relationships),
            conflicts_found=len(conflicts),
            conflicts=conflicts,
        )

    def check_single(
        self,
        new_rel: GraphRelationship,
        entity_names: Optional[Dict[str, str]] = None,
        use_llm: bool = False,
    ) -> List[ConflictEntry]:
        """Check a single new relationship against the graph."""
        report = self.check_conflicts([new_rel], entity_names, use_llm)
        return report.conflicts

    # ------------------------------------------------------------------

    def _compare_relationships(
        self,
        existing: GraphRelationship,
        new: GraphRelationship,
        entity_names: Dict[str, str],
        use_llm: bool,
    ) -> Optional[ConflictEntry]:
        """Compare two relationships and return a conflict if found."""

        src_name = entity_names.get(new.source_entity_id, new.source_entity_id)
        tgt_name = entity_names.get(new.target_entity_id, new.target_entity_id)

        existing_desc = (existing.description or "").strip().lower()
        new_desc = (new.description or "").strip().lower()

        # Same type + same description → redundant, not a conflict to flag
        if (existing.relationship_type == new.relationship_type
                and existing_desc == new_desc):
            return None

        # Different relationship type between same entities → inconsistent
        if existing.relationship_type != new.relationship_type:
            conflict_type = ConflictType.INCONSISTENT
            severity = ConflictSeverity.MEDIUM
            reasoning = (
                f"Existing relationship type '{existing.relationship_type.value}' "
                f"differs from new type '{new.relationship_type.value}' "
                f"between {src_name} and {tgt_name}."
            )
        else:
            # Same type but different descriptions — check semantically
            if use_llm and self._llm_service:
                result = self._llm_conflict_check(existing, new, src_name, tgt_name)
                if result is None:
                    return None  # LLM says no conflict
                return result

            # Heuristic: different descriptions with same type
            if existing_desc and new_desc and existing_desc != new_desc:
                # Check for obvious negation patterns
                negation_words = {"no ", "not ", "doesn't ", "does not ", "isn't ", "is not ",
                                  "hasn't ", "has not ", "won't ", "cannot ", "can't ", "lack ",
                                  "absent ", "without ", "fails to ", "unable to "}
                existing_has_neg = any(existing_desc.startswith(w) or f" {w}" in existing_desc
                                       for w in negation_words)
                new_has_neg = any(new_desc.startswith(w) or f" {w}" in new_desc
                                  for w in negation_words)

                if existing_has_neg != new_has_neg:
                    conflict_type = ConflictType.CONTRADICTORY
                    severity = ConflictSeverity.HIGH
                    reasoning = (
                        f"Potential contradiction: one description contains negation "
                        f"while the other does not. "
                        f"Existing: '{existing.description}' vs New: '{new.description}'"
                    )
                else:
                    conflict_type = ConflictType.COMPLEMENTARY
                    severity = ConflictSeverity.LOW
                    reasoning = (
                        f"Same relationship type but different descriptions. "
                        f"Likely complementary information."
                    )
                    # Low-severity complementary info — not worth flagging
                    return None
            else:
                return None

        return ConflictEntry(
            conflict_type=conflict_type.value,
            severity=severity.value,
            existing_relationship_id=existing.id,
            existing_relationship_type=existing.relationship_type.value,
            existing_description=existing.description or "",
            existing_source_chunk_ids=existing.source_chunk_ids,
            new_relationship_type=new.relationship_type.value,
            new_description=new.description or "",
            new_source_chunk_ids=new.source_chunk_ids,
            source_entity_name=src_name,
            target_entity_name=tgt_name,
            reasoning=reasoning,
        )

    def _llm_conflict_check(
        self,
        existing: GraphRelationship,
        new: GraphRelationship,
        src_name: str,
        tgt_name: str,
    ) -> Optional[ConflictEntry]:
        """Use LLM to determine if two relationship descriptions conflict."""
        prompt = (
            f"Two knowledge graph relationships connect the same entities.\n\n"
            f"Source entity: {src_name}\n"
            f"Target entity: {tgt_name}\n\n"
            f"Existing claim:\n"
            f"  Type: {existing.relationship_type.value}\n"
            f"  Description: {existing.description}\n\n"
            f"New claim:\n"
            f"  Type: {new.relationship_type.value}\n"
            f"  Description: {new.description}\n\n"
            f"Are these two claims contradictory, complementary, or redundant?\n"
            f"Respond with JSON only:\n"
            f'{{"verdict": "contradictory"|"complementary"|"redundant", '
            f'"confidence": <float 0-1>, '
            f'"reasoning": "<brief explanation>"}}'
        )

        import json
        try:
            response = self._llm_service.generate_text(
                prompt=prompt,
                system_prompt="You are a knowledge graph analyst. Determine if two claims conflict.",
                temperature=0.1,
            )
            data = json.loads(response)
            verdict = data.get("verdict", "complementary")
            reasoning = data.get("reasoning", "")

            if verdict == "redundant" or verdict == "complementary":
                return None

            return ConflictEntry(
                conflict_type=ConflictType.CONTRADICTORY.value,
                severity=ConflictSeverity.HIGH.value,
                existing_relationship_id=existing.id,
                existing_relationship_type=existing.relationship_type.value,
                existing_description=existing.description or "",
                existing_source_chunk_ids=existing.source_chunk_ids,
                new_relationship_type=new.relationship_type.value,
                new_description=new.description or "",
                new_source_chunk_ids=new.source_chunk_ids,
                source_entity_name=src_name,
                target_entity_name=tgt_name,
                reasoning=f"LLM: {reasoning}",
            )
        except Exception as exc:
            logger.warning("LLM conflict check failed: %s", exc)
            return None
