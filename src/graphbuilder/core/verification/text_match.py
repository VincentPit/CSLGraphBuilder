"""
TextMatchVerifier — Stage 1 of the cascading verification pipeline.

Checks whether terms from the relationship's source/target entity names, the
relationship type label, and optional keywords appear in the provided context
string.  No external dependencies required; purely lexical.

Pass criteria (configurable):
  ``min_match_ratio`` — fraction of required terms that must appear in the
  context (default 0.5).  Each matched term contributes proportionally to the
  confidence score.
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

from ..verification.models import (
    VerificationResult,
    VerificationStage,
    VerificationStatus,
)
from ...domain.models.graph_models import GraphRelationship


@dataclass
class TextMatchConfig:
    """Configuration for ``TextMatchVerifier``."""

    min_match_ratio: float = 0.5
    """Fraction of required terms that must appear for a PASSED verdict."""

    case_sensitive: bool = False
    """Whether term matching respects case."""

    whole_word: bool = False
    """Whether terms must appear as whole words (regex \\b boundaries)."""

    extra_terms: List[str] = field(default_factory=list)
    """Additional terms beyond entity names that must be checked."""


class TextMatchVerifier:
    """
    Lexical verifier — checks term presence in a context string.

    Parameters
    ----------
    config:
        ``TextMatchConfig`` instance.  Defaults to ``TextMatchConfig()`` if
        not provided.
    """

    def __init__(self, config: Optional[TextMatchConfig] = None) -> None:
        self._cfg = config or TextMatchConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def verify(
        self,
        relationship: GraphRelationship,
        context: str,
        source_name: Optional[str] = None,
        target_name: Optional[str] = None,
    ) -> VerificationResult:
        """
        Verify a relationship against a context string.

        Parameters
        ----------
        relationship:
            The relationship being verified.
        context:
            Free-text passage that should support the relationship (e.g. a
            document chunk, sentence, or paragraph).
        source_name:
            Human-readable name of the source entity.  Used as a required term
            if provided.
        target_name:
            Human-readable name of the target entity.  Used as a required term
            if provided.
        """
        if not context:
            return VerificationResult(
                status=VerificationStatus.FAILED,
                stage=VerificationStage.TEXT_MATCH,
                confidence=0.0,
                reasoning="Empty context provided; cannot perform text matching.",
            )

        required_terms: List[str] = []
        if source_name:
            required_terms.append(source_name)
        if target_name:
            required_terms.append(target_name)
        required_terms.extend(self._cfg.extra_terms)

        # Always include the relationship type label as a soft check
        type_label = relationship.relationship_type.value.replace("_", " ")
        soft_terms = [type_label]

        all_terms = required_terms + soft_terms

        if not all_terms:
            # Nothing to check — pass with low but non-zero confidence
            return VerificationResult(
                status=VerificationStatus.PASSED,
                stage=VerificationStage.TEXT_MATCH,
                confidence=0.3,
                reasoning="No terms to match; trivially passed.",
            )

        matched, unmatched = self._match_terms(context, all_terms)
        match_ratio = len(matched) / len(all_terms)
        # Scale ratio within required terms only for threshold check
        required_matched = [t for t in matched if t in required_terms]
        req_ratio = (
            len(required_matched) / len(required_terms) if required_terms else 1.0
        )

        status = (
            VerificationStatus.PASSED
            if req_ratio >= self._cfg.min_match_ratio
            else VerificationStatus.FAILED
        )

        confidence = round(min(match_ratio, 1.0), 4)

        reasoning_parts = []
        if matched:
            reasoning_parts.append(f"Matched terms: {matched}.")
        if unmatched:
            reasoning_parts.append(f"Unmatched terms: {unmatched}.")
        reasoning_parts.append(
            f"Required match ratio: {req_ratio:.0%} (threshold: {self._cfg.min_match_ratio:.0%})."
        )

        return VerificationResult(
            status=status,
            stage=VerificationStage.TEXT_MATCH,
            confidence=confidence,
            reasoning=" ".join(reasoning_parts),
            metadata={
                "matched": matched,
                "unmatched": unmatched,
                "match_ratio": match_ratio,
                "required_match_ratio": req_ratio,
            },
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _match_terms(
        self, context: str, terms: Sequence[str]
    ) -> tuple[List[str], List[str]]:
        haystack = context if self._cfg.case_sensitive else context.lower()

        matched, unmatched = [], []
        for term in terms:
            needle = term if self._cfg.case_sensitive else term.lower()
            if not needle:
                continue
            if self._cfg.whole_word:
                found = bool(re.search(rf"\b{re.escape(needle)}\b", haystack))
            else:
                found = needle in haystack
            (matched if found else unmatched).append(term)

        return matched, unmatched
