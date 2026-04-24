"""Lightweight verifier for ``GraphEntity`` objects.

Mirrors the design of the relationship cascade but answers a simpler
question: "is this extracted entity plausibly real?" Two cheap stages,
no LLM (LLM judgement on individual entities is expensive and rarely
worth it):

1. **Source confirmation** — does the entity name appear (case-insensitive)
   in any of the source chunks the LLM was reading from? If yes, the
   LLM didn't hallucinate the term wholesale → +confidence.
2. **Concept similarity** — compute a sentence embedding for the name
   (+ description if present) and run a vector search against existing
   entities in the graph. If a high-cosine neighbour exists, the
   concept is already known → +confidence.

Aggregated confidence is a weighted blend (text 0.45, embedding 0.55).
The pipeline then maps the score to a ``verification_status`` using the
``entity_auto_approve`` / ``entity_flag_below`` thresholds in
``VerificationConfiguration``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, List, Optional

from .models import VerificationResult, VerificationStage, VerificationStatus

if TYPE_CHECKING:
    from ...domain.models.graph_models import GraphEntity
    from ...infrastructure.repositories.graph_repository import GraphRepositoryInterface


logger = logging.getLogger(__name__)


@dataclass
class EntityVerifierConfig:
    text_weight: float = 0.45
    embedding_weight: float = 0.55
    embedding_top_k: int = 5
    embedding_min_score: float = 0.55      # below this we treat as "no match"
    embedding_full_credit: float = 0.85    # at/above this → 1.0 credit
    chunk_text_max_chars: int = 8000       # cap memory on huge chunks


class EntityVerifier:
    """Run text-match + embedding-similarity checks on a single entity.

    Parameters
    ----------
    config:
        Tuning knobs (see ``EntityVerifierConfig``).
    graph_repo:
        Required — used for the vector search step. Without it the
        embedding stage degrades to a SKIPPED result.
    """

    def __init__(
        self,
        config: Optional[EntityVerifierConfig] = None,
        graph_repo: Optional["GraphRepositoryInterface"] = None,
    ) -> None:
        self._cfg = config or EntityVerifierConfig()
        self._graph_repo = graph_repo

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def verify(
        self,
        entity: "GraphEntity",
        chunk_texts: Optional[List[str]] = None,
    ) -> VerificationResult:
        """Verify a single entity. Returns a ``VerificationResult`` whose
        ``confidence`` field can be fed straight into the same threshold
        machinery as the relationship verifier."""
        text_score, text_note = self._text_match(entity, chunk_texts or [])
        emb_score, emb_note = await self._embedding_match(entity)

        # Skip-aware weighted blend
        weights = []
        scores = []
        if text_score is not None:
            weights.append(self._cfg.text_weight)
            scores.append(text_score)
        if emb_score is not None:
            weights.append(self._cfg.embedding_weight)
            scores.append(emb_score)

        if not weights:
            return VerificationResult(
                stage=VerificationStage.CASCADING,
                status=VerificationStatus.INCONCLUSIVE,
                confidence=0.0,
                reasoning="No verification stages ran (no chunks + no embedding model)",
            )

        confidence = sum(s * w for s, w in zip(scores, weights)) / sum(weights)
        status = (
            VerificationStatus.PASSED if confidence >= 0.6
            else VerificationStatus.INCONCLUSIVE if confidence >= 0.3
            else VerificationStatus.FAILED
        )
        notes = " · ".join(n for n in (text_note, emb_note) if n)
        return VerificationResult(
            stage=VerificationStage.CASCADING,
            status=status,
            confidence=round(confidence, 3),
            reasoning=notes,
        )

    # ------------------------------------------------------------------
    # Stages
    # ------------------------------------------------------------------

    def _text_match(
        self, entity: "GraphEntity", chunk_texts: List[str]
    ) -> tuple[Optional[float], Optional[str]]:
        """Confidence based on whether the entity name appears in source.

        Returns ``(None, None)`` if no chunks were given (e.g. seeded
        entity with no provenance) — caller will skip this stage in the
        weighted blend.
        """
        if not chunk_texts:
            return None, None
        name = (entity.name or "").strip()
        if not name:
            return 0.0, "Entity has no name"
        haystack = (" \n ".join(chunk_texts))[: self._cfg.chunk_text_max_chars].lower()
        if name.lower() in haystack:
            return 1.0, f"'{name}' found in source"
        # Partial credit if any alias appears
        for alias in (getattr(entity, "aliases", set()) or set()):
            if alias and alias.lower() in haystack:
                return 0.7, f"alias '{alias}' found in source"
        return 0.0, f"'{name}' not in source chunks"

    async def _embedding_match(
        self, entity: "GraphEntity"
    ) -> tuple[Optional[float], Optional[str]]:
        """Confidence based on cosine similarity to nearest existing entity.

        Returns ``(None, None)`` if no embedding model / no graph repo
        is available — stage is then skipped in the blend.
        """
        if self._graph_repo is None:
            return None, None
        try:
            from ...infrastructure.services.embedding_factory import embed
        except Exception:
            return None, None

        text = entity.name + (f" — {entity.description}" if entity.description else "")
        vector = embed(text)
        if vector is None:
            return None, None

        try:
            hits = await self._graph_repo.vector_search_entities(
                vector,
                top_k=self._cfg.embedding_top_k,
                min_score=self._cfg.embedding_min_score,
            )
        except Exception as exc:
            logger.debug("Entity verifier vector search failed: %s", exc)
            return None, None

        # Filter out self-hits (entity is being saved + verified in-pipeline,
        # the just-saved row would otherwise score 1.0 trivially).
        hits = [(h, s) for (h, s) in hits if getattr(h, "id", None) != entity.id]
        if not hits:
            return 0.4, "No similar entity in graph (novel concept)"

        top_score = max(s for (_, s) in hits)
        # Linear scaling: min_score → 0.0 credit, full_credit → 1.0 credit
        full = self._cfg.embedding_full_credit
        floor = self._cfg.embedding_min_score
        if top_score >= full:
            score = 1.0
        else:
            span = max(full - floor, 1e-6)
            score = max(0.0, (top_score - floor) / span)
        top_name = getattr(hits[0][0], "name", "?")
        return round(score, 3), f"closest match '{top_name}' @ {top_score:.2f}"
