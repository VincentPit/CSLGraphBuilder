"""
EmbeddingVerifier — Stage 2 of the cascading verification pipeline.

Computes the cosine similarity between a sentence-transformers embedding of
the relationship description and the context, then compares it against a
configurable threshold.

Dependency: ``sentence-transformers`` (optional — if not installed the verifier
returns a SKIPPED result with a clear message instead of raising).

The model is lazy-loaded on first call and shared across instances that use
the same model name (module-level cache).
"""

import logging
from dataclasses import dataclass
from typing import Dict, Optional

from .models import VerificationResult, VerificationStage, VerificationStatus
from ...domain.models.graph_models import GraphRelationship

logger = logging.getLogger(__name__)

# Module-level model cache: model_name → SentenceTransformer instance
_MODEL_CACHE: Dict[str, object] = {}


@dataclass
class EmbeddingConfig:
    """Configuration for ``EmbeddingVerifier``."""

    model_name: str = "all-MiniLM-L6-v2"
    """sentence-transformers model to use."""

    threshold: float = 0.5
    """Minimum cosine similarity to PASS."""

    max_context_chars: int = 2_000
    """Truncate context to this many characters before embedding."""


class EmbeddingVerifier:
    """
    Semantic similarity verifier using sentence-transformers.

    Parameters
    ----------
    config:
        ``EmbeddingConfig``.  Defaults to ``EmbeddingConfig()`` if not given.
    """

    def __init__(self, config: Optional[EmbeddingConfig] = None) -> None:
        self._cfg = config or EmbeddingConfig()

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
        Verify a relationship via cosine similarity.

        The *query* is constructed from the relationship description, entity
        names, and relationship type; the *document* is the trimmed context.
        """
        if not context:
            return VerificationResult(
                status=VerificationStatus.FAILED,
                stage=VerificationStage.EMBEDDING,
                confidence=0.0,
                reasoning="Empty context; cosine similarity cannot be computed.",
            )

        try:
            model = self._get_model()
        except ImportError:
            return VerificationResult(
                status=VerificationStatus.SKIPPED,
                stage=VerificationStage.EMBEDDING,
                confidence=0.0,
                reasoning=(
                    "sentence-transformers is not installed. "
                    "Run `pip install sentence-transformers` to enable this stage."
                ),
            )

        query = self._build_query(relationship, source_name, target_name)
        doc = context[: self._cfg.max_context_chars]

        similarity = self._cosine_similarity(model, query, doc)
        passed = similarity >= self._cfg.threshold

        return VerificationResult(
            status=VerificationStatus.PASSED if passed else VerificationStatus.FAILED,
            stage=VerificationStage.EMBEDDING,
            confidence=round(float(similarity), 4),
            reasoning=(
                f"Cosine similarity {similarity:.4f} "
                f"{'≥' if passed else '<'} threshold {self._cfg.threshold}."
            ),
            metadata={
                "similarity": float(similarity),
                "threshold": self._cfg.threshold,
                "model": self._cfg.model_name,
                "query": query,
            },
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_model(self) -> object:
        name = self._cfg.model_name
        if name not in _MODEL_CACHE:
            from sentence_transformers import SentenceTransformer  # type: ignore

            logger.info("Loading sentence-transformers model '%s' …", name)
            _MODEL_CACHE[name] = SentenceTransformer(name)
        return _MODEL_CACHE[name]

    @staticmethod
    def _build_query(
        rel: GraphRelationship,
        source_name: Optional[str],
        target_name: Optional[str],
    ) -> str:
        parts = []
        if source_name:
            parts.append(source_name)
        parts.append(rel.relationship_type.value.replace("_", " "))
        if target_name:
            parts.append(target_name)
        if rel.description:
            parts.append(rel.description)
        return " ".join(parts)

    @staticmethod
    def _cosine_similarity(model: object, query: str, document: str) -> float:
        import numpy as np  # type: ignore

        embeddings = model.encode([query, document], convert_to_numpy=True)  # type: ignore
        a, b = embeddings[0], embeddings[1]
        norm = (np.linalg.norm(a) * np.linalg.norm(b))
        if norm == 0:
            return 0.0
        return float(np.dot(a, b) / norm)
