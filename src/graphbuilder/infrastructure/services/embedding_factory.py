"""Single source of truth for the sentence-embedding model.

Why this exists:
- Three places (document_pipeline, graph_repository, semantic_chunker /
  verifier) all need a sentence-transformers model.
- They previously hardcoded ``all-MiniLM-L6-v2`` (general-purpose, 384-d).
  For biomedical extraction that's leaving accuracy on the table — the
  model has never seen "Factor VIII" as a coherent concept and treats
  "EGFR" as gibberish.
- This module reads the configured model name (``EMBEDDING_MODEL`` env
  var, default ``cambridgeltl/SapBERT-from-PubMedBERT-fulltext``) once,
  loads it lazily, and hands the same instance back to every caller.

Default choice — **SapBERT** (Self-aligned PubMedBERT):
- Fine-tuned specifically for *biomedical entity linking*: the exact
  "is `TNF-alpha` the same concept as `Tumor Necrosis Factor Alpha`?"
  problem the dedup vector pre-filter and verifier are solving.
- 768-dimensional (vs MiniLM's 384) — Neo4j vector indexes will be
  recreated automatically with the new dim on first save.
- ~440 MB download on first use; cached locally afterwards.

Fallback chain:
1. Configured model (default: SapBERT)
2. ``all-MiniLM-L6-v2`` (general-purpose, ships in most installs)
3. Return ``None`` — embeddings disabled, vector pre-filter degrades to
   the LLM dedup path only (still works, just slower).
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any, Optional


logger = logging.getLogger("graphbuilder.embeddings")

# Biomedical entity-linking model. Override via the ``EMBEDDING_MODEL`` env var.
DEFAULT_MODEL = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
FALLBACK_MODEL = "all-MiniLM-L6-v2"


class _ModelHolder:
    """Wraps the loaded sentence-transformers model + its detected dim."""

    def __init__(self) -> None:
        # ``Any`` so callers retain ``.encode``, ``.get_sentence_embedding_dimension``
        # etc. without per-callsite casts; the real type is SentenceTransformer.
        self.model: Any = None
        self.name: Optional[str] = None
        self.dim: int = 0
        self._lock = threading.Lock()

    def ensure(self) -> Any:
        """Lazy-load on first call. Subsequent calls return the cache."""
        if self.model is not None:
            return self.model
        with self._lock:
            if self.model is not None:
                return self.model
            requested = os.getenv("EMBEDDING_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL
            self.model, self.name = _try_load(requested)
            if self.model is None and requested != FALLBACK_MODEL:
                logger.warning(
                    "Configured embedding model %r failed to load; falling back to %r",
                    requested, FALLBACK_MODEL,
                )
                self.model, self.name = _try_load(FALLBACK_MODEL)
            if self.model is not None:
                try:
                    self.dim = int(self.model.get_sentence_embedding_dimension())
                except Exception:
                    self.dim = 0
                logger.info(
                    "Embedding model loaded: %s (dim=%d)", self.name, self.dim,
                )
            else:
                logger.warning("No embedding model could be loaded; vector features disabled.")
        return self.model


def _try_load(name: str):
    """Attempt to load one model name. Returns (model_or_None, name)."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        logger.warning("sentence-transformers not installed; embeddings disabled.")
        return None, None
    try:
        return SentenceTransformer(name), name
    except Exception as exc:
        logger.warning("Failed to load embedding model %r: %s", name, exc)
        return None, None


_HOLDER = _ModelHolder()


def get_model() -> Any:
    """Return the cached SentenceTransformer instance, or None if unavailable."""
    return _HOLDER.ensure()


def get_model_name() -> Optional[str]:
    _HOLDER.ensure()
    return _HOLDER.name


def get_embedding_dim() -> int:
    """Return the dimensionality of the loaded model. 0 if no model loaded."""
    _HOLDER.ensure()
    return _HOLDER.dim


def embed(text: str):
    """Convenience: encode a single string, returning a Python list or None."""
    text = (text or "").strip()
    if not text:
        return None
    model = get_model()
    if model is None:
        return None
    try:
        return model.encode(text, convert_to_numpy=True).tolist()
    except Exception as exc:
        logger.debug("Embedding failed for %r: %s", text[:60], exc)
        return None
