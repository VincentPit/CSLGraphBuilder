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

import asyncio
import logging
import os
import threading
from typing import Any, List, Optional


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


# ── Async wrappers ───────────────────────────────────────────────────────
#
# ``model.encode`` is a synchronous PyTorch call that blocks for tens to
# hundreds of milliseconds per batch. Calling it directly from an async
# handler freezes the entire event loop for the duration — ``/health``
# checks, frontend polling, and any other in-flight job all stall.
#
# Two backends sit behind the public async API:
#
# * **GPU**: when ``torch.cuda.is_available()`` is true, a
#   ``GPUEmbeddingPool`` with N model copies on dedicated CUDA streams
#   handles encodes in genuine parallel. N is sized from free GPU memory.
# * **CPU**: single shared model + ``asyncio.Lock`` + thread executor.
#   The lock makes encodes serial (sentence-transformers/PyTorch have no
#   defined behaviour under concurrent inference on the same model);
#   the executor keeps the event loop responsive while the lone worker
#   thread runs the encode. On CPU there's nothing to gain from parallel
#   encodes — torch's MKL/OpenMP already saturates cores from one call.
#
# Backend selection is automatic and one-shot at first use.

_ENCODE_LOCK: Optional[asyncio.Lock] = None


def _encode_lock() -> asyncio.Lock:
    global _ENCODE_LOCK
    if _ENCODE_LOCK is None:
        _ENCODE_LOCK = asyncio.Lock()
    return _ENCODE_LOCK


# ── GPU pool plumbing (lazy, opt-in via CUDA availability) ─────────────

_GPU_POOL: Any = None  # GPUEmbeddingPool when active
_GPU_POOL_INIT_LOCK = threading.Lock()
_GPU_POOL_DISABLED = False  # latches True once we determine GPU is unusable


def _get_gpu_pool() -> Any:
    """Return the GPU pool, lazily initialising it on first call.

    Returns ``None`` if CUDA isn't available or pool init fails — callers
    should fall back to the CPU lock+executor path. The "disabled" flag
    latches so we don't probe CUDA repeatedly on a CPU-only machine.
    """
    global _GPU_POOL, _GPU_POOL_DISABLED
    if _GPU_POOL_DISABLED:
        return None
    if _GPU_POOL is not None:
        return _GPU_POOL
    with _GPU_POOL_INIT_LOCK:
        if _GPU_POOL_DISABLED:
            return None
        if _GPU_POOL is not None:
            return _GPU_POOL
        try:
            import torch
            if not torch.cuda.is_available():
                _GPU_POOL_DISABLED = True
                logger.info(
                    "CUDA not available — embedding stays on the CPU "
                    "single-worker (lock + executor) path."
                )
                return None
        except Exception as exc:  # torch missing / broken install
            _GPU_POOL_DISABLED = True
            logger.info("Torch CUDA probe failed (%s) — using CPU path.", exc)
            return None

        try:
            from .gpu_embedding_pool import GPUEmbeddingPool

            requested = os.getenv("EMBEDDING_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL
            min_w = max(1, int(os.getenv("EMBEDDING_GPU_MIN_WORKERS", "1")))
            max_w = max(min_w, int(os.getenv("EMBEDDING_GPU_MAX_WORKERS", "8")))
            mem_frac = float(os.getenv("EMBEDDING_GPU_MEMORY_FRACTION", "0.7"))
            explicit_env = os.getenv("EMBEDDING_GPU_WORKERS")
            explicit = int(explicit_env) if explicit_env else None
            device = os.getenv("EMBEDDING_GPU_DEVICE", "cuda:0")

            _GPU_POOL = GPUEmbeddingPool(
                model_name=requested,
                device=device,
                min_workers=min_w,
                max_workers=max_w,
                memory_fraction=mem_frac,
                explicit_workers=explicit,
            )
            logger.info("GPU embedding pool created (workers being loaded in background).")
            return _GPU_POOL
        except Exception as exc:
            logger.warning(
                "GPU pool construction failed (%s) — falling back to CPU path.",
                exc, exc_info=True,
            )
            _GPU_POOL_DISABLED = True
            return None


async def embed_async(text: str) -> Optional[List[float]]:
    """Async embedding for a single string.

    Uses the GPU pool when available; otherwise the CPU lock+executor path.
    """
    pool = _get_gpu_pool()
    if pool is not None:
        return await pool.embed_async(text)

    # CPU fallback
    text = (text or "").strip()
    if not text:
        return None
    if get_model() is None:
        return None
    async with _encode_lock():
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, embed, text)


async def embed_batch_async(
    texts: List[str], batch_size: int = 32
) -> List[Optional[List[float]]]:
    """Async batched embedding.

    Uses the GPU pool when available; otherwise the CPU lock+executor path.
    """
    pool = _get_gpu_pool()
    if pool is not None:
        return await pool.embed_batch_async(texts, batch_size)

    # CPU fallback
    if not texts:
        return []
    async with _encode_lock():
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, embed_batch, texts, batch_size)


def embed_batch(
    texts: List[str], batch_size: int = 32
) -> List[Optional[List[float]]]:
    """Encode a list of strings in one call, with per-batch padding.

    Returns one entry per input — ``None`` for empty/whitespace strings,
    a vector for everything else. The list always has the same length as
    ``texts`` so callers can ``zip`` it back to their objects.

    ``batch_size`` is the per-forward-pass mini-batch (sentence-transformers
    chunks the full list internally). Default 32 keeps peak memory ~100 MB
    for SapBERT on CPU; bump to 64+ on GPU where the larger forward pass
    amortizes setup overhead.

    Why this exists: ``save_entities_batch`` / ``save_relationships_batch``
    need to amortize the model's per-call overhead across hundreds of
    entities. Calling ``model.encode`` once with a list (and letting
    sentence-transformers handle batching + padding internally) is 10–50×
    faster than looping on the single-string variant for OT-sized ingests.
    """
    if not texts:
        return []

    # Build the batch and remember the original index of each non-empty entry
    # so we can splice the embeddings back into the right slots.
    cleaned: List[str] = []
    keep_idx: List[int] = []
    for i, t in enumerate(texts):
        s = (t or "").strip()
        if s:
            cleaned.append(s)
            keep_idx.append(i)

    out: List[Optional[List[float]]] = [None] * len(texts)
    if not cleaned:
        return out

    model = get_model()
    if model is None:
        return out

    try:
        vectors = model.encode(
            cleaned,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
    except Exception as exc:
        logger.debug("Batch embedding failed (%d texts): %s", len(cleaned), exc)
        return out

    for slot, vec in zip(keep_idx, vectors):
        out[slot] = vec.tolist()
    return out
