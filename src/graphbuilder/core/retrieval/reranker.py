"""Cross-encoder reranker — Stage 2 of fusion (§3.3 of docs/RAG_QA_PLAN.md).

After RRF fuses the channel rankings into a top-N candidate list, a
cross-encoder model scores each ``(query, candidate)`` pair end-to-end.
Cross-encoders are dramatically more accurate than bi-encoder cosine
because they jointly attend to both sides of the pair — but they're
quadratic in candidates × tokens, so we only run them on the post-RRF
shortlist (default 50), not on the raw retrieval results.

The default model is ``cross-encoder/ms-marco-MiniLM-L-6-v2`` — small
(22 MB), CPU-friendly, English-general. Biomedical-tuned cross-encoders
like ``pritamdeka/S-BioBert-snli-multinli-stsb`` are a config knob away
when retrieval evaluation (P13) tells us they pay off.

Lazy-loading + module-level cache mirrors :mod:`embedding_factory` so a
warm process pays the model load once. If sentence-transformers is
unavailable (or the model fails to load) the reranker degrades to a
pass-through that preserves the input order — never raises.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, List, Optional

from .models import RetrievedItem


logger = logging.getLogger("graphbuilder.qa.retrieval")


# ----------------------------------------------------------------------
# Module-level model cache
# ----------------------------------------------------------------------

_MODEL_CACHE: dict[str, Any] = {}
_LOAD_LOCK = asyncio.Lock()


async def _get_cross_encoder(model_name: str) -> Optional[Any]:
    """Lazy-load and cache a CrossEncoder. Returns ``None`` if the
    model can't be loaded — caller treats that as "skip reranking"."""
    cached = _MODEL_CACHE.get(model_name)
    if cached is not None or model_name in _MODEL_CACHE:
        return cached  # ``None`` is also a cached "failed to load"

    async with _LOAD_LOCK:
        # Re-check inside the lock — another task may have loaded it
        # while we were waiting.
        if model_name in _MODEL_CACHE:
            return _MODEL_CACHE[model_name]
        try:
            from sentence_transformers import CrossEncoder  # type: ignore

            loop = asyncio.get_running_loop()
            logger.info("Loading cross-encoder model %r …", model_name)
            model = await loop.run_in_executor(
                None, lambda: CrossEncoder(model_name)
            )
            _MODEL_CACHE[model_name] = model
            return model
        except Exception as exc:
            logger.warning(
                "Could not load cross-encoder %r (%s) — falling back to RRF order",
                model_name, exc,
            )
            _MODEL_CACHE[model_name] = None
            return None


# ----------------------------------------------------------------------
# Reranker
# ----------------------------------------------------------------------

@dataclass
class CrossEncoderConfig:
    """Tuning knobs for :class:`CrossEncoderReranker`."""

    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    """Hugging Face id of the cross-encoder model to use."""

    max_pair_chars: int = 512
    """Truncate the candidate side of each pair to this many chars before
    scoring. Cross-encoders have a small input window (often 512 tokens
    total); bigger candidates get clipped not chunked here."""

    batch_size: int = 16
    """Pairs per encode call. CPU-friendly default."""


class CrossEncoderReranker:
    """Reorder a candidate list by cross-encoder relevance to *query*."""

    def __init__(self, config: Optional[CrossEncoderConfig] = None) -> None:
        self._cfg = config or CrossEncoderConfig()

    async def rerank(
        self,
        query: str,
        items: List[RetrievedItem],
        *,
        top_k: Optional[int] = None,
    ) -> List[RetrievedItem]:
        """Score and reorder *items*. Mutates each item's ``score_rerank``.

        If the model isn't available, returns the input unchanged
        (``score_rerank`` stays ``None``) so callers can still ship.
        """
        if not items:
            return items
        model = await _get_cross_encoder(self._cfg.model_name)
        if model is None:
            return items[: top_k] if top_k is not None else items

        pairs = [(query, self._candidate_text(item)) for item in items]

        try:
            loop = asyncio.get_running_loop()
            scores = await loop.run_in_executor(
                None,
                lambda: model.predict(
                    pairs,
                    batch_size=self._cfg.batch_size,
                    show_progress_bar=False,
                ),
            )
        except Exception as exc:
            logger.warning("Cross-encoder predict failed (%s) — keeping RRF order", exc)
            return items[: top_k] if top_k is not None else items

        # ``predict`` returns a numpy array or list; coerce to floats.
        try:
            scores = [float(s) for s in scores]
        except Exception:
            scores = list(scores)

        # Min-max normalise into [0, 1] for the per-source bar in the UI.
        # Sigmoid would also work, but ms-marco-MiniLM emits raw logits
        # in roughly [-15, +15] and min-max keeps the visual contrast
        # between the best and worst items obvious.
        if scores:
            lo, hi = min(scores), max(scores)
            span = hi - lo
            normed = [(s - lo) / span if span > 0 else 0.5 for s in scores]
        else:
            normed = []

        scored = list(zip(items, scores, normed))
        scored.sort(key=lambda triple: triple[1], reverse=True)

        out: List[RetrievedItem] = []
        for item, _raw, norm in scored:
            item.score_rerank = round(float(norm), 4)
            out.append(item)
        if top_k is not None:
            out = out[: top_k]
        logger.debug(
            "rerank scored %d items, kept %d, top=%s",
            len(items), len(out),
            out[0].label if out else None,
        )
        return out

    def _candidate_text(self, item: RetrievedItem) -> str:
        """Build the candidate-side string for a (query, candidate) pair.

        Prefers a chunk preview when one was hydrated — that's the
        actual grounding text. Falls back to label + description so
        the cross-encoder always has *something* meaningful to score
        against.
        """
        parts: List[str] = [item.label or ""]
        desc = (item.metadata or {}).get("description")
        if desc:
            parts.append(str(desc))
        if item.chunk_preview:
            parts.append(item.chunk_preview)
        text = " — ".join(p for p in parts if p)
        if len(text) > self._cfg.max_pair_chars:
            text = text[: self._cfg.max_pair_chars].rstrip() + "…"
        return text


__all__ = [
    "CrossEncoderConfig",
    "CrossEncoderReranker",
]
