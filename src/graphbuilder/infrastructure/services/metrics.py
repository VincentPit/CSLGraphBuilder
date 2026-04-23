"""Process-wide pipeline metrics.

A lightweight counter for LLM call volume, token usage, latency, and
cache hit rates. Exposed by the API at ``GET /health/metrics`` and
consumed by the frontend dashboard.

Thread-safety: protected by an ``asyncio.Lock`` so concurrent chunk
extractors can update without racing. Counters are monotonic across
the process lifetime; reset via ``reset()`` for benchmarks.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class _Counters:
    llm_calls: int = 0
    llm_calls_by_type: Dict[str, int] = field(default_factory=dict)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    total_latency_seconds: float = 0.0
    llm_cache_hits: int = 0
    embedding_calls: int = 0
    embedding_cache_hits: int = 0
    documents_processed: int = 0
    chunks_processed: int = 0
    entities_saved: int = 0
    relationships_saved: int = 0
    started_at: float = field(default_factory=time.time)


class PipelineMetrics:
    """Process-wide metrics singleton."""

    def __init__(self) -> None:
        self._c = _Counters()
        self._lock = asyncio.Lock()

    async def record_llm_call(
        self,
        *,
        prompt_type: str,
        prompt_tokens: int,
        completion_tokens: int,
        latency_seconds: float,
        cache_hit: bool = False,
    ) -> None:
        async with self._lock:
            self._c.llm_calls += 1
            self._c.llm_calls_by_type[prompt_type] = (
                self._c.llm_calls_by_type.get(prompt_type, 0) + 1
            )
            self._c.prompt_tokens += prompt_tokens
            self._c.completion_tokens += completion_tokens
            self._c.total_tokens += prompt_tokens + completion_tokens
            self._c.total_latency_seconds += latency_seconds
            if cache_hit:
                self._c.llm_cache_hits += 1

    async def record_embedding(self, *, cache_hit: bool = False) -> None:
        async with self._lock:
            self._c.embedding_calls += 1
            if cache_hit:
                self._c.embedding_cache_hits += 1

    async def record_document(self) -> None:
        async with self._lock:
            self._c.documents_processed += 1

    async def record_chunks(self, n: int) -> None:
        async with self._lock:
            self._c.chunks_processed += n

    async def record_entities(self, n: int) -> None:
        async with self._lock:
            self._c.entities_saved += n

    async def record_relationships(self, n: int) -> None:
        async with self._lock:
            self._c.relationships_saved += n

    def snapshot(self) -> Dict[str, Any]:
        c = self._c
        non_cached = max(c.llm_calls - c.llm_cache_hits, 0)
        avg_latency = c.total_latency_seconds / non_cached if non_cached else 0.0
        cache_hit_rate = c.llm_cache_hits / c.llm_calls if c.llm_calls else 0.0
        emb_hit_rate = (
            c.embedding_cache_hits / c.embedding_calls if c.embedding_calls else 0.0
        )
        return {
            "uptime_seconds": round(time.time() - c.started_at, 2),
            "llm": {
                "calls": c.llm_calls,
                "calls_by_type": dict(c.llm_calls_by_type),
                "prompt_tokens": c.prompt_tokens,
                "completion_tokens": c.completion_tokens,
                "total_tokens": c.total_tokens,
                "avg_latency_ms": round(avg_latency * 1000, 1),
                "cache_hits": c.llm_cache_hits,
                "cache_hit_rate": round(cache_hit_rate, 3),
            },
            "embedding": {
                "calls": c.embedding_calls,
                "cache_hits": c.embedding_cache_hits,
                "cache_hit_rate": round(emb_hit_rate, 3),
            },
            "pipeline": {
                "documents_processed": c.documents_processed,
                "chunks_processed": c.chunks_processed,
                "entities_saved": c.entities_saved,
                "relationships_saved": c.relationships_saved,
            },
        }

    def reset(self) -> None:
        self._c = _Counters()


_INSTANCE: PipelineMetrics | None = None


def get_metrics() -> PipelineMetrics:
    global _INSTANCE
    if _INSTANCE is None:
        _INSTANCE = PipelineMetrics()
    return _INSTANCE
