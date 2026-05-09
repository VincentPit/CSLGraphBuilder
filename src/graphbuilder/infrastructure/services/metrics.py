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
from typing import Any, Dict, List


# Histograms are kept as simple lists of recent samples so the snapshot
# endpoint can compute p50/p95/avg without a separate dependency. The cap
# bounds memory at ~ samples * sizeof(float) per series.
_HISTOGRAM_MAX_SAMPLES = 1000


def _percentile(samples: List[float], pct: float) -> float:
    if not samples:
        return 0.0
    s = sorted(samples)
    k = (len(s) - 1) * pct
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    frac = k - lo
    return s[lo] + (s[hi] - s[lo]) * frac


def _histogram_snapshot(samples: List[float]) -> Dict[str, float]:
    if not samples:
        return {"count": 0, "p50": 0.0, "p95": 0.0, "avg": 0.0, "max": 0.0}
    return {
        "count": len(samples),
        "p50": round(_percentile(samples, 0.50), 4),
        "p95": round(_percentile(samples, 0.95), 4),
        "avg": round(sum(samples) / len(samples), 4),
        "max": round(max(samples), 4),
    }


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
    # ── qa.* counters (§8.3 of docs/RAG_QA_PLAN.md) ──
    qa_requests: Dict[str, int] = field(default_factory=dict)          # key = "intent|status"
    qa_tool_calls: Dict[str, int] = field(default_factory=dict)        # key = "tool|outcome"
    qa_mutations: Dict[str, int] = field(default_factory=dict)         # key = "tool|operation"
    qa_faithfulness_failures: int = 0
    qa_memory_overflow_drops: Dict[str, int] = field(default_factory=dict)  # key = layer
    qa_pending_confirmations: int = 0
    qa_active_sessions: int = 0
    # ── qa.* histograms ──
    qa_latency_seconds: Dict[str, List[float]] = field(default_factory=dict)  # phase
    qa_retrieval_hits: Dict[str, List[float]] = field(default_factory=dict)   # channel
    qa_context_tokens: Dict[str, List[float]] = field(default_factory=dict)   # slot
    qa_llm_tokens: Dict[str, List[float]] = field(default_factory=dict)       # direction
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

    # ------------------------------------------------------------------
    # qa.* recorders (§8.3 of docs/RAG_QA_PLAN.md)
    # ------------------------------------------------------------------

    @staticmethod
    def _bump(d: Dict[str, int], key: str, n: int = 1) -> None:
        d[key] = d.get(key, 0) + n

    @staticmethod
    def _record_sample(d: Dict[str, List[float]], key: str, value: float) -> None:
        bucket = d.setdefault(key, [])
        bucket.append(float(value))
        if len(bucket) > _HISTOGRAM_MAX_SAMPLES:
            # Keep most-recent N to bound memory.
            del bucket[: len(bucket) - _HISTOGRAM_MAX_SAMPLES]

    async def record_qa_request(self, *, intent: str, status: str) -> None:
        async with self._lock:
            self._bump(self._c.qa_requests, f"{intent}|{status}")

    async def record_qa_tool_call(self, *, tool: str, outcome: str) -> None:
        """outcome ∈ {ok, validation_error, denied, undone}."""
        async with self._lock:
            self._bump(self._c.qa_tool_calls, f"{tool}|{outcome}")

    async def record_qa_mutation(self, *, tool: str, operation: str) -> None:
        """operation ∈ {created, updated, merged, soft_deleted}."""
        async with self._lock:
            self._bump(self._c.qa_mutations, f"{tool}|{operation}")

    async def record_qa_faithfulness_failure(self, n: int = 1) -> None:
        async with self._lock:
            self._c.qa_faithfulness_failures += n

    async def record_qa_memory_overflow_drop(self, *, layer: str, n: int = 1) -> None:
        async with self._lock:
            self._bump(self._c.qa_memory_overflow_drops, layer, n)

    async def set_qa_pending_confirmations(self, value: int) -> None:
        async with self._lock:
            self._c.qa_pending_confirmations = max(0, int(value))

    async def set_qa_active_sessions(self, value: int) -> None:
        async with self._lock:
            self._c.qa_active_sessions = max(0, int(value))

    async def record_qa_latency(self, *, phase: str, seconds: float) -> None:
        """phase ∈ {planner, retrieval, rerank, llm, verify, total}."""
        async with self._lock:
            self._record_sample(self._c.qa_latency_seconds, phase, seconds)

    async def record_qa_retrieval_hits(self, *, channel: str, count: int) -> None:
        """channel ∈ {cypher, vector, bm25}."""
        async with self._lock:
            self._record_sample(self._c.qa_retrieval_hits, channel, count)

    async def record_qa_context_tokens(self, *, slot: str, tokens: int) -> None:
        """slot ∈ {system, working, summary, episodic, sources}."""
        async with self._lock:
            self._record_sample(self._c.qa_context_tokens, slot, tokens)

    async def record_qa_llm_tokens(self, *, direction: str, tokens: int) -> None:
        """direction ∈ {prompt, completion}."""
        async with self._lock:
            self._record_sample(self._c.qa_llm_tokens, direction, tokens)

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
            "qa": {
                "requests": dict(c.qa_requests),
                "tool_calls": dict(c.qa_tool_calls),
                "mutations": dict(c.qa_mutations),
                "faithfulness_failures": c.qa_faithfulness_failures,
                "memory_overflow_drops": dict(c.qa_memory_overflow_drops),
                "pending_confirmations": c.qa_pending_confirmations,
                "active_sessions": c.qa_active_sessions,
                "latency_seconds": {
                    phase: _histogram_snapshot(s)
                    for phase, s in c.qa_latency_seconds.items()
                },
                "retrieval_hits": {
                    channel: _histogram_snapshot(s)
                    for channel, s in c.qa_retrieval_hits.items()
                },
                "context_tokens": {
                    slot: _histogram_snapshot(s)
                    for slot, s in c.qa_context_tokens.items()
                },
                "llm_tokens": {
                    direction: _histogram_snapshot(s)
                    for direction, s in c.qa_llm_tokens.items()
                },
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
