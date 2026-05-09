"""Data models for the retrieval orchestrator.

These types are intentionally small and serialisable — the orchestrator
returns a ``(List[RetrievedItem], RetrievalTrace)`` tuple that the API
layer marshals straight into the ``/qa/ask`` response (§4 + §10 of
docs/RAG_QA_PLAN.md).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class Channel(str, Enum):
    """Retrieval channels (§3.2 of docs/RAG_QA_PLAN.md)."""

    CYPHER = "cypher"
    VECTOR_ENTITY = "vector_entity"
    VECTOR_RELATIONSHIP = "vector_relationship"
    BM25 = "bm25"


class ItemKind(str, Enum):
    ENTITY = "entity"
    RELATIONSHIP = "relationship"
    CHUNK = "chunk"


# ----------------------------------------------------------------------
# Raw hits returned by individual channels
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class RawHit:
    """One hit emitted by a single channel.

    Channels return raw hits keyed by ``(kind, id)``; the orchestrator
    fuses them into ``RetrievedItem`` instances.

    ``label`` is a short human-readable name (entity.name, or
    "src --REL--> tgt" for relationships) used in logs and the LLM
    prompt context.
    """

    kind: ItemKind
    id: str
    label: str
    channel: Channel
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ChannelResult:
    """Output of a single channel run — its raw hits + a small trace."""

    channel: Channel
    hits: List[RawHit] = field(default_factory=list)
    latency_ms: int = 0
    error: Optional[str] = None

    @property
    def hit_count(self) -> int:
        return len(self.hits)


# ----------------------------------------------------------------------
# Final fused items
# ----------------------------------------------------------------------

@dataclass
class RetrievedItem:
    """A single retrieved item that ends up in the LLM prompt + UI source list.

    Confidence components are kept separate so the frontend can render a
    multi-bar breakdown ("how confident are we — vector? cypher? bm25?")
    without backend changes. Components that didn't run are ``None``,
    not zero, so we can distinguish "no signal" from "low signal".
    """

    kind: ItemKind
    id: str
    label: str

    # Per-channel best score, normalised to [0, 1] when known.
    score_vector: Optional[float] = None
    score_bm25: Optional[float] = None
    score_cypher: Optional[float] = None

    # Fused / reranked / final.
    score_rrf: float = 0.0
    score_rerank: Optional[float] = None       # filled in by P4 cross-encoder
    final_confidence: float = 0.0

    # Provenance.
    source_url: Optional[str] = None
    source_doc_id: Optional[str] = None
    source_chunk_id: Optional[str] = None
    source_chunk_ids: List[str] = field(default_factory=list)
    chunk_preview: Optional[str] = None

    # Channels that contributed (for the debug pane).
    contributing_channels: List[Channel] = field(default_factory=list)
    reasoning: str = ""

    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind.value,
            "id": self.id,
            "label": self.label,
            "score_vector": self.score_vector,
            "score_bm25": self.score_bm25,
            "score_cypher": self.score_cypher,
            "score_rrf": round(self.score_rrf, 6),
            "score_rerank": (
                round(self.score_rerank, 6) if self.score_rerank is not None else None
            ),
            "final_confidence": round(self.final_confidence, 4),
            "source_url": self.source_url,
            "source_doc_id": self.source_doc_id,
            "source_chunk_id": self.source_chunk_id,
            "source_chunk_ids": list(self.source_chunk_ids),
            "chunk_preview": self.chunk_preview,
            "contributing_channels": [c.value for c in self.contributing_channels],
            "reasoning": self.reasoning,
            "metadata": dict(self.metadata),
        }


# ----------------------------------------------------------------------
# Trace
# ----------------------------------------------------------------------

@dataclass
class RetrievalTrace:
    """Structured record of one retrieval pass — fed into the debug pane.

    The trace keeps per-channel hit counts + latencies so we can show
    users what happened and so the eval harness can ablate channels.
    """

    query: str
    extracted_terms: List[str] = field(default_factory=list)
    channels: List[ChannelResult] = field(default_factory=list)
    rrf_top_n: int = 0
    final_top_k: int = 0
    hydrated_chunks: int = 0
    total_latency_ms: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "extracted_terms": list(self.extracted_terms),
            "channels": [
                {
                    "channel": c.channel.value,
                    "hits": c.hit_count,
                    "latency_ms": c.latency_ms,
                    "error": c.error,
                }
                for c in self.channels
            ],
            "rrf_top_n": self.rrf_top_n,
            "final_top_k": self.final_top_k,
            "hydrated_chunks": self.hydrated_chunks,
            "total_latency_ms": self.total_latency_ms,
        }


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------

@dataclass
class RetrievalConfig:
    """Tuning knobs for the orchestrator (§3.6 of docs/RAG_QA_PLAN.md).

    Defaults are deliberately conservative — top-k of 8 keeps the prompt
    small; min_score of 0.5 mirrors the existing vector_search_* defaults.
    """

    enable_cypher_channel: bool = True
    enable_vector_channel: bool = True
    enable_bm25_channel: bool = True

    vector_top_k: int = 20
    vector_min_score: float = 0.5

    bm25_limit: int = 20

    cypher_top_k: int = 10

    rrf_k: int = 60
    rrf_top_n: int = 50

    final_top_k: int = 8

    # Chunk hydration — neighbour expansion (NEXT_CHUNK ±1) lands in P4.
    hydrate_chunks: bool = True
    max_chunks_per_item: int = 2
    max_chunk_chars: int = 1200

    # Per-channel timeout so a slow channel can't stall the whole turn.
    channel_timeout_seconds: float = 5.0
