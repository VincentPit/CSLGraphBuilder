"""Retrieval package — graph + vector hybrid retrieval for the RAG Q&A
chatbot (P3 of docs/RAG_QA_PLAN.md).

Public surface:

- :class:`RetrievedItem`    — a single retrieved entity / relationship /
                              chunk with provenance + per-component
                              confidence (§4 of the plan).
- :class:`RetrievalTrace`   — structured "show your work" record of the
                              channels run + fusion + hydration.
- :class:`RetrievalConfig`  — tuning knobs for channels and fusion.
- :class:`RetrievalOrchestrator` — runs all enabled channels in
                              parallel, fuses via RRF, hydrates chunks,
                              produces final ``RetrievedItem`` list.
"""

from .models import (
    Channel,
    ChannelResult,
    RawHit,
    RetrievedItem,
    RetrievalConfig,
    RetrievalTrace,
)
from .orchestrator import RetrievalOrchestrator
from .rrf import reciprocal_rank_fusion

__all__ = [
    "Channel",
    "ChannelResult",
    "RawHit",
    "RetrievedItem",
    "RetrievalConfig",
    "RetrievalTrace",
    "RetrievalOrchestrator",
    "reciprocal_rank_fusion",
]
