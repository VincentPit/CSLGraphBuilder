"""Pydantic schemas for the /qa/ask endpoint (P5 of docs/RAG_QA_PLAN.md).

The shapes mirror the dataclasses returned by ``QAService`` / the
retrieval orchestrator, but are split into explicit request/response
models so OpenAPI docs render cleanly and the frontend client can be
generated from the schema.
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    query: str = Field(..., description="The user's question.")
    session_id: Optional[str] = Field(
        None,
        description=(
            "Existing session id to append this turn to. If omitted, a new "
            "anonymous session is created and returned in the response."
        ),
    )
    user_id: Optional[str] = Field(
        None,
        description=(
            "Optional user id. v1 treats unauthenticated traffic as a single "
            "anonymous user — pass null/omit to use that path."
        ),
    )
    top_k: Optional[int] = Field(
        None, ge=1, le=50,
        description="Override the configured number of sources to retrieve.",
    )


class SourceModel(BaseModel):
    kind: str
    id: str
    label: str
    score_vector: Optional[float] = None
    score_bm25: Optional[float] = None
    score_cypher: Optional[float] = None
    score_rrf: float
    score_rerank: Optional[float] = None
    final_confidence: float
    source_url: Optional[str] = None
    source_doc_id: Optional[str] = None
    source_chunk_id: Optional[str] = None
    source_chunk_ids: List[str] = Field(default_factory=list)
    chunk_preview: Optional[str] = None
    contributing_channels: List[str] = Field(default_factory=list)
    reasoning: str = ""


class ChannelTraceModel(BaseModel):
    channel: str
    hits: int
    latency_ms: int
    error: Optional[str] = None


class RetrievalTraceModel(BaseModel):
    query: str
    extracted_terms: List[str] = Field(default_factory=list)
    channels: List[ChannelTraceModel] = Field(default_factory=list)
    rrf_top_n: int
    final_top_k: int
    hydrated_chunks: int
    total_latency_ms: int


class MemoryEpisodicHit(BaseModel):
    turn_id: str
    score: float


class MemoryTraceModel(BaseModel):
    """Compact view of which memory layers fed this turn (§5 of the plan)."""

    working_turns: int = 0
    summary_chars: int = 0
    episodic_hit: Optional[MemoryEpisodicHit] = None
    summary_regenerated: bool = False


class AskResponse(BaseModel):
    session_id: str
    turn_id: str
    answer: str
    sources: List[SourceModel] = Field(default_factory=list)
    cited_source_indices: List[int] = Field(
        default_factory=list,
        description=(
            "1-indexed positions in `sources` that the LLM cited via [n] "
            "markers in the answer text."
        ),
    )
    retrieval_trace: RetrievalTraceModel
    memory_trace: Optional[MemoryTraceModel] = None
    request_id: Optional[str] = None
    latency_ms: int


class FeedbackRequest(BaseModel):
    rating: int = Field(..., ge=-1, le=1, description="-1=down, 0=neutral, 1=up")
    comment: Optional[str] = None


class FeedbackResponse(BaseModel):
    turn_id: str
    accepted: bool
