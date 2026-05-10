"""Pydantic schemas for the /qa/ask endpoint (P5 of docs/RAG_QA_PLAN.md).

The shapes mirror the dataclasses returned by ``QAService`` / the
retrieval orchestrator, but are split into explicit request/response
models so OpenAPI docs render cleanly and the frontend client can be
generated from the schema.
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class AblationOverride(BaseModel):
    """Per-request retrieval-config overrides (P13 ablation matrix).

    Every flag is optional — ``None`` means "use the server default".
    The eval harness flips these between runs to isolate each
    channel's contribution; production traffic should leave them all
    null (the API singleton's ``RetrievalConfig`` is the source of
    truth) unless a debug session needs to compare configs.
    """

    enable_cypher_channel: Optional[bool] = None
    enable_vector_channel: Optional[bool] = None
    enable_bm25_channel: Optional[bool] = None
    enable_cross_encoder: Optional[bool] = None
    chunk_neighbour_radius: Optional[int] = Field(None, ge=0, le=5)
    emit_chunk_items: Optional[bool] = None
    entity_type_blocklist: Optional[List[str]] = Field(
        None,
        description=(
            "Override the server's entity-type filter — pass `[]` to "
            "include Person/Document/Organization nodes that the default "
            "blocklist drops, or a custom list to substitute one. "
            "Used by the channel-quality ablation (`with_authors`) to "
            "compare author/paper-pollution-on vs -off."
        ),
    )


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
    ablation: Optional[AblationOverride] = Field(
        None,
        description=(
            "Per-request override of channel/rerank flags. Used by the "
            "P13 ablation matrix runner; null in production traffic."
        ),
    )
    enable_tools: bool = Field(
        False,
        description=(
            "Opt-in to the read-only tool-use loop (P9). When true, the "
            "LLM may call search_graph / get_entity / verify_claim before "
            "answering; tool calls are recorded on the response's "
            "tool_calls field. Default false keeps single-shot generation."
        ),
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


class ClaimVerificationModel(BaseModel):
    """One claim's faithfulness verdict (§6.2 of the plan)."""

    claim_text: str
    cited_indices: List[int] = Field(default_factory=list)
    confidence: float
    method: str = Field(
        ..., description='One of "text_match", "llm", "uncited", "refusal".',
    )
    reasoning: str = ""
    escalated_to_llm: bool = False
    matched_terms: List[str] = Field(default_factory=list)
    missing_terms: List[str] = Field(default_factory=list)


class FaithfulnessModel(BaseModel):
    """Aggregate faithfulness check + per-claim verdicts (§6.2)."""

    overall_score: Optional[float] = Field(
        None,
        description=(
            "Mean confidence over scorable cited claims. None when the "
            "answer had no [n] citations or every claim was uncited."
        ),
    )
    failed_claims: int = 0
    is_refusal: bool = False
    claims: List[ClaimVerificationModel] = Field(default_factory=list)


class ToolCallModel(BaseModel):
    """One tool call the LLM made during the agentic loop (P9)."""

    tool: str
    args: dict = Field(default_factory=dict)
    result: Optional[dict] = None
    error: Optional[str] = None
    latency_ms: int = 0
    tool_call_id: Optional[str] = None


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
    faithfulness: Optional[FaithfulnessModel] = None
    tool_calls: List[ToolCallModel] = Field(
        default_factory=list,
        description=(
            "Read-only tool calls the LLM made during this turn (P9). "
            "Empty when enable_tools=false or the model didn't reach for "
            "a tool."
        ),
    )
    request_id: Optional[str] = None
    latency_ms: int


class FeedbackRequest(BaseModel):
    rating: int = Field(..., ge=-1, le=1, description="-1=down, 0=neutral, 1=up")
    comment: Optional[str] = None


class FeedbackResponse(BaseModel):
    turn_id: str
    accepted: bool
