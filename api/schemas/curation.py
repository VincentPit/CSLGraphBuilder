"""Pydantic v2 schemas for curation and verification."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ── Curation ────────────────────────────────────────────────────────────────

class CurationEventRequest(BaseModel):
    action: str = Field(
        ...,
        description=(
            "One of: approve_entity, reject_entity, correct_entity, "
            "approve_relationship, reject_relationship, correct_relationship"
        ),
    )
    target_id: str = Field(..., description="Entity or relationship ID")
    curator: str = Field(..., min_length=1)
    reason: Optional[str] = None
    corrections: Optional[Dict[str, Any]] = None


class CurationBatchRequest(BaseModel):
    events: List[CurationEventRequest]
    dry_run: bool = False


class CurationResultResponse(BaseModel):
    success: bool
    applied: int
    failed: int
    audit_log: List[Dict[str, Any]]
    message: str


# ── Verification ─────────────────────────────────────────────────────────────

class VerificationRunRequest(BaseModel):
    relationship_ids: List[str] = Field(
        ...,
        min_length=1,
        description="IDs of the relationships to verify (required, at least one)",
    )
    enable_embedding: bool = False
    enable_llm: bool = False
    embedding_threshold: float = Field(0.5, ge=0.0, le=1.0)
    early_exit_on_pass: bool = False
    early_exit_on_fail: bool = False
    context_map: Dict[str, str] = Field(
        default_factory=dict,
        description="Optional map of relationship_id → supporting context string",
    )


class VerificationStageResult(BaseModel):
    stage: str
    status: str
    confidence: float
    reasoning: str
    metadata: Optional[dict] = None


class VerificationEntryResponse(BaseModel):
    relationship_id: str
    source_entity_id: str
    target_entity_id: str
    relationship_type: str
    status: str
    confidence: float
    reasoning: str
    stage_results: List[VerificationStageResult]


class VerificationReportResponse(BaseModel):
    total: int
    passed: int
    failed: int
    skipped: int
    report: List[VerificationEntryResponse]


# ── Text Verification ────────────────────────────────────────────────────

class TextVerificationRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        description="Free-text description / claim to verify against the knowledge graph",
    )
    enable_embedding: bool = False
    enable_llm: bool = False
    embedding_threshold: float = Field(0.5, ge=0.0, le=1.0)
    early_exit_on_pass: bool = False
    early_exit_on_fail: bool = False
    max_candidates: int = Field(20, ge=1, le=100)


class TextVerificationStageResult(BaseModel):
    stage: str
    status: str
    confidence: float
    reasoning: str
    metadata: Optional[dict] = None


class TextVerificationEntryResponse(BaseModel):
    relationship_id: str
    source_entity_id: str
    target_entity_id: str
    source_entity_name: str
    target_entity_name: str
    relationship_type: str
    relationship_description: str
    status: str
    confidence: float
    reasoning: str
    stage_results: List[TextVerificationStageResult]


class TextVerificationResponse(BaseModel):
    query_text: str
    total_candidates: int
    verified: int
    not_verified: int
    skipped: int
    best_confidence: float
    entries: List[TextVerificationEntryResponse]
