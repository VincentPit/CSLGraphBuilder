"""Pydantic v2 schemas for curation and verification."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ── Curation ────────────────────────────────────────────────────────────────

class CurationEventRequest(BaseModel):
    """Accept the shape the frontend sends.

    The frontend sends separate entity_id / relationship_id fields and simple
    action names (approve, reject, flag, correct).  We normalise here.
    """
    entity_id: Optional[str] = None
    relationship_id: Optional[str] = None
    action: str = Field(
        ...,
        description="One of: approve, reject, flag, correct",
    )
    curator_id: Optional[str] = Field(None, description="Curator identifier")
    notes: Optional[str] = None
    corrections: Optional[Dict[str, Any]] = None

    @property
    def target_id(self) -> str:
        return self.entity_id or self.relationship_id or ""

    @property
    def target_type(self) -> str:
        return "entity" if self.entity_id else "relationship"

    @property
    def resolved_action(self) -> str:
        """Map short action + target type to CurationAction value."""
        base = self.action.lower()
        if base == "flag":
            base = "reject"  # flags treated as soft rejections
        suffix = self.target_type
        return f"{base}_{suffix}"


class CurationBatchRequest(BaseModel):
    events: List[CurationEventRequest]
    dry_run: bool = False


class CurationResultResponse(BaseModel):
    processed: int
    failed: int
    errors: List[str] = []


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
    escalation_lower: float = Field(0.3, ge=0.0, le=1.0, description="Confidence below this is a decisive FAIL")
    escalation_upper: float = Field(0.7, ge=0.0, le=1.0, description="Confidence at or above this is a decisive PASS")
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
    escalation_lower: float = Field(0.3, ge=0.0, le=1.0, description="Confidence below this is a decisive FAIL")
    escalation_upper: float = Field(0.7, ge=0.0, le=1.0, description="Confidence at or above this is a decisive PASS")
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
