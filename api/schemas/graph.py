"""Pydantic v2 schemas for graph entities and relationships."""

from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class EntityResponse(BaseModel):
    id: str
    name: str
    entity_type: str
    description: Optional[str] = None
    properties: Dict[str, Any] = {}
    confidence_score: Optional[float] = None
    curated: bool = False
    rejected: bool = False
    tags: List[str] = []
    source_chunk_ids: List[str] = []
    source_document_ids: List[str] = []
    created_at: datetime
    updated_at: datetime


class RelationshipResponse(BaseModel):
    id: str
    source_entity_id: str
    target_entity_id: str
    relationship_type: str
    description: Optional[str] = None
    strength: float
    curated: bool = False
    verification_passed: Optional[bool] = None
    verification_confidence: Optional[float] = None
    source_chunk_ids: List[str] = []
    source_document_ids: List[str] = []
    created_at: datetime
    updated_at: datetime


class GraphStatsResponse(BaseModel):
    total_entities: int
    total_relationships: int
    entity_type_counts: Dict[str, int]
    relationship_type_counts: Dict[str, int]


class EntityListResponse(BaseModel):
    items: List[EntityResponse]
    total: int
    limit: int
    offset: int


class RelationshipListResponse(BaseModel):
    items: List[RelationshipResponse]
    total: int
    limit: int
    offset: int


# ── Conflict Detection ───────────────────────────────────────────────────

class ConflictCheckRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        description="Free-text claim to check for conflicts against the knowledge graph",
    )
    use_llm: bool = Field(False, description="Use LLM for semantic conflict analysis")


class ConflictEntryResponse(BaseModel):
    conflict_type: str
    severity: str
    existing_relationship_id: str
    existing_relationship_type: str
    existing_description: str
    existing_source_chunk_ids: List[str]
    new_relationship_type: str
    new_description: str
    new_source_chunk_ids: List[str]
    source_entity_name: str
    target_entity_name: str
    reasoning: str


class ConflictCheckResponse(BaseModel):
    total_checked: int
    conflicts_found: int
    conflicts: List[ConflictEntryResponse]
