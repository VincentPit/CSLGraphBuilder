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
