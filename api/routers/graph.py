"""Graph entities and relationships router."""

from typing import Annotated, List, Optional
from datetime import timezone

from fastapi import APIRouter, Depends, Query

from ..auth import require_api_key
from ..dependencies import get_app_config, get_graph_repo
from ..schemas.graph import (
    EntityListResponse,
    EntityResponse,
    GraphStatsResponse,
    RelationshipListResponse,
    RelationshipResponse,
)

router = APIRouter(prefix="/graph", tags=["graph"])


def _entity_to_response(e) -> EntityResponse:
    ann = e.metadata.annotations if e.metadata else {}
    tags = list(e.metadata.tags) if e.metadata and e.metadata.tags else []
    return EntityResponse(
        id=e.id,
        name=e.name,
        entity_type=e.entity_type.value,
        description=e.description,
        properties=dict(e.properties or {}),
        confidence_score=getattr(e.metadata, "confidence_score", None),
        curated=ann.get("curated", False),
        rejected=ann.get("rejected", False),
        tags=tags,
        created_at=e.metadata.created_at,
        updated_at=e.metadata.updated_at,
    )


def _rel_to_response(r) -> RelationshipResponse:
    ann = r.metadata.annotations if r.metadata else {}
    return RelationshipResponse(
        id=r.id,
        source_entity_id=r.source_entity_id,
        target_entity_id=r.target_entity_id,
        relationship_type=r.relationship_type.value,
        description=r.description,
        strength=r.strength,
        curated=ann.get("curated", False),
        verification_passed=ann.get("verification_passed"),
        verification_confidence=ann.get("verification_confidence"),
        created_at=r.metadata.created_at,
        updated_at=r.metadata.updated_at,
    )


@router.get("/stats", response_model=GraphStatsResponse)
async def graph_stats(
    repo=Depends(get_graph_repo),
    _=Depends(require_api_key),
):
    all_entities = await repo.get_all_entities()
    all_rels = await repo.get_all_relationships()
    entities = list(all_entities.values())
    rels = list(all_rels.values())

    entity_type_counts: dict = {}
    for e in entities:
        key = e.entity_type.value
        entity_type_counts[key] = entity_type_counts.get(key, 0) + 1

    rel_type_counts: dict = {}
    for r in rels:
        key = r.relationship_type.value
        rel_type_counts[key] = rel_type_counts.get(key, 0) + 1

    return GraphStatsResponse(
        total_entities=len(entities),
        total_relationships=len(rels),
        entity_type_counts=entity_type_counts,
        relationship_type_counts=rel_type_counts,
    )


@router.get("/entities", response_model=EntityListResponse)
async def list_entities(
    entity_type: Optional[str] = Query(None),
    limit: int = Query(500, ge=1, le=2000),
    offset: int = Query(0, ge=0),
    repo=Depends(get_graph_repo),
    _=Depends(require_api_key),
):
    all_entities = await repo.get_all_entities()
    entities = list(all_entities.values())
    if entity_type:
        entities = [e for e in entities if e.entity_type.value == entity_type]
    total = len(entities)
    page = entities[offset : offset + limit]
    return EntityListResponse(
        items=[_entity_to_response(e) for e in page],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get("/entities/{entity_id}", response_model=EntityResponse)
async def get_entity(
    entity_id: str,
    repo=Depends(get_graph_repo),
    _=Depends(require_api_key),
):
    entity = await repo.get_entity_by_id(entity_id)
    if entity is None:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Entity not found")
    return _entity_to_response(entity)


@router.get("/relationships", response_model=RelationshipListResponse)
async def list_relationships(
    relationship_type: Optional[str] = Query(None),
    source_entity_id: Optional[str] = Query(None),
    target_entity_id: Optional[str] = Query(None),
    limit: int = Query(500, ge=1, le=2000),
    offset: int = Query(0, ge=0),
    repo=Depends(get_graph_repo),
    _=Depends(require_api_key),
):
    all_rels = await repo.get_all_relationships()
    rels = list(all_rels.values())
    if relationship_type:
        rels = [r for r in rels if r.relationship_type.value == relationship_type]
    if source_entity_id:
        rels = [r for r in rels if r.source_entity_id == source_entity_id]
    if target_entity_id:
        rels = [r for r in rels if r.target_entity_id == target_entity_id]
    total = len(rels)
    page = rels[offset : offset + limit]
    return RelationshipListResponse(
        items=[_rel_to_response(r) for r in page],
        total=total,
        limit=limit,
        offset=offset,
    )
