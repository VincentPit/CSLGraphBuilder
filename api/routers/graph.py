"""Graph entities and relationships router."""

from typing import Annotated, List, Optional
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query


# Sort sentinel: items with no created_at sort below all real timestamps,
# so newest-first ordering doesn't crash on older records that pre-date
# the metadata field.
_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)

from ..auth import require_api_key
from ..dependencies import get_app_config, get_document_repo, get_graph_repo
from ..schemas.graph import (
    EntityListResponse,
    EntityResponse,
    GraphStatsResponse,
    RelationshipListResponse,
    RelationshipResponse,
    SubgraphResponse,
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
        source_trust=getattr(e.metadata, "source_trust", None),
        curated=ann.get("curated", False),
        rejected=ann.get("rejected", False),
        tags=tags,
        source_chunk_ids=list(getattr(e, "source_chunk_ids", []) or []),
        source_document_ids=list(getattr(e, "source_document_ids", []) or []),
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
        source_trust=getattr(r.metadata, "source_trust", None),
        source_chunk_ids=list(getattr(r, "source_chunk_ids", []) or []),
        source_document_ids=list(getattr(r, "source_document_ids", []) or []),
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
    """Return entities, **newest first** (by ``created_at``).

    With ``limit=500`` (the default), this gives you the most-recently
    ingested 500 entities. Items without ``created_at`` sort to the
    bottom via the module-level ``_EPOCH`` sentinel so older records
    without timestamps don't crash the comparison.
    """
    all_entities = await repo.get_all_entities()
    entities = list(all_entities.values())
    if entity_type:
        entities = [e for e in entities if e.entity_type.value == entity_type]
    entities.sort(
        key=lambda e: (e.metadata.created_at if e.metadata and e.metadata.created_at else _EPOCH),
        reverse=True,
    )
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
    # Newest first — same ordering contract as /graph/entities so the
    # frontend's default `limit=500` view shows the most-recently-added
    # records, not whatever happens to come first in the dict.
    rels.sort(
        key=lambda r: (r.metadata.created_at if r.metadata and r.metadata.created_at else _EPOCH),
        reverse=True,
    )
    total = len(rels)
    page = rels[offset : offset + limit]
    return RelationshipListResponse(
        items=[_rel_to_response(r) for r in page],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get("/subgraph", response_model=SubgraphResponse)
async def get_subgraph(
    per_type_limit: int = Query(50, ge=1, le=2000),
    exclude_types: str = Query(
        "Document",
        description="Comma-separated entity types to exclude from seeds "
        "(case-insensitive). Default skips Document — those are GWAS "
        "studies and tend to clutter the visualisation.",
    ),
    max_neighbors: int = Query(2000, ge=0, le=10000),
    max_fanout_per_seed: int = Query(
        15, ge=1, le=500,
        description="Cap on edges incident on any single seed entity. "
        "Prevents one disease/drug hub from dragging hundreds of "
        "leaf entities into the view and rendering as a solid disk.",
    ),
    drop_orphans: bool = Query(
        True,
        description="When true (default), entities with zero edges in "
        "the final response are stripped — they're visual noise around "
        "the perimeter of the graph.",
    ),
    repo=Depends(get_graph_repo),
    _=Depends(require_api_key),
):
    """Return a self-consistent slice of the graph for visualisation.

    Seeding is **per entity type**: take the newest ``per_type_limit``
    entities of every type (excluding any in ``exclude_types``). Then:
      1. Every relationship where at least one endpoint is in the seed,
         capped per-seed by ``max_fanout_per_seed`` so dense hubs don't
         dominate. Seed-seed edges win over seed-leaf when capping.
      2. Whatever "other-end" entities the kept edges reach (capped at
         ``max_neighbors`` to bound payload size).
      3. ``drop_orphans=true`` strips any entity with no surviving edge
         from the response so the frontend doesn't render it as a
         disconnected dot in space.
    Every relationship returned has both endpoints in the entity list.
    """
    exclude_set = {
        t.strip().lower() for t in exclude_types.split(",") if t.strip()
    }

    all_entities = await repo.get_all_entities()
    entities = list(all_entities.values())
    total_entities = len(entities)

    # Group by entity type, applying the exclusion filter.
    by_type: dict[str, list] = {}
    for e in entities:
        type_str = e.entity_type.value
        if type_str.lower() in exclude_set:
            continue
        by_type.setdefault(type_str, []).append(e)

    # For each type, take the newest ``per_type_limit`` entities.
    seed_entities = []
    seed_per_type: dict[str, int] = {}
    for type_str, ents in by_type.items():
        ents.sort(
            key=lambda e: (e.metadata.created_at if e.metadata and e.metadata.created_at else _EPOCH),
            reverse=True,
        )
        picked = ents[:per_type_limit]
        seed_entities.extend(picked)
        seed_per_type[type_str] = len(picked)

    seed_ids = {e.id for e in seed_entities}

    # All edges with at least one endpoint in seed.
    all_rels = await repo.get_all_relationships()
    rels_full = list(all_rels.values())
    total_relationships = len(rels_full)
    edges = [
        r for r in rels_full
        if r.source_entity_id in seed_ids or r.target_entity_id in seed_ids
    ]

    # Per-seed fan-out cap. Sort so that:
    #   - Tier 0 (both endpoints seeds): kept first — these are the
    #     "structural" edges between sampled entities, the most
    #     informative for showing how types interconnect.
    #   - Tier 1 (one endpoint seed): kept after, oldest dropped first
    #     when a hub blows past the cap.
    # Within a tier, newer edges win the cap budget.
    def _edge_sort_key(r):
        src_in = r.source_entity_id in seed_ids
        tgt_in = r.target_entity_id in seed_ids
        tier = 0 if (src_in and tgt_in) else 1
        created = r.metadata.created_at if r.metadata and r.metadata.created_at else _EPOCH
        return (tier, -created.timestamp())
    edges.sort(key=_edge_sort_key)

    fan_out_count: dict = {}
    capped_edges = []
    for r in edges:
        # Identify which endpoint(s) of this edge are seeds — those are
        # the ones whose fan-out budget this edge consumes.
        seed_endpoints = [
            eid for eid in (r.source_entity_id, r.target_entity_id)
            if eid in seed_ids
        ]
        # If keeping this edge would exceed the cap for any of those
        # seeds, drop it. Tier-0 edges burn budget on both sides.
        if any(fan_out_count.get(s, 0) >= max_fanout_per_seed for s in seed_endpoints):
            continue
        for s in seed_endpoints:
            fan_out_count[s] = fan_out_count.get(s, 0) + 1
        capped_edges.append(r)
    edges = capped_edges

    # Pull in "other end" entities that the *kept* edges reach. The
    # exclude_types filter applies here too — if the other end is an
    # excluded type, drop the edge rather than show a half-edge.
    extra_ids: set = set()
    for r in edges:
        for endpoint in (r.source_entity_id, r.target_entity_id):
            if endpoint in seed_ids or endpoint in extra_ids:
                continue
            other = all_entities.get(endpoint)
            if other is None or other.entity_type.value.lower() in exclude_set:
                continue
            extra_ids.add(endpoint)
            if len(extra_ids) >= max_neighbors:
                break
        if len(extra_ids) >= max_neighbors:
            break

    extra_entities = [all_entities[eid] for eid in extra_ids if eid in all_entities]

    # Final coherence pass: drop any edge whose other end didn't survive
    # (excluded type or max_neighbors clip). Keeps the invariant that
    # every relationship in the response is renderable.
    final_ids = seed_ids | {e.id for e in extra_entities}
    edges = [
        r for r in edges
        if r.source_entity_id in final_ids and r.target_entity_id in final_ids
    ]

    final_entities = seed_entities + extra_entities

    # Strip orphans: entities with no surviving edge are visual noise
    # in the force-directed layout (they end up as disconnected dots
    # around the perimeter).
    if drop_orphans:
        connected: set = set()
        for r in edges:
            connected.add(r.source_entity_id)
            connected.add(r.target_entity_id)
        final_entities = [e for e in final_entities if e.id in connected]
        # Recount per-type seeds after the orphan cull so the response
        # reflects what's actually rendered.
        kept_seeds = sum(1 for e in final_entities if e.id in seed_ids)
        seed_per_type = {
            t: sum(1 for e in final_entities if e.id in seed_ids and e.entity_type.value == t)
            for t in seed_per_type
        }
    else:
        kept_seeds = len(seed_entities)

    return SubgraphResponse(
        entities=[_entity_to_response(e) for e in final_entities],
        relationships=[_rel_to_response(r) for r in edges],
        seed_count=kept_seeds,
        expanded_count=len([e for e in final_entities if e.id not in seed_ids]),
        seed_per_type=seed_per_type,
        total_entities=total_entities,
        total_relationships=total_relationships,
    )


# ── Chunks ───────────────────────────────────────────────────────────────
# Lookup endpoint used by the curation page to surface the actual text
# behind a flagged extraction. Reviewers can read the source paragraph
# instead of guessing why the LLM produced a given entity / relationship.

@router.get("/chunks", summary="Lookup chunks by ID")
async def get_chunks(
    ids: str = Query(..., description="Comma-separated chunk IDs"),
    limit: int = Query(20, ge=1, le=200),
    doc_repo=Depends(get_document_repo),
    _=Depends(require_api_key),
):
    id_list = [x.strip() for x in (ids or "").split(",") if x.strip()][:limit]
    if not id_list:
        raise HTTPException(status_code=400, detail="At least one chunk id is required")

    try:
        chunks = await doc_repo.get_chunks_by_ids(id_list)
    except Exception as exc:
        # Don't 500 — the curation page is just trying to enrich a card.
        chunks = []
        return {"items": [], "missing": id_list, "error": str(exc)}

    found_ids = {c.id for c in chunks}
    missing = [i for i in id_list if i not in found_ids]
    return {
        "items": [
            {
                "id": c.id,
                "document_id": c.document_id,
                "chunk_index": getattr(c, "chunk_index", 0),
                "content": c.content,
                "character_count": getattr(c, "character_count", len(c.content or "")),
                "token_count": getattr(c, "token_count", None),
            }
            for c in chunks
        ],
        "missing": missing,
    }


# ── Type catalogs ────────────────────────────────────────────────────────
# Drives the Correct-form dropdowns on the curation page so reviewers
# can only pick valid enum values.

@router.get("/types/entities", summary="List valid entity-type values")
async def list_entity_types(_=Depends(require_api_key)):
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
    from graphbuilder.domain.models.graph_models import EntityType
    return {"items": [e.value for e in EntityType]}


@router.get("/types/relationships", summary="List valid relationship-type values")
async def list_relationship_types(_=Depends(require_api_key)):
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
    from graphbuilder.domain.models.graph_models import RelationshipType
    return {"items": [r.value for r in RelationshipType]}
