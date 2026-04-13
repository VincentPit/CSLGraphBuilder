"""Curation router — event ingestion and queue management."""

from typing import List, Optional
from fastapi import APIRouter, Depends, Query

from ..auth import require_api_key
from ..dependencies import get_app_config, get_graph_repo
from ..schemas.curation import (
    CurationBatchRequest,
    CurationResultResponse,
)

router = APIRouter(prefix="/curation", tags=["curation"])


@router.post("/events", response_model=CurationResultResponse)
def submit_curation_events(
    request: CurationBatchRequest,
    config=Depends(get_app_config),
    graph_repo=Depends(get_graph_repo),
    _=Depends(require_api_key),
):
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
    from graphbuilder.application.use_cases.curation import CurationUseCase, CurationRequest, CurationAction

    results = []
    errors = []
    for event in request.events:
        try:
            action = CurationAction(event.resolved_action)
            curator = event.curator_id or "anonymous"
            curation_req = CurationRequest(curator=curator)
            target = event.target_id
            reason = event.notes or ""

            if action == CurationAction.APPROVE_ENTITY:
                curation_req.approve_entity(target, reason)
            elif action == CurationAction.REJECT_ENTITY:
                curation_req.reject_entity(target, reason)
            elif action == CurationAction.CORRECT_ENTITY:
                curation_req.correct_entity(target, event.corrections or {}, reason)
            elif action == CurationAction.APPROVE_RELATIONSHIP:
                curation_req.approve_relationship(target, reason)
            elif action == CurationAction.REJECT_RELATIONSHIP:
                curation_req.reject_relationship(target, reason)
            elif action == CurationAction.CORRECT_RELATIONSHIP:
                curation_req.correct_relationship(target, event.corrections or {}, reason)
            else:
                errors.append(f"Unknown action: {event.resolved_action}")
                continue

            use_case = CurationUseCase(graph_repo, config)
            result = use_case.execute(curation_req)
            if result.success:
                results.append({"target_id": target, "status": "ok"})
            else:
                errors.append(result.message)
        except Exception as exc:
            errors.append(str(exc))

    return CurationResultResponse(
        processed=len(results),
        failed=len(errors),
        errors=errors,
    )


@router.get("/queue")
async def get_curation_queue(
    status: Optional[str] = Query(None, description="Filter by annotation status (rejected|flagged)"),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    graph_repo=Depends(get_graph_repo),
    _=Depends(require_api_key),
):
    """Return entities and relationships flagged or rejected by verifiers / LLM."""
    items = []

    all_entities = await graph_repo.get_all_entities()
    for ent in list(all_entities.values()):
        ann = getattr(getattr(ent, "metadata", None), "annotations", {}) or {}
        ent_status = ann.get("verification_status", "")
        if status and ent_status != status:
            continue
        if ent_status in ("rejected", "flagged", "unverified"):
            items.append(
                {
                    "type": "entity",
                    "id": ent.id,
                    "name": ent.name,
                    "entity_type": ent.entity_type.value,
                    "verification_status": ent_status,
                    "notes": ann.get("verification_notes"),
                }
            )

    all_relationships = await graph_repo.get_all_relationships()
    for rel in list(all_relationships.values()):
        ann = getattr(getattr(rel, "metadata", None), "annotations", {}) or {}
        rel_status = ann.get("verification_status", "")
        if status and rel_status != status:
            continue
        if rel_status in ("rejected", "flagged", "unverified"):
            items.append(
                {
                    "type": "relationship",
                    "id": rel.id,
                    "source_entity_id": rel.source_entity_id,
                    "target_entity_id": rel.target_entity_id,
                    "relationship_type": rel.relationship_type.value,
                    "verification_status": rel_status,
                    "notes": ann.get("verification_notes"),
                }
            )

    total = len(items)
    page = items[offset : offset + limit]
    return {"total": total, "items": page, "limit": limit, "offset": offset}
