"""Curation router — event ingestion and queue management."""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

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


_REVIEWABLE_STATUSES = ("rejected", "flagged", "unverified")
# Most-urgent → least-urgent. Drives the queue ordering in `_sort_key`.
_STATUS_RANK = {"rejected": 0, "flagged": 1, "unverified": 2}
# Sentinel epoch used when an item has no created_at; sorts after real dates.
_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


def _trim(text: Optional[str], max_len: int = 220) -> Optional[str]:
    """Truncate long descriptions for the queue payload."""
    if not text:
        return None
    text = text.strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 1].rsplit(" ", 1)[0] + "…"


def _annotation(obj: Any) -> Dict[str, Any]:
    """Extract the annotations dict off an entity or relationship."""
    return getattr(getattr(obj, "metadata", None), "annotations", {}) or {}


def _parse_iso(value: Any) -> datetime:
    """Best-effort parse of an ISO timestamp; falls back to epoch."""
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return _EPOCH
    return _EPOCH


def _sort_key(item: Dict[str, Any]) -> tuple:
    """Sort: rejected > flagged > unverified; within bucket, newest first."""
    rank = _STATUS_RANK.get(item.get("verification_status"), 99)
    created = _parse_iso(item.get("created_at"))
    # Negate the timestamp so newer (larger) sorts first under ascending sort.
    return (rank, -created.timestamp())


def _entity_to_queue_item(ent: Any, ann: Dict[str, Any], ent_status: str) -> Dict[str, Any]:
    """Shape a GraphEntity into the rich queue payload."""
    tags = list(getattr(ent.metadata, "tags", []) or []) if ent.metadata else []
    return {
        "type": "entity",
        "id": ent.id,
        "name": ent.name,
        "entity_type": ent.entity_type.value,
        "description": _trim(ent.description),
        "verification_status": ent_status,
        "notes": ann.get("verification_notes"),
        "source_chunk_count": len(getattr(ent, "source_chunk_ids", []) or []),
        "source_document_count": len(getattr(ent, "source_document_ids", []) or []),
        "source_trust": getattr(ent.metadata, "source_trust", None) if ent.metadata else None,
        "tags": tags,
        "created_at": ent.metadata.created_at.isoformat() if ent.metadata else None,
    }


def _relationship_to_queue_item(
    rel: Any,
    ann: Dict[str, Any],
    rel_status: str,
    entities_by_id: Dict[str, Any],
) -> Dict[str, Any]:
    """Shape a GraphRelationship + endpoint lookup into the queue payload.

    Resolves source/target entity *names + types* so reviewers see
    "BRCA1 (GENE) → Breast Cancer (DISEASE)" instead of opaque UUIDs.
    """
    src = entities_by_id.get(rel.source_entity_id)
    tgt = entities_by_id.get(rel.target_entity_id)
    return {
        "type": "relationship",
        "id": rel.id,
        "source_entity_id": rel.source_entity_id,
        "source_entity_name": src.name if src else None,
        "source_entity_type": src.entity_type.value if src else None,
        "target_entity_id": rel.target_entity_id,
        "target_entity_name": tgt.name if tgt else None,
        "target_entity_type": tgt.entity_type.value if tgt else None,
        "relationship_type": rel.relationship_type.value,
        "description": _trim(rel.description),
        "strength": getattr(rel, "strength", None),
        "verification_status": rel_status,
        "notes": ann.get("verification_notes"),
        "source_chunk_count": len(getattr(rel, "source_chunk_ids", []) or []),
        "source_document_count": len(getattr(rel, "source_document_ids", []) or []),
        "source_trust": getattr(rel.metadata, "source_trust", None) if rel.metadata else None,
        "created_at": rel.metadata.created_at.isoformat() if rel.metadata else None,
    }


def _passes_status_filter(item_status: str, status_filter: Optional[str]) -> bool:
    """Item must be reviewable AND match the optional status filter."""
    if item_status not in _REVIEWABLE_STATUSES:
        return False
    return status_filter is None or item_status == status_filter


@router.get("/queue")
async def get_curation_queue(
    status: Optional[str] = Query(None, description="Filter by annotation status (rejected|flagged|unverified)"),
    type_: Optional[str] = Query(None, alias="type", description="Filter by 'entity' or 'relationship'"),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    graph_repo=Depends(get_graph_repo),
    _=Depends(require_api_key),
):
    """Return entities and relationships needing human review.

    Each item carries enough context for a reviewer to make a decision
    without a follow-up request: description, source counts (chunks /
    documents), source trust level, tags, and — for relationships —
    the *names* and *types* of both endpoints (not just opaque IDs).
    """
    all_entities = await graph_repo.get_all_entities()
    items: List[Dict[str, Any]] = []

    if type_ in (None, "entity"):
        for ent in all_entities.values():
            ann = _annotation(ent)
            ent_status = ann.get("verification_status", "")
            if _passes_status_filter(ent_status, status):
                items.append(_entity_to_queue_item(ent, ann, ent_status))

    if type_ in (None, "relationship"):
        all_relationships = await graph_repo.get_all_relationships()
        for rel in all_relationships.values():
            ann = _annotation(rel)
            rel_status = ann.get("verification_status", "")
            if _passes_status_filter(rel_status, status):
                items.append(_relationship_to_queue_item(rel, ann, rel_status, all_entities))

    items.sort(key=_sort_key)
    total = len(items)
    return {
        "total": total,
        "items": items[offset : offset + limit],
        "limit": limit,
        "offset": offset,
    }
