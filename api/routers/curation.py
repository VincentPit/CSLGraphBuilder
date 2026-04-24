"""Curation router — event ingestion and queue management."""

import json
import logging
import os
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from ..auth import require_api_key
from ..dependencies import get_app_config, get_graph_repo
from ..schemas.curation import (
    CurationBatchRequest,
    CurationResultResponse,
)


# ── Audit log ────────────────────────────────────────────────────────────
# Every curation event is appended to logs/curation_audit.jsonl so the
# decision history survives backend restarts and can be replayed/queried
# later. JSONL (one record per line) means we can append cheaply and
# load only the tail without parsing the whole file.

_logger = logging.getLogger("graphbuilder.curation_audit")
_AUDIT_PATH = Path(__file__).resolve().parent.parent.parent / "logs" / "curation_audit.jsonl"
_AUDIT_LOCK = threading.Lock()
_AUDIT_MAX_LINES = 5000


def _audit_append(records: List[Dict[str, Any]]) -> None:
    """Append a batch of audit records to the JSONL file."""
    if not records:
        return
    try:
        _AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _AUDIT_LOCK:
            with open(_AUDIT_PATH, "a", encoding="utf-8") as f:
                for rec in records:
                    f.write(json.dumps(rec, default=str) + "\n")
            _audit_truncate_if_needed()
    except Exception as exc:  # pragma: no cover — disk hiccup
        _logger.warning("Failed to append audit records: %s", exc)


def _audit_truncate_if_needed() -> None:
    """Cap the file at the most-recent ``_AUDIT_MAX_LINES`` lines."""
    try:
        with open(_AUDIT_PATH, "r", encoding="utf-8") as f:
            lines = f.readlines()
        if len(lines) <= _AUDIT_MAX_LINES:
            return
        tail = lines[-_AUDIT_MAX_LINES:]
        # Atomic rewrite via temp + rename so we can't lose data on crash.
        fd, tmp = tempfile.mkstemp(prefix=".audit.", suffix=".tmp", dir=str(_AUDIT_PATH.parent))
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.writelines(tail)
        os.replace(tmp, _AUDIT_PATH)
    except Exception as exc:  # pragma: no cover
        _logger.warning("Audit truncation failed: %s", exc)


def _audit_tail(limit: int) -> List[Dict[str, Any]]:
    """Return the most-recent ``limit`` audit records, newest first."""
    if not _AUDIT_PATH.exists():
        return []
    try:
        with open(_AUDIT_PATH, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except Exception as exc:
        _logger.warning("Could not read audit log: %s", exc)
        return []
    out: List[Dict[str, Any]] = []
    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
        if len(out) >= limit:
            break
    return out

router = APIRouter(prefix="/curation", tags=["curation"])


def _build_curation_request(event: Any) -> Any:
    """Translate a transport-level CurationEvent into a domain CurationRequest.

    Centralises the action → builder-method mapping so the route handler
    stays small and readable, and so adding a new action only touches one
    place. Raises ``ValueError`` for unknown actions.
    """
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
    from graphbuilder.application.use_cases.curation import CurationAction, CurationRequest

    action = CurationAction(event.resolved_action)
    curator = event.curator_id or "anonymous"
    target = event.target_id
    reason = event.notes or ""
    corrections = event.corrections or {}
    req = CurationRequest(curator=curator)

    builders = {
        CurationAction.APPROVE_ENTITY:        lambda: req.approve_entity(target, reason),
        CurationAction.REJECT_ENTITY:         lambda: req.reject_entity(target, reason),
        CurationAction.CORRECT_ENTITY:        lambda: req.correct_entity(target, corrections, reason),
        CurationAction.APPROVE_RELATIONSHIP:  lambda: req.approve_relationship(target, reason),
        CurationAction.REJECT_RELATIONSHIP:   lambda: req.reject_relationship(target, reason),
        CurationAction.CORRECT_RELATIONSHIP:  lambda: req.correct_relationship(target, corrections, reason),
    }
    builder = builders.get(action)
    if builder is None:
        raise ValueError(f"Unknown action: {event.resolved_action}")
    builder()
    return req


def _audit_record(event: Any, *, success: bool, message: Optional[str] = None, error: Optional[str] = None) -> Dict[str, Any]:
    """Build a single audit-log entry for one event."""
    rec = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "action": getattr(event, "resolved_action", "unknown"),
        "target_id": getattr(event, "target_id", None),
        "curator": getattr(event, "curator_id", None) or "anonymous",
        "reason": getattr(event, "notes", None) or "",
        "corrections": getattr(event, "corrections", None) or {},
        "success": success,
    }
    if message is not None:
        rec["message"] = message
    if error is not None:
        rec["error"] = error
    return rec


@router.post("/events", response_model=CurationResultResponse)
async def submit_curation_events(
    request: CurationBatchRequest,
    config=Depends(get_app_config),
    graph_repo=Depends(get_graph_repo),
    _=Depends(require_api_key),
):
    """Apply a batch of curation events.

    The use case is async; this endpoint must be ``async def`` and
    ``await`` it (was previously ``def`` + non-awaited call — a silent
    bug that discarded every event). After applying, every event is
    appended to the persistent audit log so decisions survive backend
    restarts.
    """
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
    from graphbuilder.application.use_cases.curation import CurationUseCase

    use_case = CurationUseCase(config, graph_repo)
    results: List[Dict[str, Any]] = []
    errors: List[str] = []
    audit_records: List[Dict[str, Any]] = []

    for event in request.events:
        try:
            req = _build_curation_request(event)
            result = await use_case.execute(req)
            audit_records.append(_audit_record(event, success=bool(result.success), message=result.message))
            if result.success:
                results.append({"target_id": event.target_id, "status": "ok"})
            else:
                errors.append(result.message)
        except Exception as exc:
            errors.append(str(exc))
            audit_records.append(_audit_record(event, success=False, error=str(exc)))

    _audit_append(audit_records)

    return CurationResultResponse(
        processed=len(results),
        failed=len(errors),
        errors=errors,
    )


@router.get("/audit", summary="Recent curation events (newest first)")
async def get_audit_log(
    limit: int = Query(100, ge=1, le=1000),
    _=Depends(require_api_key),
):
    """Return the tail of ``logs/curation_audit.jsonl``.

    Useful for UIs that want to surface "recently approved/rejected"
    activity, and for compliance — every action a curator took is here
    with a timestamp and curator identifier.
    """
    items = _audit_tail(limit)
    return {"total": len(items), "items": items}


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
