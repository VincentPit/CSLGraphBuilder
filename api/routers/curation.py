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


def _audit_record(
    event: Any,
    *,
    success: bool,
    message: Optional[str] = None,
    error: Optional[str] = None,
    actor: str = "human",
    request_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a single audit-log entry for one event.

    ``actor`` distinguishes human curator actions ("human") from chatbot
    mutations ("chatbot") — see §8.4 of docs/RAG_QA_PLAN.md. ``request_id``
    is the QA observability spine id; for human-driven calls it falls
    back to whatever the request middleware set on the ambient context.
    """
    if request_id is None:
        # Pull from the qa observability contextvar so curation events
        # initiated through the API also get correlated.
        try:
            from graphbuilder.infrastructure.services.qa_observability import (
                get_request_id,
            )
            request_id = get_request_id()
        except Exception:
            request_id = None

    rec = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "actor": actor,
        "request_id": request_id,
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


def append_chatbot_audit_record(
    *,
    action: str,
    target_id: Optional[str],
    actor_user_id: Optional[str],
    success: bool,
    request_id: Optional[str] = None,
    confirmation_id: Optional[str] = None,
    before: Optional[Dict[str, Any]] = None,
    after: Optional[Dict[str, Any]] = None,
    reason: Optional[str] = None,
    error: Optional[str] = None,
) -> Dict[str, Any]:
    """Public helper for the QA tool layer to write into the same audit log.

    Mirrors the shape ``_audit_record`` produces so existing readers
    (``GET /curation/audit``, the curation review queue UI) work unchanged.
    """
    if request_id is None:
        try:
            from graphbuilder.infrastructure.services.qa_observability import (
                get_request_id,
            )
            request_id = get_request_id()
        except Exception:
            request_id = None

    rec: Dict[str, Any] = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "actor": "chatbot",
        "request_id": request_id,
        "action": action,
        "target_id": target_id,
        "curator": actor_user_id or "chatbot",
        "reason": reason or "",
        "corrections": {},
        "before": before or {},
        "after": after or {},
        "confirmation_id": confirmation_id,
        "success": success,
    }
    if error is not None:
        rec["error"] = error
    _audit_append([rec])
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


# ── Cypher-side filter helpers ──────────────────────────────────────────
#
# Both queue endpoints used to do `await graph_repo.get_all_entities()`
# and `get_all_relationships()` followed by an in-Python filter loop. With
# thousands of items in the graph, that round-tripped tens of MB of data
# and rehydrated 7500+ Python objects on every call — even though only
# a handful actually need curation.
#
# The new path pushes the status filter into Cypher via substring matches
# on the JSON-stringified ``metadata`` property and returns ONLY rows
# that need review. ``CONTAINS`` is a full scan, but it runs inside the
# DB on a single C++ pass; the dominant cost in the old path was the
# Python deserialization of every node into a ``GraphEntity``.
#
# Long term, the right fix is to lift ``verification_status`` and
# ``curated`` to top-level indexed properties on the node — the payload
# below is structured so that change becomes a one-line WHERE swap.

_STATUS_FRAGMENTS = {
    "rejected": '"verification_status": "rejected"',
    "flagged": '"verification_status": "flagged"',
    "unverified": '"verification_status": "unverified"',
}
_CURATED_FRAGMENT = '"curated": true'


def _build_status_predicate(status_filter: Optional[str], var: str) -> str:
    """Build the WHERE clause that selects items needing curation.

    ``var`` is the Cypher variable bound to the node (e.g. ``e`` for
    entities, ``r`` for relationships). ``status_filter`` narrows to a
    single status when set; otherwise all three reviewable statuses are
    accepted.
    """
    if status_filter and status_filter in _STATUS_FRAGMENTS:
        statuses = [status_filter]
    else:
        statuses = list(_STATUS_FRAGMENTS.keys())
    status_or = " OR ".join(
        f"{var}.metadata CONTAINS '{_STATUS_FRAGMENTS[s]}'" for s in statuses
    )
    return (
        f"({status_or}) AND NOT {var}.metadata CONTAINS '{_CURATED_FRAGMENT}'"
    )


def _row_status(metadata_str: Any) -> str:
    """Pull verification_status from the raw JSON metadata string."""
    if not metadata_str:
        return ""
    if isinstance(metadata_str, str):
        try:
            data = json.loads(metadata_str)
        except (ValueError, json.JSONDecodeError):
            return ""
    elif isinstance(metadata_str, dict):
        data = metadata_str
    else:
        return ""
    return ((data.get("annotations") or {}).get("verification_status") or "")


def _row_notes(metadata_str: Any) -> Optional[str]:
    if not metadata_str:
        return None
    if isinstance(metadata_str, str):
        try:
            data = json.loads(metadata_str)
        except (ValueError, json.JSONDecodeError):
            return None
    elif isinstance(metadata_str, dict):
        data = metadata_str
    else:
        return None
    return (data.get("annotations") or {}).get("verification_notes")


def _row_to_iso(value: Any) -> Optional[str]:
    """Coerce a Neo4j datetime / string / None to an ISO string."""
    if value is None:
        return None
    if hasattr(value, "iso_format"):  # neo4j.time.DateTime
        return value.iso_format()
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


@router.get("/queue/counts")
async def get_curation_queue_counts(
    type_: Optional[str] = Query(None, alias="type", description="Filter by 'entity' or 'relationship'"),
    graph_repo=Depends(get_graph_repo),
    _=Depends(require_api_key),
):
    """Return per-status counts of items still pending human review.

    Single Cypher round-trip with ``sum(CASE …)`` aggregations so we
    never pull node payloads into Python just to count them.
    """
    counts: Dict[str, int] = {"total": 0, "rejected": 0, "flagged": 0, "unverified": 0}

    async def _count(label_pattern: str, var: str) -> Dict[str, int]:
        not_curated = f"NOT {var}.metadata CONTAINS '{_CURATED_FRAGMENT}'"
        query = (
            f"MATCH {label_pattern} "
            f"WHERE {not_curated} "
            f"RETURN "
            f"  sum(CASE WHEN {var}.metadata CONTAINS '{_STATUS_FRAGMENTS['rejected']}' THEN 1 ELSE 0 END) AS rejected, "
            f"  sum(CASE WHEN {var}.metadata CONTAINS '{_STATUS_FRAGMENTS['flagged']}' THEN 1 ELSE 0 END) AS flagged, "
            f"  sum(CASE WHEN {var}.metadata CONTAINS '{_STATUS_FRAGMENTS['unverified']}' THEN 1 ELSE 0 END) AS unverified"
        )
        rows = await graph_repo.execute_cypher_query(query, {})
        if not rows:
            return {"rejected": 0, "flagged": 0, "unverified": 0}
        row = rows[0]
        return {
            "rejected": int(row.get("rejected") or 0),
            "flagged": int(row.get("flagged") or 0),
            "unverified": int(row.get("unverified") or 0),
        }

    if type_ in (None, "entity"):
        ent_counts = await _count("(e:Entity)", "e")
        for k in ("rejected", "flagged", "unverified"):
            counts[k] += ent_counts[k]

    if type_ in (None, "relationship"):
        rel_counts = await _count("()-[r:RELATES]->()", "r")
        for k in ("rejected", "flagged", "unverified"):
            counts[k] += rel_counts[k]

    counts["total"] = counts["rejected"] + counts["flagged"] + counts["unverified"]
    return counts


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

    The status filter runs in Cypher (``CONTAINS`` substring match on
    the JSON-stringified ``metadata`` property), so we only deserialize
    rows that actually need review — not every entity in the graph.
    """
    items: List[Dict[str, Any]] = []
    where = _build_status_predicate(status, "e")

    if type_ in (None, "entity"):
        ent_query = (
            f"MATCH (e:Entity) "
            f"WHERE {where} "
            f"RETURN "
            f"  e.id AS id, e.name AS name, e.entity_type AS entity_type, "
            f"  e.description AS description, e.source_trust AS source_trust, "
            f"  e.tags AS tags, "
            f"  size(coalesce(e.source_chunk_ids, [])) AS source_chunk_count, "
            f"  size(coalesce(e.source_document_ids, [])) AS source_document_count, "
            f"  e.created_at AS created_at, e.metadata AS metadata"
        )
        rows = await graph_repo.execute_cypher_query(ent_query, {})
        for row in rows:
            row_status = _row_status(row.get("metadata"))
            if row_status not in _REVIEWABLE_STATUSES:
                continue
            items.append({
                "type": "entity",
                "id": row.get("id"),
                "name": row.get("name"),
                "entity_type": row.get("entity_type"),
                "description": _trim(row.get("description")),
                "verification_status": row_status,
                "notes": _row_notes(row.get("metadata")),
                "source_chunk_count": int(row.get("source_chunk_count") or 0),
                "source_document_count": int(row.get("source_document_count") or 0),
                "source_trust": row.get("source_trust"),
                "tags": list(row.get("tags") or []),
                "created_at": _row_to_iso(row.get("created_at")),
            })

    if type_ in (None, "relationship"):
        rel_where = _build_status_predicate(status, "r")
        rel_query = (
            f"MATCH (s:Entity)-[r:RELATES]->(t:Entity) "
            f"WHERE {rel_where} "
            f"RETURN "
            f"  r.id AS id, r.relationship_type AS relationship_type, "
            f"  r.description AS description, r.strength AS strength, "
            f"  r.source_trust AS source_trust, "
            f"  size(coalesce(r.source_chunk_ids, [])) AS source_chunk_count, "
            f"  size(coalesce(r.source_document_ids, [])) AS source_document_count, "
            f"  r.created_at AS created_at, r.metadata AS metadata, "
            f"  s.id AS source_entity_id, s.name AS source_entity_name, s.entity_type AS source_entity_type, "
            f"  t.id AS target_entity_id, t.name AS target_entity_name, t.entity_type AS target_entity_type"
        )
        rows = await graph_repo.execute_cypher_query(rel_query, {})
        for row in rows:
            row_status = _row_status(row.get("metadata"))
            if row_status not in _REVIEWABLE_STATUSES:
                continue
            items.append({
                "type": "relationship",
                "id": row.get("id"),
                "source_entity_id": row.get("source_entity_id"),
                "source_entity_name": row.get("source_entity_name"),
                "source_entity_type": row.get("source_entity_type"),
                "target_entity_id": row.get("target_entity_id"),
                "target_entity_name": row.get("target_entity_name"),
                "target_entity_type": row.get("target_entity_type"),
                "relationship_type": row.get("relationship_type"),
                "description": _trim(row.get("description")),
                "strength": row.get("strength"),
                "verification_status": row_status,
                "notes": _row_notes(row.get("metadata")),
                "source_chunk_count": int(row.get("source_chunk_count") or 0),
                "source_document_count": int(row.get("source_document_count") or 0),
                "source_trust": row.get("source_trust"),
                "created_at": _row_to_iso(row.get("created_at")),
            })

    items.sort(key=_sort_key)
    total = len(items)
    return {
        "total": total,
        "items": items[offset : offset + limit],
        "limit": limit,
        "offset": offset,
    }
