"""Job store with structured stage/event progress + on-disk persistence.

A job models a long-running pipeline (document processing, web crawl,
ingest). Each job carries:

* A coarse ``status`` (pending/running/completed/failed/cancelled).
* An ordered list of ``stages`` — the canonical pipeline phases. The
  current stage is identified by ``current_stage``.
* An append-only ``events`` log of timestamped progress messages
  (``{ts, stage, level, message, data}``). The frontend renders this
  as a live timeline.
* A cooperative ``cancel_requested`` flag — long-running tasks should
  poll ``is_cancelled(job_id)`` between units of work and raise
  ``JobCancelled`` to short-circuit.

**Persistence**: jobs are mirrored to ``logs/jobs.json`` on every state
change (status, stage, event, result). On startup the file is loaded
back so the Job History page survives backend restarts. Jobs that were
mid-flight when the backend died are loaded but marked ``failed`` with
a synthesized event explaining the interruption — they can't be
resumed (no in-memory worker still holds the future), so this is the
only honest thing to do.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


logger = logging.getLogger("graphbuilder.job_store")

# ──────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────


class JobStatus:
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


_TERMINAL_STATUSES = {JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED}


class JobCancelled(Exception):
    """Raised inside a worker when the user requests cancellation."""


# Canonical pipeline stages — used by the document processing flow.
DOCUMENT_STAGES: List[str] = [
    "fetch",
    "chunk",
    "entities",
    "relationships",
    "verify",
    "finalize",
]

# Web-crawl flow has its own coarser stages.
CRAWL_STAGES: List[str] = ["crawl", "process"]

# Single-stage ingests (open-targets, pubmed).
INGEST_STAGES: List[str] = ["fetch", "persist"]


# Persistence file. Mirrors the rotating log location so dev environments
# get one tidy ``logs/`` directory with everything inspectable on disk.
_PERSIST_PATH = Path(__file__).resolve().parent.parent / "logs" / "jobs.json"
# How many most-recent jobs to keep on disk. Older jobs are evicted to
# stop the file from growing unbounded.
_PERSIST_MAX_JOBS = 500


# ──────────────────────────────────────────────────────────────────────
# Job model
# ──────────────────────────────────────────────────────────────────────


class Job:
    __slots__ = (
        "job_id",
        "kind",
        "status",
        "message",
        "progress",
        "stages",
        "current_stage",
        "stage_progress",
        "events",
        "result",
        "error",
        "cancel_requested",
        "created_at",
        "updated_at",
    )

    def __init__(self, job_id: str, *, kind: str = "document", stages: Optional[List[str]] = None) -> None:
        self.job_id = job_id
        self.kind = kind
        self.status = JobStatus.PENDING
        self.message: Optional[str] = None
        self.progress: float = 0.0
        self.stages: List[str] = list(stages) if stages else list(DOCUMENT_STAGES)
        self.current_stage: Optional[str] = None
        # Per-stage status: pending | running | completed | skipped | failed
        self.stage_progress: Dict[str, str] = {s: "pending" for s in self.stages}
        self.events: List[Dict[str, Any]] = []
        self.result: Optional[Dict[str, Any]] = None
        self.error: Optional[str] = None
        self.cancel_requested: bool = False
        self.created_at = datetime.now(timezone.utc)
        self.updated_at = self.created_at

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "kind": self.kind,
            "status": self.status,
            "message": self.message,
            "progress": self.progress,
            "stages": self.stages,
            "current_stage": self.current_stage,
            "stage_progress": dict(self.stage_progress),
            "events": list(self.events[-200:]),  # cap payload size
            "result": self.result,
            "error": self.error,
            "cancel_requested": self.cancel_requested,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Job":
        job = cls.__new__(cls)
        job.job_id = data["job_id"]
        job.kind = data.get("kind", "document")
        job.status = data.get("status", JobStatus.PENDING)
        job.message = data.get("message")
        job.progress = float(data.get("progress") or 0.0)
        job.stages = list(data.get("stages") or [])
        job.current_stage = data.get("current_stage")
        job.stage_progress = dict(data.get("stage_progress") or {})
        job.events = list(data.get("events") or [])
        job.result = data.get("result")
        job.error = data.get("error")
        job.cancel_requested = bool(data.get("cancel_requested", False))
        job.created_at = _parse_dt(data.get("created_at"))
        job.updated_at = _parse_dt(data.get("updated_at"))
        return job


def _parse_dt(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            pass
    return datetime.now(timezone.utc)


# ──────────────────────────────────────────────────────────────────────
# Disk persistence (atomic write, debounced)
# ──────────────────────────────────────────────────────────────────────


_STORE: Dict[str, Job] = {}
_STORE_LOCK = threading.Lock()


def _atomic_write(path: Path, payload: str) -> None:
    """Write ``payload`` to ``path`` atomically (write to temp + rename)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".jobs.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(payload)
        os.replace(tmp_path, path)
    except Exception:
        # Best-effort cleanup of the temp file on failure.
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _persist() -> None:
    """Snapshot the store to disk. Best-effort: errors are logged not raised
    so a transient I/O failure never crashes a request."""
    try:
        with _STORE_LOCK:
            jobs_sorted = sorted(
                _STORE.values(), key=lambda j: j.created_at, reverse=True
            )[:_PERSIST_MAX_JOBS]
            payload = json.dumps([j.to_dict() for j in jobs_sorted], default=str)
        _atomic_write(_PERSIST_PATH, payload)
    except Exception as exc:  # pragma: no cover — disk hiccup
        logger.warning("Failed to persist job store: %s", exc)


def _hydrate() -> None:
    """Load the persisted store at startup. Mark stale in-flight jobs as
    failed (no worker remains to drive them after a restart)."""
    if not _PERSIST_PATH.exists():
        return
    try:
        raw = _PERSIST_PATH.read_text(encoding="utf-8")
        records = json.loads(raw) if raw.strip() else []
    except Exception as exc:
        logger.warning("Could not load %s: %s", _PERSIST_PATH, exc)
        return

    loaded = 0
    interrupted = 0
    for rec in records:
        if not isinstance(rec, dict) or "job_id" not in rec:
            continue
        try:
            job = Job.from_dict(rec)
        except Exception:
            continue
        if job.status not in _TERMINAL_STATUSES:
            # Backend was killed while this was running — no honest way to
            # resume. Mark failed with a synthesised event so the timeline
            # explains what happened.
            job.status = JobStatus.FAILED
            job.error = "Backend was restarted while this job was in flight."
            job.message = "Interrupted by backend restart"
            job.events.append(
                {
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "stage": job.current_stage,
                    "level": "error",
                    "message": "Backend restart killed the worker; job marked failed.",
                    "data": {},
                }
            )
            interrupted += 1
        _STORE[job.job_id] = job
        loaded += 1

    logger.info(
        "Hydrated %d jobs from %s (%d marked failed due to interrupted run)",
        loaded,
        _PERSIST_PATH,
        interrupted,
    )


# Hydrate exactly once at module import.
_hydrate()


# ──────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────


def create_job(*, kind: str = "document", stages: Optional[List[str]] = None) -> Job:
    job_id = str(uuid.uuid4())
    job = Job(job_id, kind=kind, stages=stages)
    with _STORE_LOCK:
        _STORE[job_id] = job
    _persist()
    return job


def get_job(job_id: str) -> Optional[Job]:
    return _STORE.get(job_id)


def list_jobs(*, limit: int = 50) -> List[Job]:
    """Return jobs newest-first."""
    jobs = sorted(_STORE.values(), key=lambda j: j.created_at, reverse=True)
    return jobs[:limit]


def request_cancel(job_id: str) -> bool:
    job = _STORE.get(job_id)
    if not job or job.status in _TERMINAL_STATUSES:
        return False
    job.cancel_requested = True
    job.updated_at = datetime.now(timezone.utc)
    add_event(job_id, level="warn", message="Cancellation requested by user")
    _persist()
    return True


def is_cancelled(job_id: str) -> bool:
    job = _STORE.get(job_id)
    return bool(job and job.cancel_requested)


def update_job(
    job_id: str,
    *,
    status: Optional[str] = None,
    message: Optional[str] = None,
    progress: Optional[float] = None,
    result: Optional[Dict[str, Any]] = None,
    error: Optional[str] = None,
    current_stage: Optional[str] = None,
    stage_status: Optional[str] = None,
) -> None:
    job = _STORE.get(job_id)
    if job is None:
        return
    prior_status = job.status
    if status is not None:
        job.status = status
    if message is not None:
        job.message = message
    if progress is not None:
        job.progress = max(0.0, min(1.0, float(progress)))
    if result is not None:
        job.result = result
    if error is not None:
        job.error = error
    if current_stage is not None:
        job.current_stage = current_stage
        if current_stage in job.stage_progress:
            job.stage_progress[current_stage] = stage_status or "running"
    job.updated_at = datetime.now(timezone.utc)

    # Persist on meaningful transitions only — every progress tick would
    # thrash the disk. Status changes, stage status changes, and entry
    # into a terminal state are the moments worth saving.
    should_persist = (
        status is not None and status != prior_status
    ) or (current_stage is not None and stage_status is not None)
    if should_persist or job.status in _TERMINAL_STATUSES:
        _persist()


def add_event(
    job_id: str,
    *,
    message: str,
    level: str = "info",
    stage: Optional[str] = None,
    data: Optional[Dict[str, Any]] = None,
) -> None:
    job = _STORE.get(job_id)
    if job is None:
        return
    job.events.append(
        {
            "ts": datetime.now(timezone.utc).isoformat(),
            "stage": stage or job.current_stage,
            "level": level,
            "message": message,
            "data": data or {},
        }
    )
    job.updated_at = datetime.now(timezone.utc)
    # Persist on errors/warnings so failures are never silently lost.
    if level in ("error", "warn"):
        _persist()


def begin_stage(job_id: str, stage: str, *, message: Optional[str] = None) -> None:
    """Mark a stage as running and optionally log a starting event."""
    update_job(job_id, current_stage=stage, stage_status="running")
    add_event(job_id, stage=stage, message=message or f"Stage '{stage}' started")


def complete_stage(job_id: str, stage: str, *, message: Optional[str] = None) -> None:
    update_job(job_id, current_stage=stage, stage_status="completed")
    add_event(job_id, stage=stage, message=message or f"Stage '{stage}' completed")


def skip_stage(job_id: str, stage: str, *, message: Optional[str] = None) -> None:
    update_job(job_id, current_stage=stage, stage_status="skipped")
    add_event(job_id, stage=stage, level="info", message=message or f"Stage '{stage}' skipped")


def fail_stage(job_id: str, stage: str, *, message: str) -> None:
    update_job(job_id, current_stage=stage, stage_status="failed")
    add_event(job_id, stage=stage, level="error", message=message)
