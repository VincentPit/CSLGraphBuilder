"""In-memory job store for background task tracking."""

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional


class JobStatus:
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class Job:
    __slots__ = ("job_id", "status", "message", "progress", "result", "error",
                 "created_at", "updated_at")

    def __init__(self, job_id: str) -> None:
        self.job_id = job_id
        self.status = JobStatus.PENDING
        self.message: Optional[str] = None
        self.progress: Optional[float] = None
        self.result: Optional[Dict[str, Any]] = None
        self.error: Optional[str] = None
        self.created_at = datetime.now(timezone.utc)
        self.updated_at = datetime.now(timezone.utc)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "status": self.status,
            "message": self.message,
            "progress": self.progress,
            "result": self.result,
            "error": self.error,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


# Module-level singleton — simple enough for a single-process deployment
_STORE: Dict[str, Job] = {}


def create_job() -> Job:
    job_id = str(uuid.uuid4())
    job = Job(job_id)
    _STORE[job_id] = job
    return job


def get_job(job_id: str) -> Optional[Job]:
    return _STORE.get(job_id)


def update_job(
    job_id: str,
    *,
    status: Optional[str] = None,
    message: Optional[str] = None,
    progress: Optional[float] = None,
    result: Optional[Dict[str, Any]] = None,
    error: Optional[str] = None,
) -> None:
    job = _STORE.get(job_id)
    if job is None:
        return
    if status is not None:
        job.status = status
    if message is not None:
        job.message = message
    if progress is not None:
        job.progress = progress
    if result is not None:
        job.result = result
    if error is not None:
        job.error = error
    job.updated_at = datetime.now(timezone.utc)
