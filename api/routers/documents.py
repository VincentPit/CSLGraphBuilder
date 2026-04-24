"""Document processing router with stage-aware progress + cancellation.

Wraps the new ``DocumentExtractionPipeline``:

* ``POST /documents/process`` — kick off a background job; returns the
  job envelope (id, kind=document, ordered stages, initial state).
* ``GET /documents/jobs/{id}`` — current snapshot.
* ``GET /documents/jobs/{id}/stream`` — SSE; emits one event per change
  with the full snapshot (status, current_stage, stage_progress, last
  events). The stream ends when the job reaches a terminal state.
* ``POST /documents/jobs/{id}/cancel`` — cooperative cancel signal.
* ``GET /documents/jobs`` — recent jobs across all kinds.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from sse_starlette.sse import EventSourceResponse

from ..auth import require_api_key
from ..dependencies import get_app_config, get_document_repo, get_graph_repo, get_llm
from ..job_store import (
    DOCUMENT_STAGES,
    JobCancelled,
    JobStatus,
    add_event,
    begin_stage,
    complete_stage,
    create_job,
    get_job,
    is_cancelled,
    list_jobs,
    request_cancel,
    update_job,
)
from ..schemas.documents import (
    DocumentListResponse,
    DocumentStatusResponse,
    JobResponse,
    JobSummary,
    ProcessDocumentRequest,
)

# Make the src/ package importable when running from the repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

router = APIRouter(prefix="/documents", tags=["documents"])


# Stage weights for the global progress bar. Roughly mirrors typical
# wall-clock distribution (chunking is fast, LLM extraction dominates).
_STAGE_WEIGHTS = {
    "fetch": 0.04,
    "chunk": 0.04,
    "entities": 0.38,
    "relationships": 0.34,
    "verify": 0.16,        # cascading verifier on every new relationship
    "finalize": 0.04,
}


def _job_to_response(job) -> JobResponse:
    return JobResponse(**job.to_dict())


def _make_progress_callback(job_id: str):
    """Build a progress callback that updates job state + appends events."""

    seen_stages: set[str] = set()

    async def _cb(stage: str, message: str, fraction: float, data) -> None:
        # First time we see a stage, mark prior stages complete and begin it.
        if stage not in seen_stages:
            for prior in DOCUMENT_STAGES:
                if prior == stage:
                    break
                if prior in seen_stages:
                    continue
                seen_stages.add(prior)
                complete_stage(job_id, prior)
            seen_stages.add(stage)
            begin_stage(job_id, stage, message=f"{stage.capitalize()} stage started")

        # Compute weighted global progress: sum of fully-done stages plus
        # the in-flight stage's fractional contribution.
        total = 0.0
        for s in DOCUMENT_STAGES:
            w = _STAGE_WEIGHTS.get(s, 0.0)
            if s == stage:
                total += w * max(0.0, min(1.0, fraction))
                break
            if s in seen_stages:
                total += w
        update_job(
            job_id,
            status=JobStatus.RUNNING,
            message=message,
            progress=total,
            current_stage=stage,
        )
        if message:
            add_event(job_id, stage=stage, message=message, data=data or {})
        # Yield to the event loop so SSE can drain frequently
        await asyncio.sleep(0)

    return _cb


async def _run_processing(
    job_id: str,
    request: ProcessDocumentRequest,
    config,
    graph_repo,
    doc_repo,
    llm_service,
) -> None:
    from graphbuilder.application.use_cases.document_pipeline import (
        DocumentExtractionPipeline,
        DocumentInput,
    )

    update_job(job_id, status=JobStatus.RUNNING, progress=0.0)
    add_event(job_id, message="Pipeline started")

    if llm_service is None:
        update_job(
            job_id,
            status=JobStatus.FAILED,
            error="LLM service unavailable — set LLM_API_KEY",
            progress=1.0,
        )
        add_event(
            job_id,
            level="error",
            message="LLM service unavailable — check LLM_API_KEY",
        )
        return

    pipeline = DocumentExtractionPipeline(config, doc_repo, graph_repo, llm_service)
    progress_cb = _make_progress_callback(job_id)

    doc_input = DocumentInput(
        title=request.title or request.url or "text-input",
        content=request.text or "",
        source_url=request.url,
        tags=request.tags or [],
        chunk_size=request.chunk_size,
        chunk_overlap=request.chunk_overlap,
    )

    try:
        result = await pipeline.run(
            doc_input,
            progress=progress_cb,
            cancel_check=lambda: is_cancelled(job_id),
        )
    except JobCancelled:
        update_job(job_id, status=JobStatus.CANCELLED, progress=1.0)
        add_event(job_id, level="warn", message="Job cancelled")
        return
    except Exception as exc:
        update_job(job_id, status=JobStatus.FAILED, error=str(exc), progress=1.0)
        add_event(job_id, level="error", message=f"Pipeline crashed: {exc}")
        return

    if result.cancelled:
        update_job(
            job_id,
            status=JobStatus.CANCELLED,
            message=result.message,
            progress=1.0,
            result=result.to_dict(),
        )
        return

    if result.success:
        # Mark any unseen tail stages complete so the timeline lights up.
        for stage in DOCUMENT_STAGES:
            update_job(job_id, current_stage=stage, stage_status="completed")
        update_job(
            job_id,
            status=JobStatus.COMPLETED,
            message=result.message,
            progress=1.0,
            result=result.to_dict(),
            current_stage="finalize",
            stage_status="completed",
        )
        add_event(job_id, message=result.message)
    else:
        update_job(
            job_id,
            status=JobStatus.FAILED,
            message=result.message,
            error=result.error or result.message,
            progress=1.0,
            result=result.to_dict(),
        )
        add_event(
            job_id,
            level="error",
            message=result.error or result.message,
        )


@router.post("/process", response_model=JobResponse, status_code=202)
async def process_document(
    request: ProcessDocumentRequest,
    background_tasks: BackgroundTasks,
    config=Depends(get_app_config),
    graph_repo=Depends(get_graph_repo),
    doc_repo=Depends(get_document_repo),
    llm_service=Depends(get_llm),
    _=Depends(require_api_key),
):
    job = create_job(kind="document", stages=list(DOCUMENT_STAGES))
    background_tasks.add_task(
        _run_processing, job.job_id, request, config, graph_repo, doc_repo, llm_service
    )
    return _job_to_response(job)


@router.get("/jobs", response_model=list[JobSummary])
async def recent_jobs(
    limit: int = Query(20, ge=1, le=100),
    _=Depends(require_api_key),
):
    return [
        JobSummary(
            job_id=j.job_id,
            kind=j.kind,
            status=j.status,
            message=j.message,
            current_stage=j.current_stage,
            progress=j.progress,
            created_at=j.created_at,
            updated_at=j.updated_at,
        )
        for j in list_jobs(limit=limit)
    ]


@router.get("/jobs/{job_id}", response_model=JobResponse)
async def get_job_status(job_id: str, _=Depends(require_api_key)):
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return _job_to_response(job)


@router.post("/jobs/{job_id}/cancel", response_model=JobResponse)
async def cancel_job(job_id: str, _=Depends(require_api_key)):
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if not request_cancel(job_id):
        raise HTTPException(status_code=409, detail="Job is no longer cancellable")
    return _job_to_response(get_job(job_id))


@router.get("/jobs/{job_id}/stream")
async def stream_job_progress(job_id: str, _=Depends(require_api_key)):
    """SSE — emit a snapshot whenever the job state changes."""

    async def event_generator():
        last_signature = None
        backoff = 0.25
        while True:
            job = get_job(job_id)
            if not job:
                yield {"event": "error", "data": json.dumps({"error": "job not found"})}
                return
            snapshot = job.to_dict()
            # Use updated_at + last event count as a change signature so we
            # only push when something actually changed.
            sig = (snapshot["updated_at"], len(snapshot["events"]))
            if sig != last_signature:
                yield {"event": "progress", "data": json.dumps(snapshot)}
                last_signature = sig
                backoff = 0.25
            if job.status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
                yield {"event": "done", "data": json.dumps(snapshot)}
                return
            await asyncio.sleep(backoff)
            backoff = min(backoff + 0.1, 1.0)

    return EventSourceResponse(event_generator())


@router.get("", response_model=DocumentListResponse)
async def list_documents(
    status: Optional[str] = Query(None),
    limit: int = Query(20, ge=1, le=200),
    offset: int = Query(0, ge=0),
    doc_repo=Depends(get_document_repo),
    _=Depends(require_api_key),
):
    try:
        docs = await doc_repo.list_documents(status_filter=status, limit=limit, offset=offset)
        total = await doc_repo.count_documents(status_filter=status)
    except Exception:
        docs, total = [], 0

    def _to_status(d) -> DocumentStatusResponse:
        return DocumentStatusResponse(
            document_id=d.id,
            job_id="",
            status=d.processing_status.value if hasattr(d.processing_status, "value") else str(d.processing_status),
            url=d.source_url if hasattr(d, "source_url") else getattr(d, "url", None),
            title=d.title,
            chunks_created=getattr(d, "total_chunks", 0),
            entities_extracted=getattr(d, "extracted_entities", 0),
            relationships_extracted=getattr(d, "extracted_relationships", 0),
            created_at=d.metadata.created_at,
            updated_at=d.metadata.updated_at,
        )

    return DocumentListResponse(
        items=[_to_status(d) for d in docs],
        total=total,
        limit=limit,
        offset=offset,
    )
