"""Document processing router with background job support."""

import asyncio
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from sse_starlette.sse import EventSourceResponse

from ..auth import require_api_key
from ..dependencies import get_app_config, get_document_repo, get_graph_repo, get_llm
from ..job_store import JobStatus, create_job, get_job, update_job
from ..schemas.documents import (
    DocumentListResponse,
    DocumentStatusResponse,
    JobResponse,
    ProcessDocumentRequest,
)

router = APIRouter(prefix="/documents", tags=["documents"])


def _job_to_response(job) -> JobResponse:
    return JobResponse(**job.to_dict())


async def _run_processing(
    job_id: str,
    request: ProcessDocumentRequest,
    config,
    graph_repo,
    doc_repo,
    llm_service,
) -> None:
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

    from graphbuilder.application.use_cases.document_processing import ProcessDocumentUseCase
    from graphbuilder.domain.models.graph_models import SourceDocument

    try:
        update_job(job_id, status=JobStatus.RUNNING, message="Processing document…", progress=0.1)

        doc = SourceDocument(
            url=request.url or "",
            title=request.title or (request.url or "text-input"),
            content=request.text or "",
        )

        use_case = ProcessDocumentUseCase(config, doc_repo, graph_repo, llm_service)
        result = await use_case.execute(doc)

        update_job(
            job_id,
            status=JobStatus.COMPLETED if result.success else JobStatus.FAILED,
            message=result.message,
            progress=1.0,
            result=result.data,
            error=None if result.success else result.message,
        )
    except Exception as exc:
        update_job(job_id, status=JobStatus.FAILED, error=str(exc), progress=1.0)


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
    job = create_job()
    background_tasks.add_task(
        _run_processing, job.job_id, request, config, graph_repo, doc_repo, llm_service
    )
    return _job_to_response(job)


@router.get("/jobs/{job_id}", response_model=JobResponse)
async def get_job_status(job_id: str, _=Depends(require_api_key)):
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return _job_to_response(job)


@router.get("/jobs/{job_id}/stream")
async def stream_job_progress(job_id: str, _=Depends(require_api_key)):
    """SSE endpoint — streams job status updates until completion."""

    async def event_generator():
        import json
        while True:
            job = get_job(job_id)
            if not job:
                yield {"data": json.dumps({"error": "job not found"})}
                return
            yield {"data": json.dumps(job.to_dict())}
            if job.status in (JobStatus.COMPLETED, JobStatus.FAILED):
                return
            await asyncio.sleep(0.5)

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
            url=d.url,
            title=d.title,
            chunks_created=getattr(d, "chunk_count", 0),
            entities_extracted=getattr(d, "entity_count", 0),
            relationships_extracted=getattr(d, "relationship_count", 0),
            created_at=d.metadata.created_at,
            updated_at=d.metadata.updated_at,
        )

    return DocumentListResponse(
        items=[_to_status(d) for d in docs],
        total=total,
        limit=limit,
        offset=offset,
    )
