"""Ingest router — Open Targets and PubMed."""

from fastapi import APIRouter, BackgroundTasks, Depends

from ..auth import require_api_key
from ..dependencies import get_app_config, get_graph_repo
from ..job_store import JobStatus, create_job, update_job
from ..schemas.ingest import IngestResponse, OpenTargetsIngestRequest, PubMedIngestRequest

router = APIRouter(prefix="/ingest", tags=["ingest"])


async def _run_open_targets(job_id: str, request: OpenTargetsIngestRequest, config, graph_repo):
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
    from graphbuilder.application.use_cases.open_targets_ingestion import (
        OpenTargetsIngestionUseCase, IngestionConfig,
    )

    try:
        update_job(job_id, status=JobStatus.RUNNING, message="Fetching from Open Targets…")
        cfg = IngestionConfig(
            disease_id=request.disease_id,
            max_associations=request.max_associations,
            min_association_score=request.min_association_score,
            tag=request.tag,
        )
        use_case = OpenTargetsIngestionUseCase(graph_repo, config)
        result = await use_case.execute(cfg)
        update_job(
            job_id,
            status=JobStatus.COMPLETED if result.success else JobStatus.FAILED,
            message=result.message,
            progress=1.0,
            result=result.data,
        )
    except Exception as exc:
        update_job(job_id, status=JobStatus.FAILED, error=str(exc), progress=1.0)


async def _run_pubmed(job_id: str, request: PubMedIngestRequest, config, graph_repo):
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
    from graphbuilder.application.use_cases.pubmed_ingestion import (
        PubMedIngestionUseCase, PubMedIngestionConfig,
    )

    try:
        update_job(job_id, status=JobStatus.RUNNING, message="Searching PubMed…")
        cfg = PubMedIngestionConfig(
            query=request.query,
            max_articles=request.max_articles,
            email=request.email,
            api_key=request.api_key,
            include_mesh=request.include_mesh,
            include_keywords=request.include_keywords,
            tag=request.tag,
        )
        use_case = PubMedIngestionUseCase(graph_repo, config)
        result = await use_case.execute(cfg)
        update_job(
            job_id,
            status=JobStatus.COMPLETED if result.success else JobStatus.FAILED,
            message=result.message,
            progress=1.0,
            result=result.data,
        )
    except Exception as exc:
        update_job(job_id, status=JobStatus.FAILED, error=str(exc), progress=1.0)


@router.post("/open-targets", response_model=IngestResponse, status_code=202)
async def ingest_open_targets(
    request: OpenTargetsIngestRequest,
    background_tasks: BackgroundTasks,
    config=Depends(get_app_config),
    graph_repo=Depends(get_graph_repo),
    _=Depends(require_api_key),
):
    job = create_job()
    background_tasks.add_task(_run_open_targets, job.job_id, request, config, graph_repo)
    return IngestResponse(job_id=job.job_id, source="open-targets", status="pending")


@router.post("/pubmed", response_model=IngestResponse, status_code=202)
async def ingest_pubmed(
    request: PubMedIngestRequest,
    background_tasks: BackgroundTasks,
    config=Depends(get_app_config),
    graph_repo=Depends(get_graph_repo),
    _=Depends(require_api_key),
):
    job = create_job()
    background_tasks.add_task(_run_pubmed, job.job_id, request, config, graph_repo)
    return IngestResponse(job_id=job.job_id, source="pubmed", status="pending")
