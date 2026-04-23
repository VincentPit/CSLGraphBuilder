"""Ingest router — Open Targets, PubMed, and web crawls.

Uses the shared structured job store (stages, events, cancellation) so
the frontend can render the same timeline UI for ingest jobs as for
document processing.
"""

from __future__ import annotations

import os
import sys

from fastapi import APIRouter, BackgroundTasks, Depends

from ..auth import require_api_key
from ..dependencies import get_app_config, get_document_repo, get_graph_repo, get_llm
from ..job_store import (
    CRAWL_STAGES,
    INGEST_STAGES,
    JobStatus,
    add_event,
    begin_stage,
    complete_stage,
    create_job,
    is_cancelled,
    update_job,
)
from ..schemas.ingest import (
    CrawlIngestRequest,
    IngestResponse,
    OpenTargetsIngestRequest,
    PubMedIngestRequest,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

router = APIRouter(prefix="/ingest", tags=["ingest"])


def _ensure_graph_repo(graph_repo, config):
    from graphbuilder.infrastructure.repositories.graph_repository import (
        GraphRepositoryInterface,
        create_graph_repository,
    )

    if not isinstance(graph_repo, GraphRepositoryInterface):
        return create_graph_repository(config)
    return graph_repo


# ── Open Targets ─────────────────────────────────────────────────────────


async def _run_open_targets(
    job_id: str, request: OpenTargetsIngestRequest, config, graph_repo
):
    from graphbuilder.application.use_cases.open_targets_ingestion import (
        IngestionConfig,
        OpenTargetsIngestionUseCase,
    )

    try:
        graph_repo = _ensure_graph_repo(graph_repo, config)
        update_job(job_id, status=JobStatus.RUNNING, progress=0.0)
        begin_stage(job_id, "fetch", message="Fetching from Open Targets")
        cfg = IngestionConfig(
            disease_id=request.disease_id,
            max_associations=request.max_associations,
            min_association_score=request.min_association_score,
            tag=request.tag,
        )
        use_case = OpenTargetsIngestionUseCase(config, graph_repo)
        result = await use_case.execute(cfg)
        complete_stage(job_id, "fetch", message="Fetched associations")
        begin_stage(job_id, "persist", message="Saving to graph")
        complete_stage(job_id, "persist", message=result.message)
        update_job(
            job_id,
            status=JobStatus.COMPLETED if result.success else JobStatus.FAILED,
            message=result.message,
            progress=1.0,
            result=result.data,
            error=None if result.success else result.message,
        )
        add_event(job_id, message=result.message)
    except Exception as exc:
        update_job(job_id, status=JobStatus.FAILED, error=str(exc), progress=1.0)
        add_event(job_id, level="error", message=f"Ingest crashed: {exc}")


# ── PubMed ───────────────────────────────────────────────────────────────


async def _run_pubmed(job_id: str, request: PubMedIngestRequest, config, graph_repo):
    from graphbuilder.application.use_cases.pubmed_ingestion import (
        PubMedIngestionConfig,
        PubMedIngestionUseCase,
    )

    try:
        graph_repo = _ensure_graph_repo(graph_repo, config)
        update_job(job_id, status=JobStatus.RUNNING, progress=0.0)
        begin_stage(job_id, "fetch", message=f"Searching PubMed for '{request.query}'")
        cfg = PubMedIngestionConfig(
            query=request.query,
            max_articles=request.max_articles,
            email=request.email,
            api_key=request.api_key,
            include_mesh=request.include_mesh,
            include_keywords=request.include_keywords,
            tag=request.tag,
        )
        use_case = PubMedIngestionUseCase(config, graph_repo)
        result = await use_case.execute(cfg)
        complete_stage(job_id, "fetch", message="PubMed query complete")
        begin_stage(job_id, "persist", message="Saving to graph")
        complete_stage(job_id, "persist", message=result.message)
        update_job(
            job_id,
            status=JobStatus.COMPLETED if result.success else JobStatus.FAILED,
            message=result.message,
            progress=1.0,
            result=result.data,
            error=None if result.success else result.message,
        )
        add_event(job_id, message=result.message)
    except Exception as exc:
        update_job(job_id, status=JobStatus.FAILED, error=str(exc), progress=1.0)
        add_event(job_id, level="error", message=f"Ingest crashed: {exc}")


@router.post("/open-targets", response_model=IngestResponse, status_code=202)
async def ingest_open_targets(
    request: OpenTargetsIngestRequest,
    background_tasks: BackgroundTasks,
    config=Depends(get_app_config),
    graph_repo=Depends(get_graph_repo),
    _=Depends(require_api_key),
):
    job = create_job(kind="open-targets", stages=list(INGEST_STAGES))
    background_tasks.add_task(
        _run_open_targets, job.job_id, request, config, graph_repo
    )
    return IngestResponse(job_id=job.job_id, source="open-targets", status="pending")


@router.post("/pubmed", response_model=IngestResponse, status_code=202)
async def ingest_pubmed(
    request: PubMedIngestRequest,
    background_tasks: BackgroundTasks,
    config=Depends(get_app_config),
    graph_repo=Depends(get_graph_repo),
    _=Depends(require_api_key),
):
    job = create_job(kind="pubmed", stages=list(INGEST_STAGES))
    background_tasks.add_task(_run_pubmed, job.job_id, request, config, graph_repo)
    return IngestResponse(job_id=job.job_id, source="pubmed", status="pending")


# ── Web Crawl → Document Pipeline ────────────────────────────────────────


async def _run_crawl(
    job_id: str,
    request: CrawlIngestRequest,
    config,
    graph_repo,
    doc_repo,
    llm_service,
):
    from bs4 import BeautifulSoup

    from graphbuilder.application.use_cases.document_pipeline import (
        DocumentExtractionPipeline,
        DocumentInput,
    )
    from graphbuilder.infrastructure.config.settings import CrawlerConfiguration
    from graphbuilder.infrastructure.crawlers.crawler_cache import CrawlerCache
    from graphbuilder.infrastructure.crawlers.web_crawler import WebCrawler

    try:
        graph_repo = _ensure_graph_repo(graph_repo, config)
        update_job(job_id, status=JobStatus.RUNNING, progress=0.0)
        begin_stage(job_id, "crawl", message=f"Crawling {len(request.urls)} seed URLs")

        crawler_config = CrawlerConfiguration()
        crawler_config.max_urls = request.max_pages
        if request.allowed_domains:
            crawler_config.allowed_domains = request.allowed_domains

        cache = CrawlerCache()
        crawler = WebCrawler(crawler_config, cache=cache)
        pages = await crawler.crawl(
            start_urls=request.urls,
            extra_allowed_domains=request.allowed_domains,
        )
        complete_stage(
            job_id, "crawl", message=f"Crawled {len(pages)} pages"
        )

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
                message="LLM service unavailable; cannot extract knowledge from crawled pages.",
            )
            return

        begin_stage(
            job_id,
            "process",
            message=f"Extracting knowledge from {len(pages)} pages",
        )
        update_job(job_id, progress=0.4)
        pipeline = DocumentExtractionPipeline(config, doc_repo, graph_repo, llm_service)
        pages_processed = 0
        entities_total = 0
        relationships_total = 0
        page_count = max(len(pages), 1)

        for url, html_text in pages.items():
            if is_cancelled(job_id):
                add_event(job_id, level="warn", message="Crawl cancelled by user")
                update_job(job_id, status=JobStatus.CANCELLED, progress=1.0)
                return
            soup = BeautifulSoup(html_text, "html.parser")
            for tag in soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()
            clean_text = soup.get_text(separator="\n", strip=True)
            if len(clean_text.strip()) < 50:
                continue

            doc_input = DocumentInput(
                title=(soup.title.string if soup.title else url) or url,
                content=clean_text,
                source_url=url,
                tags=[request.tag or "web-crawl"],
            )
            try:
                result = await pipeline.run(
                    doc_input,
                    cancel_check=lambda: is_cancelled(job_id),
                )
                pages_processed += 1
                entities_total += result.entities_extracted
                relationships_total += result.relationships_extracted
            except Exception as page_exc:
                add_event(
                    job_id,
                    level="warn",
                    message=f"Failed page {url}: {page_exc}",
                )

            progress = 0.4 + 0.55 * (pages_processed / page_count)
            update_job(
                job_id,
                progress=progress,
                message=(
                    f"Processed {pages_processed}/{len(pages)} pages — "
                    f"{entities_total} entities, {relationships_total} relationships"
                ),
            )

        complete_stage(
            job_id,
            "process",
            message=f"Extraction complete ({pages_processed}/{len(pages)} pages)",
        )
        update_job(
            job_id,
            status=JobStatus.COMPLETED,
            message=(
                f"Crawled {len(pages)} pages, processed {pages_processed}, "
                f"extracted {entities_total} entities + {relationships_total} relationships"
            ),
            progress=1.0,
            result={
                "pages_crawled": len(pages),
                "pages_processed": pages_processed,
                "entities_extracted": entities_total,
                "relationships_extracted": relationships_total,
                "cache_stats": cache.stats(),
            },
        )
    except Exception as exc:
        update_job(job_id, status=JobStatus.FAILED, error=str(exc), progress=1.0)
        add_event(job_id, level="error", message=f"Crawl crashed: {exc}")


@router.post("/crawl", response_model=IngestResponse, status_code=202)
async def ingest_crawl(
    request: CrawlIngestRequest,
    background_tasks: BackgroundTasks,
    config=Depends(get_app_config),
    graph_repo=Depends(get_graph_repo),
    doc_repo=Depends(get_document_repo),
    llm_service=Depends(get_llm),
    _=Depends(require_api_key),
):
    """Crawl web pages and run them through the extraction pipeline."""
    job = create_job(kind="web-crawl", stages=list(CRAWL_STAGES))
    background_tasks.add_task(
        _run_crawl, job.job_id, request, config, graph_repo, doc_repo, llm_service
    )
    return IngestResponse(job_id=job.job_id, source="web-crawl", status="pending")
