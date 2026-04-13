"""Ingest router — Open Targets and PubMed."""

from fastapi import APIRouter, BackgroundTasks, Depends

from ..auth import require_api_key
from ..dependencies import get_app_config, get_graph_repo, get_document_repo, get_llm
from ..job_store import JobStatus, create_job, update_job
from ..schemas.ingest import IngestResponse, OpenTargetsIngestRequest, PubMedIngestRequest, CrawlIngestRequest

router = APIRouter(prefix="/ingest", tags=["ingest"])


async def _run_open_targets(job_id: str, request: OpenTargetsIngestRequest, config, graph_repo):
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
    from graphbuilder.application.use_cases.open_targets_ingestion import (
        OpenTargetsIngestionUseCase, IngestionConfig,
    )
    from graphbuilder.infrastructure.repositories.graph_repository import GraphRepositoryInterface

    try:
        # Validate graph_repo is the right type — if the dependency injector
        # passed the config object by mistake, rebuild the repo.
        if not isinstance(graph_repo, GraphRepositoryInterface):
            from graphbuilder.infrastructure.repositories.graph_repository import create_graph_repository
            graph_repo = create_graph_repository(config)

        update_job(job_id, status=JobStatus.RUNNING, message="Fetching from Open Targets…")
        cfg = IngestionConfig(
            disease_id=request.disease_id,
            max_associations=request.max_associations,
            min_association_score=request.min_association_score,
            tag=request.tag,
        )
        use_case = OpenTargetsIngestionUseCase(config, graph_repo)
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
    from graphbuilder.infrastructure.repositories.graph_repository import GraphRepositoryInterface

    try:
        if not isinstance(graph_repo, GraphRepositoryInterface):
            from graphbuilder.infrastructure.repositories.graph_repository import create_graph_repository
            graph_repo = create_graph_repository(config)

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
        use_case = PubMedIngestionUseCase(config, graph_repo)
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


# ── Web Crawl → Document Pipeline ────────────────────────────────────────

async def _run_crawl(job_id: str, request: CrawlIngestRequest, config, graph_repo, doc_repo, llm_service):
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

    from graphbuilder.infrastructure.crawlers.web_crawler import WebCrawler
    from graphbuilder.infrastructure.crawlers.crawler_cache import CrawlerCache
    from graphbuilder.infrastructure.config.settings import CrawlerConfiguration
    from graphbuilder.application.use_cases.document_processing import ProcessDocumentUseCase
    from graphbuilder.domain.models.graph_models import SourceDocument
    from graphbuilder.infrastructure.repositories.graph_repository import GraphRepositoryInterface

    try:
        if not isinstance(graph_repo, GraphRepositoryInterface):
            from graphbuilder.infrastructure.repositories.graph_repository import create_graph_repository
            graph_repo = create_graph_repository(config)

        update_job(job_id, status=JobStatus.RUNNING, message="Crawling web pages…")

        # Build crawler config with request overrides
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

        update_job(job_id, status=JobStatus.RUNNING,
                   message=f"Crawled {len(pages)} pages. Processing as documents…",
                   progress=0.3)

        # Process each crawled page through the document pipeline
        pages_processed = 0
        entities_total = 0

        for url, html_text in pages.items():
            # Strip HTML tags to get text content
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html_text, "html.parser")
            # Remove script and style elements
            for tag in soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()
            clean_text = soup.get_text(separator="\n", strip=True)

            if len(clean_text.strip()) < 50:
                continue  # Skip near-empty pages

            doc = SourceDocument(
                url=url,
                title=soup.title.string if soup.title else url,
                content=clean_text,
            )
            doc.metadata.source_trust = "extracted"
            doc.metadata.source_system = "web_crawler"
            doc.metadata.add_tag(request.tag or "web-crawl")

            try:
                use_case = ProcessDocumentUseCase(config, doc_repo, graph_repo, llm_service)
                result = await use_case.execute(doc)
                pages_processed += 1
                if result.data:
                    entities_total += result.data.get("entities_extracted", 0)
            except Exception as page_exc:
                import logging
                logging.getLogger("CrawlIngest").warning("Failed to process %s: %s", url, page_exc)

            progress = 0.3 + 0.7 * (pages_processed / max(len(pages), 1))
            update_job(job_id, progress=progress,
                       message=f"Processed {pages_processed}/{len(pages)} pages…")

        update_job(
            job_id,
            status=JobStatus.COMPLETED,
            message=f"Crawled {len(pages)} pages, processed {pages_processed} as documents, extracted {entities_total} entities",
            progress=1.0,
            result={
                "pages_crawled": len(pages),
                "pages_processed": pages_processed,
                "entities_extracted": entities_total,
                "cache_stats": cache.stats(),
            },
        )
    except Exception as exc:
        update_job(job_id, status=JobStatus.FAILED, error=str(exc), progress=1.0)


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
    """Crawl web pages and process them as documents through the knowledge extraction pipeline."""
    job = create_job()
    background_tasks.add_task(_run_crawl, job.job_id, request, config, graph_repo, doc_repo, llm_service)
    return IngestResponse(job_id=job.job_id, source="web-crawl", status="pending")
