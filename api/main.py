"""FastAPI application entry point."""

import logging
import logging.handlers
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routers import health, graph, documents, ingest, curation, verification, export, dev


def _configure_logging() -> None:
    """Set up persistent file logging under <project>/logs/."""
    logs_dir = Path(__file__).resolve().parent.parent / "logs"
    logs_dir.mkdir(exist_ok=True)

    fmt = logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Rotating file handler: 5 MB per file, keep last 5
    file_handler = logging.handlers.RotatingFileHandler(
        logs_dir / "api.log",
        maxBytes=5 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)

    # Error-only file for quick triage
    error_handler = logging.handlers.RotatingFileHandler(
        logs_dir / "api_errors.log",
        maxBytes=5 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(fmt)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.addHandler(file_handler)
    root.addHandler(error_handler)


_configure_logging()


async def _warm_embedding_model() -> None:
    """Pre-load the sentence-embedding model in the background.

    Without this, the *first* Process request after a fresh boot pays
    the SapBERT download (~30s) before the pipeline can start. Doing it
    at startup means the API is responsive immediately while the model
    downloads/loads in parallel; if a request comes in before warm-up
    finishes, the lazy-load path in ``embedding_factory`` still works,
    just slower for that one call.
    """
    log = logging.getLogger("graphbuilder.api")
    try:
        # ``run_in_executor`` so the blocking sentence-transformers load
        # doesn't block the asyncio loop or other startup work.
        import asyncio, sys, os as _os
        sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), "..", "src"))
        from graphbuilder.infrastructure.services.embedding_factory import (
            get_model, get_model_name, get_embedding_dim,
        )
        log.info("Warming embedding model in background…")
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, get_model)
        log.info("Embedding model ready: %s (dim=%d)", get_model_name(), get_embedding_dim())
    except Exception as exc:
        log.warning("Embedding model warm-up failed: %s", exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    log = logging.getLogger("graphbuilder.api")
    log.info("API startup")
    # Fire-and-forget; the API is reachable immediately.
    import asyncio
    warm_task = asyncio.create_task(_warm_embedding_model())
    yield
    log.info("API shutdown")
    if not warm_task.done():
        warm_task.cancel()


def create_app() -> FastAPI:
    app = FastAPI(
        title="GraphBuilder API",
        version="2.0.0",
        description="Knowledge-graph construction and curation service.",
        lifespan=lifespan,
    )

    allowed_origins = os.getenv("CORS_ORIGINS", "*").split(",")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health.router)
    app.include_router(graph.router)
    app.include_router(documents.router)
    app.include_router(ingest.router)
    app.include_router(curation.router)
    app.include_router(verification.router)
    app.include_router(export.router)
    app.include_router(dev.router)

    return app


app = create_app()
