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


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.getLogger("graphbuilder.api").info("API startup")
    yield
    logging.getLogger("graphbuilder.api").info("API shutdown")


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
