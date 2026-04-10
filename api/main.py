"""FastAPI application entry point."""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routers import health, graph, documents, ingest, curation, verification, export


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Nothing to warmup at startup for now — graph repo is initialised lazily
    yield


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

    return app


app = create_app()
