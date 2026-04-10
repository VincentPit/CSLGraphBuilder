"""FastAPI dependencies — factories injected via Depends()."""

import sys
import os
from functools import lru_cache
from typing import Annotated

from fastapi import Depends, HTTPException, status

# Make sure the src/ package is importable when running from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from graphbuilder.infrastructure.config.settings import GraphBuilderConfig, get_config
from graphbuilder.infrastructure.repositories.graph_repository import (
    GraphRepositoryInterface,
    create_graph_repository,
)
from graphbuilder.infrastructure.repositories.document_repository import (
    create_document_repository,
)
from graphbuilder.infrastructure.services.llm_service import create_llm_service


@lru_cache(maxsize=1)
def _build_config() -> GraphBuilderConfig:
    # Provide a placeholder LLM key so the config validator passes when no key
    # is set.  Actual LLM calls will fail with an auth error from the provider,
    # but graph / curation / health routes work without it.
    os.environ.setdefault("LLM_API_KEY", "not-configured")
    # Provide default DB credentials so read-only/in-memory routes work.
    os.environ.setdefault("NEO4J_URI", "bolt://localhost:7687")
    os.environ.setdefault("NEO4J_USER", "neo4j")
    os.environ.setdefault("NEO4J_PASSWORD", "password")
    return get_config()


def get_app_config() -> GraphBuilderConfig:
    return _build_config()


async def get_graph_repo(
    config: Annotated[GraphBuilderConfig, Depends(get_app_config)],
) -> GraphRepositoryInterface:
    return create_graph_repository(config)


async def get_document_repo(
    config: Annotated[GraphBuilderConfig, Depends(get_app_config)],
):
    return create_document_repository(config)


async def get_llm(
    config: Annotated[GraphBuilderConfig, Depends(get_app_config)],
):
    try:
        return create_llm_service(config)
    except Exception:
        return None
