"""FastAPI dependencies — factories injected via Depends()."""

import logging
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

logger = logging.getLogger(__name__)


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


def _create_neo4j_driver():
    """Create an async Neo4j driver if DATABASE_PROVIDER=neo4j."""
    uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    user = os.environ.get("NEO4J_USER", "neo4j")
    password = os.environ.get("NEO4J_PASSWORD", "password")
    try:
        from neo4j import AsyncGraphDatabase
        driver = AsyncGraphDatabase.driver(uri, auth=(user, password))
        logger.info("Neo4j async driver created for %s", uri)
        return driver
    except Exception as e:
        logger.warning("Failed to create Neo4j driver: %s — falling back to in-memory", e)
        return None


def get_app_config() -> GraphBuilderConfig:
    return _build_config()


_graph_repo_instance: GraphRepositoryInterface | None = None


async def get_graph_repo(
    config: Annotated[GraphBuilderConfig, Depends(get_app_config)],
) -> GraphRepositoryInterface:
    global _graph_repo_instance
    if _graph_repo_instance is None:
        db_provider = os.getenv("DATABASE_PROVIDER", "in_memory")
        neo4j_driver = None
        if db_provider == "neo4j":
            neo4j_driver = _create_neo4j_driver()
        _graph_repo_instance = create_graph_repository(config, neo4j_driver=neo4j_driver)
        logger.info("Graph repo initialised: %s", type(_graph_repo_instance).__name__)
    return _graph_repo_instance


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
