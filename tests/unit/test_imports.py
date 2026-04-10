"""
Smoke tests — verify every top-level module in the package can be imported
without raising ImportError or circular-import errors.

These tests require only the installed package; no credentials or running
services are needed.
"""

import importlib
import pytest

MODULES = [
    # package root
    "graphbuilder",
    # cli
    "graphbuilder.cli.main",
    # application
    "graphbuilder.application.use_cases.document_processing",
    # core
    "graphbuilder.core.graph.transformer",
    "graphbuilder.core.processing.processor",
    "graphbuilder.core.schema.extraction",
    "graphbuilder.core.utils.common_functions",
    "graphbuilder.core.utils.constants",
    # domain
    "graphbuilder.domain.entities.source_node",
    "graphbuilder.domain.models.graph_models",
    "graphbuilder.domain.models.processing_models",
    # infrastructure
    "graphbuilder.infrastructure.config.settings",
    "graphbuilder.infrastructure.crawlers.file_crawler",
    "graphbuilder.infrastructure.crawlers.json_crawler",
    "graphbuilder.infrastructure.crawlers.sync_crawler",
    "graphbuilder.infrastructure.crawlers.web_crawler",
    "graphbuilder.infrastructure.database.neo4j_client",
    "graphbuilder.infrastructure.repositories.document_repository",
    "graphbuilder.infrastructure.repositories.graph_repository",
    "graphbuilder.infrastructure.services.content_extractor",
    "graphbuilder.infrastructure.services.legacy_llm",
    "graphbuilder.infrastructure.services.llm_service",
]


@pytest.mark.parametrize("module", MODULES)
def test_module_imports(module):
    """Each module must be importable with no side effects."""
    importlib.import_module(module)
