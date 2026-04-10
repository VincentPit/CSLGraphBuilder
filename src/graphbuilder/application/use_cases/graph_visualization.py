"""
Graph Visualization Use Case.

Reads the current knowledge graph from the repository, then exports it to
one of the supported formats via ``GraphExporter``.

Supported formats
-----------------
cytoscape  — Cytoscape.js JSON  (.json)
graphml    — GraphML XML        (.graphml)
html       — Interactive HTML   (.html)
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from ...core.utils.visualization import GraphExporter
from ...domain.models.graph_models import EntityType, KnowledgeGraph, RelationshipType
from ...domain.models.processing_models import ProcessingResult

logger = logging.getLogger(__name__)

SUPPORTED_FORMATS = {"cytoscape", "graphml", "html"}


@dataclass
class VisualizationConfig:
    """Configuration for a single export run."""

    output_path: str
    format: str = "html"                          # one of SUPPORTED_FORMATS
    title: str = "Knowledge Graph"
    include_rejected: bool = False
    # Optional filters (None = include all)
    entity_types: Optional[List[str]] = None      # e.g. ["Concept", "Person"]
    relationship_types: Optional[List[str]] = None


class GraphVisualizationUseCase:
    """
    Export the knowledge graph to a portable format.

    Parameters
    ----------
    graph:
        A populated ``KnowledgeGraph`` instance.  When used through the CLI
        the graph is assembled by the ``GraphRepository`` before this use case
        is called.
    """

    def __init__(self, graph: KnowledgeGraph) -> None:
        self._graph = graph

    def execute(self, config: VisualizationConfig) -> ProcessingResult:
        fmt = config.format.lower()
        if fmt not in SUPPORTED_FORMATS:
            return ProcessingResult(
                success=False,
                message=f"Unsupported format '{fmt}'. Choose from: {', '.join(sorted(SUPPORTED_FORMATS))}",
            )

        # Apply optional entity-type / relationship-type filters
        graph = self._filter_graph(config)

        exporter = GraphExporter(graph)
        output_path = config.output_path

        try:
            if fmt == "cytoscape":
                exporter.to_cytoscape_json(
                    output_path,
                    include_rejected=config.include_rejected,
                )
            elif fmt == "graphml":
                exporter.to_graphml(
                    output_path,
                    include_rejected=config.include_rejected,
                )
            elif fmt == "html":
                exporter.to_html(
                    output_path,
                    title=config.title,
                    include_rejected=config.include_rejected,
                )
        except Exception as exc:
            logger.error("Graph export failed: %s", exc, exc_info=True)
            return ProcessingResult(success=False, message=str(exc))

        node_count = len(graph.entities)
        edge_count = len(graph.relationships)
        logger.info(
            "Exported graph: %d nodes, %d edges → %s (%s)",
            node_count, edge_count, output_path, fmt,
        )

        return ProcessingResult(
            success=True,
            message=f"Exported {node_count} nodes, {edge_count} edges → {output_path} ({fmt})",
            data={
                "format": fmt,
                "output_path": output_path,
                "nodes": node_count,
                "edges": edge_count,
            },
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _filter_graph(self, config: VisualizationConfig) -> KnowledgeGraph:
        """Return a filtered copy of the graph if type filters are set."""
        if config.entity_types is None and config.relationship_types is None:
            return self._graph

        allowed_entity_types = (
            {t.strip() for t in config.entity_types} if config.entity_types else None
        )
        allowed_rel_types = (
            {t.strip() for t in config.relationship_types} if config.relationship_types else None
        )

        filtered = KnowledgeGraph()

        for entity in self._graph.entities.values():
            if allowed_entity_types is None or entity.entity_type.value in allowed_entity_types:
                filtered.entities[entity.id] = entity

        kept_ids = set(filtered.entities.keys())
        for rel in self._graph.relationships.values():
            if rel.source_entity_id not in kept_ids or rel.target_entity_id not in kept_ids:
                continue
            if allowed_rel_types is not None and rel.relationship_type.value not in allowed_rel_types:
                continue
            filtered.relationships[rel.id] = rel

        return filtered
