"""Export router — graph export in multiple formats."""

import io
import json

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse

from graphbuilder.core.utils.visualization import GraphExporter
from graphbuilder.domain.models.graph_models import KnowledgeGraph

from ..dependencies import get_graph_repo

router = APIRouter(prefix="/export", tags=["export"])

_FORMAT_MIME = {
    "cytoscape": "application/json",
    "graphml": "application/xml",
    "html": "text/html",
    "json": "application/json",
}


@router.get("")
def export_graph(
    format: str = Query("json", description="Export format: json|cytoscape|graphml|html"),
    graph_repo=Depends(get_graph_repo),
):
    fmt = format.lower()
    if fmt not in _FORMAT_MIME:
        raise HTTPException(status_code=400, detail=f"Unknown format '{fmt}'. Choose: {', '.join(_FORMAT_MIME)}")

    graph = KnowledgeGraph()
    if hasattr(graph_repo, "entities") and isinstance(graph_repo.entities, dict):
        graph.entities = dict(graph_repo.entities)
    if hasattr(graph_repo, "relationships") and isinstance(graph_repo.relationships, dict):
        graph.relationships = dict(graph_repo.relationships)

    exporter = GraphExporter(graph)

    if fmt == "json":
        payload = {
            "entities": [entity.to_dict() for entity in graph.entities.values()],
            "relationships": [relationship.to_dict() for relationship in graph.relationships.values()],
            "statistics": graph.get_statistics(),
        }
        content = json.dumps(payload, indent=2, default=str).encode("utf-8")
    elif fmt == "cytoscape":
        content = json.dumps(exporter.to_cytoscape_json(), indent=2).encode("utf-8")
    elif fmt == "graphml":
        content = exporter.to_graphml().encode("utf-8")
    else:  # html
        content = exporter.to_html().encode("utf-8")

    media_type = _FORMAT_MIME[fmt]
    ext_map = {"cytoscape": "json", "graphml": "xml", "html": "html", "json": "json"}
    filename = f"graph_export.{ext_map[fmt]}"

    return StreamingResponse(
        io.BytesIO(content),
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
