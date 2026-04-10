"""Export router — graph export in multiple formats."""

import io
from fastapi import APIRouter, Depends, Query
from fastapi.responses import StreamingResponse

from ..dependencies import get_app_config, get_graph_repo

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
    config=Depends(get_app_config),
    graph_repo=Depends(get_graph_repo),
):
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
    from graphbuilder.application.use_cases.graph_visualization import (
        GraphVisualizationUseCase, VisualizationConfig,
    )

    fmt = format.lower()
    if fmt not in _FORMAT_MIME:
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail=f"Unknown format '{fmt}'. Choose: {', '.join(_FORMAT_MIME)}")

    vis_cfg = VisualizationConfig(output_format=fmt)
    use_case = GraphVisualizationUseCase(graph_repo, config)
    result = use_case.execute(vis_cfg)

    if not result.success:
        from fastapi import HTTPException
        raise HTTPException(status_code=500, detail=result.message)

    content = result.data.get("content", "")
    if isinstance(content, str):
        content = content.encode()

    media_type = _FORMAT_MIME[fmt]
    ext_map = {"cytoscape": "json", "graphml": "xml", "html": "html", "json": "json"}
    filename = f"graph_export.{ext_map[fmt]}"

    return StreamingResponse(
        io.BytesIO(content),
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
