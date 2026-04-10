"""
Graph visualization and export utilities.

Exports the knowledge graph to portable formats:

* **Cytoscape JSON** — for use with Cytoscape.js and Neo4j Bloom.
* **GraphML**        — XML-based interchange for Gephi, yEd, Cytoscape desktop.
* **HTML viewer**    — self-contained single-file interactive view powered by
                       vis-network (CDN), no build step required.

Typical usage::

    from graphbuilder.core.utils.visualization import GraphExporter
    from graphbuilder.domain.models.graph_models import KnowledgeGraph

    exporter = GraphExporter(graph)
    exporter.to_cytoscape_json("output/graph.json")
    exporter.to_graphml("output/graph.graphml")
    exporter.to_html("output/graph.html")
"""

import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional

from ...domain.models.graph_models import GraphEntity, GraphRelationship, KnowledgeGraph


# Colour palette for entity types (Cytoscape / HTML viewer)
_ENTITY_COLOURS: Dict[str, str] = {
    "Person":        "#4C9BE8",
    "Organization":  "#E87B4C",
    "Location":      "#4CE87B",
    "Product":       "#E8D44C",
    "Technology":    "#9B4CE8",
    "Concept":       "#4CE8D4",
    "Event":         "#E84C9B",
    "Document":      "#A0A0A0",
    "Category":      "#E8A04C",
    "Brand":         "#4CA0E8",
    "Feature":       "#A0E84C",
    "Specification": "#E84CA0",
}
_DEFAULT_COLOUR = "#CCCCCC"


class GraphExporter:
    """
    Export a ``KnowledgeGraph`` to various portable formats.

    Parameters
    ----------
    graph:
        A populated ``KnowledgeGraph`` instance.
    """

    def __init__(self, graph: KnowledgeGraph) -> None:
        self._graph = graph

    # ------------------------------------------------------------------
    # Cytoscape JSON
    # ------------------------------------------------------------------

    def to_cytoscape_json(
        self,
        output_path: Optional[str] = None,
        *,
        include_rejected: bool = False,
    ) -> Dict[str, Any]:
        """
        Serialize the graph as Cytoscape.js JSON.

        Returns the dict and optionally writes it to *output_path*.
        """
        elements: List[Dict[str, Any]] = []

        for entity in self._graph.entities.values():
            if not include_rejected and entity.metadata.annotations.get("rejected"):
                continue
            colour = _ENTITY_COLOURS.get(entity.entity_type.value, _DEFAULT_COLOUR)
            node: Dict[str, Any] = {
                "data": {
                    "id": entity.id,
                    "label": entity.name,
                    "type": entity.entity_type.value,
                    "description": entity.description or "",
                    "curated": entity.metadata.annotations.get("curated", False),
                    "rejected": entity.metadata.annotations.get("rejected", False),
                    **{k: v for k, v in entity.properties.items()},
                },
                "style": {"background-color": colour},
            }
            elements.append({"group": "nodes", **node})

        entity_ids = set(self._graph.entities.keys())
        for rel in self._graph.relationships.values():
            if rel.source_entity_id not in entity_ids or rel.target_entity_id not in entity_ids:
                continue
            if not include_rejected and rel.metadata.annotations.get("rejected"):
                continue
            edge: Dict[str, Any] = {
                "data": {
                    "id": rel.id,
                    "source": rel.source_entity_id,
                    "target": rel.target_entity_id,
                    "label": rel.relationship_type.value,
                    "strength": rel.strength,
                    "description": rel.description or "",
                    "curated": rel.metadata.annotations.get("curated", False),
                },
            }
            elements.append({"group": "edges", **edge})

        cyto = {"elements": elements}

        if output_path:
            _write_text(output_path, json.dumps(cyto, indent=2))

        return cyto

    # ------------------------------------------------------------------
    # GraphML
    # ------------------------------------------------------------------

    def to_graphml(
        self,
        output_path: Optional[str] = None,
        *,
        include_rejected: bool = False,
    ) -> str:
        """
        Serialize the graph as GraphML XML.

        Returns the XML string and optionally writes it to *output_path*.
        """
        root = ET.Element(
            "graphml",
            xmlns="http://graphml.graphdrawing.org/graphml",
        )

        # Declare attribute keys
        _key(root, "d0", "node", "label",       "string")
        _key(root, "d1", "node", "type",         "string")
        _key(root, "d2", "node", "description",  "string")
        _key(root, "d3", "node", "curated",      "boolean")
        _key(root, "d4", "node", "rejected",     "boolean")
        _key(root, "d5", "edge", "label",        "string")
        _key(root, "d6", "edge", "strength",     "double")
        _key(root, "d7", "edge", "description",  "string")
        _key(root, "d8", "edge", "curated",      "boolean")

        graph_el = ET.SubElement(root, "graph", id="G", edgedefault="directed")

        for entity in self._graph.entities.values():
            if not include_rejected and entity.metadata.annotations.get("rejected"):
                continue
            node_el = ET.SubElement(graph_el, "node", id=entity.id)
            _data(node_el, "d0", entity.name)
            _data(node_el, "d1", entity.entity_type.value)
            _data(node_el, "d2", entity.description or "")
            _data(node_el, "d3", str(entity.metadata.annotations.get("curated", False)).lower())
            _data(node_el, "d4", str(entity.metadata.annotations.get("rejected", False)).lower())

        entity_ids = set(self._graph.entities.keys())
        for rel in self._graph.relationships.values():
            if rel.source_entity_id not in entity_ids or rel.target_entity_id not in entity_ids:
                continue
            if not include_rejected and rel.metadata.annotations.get("rejected"):
                continue
            edge_el = ET.SubElement(
                graph_el, "edge",
                id=rel.id,
                source=rel.source_entity_id,
                target=rel.target_entity_id,
            )
            _data(edge_el, "d5", rel.relationship_type.value)
            _data(edge_el, "d6", str(rel.strength))
            _data(edge_el, "d7", rel.description or "")
            _data(edge_el, "d8", str(rel.metadata.annotations.get("curated", False)).lower())

        ET.indent(root, space="  ")
        xml_str = '<?xml version="1.0" encoding="UTF-8"?>\n' + ET.tostring(root, encoding="unicode")

        if output_path:
            _write_text(output_path, xml_str)

        return xml_str

    # ------------------------------------------------------------------
    # Self-contained HTML viewer
    # ------------------------------------------------------------------

    def to_html(
        self,
        output_path: Optional[str] = None,
        *,
        title: str = "Knowledge Graph",
        include_rejected: bool = False,
    ) -> str:
        """
        Generate a self-contained HTML page with an interactive vis-network
        graph viewer (loaded from CDN).

        Returns the HTML string and optionally writes it to *output_path*.
        """
        cyto = self.to_cytoscape_json(include_rejected=include_rejected)

        # Convert Cytoscape elements → vis-network nodes/edges
        vis_nodes: List[Dict[str, Any]] = []
        vis_edges: List[Dict[str, Any]] = []

        for el in cyto["elements"]:
            if el["group"] == "nodes":
                d = el["data"]
                colour = el.get("style", {}).get("background-color", _DEFAULT_COLOUR)
                vis_nodes.append({
                    "id": d["id"],
                    "label": d["label"],
                    "title": f"<b>{d['label']}</b><br>{d.get('type','')}<br>{d.get('description','')}",
                    "color": {"background": colour, "border": "#555"},
                    "font": {"size": 14},
                })
            else:
                d = el["data"]
                vis_edges.append({
                    "id": d["id"],
                    "from": d["source"],
                    "to": d["target"],
                    "label": d["label"],
                    "arrows": "to",
                    "font": {"size": 10, "align": "middle"},
                    "width": max(1, round(d.get("strength", 1.0) * 3)),
                })

        nodes_json = json.dumps(vis_nodes, indent=2)
        edges_json = json.dumps(vis_edges, indent=2)

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{title}</title>
  <script src="https://unpkg.com/vis-network@9.1.9/dist/vis-network.min.js"></script>
  <link href="https://unpkg.com/vis-network@9.1.9/dist/vis-network.min.css" rel="stylesheet">
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
             background: #1a1a2e; color: #e0e0e0; height: 100vh;
             display: flex; flex-direction: column; }}
    header {{ padding: 12px 20px; background: #16213e;
              border-bottom: 1px solid #0f3460; flex-shrink: 0; }}
    header h1 {{ font-size: 1.2rem; color: #e94560; }}
    header small {{ color: #a0a0c0; font-size: 0.8rem; }}
    #graph {{ flex: 1; }}
    #tooltip {{ position: fixed; bottom: 20px; left: 20px;
                background: #16213e; border: 1px solid #0f3460;
                border-radius: 6px; padding: 10px 14px;
                font-size: 0.85rem; max-width: 320px;
                white-space: pre-wrap; display: none; }}
  </style>
</head>
<body>
  <header>
    <h1>{title}</h1>
    <small id="stats"></small>
  </header>
  <div id="graph"></div>
  <div id="tooltip"></div>
  <script>
    const nodes = new vis.DataSet({nodes_json});
    const edges = new vis.DataSet({edges_json});
    document.getElementById("stats").textContent =
      nodes.length + " nodes · " + edges.length + " edges";

    const options = {{
      physics: {{ stabilization: {{ iterations: 200 }},
                  barnesHut: {{ gravitationalConstant: -8000, springLength: 120 }} }},
      interaction: {{ hover: true, tooltipDelay: 200 }},
    }};

    const network = new vis.Network(
      document.getElementById("graph"),
      {{ nodes, edges }},
      options
    );

    const tip = document.getElementById("tooltip");
    network.on("hoverNode", function(p) {{
      const n = nodes.get(p.node);
      if (n) {{
        tip.innerHTML = n.title || n.label;
        tip.style.display = "block";
      }}
    }});
    network.on("blurNode", () => {{ tip.style.display = "none"; }});
  </script>
</body>
</html>"""

        if output_path:
            _write_text(output_path, html)

        return html


# ------------------------------------------------------------------
# Private helpers
# ------------------------------------------------------------------


def _write_text(path: str, content: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")


def _key(parent: ET.Element, id_: str, for_: str, attr_name: str, attr_type: str) -> None:
    ET.SubElement(parent, "key", id=id_, **{"for": for_},
                  **{"attr.name": attr_name, "attr.type": attr_type})


def _data(parent: ET.Element, key: str, value: str) -> None:
    el = ET.SubElement(parent, "data", key=key)
    el.text = value
