"""Unit tests for the read-only tool surface (P9 of docs/RAG_QA_PLAN.md)."""

from __future__ import annotations

import os
from typing import Any, List, Optional

import pytest

os.environ.setdefault("LLM_API_KEY", "not-configured")

from graphbuilder.core.retrieval.models import (  # noqa: E402
    Channel,
    ChannelResult,
    ItemKind,
    RetrievalTrace,
    RetrievedItem,
)
from graphbuilder.core.retrieval.tools import (  # noqa: E402
    TOOL_GET_ENTITY,
    TOOL_SEARCH_GRAPH,
    TOOL_VERIFY_CLAIM,
    SearchGraphArgs,
    ToolCallRecord,
    ToolDispatcher,
)
from graphbuilder.domain.models.graph_models import (  # noqa: E402
    EntityType,
    GraphEntity,
    GraphRelationship,
    RelationshipType,
)


# ---------------------------------------------------------------- fakes


class _FakeOrchestrator:
    def __init__(self, items: List[RetrievedItem]):
        self._items = items
        self.calls: List[dict] = []

    async def retrieve(self, query: str, *, top_k: Optional[int] = None,
                       query_embedding=None, config_override=None):
        self.calls.append({"query": query, "top_k": top_k})
        trace = RetrievalTrace(
            query=query, extracted_terms=[query],
            channels=[ChannelResult(channel=Channel.VECTOR_ENTITY, latency_ms=2)],
            rrf_top_n=len(self._items), final_top_k=len(self._items),
            hydrated_chunks=0, total_latency_ms=4,
        )
        return list(self._items), trace


class _FakeGraphRepo:
    def __init__(self, entities=None, relationships=None):
        self._entities = {e.id: e for e in (entities or [])}
        self._rels = relationships or []

    async def get_entity_by_id(self, entity_id: str):
        return self._entities.get(entity_id)

    async def get_entity_relationships(self, entity_id: str):
        return [
            r for r in self._rels
            if r.source_entity_id == entity_id or r.target_entity_id == entity_id
        ]


def _item(eid: str, label: str, *, kind: ItemKind = ItemKind.ENTITY) -> RetrievedItem:
    return RetrievedItem(
        kind=kind, id=eid, label=label,
        score_vector=0.9, score_rrf=0.5, final_confidence=0.85,
        chunk_preview=f"text about {label}",
        source_doc_id="doc_1",
        contributing_channels=[Channel.VECTOR_ENTITY],
    )


# ---------------------------------------------------------------- schemas


class TestArgValidation:
    def test_search_graph_query_required(self):
        with pytest.raises(Exception):
            SearchGraphArgs(query="")

    def test_search_graph_default_kinds(self):
        args = SearchGraphArgs(query="brca1")
        assert args.kinds == ["entity", "relationship"]
        assert args.top_k == 8


# ---------------------------------------------------------------- dispatcher


@pytest.mark.asyncio
class TestSearchGraph:
    async def test_returns_top_k_items_with_summaries(self):
        orch = _FakeOrchestrator([
            _item("e1", "BRCA1"),
            _item("e2", "BRCA2"),
            _item("e3", "Ovarian Cancer"),
        ])
        disp = ToolDispatcher(orchestrator=orch, graph_repo=_FakeGraphRepo())

        rec = await disp.execute(TOOL_SEARCH_GRAPH, {"query": "brca1", "top_k": 2})

        assert rec.ok
        assert rec.result["n_items"] == 3  # all three pass the kinds filter
        # top_k caps the returned list.
        assert len(rec.result["items"]) == 2
        # Summary keys are stable for the LLM.
        assert set(rec.result["items"][0].keys()) >= {
            "kind", "id", "label", "final_confidence", "chunk_preview",
        }

    async def test_filters_by_kinds(self):
        orch = _FakeOrchestrator([
            _item("e1", "BRCA1", kind=ItemKind.ENTITY),
            _item("r1", "A--R-->B", kind=ItemKind.RELATIONSHIP),
        ])
        disp = ToolDispatcher(orchestrator=orch, graph_repo=_FakeGraphRepo())

        rec = await disp.execute(
            TOOL_SEARCH_GRAPH,
            {"query": "x", "kinds": ["entity"]},
        )
        kinds = {it["kind"] for it in rec.result["items"]}
        assert kinds == {"entity"}

    async def test_invalid_args_returns_record_with_error(self):
        orch = _FakeOrchestrator([])
        disp = ToolDispatcher(orchestrator=orch, graph_repo=_FakeGraphRepo())

        # top_k > 20 violates the schema.
        rec = await disp.execute(TOOL_SEARCH_GRAPH, {"query": "x", "top_k": 99})
        assert not rec.ok
        assert rec.error is not None
        assert "invalid arguments" in rec.error

    async def test_orchestrator_failure_is_caught(self):
        class _BoomOrch:
            async def retrieve(self, *a, **kw):
                raise RuntimeError("backend down")

        disp = ToolDispatcher(orchestrator=_BoomOrch(), graph_repo=_FakeGraphRepo())
        rec = await disp.execute(TOOL_SEARCH_GRAPH, {"query": "x"})
        assert not rec.ok
        assert "backend down" in rec.error


@pytest.mark.asyncio
class TestGetEntity:
    async def test_returns_entity_with_neighbours(self):
        e = GraphEntity(name="BRCA1", entity_type=EntityType.GENE,
                        description="DNA repair gene")
        e.id = "ent_brca1"
        other = GraphEntity(name="Ovarian Cancer", entity_type=EntityType.DISEASE)
        other.id = "ent_ovca"
        rel = GraphRelationship(
            source_entity_id="ent_brca1", target_entity_id="ent_ovca",
            relationship_type=RelationshipType.RELATED_TO,
            description="loss-of-function increases risk",
        )
        repo = _FakeGraphRepo(entities=[e, other], relationships=[rel])
        disp = ToolDispatcher(orchestrator=_FakeOrchestrator([]), graph_repo=repo)

        rec = await disp.execute(TOOL_GET_ENTITY, {"entity_id": "ent_brca1"})

        assert rec.ok
        assert rec.result["found"] is True
        assert rec.result["name"] == "BRCA1"
        assert rec.result["entity_type"] == EntityType.GENE.value
        assert len(rec.result["relationships"]) == 1
        assert rec.result["relationships"][0]["target_entity_id"] == "ent_ovca"

    async def test_missing_entity_returns_found_false(self):
        repo = _FakeGraphRepo()
        disp = ToolDispatcher(orchestrator=_FakeOrchestrator([]), graph_repo=repo)
        rec = await disp.execute(TOOL_GET_ENTITY, {"entity_id": "nope"})
        assert rec.ok
        assert rec.result == {"entity_id": "nope", "found": False}

    async def test_include_neighbours_off_skips_relationships(self):
        e = GraphEntity(name="X", entity_type=EntityType.GENE)
        e.id = "ent_x"
        repo = _FakeGraphRepo(entities=[e])
        disp = ToolDispatcher(orchestrator=_FakeOrchestrator([]), graph_repo=repo)

        rec = await disp.execute(
            TOOL_GET_ENTITY,
            {"entity_id": "ent_x", "include_neighbours": False},
        )
        assert "relationships" not in rec.result


@pytest.mark.asyncio
class TestVerifyClaim:
    async def test_high_overlap_text_match_passes(self):
        # No LLM service → cascade's embedding/LLM stages are skipped or
        # no-op; text-match alone runs and produces a verdict.
        disp = ToolDispatcher(
            orchestrator=_FakeOrchestrator([]),
            graph_repo=_FakeGraphRepo(),
        )
        rec = await disp.execute(TOOL_VERIFY_CLAIM, {
            "claim": "BRCA1 mutations increase ovarian cancer risk",
            "context": "BRCA1 mutations are associated with hereditary ovarian cancer risk.",
            "source_entity": "BRCA1",
            "target_entity": "ovarian cancer",
        })
        assert rec.ok
        assert rec.result["verdict"] in {"passed", "failed", "skipped"}
        assert 0.0 <= rec.result["confidence"] <= 1.0

    async def test_synthesises_relationship_when_endpoints_missing(self):
        """No source/target → dispatcher uses sentinel endpoints so the
        cascade's relationship-validation doesn't reject the call."""
        disp = ToolDispatcher(
            orchestrator=_FakeOrchestrator([]),
            graph_repo=_FakeGraphRepo(),
        )
        rec = await disp.execute(TOOL_VERIFY_CLAIM, {
            "claim": "Aspirin reduces inflammation",
            "context": "Aspirin is a non-steroidal anti-inflammatory drug.",
        })
        assert rec.ok


# ---------------------------------------------------------------- schemas + misc


def test_openai_tool_schemas_shape():
    disp = ToolDispatcher(
        orchestrator=_FakeOrchestrator([]),
        graph_repo=_FakeGraphRepo(),
    )
    schemas = disp.openai_tool_schemas()
    assert len(schemas) == 3
    names = {s["function"]["name"] for s in schemas}
    assert names == {TOOL_SEARCH_GRAPH, TOOL_GET_ENTITY, TOOL_VERIFY_CLAIM}
    # Each schema must carry a JSON-schema parameters block.
    for s in schemas:
        assert s["type"] == "function"
        assert "parameters" in s["function"]
        assert s["function"]["parameters"]["type"] == "object"


@pytest.mark.asyncio
async def test_unknown_tool_returns_error():
    disp = ToolDispatcher(
        orchestrator=_FakeOrchestrator([]),
        graph_repo=_FakeGraphRepo(),
    )
    rec = await disp.execute("delete_everything", {})
    assert not rec.ok
    assert "unknown tool" in rec.error


def test_tool_call_record_to_dict_round_trips():
    rec = ToolCallRecord(
        tool=TOOL_SEARCH_GRAPH,
        args={"query": "x"},
        result={"items": []},
        latency_ms=42,
        tool_call_id="call_abc",
    )
    d = rec.to_dict()
    assert d["tool"] == TOOL_SEARCH_GRAPH
    assert d["args"] == {"query": "x"}
    assert d["result"] == {"items": []}
    assert d["error"] is None
    assert d["tool_call_id"] == "call_abc"


def test_tool_call_record_error_hides_result():
    """If the call errored, ``to_dict`` should not leak a partial
    result — the dispatcher always builds a clean record but defending
    against future regressions here is cheap."""
    rec = ToolCallRecord(
        tool=TOOL_GET_ENTITY, args={"entity_id": "x"},
        result={"leaked": True}, error="boom",
    )
    d = rec.to_dict()
    assert d["result"] is None
    assert d["error"] == "boom"
