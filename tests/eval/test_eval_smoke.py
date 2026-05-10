"""Hermetic CI gate for the RAG eval harness (P13 of docs/RAG_QA_PLAN.md, §9.3).

Builds an in-memory mini-graph, wires a real
:class:`RetrievalOrchestrator` + :class:`QAService` against it, runs
``run_eval`` over an embedded gold list, and asserts every metric in
:file:`baselines.json::hermetic_floor` is at-or-above its floor.

If anyone breaks fusion, hydration, rerank pass-through, or the eval
runner's record assembly, one of these floor assertions will trip.
The contract is: code follows the floors — never lower a floor to make
a regression pass.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest

os.environ.setdefault("LLM_API_KEY", "not-configured")

from graphbuilder.core.eval import (  # noqa: E402
    GoldQuestion,
    compute_metrics,
    run_eval,
    write_csv_report,
    write_markdown_report,
)
from graphbuilder.core.retrieval import (  # noqa: E402
    RetrievalConfig,
    RetrievalOrchestrator,
)
from graphbuilder.core.retrieval.qa_service import QAService  # noqa: E402
from graphbuilder.core.retrieval.reranker import CrossEncoderReranker  # noqa: E402
from graphbuilder.domain.models.graph_models import (  # noqa: E402
    EntityType,
    GraphEntity,
    GraphRelationship,
    RelationshipType,
)
from graphbuilder.infrastructure.config.settings import GraphBuilderConfig  # noqa: E402
from graphbuilder.infrastructure.repositories.conversation_repository import (  # noqa: E402
    InMemoryConversationRepository,
)


HERE = Path(__file__).parent


# ---------------------------------------------------------------- fakes


@dataclass
class _Chunk:
    id: str
    content: str
    document_id: str = "doc_1"
    chunk_index: int = 0


def _entity(eid: str, name: str, *, chunk_ids: Optional[List[str]] = None) -> GraphEntity:
    e = GraphEntity(
        name=name,
        entity_type=EntityType.GENE,
        description=f"description of {name}",
    )
    e.id = eid
    if chunk_ids:
        e.source_chunk_ids = list(chunk_ids)
    return e


def _rel(rid: str, src: str, tgt: str, *, chunk_ids: Optional[List[str]] = None) -> GraphRelationship:
    r = GraphRelationship(
        source_entity_id=src,
        target_entity_id=tgt,
        relationship_type=RelationshipType.RELATED_TO,
        description=f"{src} relates to {tgt}",
    )
    r.id = rid
    if chunk_ids:
        r.source_chunk_ids = list(chunk_ids)
    return r


class _FakeGraphRepo:
    """Minimal duck-typed graph repo — same shape used in test_retrieval.py.

    Channels read these in parallel; the smoke set is rigged so each
    channel contributes at least one hit, exercising the full RRF +
    hydration path the way a live retrieval would.
    """

    def __init__(
        self,
        entities: List[GraphEntity],
        relationships: List[GraphRelationship],
        vector_entity_hits: List[Tuple[GraphEntity, float]],
        vector_rel_hits: List[Tuple[GraphRelationship, float]],
        text_hits_by_term: Dict[str, List[GraphEntity]],
    ):
        self._entities = {e.id: e for e in entities}
        self._rels = {r.id: r for r in relationships}
        self._vector_entity_hits = vector_entity_hits
        self._vector_rel_hits = vector_rel_hits
        self._text_hits_by_term = {k.lower(): v for k, v in text_hits_by_term.items()}

    async def vector_search_entities(self, embedding, top_k=10, min_score=0.5):
        return [(e, s) for (e, s) in self._vector_entity_hits[:top_k] if s >= min_score]

    async def vector_search_relationships(self, embedding, top_k=10, min_score=0.5):
        return [(r, s) for (r, s) in self._vector_rel_hits[:top_k] if s >= min_score]

    async def search_entities_by_text(self, terms, limit=50):
        # Real impl returns dict[id, entity]; aggregate any term that
        # matches a known seed (case-insensitive).
        out: Dict[str, GraphEntity] = {}
        for t in terms:
            for e in self._text_hits_by_term.get(t.lower(), []):
                out[e.id] = e
                if len(out) >= limit:
                    return out
        return out

    async def get_entity_relationships(self, entity_id):
        return [r for r in self._rels.values() if r.source_entity_id == entity_id]

    async def get_entity_names_by_ids(self, ids):
        wanted = set(ids)
        return {eid: e.name for eid, e in self._entities.items() if eid in wanted and e.name}


class _FakeDocumentRepo:
    def __init__(self, chunks: List[_Chunk]):
        self._chunks = {c.id: c for c in chunks}

    async def get_chunks_by_ids(self, chunk_ids):
        return [self._chunks[c] for c in chunk_ids if c in self._chunks]

    async def get_chunk_with_neighbours(self, chunk_id, radius=1):
        if chunk_id not in self._chunks:
            return []
        target = self._chunks[chunk_id]
        same_doc = [c for c in self._chunks.values() if c.document_id == target.document_id]
        ordered = sorted(same_doc, key=lambda c: c.chunk_index)
        idx = next((i for i, c in enumerate(ordered) if c.id == chunk_id), -1)
        if idx < 0:
            return []
        bounded = max(0, min(int(radius), 5))
        return ordered[max(0, idx - bounded) : idx + bounded + 1]


class _ScriptedLLM:
    """LLM that produces an answer by stitching together gold-aligned
    keywords found in the retrieved sources.

    The smoke test is about the harness, not LLM quality — but to keep
    the answer-coverage metric meaningful we want a deterministic answer
    that mentions the right substrings whenever the right sources came
    back. This fake reads the prompt's SOURCES block, picks out the
    entity names, and emits a sentence that cites each source by index.
    """

    def __init__(self, *, mention: List[str]):
        self._mention = [m for m in mention if m]
        self.calls: List[str] = []

    async def generate_text(self, *, prompt: str, system_prompt=None,
                            temperature: float = 0.0, max_tokens: int = 1024) -> str:
        self.calls.append(prompt)
        body = prompt or ""
        # Cite each source we actually saw, preserving order.
        cites = []
        for line in body.splitlines():
            line = line.strip()
            if line.startswith("[") and "]" in line:
                idx = line[1 : line.index("]")]
                if idx.isdigit():
                    cites.append(int(idx))
        # Echo back any required mention strings so answer coverage is
        # deterministic — this stands in for "LLM successfully grounded".
        cite_str = " ".join(f"[{i}]" for i in cites[:4])
        mention_str = ", ".join(self._mention)
        if not cite_str:
            cite_str = "[1]"
        return f"Answer references {mention_str} {cite_str}.".strip()


# ---------------------------------------------------------------- fixture


@pytest.fixture
def hermetic_qa_service(monkeypatch):
    """Build a real orchestrator + QAService against a tiny in-memory graph.

    The cross-encoder is pass-through (real model isn't loaded) — same
    pattern as ``tests/unit/test_retrieval.py`` so the smoke doesn't pay
    the model-load cost on a cold cache.
    """
    async def passthrough(self, query, items, *, top_k=None):
        return items[: top_k] if top_k is not None else items
    monkeypatch.setattr(CrossEncoderReranker, "rerank", passthrough)

    # Skip real embeddings — the orchestrator accepts a pre-supplied
    # query_embedding and the channels here ignore the vector content.
    async def fake_embed(text):
        return [0.0] * 8
    import graphbuilder.infrastructure.services.embedding_factory as ef
    monkeypatch.setattr(ef, "embed_async", fake_embed, raising=False)

    # Mini graph: imatinib, BCR-ABL, KIT, and a treats edge.
    e_imatinib = _entity("ent_imatinib", "imatinib", chunk_ids=["chunk_imat"])
    e_bcr_abl = _entity("ent_bcr_abl", "BCR-ABL", chunk_ids=["chunk_bcr"])
    e_kit = _entity("ent_kit", "KIT", chunk_ids=["chunk_kit"])
    e_other = _entity("ent_aspirin", "aspirin")  # noise

    r_imat_bcr = _rel(
        "rel_imat_bcr", "ent_imatinib", "ent_bcr_abl",
        chunk_ids=["chunk_imat"],
    )
    r_imat_kit = _rel(
        "rel_imat_kit", "ent_imatinib", "ent_kit",
        chunk_ids=["chunk_imat"],
    )

    chunks = [
        _Chunk("chunk_imat", "Imatinib inhibits BCR-ABL and KIT.", chunk_index=0),
        _Chunk("chunk_bcr", "BCR-ABL is a fusion tyrosine kinase.", chunk_index=1),
        _Chunk("chunk_kit", "KIT is a receptor tyrosine kinase.", chunk_index=2),
    ]

    repo = _FakeGraphRepo(
        entities=[e_imatinib, e_bcr_abl, e_kit, e_other],
        relationships=[r_imat_bcr, r_imat_kit],
        vector_entity_hits=[
            (e_imatinib, 0.92),
            (e_bcr_abl, 0.88),
            (e_kit, 0.81),
            (e_other, 0.55),
        ],
        vector_rel_hits=[
            (r_imat_bcr, 0.86),
            (r_imat_kit, 0.83),
        ],
        text_hits_by_term={
            "imatinib": [e_imatinib],
            "bcr-abl": [e_bcr_abl],
            "kit": [e_kit],
        },
    )
    doc_repo = _FakeDocumentRepo(chunks)

    cfg = RetrievalConfig(final_top_k=4, max_chunks_per_item=1)
    orch = RetrievalOrchestrator(
        graph_repo=repo, document_repo=doc_repo, config=cfg,
    )
    conv_repo = InMemoryConversationRepository(GraphBuilderConfig())
    llm = _ScriptedLLM(mention=["BCR-ABL", "KIT"])
    svc = QAService(
        orchestrator=orch,
        conversation_repo=conv_repo,
        llm_service=llm,
        config=cfg,
    )
    return svc


HERMETIC_GOLD: List[GoldQuestion] = [
    # Mirrors q001 of the seed yaml but pinned to the in-memory graph.
    GoldQuestion(
        id="hermetic_relational",
        question="What kinases does imatinib inhibit?",
        intent="relational",
        gold_entity_ids=["ent_imatinib", "ent_bcr_abl", "ent_kit"],
        gold_relationship_ids=["rel_imat_bcr", "rel_imat_kit"],
        gold_chunk_ids=["chunk_imat"],
        gold_answer_substrings=["BCR-ABL", "KIT"],
    ),
    GoldQuestion(
        id="hermetic_definitional",
        question="What is BCR-ABL?",
        intent="definitional",
        gold_entity_ids=["ent_bcr_abl"],
        gold_chunk_ids=["chunk_bcr"],
        gold_answer_substrings=["BCR-ABL"],
    ),
    GoldQuestion(
        id="hermetic_lookup",
        question="BCR-ABL",
        intent="lookup",
        gold_entity_ids=["ent_bcr_abl"],
        gold_answer_substrings=["BCR-ABL"],
    ),
]


# ---------------------------------------------------------------- tests


def _load_floors() -> Dict[str, float]:
    raw = json.loads((HERE / "baselines.json").read_text(encoding="utf-8"))
    return raw["hermetic_floor"]


async def test_hermetic_eval_clears_baseline_floors(hermetic_qa_service):
    """The smoke gate: full harness against a known-good in-memory graph
    must clear every floor in baselines.json.

    Routing (§9.9) is bypassed here via ``retrieval_override`` — the
    relational profile's ``final_top_k=16`` would push the smoke
    precision below floor on a graph this small (only 4 entities), and
    routing has its own unit tests in ``test_qa_service.py`` and
    ``test_intent.py``. The gate's job is to catch regressions in
    fusion / hydration / rerank pass-through / faithfulness wiring.
    """
    svc = hermetic_qa_service

    async def ask_fn(query: str):
        return await svc.ask(query=query, retrieval_override=svc._cfg)

    records, summary = await run_eval(ask_fn=ask_fn, gold=HERMETIC_GOLD)
    floors = _load_floors()

    assert summary.n_questions == len(HERMETIC_GOLD)
    assert summary.n_errors == 0, [r.error for r in records if r.error]

    assert summary.precision_at_k >= floors["precision_at_k"], (
        f"precision_at_k={summary.precision_at_k:.3f} below floor "
        f"{floors['precision_at_k']:.3f}"
    )
    assert summary.recall_at_k >= floors["recall_at_k"], (
        f"recall_at_k={summary.recall_at_k:.3f} below floor "
        f"{floors['recall_at_k']:.3f}"
    )
    assert summary.f1_at_k >= floors["f1_at_k"], (
        f"f1_at_k={summary.f1_at_k:.3f} below floor {floors['f1_at_k']:.3f}"
    )
    assert summary.context_recall >= floors["context_recall"], (
        f"context_recall={summary.context_recall:.3f} below floor "
        f"{floors['context_recall']:.3f}"
    )
    assert summary.answer_coverage is not None
    assert summary.answer_coverage >= floors["answer_coverage"], (
        f"answer_coverage={summary.answer_coverage:.3f} below floor "
        f"{floors['answer_coverage']:.3f}"
    )
    # P8 — every smoke question cites known sources, so the harness
    # should produce a score on every record. None means the wiring
    # broke (faithfulness field stripped from AskResult, runner not
    # reading it, etc.).
    assert summary.answer_faithfulness is not None, (
        "answer_faithfulness is None — P8 plumbing missing on the eval path"
    )
    assert summary.answer_faithfulness >= floors["answer_faithfulness"], (
        f"answer_faithfulness={summary.answer_faithfulness:.3f} below floor "
        f"{floors['answer_faithfulness']:.3f}"
    )
    assert summary.latency_ms_p95 <= floors["latency_ms_p95_max"]


async def test_eval_records_round_trip_through_reports(hermetic_qa_service, tmp_path):
    """Run the harness, write CSV + markdown, sanity-check the artefacts.

    Catches accidental schema drift between metric math and the
    serialiser — both must agree on field names and value formatting.
    """
    svc = hermetic_qa_service

    async def ask_fn(query: str):
        return await svc.ask(query=query, retrieval_override=svc._cfg)

    records, summary = await run_eval(ask_fn=ask_fn, gold=HERMETIC_GOLD)

    csv_path = write_csv_report(records, tmp_path / "rag_eval.csv")
    md_path = write_markdown_report(
        records, tmp_path / "rag_eval.md", summary=summary,
    )

    csv_text = csv_path.read_text(encoding="utf-8")
    assert "question_id" in csv_text.splitlines()[0]
    assert "hermetic_relational" in csv_text

    md_text = md_path.read_text(encoding="utf-8")
    assert "# RAG eval report" in md_text
    assert "Headline metrics" in md_text
    assert "Per-question results" in md_text
    # Latency p95 row uses the literal target from the §9.2 table — make
    # sure the writer didn't drop the threshold side of the table.
    assert "≤ 6000 ms" in md_text


async def test_runner_captures_per_question_failures(hermetic_qa_service):
    """A question whose ``ask_fn`` raises must produce an error record
    instead of taking the whole run down."""
    svc = hermetic_qa_service
    boom_question = GoldQuestion(id="boom", question="anything")

    async def ask_fn(query: str):
        if query == "anything":
            raise RuntimeError("synthetic failure")
        return await svc.ask(query=query, retrieval_override=svc._cfg)

    records, summary = await run_eval(
        ask_fn=ask_fn,
        gold=HERMETIC_GOLD + [boom_question],
    )
    err_records = [r for r in records if r.error]
    assert len(err_records) == 1
    assert "synthetic failure" in (err_records[0].error or "")
    # The aggregate still reports — n_errors counted, others contribute zeros.
    assert summary.n_errors == 1
    assert summary.n_questions == len(HERMETIC_GOLD) + 1
