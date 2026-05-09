"""Tests for the cross-encoder reranker + chunk-neighbour expansion (P4
of docs/RAG_QA_PLAN.md).

Two surfaces under test:

1. :class:`CrossEncoderReranker` itself — pass-through when the model
   isn't available, ranking by raw score, min-max normalisation,
   ``top_k`` trim. Tests inject a fake ``predict`` so they don't load
   sentence-transformers from disk.

2. ``DocumentRepository.get_chunk_with_neighbours`` on the in-memory
   impl — exercises the ±radius window without needing Neo4j.

3. End-to-end orchestrator behaviour with a fake reranker — confirms
   rerank reorders the candidate list and the final-confidence math
   blends ``score_rerank`` in.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional

import pytest

os.environ.setdefault("LLM_API_KEY", "not-configured")

from graphbuilder.core.retrieval.models import (  # noqa: E402
    Channel,
    ItemKind,
    RetrievalConfig,
    RetrievedItem,
)
from graphbuilder.core.retrieval.orchestrator import (  # noqa: E402
    RetrievalOrchestrator,
    _compute_final_confidence,
)
from graphbuilder.core.retrieval.reranker import (  # noqa: E402
    CrossEncoderConfig,
    CrossEncoderReranker,
    _MODEL_CACHE,
)
from graphbuilder.domain.models.graph_models import (  # noqa: E402
    DocumentChunk,
    SourceDocument,
)
from graphbuilder.infrastructure.config.settings import GraphBuilderConfig  # noqa: E402
from graphbuilder.infrastructure.repositories.document_repository import (  # noqa: E402
    InMemoryDocumentRepository,
)


# Reuse the FakeGraphRepo / entity helpers from test_retrieval.
from tests.unit.test_retrieval import (  # noqa: E402
    FakeChunk,
    FakeDocumentRepo,
    FakeGraphRepo,
    _make_entity,
    _make_rel,
)


# ---------------------------------------------------------------- reranker


def _item(eid: str, label: str, *, vector: float = 0.5) -> RetrievedItem:
    return RetrievedItem(
        kind=ItemKind.ENTITY,
        id=eid,
        label=label,
        score_vector=vector,
        score_rrf=0.5,
        contributing_channels=[Channel.VECTOR_ENTITY],
    )


class _FakeCrossEncoder:
    """Stand-in for sentence_transformers.CrossEncoder.

    ``scores_by_label`` lets each test pin a specific score for each
    candidate so the assertion can compare order, not magnitudes.
    """

    def __init__(self, scores_by_label: dict[str, float]):
        self._scores = scores_by_label
        self.calls = 0

    def predict(self, pairs, batch_size=16, show_progress_bar=False):
        self.calls += 1
        return [self._scores.get(p[1].split(" — ")[0], 0.0) for p in pairs]


@pytest.fixture(autouse=True)
def _reset_model_cache():
    # A previous test's cached model (real or stub) must not leak across
    # tests — each rerank case sets up its own CrossEncoder via the cache.
    _MODEL_CACHE.clear()
    yield
    _MODEL_CACHE.clear()


async def test_reranker_passthrough_when_model_unavailable(monkeypatch):
    # _get_cross_encoder caches None when the load fails; we mark the
    # name as a known-failure here.
    _MODEL_CACHE["never-loads"] = None
    rr = CrossEncoderReranker(CrossEncoderConfig(model_name="never-loads"))
    items = [_item("a", "A"), _item("b", "B"), _item("c", "C")]
    out = await rr.rerank("q", items)
    assert out == items
    # No score_rerank assigned because rerank didn't run.
    assert all(i.score_rerank is None for i in out)


async def test_reranker_passthrough_with_top_k_trims_in_rrf_order():
    _MODEL_CACHE["never-loads"] = None
    rr = CrossEncoderReranker(CrossEncoderConfig(model_name="never-loads"))
    items = [_item("a", "A"), _item("b", "B"), _item("c", "C")]
    out = await rr.rerank("q", items, top_k=2)
    assert [i.id for i in out] == ["a", "b"]


async def test_reranker_reorders_by_predicted_score():
    fake = _FakeCrossEncoder({"A": 0.1, "B": 0.9, "C": 0.5})
    _MODEL_CACHE["fake-model"] = fake
    rr = CrossEncoderReranker(CrossEncoderConfig(model_name="fake-model"))

    items = [_item("a", "A"), _item("b", "B"), _item("c", "C")]
    out = await rr.rerank("q", items)
    assert [i.id for i in out] == ["b", "c", "a"]
    assert fake.calls == 1


async def test_reranker_normalises_scores_to_zero_one_range():
    fake = _FakeCrossEncoder({"A": -2.0, "B": 0.0, "C": 4.0})
    _MODEL_CACHE["fake-model"] = fake
    rr = CrossEncoderReranker(CrossEncoderConfig(model_name="fake-model"))

    items = [_item("a", "A"), _item("b", "B"), _item("c", "C")]
    out = await rr.rerank("q", items)
    # Top item gets 1.0, bottom gets 0.0 after min-max normalisation.
    by_id = {i.id: i.score_rerank for i in out}
    assert by_id["c"] == 1.0
    assert by_id["a"] == 0.0
    assert 0.0 < (by_id["b"] or 0) < 1.0


async def test_reranker_top_k_trims_after_reorder():
    fake = _FakeCrossEncoder({"A": 0.1, "B": 0.9, "C": 0.5, "D": 0.7})
    _MODEL_CACHE["fake-model"] = fake
    rr = CrossEncoderReranker(CrossEncoderConfig(model_name="fake-model"))

    items = [_item("a", "A"), _item("b", "B"), _item("c", "C"), _item("d", "D")]
    out = await rr.rerank("q", items, top_k=2)
    assert [i.id for i in out] == ["b", "d"]


async def test_reranker_predict_failure_falls_back_to_input_order():
    class _Boom:
        def predict(self, *a, **kw):
            raise RuntimeError("oom")
    _MODEL_CACHE["boom"] = _Boom()
    rr = CrossEncoderReranker(CrossEncoderConfig(model_name="boom"))

    items = [_item("a", "A"), _item("b", "B")]
    out = await rr.rerank("q", items)
    assert [i.id for i in out] == ["a", "b"]
    assert all(i.score_rerank is None for i in out)


async def test_reranker_empty_items_short_circuits():
    rr = CrossEncoderReranker(CrossEncoderConfig(model_name="any"))
    out = await rr.rerank("q", [])
    assert out == []


async def test_reranker_truncates_long_candidate_text():
    fake = _FakeCrossEncoder({})  # default 0.0
    _MODEL_CACHE["fake-model"] = fake
    rr = CrossEncoderReranker(
        CrossEncoderConfig(model_name="fake-model", max_pair_chars=20)
    )
    item = RetrievedItem(
        kind=ItemKind.ENTITY,
        id="a",
        label="ShortLabel",
        score_rrf=0.5,
        chunk_preview="x" * 1000,  # would otherwise blow past max_pair_chars
        contributing_channels=[Channel.VECTOR_ENTITY],
    )
    # We can't observe the truncated string without instrumenting predict;
    # round-trip via _candidate_text instead.
    text = rr._candidate_text(item)
    # Truncation appends a 1-char ellipsis after slicing to max_pair_chars,
    # so the cap is max_pair_chars + 1.
    assert len(text) <= 21
    assert text.endswith("…")


# ---------------------------------------------------------------- final confidence


def test_final_confidence_blends_rerank_when_present():
    item = RetrievedItem(
        kind=ItemKind.ENTITY, id="a", label="A",
        score_vector=0.6, score_rerank=0.9, score_rrf=0.5,
        contributing_channels=[Channel.VECTOR_ENTITY],
    )
    # 0.7 * 0.9 + 0.3 * 0.6 + bonus(0) = 0.63 + 0.18 = 0.81
    assert _compute_final_confidence(item) == pytest.approx(0.81, abs=0.001)


def test_final_confidence_falls_back_to_channel_max_without_rerank():
    item = RetrievedItem(
        kind=ItemKind.ENTITY, id="a", label="A",
        score_vector=0.6, score_bm25=0.4, score_rrf=0.5,
        contributing_channels=[Channel.VECTOR_ENTITY, Channel.BM25],
    )
    # max(0.6, 0.4) + 0.05*(2-1) bonus = 0.65
    assert _compute_final_confidence(item) == pytest.approx(0.65, abs=0.001)


def test_final_confidence_clipped_to_one():
    item = RetrievedItem(
        kind=ItemKind.ENTITY, id="a", label="A",
        score_vector=1.0, score_bm25=1.0, score_cypher=1.0, score_rerank=1.0,
        score_rrf=0.5,
        contributing_channels=[
            Channel.VECTOR_ENTITY, Channel.BM25, Channel.CYPHER,
        ],
    )
    assert _compute_final_confidence(item) == 1.0


# ---------------------------------------------------------------- chunk neighbours (in-memory)


def _chunk(cid: str, idx: int, content: str, doc_id: str = "doc_1") -> DocumentChunk:
    """Build a real DocumentChunk with the index/document we want.

    DocumentChunk's __post_init__ assigns an auto uuid id, so we set
    the desired id and chunk_index explicitly after construction.
    """
    c = DocumentChunk(
        document_id=doc_id,
        content=content,
        chunk_index=idx,
        start_position=idx * 100,
        end_position=idx * 100 + len(content),
        token_count=len(content.split()),
        character_count=len(content),
    )
    c.id = cid
    return c


@pytest.fixture
def repo_with_chunks() -> InMemoryDocumentRepository:
    repo = InMemoryDocumentRepository(GraphBuilderConfig())
    # Five chunks in one document, ordered by chunk_index.
    repo.chunks["doc_1"] = [
        _chunk("c0", 0, "first"),
        _chunk("c1", 1, "second"),
        _chunk("c2", 2, "third"),
        _chunk("c3", 3, "fourth"),
        _chunk("c4", 4, "fifth"),
    ]
    return repo


async def test_neighbours_radius_one_returns_target_plus_neighbours(repo_with_chunks):
    chunks = await repo_with_chunks.get_chunk_with_neighbours("c2", radius=1)
    assert [c.id for c in chunks] == ["c1", "c2", "c3"]


async def test_neighbours_radius_two(repo_with_chunks):
    chunks = await repo_with_chunks.get_chunk_with_neighbours("c2", radius=2)
    assert [c.id for c in chunks] == ["c0", "c1", "c2", "c3", "c4"]


async def test_neighbours_radius_zero_returns_only_target(repo_with_chunks):
    chunks = await repo_with_chunks.get_chunk_with_neighbours("c2", radius=0)
    assert [c.id for c in chunks] == ["c2"]


async def test_neighbours_truncated_at_document_start(repo_with_chunks):
    # c0 is the first chunk; ±1 window should clip to {c0, c1}.
    chunks = await repo_with_chunks.get_chunk_with_neighbours("c0", radius=1)
    assert [c.id for c in chunks] == ["c0", "c1"]


async def test_neighbours_truncated_at_document_end(repo_with_chunks):
    chunks = await repo_with_chunks.get_chunk_with_neighbours("c4", radius=1)
    assert [c.id for c in chunks] == ["c3", "c4"]


async def test_neighbours_unknown_chunk_returns_empty(repo_with_chunks):
    assert await repo_with_chunks.get_chunk_with_neighbours("missing") == []


async def test_neighbours_radius_clamped_to_five(repo_with_chunks):
    """A caller asking for radius=999 should get the same window as
    radius=5 — the bound prevents pathological pulls of entire documents."""
    chunks = await repo_with_chunks.get_chunk_with_neighbours("c2", radius=999)
    # We only have 5 chunks total, so result is the whole document.
    assert [c.id for c in chunks] == ["c0", "c1", "c2", "c3", "c4"]


# ---------------------------------------------------------------- orchestrator with rerank


async def test_orchestrator_rerank_reorders_final_items(monkeypatch):
    """A real rerank that ranks Imatinib higher than TP53 should make
    the orchestrator return Imatinib first even when RRF placed TP53
    above it (via a stronger BM25 hit, say)."""
    from graphbuilder.infrastructure.services import embedding_factory

    async def fake_embed_async(text):
        return [0.1] * 4
    monkeypatch.setattr(embedding_factory, "embed_async", fake_embed_async)

    imatinib = _make_entity("e1", "Imatinib", chunk_ids=["c1"])
    tp53 = _make_entity("e2", "TP53", chunk_ids=["c2"])

    # BM25 returns TP53 first; vector returns Imatinib first.
    repo = FakeGraphRepo(
        vector_entity_hits=[(imatinib, 0.85), (tp53, 0.6)],
        text_search_hits=[tp53, imatinib],   # BM25 ranks TP53 higher
    )
    docs = FakeDocumentRepo([
        FakeChunk("c1", "Imatinib content", chunk_index=0),
        FakeChunk("c2", "TP53 content", chunk_index=0, document_id="doc_2"),
    ])

    fake = _FakeCrossEncoder({"Imatinib": 0.95, "TP53": 0.10})
    _MODEL_CACHE["fake-rerank"] = fake
    cfg = RetrievalConfig(
        cross_encoder_model="fake-rerank",
        chunk_neighbour_radius=0,  # keep this test focused on rerank
    )
    orch = RetrievalOrchestrator(graph_repo=repo, document_repo=docs, config=cfg)

    items, _ = await orch.retrieve("does Imatinib do anything?")
    assert items[0].id == "e1"
    assert items[0].score_rerank is not None
    # Rerank score is normalised so the top item gets 1.0.
    assert items[0].score_rerank == 1.0


async def test_orchestrator_rerank_failure_does_not_break_pipeline(monkeypatch):
    """If the cross-encoder model can't load, items still come back —
    just in RRF order. This is the production fallback path."""
    from graphbuilder.infrastructure.services import embedding_factory

    async def fake_embed_async(text):
        return [0.1] * 4
    monkeypatch.setattr(embedding_factory, "embed_async", fake_embed_async)

    e1 = _make_entity("e1", "Imatinib")
    repo = FakeGraphRepo(
        vector_entity_hits=[(e1, 0.9)], text_search_hits=[e1],
    )
    _MODEL_CACHE["never-loads"] = None  # cached failure
    cfg = RetrievalConfig(
        cross_encoder_model="never-loads",
        chunk_neighbour_radius=0,
    )
    orch = RetrievalOrchestrator(graph_repo=repo, config=cfg)
    items, _ = await orch.retrieve("Imatinib")
    assert items
    assert all(i.score_rerank is None for i in items)


async def test_orchestrator_neighbour_expansion_joins_chunks(monkeypatch):
    """With radius=1 and three sequential chunks in one doc, the cited
    chunk's preview should include text from its neighbours."""
    from graphbuilder.infrastructure.services import embedding_factory

    async def fake_embed_async(text):
        return [0.1] * 4
    monkeypatch.setattr(embedding_factory, "embed_async", fake_embed_async)

    e1 = _make_entity("e1", "Imatinib", chunk_ids=["c1"])
    repo = FakeGraphRepo(
        vector_entity_hits=[(e1, 0.9)],
        text_search_hits=[e1],
    )
    docs = FakeDocumentRepo([
        FakeChunk("c0", "BEFORE-TEXT", chunk_index=0),
        FakeChunk("c1", "MAIN-TEXT", chunk_index=1),
        FakeChunk("c2", "AFTER-TEXT", chunk_index=2),
    ])

    _MODEL_CACHE["pass"] = None  # disable rerank
    cfg = RetrievalConfig(
        chunk_neighbour_radius=1,
        cross_encoder_model="pass",
    )
    orch = RetrievalOrchestrator(graph_repo=repo, document_repo=docs, config=cfg)
    items, _ = await orch.retrieve("Imatinib")
    e1_item = next(i for i in items if i.id == "e1")
    assert e1_item.chunk_preview is not None
    # Neighbour text from c0 and c2 should be joined into the preview.
    assert "BEFORE-TEXT" in e1_item.chunk_preview
    assert "MAIN-TEXT" in e1_item.chunk_preview
    assert "AFTER-TEXT" in e1_item.chunk_preview
    # The anchor stays c1 even after the join.
    assert e1_item.source_chunk_id == "c1"


async def test_orchestrator_radius_zero_uses_legacy_single_chunk_path(monkeypatch):
    """radius=0 keeps the original behaviour — single chunk, no join."""
    from graphbuilder.infrastructure.services import embedding_factory

    async def fake_embed_async(text):
        return [0.1] * 4
    monkeypatch.setattr(embedding_factory, "embed_async", fake_embed_async)

    e1 = _make_entity("e1", "Imatinib", chunk_ids=["c1"])
    repo = FakeGraphRepo(
        vector_entity_hits=[(e1, 0.9)],
        text_search_hits=[e1],
    )
    docs = FakeDocumentRepo([
        FakeChunk("c0", "BEFORE-TEXT", chunk_index=0),
        FakeChunk("c1", "MAIN-TEXT", chunk_index=1),
        FakeChunk("c2", "AFTER-TEXT", chunk_index=2),
    ])

    _MODEL_CACHE["pass"] = None
    cfg = RetrievalConfig(
        chunk_neighbour_radius=0,
        cross_encoder_model="pass",
    )
    orch = RetrievalOrchestrator(graph_repo=repo, document_repo=docs, config=cfg)
    items, _ = await orch.retrieve("Imatinib")
    e1_item = next(i for i in items if i.id == "e1")
    assert e1_item.chunk_preview == "MAIN-TEXT"
    assert "BEFORE-TEXT" not in (e1_item.chunk_preview or "")
