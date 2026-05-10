"""Tests for the retrieval package (P3 of docs/RAG_QA_PLAN.md).

Covers term extraction, RRF fusion, the three channels, and the
orchestrator end-to-end. Uses a minimal duck-typed FakeGraphRepo to
avoid pulling in Neo4j or sentence-transformers — the contract under
test is the orchestrator's logic, not the persistence layer.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import pytest

os.environ.setdefault("LLM_API_KEY", "not-configured")

from graphbuilder.core.retrieval import (  # noqa: E402
    Channel,
    ChannelResult,
    RawHit,
    RetrievalConfig,
    RetrievalOrchestrator,
    RetrievedItem,
    reciprocal_rank_fusion,
)
from graphbuilder.core.retrieval.channels import (  # noqa: E402
    Bm25Channel,
    CypherChannel,
    VectorChannel,
)
from graphbuilder.core.retrieval.models import ItemKind  # noqa: E402
from graphbuilder.core.retrieval.reranker import CrossEncoderReranker  # noqa: E402
from graphbuilder.core.retrieval.term_extraction import extract_terms  # noqa: E402


@pytest.fixture(autouse=True)
def _stub_cross_encoder(monkeypatch):
    """No-op the cross-encoder for the whole module.

    Existing channel/RRF/orchestrator tests don't care about rerank
    quality and shouldn't pay the model-load cost (or hit the network
    on a cold cache). Tests that *do* exercise rerank construct their
    own ``CrossEncoderReranker`` and inject it via the orchestrator
    constructor — bypassing this stub.
    """
    async def passthrough(self, query, items, *, top_k=None):
        return items[: top_k] if top_k is not None else items
    monkeypatch.setattr(CrossEncoderReranker, "rerank", passthrough)
from graphbuilder.domain.models.graph_models import (  # noqa: E402
    EntityType,
    GraphEntity,
    GraphRelationship,
    RelationshipType,
)


# ---------------------------------------------------------------- fakes


@dataclass
class FakeChunk:
    id: str
    content: str
    document_id: str = "doc_1"
    chunk_index: int = 0


class FakeGraphRepo:
    """Just enough of GraphRepositoryInterface to drive the orchestrator."""

    def __init__(
        self,
        entities: Optional[List[GraphEntity]] = None,
        relationships: Optional[List[GraphRelationship]] = None,
        vector_entity_hits: Optional[List[Tuple[GraphEntity, float]]] = None,
        vector_rel_hits: Optional[List[Tuple[GraphRelationship, float]]] = None,
        text_search_hits: Optional[List[GraphEntity]] = None,
        relationships_by_entity: Optional[Dict[str, List[GraphRelationship]]] = None,
    ):
        self._entities = {e.id: e for e in (entities or [])}
        self._rels = {r.id: r for r in (relationships or [])}
        self._vector_entity_hits = vector_entity_hits or []
        self._vector_rel_hits = vector_rel_hits or []
        self._text_search_hits = text_search_hits or []
        self._rels_by_entity = relationships_by_entity or {}

    async def vector_search_entities(self, embedding, top_k=10, min_score=0.5):
        return [
            (e, s) for (e, s) in self._vector_entity_hits[:top_k]
            if s >= min_score
        ]

    async def vector_search_relationships(self, embedding, top_k=10, min_score=0.5):
        return [
            (r, s) for (r, s) in self._vector_rel_hits[:top_k]
            if s >= min_score
        ]

    async def search_entities_by_text(self, terms, limit=50):
        # Real impl returns dict {id: GraphEntity}; we mirror that.
        return {e.id: e for e in self._text_search_hits[:limit]}

    async def get_entity_relationships(self, entity_id):
        return list(self._rels_by_entity.get(entity_id, []))

    async def get_entity_names_by_ids(self, ids):
        # Look across both the explicit `entities` map and any anchor
        # entity that only appeared in vector/text-search hit lists, so
        # the fix-#3 label-resolution path has data to work with.
        all_seen = dict(self._entities)
        for e, _ in self._vector_entity_hits:
            all_seen.setdefault(e.id, e)
        for e in self._text_search_hits:
            all_seen.setdefault(e.id, e)
        wanted = set(ids)
        return {eid: e.name for eid, e in all_seen.items() if eid in wanted and e.name}


class FakeDocumentRepo:
    def __init__(self, chunks: Optional[List[FakeChunk]] = None):
        self._chunks = {c.id: c for c in (chunks or [])}

    async def get_chunks_by_ids(self, chunk_ids):
        return [self._chunks[cid] for cid in chunk_ids if cid in self._chunks]

    async def get_chunk_with_neighbours(self, chunk_id, radius=1):
        # Match the in-memory implementation: locate the target's
        # document and slice ±radius around it by chunk_index.
        if chunk_id not in self._chunks:
            return []
        target = self._chunks[chunk_id]
        same_doc = [c for c in self._chunks.values() if c.document_id == target.document_id]
        ordered = sorted(same_doc, key=lambda c: getattr(c, "chunk_index", 0))
        idx = next((i for i, c in enumerate(ordered) if c.id == chunk_id), -1)
        if idx < 0:
            return []
        bounded = max(0, min(int(radius), 5))
        return ordered[max(0, idx - bounded) : idx + bounded + 1]


def _make_entity(
    eid: str,
    name: str,
    *,
    chunk_ids: Optional[List[str]] = None,
    entity_type: EntityType = EntityType.GENE,
) -> GraphEntity:
    # GraphEntity is a @dataclass subclass of DomainEntity; the id is
    # set in DomainEntity.__init__ via __post_init__. Assign post-construct.
    e = GraphEntity(
        name=name,
        entity_type=entity_type,
        description=f"description of {name}",
    )
    e.id = eid
    if chunk_ids:
        e.source_chunk_ids = list(chunk_ids)
    return e


def _make_rel(rid: str, src: str, tgt: str, *, chunk_ids: Optional[List[str]] = None) -> GraphRelationship:
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


# ---------------------------------------------------------------- term extraction


def test_extract_terms_keeps_quoted_phrases_verbatim():
    terms = extract_terms('what does "TNF-alpha" do in "Crohn disease"?')
    assert "TNF-alpha" in terms
    assert "Crohn disease" in terms


def test_extract_terms_drops_stopwords():
    terms = extract_terms("what is the role of imatinib?")
    # "what", "is", "the", "of" are stopwords. "imatinib" is long enough.
    lower_terms = [t.lower() for t in terms]
    for stop in ("what", "is", "the", "of"):
        assert stop not in lower_terms
    assert "imatinib" in lower_terms


def test_extract_terms_preserves_gene_symbols_and_identifiers():
    terms = extract_terms("does BRCA1 interact with TP53 and CHEMBL12345?")
    assert "BRCA1" in terms
    assert "TP53" in terms
    assert "CHEMBL12345" in terms


def test_extract_terms_capitalised_proper_nouns_after_first_word():
    # "Aspirin" first word — the capitalisation heuristic only kicks in
    # for non-first positions; but length>=4 catches it anyway.
    terms = extract_terms("Aspirin treats headache and migraine.")
    lower = [t.lower() for t in terms]
    assert "aspirin" in lower
    assert "headache" in lower
    assert "migraine" in lower


def test_extract_terms_caps_at_max():
    query = " ".join(f"GENE{i}" for i in range(50))
    terms = extract_terms(query, max_terms=5)
    assert len(terms) == 5


def test_extract_terms_empty_query():
    assert extract_terms("") == []
    assert extract_terms("   ") == []


def test_extract_terms_dedupes():
    terms = extract_terms("imatinib imatinib IMATINIB")
    # "imatinib" lowercase + "IMATINIB" uppercase are technically different
    # tokens; we dedup exact matches only. So we expect at most 2 entries.
    assert len(terms) <= 2


# ---------------------------------------------------------------- RRF


def test_rrf_single_channel_preserves_order():
    fused = reciprocal_rank_fusion(
        [("vector", ["a", "b", "c"])], k=60
    )
    assert [item for item, _ in fused] == ["a", "b", "c"]


def test_rrf_combines_channels_and_boosts_overlap():
    """An item that appears in both channels at rank 1 should beat an
    item that only appears in one channel at rank 1."""
    fused = reciprocal_rank_fusion(
        [
            ("vector", ["a", "b", "c"]),
            ("bm25",   ["a", "x", "y"]),
        ],
        k=60,
    )
    fused_dict = dict(fused)
    # 'a' appears in both → 2 * (1/61) ≈ 0.0328
    # 'b' only in vector at rank 2 → 1/62 ≈ 0.0161
    assert fused_dict["a"] > fused_dict["b"]
    assert fused_dict["a"] > fused_dict["x"]


def test_rrf_top_n_caps_output():
    fused = reciprocal_rank_fusion(
        [("vec", [f"d{i}" for i in range(20)])], top_n=5
    )
    assert len(fused) == 5
    assert fused[0][0] == "d0"


def test_rrf_empty_input_returns_empty():
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([("v", [])]) == []


def test_rrf_invalid_k_raises():
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([("v", ["a"])], k=0)


def test_rrf_score_uses_standard_formula():
    """rrf(d) = sum(1 / (k + rank)). Single channel, rank 1, k=60 → 1/61."""
    fused = reciprocal_rank_fusion([("v", ["a"])], k=60)
    assert fused[0][0] == "a"
    assert fused[0][1] == pytest.approx(1.0 / 61, rel=1e-6)


def test_rrf_stable_for_ties():
    """Same fused score → first-seen channel order wins (stable sort)."""
    fused = reciprocal_rank_fusion(
        [
            ("c1", ["a"]),  # a gets 1/61
            ("c2", ["b"]),  # b gets 1/61
        ],
        k=60,
    )
    # With both items at the same score, the one seen first ('a') wins.
    assert fused[0][0] == "a"
    assert fused[1][0] == "b"


# ---------------------------------------------------------------- channels


async def test_vector_channel_emits_entity_and_relationship_results():
    e1 = _make_entity("e1", "Imatinib")
    e2 = _make_entity("e2", "BCR-ABL")
    r1 = _make_rel("r1", "e1", "e2")
    repo = FakeGraphRepo(
        vector_entity_hits=[(e1, 0.9), (e2, 0.8)],
        vector_rel_hits=[(r1, 0.85)],
    )
    cfg = RetrievalConfig()
    channel = VectorChannel(repo, cfg)
    results = await channel.run("imatinib targets BCR-ABL", [0.1] * 4, [])
    assert [r.channel for r in results] == [
        Channel.VECTOR_ENTITY, Channel.VECTOR_RELATIONSHIP
    ]
    ent_result = results[0]
    rel_result = results[1]
    assert ent_result.hit_count == 2
    assert rel_result.hit_count == 1
    assert ent_result.hits[0].label == "Imatinib"
    assert rel_result.hits[0].kind == ItemKind.RELATIONSHIP


async def test_vector_channel_handles_missing_embedding():
    repo = FakeGraphRepo()
    cfg = RetrievalConfig()
    channel = VectorChannel(repo, cfg)
    results = await channel.run("anything", None, [])
    assert all(r.error == "no query embedding (embedding model unavailable)" for r in results)
    assert all(r.hit_count == 0 for r in results)


async def test_vector_channel_disabled_short_circuits():
    repo = FakeGraphRepo(vector_entity_hits=[(_make_entity("e1", "A"), 0.9)])
    cfg = RetrievalConfig(enable_vector_channel=False)
    channel = VectorChannel(repo, cfg)
    assert await channel.run("q", [0.1], []) == []


async def test_bm25_channel_uses_terms_and_returns_synthetic_score():
    e1 = _make_entity("e1", "BRCA1")
    e2 = _make_entity("e2", "TP53")
    repo = FakeGraphRepo(text_search_hits=[e1, e2])
    cfg = RetrievalConfig()
    channel = Bm25Channel(repo, cfg)
    [result] = await channel.run("brca1 and tp53", None, ["BRCA1", "TP53"])
    assert result.hit_count == 2
    # First hit gets score 1.0, second gets ≤ 1.0 (rank-based decay).
    assert result.hits[0].score == 1.0
    assert result.hits[1].score < result.hits[0].score


async def test_bm25_channel_empty_terms_records_error():
    repo = FakeGraphRepo()
    cfg = RetrievalConfig()
    channel = Bm25Channel(repo, cfg)
    [result] = await channel.run("?", None, [])
    assert result.error and "no candidate terms" in result.error
    assert result.hit_count == 0


async def test_cypher_channel_emits_anchor_and_neighbours():
    anchor = _make_entity("e1", "Imatinib")
    neighbour = _make_entity("e2", "BCR-ABL")
    edge = _make_rel("r1", "e1", "e2")
    repo = FakeGraphRepo(
        text_search_hits=[anchor],
        relationships_by_entity={"e1": [edge]},
        entities=[anchor, neighbour],
        relationships=[edge],
    )
    cfg = RetrievalConfig()
    channel = CypherChannel(repo, cfg)
    [result] = await channel.run("imatinib", None, ["imatinib"])
    kinds = {h.kind for h in result.hits}
    ids = {h.id for h in result.hits}
    assert ItemKind.ENTITY in kinds and ItemKind.RELATIONSHIP in kinds
    assert "e1" in ids and "r1" in ids


async def test_cypher_channel_empty_terms_records_error():
    repo = FakeGraphRepo()
    cfg = RetrievalConfig()
    channel = CypherChannel(repo, cfg)
    [result] = await channel.run("?", None, [])
    assert result.error and "no candidate terms" in result.error


# ---------------------------------------------------------------- type blocklist


async def test_vector_channel_drops_blocklisted_types():
    """Person/Document/Organization entities must not survive the
    vector channel even when they appear in the underlying index. The
    channel-quality investigation showed these types dominate noise on
    biomedical Q&A — the default blocklist removes them."""
    gene = _make_entity("g1", "BRCA1", entity_type=EntityType.GENE)
    person = _make_entity("p1", "Levin B", entity_type=EntityType.PERSON)
    doc = _make_entity("d1", "Some paper", entity_type=EntityType.DOCUMENT)
    repo = FakeGraphRepo(
        vector_entity_hits=[(gene, 0.92), (person, 0.91), (doc, 0.85)],
    )
    channel = VectorChannel(repo, RetrievalConfig())
    [ent_result, _] = await channel.run("BRCA1", [0.0] * 4, ["BRCA1"])
    surviving_ids = {h.id for h in ent_result.hits}
    assert "g1" in surviving_ids
    assert "p1" not in surviving_ids
    assert "d1" not in surviving_ids


async def test_blocklist_can_be_disabled_via_empty_tuple():
    """Empty blocklist is the documented escape hatch — every entity
    flows through. Useful for "find papers about X" later, and the
    contract the API's `entity_type_blocklist: []` override depends on."""
    gene = _make_entity("g1", "BRCA1", entity_type=EntityType.GENE)
    person = _make_entity("p1", "Levin B", entity_type=EntityType.PERSON)
    repo = FakeGraphRepo(vector_entity_hits=[(gene, 0.92), (person, 0.91)])
    cfg = RetrievalConfig(entity_type_blocklist=())
    channel = VectorChannel(repo, cfg)
    [ent_result, _] = await channel.run("BRCA1", [0.0] * 4, ["BRCA1"])
    surviving_ids = {h.id for h in ent_result.hits}
    assert {"g1", "p1"} <= surviving_ids


async def test_bm25_channel_drops_blocklisted_types():
    gene = _make_entity("g1", "BRCA1", entity_type=EntityType.GENE)
    person = _make_entity("p1", "BRCA1 author", entity_type=EntityType.PERSON)
    repo = FakeGraphRepo(text_search_hits=[gene, person])
    channel = Bm25Channel(repo, RetrievalConfig())
    [result] = await channel.run("BRCA1", None, ["BRCA1"])
    surviving_ids = {h.id for h in result.hits}
    assert "g1" in surviving_ids
    assert "p1" not in surviving_ids


async def test_orchestrator_resolves_relationship_labels_to_entity_names(monkeypatch):
    """Cypher emits relationships with bare-UUID labels (``"<src_id> --REL--> <tgt_id>"``)
    because the channel doesn't have the entity-name lookup. Fix #3 makes the
    orchestrator batch-resolve those names after channels return so the
    cross-encoder rerank scores readable text instead of UUIDs.
    """
    imatinib = _make_entity("e1", "Imatinib")
    bcr_abl = _make_entity("e2", "BCR-ABL")
    rel = _make_rel("r1", "e1", "e2")

    repo = FakeGraphRepo(
        entities=[imatinib, bcr_abl],
        relationships=[rel],
        vector_rel_hits=[(rel, 0.92)],
        # Anchor the BM25/Cypher channels through text search so all
        # three channels surface the rel.
        text_search_hits=[imatinib],
        relationships_by_entity={"e1": [rel]},
    )

    from graphbuilder.infrastructure.services import embedding_factory
    monkeypatch.setattr(
        embedding_factory, "embed_async", lambda *_: _async_const([0.1] * 4),
    )

    orch = RetrievalOrchestrator(graph_repo=repo)
    items, _ = await orch.retrieve("Imatinib targets BCR-ABL")
    rel_item = next((i for i in items if i.kind is ItemKind.RELATIONSHIP), None)
    assert rel_item is not None
    # Label should now be human-readable, not a UUID-looking string.
    assert "Imatinib" in rel_item.label
    assert "BCR-ABL" in rel_item.label
    assert "e1" not in rel_item.label and "e2" not in rel_item.label


async def test_orchestrator_keeps_uuid_label_when_resolution_fails(monkeypatch):
    """When the repo's name lookup returns nothing, the orchestrator must
    keep the channel-emitted label rather than emit a half-resolved one
    (``"None --REL--> None"`` would be even more confusing for rerank)."""
    rel = _make_rel("r1", "e1", "e2")
    # Repo has the rel in vector hits but no entities → name_map is empty.
    repo = FakeGraphRepo(vector_rel_hits=[(rel, 0.9)])

    from graphbuilder.infrastructure.services import embedding_factory
    monkeypatch.setattr(
        embedding_factory, "embed_async", lambda *_: _async_const([0.1] * 4),
    )

    orch = RetrievalOrchestrator(graph_repo=repo)
    items, _ = await orch.retrieve("anything")
    rel_item = next((i for i in items if i.kind is ItemKind.RELATIONSHIP), None)
    assert rel_item is not None
    # Falls back to the channel's UUID-style label — no "None" leakage.
    assert "None" not in rel_item.label
    assert "e1" in rel_item.label  # original label preserved


async def test_cypher_channel_drops_blocklisted_anchors():
    """Blocked anchors shouldn't enter the loop — otherwise the entire
    1-hop neighbourhood of a Person/Document/Org would still leak in
    through the per-rel emission path."""
    gene = _make_entity("g1", "BRCA1", entity_type=EntityType.GENE)
    person = _make_entity("p1", "BRCA1 reviewer", entity_type=EntityType.PERSON)
    person_rel = _make_rel("r_person", "p1", "g1")
    repo = FakeGraphRepo(
        text_search_hits=[gene, person],
        relationships_by_entity={"p1": [person_rel], "g1": []},
    )
    channel = CypherChannel(repo, RetrievalConfig())
    [result] = await channel.run("BRCA1", None, ["BRCA1"])
    ids = {h.id for h in result.hits}
    assert "g1" in ids
    assert "p1" not in ids
    assert "r_person" not in ids   # the person's edge must not leak


# ---------------------------------------------------------------- orchestrator


@pytest.fixture
def orchestrator_setup():
    """Build a small graph with overlapping channel hits and chunks."""
    imatinib = _make_entity("e1", "Imatinib", chunk_ids=["c1"])
    bcr_abl = _make_entity("e2", "BCR-ABL", chunk_ids=["c1"])
    tp53 = _make_entity("e3", "TP53", chunk_ids=["c2"])
    rel = _make_rel("r1", "e1", "e2", chunk_ids=["c1"])

    repo = FakeGraphRepo(
        vector_entity_hits=[(imatinib, 0.92), (bcr_abl, 0.81), (tp53, 0.55)],
        vector_rel_hits=[(rel, 0.88)],
        text_search_hits=[imatinib, tp53],
        relationships_by_entity={"e1": [rel]},
        entities=[imatinib, bcr_abl, tp53],
        relationships=[rel],
    )
    docs = FakeDocumentRepo(
        chunks=[
            FakeChunk("c1", "Imatinib inhibits BCR-ABL fusion kinase activity."),
            FakeChunk("c2", "TP53 is a tumour suppressor gene."),
        ]
    )
    return repo, docs


async def test_orchestrator_runs_all_channels_and_fuses(orchestrator_setup, monkeypatch):
    repo, docs = orchestrator_setup

    # Stub the embedding factory so we don't need sentence-transformers.
    from graphbuilder.infrastructure.services import embedding_factory

    async def fake_embed_async(text):
        return [0.1] * 4

    monkeypatch.setattr(embedding_factory, "embed_async", fake_embed_async)

    cfg = RetrievalConfig(final_top_k=5)
    orch = RetrievalOrchestrator(graph_repo=repo, document_repo=docs, config=cfg)
    items, trace = await orch.retrieve("does Imatinib inhibit BCR-ABL?")

    assert len(items) > 0
    # Imatinib appeared in vector + bm25 + cypher → should outrank tp53.
    item_ids = [item.id for item in items]
    assert "e1" in item_ids
    assert item_ids.index("e1") < item_ids.index("e3") if "e3" in item_ids else True

    e1 = next(i for i in items if i.id == "e1")
    # Imatinib has at least vector + bm25 + cypher contributing.
    assert {Channel.VECTOR_ENTITY, Channel.BM25, Channel.CYPHER}.issubset(
        set(e1.contributing_channels)
    )
    # Multi-channel agreement should produce a bonus over the raw max channel score.
    base = max(s for s in (e1.score_vector, e1.score_bm25, e1.score_cypher) if s)
    assert e1.final_confidence >= base


async def test_orchestrator_hydrates_chunk_preview(orchestrator_setup, monkeypatch):
    repo, docs = orchestrator_setup

    from graphbuilder.infrastructure.services import embedding_factory

    async def fake_embed_async(text):
        return [0.1] * 4

    monkeypatch.setattr(embedding_factory, "embed_async", fake_embed_async)

    orch = RetrievalOrchestrator(graph_repo=repo, document_repo=docs)
    items, trace = await orch.retrieve("Imatinib")
    e1 = next((i for i in items if i.id == "e1"), None)
    assert e1 is not None
    assert e1.chunk_preview is not None
    assert "Imatinib" in e1.chunk_preview
    assert e1.source_chunk_id == "c1"
    assert trace.hydrated_chunks >= 1


async def test_orchestrator_promotes_chunks_to_first_class_items(orchestrator_setup, monkeypatch):
    """Hydrated chunks must appear as their own ``RetrievedItem(kind=chunk)``
    rows after the entity / relationship items.

    Locks in the P13 fix: without this promotion the eval harness's
    ``gold_chunk_ids`` check could never match because chunks were
    only metadata on their parent entity. The chunk row carries the
    real chunk id (so eval composite ids line up) and inherits the
    parent's confidence (slightly discounted) so confidence-sorted
    UIs don't surface a chunk above the entity it came from.
    """
    repo, docs = orchestrator_setup

    from graphbuilder.infrastructure.services import embedding_factory

    async def fake_embed_async(text):
        return [0.1] * 4

    monkeypatch.setattr(embedding_factory, "embed_async", fake_embed_async)

    orch = RetrievalOrchestrator(graph_repo=repo, document_repo=docs)
    items, _ = await orch.retrieve("Imatinib")

    chunk_items = [i for i in items if i.kind is ItemKind.CHUNK]
    assert chunk_items, "expected at least one promoted chunk item"
    parent = next(i for i in items if i.id == "e1")
    chunk = next((c for c in chunk_items if c.id == parent.source_chunk_id), None)
    assert chunk is not None
    # Chunk row carries the real chunk id and a content preview.
    assert chunk.id == "c1"
    assert "Imatinib" in (chunk.chunk_preview or "")
    # Confidence discounted vs parent so confidence-sorted UIs keep order.
    assert chunk.final_confidence <= parent.final_confidence
    # The metadata pointer back to the promoting parent helps the
    # frontend group chunk rows under their entity.
    assert chunk.metadata.get("promoted_from") == parent.id


async def test_orchestrator_chunk_promotion_can_be_disabled(orchestrator_setup, monkeypatch):
    """``cfg.emit_chunk_items=False`` restores the legacy 'chunks only
    as metadata' behaviour for callers that depend on it."""
    repo, docs = orchestrator_setup

    from graphbuilder.infrastructure.services import embedding_factory

    async def fake_embed_async(text):
        return [0.1] * 4

    monkeypatch.setattr(embedding_factory, "embed_async", fake_embed_async)

    cfg = RetrievalConfig(emit_chunk_items=False)
    orch = RetrievalOrchestrator(graph_repo=repo, document_repo=docs, config=cfg)
    items, _ = await orch.retrieve("Imatinib")
    assert all(i.kind is not ItemKind.CHUNK for i in items)


async def test_orchestrator_no_doc_repo_skips_hydration(monkeypatch):
    e1 = _make_entity("e1", "Imatinib", chunk_ids=["c1"])
    repo = FakeGraphRepo(
        vector_entity_hits=[(e1, 0.9)],
        text_search_hits=[e1],
    )

    from graphbuilder.infrastructure.services import embedding_factory
    monkeypatch.setattr(embedding_factory, "embed_async", lambda *_: _async_const([0.1] * 4))

    orch = RetrievalOrchestrator(graph_repo=repo, document_repo=None)
    items, trace = await orch.retrieve("Imatinib")
    assert items
    assert all(item.chunk_preview is None for item in items)
    assert trace.hydrated_chunks == 0


async def _async_const(value):
    return value


async def test_orchestrator_disabled_channel_omits_results(monkeypatch):
    e1 = _make_entity("e1", "Imatinib")
    repo = FakeGraphRepo(
        vector_entity_hits=[(e1, 0.9)],
        text_search_hits=[e1],
    )

    from graphbuilder.infrastructure.services import embedding_factory

    async def fake_embed_async(text):
        return [0.1] * 4

    monkeypatch.setattr(embedding_factory, "embed_async", fake_embed_async)

    cfg = RetrievalConfig(
        enable_bm25_channel=False,
        enable_cypher_channel=False,
    )
    orch = RetrievalOrchestrator(graph_repo=repo, config=cfg)
    items, trace = await orch.retrieve("Imatinib")
    channels_run = {cr.channel for cr in trace.channels}
    assert Channel.BM25 not in channels_run
    assert Channel.CYPHER not in channels_run
    # Vector still ran (entity + relationship sub-channels).
    assert Channel.VECTOR_ENTITY in channels_run


async def test_orchestrator_handles_no_embedding(monkeypatch):
    """When embedding factory returns None, vector channel records an error
    but BM25 + Cypher still produce results."""
    e1 = _make_entity("e1", "Imatinib")
    repo = FakeGraphRepo(text_search_hits=[e1])

    from graphbuilder.infrastructure.services import embedding_factory

    async def fake_embed_async(text):
        return None

    monkeypatch.setattr(embedding_factory, "embed_async", fake_embed_async)

    orch = RetrievalOrchestrator(graph_repo=repo)
    items, trace = await orch.retrieve("Imatinib")
    assert items, "expected BM25 / Cypher to still produce items"
    vec_results = [c for c in trace.channels if c.channel == Channel.VECTOR_ENTITY]
    assert vec_results and vec_results[0].error


async def test_orchestrator_empty_query_returns_empty(monkeypatch):
    repo = FakeGraphRepo()
    from graphbuilder.infrastructure.services import embedding_factory

    async def fake_embed_async(text):
        return None

    monkeypatch.setattr(embedding_factory, "embed_async", fake_embed_async)

    orch = RetrievalOrchestrator(graph_repo=repo)
    items, trace = await orch.retrieve("")
    assert items == []
    assert trace.extracted_terms == []


# ---------------------------------------------------------------- model serialisation


def test_retrieved_item_to_dict_round_trips_essential_fields():
    item = RetrievedItem(
        kind=ItemKind.ENTITY,
        id="e1",
        label="Imatinib",
        score_vector=0.9,
        score_bm25=0.7,
        score_rrf=0.123456,
        contributing_channels=[Channel.VECTOR_ENTITY, Channel.BM25],
        source_chunk_ids=["c1"],
    )
    d = item.to_dict()
    assert d["id"] == "e1"
    assert d["kind"] == "entity"
    assert d["score_vector"] == 0.9
    assert d["score_bm25"] == 0.7
    assert d["score_rrf"] == 0.123456
    assert d["contributing_channels"] == ["vector_entity", "bm25"]
    assert d["source_chunk_ids"] == ["c1"]
