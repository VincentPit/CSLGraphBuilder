"""Retrieval channels — each channel knows how to fetch hits for a query.

Three channels (§3.2 of docs/RAG_QA_PLAN.md):

1. :class:`VectorChannel`   — entity-name + relationship-description
                              cosine similarity via the existing Neo4j
                              vector indexes. Highest recall when
                              phrasing differs from canonical names.
2. :class:`Bm25Channel`     — substring / fulltext lookup via
                              ``GraphRepositoryInterface.search_entities_by_text``.
                              Catches exact identifiers (gene symbols,
                              drug codes) that vector search blurs.
3. :class:`CypherChannel`   — single hand-written 1-hop template:
                              given a candidate entity name from
                              :func:`extract_terms`, fetch the entity
                              + its incident relationships. Provides
                              the graph-shaped signal the LLM needs
                              for relational questions.

All channels are async, share a uniform :meth:`run` signature, and
return :class:`ChannelResult` objects so the orchestrator can fuse +
trace uniformly.
"""

from __future__ import annotations

import logging
import time
from typing import Any, List, Optional, Sequence

from .models import (
    Channel,
    ChannelResult,
    EmbeddingOrAwaitable,
    ItemKind,
    RawHit,
    RetrievalConfig,
    resolve_embedding,
)


logger = logging.getLogger("graphbuilder.qa.retrieval")


# ----------------------------------------------------------------------
# Shared filtering
# ----------------------------------------------------------------------

def _entity_type_str(entity: Any) -> Optional[str]:
    """Pull the ``entity_type`` value off either an enum or a raw string.

    Domain models hold ``entity_type`` as an :class:`EntityType` enum;
    Neo4j round-trips it as a string. Either form must compare cleanly
    against the blocklist tuple.
    """
    et = getattr(entity, "entity_type", None)
    if et is None:
        return None
    val = getattr(et, "value", None)
    return val if isinstance(val, str) else (et if isinstance(et, str) else str(et))


def _is_blocked(entity: Any, blocklist: Sequence[str]) -> bool:
    """``True`` iff ``entity`` should be dropped on type grounds.

    Reasoning lives in :class:`RetrievalConfig.entity_type_blocklist` —
    Person, Document, Organization are author/paper/affiliation
    metadata, not the biomedical content the user is asking about.
    Empty blocklist disables filtering entirely.
    """
    if not blocklist:
        return False
    return _entity_type_str(entity) in blocklist


# ----------------------------------------------------------------------
# Vector channel
# ----------------------------------------------------------------------

class VectorChannel:
    """Runs entity + relationship vector search in parallel.

    Neo4j's vector index returns cosine scores in [0, 1]. We split into
    two ``ChannelResult``s — one per index — so the trace shows them
    distinctly and RRF can give them independent rank votes.
    """

    def __init__(self, graph_repo: Any, config: RetrievalConfig):
        self._repo = graph_repo
        self._cfg = config

    async def run(
        self,
        query: str,
        query_embedding: EmbeddingOrAwaitable,
        terms: Sequence[str],
        *,
        cfg: Optional[RetrievalConfig] = None,
    ) -> List[ChannelResult]:
        # Per-call ``cfg`` lets the orchestrator's ``config_override``
        # (intent profiles, P13 ablations) reach the channel-level
        # knobs — without this, channels would always read their
        # construction-time ``self._cfg`` and overrides would silently
        # apply to top-level toggles only.
        cfg = cfg or self._cfg
        if not cfg.enable_vector_channel:
            return []
        run_rel = cfg.enable_vector_relationship
        # Block on the embedding only now — by the time we get here it's
        # usually already done (it was kicked off before the channels ran).
        query_embedding = await resolve_embedding(query_embedding)
        if not query_embedding:
            results: List[ChannelResult] = [
                ChannelResult(
                    channel=Channel.VECTOR_ENTITY,
                    error="no query embedding (embedding model unavailable)",
                ),
            ]
            if run_rel:
                results.append(
                    ChannelResult(
                        channel=Channel.VECTOR_RELATIONSHIP,
                        error="no query embedding (embedding model unavailable)",
                    )
                )
            return results

        # Entities.
        ent_result = ChannelResult(channel=Channel.VECTOR_ENTITY)
        t0 = time.perf_counter()
        try:
            # Over-fetch by 2× when a blocklist is set so we still
            # have a healthy candidate pool after filtering. Cheap on
            # Neo4j's vector index; trims dominated by min_score, not k.
            block = cfg.entity_type_blocklist
            fetch_k = cfg.vector_top_k * (2 if block else 1)
            ent_hits = await self._repo.vector_search_entities(
                query_embedding,
                top_k=fetch_k,
                min_score=cfg.vector_min_score,
            )
            kept = 0
            for entity, score in ent_hits:
                if _is_blocked(entity, block):
                    continue
                if kept >= cfg.vector_top_k:
                    break
                kept += 1
                ent_result.hits.append(
                    RawHit(
                        kind=ItemKind.ENTITY,
                        id=entity.id,
                        label=entity.name or entity.id,
                        channel=Channel.VECTOR_ENTITY,
                        score=float(score),
                        metadata={
                            "entity_type": getattr(
                                getattr(entity, "entity_type", None), "value", None
                            ),
                            "description": getattr(entity, "description", None),
                            "source_chunk_ids": list(
                                getattr(entity, "source_chunk_ids", []) or []
                            ),
                            "source_document_ids": list(
                                getattr(entity, "source_document_ids", []) or []
                            ),
                        },
                    )
                )
        except Exception as exc:
            logger.warning("vector_entity channel failed: %s", exc)
            ent_result.error = str(exc)
        ent_result.latency_ms = int((time.perf_counter() - t0) * 1000)

        # Relationships. Skipped entirely when the rel sub-toggle is
        # off — caller gets only the entity result, no error stub.
        if not run_rel:
            return [ent_result]

        rel_result = ChannelResult(channel=Channel.VECTOR_RELATIONSHIP)
        rel_top_k = cfg.vector_relationship_top_k or cfg.vector_top_k
        t0 = time.perf_counter()
        try:
            rel_hits = await self._repo.vector_search_relationships(
                query_embedding,
                top_k=rel_top_k,
                min_score=cfg.vector_min_score,
            )
            for rel, score in rel_hits:
                rel_result.hits.append(
                    RawHit(
                        kind=ItemKind.RELATIONSHIP,
                        id=rel.id,
                        label=_relationship_label(rel),
                        channel=Channel.VECTOR_RELATIONSHIP,
                        score=float(score),
                        metadata={
                            "source_entity_id": rel.source_entity_id,
                            "target_entity_id": rel.target_entity_id,
                            "relationship_type": getattr(
                                getattr(rel, "relationship_type", None), "value", None
                            ),
                            "description": getattr(rel, "description", None),
                            "source_chunk_ids": list(
                                getattr(rel, "source_chunk_ids", []) or []
                            ),
                            "source_document_ids": list(
                                getattr(rel, "source_document_ids", []) or []
                            ),
                        },
                    )
                )
        except Exception as exc:
            logger.warning("vector_relationship channel failed: %s", exc)
            rel_result.error = str(exc)
        rel_result.latency_ms = int((time.perf_counter() - t0) * 1000)

        return [ent_result, rel_result]


# ----------------------------------------------------------------------
# BM25 / fulltext channel
# ----------------------------------------------------------------------

class Bm25Channel:
    """Substring / fulltext entity search via the existing repo helper.

    Neo4j's fulltext index is BM25-style internally; the repo exposes it
    through ``search_entities_by_text(terms, limit)``. The channel
    iterates that result and gives each entity rank-based ordering.
    Scores from the fulltext API aren't comparable across channels —
    we only care about *order*, which RRF will use.
    """

    def __init__(self, graph_repo: Any, config: RetrievalConfig):
        self._repo = graph_repo
        self._cfg = config

    async def run(
        self,
        query: str,
        query_embedding: EmbeddingOrAwaitable,  # unused — BM25 is purely lexical
        terms: Sequence[str],
        *,
        cfg: Optional[RetrievalConfig] = None,
    ) -> List[ChannelResult]:
        cfg = cfg or self._cfg
        if not cfg.enable_bm25_channel:
            return []
        result = ChannelResult(channel=Channel.BM25)
        if not terms:
            result.error = "no candidate terms extracted from query"
            return [result]

        t0 = time.perf_counter()
        try:
            # The repo returns a dict {entity_id: GraphEntity}. We don't
            # get a per-entity score from the fulltext API, but iteration
            # order roughly mirrors index relevance for short term lists.
            block = cfg.entity_type_blocklist
            fetch_n = cfg.bm25_limit * (2 if block else 1)
            hits = await self._repo.search_entities_by_text(
                list(terms), limit=fetch_n,
            )
            # ``hits`` may be a dict or list depending on impl — handle both.
            if isinstance(hits, dict):
                entities = list(hits.values())
            else:
                entities = list(hits)
            entities = [e for e in entities if not _is_blocked(e, block)]
            entities = entities[: cfg.bm25_limit]
            for rank, entity in enumerate(entities, start=1):
                # Synthesise a 0..1 score from rank for the per-component
                # bar in the UI; the RRF stage uses rank, not this score.
                synthetic = max(0.0, 1.0 - (rank - 1) / max(len(entities), 1))
                result.hits.append(
                    RawHit(
                        kind=ItemKind.ENTITY,
                        id=entity.id,
                        label=entity.name or entity.id,
                        channel=Channel.BM25,
                        score=round(synthetic, 4),
                        metadata={
                            "entity_type": getattr(
                                getattr(entity, "entity_type", None), "value", None
                            ),
                            "description": getattr(entity, "description", None),
                            "source_chunk_ids": list(
                                getattr(entity, "source_chunk_ids", []) or []
                            ),
                            "source_document_ids": list(
                                getattr(entity, "source_document_ids", []) or []
                            ),
                        },
                    )
                )
        except Exception as exc:
            logger.warning("bm25 channel failed: %s", exc)
            result.error = str(exc)
        result.latency_ms = int((time.perf_counter() - t0) * 1000)
        return [result]


# ----------------------------------------------------------------------
# Cypher channel — single 1-hop template
# ----------------------------------------------------------------------

class CypherChannel:
    """Graph-shaped channel: 1-hop neighbourhood around matched entities.

    For v1 we ship a single hand-written template (see §3.2 — "10
    templates cover the main intents" is the long-term goal). The
    template:

      1. Use BM25 hits as anchor entities (or the LLM's later
         ``search_graph`` tool when that lands in P9).
      2. Fetch each anchor's incident relationships via
         ``graph_repo.get_entity_relationships(entity_id)``.
      3. Return the relationships and their other-end entities as
         hits — these are the graph-shaped pieces a relational
         question ("what does X target?") needs.

    Scores are rank-based — the closer the anchor was in the BM25
    ranking, the higher the score for its incident edges.
    """

    def __init__(self, graph_repo: Any, config: RetrievalConfig):
        self._repo = graph_repo
        self._cfg = config

    async def run(
        self,
        query: str,
        query_embedding: EmbeddingOrAwaitable,
        terms: Sequence[str],
        *,
        cfg: Optional[RetrievalConfig] = None,
    ) -> List[ChannelResult]:
        cfg = cfg or self._cfg
        if not cfg.enable_cypher_channel:
            return []
        result = ChannelResult(channel=Channel.CYPHER)
        # Resolve the embedding before deciding the anchor strategy —
        # ``_fetch_anchors`` prefers vector anchors when one is available
        # and falls back to term anchors only if it resolved to ``None``.
        query_embedding = await resolve_embedding(query_embedding)
        if not query_embedding and not terms:
            # Need *something* to anchor on. Vector path needs an
            # embedding; BM25 fallback needs terms. Surface the
            # missing-input case explicitly in the trace.
            result.error = "no query embedding and no candidate terms"
            return [result]

        t0 = time.perf_counter()
        try:
            block = cfg.entity_type_blocklist
            anchors = await self._fetch_anchors(query_embedding, terms, block, cfg)
            seen_ids: set[str] = set()
            for anchor_rank, anchor in enumerate(anchors, start=1):
                # Re-emit the anchor itself as a Cypher-channel hit so
                # the entity benefits from being in the graph-shaped
                # ranking too — not just BM25.
                if anchor.id not in seen_ids:
                    seen_ids.add(anchor.id)
                    result.hits.append(
                        RawHit(
                            kind=ItemKind.ENTITY,
                            id=anchor.id,
                            label=anchor.name or anchor.id,
                            channel=Channel.CYPHER,
                            score=_anchor_score(anchor_rank),
                            metadata={
                                "anchor_rank": anchor_rank,
                                "entity_type": getattr(
                                    getattr(anchor, "entity_type", None), "value", None
                                ),
                                "description": getattr(anchor, "description", None),
                                "source_chunk_ids": list(
                                    getattr(anchor, "source_chunk_ids", []) or []
                                ),
                                "source_document_ids": list(
                                    getattr(anchor, "source_document_ids", []) or []
                                ),
                            },
                        )
                    )

                # Fetch 1-hop incident relationships.
                try:
                    rels = await self._repo.get_entity_relationships(anchor.id)
                except Exception as exc:
                    logger.debug(
                        "cypher channel: get_entity_relationships(%s) failed: %s",
                        anchor.id, exc,
                    )
                    continue
                for rel in rels[: cfg.cypher_top_k]:
                    if rel.id in seen_ids:
                        continue
                    seen_ids.add(rel.id)
                    result.hits.append(
                        RawHit(
                            kind=ItemKind.RELATIONSHIP,
                            id=rel.id,
                            label=_relationship_label(rel),
                            channel=Channel.CYPHER,
                            score=_anchor_score(anchor_rank) * 0.9,
                            metadata={
                                "anchor_entity_id": anchor.id,
                                "anchor_rank": anchor_rank,
                                "source_entity_id": rel.source_entity_id,
                                "target_entity_id": rel.target_entity_id,
                                "relationship_type": getattr(
                                    getattr(rel, "relationship_type", None), "value", None
                                ),
                                "description": getattr(rel, "description", None),
                                "source_chunk_ids": list(
                                    getattr(rel, "source_chunk_ids", []) or []
                                ),
                                "source_document_ids": list(
                                    getattr(rel, "source_document_ids", []) or []
                                ),
                            },
                        )
                    )
        except Exception as exc:
            logger.warning("cypher channel failed: %s", exc)
            result.error = str(exc)
        result.latency_ms = int((time.perf_counter() - t0) * 1000)
        return [result]

    async def _fetch_anchors(
        self,
        query_embedding: Optional[List[float]],
        terms: Sequence[str],
        block: Sequence[str],
        cfg: RetrievalConfig,
    ) -> List[Any]:
        """Pick anchor entities to expand 1-hop neighbourhoods around.

        Fix #2 of the channel-quality plan (§9.6 of docs/RAG_QA_PLAN.md):
        anchor on **vector** hits when an embedding is available. The
        previous BM25-anchor approach used the same substring matcher
        as the BM25 channel, so any noise BM25 surfaced (e.g. Concept
        entities containing the gene symbol as a substring,
        ``"BRCA1-associated genome surveillance complex"``) seeded the
        Cypher channel's neighbourhood expansion too — two channels
        compounding the same mistake. Vector embeddings are semantic,
        which is more aligned with what biomedical Q&A users mean.

        Falls back to BM25 (the legacy behaviour) when no embedding is
        available — typically the cold-start case when the embedding
        model failed to load. Better degraded-anchors than no anchors.
        """
        fetch_k = cfg.cypher_top_k * (2 if block else 1)

        if query_embedding:
            ranked = await self._repo.vector_search_entities(
                query_embedding,
                top_k=fetch_k,
                min_score=cfg.vector_min_score,
            )
            anchors = [entity for entity, _score in ranked]
        else:
            hits_by_term = await self._repo.search_entities_by_text(
                list(terms), limit=fetch_k,
            )
            anchors = (
                list(hits_by_term.values())
                if isinstance(hits_by_term, dict)
                else list(hits_by_term)
            )

        # Blocked-type anchors would also drag in their entire 1-hop
        # neighbourhood (the whole point of this channel), so filter at
        # the anchor level — that's where the leverage is, not in the
        # per-rel emission below.
        anchors = [a for a in anchors if not _is_blocked(a, block)]
        return anchors[: cfg.cypher_top_k]


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _anchor_score(rank: int) -> float:
    """Convert an anchor's rank to a 0..1 score for the Cypher channel."""
    return round(max(0.0, 1.0 - (rank - 1) * 0.1), 4)


def _relationship_label(rel: Any) -> str:
    rtype = getattr(getattr(rel, "relationship_type", None), "value", None) or "RELATED_TO"
    src = rel.source_entity_id
    tgt = rel.target_entity_id
    return f"{src} --{rtype}--> {tgt}"


__all__ = ["VectorChannel", "Bm25Channel", "CypherChannel"]
