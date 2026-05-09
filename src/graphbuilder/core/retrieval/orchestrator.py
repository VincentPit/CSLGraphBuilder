"""Retrieval orchestrator — runs the channels, fuses, hydrates chunks.

End-to-end flow (§3 of docs/RAG_QA_PLAN.md):

    1. Embed query (async, via embedding_factory).
    2. Extract candidate terms for BM25 + Cypher anchors.
    3. Run all enabled channels in parallel (with per-channel timeout).
    4. Build per-channel rankings keyed by composite item id
       ``"<kind>:<id>"`` so an entity can't collide with a relationship.
    5. Fuse via Reciprocal Rank Fusion → top-N candidates.
    6. Trim to ``final_top_k``, build :class:`RetrievedItem` per
       composite id with provenance + per-component scores.
    7. Hydrate source chunks (fetch the first-cited chunk, attach a
       short preview). Neighbour expansion lands in P4.
    8. Emit metrics + structured log + a :class:`RetrievalTrace`.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

from .channels import Bm25Channel, CypherChannel, VectorChannel
from .models import (
    Channel,
    ChannelResult,
    ItemKind,
    RawHit,
    RetrievalConfig,
    RetrievalTrace,
    RetrievedItem,
)
from .rrf import reciprocal_rank_fusion
from .term_extraction import extract_terms


logger = logging.getLogger("graphbuilder.qa.retrieval")


# Composite id for fusion: keeps entities and relationships in disjoint id-spaces.
def _ckey(kind: ItemKind, item_id: str) -> str:
    return f"{kind.value}:{item_id}"


class RetrievalOrchestrator:
    """Orchestrates the three retrieval channels + fusion + hydration."""

    def __init__(
        self,
        graph_repo: Any,
        document_repo: Optional[Any] = None,
        config: Optional[RetrievalConfig] = None,
    ) -> None:
        self._graph_repo = graph_repo
        self._document_repo = document_repo
        self._cfg = config or RetrievalConfig()
        self._vector_channel = VectorChannel(graph_repo, self._cfg)
        self._bm25_channel = Bm25Channel(graph_repo, self._cfg)
        self._cypher_channel = CypherChannel(graph_repo, self._cfg)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def retrieve(
        self,
        query: str,
        *,
        top_k: Optional[int] = None,
    ) -> Tuple[List[RetrievedItem], RetrievalTrace]:
        """Retrieve and fuse for one query.

        Returns ``(items, trace)`` — items are already sorted by RRF
        score, descending; trace is a structured record for the
        debug pane and metrics.
        """
        final_top_k = top_k or self._cfg.final_top_k
        wall_start = time.perf_counter()

        terms = extract_terms(query)
        query_embedding = await self._embed_query(query)

        channel_results = await self._run_channels(query, query_embedding, terms)

        # Build hit index: composite_id -> list of RawHit (across channels)
        hit_index: Dict[str, List[RawHit]] = {}
        for cr in channel_results:
            for hit in cr.hits:
                hit_index.setdefault(_ckey(hit.kind, hit.id), []).append(hit)

        # Per-channel rankings for RRF, keyed by composite id.
        rankings: List[Tuple[Channel, List[str]]] = []
        for cr in channel_results:
            if not cr.hits:
                continue
            rankings.append(
                (cr.channel, [_ckey(h.kind, h.id) for h in cr.hits])
            )

        fused = reciprocal_rank_fusion(
            rankings, k=self._cfg.rrf_k, top_n=self._cfg.rrf_top_n
        )

        items: List[RetrievedItem] = []
        for composite_id, rrf_score in fused[:final_top_k]:
            hits = hit_index.get(composite_id, [])
            if not hits:
                continue
            item = self._build_item(composite_id, hits, rrf_score)
            items.append(item)

        # Chunk hydration — sequential and bounded so a slow doc-repo
        # query can't blow up the turn latency budget.
        hydrated_count = 0
        if self._cfg.hydrate_chunks and self._document_repo is not None:
            hydrated_count = await self._hydrate_chunks(items)

        # Final confidence — for v1 it's max(per-channel score) + a
        # bump if multiple channels contributed. The cross-encoder
        # rerank in P4 will replace this with a learned score.
        for item in items:
            item.final_confidence = _compute_final_confidence(item)

        total_latency_ms = int((time.perf_counter() - wall_start) * 1000)
        trace = RetrievalTrace(
            query=query,
            extracted_terms=terms,
            channels=channel_results,
            rrf_top_n=len(fused),
            final_top_k=len(items),
            hydrated_chunks=hydrated_count,
            total_latency_ms=total_latency_ms,
        )
        await self._record_metrics(channel_results, total_latency_ms)
        logger.info(
            "retrieval done query=%r terms=%s channels=%s fused=%d top_k=%d latency_ms=%d",
            query[:120],
            terms,
            {cr.channel.value: cr.hit_count for cr in channel_results},
            len(fused),
            len(items),
            total_latency_ms,
        )
        return items, trace

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _embed_query(self, query: str) -> Optional[List[float]]:
        """Embed *query*, gracefully degrading to ``None`` on failure."""
        try:
            from ...infrastructure.services.embedding_factory import embed_async
        except Exception:
            return None
        try:
            return await embed_async(query)
        except Exception as exc:
            logger.debug("query embedding failed: %s", exc)
            return None

    async def _run_channels(
        self,
        query: str,
        query_embedding: Optional[List[float]],
        terms: List[str],
    ) -> List[ChannelResult]:
        """Run all channels concurrently with a per-channel timeout."""

        async def _bounded(label: str, coro):
            try:
                return await asyncio.wait_for(
                    coro, timeout=self._cfg.channel_timeout_seconds
                )
            except asyncio.TimeoutError:
                logger.warning("%s channel timed out", label)
                return [ChannelResult(channel=Channel(label.split(":")[-1]),
                                      error="timeout")]
            except Exception as exc:
                logger.warning("%s channel raised: %s", label, exc)
                return []

        results = await asyncio.gather(
            _bounded(
                "channel:vector_entity",
                self._vector_channel.run(query, query_embedding, terms),
            ),
            _bounded(
                "channel:bm25",
                self._bm25_channel.run(query, query_embedding, terms),
            ),
            _bounded(
                "channel:cypher",
                self._cypher_channel.run(query, query_embedding, terms),
            ),
            return_exceptions=False,
        )

        flat: List[ChannelResult] = []
        for sub in results:
            flat.extend(sub or [])
        return flat

    def _build_item(
        self, composite_id: str, hits: List[RawHit], rrf_score: float
    ) -> RetrievedItem:
        kind_str, item_id = composite_id.split(":", 1)
        kind = ItemKind(kind_str)
        # Prefer the longest non-empty label (entity names trump rel
        # arrow strings only when the rel doesn't have a meaningful
        # type). Fall back to the first hit's label.
        label = max(
            (h.label for h in hits if h.label),
            default=hits[0].label,
            key=len,
        )

        score_vector: Optional[float] = None
        score_bm25: Optional[float] = None
        score_cypher: Optional[float] = None
        contributing: List[Channel] = []
        merged_meta: Dict[str, Any] = {}
        chunk_ids: List[str] = []

        for h in hits:
            contributing.append(h.channel)
            merged_meta.update(h.metadata or {})
            if h.channel in (Channel.VECTOR_ENTITY, Channel.VECTOR_RELATIONSHIP):
                score_vector = max(score_vector or 0.0, h.score)
            elif h.channel is Channel.BM25:
                score_bm25 = max(score_bm25 or 0.0, h.score)
            elif h.channel is Channel.CYPHER:
                score_cypher = max(score_cypher or 0.0, h.score)
            for cid in h.metadata.get("source_chunk_ids", []) or []:
                if cid and cid not in chunk_ids:
                    chunk_ids.append(cid)

        # Dedup contributing list, preserving channel order of appearance.
        seen_chans: List[Channel] = []
        for c in contributing:
            if c not in seen_chans:
                seen_chans.append(c)

        reasoning = _build_reasoning(seen_chans, score_vector, score_bm25, score_cypher)

        return RetrievedItem(
            kind=kind,
            id=item_id,
            label=label,
            score_vector=score_vector,
            score_bm25=score_bm25,
            score_cypher=score_cypher,
            score_rrf=rrf_score,
            score_rerank=None,
            final_confidence=0.0,    # filled in by caller after hydration
            source_url=merged_meta.get("source_url"),
            source_doc_id=(merged_meta.get("source_document_ids") or [None])[0]
                if merged_meta.get("source_document_ids")
                else None,
            source_chunk_id=chunk_ids[0] if chunk_ids else None,
            source_chunk_ids=chunk_ids,
            chunk_preview=None,
            contributing_channels=seen_chans,
            reasoning=reasoning,
            metadata=merged_meta,
        )

    async def _hydrate_chunks(self, items: List[RetrievedItem]) -> int:
        """Attach a short text preview to each item from its first cited chunk.

        Bounded by ``max_chunks_per_item`` (we currently use one) and
        ``max_chunk_chars``. This is intentionally simple — neighbour
        expansion via ``:NEXT_CHUNK`` belongs in P4.
        """
        if self._document_repo is None:
            return 0

        # Collect a unique set of chunk ids across items.
        wanted: List[str] = []
        for item in items:
            for cid in item.source_chunk_ids[: self._cfg.max_chunks_per_item]:
                if cid and cid not in wanted:
                    wanted.append(cid)

        if not wanted:
            return 0

        try:
            chunks = await self._document_repo.get_chunks_by_ids(wanted)
        except Exception as exc:
            logger.warning("chunk hydration failed: %s", exc)
            return 0

        by_id = {c.id: c for c in (chunks or [])}
        hydrated = 0
        for item in items:
            for cid in item.source_chunk_ids[: self._cfg.max_chunks_per_item]:
                chunk = by_id.get(cid)
                if not chunk:
                    continue
                content = (chunk.content or "")[: self._cfg.max_chunk_chars]
                item.chunk_preview = content
                item.source_chunk_id = chunk.id
                item.source_doc_id = item.source_doc_id or chunk.document_id
                hydrated += 1
                break  # one preview per item is enough for v1
        return hydrated

    async def _record_metrics(
        self,
        channel_results: List[ChannelResult],
        total_latency_ms: int,
    ) -> None:
        try:
            from ...infrastructure.services.metrics import get_metrics
        except Exception:
            return
        m = get_metrics()
        await m.record_qa_latency(phase="retrieval", seconds=total_latency_ms / 1000.0)
        for cr in channel_results:
            await m.record_qa_retrieval_hits(
                channel=cr.channel.value, count=cr.hit_count
            )


# ----------------------------------------------------------------------
# Confidence + reasoning helpers
# ----------------------------------------------------------------------

def _compute_final_confidence(item: RetrievedItem) -> float:
    """Blend per-channel scores into a single 0..1 confidence.

    v1 formula (replaced by cross-encoder rerank in P4):

        base = max(score_vector, score_bm25, score_cypher)
        bonus = 0.05 * (number_of_contributing_channels - 1)
        final = clip(base + bonus, 0, 1)

    The "multiple channels agreed" bonus rewards items that show up in
    more than one ranking — those are unlikely to be coincidental.
    """
    components = [
        s for s in (item.score_vector, item.score_bm25, item.score_cypher)
        if s is not None
    ]
    base = max(components) if components else 0.0
    bonus = 0.05 * max(0, len(item.contributing_channels) - 1)
    return round(min(1.0, base + bonus), 4)


def _build_reasoning(
    channels: List[Channel],
    score_vector: Optional[float],
    score_bm25: Optional[float],
    score_cypher: Optional[float],
) -> str:
    parts = []
    if Channel.VECTOR_ENTITY in channels or Channel.VECTOR_RELATIONSHIP in channels:
        parts.append(f"vector={score_vector:.2f}" if score_vector is not None else "vector")
    if Channel.BM25 in channels:
        parts.append(f"bm25={score_bm25:.2f}" if score_bm25 is not None else "bm25")
    if Channel.CYPHER in channels:
        parts.append(f"cypher={score_cypher:.2f}" if score_cypher is not None else "cypher")
    return "matched via " + " + ".join(parts) if parts else "no channel signal"


__all__ = ["RetrievalOrchestrator"]
