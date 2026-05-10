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
from .reranker import CrossEncoderConfig, CrossEncoderReranker
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
        reranker: Optional[CrossEncoderReranker] = None,
    ) -> None:
        self._graph_repo = graph_repo
        self._document_repo = document_repo
        self._cfg = config or RetrievalConfig()
        self._vector_channel = VectorChannel(graph_repo, self._cfg)
        self._bm25_channel = Bm25Channel(graph_repo, self._cfg)
        self._cypher_channel = CypherChannel(graph_repo, self._cfg)
        # Reranker is constructed lazily — the model only loads on the
        # first `retrieve()` call, after the orchestrator is wired up.
        self._reranker = reranker or CrossEncoderReranker(
            CrossEncoderConfig(model_name=self._cfg.cross_encoder_model)
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def retrieve(
        self,
        query: str,
        *,
        top_k: Optional[int] = None,
        query_embedding: Optional[List[float]] = None,
        config_override: Optional[RetrievalConfig] = None,
    ) -> Tuple[List[RetrievedItem], RetrievalTrace]:
        """Retrieve and fuse for one query.

        ``query_embedding`` lets a caller (e.g. ``QAService``) pass in
        an embedding it already has so we don't re-embed for retrieval +
        memory recall on the same turn. ``None`` means "embed for me".

        ``config_override`` is the P13 ablation hook: callers (the eval
        runner) can pass a per-request ``RetrievalConfig`` to flip
        channel toggles, rerank, or chunk-promotion without rebuilding
        the orchestrator singleton. ``None`` keeps the construction-
        time config — the production-traffic path.

        Returns ``(items, trace)`` — items are sorted by rerank score
        when the cross-encoder ran, otherwise by RRF score descending.
        """
        cfg = config_override or self._cfg
        final_top_k = top_k or cfg.final_top_k
        wall_start = time.perf_counter()

        terms = extract_terms(query)
        if query_embedding is None:
            query_embedding = await self._embed_query(query)

        channel_results = await self._run_channels(query, query_embedding, terms, cfg=cfg)

        # Resolve rel-hit source/target ids → names so the cross-encoder
        # rerank scores readable text instead of bare UUIDs (fix #3 of
        # the channel-quality plan, §9.6 of docs/RAG_QA_PLAN.md). Single
        # batch repo call; missing ids fall back to the original UUID
        # label inside ``_build_item``.
        name_map = await self._resolve_relationship_names(channel_results)

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
            rankings, k=cfg.rrf_k, top_n=cfg.rrf_top_n
        )

        # Build RetrievedItem objects for every top_n candidate so the
        # cross-encoder reranks the full shortlist, not just the
        # already-trimmed final_top_k. Rerank uses label + description,
        # which doesn't need chunks to be hydrated yet — chunks are
        # fetched after the trim so we don't hydrate items that the
        # rerank would have dropped.
        candidates: List[RetrievedItem] = []
        for composite_id, rrf_score in fused:
            hits = hit_index.get(composite_id, [])
            if not hits:
                continue
            candidates.append(
                self._build_item(composite_id, hits, rrf_score, name_map=name_map)
            )

        if cfg.enable_cross_encoder and candidates:
            t_rerank = time.perf_counter()
            candidates = await self._reranker.rerank(query, candidates)
            await self._record_rerank_latency(time.perf_counter() - t_rerank)

        items = candidates[:final_top_k]

        # Chunk hydration with optional ±radius :NEXT_CHUNK expansion so
        # the LLM sees the surrounding paragraph not just the matched
        # sentence. Sequential + bounded — a slow doc-repo query can't
        # blow up the turn latency budget thanks to channel timeout.
        hydrated_count = 0
        if cfg.hydrate_chunks and self._document_repo is not None:
            hydrated_count = await self._hydrate_chunks(items, cfg=cfg)

        # Final confidence — when the cross-encoder ran we use it as the
        # primary signal; otherwise we fall back to the per-channel max
        # plus a "multi-channel agreement" bonus.
        for item in items:
            item.final_confidence = _compute_final_confidence(item)

        # Promote each unique hydrated chunk to a first-class
        # ``RetrievedItem(kind=chunk)``. Without this step a chunk only
        # appears as ``chunk_preview`` metadata on its parent entity —
        # the eval harness's ``gold_chunk_ids`` could never match, and
        # the frontend's source list would never expose the chunk as
        # its own row. Companions inherit the parent's confidence
        # (slightly discounted) and are appended *after* the entity /
        # relationship items so citation indices for the entities
        # remain stable for clients that already render them.
        if cfg.emit_chunk_items:
            chunk_items = self._emit_chunk_items(items, cfg=cfg)
            items = items + chunk_items

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
        *,
        cfg: Optional[RetrievalConfig] = None,
    ) -> List[ChannelResult]:
        """Run all channels concurrently with a per-channel timeout.

        ``cfg.enable_*_channel`` flags act as per-request circuit
        breakers — disabled channels are skipped entirely (no Cypher
        round-trip, no embedding cost) so ablations measuring "what if
        this channel was off?" stay honest about latency, not just
        about which results came back.
        """
        cfg = cfg or self._cfg

        async def _bounded(label: str, coro):
            try:
                return await asyncio.wait_for(
                    coro, timeout=cfg.channel_timeout_seconds
                )
            except asyncio.TimeoutError:
                logger.warning("%s channel timed out", label)
                return [ChannelResult(channel=Channel(label.split(":")[-1]),
                                      error="timeout")]
            except Exception as exc:
                logger.warning("%s channel raised: %s", label, exc)
                return []

        coros = []
        if cfg.enable_vector_channel:
            coros.append(_bounded(
                "channel:vector_entity",
                self._vector_channel.run(query, query_embedding, terms),
            ))
        if cfg.enable_bm25_channel:
            coros.append(_bounded(
                "channel:bm25",
                self._bm25_channel.run(query, query_embedding, terms),
            ))
        if cfg.enable_cypher_channel:
            coros.append(_bounded(
                "channel:cypher",
                self._cypher_channel.run(query, query_embedding, terms),
            ))

        if not coros:
            return []
        results = await asyncio.gather(*coros, return_exceptions=False)
        flat: List[ChannelResult] = []
        for sub in results:
            flat.extend(sub or [])
        return flat

    def _build_item(
        self,
        composite_id: str,
        hits: List[RawHit],
        rrf_score: float,
        *,
        name_map: Optional[Dict[str, str]] = None,
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

        # For relationship items, rewrite the label using resolved
        # source/target entity names so the cross-encoder rerank scores
        # readable text. Without this, Cypher emits ``"<src_uuid>
        # --INFLUENCES--> <tgt_uuid>"`` which the rerank can't tell
        # apart from any other UUID-only string.
        if kind is ItemKind.RELATIONSHIP and name_map:
            resolved = _resolve_rel_label(hits, name_map)
            if resolved is not None:
                label = resolved

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

    async def _hydrate_chunks(
        self,
        items: List[RetrievedItem],
        *,
        cfg: Optional[RetrievalConfig] = None,
    ) -> int:
        """Attach a chunk preview to each item, with ±radius neighbour
        expansion so the LLM sees the surrounding paragraph.

        For each item we walk the ``:NEXT_CHUNK`` linked list ±``cfg.
        chunk_neighbour_radius`` from its first cited chunk, then
        concatenate the neighbour contents (oldest → newest) capped at
        ``max_chunk_chars``. Falls back to the legacy single-chunk
        path when ``chunk_neighbour_radius == 0`` so callers can opt
        out without taking an extra Cypher round-trip.
        """
        if self._document_repo is None:
            return 0

        cfg = cfg or self._cfg
        radius = cfg.chunk_neighbour_radius

        # Fast path: radius=0 → batch fetch by id like the original
        # behaviour. Saves N Cypher calls when neighbours aren't needed.
        if radius <= 0:
            wanted: List[str] = []
            for item in items:
                for cid in item.source_chunk_ids[: cfg.max_chunks_per_item]:
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
                for cid in item.source_chunk_ids[: cfg.max_chunks_per_item]:
                    chunk = by_id.get(cid)
                    if not chunk:
                        continue
                    content = (chunk.content or "")[: cfg.max_chunk_chars]
                    item.chunk_preview = content
                    item.source_chunk_id = chunk.id
                    item.source_doc_id = item.source_doc_id or chunk.document_id
                    hydrated += 1
                    break
            return hydrated

        # Neighbour-expanded path: one repo call per item, in parallel.
        # Each call returns the chunk + its prev/next neighbours.
        async def _fetch(item_chunk_id: str) -> List[Any]:
            try:
                return await self._document_repo.get_chunk_with_neighbours(
                    item_chunk_id, radius=radius
                )
            except Exception as exc:
                logger.debug("get_chunk_with_neighbours(%s) failed: %s", item_chunk_id, exc)
                return []

        targets = [
            item.source_chunk_ids[0] if item.source_chunk_ids else None
            for item in items
        ]
        fetched = await asyncio.gather(
            *(_fetch(t) for t in targets if t),
            return_exceptions=False,
        )
        # Re-index by item — gather drops the entries where target was
        # None, so we walk both lists in step.
        results_iter = iter(fetched)
        hydrated = 0
        for item, target in zip(items, targets):
            if not target:
                continue
            neighbours = next(results_iter, [])
            if not neighbours:
                continue
            ordered = sorted(neighbours, key=lambda c: getattr(c, "chunk_index", 0))
            joined = " … ".join(
                (c.content or "").strip() for c in ordered if c.content
            )
            content = joined[: cfg.max_chunk_chars]
            anchor = next((c for c in ordered if c.id == target), ordered[0])
            item.chunk_preview = content
            item.source_chunk_id = anchor.id
            item.source_doc_id = item.source_doc_id or anchor.document_id
            hydrated += 1
        return hydrated

    async def _resolve_relationship_names(
        self, channel_results: List[ChannelResult]
    ) -> Dict[str, str]:
        """Batch ``id -> name`` lookup for every distinct source/target
        of a relationship hit across the channels.

        Single repo round-trip (Neo4j: ``MATCH WHERE id IN $ids``).
        Returns an empty map on any repo failure so the orchestrator
        falls through to the original UUID label rather than crashing
        the turn — labels are quality, not correctness.
        """
        ids: set[str] = set()
        for cr in channel_results:
            for hit in cr.hits:
                if hit.kind is not ItemKind.RELATIONSHIP:
                    continue
                src = hit.metadata.get("source_entity_id")
                tgt = hit.metadata.get("target_entity_id")
                if src:
                    ids.add(src)
                if tgt:
                    ids.add(tgt)
        if not ids:
            return {}
        try:
            return await self._graph_repo.get_entity_names_by_ids(list(ids))
        except Exception as exc:  # noqa: BLE001 — quality, not correctness
            logger.debug("relationship-name resolution failed: %s", exc)
            return {}

    def _emit_chunk_items(
        self,
        items: List[RetrievedItem],
        *,
        cfg: Optional[RetrievalConfig] = None,
    ) -> List[RetrievedItem]:
        """Build deduped chunk companions from hydrated entity/rel items.

        Ordering: chunks appear in the order their parent items first
        cite them. Each parent contributes at most one companion (its
        ``source_chunk_id`` after hydration), so the total can never
        exceed ``len(items)`` and is further capped by ``cfg.max_chunk_items``.
        """
        cfg = cfg or self._cfg
        out: List[RetrievedItem] = []
        seen: set[str] = set()
        cap = max(0, cfg.max_chunk_items)
        for parent in items:
            if parent.kind is ItemKind.CHUNK:
                continue  # never re-promote
            cid = parent.source_chunk_id
            if not cid or cid in seen:
                continue
            preview = parent.chunk_preview or ""
            if not preview.strip():
                # Hydration didn't run (no doc repo, or chunk lookup
                # missed). Don't promote a phantom chunk row.
                continue
            seen.add(cid)
            label = preview.strip().split("\n", 1)[0]
            if not label:
                label = f"chunk {cid[:8]}"
            elif len(label) > 120:
                label = label[:117].rstrip() + "…"
            # Slight discount so a chunk companion never outranks its
            # parent in confidence-sorted UIs. Keeps numerical ordering
            # of the existing items unchanged.
            confidence = max(0.0, round(parent.final_confidence - 0.05, 4))
            out.append(
                RetrievedItem(
                    kind=ItemKind.CHUNK,
                    id=cid,
                    label=label,
                    score_vector=parent.score_vector,
                    score_bm25=parent.score_bm25,
                    score_cypher=parent.score_cypher,
                    score_rrf=parent.score_rrf,
                    score_rerank=parent.score_rerank,
                    final_confidence=confidence,
                    source_url=parent.source_url,
                    source_doc_id=parent.source_doc_id,
                    source_chunk_id=cid,
                    source_chunk_ids=[cid],
                    chunk_preview=preview,
                    contributing_channels=list(parent.contributing_channels),
                    reasoning=f"hydrated from {parent.kind.value} {parent.id[:8]}",
                    metadata={"promoted_from": parent.id, "promoted_kind": parent.kind.value},
                )
            )
            if len(out) >= cap:
                break
        return out

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

    async def _record_rerank_latency(self, seconds: float) -> None:
        try:
            from ...infrastructure.services.metrics import get_metrics
        except Exception:
            return
        await get_metrics().record_qa_latency(phase="rerank", seconds=seconds)


# ----------------------------------------------------------------------
# Confidence + reasoning helpers
# ----------------------------------------------------------------------

def _resolve_rel_label(
    hits: List[RawHit], name_map: Dict[str, str]
) -> Optional[str]:
    """Rebuild a relationship label as ``"<src_name> --REL--> <tgt_name>"``.

    Pulls source_entity_id / target_entity_id / relationship_type from
    the first hit's metadata that has them. Returns ``None`` when
    neither end resolves to a name — the caller keeps the original
    UUID-based label rather than emit a half-resolved one (which would
    be even more confusing for the rerank).
    """
    src_id: Optional[str] = None
    tgt_id: Optional[str] = None
    rel_type: Optional[str] = None
    for h in hits:
        meta = h.metadata or {}
        src_id = src_id or meta.get("source_entity_id")
        tgt_id = tgt_id or meta.get("target_entity_id")
        rel_type = rel_type or meta.get("relationship_type")
        if src_id and tgt_id and rel_type:
            break
    if not src_id and not tgt_id:
        return None
    src_name = name_map.get(src_id or "")
    tgt_name = name_map.get(tgt_id or "")
    # Only rewrite when at least one end resolved — partial resolution
    # ("BRCA1 --INHIBITS--> 5e75dd33") is still useful to the rerank.
    if not src_name and not tgt_name:
        return None
    src_label = src_name or (src_id[:8] if src_id else "?")
    tgt_label = tgt_name or (tgt_id[:8] if tgt_id else "?")
    rel_label = rel_type or "RELATES"
    return f"{src_label} --{rel_label}--> {tgt_label}"


def _compute_final_confidence(item: RetrievedItem) -> float:
    """Blend per-channel + rerank scores into a single 0..1 confidence.

    With cross-encoder rerank (P4):

        primary = score_rerank                   (weight 0.7)
        channel = max(score_vector, bm25, cypher)  (weight 0.3)
        bonus   = 0.05 * (n_channels - 1)
        final   = clip(0.7*primary + 0.3*channel + bonus, 0, 1)

    Without rerank (cross-encoder unavailable / disabled):

        base    = max(score_vector, bm25, cypher)
        bonus   = 0.05 * (n_channels - 1)
        final   = clip(base + bonus, 0, 1)

    The "multiple channels agreed" bonus rewards items that show up
    in more than one ranking — those are unlikely to be coincidental.
    Citation-coverage (§4 of the plan, weight 0.2) is layered on later
    by the QA service after the LLM has answered.
    """
    components = [
        s for s in (item.score_vector, item.score_bm25, item.score_cypher)
        if s is not None
    ]
    channel_max = max(components) if components else 0.0
    bonus = 0.05 * max(0, len(item.contributing_channels) - 1)
    if item.score_rerank is not None:
        blended = 0.7 * item.score_rerank + 0.3 * channel_max
    else:
        blended = channel_max
    return round(min(1.0, blended + bonus), 4)


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
