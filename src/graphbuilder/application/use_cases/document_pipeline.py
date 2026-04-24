"""Document extraction pipeline — the orchestrator the API actually uses.

A clean, lean replacement for the legacy ``ProcessDocumentUseCase`` task
state-machine. Drives the full document → graph flow with:

* **Stage-aware progress callbacks** — each stage (fetch, chunk,
  entities, relationships, finalize) reports start, fine-grained
  per-chunk progress, and completion. The API forwards these as SSE
  events to the frontend.
* **Cooperative cancellation** — between every chunk batch the pipeline
  invokes ``cancel_check()``; if it returns truthy, the run aborts
  cleanly, persisting whatever's already extracted.
* **Bounded parallel chunk processing** — entity and relationship
  extraction run with an ``asyncio.Semaphore`` so multiple chunks hit
  the LLM concurrently without flooding the provider.
* **Process-wide caches** — ``LLMDedupCache`` skips repeat dedup calls
  on identical (new, candidates) sets; ``EmbeddingCache`` skips repeat
  sentence-transformer encodes for the same string. Both report hits
  to ``PipelineMetrics``.

The legacy ``ProcessDocumentUseCase`` stays put for its tests; this is
the pragmatic path the FastAPI ``/documents/process`` endpoint runs.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional
from collections.abc import Awaitable

from ...domain.models.graph_models import (
    DocumentChunk,
    EntityType,
    GraphEntity,
    GraphRelationship,
    ProcessingStatus,
    RelationshipType,
    SourceDocument,
)
from ...domain.models.processing_models import ProcessingResult
from ...infrastructure.config.settings import GraphBuilderConfig
from ...infrastructure.repositories.document_repository import (
    DocumentRepositoryInterface,
)
from ...infrastructure.repositories.graph_repository import GraphRepositoryInterface
from ...infrastructure.services.cache import get_dedup_cache, get_embedding_cache
from ...infrastructure.services.llm_service import LLMServiceInterface
from ...infrastructure.services.metrics import get_metrics


logger = logging.getLogger("graphbuilder.pipeline")


# Public stage identifiers. Frontend renders them as the timeline rail.
STAGES = ("fetch", "chunk", "entities", "relationships", "finalize")


# Callback shape: stage, message, fraction in [0,1], extra structured data
ProgressCallback = Callable[[str, str, float, Optional[Dict[str, Any]]], Any]
CancelCheck = Callable[[], bool]


class PipelineCancelled(Exception):
    """Raised when ``cancel_check()`` returns true mid-run."""


@dataclass
class DocumentInput:
    """Lean input shape for the pipeline. Pre-fetched content preferred."""

    title: str
    content: str = ""
    source_url: Optional[str] = None
    file_path: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    chunk_size: Optional[int] = None
    chunk_overlap: Optional[int] = None
    allowed_nodes: Optional[List[str]] = None
    allowed_relationships: Optional[List[str]] = None


@dataclass
class PipelineResult:
    success: bool
    message: str
    document_id: Optional[str] = None
    chunks_created: int = 0
    entities_extracted: int = 0
    entities_merged: int = 0
    relationships_extracted: int = 0
    relationships_merged: int = 0
    # Verification summary — populated by _stage_verify
    relationships_verified: int = 0   # auto-approved (verification_status="verified")
    relationships_flagged: int = 0    # needs human review (status="flagged")
    relationships_rejected: int = 0   # conflicts with trusted data ("rejected")
    relationships_unreviewed: int = 0 # cascade was inconclusive but not flag-worthy
    entities_verified: int = 0
    entities_flagged: int = 0
    entities_rejected: int = 0
    entities_unreviewed: int = 0
    cancelled: bool = False
    duration_seconds: float = 0.0
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "message": self.message,
            "document_id": self.document_id,
            "chunks_created": self.chunks_created,
            "entities_extracted": self.entities_extracted,
            "entities_merged": self.entities_merged,
            "relationships_extracted": self.relationships_extracted,
            "relationships_merged": self.relationships_merged,
            "entities_verified": self.entities_verified,
            "entities_flagged": self.entities_flagged,
            "entities_rejected": self.entities_rejected,
            "entities_unreviewed": self.entities_unreviewed,
            "relationships_verified": self.relationships_verified,
            "relationships_flagged": self.relationships_flagged,
            "relationships_rejected": self.relationships_rejected,
            "relationships_unreviewed": self.relationships_unreviewed,
            "cancelled": self.cancelled,
            "duration_seconds": round(self.duration_seconds, 2),
            "error": self.error,
        }


class DocumentExtractionPipeline:
    """Drives a single document end-to-end through the extraction stack."""

    # Vector pre-filter parameters (cheap; LLM makes the final dedup call)
    VECTOR_PREFILTER_THRESHOLD = 0.4
    VECTOR_PREFILTER_TOP_K = 5
    DEDUP_CONFIDENCE_THRESHOLD = 0.7

    def __init__(
        self,
        config: GraphBuilderConfig,
        document_repo: DocumentRepositoryInterface,
        graph_repo: GraphRepositoryInterface,
        llm_service: LLMServiceInterface,
    ) -> None:
        self.config = config
        self.document_repo = document_repo
        self.graph_repo = graph_repo
        self.llm_service = llm_service
        self.metrics = get_metrics()
        self.dedup_cache = get_dedup_cache()
        self.embedding_cache = get_embedding_cache()
        # Concurrency bound for parallel per-chunk LLM calls.
        # Capped to the configured parallel_workers (default 4).
        try:
            workers = int(getattr(config.processing, "parallel_workers", 4) or 4)
        except Exception:
            workers = 4
        self._chunk_semaphore = asyncio.Semaphore(max(1, min(workers, 8)))

    # ------------------------------------------------------------------
    # Top-level entry point
    # ------------------------------------------------------------------

    async def run(
        self,
        doc_input: DocumentInput,
        *,
        progress: Optional[ProgressCallback] = None,
        cancel_check: Optional[CancelCheck] = None,
    ) -> PipelineResult:
        """Run the full pipeline for one document.

        ``progress(stage, message, fraction, data)`` is invoked at stage
        boundaries and inside long stages. ``cancel_check()`` is polled
        between chunk batches; truthy aborts the run.
        """

        started = datetime.now(timezone.utc)
        result = PipelineResult(success=True, message="Started")
        progress = progress or (lambda *a, **k: None)
        cancel_check = cancel_check or (lambda: False)

        try:
            await _maybe_await(progress("fetch", "Resolving content", 0.0, None))
            document, content = await self._stage_fetch(doc_input, progress)
            result.document_id = document.id
            self._raise_if_cancelled(cancel_check)

            await _maybe_await(progress("chunk", "Splitting into chunks", 0.0, None))
            chunks = await self._stage_chunk(document, content, doc_input, progress)
            result.chunks_created = len(chunks)
            self._raise_if_cancelled(cancel_check)

            await _maybe_await(progress("entities", "Extracting entities", 0.0, None))
            ent_summary = await self._stage_entities(
                document, chunks, doc_input, progress, cancel_check
            )
            result.entities_extracted = ent_summary["extracted"]
            result.entities_merged = ent_summary["merged"]
            self._raise_if_cancelled(cancel_check)

            await _maybe_await(
                progress("relationships", "Extracting relationships", 0.0, None)
            )
            rel_summary = await self._stage_relationships(
                document, chunks, doc_input, progress, cancel_check
            )
            result.relationships_extracted = rel_summary["extracted"]
            result.relationships_merged = rel_summary["merged"]
            self._raise_if_cancelled(cancel_check)

            # Stage 5 — auto-verify newly-saved relationships and tag them
            # verified / flagged / rejected per the configured thresholds so
            # the curation queue surfaces only the items that actually need
            # a human eye.
            if getattr(self.config, "verification", None) and self.config.verification.enabled:
                await _maybe_await(progress("verify", "Verifying new relationships", 0.0, None))
                verify_summary = await self._stage_verify(
                    document, progress, cancel_check
                )
                result.relationships_verified   = verify_summary["verified"]
                result.relationships_flagged    = verify_summary["flagged"]
                result.relationships_rejected   = verify_summary["rejected"]
                result.relationships_unreviewed = verify_summary["unreviewed"]
                result.entities_verified   = verify_summary.get("entity_verified",   0)
                result.entities_flagged    = verify_summary.get("entity_flagged",    0)
                result.entities_rejected   = verify_summary.get("entity_rejected",   0)
                result.entities_unreviewed = verify_summary.get("entity_unreviewed", 0)

            await _maybe_await(progress("finalize", "Persisting summary", 0.5, None))
            await self._stage_finalize(document, result)
            await _maybe_await(progress("finalize", "Done", 1.0, None))

            verify_summary = ""
            rel_total = (
                result.relationships_verified
                + result.relationships_flagged
                + result.relationships_rejected
                + result.relationships_unreviewed
            )
            ent_total = (
                result.entities_verified
                + result.entities_flagged
                + result.entities_rejected
                + result.entities_unreviewed
            )
            if rel_total + ent_total > 0:
                ent_part = (
                    f"entities {result.entities_verified}✓ {result.entities_flagged}⚠ {result.entities_rejected}✗"
                    if ent_total else "entities -"
                )
                rel_part = (
                    f"rels {result.relationships_verified}✓ {result.relationships_flagged}⚠ {result.relationships_rejected}✗"
                    if rel_total else "rels -"
                )
                verify_summary = f" · auto-verify: {ent_part} · {rel_part}"
            result.message = (
                f"Extracted {result.entities_extracted} entities and "
                f"{result.relationships_extracted} relationships from "
                f"{result.chunks_created} chunks{verify_summary}"
            )
            await self.metrics.record_document()

        except PipelineCancelled:
            result.cancelled = True
            result.success = False
            result.message = "Cancelled by user"
            try:
                await self._stage_finalize(document, result, status=ProcessingStatus.FAILED)
            except Exception:  # pragma: no cover — cancellation cleanup
                pass
        except Exception as exc:
            logger.exception("Pipeline failed: %s", exc)
            result.success = False
            result.error = str(exc)
            result.message = f"Pipeline failed: {exc}"

        result.duration_seconds = (
            datetime.now(timezone.utc) - started
        ).total_seconds()
        return result

    # ------------------------------------------------------------------
    # Stage 1 — fetch (URL → text or use provided content)
    # ------------------------------------------------------------------

    async def _stage_fetch(
        self, doc_input: DocumentInput, progress: ProgressCallback
    ) -> tuple[SourceDocument, str]:
        content = (doc_input.content or "").strip()
        if not content and doc_input.source_url:
            content = await _fetch_url(doc_input.source_url)
            await _maybe_await(
                progress(
                    "fetch",
                    f"Fetched {len(content)} chars from URL",
                    1.0,
                    {"chars": len(content)},
                )
            )
        elif not content:
            raise ValueError("Pipeline requires either content or source_url")
        else:
            await _maybe_await(
                progress(
                    "fetch",
                    f"Using provided content ({len(content)} chars)",
                    1.0,
                    {"chars": len(content)},
                )
            )

        # Build a SourceDocument that satisfies validation. The model
        # demands either source_url or file_path — synthesize one if
        # the caller passed plain text only.
        source_url = doc_input.source_url or f"text://{uuid.uuid4()}"
        document = SourceDocument(
            title=doc_input.title or source_url,
            source_url=source_url,
            file_path=doc_input.file_path,
            content_length=len(content),
        )
        if doc_input.tags:
            for tag in doc_input.tags:
                document.metadata.add_tag(tag)
        await self.document_repo.save(document)
        return document, content

    # ------------------------------------------------------------------
    # Stage 2 — chunk
    # ------------------------------------------------------------------

    async def _stage_chunk(
        self,
        document: SourceDocument,
        content: str,
        doc_input: DocumentInput,
        progress: ProgressCallback,
    ) -> List[DocumentChunk]:
        chunk_size = doc_input.chunk_size or getattr(
            self.config.processing, "chunk_size", 512
        )
        try:
            from graphbuilder.core.processing.semantic_chunker import (
                SemanticChunker,
                SemanticChunkerConfig,
            )

            chunker = SemanticChunker(
                SemanticChunkerConfig(
                    max_chunk_tokens=chunk_size,
                    min_chunk_tokens=30,
                    similarity_threshold=0.5,
                )
            )
            chunks = chunker.chunk(content, document.id)
        except Exception as exc:  # fall back to fixed-size split
            logger.warning("Semantic chunker failed (%s); falling back to fixed-size", exc)
            chunks = _fixed_size_chunks(
                content,
                document.id,
                chunk_size=chunk_size,
                overlap=doc_input.chunk_overlap or 50,
            )

        if not chunks:
            return []

        if hasattr(self.document_repo, "save_chunks_with_links"):
            try:
                await self.document_repo.save_chunks_with_links(chunks)
            except Exception as exc:
                logger.warning("Batch chunk save failed (%s); falling back to per-chunk", exc)
                for chunk in chunks:
                    await self.document_repo.save_chunk(chunk)
        else:
            for chunk in chunks:
                await self.document_repo.save_chunk(chunk)

        document.total_chunks = len(chunks)
        await self.document_repo.update(document)
        await self.metrics.record_chunks(len(chunks))

        await _maybe_await(
            progress(
                "chunk",
                f"Created {len(chunks)} chunks",
                1.0,
                {"count": len(chunks)},
            )
        )
        return chunks

    # ------------------------------------------------------------------
    # Stage 3 — entities (parallel per-chunk, cache-aware dedup)
    # ------------------------------------------------------------------

    async def _stage_entities(
        self,
        document: SourceDocument,
        chunks: List[DocumentChunk],
        doc_input: DocumentInput,
        progress: ProgressCallback,
        cancel_check: CancelCheck,
    ) -> Dict[str, int]:
        if not chunks:
            return {"extracted": 0, "merged": 0}

        completed = 0
        extracted = 0
        merged = 0
        lock = asyncio.Lock()
        extraction_config = {"allowed_nodes": doc_input.allowed_nodes or []}

        async def _process(chunk: DocumentChunk) -> tuple[int, int]:
            self._raise_if_cancelled(cancel_check)
            async with self._chunk_semaphore:
                self._raise_if_cancelled(cancel_check)
                ext_result = await self.llm_service.extract_entities(
                    chunk.content, extraction_config
                )
                if not ext_result.success:
                    return 0, 0
                raw = ext_result.data.get("entities", [])
                if not raw:
                    return 0, 0

                chunk_entities: List[GraphEntity] = []
                for ed in raw:
                    try:
                        etype = EntityType(ed.get("type", "CONCEPT"))
                    except ValueError:
                        etype = EntityType.CONCEPT
                    entity = GraphEntity(
                        name=ed.get("name", "").strip(),
                        entity_type=etype,
                        description=ed.get("description"),
                        properties=ed.get("properties", {}),
                        source_chunk_ids=[chunk.id],
                        source_document_ids=[chunk.document_id],
                    )
                    if not entity.name:
                        continue
                    entity.metadata.source_trust = "extracted"
                    # Land in the curation queue by default. A reviewer flips
                    # this to "verified" via /curation/events when they
                    # approve the extraction.
                    entity.metadata.add_annotation("verification_status", "unverified")
                    chunk_entities.append(entity)

                if not chunk_entities:
                    return 0, 0

                # Vector pre-filter to gather dedup candidates
                candidate_map: Dict[str, GraphEntity] = {}
                for entity in chunk_entities:
                    emb = await self._embed(_entity_text(entity))
                    if emb is None:
                        continue
                    try:
                        hits = await self.graph_repo.vector_search_entities(
                            emb,
                            top_k=self.VECTOR_PREFILTER_TOP_K,
                            min_score=self.VECTOR_PREFILTER_THRESHOLD,
                        )
                    except Exception:
                        hits = []
                    for hit_ent, _score in hits:
                        candidate_map[hit_ent.name.lower()] = hit_ent

                # Cache-aware LLM dedup
                merge_targets: Dict[str, GraphEntity] = {}
                if candidate_map:
                    new_dicts = [
                        {
                            "name": e.name,
                            "type": e.entity_type.value,
                            "description": e.description or "",
                        }
                        for e in chunk_entities
                    ]
                    existing_dicts = [
                        {
                            "name": e.name,
                            "type": e.entity_type.value,
                            "description": e.description or "",
                        }
                        for e in candidate_map.values()
                    ]
                    cache_key = self.dedup_cache.key_for_entities(
                        new_dicts, existing_dicts
                    )
                    cached = await self.dedup_cache.get(cache_key)
                    if cached is not None:
                        await self.metrics.record_llm_call(
                            prompt_type="entity_dedup",
                            prompt_tokens=0,
                            completion_tokens=0,
                            latency_seconds=0.0,
                            cache_hit=True,
                        )
                        matches = cached
                    else:
                        try:
                            dedup_result = (
                                await self.llm_service.resolve_entity_duplicates(
                                    new_dicts, existing_dicts
                                )
                            )
                            matches = (
                                dedup_result.data.get("matches", [])
                                if dedup_result.success
                                else []
                            )
                            await self.dedup_cache.set(cache_key, matches)
                        except Exception as err:
                            logger.debug("Entity LLM dedup failed: %s", err)
                            matches = []

                    for match in matches:
                        if match.get("confidence", 0) < self.DEDUP_CONFIDENCE_THRESHOLD:
                            continue
                        existing_name = (match.get("existing_name") or "").lower()
                        new_name = (match.get("new_name") or "").lower()
                        if existing_name in candidate_map:
                            merge_targets[new_name] = candidate_map[existing_name]

                local_extracted = 0
                local_merged = 0
                for entity in chunk_entities:
                    target = merge_targets.get(entity.name.lower())
                    if target and target.entity_type == entity.entity_type:
                        target.source_chunk_ids = list(
                            dict.fromkeys(target.source_chunk_ids + entity.source_chunk_ids)
                        )
                        target.source_document_ids = list(
                            dict.fromkeys(
                                target.source_document_ids + entity.source_document_ids
                            )
                        )
                        if entity.description and (
                            not target.description
                            or len(entity.description) > len(target.description)
                        ):
                            target.description = entity.description
                        await self.graph_repo.save_entity(target)
                        local_merged += 1
                    else:
                        await self.graph_repo.save_entity(entity)
                    local_extracted += 1

                await self.metrics.record_entities(local_extracted)
                return local_extracted, local_merged

        async def _wrapped(chunk: DocumentChunk) -> None:
            nonlocal completed, extracted, merged
            try:
                e, m = await _process(chunk)
            except PipelineCancelled:
                raise
            except Exception as exc:
                logger.warning("Entity extraction error in chunk %s: %s", chunk.id, exc)
                e, m = 0, 0
            async with lock:
                completed += 1
                extracted += e
                merged += m
                fraction = completed / len(chunks)
                await _maybe_await(
                    progress(
                        "entities",
                        f"Processed {completed}/{len(chunks)} chunks "
                        f"({extracted} entities, {merged} merged)",
                        fraction,
                        {
                            "completed": completed,
                            "total": len(chunks),
                            "extracted": extracted,
                            "merged": merged,
                        },
                    )
                )

        await _gather_with_cancel(
            [_wrapped(c) for c in chunks], cancel_check=cancel_check
        )
        return {"extracted": extracted, "merged": merged}

    # ------------------------------------------------------------------
    # Stage 4 — relationships (parallel per-chunk)
    # ------------------------------------------------------------------

    async def _stage_relationships(
        self,
        document: SourceDocument,
        chunks: List[DocumentChunk],
        doc_input: DocumentInput,
        progress: ProgressCallback,
        cancel_check: CancelCheck,
    ) -> Dict[str, int]:
        if not chunks:
            return {"extracted": 0, "merged": 0}

        all_entities = await self.graph_repo.get_all_entities()
        doc_entity_list: List[Dict[str, Any]] = []
        entity_name_to_id: Dict[str, str] = {}
        entity_id_to_name: Dict[str, str] = {}
        for entity in all_entities.values():
            if document.id in (entity.source_document_ids or []):
                doc_entity_list.append(
                    {
                        "name": entity.name,
                        "type": entity.entity_type.value,
                        "description": entity.description or "",
                    }
                )
                entity_name_to_id[entity.name.lower()] = entity.id
                entity_id_to_name[entity.id] = entity.name
        if not doc_entity_list:
            await _maybe_await(
                progress("relationships", "No entities — skipping", 1.0, None)
            )
            return {"extracted": 0, "merged": 0}

        completed = 0
        extracted = 0
        merged = 0
        lock = asyncio.Lock()

        async def _process(chunk: DocumentChunk) -> tuple[int, int]:
            self._raise_if_cancelled(cancel_check)
            async with self._chunk_semaphore:
                self._raise_if_cancelled(cancel_check)
                ext_result = await self.llm_service.extract_relationships(
                    chunk.content, doc_entity_list, {}
                )
                if not ext_result.success:
                    return 0, 0
                rels = ext_result.data.get("relationships", [])
                if not rels:
                    return 0, 0

                local_extracted = 0
                local_merged = 0
                for rel_data in rels:
                    src_raw = (rel_data.get("source_entity") or "").strip()
                    tgt_raw = (rel_data.get("target_entity") or "").strip()
                    src_id = entity_name_to_id.get(src_raw.lower())
                    tgt_id = entity_name_to_id.get(tgt_raw.lower())
                    if not src_id or not tgt_id:
                        continue
                    try:
                        rtype = RelationshipType(
                            rel_data.get("relationship_type", "RELATED_TO")
                        )
                    except ValueError:
                        rtype = RelationshipType.RELATED_TO

                    relationship = GraphRelationship(
                        source_entity_id=src_id,
                        target_entity_id=tgt_id,
                        relationship_type=rtype,
                        description=rel_data.get("description"),
                        properties=rel_data.get("properties", {}),
                        strength=float(rel_data.get("confidence", 1.0)),
                        source_chunk_ids=[chunk.id],
                        source_document_ids=[chunk.document_id],
                    )
                    relationship.metadata.source_trust = "extracted"
                    relationship.metadata.add_annotation("verification_status", "unverified")

                    # Cache-aware LLM relationship dedup
                    merged_existing = await self._maybe_merge_relationship(
                        relationship, src_id, tgt_id, rtype, rel_data, entity_id_to_name
                    )
                    if merged_existing:
                        local_merged += 1
                        local_extracted += 1
                        continue
                    await self.graph_repo.save_relationship(relationship)
                    local_extracted += 1

                await self.metrics.record_relationships(local_extracted)
                return local_extracted, local_merged

        async def _wrapped(chunk: DocumentChunk) -> None:
            nonlocal completed, extracted, merged
            try:
                e, m = await _process(chunk)
            except PipelineCancelled:
                raise
            except Exception as exc:
                logger.warning(
                    "Relationship extraction error in chunk %s: %s", chunk.id, exc
                )
                e, m = 0, 0
            async with lock:
                completed += 1
                extracted += e
                merged += m
                fraction = completed / len(chunks)
                await _maybe_await(
                    progress(
                        "relationships",
                        f"Processed {completed}/{len(chunks)} chunks "
                        f"({extracted} relationships, {merged} merged)",
                        fraction,
                        {
                            "completed": completed,
                            "total": len(chunks),
                            "extracted": extracted,
                            "merged": merged,
                        },
                    )
                )

        await _gather_with_cancel(
            [_wrapped(c) for c in chunks], cancel_check=cancel_check
        )
        return {"extracted": extracted, "merged": merged}

    async def _maybe_merge_relationship(
        self,
        relationship: GraphRelationship,
        src_id: str,
        tgt_id: str,
        rtype: RelationshipType,
        rel_data: Dict[str, Any],
        entity_id_to_name: Dict[str, str],
    ) -> bool:
        """Return True if relationship was merged into an existing one."""
        try:
            existing_rels = await self.graph_repo.get_entity_relationships(src_id)
        except Exception:
            return False
        same_pair = [r for r in existing_rels if r.target_entity_id == tgt_id]
        if not same_pair:
            return False

        # Exact-type duplicate handled by save_relationship's MERGE path;
        # we still call save_relationship below so the existing edge picks
        # up the new chunk provenance.
        if any(r.relationship_type == rtype for r in same_pair):
            await self.graph_repo.save_relationship(relationship)
            return False

        src_name = entity_id_to_name.get(src_id, "")
        tgt_name = entity_id_to_name.get(tgt_id, "")
        new_rel = {
            "source": src_name,
            "target": tgt_name,
            "type": rtype.value,
            "description": rel_data.get("description", ""),
        }
        existing_dicts = [
            {
                "source": src_name,
                "target": tgt_name,
                "type": r.relationship_type.value,
                "description": r.description or "",
            }
            for r in same_pair
        ]
        cache_key = self.dedup_cache.key_for_relationship(new_rel, existing_dicts)
        cached = await self.dedup_cache.get(cache_key)
        if cached is not None:
            await self.metrics.record_llm_call(
                prompt_type="rel_dedup",
                prompt_tokens=0,
                completion_tokens=0,
                latency_seconds=0.0,
                cache_hit=True,
            )
            decision = cached
        else:
            try:
                dup_result = await self.llm_service.check_relationship_duplicates(
                    new_rel, existing_dicts
                )
                decision = {
                    "duplicate_of": dup_result.data.get("duplicate_of"),
                    "confidence": dup_result.data.get("confidence", 0),
                }
                await self.dedup_cache.set(cache_key, decision)
            except Exception as err:
                logger.debug("Relationship LLM dedup failed: %s", err)
                decision = {"duplicate_of": None, "confidence": 0}

        idx = decision.get("duplicate_of")
        if idx is not None and decision.get("confidence", 0) >= self.DEDUP_CONFIDENCE_THRESHOLD:
            existing = same_pair[idx]
            existing.source_chunk_ids = list(
                dict.fromkeys(existing.source_chunk_ids + relationship.source_chunk_ids)
            )
            existing.source_document_ids = list(
                dict.fromkeys(
                    existing.source_document_ids + relationship.source_document_ids
                )
            )
            await self.graph_repo.save_relationship(existing)
            return True
        await self.graph_repo.save_relationship(relationship)
        return False

    # ------------------------------------------------------------------
    # Stage 5 — auto-verify (cascade → confidence → status)
    # ------------------------------------------------------------------

    async def _stage_verify(
        self,
        document: SourceDocument,
        progress: ProgressCallback,
        cancel_check: CancelCheck,
    ) -> Dict[str, int]:
        """Auto-verify newly-saved entities and relationships.

        Two passes:
          1. Entities — fast (text-match in source + cosine similarity to
             existing graph), no LLM ever. Run first so the relationship
             pass sees up-to-date entity statuses.
          2. Relationships — cascading verifier (text-match → embedding →
             LLM, with the LLM stage skipped in batch mode by default).

        For both, the (confidence, conflict, source_trust) triple maps to
        a ``verification_status`` per the README standard. We fetch
        targets (rather than threading IDs through earlier stages) so a
        previous crashed pipeline's unverified leftovers also get picked
        up.

        Returns counts under both ``entity_*`` and ``rel_*`` prefixes for
        ergonomics; the caller only consumes the relationship counts in
        the legacy keys (verified/flagged/rejected/unreviewed) plus the
        new entity_* counterparts.
        """
        from ...core.verification import (
            CascadingVerifier,
            CascadingVerifierConfig,
            EntityVerifier,
        )

        cfg = self.config.verification

        # ── Pass 1 — entities ─────────────────────────────────────────
        ent_counts = await self._verify_entities_pass(
            document, EntityVerifier(graph_repo=self.graph_repo), cfg, progress, cancel_check,
        )

        # Pull all relationships for this document that are still flagged as
        # unverified (the default tag set in _stage_relationships).
        all_rels = await self.graph_repo.get_all_relationships()
        targets = []
        for rel in all_rels.values():
            if document.id not in (rel.source_document_ids or []):
                continue
            ann = getattr(getattr(rel, "metadata", None), "annotations", {}) or {}
            if ann.get("verification_status") in ("unverified", None, ""):
                targets.append(rel)

        if not targets:
            await _maybe_await(progress("verify", "Nothing to verify", 1.0, None))
            return {"verified": 0, "flagged": 0, "rejected": 0, "unreviewed": 0}

        # Build a verifier. Skip the LLM stage in batch mode for speed —
        # the user can still trigger it from the /verification page.
        verifier_cfg = CascadingVerifierConfig(
            enable_llm=not cfg.batch_skip_llm,
        )
        verifier = CascadingVerifier(
            config=verifier_cfg,
            llm_service=self.llm_service if not cfg.batch_skip_llm else None,
            graph_repo=self.graph_repo,
        )

        # Resolve entity-id → name for the verifier kwargs.
        all_entities = await self.graph_repo.get_all_entities()
        id_to_name = {e.id: e.name for e in all_entities.values()}

        # Pre-fetch source chunks for all targets in one round-trip so each
        # verifier call doesn't hit Neo4j.
        wanted_chunk_ids: set[str] = set()
        for rel in targets:
            wanted_chunk_ids.update(rel.source_chunk_ids or [])
        chunk_text: Dict[str, str] = {}
        if wanted_chunk_ids and hasattr(self.document_repo, "get_chunks_by_ids"):
            try:
                chunks = await self.document_repo.get_chunks_by_ids(list(wanted_chunk_ids))
                for c in chunks:
                    chunk_text[c.id] = c.content
            except Exception as exc:
                logger.debug("Verify stage: chunk lookup failed (%s)", exc)

        sem = asyncio.Semaphore(max(1, int(cfg.parallel_workers)))
        counts = {"verified": 0, "flagged": 0, "rejected": 0, "unreviewed": 0}
        completed = 0
        lock = asyncio.Lock()

        async def _verify_one(rel) -> None:
            nonlocal completed
            self._raise_if_cancelled(cancel_check)
            async with sem:
                self._raise_if_cancelled(cancel_check)
                # Build context from the rel's source chunks (truncated).
                ctx_parts = [chunk_text[cid] for cid in (rel.source_chunk_ids or []) if cid in chunk_text]
                context = ("\n\n".join(ctx_parts))[:4000] if ctx_parts else ""
                src_name = id_to_name.get(rel.source_entity_id)
                tgt_name = id_to_name.get(rel.target_entity_id)

                # The verifier is sync; run in a thread so the LLM call (if
                # enabled) doesn't block the event loop.
                try:
                    vr = await asyncio.get_running_loop().run_in_executor(
                        None,
                        lambda: verifier.verify(
                            relationship=rel,
                            context=context,
                            source_name=src_name,
                            target_name=tgt_name,
                        ),
                    )
                except Exception as exc:
                    logger.debug("Verify failed for rel %s: %s", rel.id, exc)
                    vr = None

                status, notes = self._classify(vr, rel, cfg)
                rel.metadata.add_annotation("verification_status", status)
                if notes:
                    rel.metadata.add_annotation("verification_notes", notes)
                if vr is not None:
                    rel.metadata.add_annotation(
                        "verification_confidence", round(float(vr.confidence or 0.0), 3)
                    )
                try:
                    await self.graph_repo.save_relationship(rel)
                except Exception as exc:
                    logger.warning("Could not persist verification for %s: %s", rel.id, exc)

                async with lock:
                    counts[status if status in counts else "unreviewed"] += 1
                    completed += 1
                    fraction = completed / len(targets)
                    await _maybe_await(progress(
                        "verify",
                        f"Verified {completed}/{len(targets)} "
                        f"({counts['verified']} verified · {counts['flagged']} flagged · {counts['rejected']} rejected)",
                        fraction,
                        {"completed": completed, "total": len(targets), **counts},
                    ))

        await _gather_with_cancel([_verify_one(r) for r in targets], cancel_check=cancel_check)
        # Merge entity counts in under the entity_* prefix so the caller can
        # populate PipelineResult fields independently of relationship counts.
        return {
            **counts,
            "entity_verified":   ent_counts["verified"],
            "entity_flagged":    ent_counts["flagged"],
            "entity_rejected":   ent_counts["rejected"],
            "entity_unreviewed": ent_counts["unreviewed"],
        }

    async def _verify_entities_pass(
        self,
        document: SourceDocument,
        verifier: Any,
        cfg: Any,
        progress: ProgressCallback,
        cancel_check: CancelCheck,
    ) -> Dict[str, int]:
        """First half of the verify stage: classify newly-saved entities.

        Cheap by design — text-match against this run's chunk text +
        embedding similarity to the existing graph. No LLM stage. Same
        threshold mapping as relationships, using the entity-specific
        ``entity_auto_approve`` / ``entity_flag_below`` knobs.
        """
        all_entities = await self.graph_repo.get_all_entities()
        targets = []
        for ent in all_entities.values():
            if document.id not in (ent.source_document_ids or []):
                continue
            ann = getattr(getattr(ent, "metadata", None), "annotations", {}) or {}
            if ann.get("verification_status") in ("unverified", None, ""):
                targets.append(ent)
        if not targets:
            return {"verified": 0, "flagged": 0, "rejected": 0, "unreviewed": 0}

        # Pull every chunk this document spawned in one round-trip — same set
        # the relationship pass needs, so this is shared work.
        wanted_chunk_ids: set[str] = set()
        for ent in targets:
            wanted_chunk_ids.update(ent.source_chunk_ids or [])
        chunk_text_by_id: Dict[str, str] = {}
        if wanted_chunk_ids and hasattr(self.document_repo, "get_chunks_by_ids"):
            try:
                chunks = await self.document_repo.get_chunks_by_ids(list(wanted_chunk_ids))
                for c in chunks:
                    chunk_text_by_id[c.id] = c.content
            except Exception as exc:
                logger.debug("Entity verify: chunk lookup failed (%s)", exc)

        sem = asyncio.Semaphore(max(1, int(cfg.parallel_workers)))
        counts = {"verified": 0, "flagged": 0, "rejected": 0, "unreviewed": 0}
        completed = 0
        lock = asyncio.Lock()

        async def _verify_one(ent: Any) -> None:
            nonlocal completed
            self._raise_if_cancelled(cancel_check)
            async with sem:
                self._raise_if_cancelled(cancel_check)
                chunk_texts = [chunk_text_by_id[cid] for cid in (ent.source_chunk_ids or []) if cid in chunk_text_by_id]
                try:
                    vr = await verifier.verify(ent, chunk_texts=chunk_texts)
                except Exception as exc:
                    logger.debug("Entity verifier failed for %s: %s", ent.id, exc)
                    vr = None
                status, notes = self._classify_entity(vr, ent, cfg)
                ent.metadata.add_annotation("verification_status", status)
                if notes:
                    ent.metadata.add_annotation("verification_notes", notes)
                if vr is not None:
                    ent.metadata.add_annotation(
                        "verification_confidence", round(float(vr.confidence or 0.0), 3),
                    )
                try:
                    await self.graph_repo.save_entity(ent)
                except Exception as exc:
                    logger.warning("Could not persist entity verification for %s: %s", ent.id, exc)
                async with lock:
                    counts[status if status in counts else "unreviewed"] += 1
                    completed += 1
                    fraction = completed / len(targets) * 0.5  # entity pass owns first half of the bar
                    await _maybe_await(progress(
                        "verify",
                        f"Verified {completed}/{len(targets)} entities "
                        f"({counts['verified']} verified · {counts['flagged']} flagged)",
                        fraction,
                        {"phase": "entities", "completed": completed, "total": len(targets), **counts},
                    ))

        await _gather_with_cancel([_verify_one(e) for e in targets], cancel_check=cancel_check)
        return counts

    @staticmethod
    def _classify_entity(verification_result: Any, entity: Any, cfg: Any) -> tuple[str, Optional[str]]:
        """Same shape as ``_classify`` but using the entity thresholds."""
        source_trust = getattr(getattr(entity, "metadata", None), "source_trust", None)
        if verification_result is None:
            return "unverified", "Entity verifier did not run"
        confidence = float(verification_result.confidence or 0.0)
        if confidence >= cfg.entity_auto_approve:
            return "verified", f"Auto-approved at {confidence:.2f} confidence"
        if source_trust == "reviewed" and confidence >= cfg.trusted_auto_approve:
            return "verified", f"Trusted source + {confidence:.2f} confidence"
        if confidence < cfg.entity_flag_below:
            return "flagged", f"Low confidence ({confidence:.2f}) — needs review"
        return "unverified", f"Inconclusive ({confidence:.2f})"

    @staticmethod
    def _classify(
        verification_result: Any,
        relationship: Any,
        cfg: Any,
    ) -> tuple[str, Optional[str]]:
        """Map the verifier's output to a ``verification_status`` string.

        Rules — relationship variant of the table in README:

            conflict_detected           → cfg.treat_conflict_as ("rejected")
            confidence ≥ rel_auto_approve OR
              (source_trust=="reviewed" AND confidence ≥ trusted_auto_approve)
                                        → "verified"
            confidence < rel_flag_below → "flagged"
            anything else               → "unverified"

        ``verification_result`` may be None if the verifier crashed; we fall
        back to "unverified" rather than dropping the relationship.
        """
        source_trust = getattr(getattr(relationship, "metadata", None), "source_trust", None)

        # Conflict signal: VerificationResult exposes a `conflict_detected`
        # field on some stages; we also treat an explicit FAILED status as a
        # soft conflict for now.
        conflict = False
        if verification_result is not None:
            for stage_res in (getattr(verification_result, "stage_results", []) or []):
                if getattr(stage_res, "conflict_detected", False):
                    conflict = True
                    break
        if conflict:
            return cfg.treat_conflict_as, "Conflicts with existing trusted data"

        if verification_result is None:
            return "unverified", "Verifier did not run (error)"

        confidence = float(verification_result.confidence or 0.0)
        if confidence >= cfg.relationship_auto_approve:
            return "verified", f"Auto-approved at {confidence:.2f} confidence"
        if source_trust == "reviewed" and confidence >= cfg.trusted_auto_approve:
            return "verified", f"Trusted source + {confidence:.2f} confidence"
        if confidence < cfg.relationship_flag_below:
            return "flagged", f"Low confidence ({confidence:.2f}) — needs review"
        return "unverified", f"Inconclusive ({confidence:.2f})"

    # ------------------------------------------------------------------
    # Stage 6 — finalize
    # ------------------------------------------------------------------

    async def _stage_finalize(
        self,
        document: SourceDocument,
        result: PipelineResult,
        *,
        status: ProcessingStatus = ProcessingStatus.COMPLETED,
    ) -> None:
        document.set_extraction_results(
            result.entities_extracted, result.relationships_extracted
        )
        document.update_processing_status(status)
        try:
            await self.document_repo.update(document)
        except Exception as err:
            logger.debug("Document update failed (non-fatal): %s", err)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_embedding_model(self):
        """Return the shared sentence-transformers model from the factory.

        Routed through ``embedding_factory`` so the pipeline, graph repo,
        chunker, and verifier all share one model instance — and so
        switching to SapBERT (or any other model) is an env var, not a
        code change. See infrastructure.services.embedding_factory.
        """
        from ...infrastructure.services.embedding_factory import get_model
        return get_model()

    async def _embed(self, text: str) -> Optional[list]:
        text = (text or "").strip()
        if not text:
            return None
        cached = await self.embedding_cache.get(text)
        if cached is not None:
            await self.metrics.record_embedding(cache_hit=True)
            return cached
        model = self._get_embedding_model()
        if model is None:
            return None
        try:
            vec = (
                await asyncio.get_running_loop().run_in_executor(
                    None, lambda: model.encode(text, convert_to_numpy=True).tolist()
                )
            )
        except Exception as exc:
            logger.debug("Embedding failed for %r: %s", text[:60], exc)
            return None
        await self.embedding_cache.set(text, vec)
        await self.metrics.record_embedding(cache_hit=False)
        return vec

    @staticmethod
    def _raise_if_cancelled(cancel_check: CancelCheck) -> None:
        if cancel_check():
            raise PipelineCancelled()


# ---------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------


def _entity_text(entity: GraphEntity) -> str:
    parts = [entity.name]
    if entity.description:
        parts.append(entity.description)
    return " — ".join(parts)


def _fixed_size_chunks(
    content: str, document_id: str, *, chunk_size: int, overlap: int
) -> List[DocumentChunk]:
    chunks: List[DocumentChunk] = []
    start = 0
    idx = 0
    while start < len(content):
        end = min(start + chunk_size, len(content))
        text = content[start:end].strip()
        if text:
            chunks.append(
                DocumentChunk(
                    content=text,
                    document_id=document_id,
                    chunk_index=idx,
                    token_count=len(text.split()),
                    character_count=len(text),
                    start_position=start,
                    end_position=end,
                )
            )
            idx += 1
        if end >= len(content):
            break
        start = max(end - overlap, start + 1)
    return chunks


async def _fetch_url(url: str) -> str:
    """Fetch a URL and return cleaned text. aiohttp + BeautifulSoup."""
    import aiohttp
    from bs4 import BeautifulSoup

    timeout = aiohttp.ClientTimeout(total=30)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.get(url, headers={"User-Agent": "GraphBuilder/2.1"}) as resp:
            resp.raise_for_status()
            html = await resp.text()
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    return soup.get_text(separator="\n", strip=True)


async def _maybe_await(value: Any) -> None:
    if asyncio.iscoroutine(value) or isinstance(value, Awaitable):
        await value


async def _gather_with_cancel(coros, *, cancel_check: CancelCheck) -> None:
    """Run coros concurrently; if any raises PipelineCancelled, abort."""
    tasks = [asyncio.create_task(c) for c in coros]
    try:
        for fut in asyncio.as_completed(tasks):
            await fut
    except PipelineCancelled:
        for t in tasks:
            if not t.done():
                t.cancel()
        # Drain
        for t in tasks:
            try:
                await t
            except (asyncio.CancelledError, PipelineCancelled, Exception):
                pass
        raise
