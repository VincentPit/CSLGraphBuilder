"""QA service — ties retrieval, the LLM, and conversation persistence
together into the single ``ask(query)`` flow used by ``/qa/ask``.

v1 scope (P5 of docs/RAG_QA_PLAN.md), now extended for P6+P7:
- Memory layers (working / rolling-summary / episodic recall) build
  in parallel with retrieval and feed into the LLM prompt.
- Per-turn ``query_embedding`` is persisted on the turn so future
  turns in the same session can episodic-recall it.
- No tool-use (P9/P10) yet — the LLM only generates an answer.
- No faithfulness check (P8) yet — the answer is returned as-is, with
  citation IDs extracted for future blending into source confidence.

Each call:
1. Resolves or creates a session (anonymous if no user_id supplied).
2. Embeds the query once.
3. Runs retrieval + memory.build in parallel.
4. Renders a system + memory + sources + question prompt.
5. Calls the LLM service for a free-form answer.
6. Extracts ``[n]`` citation indices from the answer and uses them to
   record which entity / relationship / chunk ids the turn cited.
7. Persists a :class:`ConversationTurn` (with the embedding) and
   returns the bundle.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, List, Optional

from .intent import INTENT_PROFILES, apply_profile, classify_intent
from .memory import MemoryConfig, MemoryContext, MemoryService
from .models import RetrievalConfig, RetrievalTrace, RetrievedItem
from .orchestrator import RetrievalOrchestrator


logger = logging.getLogger("graphbuilder.qa.api")


ANONYMOUS_USER_ID: Optional[str] = None
"""Sentinel for v1 — every unauthenticated chat lives under ``user_id=None``."""


# ----------------------------------------------------------------------
# System prompt + renderers
# ----------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a biomedical knowledge-graph assistant.

Rules:
- Answer ONLY from the SOURCES below. Do not use outside knowledge.
- Cite each factual claim with [n] where n is the 1-indexed source number.
- If the sources do not contain enough information, reply exactly:
  "I cannot find this in the knowledge base."
- Be concise. Prefer 2-5 sentences for short questions, 1 short paragraph
  for "tell me about" requests.
- When the user asks for a list, return a bulleted list. Otherwise prose.
"""


def _render_sources(items: List[RetrievedItem]) -> str:
    """Format the retrieved items as a numbered SOURCES block.

    Source numbering is 1-indexed and stable for the duration of the
    response — citation indices in the LLM's answer map back via the
    same 1-indexed list ``items``.
    """
    if not items:
        return "(no relevant sources found in the knowledge base)"
    lines = []
    for i, item in enumerate(items, start=1):
        head = f"[{i}] {item.kind.value.title()}: {item.label}"
        lines.append(head)
        if item.metadata.get("description"):
            lines.append(f"    Description: {item.metadata['description']}")
        if item.chunk_preview:
            preview = item.chunk_preview.strip().replace("\n", " ")
            if len(preview) > 400:
                preview = preview[:400].rstrip() + "…"
            origin = item.source_doc_id or "unknown"
            lines.append(f"    From {origin}: \"{preview}\"")
    return "\n".join(lines)


def _render_user_prompt(
    query: str,
    items: List[RetrievedItem],
    memory_block: str = "",
) -> str:
    """Stitch the prompt sections together. The memory block (if any)
    sits ABOVE sources so the LLM has conversational context before it
    starts grounding against the retrieved evidence."""
    parts: List[str] = []
    if memory_block:
        parts.append(memory_block)
    parts.extend([
        f"SOURCES\n{_render_sources(items)}",
        f"QUESTION\n{query}",
    ])
    return "\n\n".join(parts)


_CITATION_RE = re.compile(r"\[(\d+)\]")


def _extract_cited_indices(answer: str) -> List[int]:
    """Pull 1-indexed citation numbers out of an answer.

    The LLM emits ``[3]`` style markers; we just collect the unique set
    in first-appearance order. Out-of-range / non-numeric markers are
    filtered upstream by ``QAService`` against the actual source list.
    """
    seen: set[int] = set()
    out: List[int] = []
    for m in _CITATION_RE.finditer(answer or ""):
        try:
            idx = int(m.group(1))
        except ValueError:
            continue
        if idx not in seen:
            seen.add(idx)
            out.append(idx)
    return out


# ----------------------------------------------------------------------
# Result dataclass
# ----------------------------------------------------------------------

@dataclass
class AskResult:
    session_id: str
    turn_id: str
    answer: str
    sources: List[RetrievedItem] = field(default_factory=list)
    cited_source_indices: List[int] = field(default_factory=list)
    retrieval_trace: Optional[RetrievalTrace] = None
    memory_trace: Optional[dict] = None
    request_id: Optional[str] = None
    latency_ms: int = 0


# ----------------------------------------------------------------------
# Service
# ----------------------------------------------------------------------

class QAService:
    """Glues retrieval + memory + LLM + conversation persistence."""

    def __init__(
        self,
        *,
        orchestrator: RetrievalOrchestrator,
        conversation_repo: Any,
        llm_service: Optional[Any],
        config: Optional[RetrievalConfig] = None,
        memory: Optional[MemoryService] = None,
        memory_config: Optional[MemoryConfig] = None,
    ):
        self._orch = orchestrator
        self._conv = conversation_repo
        self._llm = llm_service
        self._cfg = config or RetrievalConfig()
        self._memory = memory or MemoryService(
            conversation_repo=conversation_repo,
            llm_service=llm_service,
            config=memory_config or MemoryConfig(),
        )

    async def ask(
        self,
        *,
        query: str,
        session_id: Optional[str] = None,
        user_id: Optional[str] = ANONYMOUS_USER_ID,
        top_k: Optional[int] = None,
        retrieval_override: Optional[RetrievalConfig] = None,
    ) -> AskResult:
        wall_start = time.perf_counter()
        if not (query or "").strip():
            raise ValueError("query must not be empty")

        # Lazy import so the service module is importable in environments
        # that don't yet have the rest of the qa observability stack.
        from ...infrastructure.services.qa_observability import get_request_id
        request_id = get_request_id()

        session = await self._resolve_session(session_id, user_id)

        # Embed once and share across retrieval + memory recall so we
        # don't pay the model cost twice on a single turn.
        query_embedding = await self._embed_query(query)

        # Pick the per-intent retrieval config. An explicit
        # ``retrieval_override`` wins so the eval harness's ablation
        # hook stays honest — when a caller asks for a specific config,
        # we don't fuzz it through the classifier. Otherwise classify
        # the query and apply the matching profile over the base cfg.
        if retrieval_override is not None:
            intent_label: Optional[str] = None
            cfg_for_turn: Optional[RetrievalConfig] = retrieval_override
        else:
            intent_label = classify_intent(query)
            cfg_for_turn = apply_profile(self._cfg, INTENT_PROFILES[intent_label])

        # Retrieval and memory build are independent — gather them.
        items_trace_task = self._orch.retrieve(
            query, top_k=top_k, query_embedding=query_embedding,
            config_override=cfg_for_turn,
        )
        memory_task = self._memory.build(
            session_id=session.id,
            query=query,
            query_embedding=query_embedding,
        )
        (items, trace), memory_ctx = await asyncio.gather(
            items_trace_task, memory_task,
        )

        # Stamp the chosen intent on the trace so the eval harness, the
        # debug pane, and structured logs can all see which profile
        # actually drove this turn. ``None`` when override bypassed
        # routing — keeps the field honest.
        if intent_label is not None:
            trace.intent = intent_label

        answer = await self._generate_answer(query, items, memory_ctx)
        cited_indices = [
            i for i in _extract_cited_indices(answer)
            if 1 <= i <= len(items)
        ]

        cited_entity_ids: List[str] = []
        cited_rel_ids: List[str] = []
        cited_chunk_ids: List[str] = []
        for idx in cited_indices:
            item = items[idx - 1]
            if item.kind.value == "entity":
                cited_entity_ids.append(item.id)
            elif item.kind.value == "relationship":
                cited_rel_ids.append(item.id)
            if item.source_chunk_id and item.source_chunk_id not in cited_chunk_ids:
                cited_chunk_ids.append(item.source_chunk_id)

        turn = await self._append_turn(
            session_id=session.id,
            query=query,
            answer=answer,
            request_id=request_id,
            cited_entity_ids=cited_entity_ids,
            cited_rel_ids=cited_rel_ids,
            cited_chunk_ids=cited_chunk_ids,
            query_embedding=query_embedding,
        )

        latency_ms = int((time.perf_counter() - wall_start) * 1000)

        # qa_request metric: now labeled by chosen intent (or "any" when
        # routing was bypassed via retrieval_override). Cardinality is
        # bounded — the classifier returns one of three values plus the
        # "any" fallback.
        await self._record_request_metric(status="ok", intent=intent_label or "any")
        await self._record_total_latency(latency_ms)
        await self._record_memory_tokens(memory_ctx)

        logger.info(
            "ask done session=%s turn=%s intent=%s query=%r sources=%d cited=%d "
            "memory(working=%d summary=%d episodic=%s) latency_ms=%d",
            session.id, turn.id, intent_label or "override", query[:100],
            len(items), len(cited_indices),
            len(memory_ctx.working_turns), memory_ctx.summary_chars,
            (memory_ctx.episodic_hit[0].id if memory_ctx.episodic_hit else None),
            latency_ms,
        )

        return AskResult(
            session_id=session.id,
            turn_id=turn.id,
            answer=answer,
            sources=items,
            cited_source_indices=cited_indices,
            retrieval_trace=trace,
            memory_trace=memory_ctx.to_trace_dict(),
            request_id=request_id,
            latency_ms=latency_ms,
        )

    # ------------------------------------------------------------------
    # Embedding helper
    # ------------------------------------------------------------------

    async def _embed_query(self, query: str) -> Optional[List[float]]:
        try:
            from ...infrastructure.services.embedding_factory import embed_async
        except Exception:
            return None
        try:
            return await embed_async(query)
        except Exception as exc:
            logger.debug("ask: query embedding failed: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Session helpers
    # ------------------------------------------------------------------

    async def _resolve_session(self, session_id: Optional[str], user_id: Optional[str]):
        from ...domain.models.conversation_models import ConversationSession

        if session_id:
            existing = await self._conv.get_session(session_id)
            if existing is not None:
                return existing
            # If the caller named a session that doesn't exist, surface a
            # clear error rather than silently creating a fresh one — they
            # may be referring to a session that was deleted.
            raise LookupError(f"session not found: {session_id}")

        new_session = ConversationSession(
            id=f"session_{uuid.uuid4().hex[:16]}",
            user_id=user_id if user_id is not None else ANONYMOUS_USER_ID,
        )
        return await self._conv.create_session(new_session)

    async def _append_turn(
        self,
        *,
        session_id: str,
        query: str,
        answer: str,
        request_id: Optional[str],
        cited_entity_ids: List[str],
        cited_rel_ids: List[str],
        cited_chunk_ids: List[str],
        query_embedding: Optional[List[float]] = None,
    ):
        from ...domain.models.conversation_models import ConversationTurn

        # idx = current turn count (atomic enough for v1 single-instance API).
        session = await self._conv.get_session(session_id)
        idx = session.turn_count if session else 0
        turn = ConversationTurn(
            session_id=session_id,
            idx=idx,
            user_query=query,
            llm_answer=answer,
            request_id=request_id,
            cited_entity_ids=cited_entity_ids,
            cited_relationship_ids=cited_rel_ids,
            cited_chunk_ids=cited_chunk_ids,
            query_embedding=query_embedding,
        )
        return await self._conv.append_turn(turn)

    # ------------------------------------------------------------------
    # LLM
    # ------------------------------------------------------------------

    async def _generate_answer(
        self,
        query: str,
        items: List[RetrievedItem],
        memory_ctx: MemoryContext,
    ) -> str:
        if self._llm is None:
            # Graceful degradation — if no LLM is configured, surface the
            # retrieved context so the user at least sees what would have
            # been used. The frontend can still render sources + confidence.
            return (
                "(no LLM configured — retrieval returned "
                f"{len(items)} source(s); see the Sources panel)"
            )
        prompt = _render_user_prompt(query, items, memory_ctx.rendered_block)
        t0 = time.perf_counter()
        try:
            answer = await self._llm.generate_text(
                prompt=prompt,
                system_prompt=_SYSTEM_PROMPT,
                temperature=0.1,
                max_tokens=1024,
            )
        except Exception as exc:
            logger.error("LLM call failed: %s", exc, exc_info=True)
            return f"(LLM call failed: {exc})"
        latency = time.perf_counter() - t0
        await self._record_llm_latency(latency)
        return answer

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    async def _record_request_metric(self, *, status: str, intent: str = "any") -> None:
        try:
            from ...infrastructure.services.metrics import get_metrics
        except Exception:
            return
        await get_metrics().record_qa_request(intent=intent, status=status)

    async def _record_total_latency(self, latency_ms: int) -> None:
        try:
            from ...infrastructure.services.metrics import get_metrics
        except Exception:
            return
        await get_metrics().record_qa_latency(phase="total", seconds=latency_ms / 1000.0)

    async def _record_llm_latency(self, seconds: float) -> None:
        try:
            from ...infrastructure.services.metrics import get_metrics
        except Exception:
            return
        await get_metrics().record_qa_latency(phase="llm", seconds=seconds)

    async def _record_memory_tokens(self, ctx: MemoryContext) -> None:
        """Surface the per-slot character counts as ``qa_context_tokens``
        histograms. We use a coarse 4-chars-per-token heuristic for the
        token-axis since the actual tokeniser depends on the LLM provider."""
        try:
            from ...infrastructure.services.metrics import get_metrics
        except Exception:
            return
        m = get_metrics()
        if ctx.working_chars:
            await m.record_qa_context_tokens(slot="working", tokens=ctx.working_chars // 4)
        if ctx.summary_chars:
            await m.record_qa_context_tokens(slot="summary", tokens=ctx.summary_chars // 4)
        if ctx.episodic_chars:
            await m.record_qa_context_tokens(slot="episodic", tokens=ctx.episodic_chars // 4)


__all__ = [
    "ANONYMOUS_USER_ID",
    "AskResult",
    "MemoryConfig",
    "MemoryContext",
    "MemoryService",
    "QAService",
]
