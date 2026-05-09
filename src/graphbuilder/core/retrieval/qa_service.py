"""QA service — ties retrieval, the LLM, and conversation persistence
together into the single ``ask(query)`` flow used by ``/qa/ask``.

v1 scope (P5 of docs/RAG_QA_PLAN.md):
- No memory layers yet (P6/P7) — prior turns are persisted but not
  injected into the prompt.
- No tool-use (P9/P10) — the LLM only generates an answer.
- No faithfulness check (P8) — the answer is returned as-is, with
  citation IDs extracted for future blending into source confidence.

Each call:
1. Resolves or creates a session (anonymous if no user_id supplied).
2. Runs the :class:`RetrievalOrchestrator` to produce sources.
3. Renders a system + sources + question prompt.
4. Calls the LLM service for a free-form answer.
5. Extracts ``[n]`` citation indices from the answer and uses them to
   record which entity / relationship / chunk ids the turn cited.
6. Persists a :class:`ConversationTurn` and returns the bundle.
"""

from __future__ import annotations

import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, List, Optional

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


def _render_user_prompt(query: str, items: List[RetrievedItem]) -> str:
    return (
        "SOURCES\n"
        f"{_render_sources(items)}\n\n"
        "QUESTION\n"
        f"{query}"
    )


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
    request_id: Optional[str] = None
    latency_ms: int = 0


# ----------------------------------------------------------------------
# Service
# ----------------------------------------------------------------------

class QAService:
    """Glues retrieval + LLM + conversation persistence."""

    def __init__(
        self,
        *,
        orchestrator: RetrievalOrchestrator,
        conversation_repo: Any,
        llm_service: Optional[Any],
        config: Optional[RetrievalConfig] = None,
    ):
        self._orch = orchestrator
        self._conv = conversation_repo
        self._llm = llm_service
        self._cfg = config or RetrievalConfig()

    async def ask(
        self,
        *,
        query: str,
        session_id: Optional[str] = None,
        user_id: Optional[str] = ANONYMOUS_USER_ID,
        top_k: Optional[int] = None,
    ) -> AskResult:
        wall_start = time.perf_counter()
        if not (query or "").strip():
            raise ValueError("query must not be empty")

        # Lazy import so the service module is importable in environments
        # that don't yet have the rest of the qa observability stack.
        from ...infrastructure.services.qa_observability import get_request_id
        request_id = get_request_id()

        session = await self._resolve_session(session_id, user_id)

        items, trace = await self._orch.retrieve(query, top_k=top_k)

        answer = await self._generate_answer(query, items)
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
        )

        latency_ms = int((time.perf_counter() - wall_start) * 1000)

        # qa_request metric — (intent="any", status="ok") as v1 baseline.
        await self._record_request_metric(status="ok")
        await self._record_total_latency(latency_ms)

        logger.info(
            "ask done session=%s turn=%s query=%r sources=%d cited=%d latency_ms=%d",
            session.id, turn.id, query[:100], len(items), len(cited_indices), latency_ms,
        )

        return AskResult(
            session_id=session.id,
            turn_id=turn.id,
            answer=answer,
            sources=items,
            cited_source_indices=cited_indices,
            retrieval_trace=trace,
            request_id=request_id,
            latency_ms=latency_ms,
        )

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
        )
        return await self._conv.append_turn(turn)

    # ------------------------------------------------------------------
    # LLM
    # ------------------------------------------------------------------

    async def _generate_answer(self, query: str, items: List[RetrievedItem]) -> str:
        if self._llm is None:
            # Graceful degradation — if no LLM is configured, surface the
            # retrieved context so the user at least sees what would have
            # been used. The frontend can still render sources + confidence.
            return (
                "(no LLM configured — retrieval returned "
                f"{len(items)} source(s); see the Sources panel)"
            )
        prompt = _render_user_prompt(query, items)
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

    async def _record_request_metric(self, *, status: str) -> None:
        try:
            from ...infrastructure.services.metrics import get_metrics
        except Exception:
            return
        await get_metrics().record_qa_request(intent="any", status=status)

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


__all__ = [
    "ANONYMOUS_USER_ID",
    "AskResult",
    "QAService",
]
