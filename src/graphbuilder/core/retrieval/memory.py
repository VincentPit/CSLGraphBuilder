"""Conversation memory — working / summary / episodic layers.

Implements the first three layers of §5 of docs/RAG_QA_PLAN.md so the
chatbot stays coherent across long conversations:

* **Working memory** — last N turns rendered verbatim into the prompt.
* **Rolling summary** — older turns compressed to a paragraph, kept on
  the session so we don't re-summarise on every /ask.
* **Episodic recall** — a vector search over prior turns in the same
  session for the one most relevant to the new question. Surfaces "what
  about its side effects?" by pulling in the original turn that named
  the drug, which the LLM then resolves the pronoun against.

Cross-session semantic memory (§5.4) lands in P14 and isn't here.

The :class:`MemoryService` keeps no per-session state — every call reads
from the conversation repo and writes the new summary back via
``update_session_summary`` so the cost is amortised across turns. When
``MemoryConfig.background_summary_refresh`` is on (the default), a stale
summary's regeneration is dispatched as a *detached* asyncio task so the
answer's critical path never pays the summariser-LLM cost; the current
turn is served the previous (cached) summary, which at most omits the
single most-recent turn — and that turn is already in working memory
verbatim. So the service holds a transient handle to those tasks only
for the duration of the regen.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

from ...domain.models.conversation_models import ConversationTurn
from .models import EmbeddingOrAwaitable, resolve_embedding


logger = logging.getLogger("graphbuilder.qa.memory")


# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------

@dataclass
class MemoryConfig:
    """Tuning knobs for :class:`MemoryService`. Defaults match §5.5."""

    # Working memory.
    working_memory_turns: int = 3
    """Most-recent N turns rendered verbatim into the prompt."""

    max_working_chars: int = 3000
    """Hard cap on the rendered working-memory block. When the verbatim
    turns exceed this, oldest turns are dropped first."""

    # Rolling summary.
    enable_summary: bool = True

    background_summary_refresh: bool = True
    """When True (default), a stale rolling summary is regenerated in a
    detached ``asyncio`` task and *this* turn is served the previous
    (cached) summary — keeping the summariser LLM off the answer's
    critical path. A one-turn-stale summary is harmless: the only turn it
    omits is already rendered verbatim in working memory. Set False for
    deterministic, synchronous behaviour (eval harness, unit tests). Also
    falls back to synchronous regen when no event loop is running."""

    max_summary_chars: int = 2000
    """Cap on the rolling summary block (≈ 500 tokens at 4 chars/token)."""

    summarise_chunk_chars: int = 1200
    """Per-turn snippet fed into the summariser when older turns are
    long. Bigger gives more fidelity at higher cost."""

    # Episodic recall.
    enable_episodic_recall: bool = True
    episodic_top_k: int = 1
    """How many prior turns to pull in. v1 = 1; the prompt already has
    the working memory window so most additions become noise."""

    episodic_min_score: float = 0.65
    """Cosine threshold for accepting an episodic hit."""

    max_episodic_chars: int = 800


# ----------------------------------------------------------------------
# Output
# ----------------------------------------------------------------------

@dataclass
class MemoryContext:
    """Snapshot of memory layers used for one turn.

    The :class:`QAService` consumes ``rendered_block`` directly into the
    prompt; the other fields are surfaced into the retrieval trace and
    metrics for the debug pane.
    """

    working_turns: List[ConversationTurn] = field(default_factory=list)
    rolling_summary: str = ""
    episodic_hit: Optional[Tuple[ConversationTurn, float]] = None
    rendered_block: str = ""

    # Stats for the trace.
    working_chars: int = 0
    summary_chars: int = 0
    episodic_chars: int = 0
    summary_regenerated: bool = False

    def to_trace_dict(self) -> dict:
        return {
            "working_turns": len(self.working_turns),
            "summary_chars": self.summary_chars,
            "episodic_hit": (
                {
                    "turn_id": self.episodic_hit[0].id,
                    "score": round(self.episodic_hit[1], 4),
                }
                if self.episodic_hit else None
            ),
            "summary_regenerated": self.summary_regenerated,
        }


# ----------------------------------------------------------------------
# Service
# ----------------------------------------------------------------------

# Block-header strings as constants so test-asserts don't drift from
# what the LLM actually sees.
_HEADER_SUMMARY = "ROLLING SUMMARY OF EARLIER CONVERSATION"
_HEADER_RECENT = "RECENT TURNS"
_HEADER_EPISODIC = "POSSIBLY RELEVANT EARLIER TURN"


class MemoryService:
    """Builds + persists the memory layers for one conversation turn."""

    def __init__(
        self,
        *,
        conversation_repo: Any,
        llm_service: Optional[Any],
        config: Optional[MemoryConfig] = None,
    ):
        self._conv = conversation_repo
        self._llm = llm_service
        self._cfg = config or MemoryConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def build(
        self,
        *,
        session_id: Optional[str],
        query: str,
        query_embedding: EmbeddingOrAwaitable,
    ) -> MemoryContext:
        """Build the memory context for the upcoming /ask turn.

        Pre-conditions:
        - ``session_id`` may be ``None`` for a brand-new session — we
          short-circuit and return an empty context (nothing to recall).
        - ``query_embedding`` may be ``None`` (embedding model
          unavailable → skip episodic recall, still render working memory
          + summary), a vector, or an *awaitable* producing one. Only
          episodic recall consumes it, so passing an in-flight embed task
          lets the turn-loading + summary work overlap the embedder.
        """
        if not session_id:
            return MemoryContext()

        # Load existing turns so we can split into working / older.
        existing_turns = await self._conv.get_turns_by_session(
            session_id, limit=200, offset=0,
        )
        if not existing_turns:
            return MemoryContext()

        n_working = self._cfg.working_memory_turns
        if n_working <= 0:
            working = []
            older = list(existing_turns)
        else:
            working = list(existing_turns[-n_working:])
            older = list(existing_turns[:-n_working])

        rolling_summary = ""
        summary_regenerated = False
        if older and self._cfg.enable_summary:
            rolling_summary, summary_regenerated = await self._maybe_refresh_summary(
                session_id=session_id, older_turns=older,
            )

        episodic_hit: Optional[Tuple[ConversationTurn, float]] = None
        if self._cfg.enable_episodic_recall:
            # Resolve the embedding only here — nothing above needed it,
            # so it overlapped the (caller-kicked-off) embed task.
            resolved_embedding = await resolve_embedding(query_embedding)
            if resolved_embedding is not None:
                episodic_hit = await self._episodic_recall(
                    session_id=session_id,
                    query_embedding=resolved_embedding,
                    exclude_turn_ids={t.id for t in working},
                )

        rendered_block, stats = self._render(
            rolling_summary=rolling_summary,
            working_turns=working,
            episodic_hit=episodic_hit,
        )

        return MemoryContext(
            working_turns=working,
            rolling_summary=rolling_summary,
            episodic_hit=episodic_hit,
            rendered_block=rendered_block,
            working_chars=stats["working"],
            summary_chars=stats["summary"],
            episodic_chars=stats["episodic"],
            summary_regenerated=summary_regenerated,
        )

    # ------------------------------------------------------------------
    # Rolling summary
    # ------------------------------------------------------------------

    async def _maybe_refresh_summary(
        self,
        *,
        session_id: str,
        older_turns: List[ConversationTurn],
    ) -> Tuple[str, bool]:
        """Return ``(summary_for_this_turn, regenerated_synchronously)``.

        The cached summary lives on ``session.summary``, prefixed with a
        ``[summary covers N turn(s)]`` marker so staleness is a string
        compare — no extra schema field. We "regenerate" whenever the
        number of older turns changed since the last summarise.

        On a stale (or missing) marker the behaviour forks on
        ``MemoryConfig.background_summary_refresh``:

        * **Background (default)** — dispatch the regen as a detached
          task and serve what we already have: the stale cached summary,
          or — for a never-summarised session — a deterministic
          ``_fallback_summary`` stopgap the task will overwrite shortly.
          The summariser LLM never touches this turn's critical path.
        * **Synchronous** — block on the regen (LLM, or deterministic
          fallback when no LLM / on LLM error) and persist it before
          returning. Used when the flag is off, or when there's no event
          loop to attach a task to.

        ``regenerated_synchronously`` is True only for the synchronous
        path; a backgrounded regen leaves it False because *this* turn
        was served the cached/stopgap text.
        """
        session = await self._conv.get_session(session_id)
        cached = (session.summary if session else "") or ""

        marker_for = self._summary_marker(len(older_turns))
        if cached.startswith(marker_for):
            return cached, False

        # Stale or missing marker → a regen is due.
        if self._cfg.background_summary_refresh and self._schedule_summary_refresh(
            session_id, older_turns, marker_for,
        ):
            if cached:
                return cached, False
            # Never summarised before — there's nothing cached to serve,
            # so hand back a cheap deterministic stopgap (capped to the
            # block budget); the background task replaces it next turn.
            stopgap = f"{marker_for}\n{self._fallback_summary(older_turns)}"
            return self._truncate(stopgap, self._cfg.max_summary_chars), False

        # Synchronous path: flag off, or no running loop to background on.
        new_summary = await self._build_summary_text(older_turns, marker_for)
        await self._conv.update_session_summary(session_id, new_summary)
        return new_summary, True

    def _schedule_summary_refresh(
        self,
        session_id: str,
        older_turns: List[ConversationTurn],
        marker_for: str,
    ) -> bool:
        """Fire-and-forget a rolling-summary regen. Returns False when
        there's no running event loop (caller then regenerates inline).

        Overlapping turns on the same session can each spawn a task —
        harmless: ``update_session_summary`` writes are last-wins and the
        marker makes a redundant regen a cheap skip on the next turn. Not
        worth a per-session lock for the single-instance v1 API."""
        coro = self._regenerate_and_persist(session_id, older_turns, marker_for)
        try:
            task = asyncio.create_task(coro)
        except RuntimeError:
            coro.close()  # never scheduled — close it so it's not "never awaited"
            return False
        task.add_done_callback(self._on_summary_refresh_done)
        return True

    @staticmethod
    def _on_summary_refresh_done(task: "asyncio.Task[Any]") -> None:
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            logger.warning("background rolling-summary refresh failed: %s", exc)

    async def _regenerate_and_persist(
        self,
        session_id: str,
        older_turns: List[ConversationTurn],
        marker_for: str,
    ) -> None:
        new_summary = await self._build_summary_text(older_turns, marker_for)
        await self._conv.update_session_summary(session_id, new_summary)

    async def _build_summary_text(
        self, older_turns: List[ConversationTurn], marker_for: str,
    ) -> str:
        """Produce the marker-prefixed summary — LLM when available, a
        deterministic Q/A concat otherwise. Never raises: an LLM failure
        degrades to the concat. Output is capped to the block budget."""
        if self._llm is None:
            text = self._fallback_summary(older_turns)
        else:
            try:
                text = await self._summarise_via_llm(older_turns)
            except Exception as exc:  # noqa: BLE001
                logger.warning("summary LLM call failed: %s — using fallback", exc)
                text = self._fallback_summary(older_turns)
        return self._truncate(f"{marker_for}\n{text}", self._cfg.max_summary_chars)

    @staticmethod
    def _summary_marker(n_turns: int) -> str:
        return f"[summary covers {n_turns} turn{'s' if n_turns != 1 else ''}]"

    def _fallback_summary(self, older: List[ConversationTurn]) -> str:
        """Cheap text-only summary used when no LLM is configured."""
        lines = []
        for t in older:
            q = (t.user_query or "").strip().replace("\n", " ")
            a = (t.llm_answer or "").strip().replace("\n", " ")
            if len(q) > 120:
                q = q[:120].rstrip() + "…"
            if len(a) > 240:
                a = a[:240].rstrip() + "…"
            lines.append(f"- Q: {q}\n  A: {a}")
        return "\n".join(lines)

    async def _summarise_via_llm(self, older: List[ConversationTurn]) -> str:
        snippets = []
        for t in older:
            q = (t.user_query or "").strip()
            a = (t.llm_answer or "").strip()
            cap = self._cfg.summarise_chunk_chars // 2
            if len(q) > cap:
                q = q[:cap].rstrip() + "…"
            if len(a) > cap:
                a = a[:cap].rstrip() + "…"
            snippets.append(f"USER: {q}\nASSISTANT: {a}")

        budget_chars = self._cfg.max_summary_chars - 60  # leave headroom for marker + ellipsis
        budget_tokens = max(120, budget_chars // 4)
        system = (
            "You compress chat history. Produce a 2-4 sentence running "
            "summary that preserves any biomedical entities mentioned "
            "(genes, drugs, diseases, identifiers like CHEMBL/ENSG). Keep "
            "user-stated preferences if any. No preamble, plain text only."
        )
        user = (
            "Summarise the following turns. Stay under "
            f"{budget_chars} characters total.\n\n"
            + "\n\n".join(snippets)
        )
        return await self._llm.generate_text(
            prompt=user,
            system_prompt=system,
            temperature=0.0,
            max_tokens=budget_tokens,
        )

    # ------------------------------------------------------------------
    # Episodic recall
    # ------------------------------------------------------------------

    async def _episodic_recall(
        self,
        *,
        session_id: str,
        query_embedding: List[float],
        exclude_turn_ids: set[str],
    ) -> Optional[Tuple[ConversationTurn, float]]:
        try:
            # Pull a few hits so we can filter out working-memory turns
            # before falling back to a less-relevant one.
            hits = await self._conv.vector_search_turns(
                query_embedding=query_embedding,
                top_k=self._cfg.episodic_top_k + len(exclude_turn_ids),
                min_score=self._cfg.episodic_min_score,
                session_id=session_id,
            )
        except Exception as exc:
            logger.warning("episodic recall vector search failed: %s", exc)
            return None
        for turn, score in hits:
            if turn.id in exclude_turn_ids:
                continue
            return (turn, float(score))
        return None

    # ------------------------------------------------------------------
    # Renderer
    # ------------------------------------------------------------------

    def _render(
        self,
        *,
        rolling_summary: str,
        working_turns: List[ConversationTurn],
        episodic_hit: Optional[Tuple[ConversationTurn, float]],
    ) -> Tuple[str, dict]:
        """Render the memory block for the LLM prompt.

        Returns the rendered string + a small dict of byte counts per
        slot so the orchestrator can log them as ``qa_context_tokens``.
        """
        parts: List[str] = []
        stats = {"working": 0, "summary": 0, "episodic": 0}

        if rolling_summary:
            block = self._truncate(rolling_summary, self._cfg.max_summary_chars)
            stats["summary"] = len(block)
            parts.append(f"{_HEADER_SUMMARY}\n{block}")

        if working_turns:
            rendered = self._render_working_turns(working_turns)
            stats["working"] = len(rendered)
            parts.append(f"{_HEADER_RECENT}\n{rendered}")

        if episodic_hit is not None:
            turn, score = episodic_hit
            block = self._render_one_turn(
                turn, max_chars=self._cfg.max_episodic_chars,
            )
            stats["episodic"] = len(block)
            parts.append(
                f"{_HEADER_EPISODIC} (turn idx {turn.idx}, sim {score:.2f})\n{block}"
            )

        return "\n\n".join(parts), stats

    def _render_working_turns(self, turns: List[ConversationTurn]) -> str:
        """Render the working window oldest→newest, dropping oldest
        turns first if the budget is exceeded."""
        rendered_each = [self._render_one_turn(t) for t in turns]
        joined = "\n\n".join(rendered_each)
        if len(joined) <= self._cfg.max_working_chars:
            return joined
        # Trim from the front.
        while rendered_each and len(joined) > self._cfg.max_working_chars:
            rendered_each.pop(0)
            joined = "\n\n".join(rendered_each)
        return joined

    @staticmethod
    def _render_one_turn(t: ConversationTurn, *, max_chars: Optional[int] = None) -> str:
        q = (t.user_query or "").strip()
        a = (t.llm_answer or "").strip()
        rendered = f"USER: {q}\nASSISTANT: {a}"
        if max_chars is not None and len(rendered) > max_chars:
            rendered = rendered[:max_chars].rstrip() + "…"
        return rendered

    @staticmethod
    def _truncate(text: str, max_chars: int) -> str:
        if len(text) <= max_chars:
            return text
        return text[:max_chars].rstrip() + "…"


__all__ = [
    "MemoryConfig",
    "MemoryContext",
    "MemoryService",
]
