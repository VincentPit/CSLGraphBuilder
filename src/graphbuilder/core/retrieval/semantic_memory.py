"""Cross-session semantic memory — §5.4 / P14 of docs/RAG_QA_PLAN.md.

Extends the per-session memory layers (working / rolling-summary /
episodic) with a *per-user* persona summary that survives across
sessions. The flow:

1. **Load** (every turn): :meth:`SemanticMemoryService.load_persona`
   fetches the cached persona text from ``User.metadata`` and the
   :class:`QAService` splices it into the system prompt above SOURCES.
   Cheap — just a dict read on an existing user record.

2. **Refresh** (rare): :meth:`refresh_persona` re-summarises the user's
   recent sessions into a single persona string + an embedding. Called:

   - synchronously on session **delete** (we have the turns in hand,
     and the curator-style "I'm done with this conversation" event is
     the clearest "session ended" signal), and
   - in the background when a **new** session starts for a user who has
     accumulated more completed sessions than the cached persona covers.

Storage lives on ``User.metadata`` (no schema migration needed — the
metadata bag was reserved for exactly this in user_models.py:36):

    metadata = {
        "semantic_summary": str,                  # the persona text
        "semantic_embedding": List[float],        # for cross-session vector recall (future use)
        "semantic_summary_covers_sessions": int,  # freshness marker — mirrors the
                                                  #   "[summary covers N turns]" trick from §5.2
        "semantic_summary_updated_at": str,       # ISO timestamp for debugging
    }

The service degrades gracefully — anonymous traffic (``user_id=None``),
missing LLM, or summary failure all return an empty persona so the QA
loop keeps working.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, List, Optional

from ...domain.models.conversation_models import ConversationTurn


logger = logging.getLogger("graphbuilder.qa.semantic_memory")


# Metadata keys — kept here so tests + the renderer don't drift apart.
META_SUMMARY = "semantic_summary"
META_EMBEDDING = "semantic_embedding"
META_COVERS = "semantic_summary_covers_sessions"
META_UPDATED_AT = "semantic_summary_updated_at"


@dataclass
class SemanticMemoryConfig:
    """Tuning knobs for :class:`SemanticMemoryService`. Defaults match §5.4."""

    enable: bool = True
    """Master switch — turn off to disable both load + refresh."""

    max_summary_chars: int = 1200
    """Cap on the persona text. ≈300 tokens at 4 chars/token leaves
    headroom in the 1.5k system-prompt budget from §5.5."""

    max_sessions_per_refresh: int = 5
    """How many recent sessions to feed into the summariser. More gives
    a richer persona but blows up token cost — five sessions × dozens of
    turns is already comfortably in the LLM's window."""

    max_turns_per_session: int = 8
    """Per-session turn cap fed into the summariser. The turns are
    rendered Q/A-style; this keeps a chatty session from dominating the
    persona prompt."""

    min_turns_to_summarise: int = 2
    """Skip sessions with fewer turns than this — a one-turn session is
    usually a "test" or an abandoned chat and contributes more noise
    than signal."""

    min_new_sessions_to_refresh: int = 1
    """Background refresh triggers only when the user has at least this
    many completed sessions beyond what the cached persona covers."""


# ----------------------------------------------------------------------
# Service
# ----------------------------------------------------------------------

_SUMMARISER_SYSTEM = (
    "You compress a user's chat history into a 2-4 sentence persona "
    "summary for a biomedical knowledge-graph assistant. Capture: their "
    "apparent role (clinician / researcher / student), recurring topics "
    "(specific genes, drugs, diseases, identifiers), and any stated "
    "preferences (answer depth, format). Use neutral third-person, no "
    "preamble, plain text only."
)


class SemanticMemoryService:
    """Loads + refreshes the per-user persona summary."""

    def __init__(
        self,
        *,
        user_repo: Any,
        conversation_repo: Any,
        llm_service: Optional[Any],
        config: Optional[SemanticMemoryConfig] = None,
    ):
        self._users = user_repo
        self._conv = conversation_repo
        self._llm = llm_service
        self._cfg = config or SemanticMemoryConfig()

    # ------------------------------------------------------------------
    # Read path — splice into the system prompt
    # ------------------------------------------------------------------

    async def load_persona(self, user_id: Optional[str]) -> str:
        """Return the cached persona text, or ``""`` if unavailable.

        Anonymous (``user_id is None``) and disabled-by-config callers
        get an empty string — :class:`QAService` then renders no
        ``USER SEMANTIC SUMMARY`` block.
        """
        if not self._cfg.enable or not user_id:
            return ""
        try:
            user = await self._users.get_user(user_id)
        except Exception as exc:  # noqa: BLE001
            logger.debug("load_persona: get_user failed: %s", exc)
            return ""
        if user is None:
            return ""
        text = user.metadata.get(META_SUMMARY) if user.metadata else None
        if not isinstance(text, str):
            return ""
        return text.strip()

    # ------------------------------------------------------------------
    # Write path — refresh
    # ------------------------------------------------------------------

    async def refresh_persona(
        self,
        user_id: Optional[str],
        *,
        force: bool = False,
        include_session_ids: Optional[List[str]] = None,
    ) -> Optional[str]:
        """Recompute the persona summary for ``user_id``.

        Returns the new summary text, or ``None`` if no refresh ran
        (user unknown, no eligible sessions, freshness not met, or
        feature disabled).

        ``force=True`` skips the "new sessions since cache" check —
        used by the explicit session-delete trigger where we always
        want the just-ended session to flow into the persona.

        ``include_session_ids`` lets a caller pin in extra sessions
        whose turns might not have landed in the repo yet (used by the
        delete-trigger: we fetch the to-be-deleted session's turns
        first and pass them through). Without it, the service relies
        on whatever the conversation repo already exposes.
        """
        if not self._cfg.enable or not user_id:
            return None

        user = await self._users.get_user(user_id)
        if user is None:
            return None

        sessions = await self._conv.list_sessions(
            user_id=user_id, limit=self._cfg.max_sessions_per_refresh,
        )
        # Bring in any extras explicitly requested (e.g. a session about
        # to be deleted that already exists in list_sessions output, or
        # in theory one being pinned from a different path). Dedup by id.
        if include_session_ids:
            seen_ids = {s.id for s in sessions}
            for sid in include_session_ids:
                if sid in seen_ids:
                    continue
                extra = await self._conv.get_session(sid)
                if extra is not None:
                    sessions.append(extra)
                    seen_ids.add(sid)

        if not sessions:
            return None

        # Freshness gate — bail if the cached persona already covers as
        # many sessions as we'd feed it now, unless the caller forces.
        cached_covers = 0
        if user.metadata:
            try:
                cached_covers = int(user.metadata.get(META_COVERS) or 0)
            except (TypeError, ValueError):
                cached_covers = 0
        if not force:
            min_new = max(1, self._cfg.min_new_sessions_to_refresh)
            if len(sessions) < cached_covers + min_new:
                return None

        # Pull turns per session. Sessions with too few turns get
        # filtered — they're usually abandoned chats and add noise.
        session_blocks: List[str] = []
        used_session_count = 0
        for sess in sessions:
            turns = await self._conv.get_turns_by_session(
                sess.id, limit=self._cfg.max_turns_per_session, offset=0,
            )
            if len(turns) < self._cfg.min_turns_to_summarise:
                continue
            session_blocks.append(self._render_session_block(sess.id, turns))
            used_session_count += 1

        if not session_blocks:
            return None

        existing_persona = ""
        if user.metadata:
            existing_persona = (user.metadata.get(META_SUMMARY) or "").strip()

        new_summary = await self._summarise(
            existing_persona=existing_persona,
            session_blocks=session_blocks,
        )
        if not new_summary:
            return None

        # Embed the new persona for future cross-session recall. Failure
        # here doesn't block the text-only persona path.
        embedding = await self._embed(new_summary)

        await self._users.update_user(
            user_id,
            metadata={
                META_SUMMARY: new_summary,
                META_EMBEDDING: embedding or [],
                META_COVERS: used_session_count,
                META_UPDATED_AT: datetime.now(timezone.utc).isoformat(),
            },
        )
        logger.info(
            "semantic memory refreshed user=%s sessions=%d chars=%d embed_dim=%s",
            user_id, used_session_count, len(new_summary),
            len(embedding) if embedding else 0,
        )
        return new_summary

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _render_session_block(
        self, session_id: str, turns: List[ConversationTurn],
    ) -> str:
        lines = [f"=== SESSION {session_id} ==="]
        cap = self._cfg.max_summary_chars  # tight per-turn cap → keeps prompt sane
        for t in turns:
            q = (t.user_query or "").strip().replace("\n", " ")
            a = (t.llm_answer or "").strip().replace("\n", " ")
            if len(q) > cap:
                q = q[:cap].rstrip() + "…"
            if len(a) > cap:
                a = a[:cap].rstrip() + "…"
            lines.append(f"USER: {q}\nASSISTANT: {a}")
        return "\n".join(lines)

    async def _summarise(
        self, *, existing_persona: str, session_blocks: List[str],
    ) -> str:
        """LLM-driven persona refresh with a deterministic fallback."""
        if self._llm is None:
            return self._fallback_summary(
                existing_persona=existing_persona,
                session_blocks=session_blocks,
            )
        budget_chars = max(200, self._cfg.max_summary_chars - 40)
        budget_tokens = max(120, budget_chars // 4)

        prompt_parts: List[str] = []
        if existing_persona:
            prompt_parts.append(
                "EXISTING PERSONA (carry forward unless contradicted):\n"
                + existing_persona
            )
        prompt_parts.append(
            "RECENT SESSIONS:\n" + "\n\n".join(session_blocks)
        )
        prompt_parts.append(
            "Produce the updated persona summary. Stay under "
            f"{budget_chars} characters."
        )
        try:
            text = await self._llm.generate_text(
                prompt="\n\n".join(prompt_parts),
                system_prompt=_SUMMARISER_SYSTEM,
                temperature=0.0,
                max_tokens=budget_tokens,
            )
        except Exception as exc:
            logger.warning("semantic memory LLM call failed: %s — using fallback", exc)
            text = self._fallback_summary(
                existing_persona=existing_persona,
                session_blocks=session_blocks,
            )
        text = (text or "").strip()
        if len(text) > self._cfg.max_summary_chars:
            text = text[: self._cfg.max_summary_chars].rstrip() + "…"
        return text

    def _fallback_summary(
        self, *, existing_persona: str, session_blocks: List[str],
    ) -> str:
        """No-LLM path: keep existing persona + tag the new session ids
        that contributed. Better than nothing for hermetic tests."""
        n = len(session_blocks)
        base = (existing_persona or "User profile unavailable.").strip()
        addendum = f"Activity covers {n} recent session{'s' if n != 1 else ''}."
        joined = f"{base} {addendum}".strip()
        if len(joined) > self._cfg.max_summary_chars:
            joined = joined[: self._cfg.max_summary_chars].rstrip() + "…"
        return joined

    async def _embed(self, text: str) -> Optional[List[float]]:
        if not text:
            return None
        try:
            from ...infrastructure.services.embedding_factory import embed_async
        except Exception:
            return None
        try:
            return await embed_async(text)
        except Exception as exc:
            logger.debug("semantic memory: embed failed: %s", exc)
            return None


__all__ = [
    "META_COVERS",
    "META_EMBEDDING",
    "META_SUMMARY",
    "META_UPDATED_AT",
    "SemanticMemoryConfig",
    "SemanticMemoryService",
]
