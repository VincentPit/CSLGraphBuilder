"""QA service — ties retrieval, the LLM, and conversation persistence
together into the single ``ask(query)`` flow used by ``/qa/ask``.

v1 scope (P5 of docs/RAG_QA_PLAN.md), now extended for P6–P8 + P11:
- Memory layers (working / rolling-summary / episodic recall) build
  in parallel with retrieval and feed into the LLM prompt.
- Per-turn ``query_embedding`` is persisted on the turn so future
  turns in the same session can episodic-recall it.
- No tool-use (P9/P10) yet — the LLM only generates an answer.
- Faithfulness check (P8) runs after generation; per-claim confidence
  + an aggregate ``answer_faithfulness`` score ride along on the
  result for the frontend's yellow-underline UI and the eval harness.
- Streaming counterpart ``ask_stream`` (P11) yields the same data as
  ``ask`` but as a sequence of SSE-shaped events: phase / retrieval /
  delta / done / error. Same retrieval + memory + faithfulness wiring;
  the LLM step uses the provider's stream API when available.

Each call:
1. Resolves or creates a session (anonymous if no user_id supplied).
2. Embeds the query once.
3. Runs retrieval + memory.build in parallel.
4. Renders a system + memory + sources + question prompt.
5. Calls the LLM service for a free-form answer.
6. Extracts ``[n]`` citation indices from the answer and uses them to
   record which entity / relationship / chunk ids the turn cited.
7. Runs the faithfulness checker against the cited sources.
8. Persists a :class:`ConversationTurn` (with the embedding) and
   returns the bundle.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Optional

from .faithfulness import (
    FaithfulnessChecker,
    FaithfulnessConfig,
    FaithfulnessResult,
)
from .intent import INTENT_PROFILES, apply_profile, classify_intent
from .memory import MemoryConfig, MemoryContext, MemoryService
from .models import RetrievalConfig, RetrievalTrace, RetrievedItem
from .orchestrator import RetrievalOrchestrator
from .mutation_tools import MutationToolDispatcher, is_mutation
from .tools import ToolCallRecord, ToolDispatcher


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

_TOOLS_PROMPT_SUFFIX = """

TOOLS
You may call the listed tools to gather more information before answering:
- search_graph: free-text search over the knowledge graph.
- get_entity: fetch one entity (by id) + its 1-hop neighbourhood.
- verify_claim: score how well a context supports a claim.

Never invent entity ids — first call search_graph or get_entity to confirm a
target exists. Call tools sparingly; the SOURCES block is usually enough.
When you have what you need, produce the final answer in the usual format
(no further tool calls). If the tools return nothing useful, reply with the
configured refusal phrase rather than guessing.
"""


_MUTATION_PROMPT_SUFFIX = """

MUTATING TOOLS
You may also propose graph changes when the user explicitly asks for them:
- propose_entity / propose_relationship: create new content
- update_entity: patch fields on an existing entity
- merge_entities: collapse two duplicate entities
- soft_delete_entity / soft_delete_relationship: hide rejected content

Mutating tools never apply directly — they queue a proposal that a human
curator must approve. Always confirm the user's intent before calling one.
Tell the user the proposal_id you got back so they can track the review.
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
    # Per-claim faithfulness verdicts + the aggregate score (P8). The
    # eval harness reads ``faithfulness.overall_score`` to populate
    # ``answer_faithfulness``; ``None`` means the answer had no
    # scorable cited claims (e.g. an LLM that didn't cite anything).
    faithfulness: Optional[FaithfulnessResult] = None
    # Read-only tool calls the LLM made during this turn (P9). Empty
    # list when tool-use is disabled or the LLM didn't reach for a
    # tool. Each record has the tool name, args, result/error, and
    # latency — see ToolCallRecord.
    tool_calls: List[ToolCallRecord] = field(default_factory=list)


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
        faithfulness: Optional[FaithfulnessChecker] = None,
        faithfulness_config: Optional[FaithfulnessConfig] = None,
        tool_dispatcher: Optional[ToolDispatcher] = None,
        max_tool_calls_per_turn: int = 5,
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
        # Faithfulness checker is constructed eagerly so production
        # traffic never pays the import on the hot path. The default
        # config is lexical-only; flip ``enable_llm_escalation`` on
        # the config to opt into LLM verdicts on borderline claims.
        self._faithfulness = faithfulness or FaithfulnessChecker(
            config=faithfulness_config or FaithfulnessConfig(),
            llm_service=llm_service,
        )
        # P9 — tool-use is opt-in per request. When the caller passes a
        # dispatcher we can run the agentic loop; without one, ``ask``
        # falls back to the single-call generate path. The cap mirrors
        # §7.7's ``RAGToolConfig.max_tool_calls_per_turn`` so a
        # runaway loop can't fan out forever.
        self._tools = tool_dispatcher
        # P10 — mutating tools are a separate dispatcher so the read-
        # only and write surfaces can be enabled / disabled
        # independently. Wired up via ``set_mutation_dispatcher`` from
        # the API layer (the dispatcher needs access to the api/
        # proposed-mutation store, which core/ shouldn't import).
        self._mutation_tools: Optional[MutationToolDispatcher] = None
        self._max_tool_calls = max(0, int(max_tool_calls_per_turn))

    def set_mutation_dispatcher(
        self, dispatcher: Optional[MutationToolDispatcher],
    ) -> None:
        """Wire the mutating tool dispatcher post-construction (P10).

        The dispatcher needs the api/ proposed-mutation store, which
        core/ shouldn't import — so the API layer constructs it and
        injects via this setter rather than via ``__init__``.
        """
        self._mutation_tools = dispatcher

    async def ask(
        self,
        *,
        query: str,
        session_id: Optional[str] = None,
        user_id: Optional[str] = ANONYMOUS_USER_ID,
        top_k: Optional[int] = None,
        retrieval_override: Optional[RetrievalConfig] = None,
        enable_tools: bool = False,
        enable_mutations: bool = False,
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

        tool_calls: List[ToolCallRecord] = []
        # The agentic loop runs whenever EITHER tool surface is
        # enabled. Read-only and mutating dispatchers are merged
        # inside the loop; the LLM picks any tool from the union, and
        # we route by name. With both flags off we keep the original
        # single-shot path for zero overhead.
        agentic_active = self._llm is not None and (
            (enable_tools and self._tools is not None)
            or (enable_mutations and self._mutation_tools is not None)
        )
        if agentic_active:
            answer, tool_calls = await self._generate_answer_agentic(
                query, items, memory_ctx,
                enable_tools=enable_tools,
                enable_mutations=enable_mutations,
                proposer_user_id=user_id,
            )
        else:
            answer = await self._generate_answer(query, items, memory_ctx)

        cited_indices = [
            i for i in _extract_cited_indices(answer)
            if 1 <= i <= len(items)
        ]

        cited_entity_ids, cited_rel_ids, cited_chunk_ids = self._collect_cited_ids(
            items, cited_indices,
        )

        # Faithfulness check (P8). Runs against the same 1-indexed
        # source list the LLM saw so [n] markers line up. Lexical-only
        # by default — borderline-band escalation to the LLM is opt-in
        # via FaithfulnessConfig.enable_llm_escalation.
        faithfulness_result = await self._check_faithfulness(answer, items)

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
        if faithfulness_result and faithfulness_result.failed_claims:
            await self._record_faithfulness_failures(faithfulness_result.failed_claims)

        logger.info(
            "ask done session=%s turn=%s intent=%s query=%r sources=%d cited=%d "
            "memory(working=%d summary=%d episodic=%s) faithfulness=%s "
            "latency_ms=%d",
            session.id, turn.id, intent_label or "override", query[:100],
            len(items), len(cited_indices),
            len(memory_ctx.working_turns), memory_ctx.summary_chars,
            (memory_ctx.episodic_hit[0].id if memory_ctx.episodic_hit else None),
            (
                f"{faithfulness_result.overall_score:.2f}"
                if faithfulness_result and faithfulness_result.overall_score is not None
                else "n/a"
            ),
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
            faithfulness=faithfulness_result,
            tool_calls=tool_calls,
        )

    # ------------------------------------------------------------------
    # Streaming variant (P11)
    # ------------------------------------------------------------------
    #
    # Same flow as ``ask`` but yields a sequence of structured event
    # dicts rather than building one ``AskResult``. The router maps
    # each event dict to one SSE frame:
    #
    #   - ``phase``        — coarse progress signal ("retrieving",
    #                        "generating") so the frontend can swap
    #                        spinner copy without parsing details.
    #   - ``retrieval``    — sources + retrieval_trace + memory_trace,
    #                        emitted once retrieval+memory complete.
    #   - ``delta``        — token chunk(s) from the LLM. May fire many
    #                        times per turn.
    #   - ``done``         — final event: turn_id, session_id,
    #                        cited_source_indices, faithfulness, total
    #                        latency. Indicates the stream is closed.
    #   - ``error``        — emitted in place of ``done`` when the call
    #                        fails. The frontend should surface the
    #                        ``error.message`` and stop reading.
    #
    # Persistence + metrics + faithfulness all run inside the generator
    # so a client that hangs up mid-stream still triggers the cleanup
    # path via the ``finally`` blocks below.

    async def ask_stream(
        self,
        *,
        query: str,
        session_id: Optional[str] = None,
        user_id: Optional[str] = ANONYMOUS_USER_ID,
        top_k: Optional[int] = None,
        retrieval_override: Optional[RetrievalConfig] = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        """Yield streaming events for a single ``/qa/ask/stream`` turn."""
        wall_start = time.perf_counter()
        if not (query or "").strip():
            raise ValueError("query must not be empty")

        from ...infrastructure.services.qa_observability import get_request_id
        request_id = get_request_id()

        try:
            session = await self._resolve_session(session_id, user_id)
        except LookupError as exc:
            yield {"event": "error", "data": {"message": str(exc), "kind": "session_not_found"}}
            return

        # Tell the client retrieval has started before we await anything
        # expensive — keeps the spinner snappy.
        yield {"event": "phase", "data": {"phase": "retrieving", "request_id": request_id}}

        query_embedding = await self._embed_query(query)

        if retrieval_override is not None:
            intent_label: Optional[str] = None
            cfg_for_turn: Optional[RetrievalConfig] = retrieval_override
        else:
            intent_label = classify_intent(query)
            cfg_for_turn = apply_profile(self._cfg, INTENT_PROFILES[intent_label])

        items_trace_task = self._orch.retrieve(
            query, top_k=top_k, query_embedding=query_embedding,
            config_override=cfg_for_turn,
        )
        memory_task = self._memory.build(
            session_id=session.id,
            query=query,
            query_embedding=query_embedding,
        )
        try:
            (items, trace), memory_ctx = await asyncio.gather(
                items_trace_task, memory_task,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("ask_stream retrieval failed: %s", exc, exc_info=True)
            yield {"event": "error", "data": {"message": str(exc), "kind": "retrieval_failed"}}
            return

        if intent_label is not None:
            trace.intent = intent_label

        # Snapshot retrieval result so the frontend can render source
        # cards + the retrieval-trace panel before the LLM finishes.
        yield {
            "event": "retrieval",
            "data": {
                "sources": [it.to_dict() for it in items],
                "retrieval_trace": trace.to_dict(),
                "memory_trace": memory_ctx.to_trace_dict(),
                "intent": intent_label,
            },
        }

        # Generation phase — actual token streaming below.
        yield {"event": "phase", "data": {"phase": "generating"}}

        answer_chunks: List[str] = []
        prompt = _render_user_prompt(query, items, memory_ctx.rendered_block)
        gen_start = time.perf_counter()

        try:
            async for chunk in self._stream_answer(prompt):
                answer_chunks.append(chunk)
                yield {"event": "delta", "data": {"text": chunk}}
        except Exception as exc:  # noqa: BLE001
            logger.error("ask_stream generation failed: %s", exc, exc_info=True)
            # Still record the (partial) answer so the turn isn't lost.
            answer = "".join(answer_chunks) or f"(LLM stream failed: {exc})"
            yield {"event": "error", "data": {"message": str(exc), "kind": "llm_failed"}}
            await self._record_llm_latency(time.perf_counter() - gen_start)
            return

        await self._record_llm_latency(time.perf_counter() - gen_start)

        answer = "".join(answer_chunks)
        cited_indices = [
            i for i in _extract_cited_indices(answer)
            if 1 <= i <= len(items)
        ]
        cited_entity_ids, cited_rel_ids, cited_chunk_ids = self._collect_cited_ids(
            items, cited_indices,
        )

        faithfulness_result = await self._check_faithfulness(answer, items)

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

        await self._record_request_metric(status="ok", intent=intent_label or "any")
        await self._record_total_latency(latency_ms)
        await self._record_memory_tokens(memory_ctx)
        if faithfulness_result and faithfulness_result.failed_claims:
            await self._record_faithfulness_failures(faithfulness_result.failed_claims)

        logger.info(
            "ask_stream done session=%s turn=%s intent=%s sources=%d cited=%d "
            "chunks=%d faithfulness=%s latency_ms=%d",
            session.id, turn.id, intent_label or "override",
            len(items), len(cited_indices), len(answer_chunks),
            (
                f"{faithfulness_result.overall_score:.2f}"
                if faithfulness_result and faithfulness_result.overall_score is not None
                else "n/a"
            ),
            latency_ms,
        )

        yield {
            "event": "done",
            "data": {
                "session_id": session.id,
                "turn_id": turn.id,
                "answer": answer,
                "cited_source_indices": cited_indices,
                "faithfulness": (
                    faithfulness_result.to_dict() if faithfulness_result else None
                ),
                "request_id": request_id,
                "latency_ms": latency_ms,
            },
        }

    async def _stream_answer(self, prompt: str) -> AsyncIterator[str]:
        """Yield answer chunks. Falls back to a single chunk when the LLM
        service doesn't expose a streaming method (e.g. test fakes or a
        provider without ``stream=True`` support)."""
        if self._llm is None:
            yield (
                "(no LLM configured — retrieval already streamed; see Sources panel)"
            )
            return
        stream_fn = getattr(self._llm, "generate_text_stream", None)
        if stream_fn is None:
            # Graceful fallback: call the non-streaming method and emit
            # the full answer as one chunk. Keeps the SSE contract the
            # same shape regardless of provider capability.
            answer = await self._llm.generate_text(
                prompt=prompt,
                system_prompt=_SYSTEM_PROMPT,
                temperature=0.1,
                max_tokens=1024,
            )
            yield answer
            return

        async for chunk in stream_fn(
            prompt=prompt,
            system_prompt=_SYSTEM_PROMPT,
            temperature=0.1,
            max_tokens=1024,
        ):
            if chunk:
                yield chunk

    # ------------------------------------------------------------------
    # P9 — agentic generation (read-only tools)
    # ------------------------------------------------------------------

    async def _generate_answer_agentic(
        self,
        query: str,
        items: List[RetrievedItem],
        memory_ctx: MemoryContext,
        *,
        enable_tools: bool = True,
        enable_mutations: bool = False,
        proposer_user_id: Optional[str] = None,
    ) -> tuple[str, List[ToolCallRecord]]:
        """Run the tool-using loop and return ``(answer, tool_calls)``.

        Falls back to the single-call ``_generate_answer`` path when:
        - the LLM service doesn't expose ``generate_with_tools`` (e.g.
          provider stub in a test), or
        - the model hits the per-turn tool-call cap without producing
          an answer (we then ask for a final answer with no tools).

        Either fallback still records every tool call that DID fire so
        the trace stays honest.
        """
        # Build the union of dispatcher schemas the LLM may pick from.
        read_active = enable_tools and self._tools is not None
        mut_active = enable_mutations and self._mutation_tools is not None
        if not (read_active or mut_active) or self._llm is None:
            return await self._generate_answer(query, items, memory_ctx), []

        tools_fn = getattr(self._llm, "generate_with_tools", None)
        if tools_fn is None:
            # Provider can't function-call — degrade silently so the
            # eval harness and tests with a non-streaming fake LLM
            # don't break.
            return await self._generate_answer(query, items, memory_ctx), []

        user_prompt = _render_user_prompt(query, items, memory_ctx.rendered_block)
        system_prompt = _SYSTEM_PROMPT + _TOOLS_PROMPT_SUFFIX
        if mut_active:
            system_prompt += _MUTATION_PROMPT_SUFFIX
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        tools_schema: List[Dict[str, Any]] = []
        if read_active:
            tools_schema += self._tools.openai_tool_schemas()
        if mut_active:
            tools_schema += self._mutation_tools.openai_tool_schemas()
        records: List[ToolCallRecord] = []

        for _ in range(self._max_tool_calls + 1):
            response = await tools_fn(
                messages=messages,
                tools=tools_schema,
                temperature=0.1,
                max_tokens=1024,
            )
            tool_calls = response.get("tool_calls") or []
            content = response.get("content")

            if not tool_calls:
                # Model stopped calling tools → this is the final answer.
                if content:
                    return content, records
                # Empty answer + no tool call — log and fall back to a
                # non-tool generate so the turn still produces text.
                logger.warning(
                    "agentic loop: empty content + no tool calls; falling back",
                )
                return await self._generate_answer(query, items, memory_ctx), records

            # Append the model's tool_call turn to the conversation
            # before executing — OpenAI's protocol requires the
            # assistant message with tool_calls to precede the tool
            # messages that respond to it.
            messages.append({
                "role": "assistant",
                "content": content,
                "tool_calls": [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": json.dumps(tc["arguments"]),
                        },
                    }
                    for tc in tool_calls
                ],
            })

            # Hit the cap? Strip tools from the next call so the model
            # has to produce a plain answer — preserves the existing
            # tool-call records on the way out.
            if len(records) + len(tool_calls) > self._max_tool_calls:
                logger.info(
                    "agentic loop: hit max_tool_calls_per_turn=%d, "
                    "forcing final answer", self._max_tool_calls,
                )
                final = await tools_fn(
                    messages=messages + [{
                        "role": "user",
                        "content": "Answer now from what you have; no more tool calls.",
                    }],
                    tools=[],
                    temperature=0.1,
                    max_tokens=1024,
                )
                return final.get("content") or "(no answer generated)", records

            for tc in tool_calls:
                record = await self._dispatch_tool_call(
                    name=tc["name"],
                    args=tc["arguments"],
                    tool_call_id=tc["id"],
                    proposer_user_id=proposer_user_id,
                    read_active=read_active,
                    mut_active=mut_active,
                )
                records.append(record)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": json.dumps(record.to_dict()["result"] or {
                        "error": record.error,
                    }),
                })

        # Loop exit without an answer — shouldn't happen given the cap
        # check above, but stay defensive.
        logger.warning("agentic loop fell through without an answer")
        return await self._generate_answer(query, items, memory_ctx), records

    async def _dispatch_tool_call(
        self,
        *,
        name: str,
        args: Dict[str, Any],
        tool_call_id: str,
        proposer_user_id: Optional[str],
        read_active: bool,
        mut_active: bool,
    ) -> ToolCallRecord:
        """Route one tool call to the right dispatcher.

        Mutating tools always go through the mutation dispatcher (so
        provenance is recorded on the proposal); read-only tools go
        through the read dispatcher. A model that calls a mutating
        tool when ``enable_mutations`` is False gets an explicit error
        record so it can recover instead of silently producing nothing.
        """
        if is_mutation(name):
            if not mut_active:
                return ToolCallRecord(
                    tool=name, args=args,
                    error=(
                        "mutating tools are not enabled for this request "
                        "(set enable_mutations=true)"
                    ),
                    tool_call_id=tool_call_id,
                )
            return await self._mutation_tools.execute(
                name, args,
                tool_call_id=tool_call_id,
                proposer_user_id=proposer_user_id,
            )
        if not read_active:
            return ToolCallRecord(
                tool=name, args=args,
                error=(
                    "read-only tools are not enabled for this request "
                    "(set enable_tools=true)"
                ),
                tool_call_id=tool_call_id,
            )
        return await self._tools.execute(
            name, args, tool_call_id=tool_call_id,
        )

    @staticmethod
    def _collect_cited_ids(
        items: List[RetrievedItem],
        cited_indices: List[int],
    ) -> tuple[List[str], List[str], List[str]]:
        """Bucket cited source ids by kind for ``ConversationTurn``."""
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
        return cited_entity_ids, cited_rel_ids, cited_chunk_ids

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

    async def _record_faithfulness_failures(self, n: int) -> None:
        try:
            from ...infrastructure.services.metrics import get_metrics
        except Exception:
            return
        await get_metrics().record_qa_faithfulness_failure(n=n)

    # ------------------------------------------------------------------
    # Faithfulness
    # ------------------------------------------------------------------

    async def _check_faithfulness(
        self,
        answer: str,
        items: List[RetrievedItem],
    ) -> Optional[FaithfulnessResult]:
        """Run the per-claim faithfulness cascade. Failures degrade
        gracefully — a checker exception leaves the field as ``None``
        rather than taking the whole response down."""
        try:
            return await self._faithfulness.check(answer=answer, sources=items)
        except Exception as exc:  # noqa: BLE001
            logger.warning("faithfulness check failed: %s", exc, exc_info=True)
            return None

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
    "FaithfulnessChecker",
    "FaithfulnessConfig",
    "FaithfulnessResult",
    "MemoryConfig",
    "MemoryContext",
    "MemoryService",
    "MutationToolDispatcher",
    "QAService",
    "ToolCallRecord",
    "ToolDispatcher",
]
