"""QA router — POST /qa/ask + a few session helpers (P5 of docs/RAG_QA_PLAN.md).

The router is deliberately thin: it adapts FastAPI dependencies to the
``QAService`` and translates dataclasses to Pydantic responses. Tool-use
(P9/P10) is not wired here yet; faithfulness (P8) rides along on the
``faithfulness`` field of ``AskResponse``. Streaming (P11) is exposed
as ``POST /qa/ask/stream`` using the same SSE pattern as
``/documents/jobs/{id}/stream``.
"""

from __future__ import annotations

import json
import logging
import sys
import os
from typing import Annotated, Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from sse_starlette.sse import EventSourceResponse

from ..auth import require_api_key
from ..dependencies import (
    get_app_config,
    get_conversation_repo,
    get_document_repo,
    get_graph_repo,
    get_llm,
    get_user_repo,
)
from ..schemas.qa import (
    AskRequest,
    AskResponse,
    ChannelTraceModel,
    ClaimVerificationModel,
    FaithfulnessModel,
    FeedbackRequest,
    FeedbackResponse,
    MemoryEpisodicHit,
    MemoryTraceModel,
    RetrievalTraceModel,
    SourceModel,
)


# Make sure the src/ package is importable when running from repo root.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from graphbuilder.core.retrieval import (  # noqa: E402
    RetrievalConfig,
    RetrievalOrchestrator,
)
from graphbuilder.core.retrieval.qa_service import QAService  # noqa: E402


logger = logging.getLogger("graphbuilder.qa.api")

router = APIRouter(prefix="/qa", tags=["qa"])


# ----------------------------------------------------------------------
# X-User-Id resolver
# ----------------------------------------------------------------------
#
# Lightweight chat-only identity (§14.1 of docs/RAG_QA_PLAN.md, revised
# 2026-05-09). Lenient: the header is OPTIONAL — when missing we fall
# through to the existing anonymous-bucket behaviour so older clients
# keep working. When present and valid we ``touch_user`` for the
# last-seen freshness signal and surface the id back to handlers.
#
# When present-but-unknown (id was deleted, or someone fabricated one)
# we 401 — silently creating a stand-in would let stale localStorage
# write into the wrong account if/when ids are ever recycled.

async def get_chat_user_id(
    request: Request,
    user_repo=Depends(get_user_repo),
) -> Optional[str]:
    """Resolve and validate the X-User-Id header for /qa/* routes."""
    raw = request.headers.get("X-User-Id")
    if not raw:
        return None
    raw = raw.strip()
    if not raw:
        return None
    if len(raw) > 128:
        raise HTTPException(status_code=400, detail="X-User-Id too long")
    user = await user_repo.touch_user(raw)
    if user is None:
        # The id is well-formed but doesn't resolve to an account —
        # surface that explicitly so the frontend can reset localStorage.
        raise HTTPException(
            status_code=401,
            detail="Unknown X-User-Id; clear localStorage and re-register.",
        )
    return user.id


# ----------------------------------------------------------------------
# Service factory — singleton per process so the orchestrator + repos
# don't reallocate on every request.
# ----------------------------------------------------------------------

_qa_service_singleton: QAService | None = None


def _get_qa_service(
    *,
    config: Any,
    graph_repo: Any,
    document_repo: Any,
    conversation_repo: Any,
    llm_service: Any,
) -> QAService:
    global _qa_service_singleton
    if _qa_service_singleton is None:
        retrieval_cfg = RetrievalConfig()
        orchestrator = RetrievalOrchestrator(
            graph_repo=graph_repo,
            document_repo=document_repo,
            config=retrieval_cfg,
        )
        _qa_service_singleton = QAService(
            orchestrator=orchestrator,
            conversation_repo=conversation_repo,
            llm_service=llm_service,
            config=retrieval_cfg,
        )
        logger.info("QAService initialised (single-process singleton)")
    return _qa_service_singleton


# ----------------------------------------------------------------------
# Routes
# ----------------------------------------------------------------------

@router.post("/ask", response_model=AskResponse)
async def ask(
    request: Request,
    body: AskRequest,
    config=Depends(get_app_config),
    graph_repo=Depends(get_graph_repo),
    document_repo=Depends(get_document_repo),
    conversation_repo=Depends(get_conversation_repo),
    llm_service=Depends(get_llm),
    chat_user_id: Optional[str] = Depends(get_chat_user_id),
    _=Depends(require_api_key),
) -> AskResponse:
    if not body.query or not body.query.strip():
        raise HTTPException(status_code=400, detail="query must not be empty")

    service = _get_qa_service(
        config=config,
        graph_repo=graph_repo,
        document_repo=document_repo,
        conversation_repo=conversation_repo,
        llm_service=llm_service,
    )

    # Header takes precedence over the deprecated body field — the
    # header is what the new frontend sends, and is also the only
    # value we've validated against the user repo.
    effective_user_id = chat_user_id if chat_user_id is not None else body.user_id

    # Build a per-request RetrievalConfig override when the eval runner
    # is asking for an ablation. Every flag is optional — unset fields
    # fall back to the singleton's config so production traffic that
    # never sets ``ablation`` is unaffected.
    retrieval_override = _build_ablation_override(service, body.ablation)

    try:
        result = await service.ask(
            query=body.query,
            session_id=body.session_id,
            user_id=effective_user_id,
            top_k=body.top_k,
            retrieval_override=retrieval_override,
        )
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return _to_ask_response(result)


@router.post("/ask/stream", summary="Streaming /qa/ask via SSE (P11)")
async def ask_stream(
    request: Request,
    body: AskRequest,
    config=Depends(get_app_config),
    graph_repo=Depends(get_graph_repo),
    document_repo=Depends(get_document_repo),
    conversation_repo=Depends(get_conversation_repo),
    llm_service=Depends(get_llm),
    chat_user_id: Optional[str] = Depends(get_chat_user_id),
    _=Depends(require_api_key),
) -> EventSourceResponse:
    """Server-Sent Events: same data as ``/qa/ask`` but streamed.

    Event sequence per turn:

    1. ``phase`` (``retrieving``)
    2. ``retrieval`` — sources + retrieval_trace + memory_trace
    3. ``phase`` (``generating``)
    4. ``delta`` — repeated; each carries a token chunk
    5. ``done`` — turn_id, session_id, cited indices, faithfulness, latency

    On any failure an ``error`` event is emitted instead of the next
    expected event and the stream closes; the client should display
    ``error.message`` and stop reading.
    """
    if not body.query or not body.query.strip():
        raise HTTPException(status_code=400, detail="query must not be empty")

    service = _get_qa_service(
        config=config,
        graph_repo=graph_repo,
        document_repo=document_repo,
        conversation_repo=conversation_repo,
        llm_service=llm_service,
    )

    effective_user_id = chat_user_id if chat_user_id is not None else body.user_id
    retrieval_override = _build_ablation_override(service, body.ablation)

    async def event_generator():
        try:
            async for ev in service.ask_stream(
                query=body.query,
                session_id=body.session_id,
                user_id=effective_user_id,
                top_k=body.top_k,
                retrieval_override=retrieval_override,
            ):
                # Each event is a {"event": <name>, "data": <dict>} pair.
                # ``sse_starlette`` accepts that shape directly when we
                # JSON-encode the data field — keeps the SSE frame's
                # ``data:`` line a single-line JSON object the client
                # can ``JSON.parse``.
                yield {
                    "event": ev["event"],
                    "data": json.dumps(ev["data"]),
                }
                # If the service signalled the end of the stream, close
                # cleanly. ``sse_starlette`` would also close on return,
                # but being explicit avoids a wasted iteration if the
                # generator yields anything after ``done`` / ``error``.
                if ev["event"] in ("done", "error"):
                    return
        except Exception as exc:  # noqa: BLE001 — last-line safety net
            logger.error("/qa/ask/stream generator crashed: %s", exc, exc_info=True)
            yield {
                "event": "error",
                "data": json.dumps(
                    {"message": str(exc), "kind": "internal_error"}
                ),
            }

    return EventSourceResponse(event_generator())


def _build_ablation_override(service: QAService, ablation: Any) -> Any:
    """Clone the service's ``RetrievalConfig`` and apply only the
    fields the request actually set. Returns ``None`` when no override
    is requested so the orchestrator stays on its singleton config.
    """
    if ablation is None:
        return None
    base = service._cfg  # noqa: SLF001 — singleton-only, not a public surface
    from dataclasses import replace
    overrides = {
        k: v for k, v in {
            "enable_cypher_channel": ablation.enable_cypher_channel,
            "enable_vector_channel": ablation.enable_vector_channel,
            "enable_bm25_channel":   ablation.enable_bm25_channel,
            "enable_cross_encoder":  ablation.enable_cross_encoder,
            "chunk_neighbour_radius": ablation.chunk_neighbour_radius,
            "emit_chunk_items":      ablation.emit_chunk_items,
        }.items() if v is not None
    }
    # Blocklist needs the empty-list-is-meaningful escape hatch (`[]`
    # = "drop nothing, include authors/papers"). Pydantic models
    # surface unset as None vs. set-to-empty as `[]`, so the truthy
    # filter above would silently swallow `[]`. Treat None as unset
    # and pass a tuple through for everything else.
    if ablation.entity_type_blocklist is not None:
        overrides["entity_type_blocklist"] = tuple(ablation.entity_type_blocklist)
    if not overrides:
        return None
    return replace(base, **overrides)


@router.get("/sessions/{session_id}", summary="Fetch a session and its turns")
async def get_session(
    session_id: str,
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    conversation_repo=Depends(get_conversation_repo),
    chat_user_id: Optional[str] = Depends(get_chat_user_id),
    _=Depends(require_api_key),
):
    session = await conversation_repo.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="session not found")
    # Authorisation: a session that belongs to a user can only be read
    # by that user. Anonymous (user_id=None) sessions stay readable
    # by anyone, matching the pre-identity behaviour.
    if session.user_id is not None and session.user_id != chat_user_id:
        raise HTTPException(status_code=404, detail="session not found")
    turns = await conversation_repo.get_turns_by_session(
        session_id, limit=limit, offset=offset,
    )
    return {
        "session": session.to_dict(),
        "turns": [t.to_dict() for t in turns],
    }


@router.get("/sessions", summary="List sessions for the current user (or anonymous)")
async def list_sessions(
    user_id: str | None = Query(
        None,
        description=(
            "Override filter; when omitted, defaults to the X-User-Id "
            "header (or anonymous if neither is set)."
        ),
    ),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    conversation_repo=Depends(get_conversation_repo),
    chat_user_id: Optional[str] = Depends(get_chat_user_id),
    _=Depends(require_api_key),
):
    # Default to the header-resolved user when no explicit filter is
    # given, so the chat sidebar shows YOUR sessions without the
    # frontend having to pass user_id twice.
    effective = user_id if user_id is not None else chat_user_id
    sessions = await conversation_repo.list_sessions(
        user_id=effective, limit=limit, offset=offset
    )
    return {"sessions": [s.to_dict() for s in sessions]}


@router.delete(
    "/sessions/{session_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a session and its turns",
)
async def delete_session(
    session_id: str,
    conversation_repo=Depends(get_conversation_repo),
    chat_user_id: Optional[str] = Depends(get_chat_user_id),
    _=Depends(require_api_key),
):
    # Same auth rule as get_session: only the owner can delete a
    # user-scoped session. Treat ownership mismatch as 404 not 403
    # so we don't leak whether the id exists for someone else.
    existing = await conversation_repo.get_session(session_id)
    if existing is None:
        raise HTTPException(status_code=404, detail="session not found")
    if existing.user_id is not None and existing.user_id != chat_user_id:
        raise HTTPException(status_code=404, detail="session not found")
    deleted = await conversation_repo.delete_session(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="session not found")
    return None


@router.post("/turns/{turn_id}/feedback", response_model=FeedbackResponse)
async def post_feedback(
    turn_id: str,
    body: FeedbackRequest,
    conversation_repo=Depends(get_conversation_repo),
    _=Depends(require_api_key),
) -> FeedbackResponse:
    accepted = await conversation_repo.record_feedback(
        turn_id, rating=body.rating, comment=body.comment,
    )
    if not accepted:
        raise HTTPException(status_code=404, detail="turn not found")
    return FeedbackResponse(turn_id=turn_id, accepted=True)


# ----------------------------------------------------------------------
# Translators
# ----------------------------------------------------------------------

def _to_ask_response(result) -> AskResponse:
    return AskResponse(
        session_id=result.session_id,
        turn_id=result.turn_id,
        answer=result.answer,
        sources=[_to_source_model(s) for s in result.sources],
        cited_source_indices=list(result.cited_source_indices),
        retrieval_trace=_to_trace_model(result.retrieval_trace),
        memory_trace=_to_memory_trace_model(result.memory_trace),
        faithfulness=_to_faithfulness_model(getattr(result, "faithfulness", None)),
        request_id=result.request_id,
        latency_ms=result.latency_ms,
    )


def _to_faithfulness_model(fr) -> FaithfulnessModel | None:
    """Adapt a ``FaithfulnessResult`` into the response schema."""
    if fr is None:
        return None
    return FaithfulnessModel(
        overall_score=fr.overall_score,
        failed_claims=fr.failed_claims,
        is_refusal=fr.is_refusal,
        claims=[
            ClaimVerificationModel(
                claim_text=c.claim_text,
                cited_indices=list(c.cited_indices),
                confidence=c.confidence,
                method=c.method,
                reasoning=c.reasoning,
                escalated_to_llm=c.escalated_to_llm,
                matched_terms=list(c.matched_terms),
                missing_terms=list(c.missing_terms),
            )
            for c in fr.claims
        ],
    )


def _to_memory_trace_model(trace_dict) -> MemoryTraceModel | None:
    """Adapt the dict shape MemoryContext.to_trace_dict() produces."""
    if not trace_dict:
        return None
    hit = trace_dict.get("episodic_hit")
    return MemoryTraceModel(
        working_turns=int(trace_dict.get("working_turns") or 0),
        summary_chars=int(trace_dict.get("summary_chars") or 0),
        episodic_hit=(
            MemoryEpisodicHit(turn_id=hit["turn_id"], score=float(hit["score"]))
            if hit else None
        ),
        summary_regenerated=bool(trace_dict.get("summary_regenerated")),
    )


def _to_source_model(item) -> SourceModel:
    return SourceModel(
        kind=item.kind.value,
        id=item.id,
        label=item.label,
        score_vector=item.score_vector,
        score_bm25=item.score_bm25,
        score_cypher=item.score_cypher,
        score_rrf=item.score_rrf,
        score_rerank=item.score_rerank,
        final_confidence=item.final_confidence,
        source_url=item.source_url,
        source_doc_id=item.source_doc_id,
        source_chunk_id=item.source_chunk_id,
        source_chunk_ids=list(item.source_chunk_ids),
        chunk_preview=item.chunk_preview,
        contributing_channels=[c.value for c in item.contributing_channels],
        reasoning=item.reasoning,
    )


def _to_trace_model(trace) -> RetrievalTraceModel:
    if trace is None:
        return RetrievalTraceModel(
            query="", extracted_terms=[], channels=[],
            rrf_top_n=0, final_top_k=0, hydrated_chunks=0,
            total_latency_ms=0,
        )
    return RetrievalTraceModel(
        query=trace.query,
        extracted_terms=list(trace.extracted_terms),
        channels=[
            ChannelTraceModel(
                channel=c.channel.value,
                hits=c.hit_count,
                latency_ms=c.latency_ms,
                error=c.error,
            )
            for c in trace.channels
        ],
        rrf_top_n=trace.rrf_top_n,
        final_top_k=trace.final_top_k,
        hydrated_chunks=trace.hydrated_chunks,
        total_latency_ms=trace.total_latency_ms,
    )
