"""QA router — POST /qa/ask + a few session helpers (P5 of docs/RAG_QA_PLAN.md).

The router is deliberately thin: it adapts FastAPI dependencies to the
``QAService`` and translates dataclasses to Pydantic responses. Streaming
(P11), tool-use (P9/P10), and faithfulness (P8) are not wired here yet.
"""

from __future__ import annotations

import logging
import sys
import os
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status

from ..auth import require_api_key
from ..dependencies import (
    get_app_config,
    get_conversation_repo,
    get_document_repo,
    get_graph_repo,
    get_llm,
)
from ..schemas.qa import (
    AskRequest,
    AskResponse,
    ChannelTraceModel,
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

    try:
        result = await service.ask(
            query=body.query,
            session_id=body.session_id,
            user_id=body.user_id,
            top_k=body.top_k,
        )
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return _to_ask_response(result)


@router.get("/sessions/{session_id}", summary="Fetch a session and its turns")
async def get_session(
    session_id: str,
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    conversation_repo=Depends(get_conversation_repo),
    _=Depends(require_api_key),
):
    session = await conversation_repo.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="session not found")
    turns = await conversation_repo.get_turns_by_session(
        session_id, limit=limit, offset=offset,
    )
    return {
        "session": session.to_dict(),
        "turns": [t.to_dict() for t in turns],
    }


@router.get("/sessions", summary="List sessions for the anonymous user (v1)")
async def list_sessions(
    user_id: str | None = Query(None),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    conversation_repo=Depends(get_conversation_repo),
    _=Depends(require_api_key),
):
    sessions = await conversation_repo.list_sessions(
        user_id=user_id, limit=limit, offset=offset
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
    _=Depends(require_api_key),
):
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
        request_id=result.request_id,
        latency_ms=result.latency_ms,
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
