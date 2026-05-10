"""Read-only tool surface for the chat LLM (P9 of docs/RAG_QA_PLAN.md).

The LLM doesn't write raw Cypher; it picks a tool from a fixed list,
the dispatcher validates the args against a Pydantic schema, and the
backing function (orchestrator / graph repo / cascading verifier) does
the work. Every call returns a ``ToolCallRecord`` the API surfaces in
``AskResponse.tool_calls`` and the eval harness can use to spot when
the model went looking past the initial retrieved sources.

P9 is **read-only only** — ``search_graph``, ``get_entity``,
``verify_claim``. P10 (mutating tools) ships separately and routes
through the curation review queue per §14.6's resolution; it lives in
a different module so this one can stay narrow.

Design choices worth flagging:

- **Schemas are Pydantic, not JSON-Schema-by-hand.** ``model_json_schema()``
  feeds straight into OpenAI's function-calling format, and a single
  validate-then-dispatch keeps argument coercion in one place.

- **Errors never raise out of ``execute``.** A bad ``entity_id`` or
  malformed args returns a ``ToolCallRecord`` with ``error`` set; the
  LLM sees the error in its tool-message and can retry. Raising would
  abort the agentic loop and lose the partial work.

- **No agentic loop in this module.** The dispatcher just runs one
  call. The loop that decides "another tool call or final answer"
  lives in :class:`QAService` so it can keep its existing memory +
  retrieval wiring on the hot path.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, ValidationError

from .models import ItemKind


logger = logging.getLogger("graphbuilder.qa.tools")


# ----------------------------------------------------------------------
# Tool names + arg schemas
# ----------------------------------------------------------------------


TOOL_SEARCH_GRAPH = "search_graph"
TOOL_GET_ENTITY = "get_entity"
TOOL_VERIFY_CLAIM = "verify_claim"


class SearchGraphArgs(BaseModel):
    """Args for ``search_graph`` — wraps the retrieval orchestrator."""

    query: str = Field(..., min_length=1, description="Free-text search query.")
    top_k: int = Field(
        8, ge=1, le=20,
        description="Maximum number of items to return (default 8).",
    )
    kinds: List[str] = Field(
        default_factory=lambda: ["entity", "relationship"],
        description=(
            "Limit to entity / relationship / chunk kinds. Defaults to "
            "entity + relationship; pass [\"chunk\"] to fetch chunk "
            "neighbours of cited entities."
        ),
    )


class GetEntityArgs(BaseModel):
    """Args for ``get_entity`` — fetch one entity + its 1-hop neighbourhood."""

    entity_id: str = Field(..., min_length=1, description="Entity id to inspect.")
    include_neighbours: bool = Field(
        True,
        description=(
            "Include the 1-hop relationship neighbourhood. Off saves a "
            "round-trip when the LLM only needs entity properties."
        ),
    )


class VerifyClaimArgs(BaseModel):
    """Args for ``verify_claim`` — run the 3-stage cascade against a claim.

    The dispatcher synthesises a placeholder ``GraphRelationship`` from
    the claim text + endpoints so the cascade's
    ``TextMatchVerifier`` / ``EmbeddingVerifier`` / ``LLMVerifier``
    can score it. When ``source_entity`` / ``target_entity`` are
    omitted the verifier falls back to claim-text-only matching, which
    is rougher but still useful for "does this chunk support this
    statement?" style checks.
    """

    claim: str = Field(..., min_length=1, description="The claim text to verify.")
    context: str = Field(
        ..., min_length=1,
        description="Free-text context (chunk content) the claim should be supported by.",
    )
    source_entity: Optional[str] = Field(
        None, description="Source entity name (when claim is relational)."
    )
    target_entity: Optional[str] = Field(
        None, description="Target entity name (when claim is relational)."
    )


# ----------------------------------------------------------------------
# Result records
# ----------------------------------------------------------------------


@dataclass
class ToolCallRecord:
    """One tool-call → result pair, recorded on the turn trace.

    The LLM's ``tool_call_id`` is preserved so OpenAI-style follow-ups
    line up (``role="tool", tool_call_id=...``); callers that bypass
    function-calling can leave it ``None``.
    """

    tool: str
    args: Dict[str, Any]
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    latency_ms: int = 0
    tool_call_id: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool": self.tool,
            "args": dict(self.args),
            "result": self.result if self.error is None else None,
            "error": self.error,
            "latency_ms": self.latency_ms,
            "tool_call_id": self.tool_call_id,
        }


# ----------------------------------------------------------------------
# Dispatcher
# ----------------------------------------------------------------------


class ToolDispatcher:
    """Routes validated tool calls to the existing services that back them.

    Construction is cheap and stateless — pass the already-built
    orchestrator + repos. The dispatcher itself doesn't decide when to
    run; that's the QA-service loop's job.
    """

    def __init__(
        self,
        *,
        orchestrator: Any,
        graph_repo: Any,
        llm_service: Optional[Any] = None,
    ) -> None:
        self._orch = orchestrator
        self._graph_repo = graph_repo
        self._llm = llm_service
        # Lazy-built — cascading verifier only matters for ``verify_claim``.
        self._verifier: Optional[Any] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def execute(
        self,
        tool: str,
        args: Dict[str, Any],
        *,
        tool_call_id: Optional[str] = None,
    ) -> ToolCallRecord:
        """Validate args + dispatch. Never raises."""
        t0 = time.perf_counter()
        try:
            if tool == TOOL_SEARCH_GRAPH:
                parsed = SearchGraphArgs(**args)
                result = await self._do_search_graph(parsed)
            elif tool == TOOL_GET_ENTITY:
                parsed = GetEntityArgs(**args)
                result = await self._do_get_entity(parsed)
            elif tool == TOOL_VERIFY_CLAIM:
                parsed = VerifyClaimArgs(**args)
                result = await self._do_verify_claim(parsed)
            else:
                return ToolCallRecord(
                    tool=tool, args=args,
                    error=f"unknown tool: {tool}",
                    latency_ms=_ms(t0), tool_call_id=tool_call_id,
                )
        except ValidationError as ve:
            return ToolCallRecord(
                tool=tool, args=args,
                error=f"invalid arguments: {ve.errors()}",
                latency_ms=_ms(t0), tool_call_id=tool_call_id,
            )
        except Exception as exc:  # noqa: BLE001 — surfaced to LLM, not raised
            logger.warning("tool %s raised: %s", tool, exc, exc_info=True)
            return ToolCallRecord(
                tool=tool, args=args,
                error=f"tool execution failed: {exc}",
                latency_ms=_ms(t0), tool_call_id=tool_call_id,
            )

        return ToolCallRecord(
            tool=tool, args=args, result=result,
            latency_ms=_ms(t0), tool_call_id=tool_call_id,
        )

    def openai_tool_schemas(self) -> List[Dict[str, Any]]:
        """Return the OpenAI ``tools=[...]`` payload for function-calling.

        Each entry is ``{"type": "function", "function": {...}}`` with a
        ``parameters`` JSON schema derived from the Pydantic model.
        Keeps schema-generation in one place so the LLM service doesn't
        need to know about Pydantic.
        """
        return [
            _openai_tool(
                TOOL_SEARCH_GRAPH,
                "Search the knowledge graph. Returns entities and/or "
                "relationships matching the query. Use this when the "
                "initial SOURCES block doesn't contain enough to answer.",
                SearchGraphArgs,
            ),
            _openai_tool(
                TOOL_GET_ENTITY,
                "Fetch one entity (by id) plus its 1-hop relationship "
                "neighbourhood. Use after search_graph returns a "
                "promising id you want to inspect more closely.",
                GetEntityArgs,
            ),
            _openai_tool(
                TOOL_VERIFY_CLAIM,
                "Score how well a free-text context supports a claim, "
                "using a 3-stage text/embedding/LLM cascade. Use before "
                "stating a fact you're uncertain about.",
                VerifyClaimArgs,
            ),
        ]

    # ------------------------------------------------------------------
    # Per-tool implementations
    # ------------------------------------------------------------------

    async def _do_search_graph(self, args: SearchGraphArgs) -> Dict[str, Any]:
        items, trace = await self._orch.retrieve(args.query, top_k=args.top_k)
        kinds_filter = {k.lower() for k in args.kinds}
        filtered = [
            it for it in items
            if it.kind.value in kinds_filter
        ]
        return {
            "query": args.query,
            "n_items": len(filtered),
            "items": [_item_summary(it) for it in filtered[: args.top_k]],
            "trace": {
                "rrf_top_n": trace.rrf_top_n,
                "hydrated_chunks": trace.hydrated_chunks,
                "total_latency_ms": trace.total_latency_ms,
            },
        }

    async def _do_get_entity(self, args: GetEntityArgs) -> Dict[str, Any]:
        entity = await self._graph_repo.get_entity_by_id(args.entity_id)
        if entity is None:
            return {"entity_id": args.entity_id, "found": False}
        out: Dict[str, Any] = {
            "entity_id": entity.id,
            "name": getattr(entity, "name", None),
            "entity_type": _enum_value(getattr(entity, "entity_type", None)),
            "description": getattr(entity, "description", None),
            "external_ids": list(getattr(entity, "external_ids", []) or []),
            "aliases": list(getattr(entity, "aliases", []) or []),
            "found": True,
        }
        if args.include_neighbours:
            rels = await self._graph_repo.get_entity_relationships(entity.id)
            out["relationships"] = [_relationship_summary(r) for r in rels]
        return out

    async def _do_verify_claim(self, args: VerifyClaimArgs) -> Dict[str, Any]:
        verifier = self._get_verifier()
        rel = _synthesise_relationship(
            args.source_entity, args.target_entity, args.claim,
        )
        result = await verifier.verify(
            relationship=rel,
            context=args.context,
            source_name=args.source_entity,
            target_name=args.target_entity,
        )
        return {
            "verdict": result.status.value,
            "confidence": round(result.confidence, 4),
            "stage": result.stage.value,
            "reasoning": result.reasoning,
            "stage_results": [
                {
                    "stage": sr.stage.value,
                    "status": sr.status.value,
                    "confidence": round(sr.confidence, 4),
                }
                for sr in result.stage_results
            ],
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_verifier(self):
        if self._verifier is not None:
            return self._verifier
        # Local import keeps the verification module optional for
        # codepaths that never call verify_claim (e.g. the eval gate
        # exercising only search_graph / get_entity).
        from ..verification.cascading import (
            CascadingVerifier, CascadingVerifierConfig,
        )
        # No graph_repo passed to the embedding stage — verify_claim
        # operates on free-text context, not a vector lookup.
        self._verifier = CascadingVerifier(
            config=CascadingVerifierConfig(),
            llm_service=self._llm,
            graph_repo=self._graph_repo,
        )
        return self._verifier


# ----------------------------------------------------------------------
# Schema / serialisation helpers
# ----------------------------------------------------------------------


def _openai_tool(name: str, description: str, model: type[BaseModel]) -> Dict[str, Any]:
    """Build one ``{"type": "function", "function": {...}}`` entry."""
    schema = model.model_json_schema()
    # OpenAI's function-calling spec doesn't allow ``$defs`` at the
    # top level of the parameters; inline them with model_json_schema's
    # built-in resolution by stripping unused keys we don't need.
    schema.pop("title", None)
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": schema,
        },
    }


def _item_summary(item: Any) -> Dict[str, Any]:
    """Compact one ``RetrievedItem`` for the tool message — keep just the
    fields the LLM needs to decide if it's worth a follow-up call."""
    return {
        "kind": item.kind.value if hasattr(item.kind, "value") else str(item.kind),
        "id": item.id,
        "label": item.label,
        "final_confidence": round(item.final_confidence, 4),
        "chunk_preview": (item.chunk_preview or "")[:240] or None,
        "source_doc_id": item.source_doc_id,
    }


def _relationship_summary(rel: Any) -> Dict[str, Any]:
    return {
        "id": getattr(rel, "id", None),
        "source_entity_id": getattr(rel, "source_entity_id", None),
        "target_entity_id": getattr(rel, "target_entity_id", None),
        "relationship_type": _enum_value(getattr(rel, "relationship_type", None)),
        "description": getattr(rel, "description", None),
    }


def _enum_value(val: Any) -> Optional[str]:
    if val is None:
        return None
    return getattr(val, "value", str(val))


def _synthesise_relationship(
    source_name: Optional[str],
    target_name: Optional[str],
    description: str,
):
    """Build a placeholder ``GraphRelationship`` for ``verify_claim``.

    The cascade signature requires a relationship dataclass, but for a
    free-text claim we only have prose. We slot the source/target names
    (or sentinels when absent) into the relationship and use the claim
    as the description so the text-match stage sees real content.
    """
    from ...domain.models.graph_models import GraphRelationship, RelationshipType

    src = source_name or "claim_subject"
    tgt = target_name or "claim_object"
    # GraphRelationship validates that source != target, so when the
    # LLM omits both endpoints we make them disjoint sentinels.
    if src == tgt:
        tgt = f"{tgt}_target"
    return GraphRelationship(
        source_entity_id=src,
        target_entity_id=tgt,
        relationship_type=RelationshipType.RELATED_TO,
        description=description,
    )


def _ms(t0: float) -> int:
    return int((time.perf_counter() - t0) * 1000)


__all__ = [
    "GetEntityArgs",
    "SearchGraphArgs",
    "TOOL_GET_ENTITY",
    "TOOL_SEARCH_GRAPH",
    "TOOL_VERIFY_CLAIM",
    "ToolCallRecord",
    "ToolDispatcher",
    "VerifyClaimArgs",
]
