"""Mutating tool surface (P10 of docs/RAG_QA_PLAN.md).

Per §14.6's resolution, every mutating tool call from the chat LLM
queues into the proposed-mutation store rather than auto-applying.
A curator promotes the proposal through ``POST /qa/proposals/{id}/apply``,
which runs :class:`MutationApplier` against the graph repo to perform
the real write.

Design split vs. P9's :mod:`tools`:

- :mod:`tools`         — read-only dispatcher (executes inline, no queue)
- :mod:`mutation_tools` — write dispatcher (validates → enqueues only)
- :mod:`mutation_applier` — pure apply logic (decoupled so the curator
  endpoint can call it without touching the chat code path)

The split keeps each surface narrow and means the read-only loop in
P9 has zero coupling to mutation infrastructure.

Tool surface (six tools per §7.2):

- ``propose_entity``           — create a new entity
- ``propose_relationship``     — create a new relationship between two ids
- ``update_entity``            — patch description / aliases / external_ids
- ``merge_entities``           — collapse two entities (curator confirms)
- ``soft_delete_entity``       — annotate verification_status="rejected"
- ``soft_delete_relationship`` — same pattern for relationships

All six produce a :class:`~graphbuilder.core.retrieval.tools.ToolCallRecord`
with ``result={"status": "queued", "proposal_id": ..., "summary": ...}``.
The LLM sees the queued status and tells the user "Proposed for
curator review". Hard delete is intentionally absent — mirrors §7.2.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel, Field, ValidationError

from .tools import ToolCallRecord


# Signature the dispatcher needs from the queue. The api/ layer wires
# this up to ``api.proposed_mutation_store.add_proposal``; tests pass a
# fake. Must return any object with a ``.proposal_id`` attribute (or a
# dict with a ``proposal_id`` key) — the dispatcher doesn't otherwise
# care about the ProposedMutation shape.
EnqueueProposalFn = Callable[..., Any]


logger = logging.getLogger("graphbuilder.qa.mutation_tools")


# ----------------------------------------------------------------------
# Tool names
# ----------------------------------------------------------------------


TOOL_PROPOSE_ENTITY = "propose_entity"
TOOL_PROPOSE_RELATIONSHIP = "propose_relationship"
TOOL_UPDATE_ENTITY = "update_entity"
TOOL_MERGE_ENTITIES = "merge_entities"
TOOL_SOFT_DELETE_ENTITY = "soft_delete_entity"
TOOL_SOFT_DELETE_RELATIONSHIP = "soft_delete_relationship"


# ----------------------------------------------------------------------
# Pydantic arg schemas
# ----------------------------------------------------------------------


class ProposeEntityArgs(BaseModel):
    """Args for ``propose_entity`` — create a new entity."""

    name: str = Field(..., min_length=1, description="Entity name (canonical form).")
    entity_type: str = Field(
        ..., min_length=1,
        description=(
            "Entity type — must match a value of EntityType "
            "(e.g. 'gene', 'drug', 'disease'). Case-insensitive."
        ),
    )
    description: Optional[str] = Field(
        None, description="Free-text description (1–3 sentences)."
    )
    aliases: List[str] = Field(default_factory=list)
    external_ids: Dict[str, str] = Field(
        default_factory=dict,
        description="Map of external system → identifier (e.g. {\"hgnc\": \"1100\"}).",
    )


class ProposeRelationshipArgs(BaseModel):
    """Args for ``propose_relationship`` — link two existing entities."""

    source_entity_id: str = Field(..., min_length=1)
    target_entity_id: str = Field(..., min_length=1)
    relationship_type: str = Field(
        ..., min_length=1,
        description=(
            "Must match a value of RelationshipType "
            "(e.g. 'inhibits', 'related_to'). Case-insensitive."
        ),
    )
    description: Optional[str] = Field(
        None, description="Free-text justification or context."
    )
    strength: float = Field(1.0, ge=0.0, le=1.0)


class UpdateEntityArgs(BaseModel):
    """Args for ``update_entity`` — patch fields on an existing entity."""

    entity_id: str = Field(..., min_length=1)
    description: Optional[str] = None
    add_aliases: List[str] = Field(default_factory=list)
    add_external_ids: Dict[str, str] = Field(default_factory=dict)
    reason: Optional[str] = Field(
        None,
        description="Why this update is warranted; surfaces in the curation UI.",
    )


class MergeEntitiesArgs(BaseModel):
    """Args for ``merge_entities`` — collapse two ids into one."""

    keep_entity_id: str = Field(
        ..., min_length=1,
        description="The entity that survives the merge (canonical id).",
    )
    merge_entity_id: str = Field(
        ..., min_length=1,
        description="The duplicate entity that gets folded into keep_entity_id.",
    )
    reason: Optional[str] = Field(
        None, description="Why these are duplicates (e.g. 'same gene, different alias').",
    )


class SoftDeleteEntityArgs(BaseModel):
    """Args for ``soft_delete_entity`` — mark verification_status=rejected."""

    entity_id: str = Field(..., min_length=1)
    reason: str = Field(
        ..., min_length=1,
        description="Why this entity should be hidden — required, not optional.",
    )


class SoftDeleteRelationshipArgs(BaseModel):
    """Args for ``soft_delete_relationship`` — same pattern for an edge."""

    relationship_id: str = Field(..., min_length=1)
    reason: str = Field(..., min_length=1)


# Tool name → schema lookup so the applier can re-validate the persisted args
# before running the apply step (the args might have been hand-edited in the
# curation UI, or stored when an older version of the schema was active).
_SCHEMA_BY_TOOL = {
    TOOL_PROPOSE_ENTITY: ProposeEntityArgs,
    TOOL_PROPOSE_RELATIONSHIP: ProposeRelationshipArgs,
    TOOL_UPDATE_ENTITY: UpdateEntityArgs,
    TOOL_MERGE_ENTITIES: MergeEntitiesArgs,
    TOOL_SOFT_DELETE_ENTITY: SoftDeleteEntityArgs,
    TOOL_SOFT_DELETE_RELATIONSHIP: SoftDeleteRelationshipArgs,
}


def schema_for(tool: str) -> Optional[type[BaseModel]]:
    """Return the Pydantic schema for a mutating tool, or None if unknown."""
    return _SCHEMA_BY_TOOL.get(tool)


def is_mutation(tool: str) -> bool:
    """True if ``tool`` is one of the mutating tools (write surface)."""
    return tool in _SCHEMA_BY_TOOL


# ----------------------------------------------------------------------
# Dispatcher
# ----------------------------------------------------------------------


class MutationToolDispatcher:
    """Validate + enqueue mutating tool calls — never writes the graph.

    Takes an ``enqueue_fn`` rather than reaching into the api/ package
    directly: this module lives under ``core/retrieval`` and shouldn't
    depend on the API layer. The router wires
    ``api.proposed_mutation_store.add_proposal`` in production; tests
    pass a fake.
    """

    def __init__(self, enqueue_fn: EnqueueProposalFn) -> None:
        self._enqueue = enqueue_fn

    async def execute(
        self,
        tool: str,
        args: Dict[str, Any],
        *,
        tool_call_id: Optional[str] = None,
        proposer_user_id: Optional[str] = None,
        proposer_session_id: Optional[str] = None,
        proposer_turn_id: Optional[str] = None,
        request_id: Optional[str] = None,
    ) -> ToolCallRecord:
        """Validate + queue. Never raises — bad args / store errors come
        back as a ``ToolCallRecord`` with ``error`` set so the LLM can
        recover."""
        t0 = time.perf_counter()
        schema = _SCHEMA_BY_TOOL.get(tool)
        if schema is None:
            return ToolCallRecord(
                tool=tool, args=args,
                error=f"unknown mutating tool: {tool}",
                latency_ms=_ms(t0), tool_call_id=tool_call_id,
            )
        try:
            parsed = schema(**args)
        except ValidationError as ve:
            return ToolCallRecord(
                tool=tool, args=args,
                error=f"invalid arguments: {ve.errors()}",
                latency_ms=_ms(t0), tool_call_id=tool_call_id,
            )

        summary = _summarise(tool, parsed)
        try:
            proposal = self._enqueue(
                tool=tool,
                args=parsed.model_dump(),
                summary=summary,
                proposer_user_id=proposer_user_id,
                proposer_session_id=proposer_session_id,
                proposer_turn_id=proposer_turn_id,
                request_id=request_id,
            )
        except Exception as exc:  # noqa: BLE001 — surfaced to LLM
            logger.warning("enqueue %s failed: %s", tool, exc, exc_info=True)
            return ToolCallRecord(
                tool=tool, args=parsed.model_dump(),
                error=f"could not enqueue proposal: {exc}",
                latency_ms=_ms(t0), tool_call_id=tool_call_id,
            )

        proposal_id = getattr(proposal, "proposal_id", None) or (
            proposal.get("proposal_id") if isinstance(proposal, dict) else None
        )
        result: Dict[str, Any] = {
            "status": "queued",
            "proposal_id": proposal_id,
            "summary": summary,
            "review_url": f"/qa/proposals/{proposal_id}" if proposal_id else None,
        }
        return ToolCallRecord(
            tool=tool, args=parsed.model_dump(), result=result,
            latency_ms=_ms(t0), tool_call_id=tool_call_id,
        )

    def openai_tool_schemas(self) -> List[Dict[str, Any]]:
        """Function-calling payload for the LLM. Same shape as the
        read-only :class:`ToolDispatcher` so the QA service can union
        the two lists into a single ``tools=[]`` array."""
        return [
            _openai_tool(
                TOOL_PROPOSE_ENTITY,
                "Propose a new entity for the knowledge graph. Goes to "
                "curator review; never auto-applied. Use only when the "
                "user explicitly asks to add an entity.",
                ProposeEntityArgs,
            ),
            _openai_tool(
                TOOL_PROPOSE_RELATIONSHIP,
                "Propose a new relationship between two existing "
                "entities. Both ids must already exist — call "
                "search_graph first if unsure. Goes to curator review.",
                ProposeRelationshipArgs,
            ),
            _openai_tool(
                TOOL_UPDATE_ENTITY,
                "Patch an existing entity's description / aliases / "
                "external_ids. Goes to curator review.",
                UpdateEntityArgs,
            ),
            _openai_tool(
                TOOL_MERGE_ENTITIES,
                "Collapse two entities into one. Use only when the "
                "user identifies them as the same concept. Goes to "
                "curator review.",
                MergeEntitiesArgs,
            ),
            _openai_tool(
                TOOL_SOFT_DELETE_ENTITY,
                "Mark an entity as rejected (hidden from default "
                "queries; node remains for audit). Reason is required.",
                SoftDeleteEntityArgs,
            ),
            _openai_tool(
                TOOL_SOFT_DELETE_RELATIONSHIP,
                "Mark a relationship as rejected. Reason is required.",
                SoftDeleteRelationshipArgs,
            ),
        ]


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _summarise(tool: str, parsed: BaseModel) -> str:
    """Build a one-line human-readable summary for the curation UI."""
    d = parsed.model_dump()
    if tool == TOOL_PROPOSE_ENTITY:
        return f"Propose entity: {d.get('name')} ({d.get('entity_type')})"
    if tool == TOOL_PROPOSE_RELATIONSHIP:
        return (
            f"Propose relationship: {d.get('source_entity_id')} "
            f"--{d.get('relationship_type')}--> {d.get('target_entity_id')}"
        )
    if tool == TOOL_UPDATE_ENTITY:
        bits: List[str] = []
        if d.get("description"):
            bits.append("description")
        if d.get("add_aliases"):
            bits.append(f"+{len(d['add_aliases'])} aliases")
        if d.get("add_external_ids"):
            bits.append(f"+{len(d['add_external_ids'])} external_ids")
        return f"Update entity {d.get('entity_id')}: " + (", ".join(bits) or "no-op")
    if tool == TOOL_MERGE_ENTITIES:
        return f"Merge {d.get('merge_entity_id')} → {d.get('keep_entity_id')}"
    if tool == TOOL_SOFT_DELETE_ENTITY:
        return f"Soft-delete entity {d.get('entity_id')}: {d.get('reason')}"
    if tool == TOOL_SOFT_DELETE_RELATIONSHIP:
        return f"Soft-delete relationship {d.get('relationship_id')}: {d.get('reason')}"
    return f"{tool}({d})"


def _openai_tool(
    name: str, description: str, model: type[BaseModel],
) -> Dict[str, Any]:
    schema = model.model_json_schema()
    schema.pop("title", None)
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": schema,
        },
    }


def _ms(t0: float) -> int:
    return int((time.perf_counter() - t0) * 1000)


__all__ = [
    "MergeEntitiesArgs",
    "MutationToolDispatcher",
    "ProposeEntityArgs",
    "ProposeRelationshipArgs",
    "SoftDeleteEntityArgs",
    "SoftDeleteRelationshipArgs",
    "TOOL_MERGE_ENTITIES",
    "TOOL_PROPOSE_ENTITY",
    "TOOL_PROPOSE_RELATIONSHIP",
    "TOOL_SOFT_DELETE_ENTITY",
    "TOOL_SOFT_DELETE_RELATIONSHIP",
    "TOOL_UPDATE_ENTITY",
    "UpdateEntityArgs",
    "is_mutation",
    "schema_for",
]
