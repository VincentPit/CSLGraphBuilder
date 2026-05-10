"""Apply approved chatbot proposals to the graph (P10).

The :class:`MutationToolDispatcher` only enqueues; the actual graph
write happens here, after a curator promotes a ``ProposedMutation``
through ``POST /qa/proposals/{id}/apply``. Splitting them keeps the
chat path (P9 read + P10 enqueue) entirely free of repo-write
imports, and lets the curator UI run the same apply logic through a
plain function call without going through the LLM.

Each apply handler:

1. Re-validates the persisted ``args`` against the matching schema.
   Args might have been hand-edited in the curation UI or stored when
   an older schema version was active — re-validation catches both.
2. Calls the existing ``graph_repo`` method (``save_entity`` /
   ``save_relationship`` / ``merge_entities``). No new write paths.
3. Returns the affected target id (newly created or existing) so the
   store can pin it on the row for audit.

Failures bubble up as exceptions; the router wrapper catches them and
records ``apply_error`` on the proposal so the curator UI can show a
"retry" button without losing the row.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from .mutation_tools import (
    TOOL_MERGE_ENTITIES,
    TOOL_PROPOSE_ENTITY,
    TOOL_PROPOSE_RELATIONSHIP,
    TOOL_SOFT_DELETE_ENTITY,
    TOOL_SOFT_DELETE_RELATIONSHIP,
    TOOL_UPDATE_ENTITY,
    schema_for,
)


logger = logging.getLogger("graphbuilder.qa.mutation_applier")


# Marker stored on the entity / relationship's metadata.annotations
# dict when soft-deleted. Mirrors the existing curation queue
# convention referenced in graph_repository.py:1549.
_REJECTED = "rejected"


class MutationApplier:
    """Applies approved chatbot proposals to the graph.

    Construction is cheap and stateless. Pass the same ``graph_repo``
    the rest of the API uses; we don't reach for a fresh instance.
    """

    def __init__(self, graph_repo: Any) -> None:
        self._graph_repo = graph_repo

    async def apply(self, *, tool: str, args: Dict[str, Any]) -> str:
        """Run the apply handler for ``tool`` and return the target id.

        Returns the id of the entity or relationship that was created /
        updated / soft-deleted. ``merge_entities`` returns the surviving
        ``keep_entity_id``.
        """
        schema = schema_for(tool)
        if schema is None:
            raise ValueError(f"unknown mutating tool: {tool}")
        # Re-validate the persisted args. Curators can edit them, and
        # an older schema's saved row should fail loudly here rather
        # than corrupting the graph.
        parsed = schema(**args)

        if tool == TOOL_PROPOSE_ENTITY:
            return await self._apply_propose_entity(parsed.model_dump())
        if tool == TOOL_PROPOSE_RELATIONSHIP:
            return await self._apply_propose_relationship(parsed.model_dump())
        if tool == TOOL_UPDATE_ENTITY:
            return await self._apply_update_entity(parsed.model_dump())
        if tool == TOOL_MERGE_ENTITIES:
            return await self._apply_merge_entities(parsed.model_dump())
        if tool == TOOL_SOFT_DELETE_ENTITY:
            return await self._apply_soft_delete_entity(parsed.model_dump())
        if tool == TOOL_SOFT_DELETE_RELATIONSHIP:
            return await self._apply_soft_delete_relationship(parsed.model_dump())
        raise ValueError(f"no apply handler for tool: {tool}")

    # ------------------------------------------------------------------
    # Per-tool implementations
    # ------------------------------------------------------------------

    async def _apply_propose_entity(self, args: Dict[str, Any]) -> str:
        from ...domain.models.graph_models import EntityType, GraphEntity

        entity = GraphEntity(
            name=args["name"],
            entity_type=_coerce_entity_type(args["entity_type"]),
            description=args.get("description"),
            external_ids=dict(args.get("external_ids") or {}),
        )
        for alias in args.get("aliases") or []:
            entity.add_alias(alias)
        # Mark provenance — chatbot proposals carry the proposer thread
        # via the store row, but the entity itself should also record
        # that it landed via the curation path so downstream queries can
        # filter on ``annotations.verification_status``.
        entity.metadata.annotations["verification_status"] = "curated"
        entity.metadata.annotations["origin"] = "chatbot_proposal"
        saved = await self._graph_repo.save_entity(entity)
        return saved.id

    async def _apply_propose_relationship(self, args: Dict[str, Any]) -> str:
        from ...domain.models.graph_models import (
            GraphRelationship, RelationshipType,
        )

        rel = GraphRelationship(
            source_entity_id=args["source_entity_id"],
            target_entity_id=args["target_entity_id"],
            relationship_type=_coerce_relationship_type(args["relationship_type"]),
            description=args.get("description"),
            strength=float(args.get("strength", 1.0)),
        )
        rel.metadata.annotations["verification_status"] = "curated"
        rel.metadata.annotations["origin"] = "chatbot_proposal"
        saved = await self._graph_repo.save_relationship(rel)
        return saved.id

    async def _apply_update_entity(self, args: Dict[str, Any]) -> str:
        existing = await self._graph_repo.get_entity_by_id(args["entity_id"])
        if existing is None:
            raise LookupError(f"entity not found: {args['entity_id']}")
        # Patch in place — save_entity is upsert-shaped (matches on
        # name + entity_type) so we keep those steady and update the
        # mutable fields.
        if args.get("description"):
            existing.description = args["description"]
        for alias in args.get("add_aliases") or []:
            existing.add_alias(alias)
        for system, ext_id in (args.get("add_external_ids") or {}).items():
            existing.add_external_id(system, ext_id)
        if args.get("reason"):
            existing.metadata.annotations["last_update_reason"] = args["reason"]
        existing.metadata.annotations["origin"] = "chatbot_proposal"
        saved = await self._graph_repo.save_entity(existing)
        return saved.id

    async def _apply_merge_entities(self, args: Dict[str, Any]) -> str:
        # ``merge_entities`` is a repo-level method (graph_repository.py).
        # It rewrites incoming/outgoing edges to the surviving entity
        # and unions provenance — mirrors the dedup CLI.
        merge_fn = getattr(self._graph_repo, "merge_entities", None)
        if merge_fn is None:
            raise NotImplementedError(
                "graph_repo.merge_entities is required for the merge_entities tool"
            )
        await merge_fn(
            keep_entity_id=args["keep_entity_id"],
            merge_entity_id=args["merge_entity_id"],
        )
        return args["keep_entity_id"]

    async def _apply_soft_delete_entity(self, args: Dict[str, Any]) -> str:
        existing = await self._graph_repo.get_entity_by_id(args["entity_id"])
        if existing is None:
            raise LookupError(f"entity not found: {args['entity_id']}")
        existing.metadata.annotations["verification_status"] = _REJECTED
        existing.metadata.annotations["rejection_reason"] = args["reason"]
        existing.metadata.annotations["origin"] = "chatbot_proposal"
        saved = await self._graph_repo.save_entity(existing)
        return saved.id

    async def _apply_soft_delete_relationship(self, args: Dict[str, Any]) -> str:
        # The graph repo doesn't currently expose a "fetch one
        # relationship by id" + "save" round-trip the way it does for
        # entities, so we delegate to a method on the repo when one
        # exists, else surface the gap honestly. This keeps the apply
        # path easy to fix in a follow-up without changing the tool
        # surface or the curator UI contract.
        soft_delete_fn = getattr(
            self._graph_repo, "soft_delete_relationship", None,
        )
        if soft_delete_fn is None:
            raise NotImplementedError(
                "graph_repo.soft_delete_relationship is not yet implemented; "
                "this proposal must be rejected or applied manually"
            )
        await soft_delete_fn(
            relationship_id=args["relationship_id"],
            reason=args["reason"],
        )
        return args["relationship_id"]


# ----------------------------------------------------------------------
# Enum coercion — accept both lowercase strings ("gene") and the
# canonical enum value ("GENE") so the LLM doesn't have to memorise
# the exact casing.
# ----------------------------------------------------------------------


def _coerce_entity_type(raw: str) -> Any:
    from ...domain.models.graph_models import EntityType

    try:
        return EntityType(raw)
    except ValueError:
        pass
    upper = raw.upper()
    try:
        return EntityType(upper)
    except ValueError:
        pass
    # Last-chance: title-case ("Person", "Organization").
    title = raw.title()
    try:
        return EntityType(title)
    except ValueError as exc:
        raise ValueError(
            f"unknown entity_type: {raw!r}. Valid values: "
            f"{[e.value for e in EntityType]}"
        ) from exc


def _coerce_relationship_type(raw: str) -> Any:
    from ...domain.models.graph_models import RelationshipType

    upper = raw.upper().replace(" ", "_")
    try:
        return RelationshipType(upper)
    except ValueError as exc:
        raise ValueError(
            f"unknown relationship_type: {raw!r}. Valid values: "
            f"{[e.value for e in RelationshipType]}"
        ) from exc


__all__ = ["MutationApplier"]
