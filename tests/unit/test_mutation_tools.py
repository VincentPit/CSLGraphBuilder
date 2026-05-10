"""Unit tests for the mutating tool surface (P10 of docs/RAG_QA_PLAN.md).

Three layers exercised here:

1. ``proposed_mutation_store`` — process-scoped queue (add/get/list/decide/apply).
2. ``MutationToolDispatcher`` — schema-validates and enqueues; never writes.
3. ``MutationApplier``         — per-tool apply handlers against graph_repo.

Router-level tests (curator approve/reject endpoints) live in
``test_qa_router.py``.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import pytest

os.environ.setdefault("LLM_API_KEY", "not-configured")

from api import proposed_mutation_store as store  # noqa: E402
from graphbuilder.core.retrieval.mutation_applier import MutationApplier  # noqa: E402
from graphbuilder.core.retrieval.mutation_tools import (  # noqa: E402
    TOOL_MERGE_ENTITIES,
    TOOL_PROPOSE_ENTITY,
    TOOL_PROPOSE_RELATIONSHIP,
    TOOL_SOFT_DELETE_ENTITY,
    TOOL_UPDATE_ENTITY,
    MutationToolDispatcher,
    is_mutation,
    schema_for,
)
from graphbuilder.domain.models.graph_models import (  # noqa: E402
    EntityType,
    GraphEntity,
    GraphRelationship,
    RelationshipType,
)


# ---------------------------------------------------------------- store


@pytest.fixture(autouse=True)
def _reset_store():
    store.reset_for_tests()
    yield
    store.reset_for_tests()


class TestProposedMutationStore:
    def test_add_then_get_round_trip(self):
        p = store.add_proposal(
            tool=TOOL_PROPOSE_ENTITY,
            args={"name": "BRCA1", "entity_type": "GENE"},
            summary="Propose entity: BRCA1 (GENE)",
            proposer_user_id="u_1",
        )
        assert p.proposal_id.startswith("prop_")
        assert p.status == "pending"
        assert p.proposer_user_id == "u_1"

        fetched = store.get_proposal(p.proposal_id)
        assert fetched is not None
        assert fetched.tool == TOOL_PROPOSE_ENTITY

    def test_list_filters_by_status(self):
        a = store.add_proposal(tool=TOOL_PROPOSE_ENTITY,
                               args={"name": "A", "entity_type": "GENE"},
                               summary="A")
        b = store.add_proposal(tool=TOOL_PROPOSE_ENTITY,
                               args={"name": "B", "entity_type": "GENE"},
                               summary="B")
        store.mark_decided(a.proposal_id, "approved")

        pending = store.list_proposals(status="pending")
        assert {p.proposal_id for p in pending} == {b.proposal_id}

        all_rows = store.list_proposals(status=None)
        assert {p.proposal_id for p in all_rows} == {a.proposal_id, b.proposal_id}

    def test_mark_decided_invalid_status_raises(self):
        p = store.add_proposal(tool=TOOL_PROPOSE_ENTITY,
                               args={"name": "X", "entity_type": "GENE"},
                               summary="X")
        with pytest.raises(ValueError):
            store.mark_decided(p.proposal_id, "approveeed")  # type: ignore[arg-type]

    def test_mark_applied_pins_target_and_clears_error(self):
        p = store.add_proposal(tool=TOOL_PROPOSE_ENTITY,
                               args={"name": "X", "entity_type": "GENE"},
                               summary="X")
        store.mark_decided(p.proposal_id, "approved")
        store.mark_applied(p.proposal_id, target_id="ent_X_id")
        row = store.get_proposal(p.proposal_id)
        assert row.applied_target_id == "ent_X_id"
        assert row.applied_at is not None
        assert row.apply_error is None

    def test_mark_applied_with_error_does_not_set_applied_at(self):
        p = store.add_proposal(tool=TOOL_PROPOSE_ENTITY,
                               args={"name": "X", "entity_type": "GENE"},
                               summary="X")
        store.mark_decided(p.proposal_id, "approved")
        store.mark_applied(p.proposal_id, error="boom")
        row = store.get_proposal(p.proposal_id)
        assert row.apply_error == "boom"
        assert row.applied_at is None
        assert row.applied_target_id is None


# ---------------------------------------------------------------- dispatcher


@pytest.mark.asyncio
class TestMutationToolDispatcher:
    async def test_propose_entity_validates_and_enqueues(self):
        captured: List[dict] = []

        def _enqueue(**kwargs):
            captured.append(kwargs)
            return store.add_proposal(**kwargs)

        disp = MutationToolDispatcher(enqueue_fn=_enqueue)
        rec = await disp.execute(
            TOOL_PROPOSE_ENTITY,
            {"name": "BRCA1", "entity_type": "GENE", "description": "DNA repair"},
            tool_call_id="call_1",
            proposer_user_id="u_1",
            proposer_session_id="s_1",
        )
        assert rec.ok
        assert rec.result["status"] == "queued"
        assert rec.result["proposal_id"].startswith("prop_")
        # Provenance threaded through to the store.
        assert captured[0]["proposer_user_id"] == "u_1"
        assert captured[0]["proposer_session_id"] == "s_1"

        # Row exists in the store with the validated args.
        row = store.get_proposal(rec.result["proposal_id"])
        assert row is not None
        assert row.tool == TOOL_PROPOSE_ENTITY
        assert row.args["name"] == "BRCA1"
        assert row.summary.startswith("Propose entity: BRCA1")

    async def test_invalid_args_returns_error_record(self):
        disp = MutationToolDispatcher(enqueue_fn=store.add_proposal)
        rec = await disp.execute(
            TOOL_PROPOSE_ENTITY,
            {"name": "", "entity_type": "GENE"},  # empty name violates min_length
        )
        assert not rec.ok
        assert "invalid arguments" in rec.error
        # Nothing queued on validation failure.
        assert store.list_proposals() == []

    async def test_unknown_tool_name(self):
        disp = MutationToolDispatcher(enqueue_fn=store.add_proposal)
        rec = await disp.execute("hard_delete_everything", {})
        assert not rec.ok
        assert "unknown mutating tool" in rec.error

    async def test_enqueue_failure_surfaces_in_record(self):
        def _bad_enqueue(**kwargs):
            raise RuntimeError("queue is down")

        disp = MutationToolDispatcher(enqueue_fn=_bad_enqueue)
        rec = await disp.execute(
            TOOL_SOFT_DELETE_ENTITY,
            {"entity_id": "e1", "reason": "duplicate"},
        )
        assert not rec.ok
        assert "could not enqueue" in rec.error

    async def test_soft_delete_requires_reason(self):
        disp = MutationToolDispatcher(enqueue_fn=store.add_proposal)
        rec = await disp.execute(TOOL_SOFT_DELETE_ENTITY, {"entity_id": "e1"})
        # Missing required ``reason`` → schema rejects.
        assert not rec.ok

    async def test_openai_tool_schemas_includes_all_six(self):
        disp = MutationToolDispatcher(enqueue_fn=store.add_proposal)
        names = {s["function"]["name"] for s in disp.openai_tool_schemas()}
        assert names == {
            TOOL_PROPOSE_ENTITY,
            TOOL_PROPOSE_RELATIONSHIP,
            TOOL_UPDATE_ENTITY,
            TOOL_MERGE_ENTITIES,
            TOOL_SOFT_DELETE_ENTITY,
            "soft_delete_relationship",
        }


def test_is_mutation_separates_read_vs_write():
    assert is_mutation(TOOL_PROPOSE_ENTITY) is True
    assert is_mutation(TOOL_SOFT_DELETE_ENTITY) is True
    assert is_mutation("search_graph") is False
    assert is_mutation("get_entity") is False


def test_schema_for_returns_pydantic_model():
    schema = schema_for(TOOL_PROPOSE_ENTITY)
    assert schema is not None
    parsed = schema(name="X", entity_type="GENE")
    assert parsed.name == "X"


# ---------------------------------------------------------------- applier


class _RecordingGraphRepo:
    """In-memory graph repo just rich enough to exercise apply paths.

    Tracks save_entity / save_relationship / merge_entities calls so
    tests can assert the right method fired with the right shape.
    """

    def __init__(self) -> None:
        self._entities: Dict[str, GraphEntity] = {}
        self._rels: Dict[str, GraphRelationship] = {}
        self.save_entity_calls: List[GraphEntity] = []
        self.save_relationship_calls: List[GraphRelationship] = []
        self.merge_calls: List[Dict[str, Any]] = []
        self.soft_delete_rel_calls: List[Dict[str, Any]] = []

    async def save_entity(self, entity: GraphEntity, **_kw) -> GraphEntity:
        self.save_entity_calls.append(entity)
        self._entities[entity.id] = entity
        return entity

    async def save_relationship(self, rel: GraphRelationship, **_kw) -> GraphRelationship:
        self.save_relationship_calls.append(rel)
        self._rels[rel.id] = rel
        return rel

    async def get_entity_by_id(self, entity_id: str) -> Optional[GraphEntity]:
        return self._entities.get(entity_id)

    async def merge_entities(self, *, keep_entity_id: str, merge_entity_id: str):
        self.merge_calls.append({
            "keep_entity_id": keep_entity_id,
            "merge_entity_id": merge_entity_id,
        })
        # Drop the merged-away entity if it's known.
        self._entities.pop(merge_entity_id, None)

    async def soft_delete_relationship(self, *, relationship_id: str, reason: str):
        self.soft_delete_rel_calls.append({
            "relationship_id": relationship_id, "reason": reason,
        })


@pytest.mark.asyncio
class TestMutationApplier:
    async def test_apply_propose_entity_creates_via_save_entity(self):
        repo = _RecordingGraphRepo()
        applier = MutationApplier(graph_repo=repo)
        target_id = await applier.apply(
            tool=TOOL_PROPOSE_ENTITY,
            args={
                "name": "BRCA1", "entity_type": "GENE",
                "description": "DNA repair", "aliases": ["BRCA-1"],
                "external_ids": {"hgnc": "1100"},
            },
        )
        assert len(repo.save_entity_calls) == 1
        saved = repo.save_entity_calls[0]
        assert saved.name == "BRCA1"
        assert saved.entity_type == EntityType.GENE
        assert saved.description == "DNA repair"
        assert "BRCA-1" in saved.aliases
        assert saved.external_ids == {"hgnc": "1100"}
        # Provenance markers land on metadata.annotations.
        assert saved.metadata.annotations["origin"] == "chatbot_proposal"
        assert saved.metadata.annotations["verification_status"] == "curated"
        assert target_id == saved.id

    async def test_apply_propose_entity_accepts_lowercase_type(self):
        repo = _RecordingGraphRepo()
        applier = MutationApplier(graph_repo=repo)
        await applier.apply(
            tool=TOOL_PROPOSE_ENTITY,
            args={"name": "BRCA1", "entity_type": "gene"},
        )
        # Lowercase coerced to the canonical EntityType.
        assert repo.save_entity_calls[0].entity_type == EntityType.GENE

    async def test_apply_propose_entity_unknown_type_raises(self):
        repo = _RecordingGraphRepo()
        applier = MutationApplier(graph_repo=repo)
        with pytest.raises(ValueError):
            await applier.apply(
                tool=TOOL_PROPOSE_ENTITY,
                args={"name": "BRCA1", "entity_type": "wizardry"},
            )

    async def test_apply_propose_relationship_creates_via_save_relationship(self):
        repo = _RecordingGraphRepo()
        applier = MutationApplier(graph_repo=repo)
        await applier.apply(
            tool=TOOL_PROPOSE_RELATIONSHIP,
            args={
                "source_entity_id": "ent_a",
                "target_entity_id": "ent_b",
                "relationship_type": "RELATED_TO",
                "description": "x",
                "strength": 0.8,
            },
        )
        assert len(repo.save_relationship_calls) == 1
        saved = repo.save_relationship_calls[0]
        assert saved.source_entity_id == "ent_a"
        assert saved.target_entity_id == "ent_b"
        assert saved.relationship_type == RelationshipType.RELATED_TO
        assert saved.strength == 0.8

    async def test_apply_update_entity_patches_existing(self):
        e = GraphEntity(name="BRCA1", entity_type=EntityType.GENE,
                        description="old description")
        e.id = "ent_brca1"
        repo = _RecordingGraphRepo()
        repo._entities["ent_brca1"] = e

        applier = MutationApplier(graph_repo=repo)
        await applier.apply(
            tool=TOOL_UPDATE_ENTITY,
            args={
                "entity_id": "ent_brca1",
                "description": "new description",
                "add_aliases": ["BRCA-1"],
                "add_external_ids": {"hgnc": "1100"},
                "reason": "external sync",
            },
        )
        saved = repo.save_entity_calls[0]
        assert saved.description == "new description"
        assert "BRCA-1" in saved.aliases
        assert saved.external_ids["hgnc"] == "1100"
        assert saved.metadata.annotations["last_update_reason"] == "external sync"

    async def test_apply_update_entity_missing_target_raises(self):
        repo = _RecordingGraphRepo()
        applier = MutationApplier(graph_repo=repo)
        with pytest.raises(LookupError):
            await applier.apply(
                tool=TOOL_UPDATE_ENTITY,
                args={"entity_id": "nope", "description": "x"},
            )

    async def test_apply_merge_entities_calls_repo_merge(self):
        repo = _RecordingGraphRepo()
        applier = MutationApplier(graph_repo=repo)
        target_id = await applier.apply(
            tool=TOOL_MERGE_ENTITIES,
            args={
                "keep_entity_id": "ent_a",
                "merge_entity_id": "ent_b",
                "reason": "duplicates",
            },
        )
        assert repo.merge_calls == [{
            "keep_entity_id": "ent_a", "merge_entity_id": "ent_b",
        }]
        assert target_id == "ent_a"  # surviving id

    async def test_apply_soft_delete_entity_marks_rejected(self):
        e = GraphEntity(name="X", entity_type=EntityType.GENE)
        e.id = "ent_x"
        repo = _RecordingGraphRepo()
        repo._entities["ent_x"] = e
        applier = MutationApplier(graph_repo=repo)
        await applier.apply(
            tool=TOOL_SOFT_DELETE_ENTITY,
            args={"entity_id": "ent_x", "reason": "test artifact"},
        )
        saved = repo.save_entity_calls[0]
        assert saved.metadata.annotations["verification_status"] == "rejected"
        assert saved.metadata.annotations["rejection_reason"] == "test artifact"

    async def test_apply_soft_delete_relationship_calls_repo_method(self):
        repo = _RecordingGraphRepo()
        applier = MutationApplier(graph_repo=repo)
        await applier.apply(
            tool="soft_delete_relationship",
            args={"relationship_id": "rel_42", "reason": "wrong"},
        )
        assert repo.soft_delete_rel_calls == [
            {"relationship_id": "rel_42", "reason": "wrong"}
        ]

    async def test_apply_unknown_tool_raises(self):
        repo = _RecordingGraphRepo()
        applier = MutationApplier(graph_repo=repo)
        with pytest.raises(ValueError):
            await applier.apply(tool="who_knows", args={})

    async def test_apply_soft_delete_relationship_without_repo_method_raises(self):
        class _MinimalRepo:
            async def get_entity_by_id(self, _): return None
            async def save_entity(self, e, **_): return e
            async def save_relationship(self, r, **_): return r
        applier = MutationApplier(graph_repo=_MinimalRepo())
        with pytest.raises(NotImplementedError):
            await applier.apply(
                tool="soft_delete_relationship",
                args={"relationship_id": "r", "reason": "x"},
            )
