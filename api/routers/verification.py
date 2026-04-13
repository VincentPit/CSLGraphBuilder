"""Verification router — run relationship verification pipeline."""

from fastapi import APIRouter, Depends

from ..auth import require_api_key
from ..dependencies import get_app_config, get_graph_repo
from ..schemas.curation import (
    VerificationReportResponse, VerificationEntryResponse, VerificationStageResult, VerificationRunRequest,
    TextVerificationRequest, TextVerificationResponse, TextVerificationEntryResponse, TextVerificationStageResult,
)
from ..schemas.graph import (
    ConflictCheckRequest, ConflictCheckResponse, ConflictEntryResponse,
    PendingReviewItem, PendingReviewListResponse, ReviewDecisionRequest,
)

router = APIRouter(prefix="/verification", tags=["verification"])


@router.post("/run", response_model=VerificationReportResponse)
async def run_verification(
    request: VerificationRunRequest,
    config=Depends(get_app_config),
    graph_repo=Depends(get_graph_repo),
    _=Depends(require_api_key),
):
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
    from graphbuilder.application.use_cases.relationship_verification import (
        RelationshipVerificationUseCase, VerificationConfig,
    )
    from graphbuilder.core.verification.cascading import CascadingVerifierConfig

    cascading_cfg = CascadingVerifierConfig(
        enable_text_match=True,
        enable_embedding=request.enable_embedding,
        enable_llm=request.enable_llm,
        early_exit_on_pass=request.early_exit_on_pass,
        early_exit_on_fail=request.early_exit_on_fail,
    )
    ver_cfg = VerificationConfig(
        cascading=cascading_cfg,
        context_map=request.context_map,
    )

    # Build a KnowledgeGraph containing only the requested relationships
    from graphbuilder.domain.models.graph_models import KnowledgeGraph
    requested = set(request.relationship_ids)
    kg = KnowledgeGraph()

    all_rels = await graph_repo.get_all_relationships()
    for rel in all_rels.values():

        if rel.id in requested:
            kg.relationships[rel.id] = rel

    all_entities = await graph_repo.get_all_entities()
    needed_ids = set()
    for rel in kg.relationships.values():
        needed_ids.add(rel.source_entity_id)
        needed_ids.add(rel.target_entity_id)
    for ent in all_entities.values():
        if ent.id in needed_ids:
            kg.entities[ent.id] = ent

    use_case = RelationshipVerificationUseCase(kg, graph_repo=graph_repo)
    result = use_case.execute(ver_cfg)

    entries: list[VerificationEntryResponse] = []
    if result.success and result.data:
        for item in result.data.get("report", []):
            stages = [
                VerificationStageResult(
                    stage=s["stage"],
                    status=s["status"],
                    confidence=s.get("confidence", 0.0),
                    reasoning=s.get("reasoning", ""),
                    metadata=s.get("metadata"),
                )
                for s in item.get("stage_results", [])
            ]
            entries.append(
                VerificationEntryResponse(
                    relationship_id=item["relationship_id"],
                    source_entity_id=item.get("source_entity_id", ""),
                    target_entity_id=item.get("target_entity_id", ""),
                    relationship_type=item.get("relationship_type", ""),
                    status=item["status"],
                    confidence=item.get("confidence", 0.0),
                    reasoning=item.get("reasoning", ""),
                    stage_results=stages,
                )
            )

    return VerificationReportResponse(
        total=result.data.get("total", len(entries)) if result.data else len(entries),
        passed=result.data.get("passed", 0) if result.data else 0,
        failed=result.data.get("failed", 0) if result.data else 0,
        skipped=result.data.get("skipped", 0) if result.data else 0,
        report=entries,
    )


@router.post("/text", response_model=TextVerificationResponse)
async def verify_text(
    request: TextVerificationRequest,
    config=Depends(get_app_config),
    graph_repo=Depends(get_graph_repo),
    _=Depends(require_api_key),
):
    """Verify a free-text description against the knowledge graph."""
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
    from graphbuilder.application.use_cases.text_verification import (
        TextVerificationUseCase, TextVerificationConfig, extract_search_terms,
    )
    from graphbuilder.core.verification.cascading import CascadingVerifierConfig
    from graphbuilder.domain.models.graph_models import KnowledgeGraph

    # Extract search terms and find candidate entities
    terms = extract_search_terms(request.text)
    if not terms:
        return TextVerificationResponse(
            query_text=request.text,
            total_candidates=0,
            verified=0,
            not_verified=0,
            skipped=0,
            best_confidence=0.0,
            entries=[],
        )

    matched_entities = await graph_repo.search_entities_by_text(terms, limit=50)

    # Find relationships connected to matched entities
    kg = KnowledgeGraph()
    kg.entities = matched_entities
    matched_entity_ids = set(matched_entities.keys())

    all_rels = await graph_repo.get_all_relationships()
    for rel in all_rels.values():
        if rel.source_entity_id in matched_entity_ids or rel.target_entity_id in matched_entity_ids:
            kg.relationships[rel.id] = rel

    # Pull in any missing entities referenced by relationships
    all_entities = await graph_repo.get_all_entities()
    for rel in kg.relationships.values():
        for eid in (rel.source_entity_id, rel.target_entity_id):
            if eid not in kg.entities and eid in all_entities:
                kg.entities[eid] = all_entities[eid]

    cascading_cfg = CascadingVerifierConfig(
        enable_text_match=True,
        enable_embedding=request.enable_embedding,
        enable_llm=request.enable_llm,
        early_exit_on_pass=request.early_exit_on_pass,
        early_exit_on_fail=request.early_exit_on_fail,
    )
    ver_cfg = TextVerificationConfig(
        cascading=cascading_cfg,
        max_candidates=request.max_candidates,
    )

    use_case = TextVerificationUseCase(kg, graph_repo=graph_repo)
    report = use_case.execute(request.text, ver_cfg)

    entries = [
        TextVerificationEntryResponse(
            relationship_id=e.relationship_id,
            source_entity_id=e.source_entity_id,
            target_entity_id=e.target_entity_id,
            source_entity_name=e.source_entity_name,
            target_entity_name=e.target_entity_name,
            relationship_type=e.relationship_type,
            relationship_description=e.relationship_description,
            status=e.status,
            confidence=e.confidence,
            reasoning=e.reasoning,
            stage_results=[
                TextVerificationStageResult(**s) for s in e.stage_results
            ],
        )
        for e in report.entries
    ]

    return TextVerificationResponse(
        query_text=report.query_text,
        total_candidates=report.total_candidates,
        verified=report.verified,
        not_verified=report.not_verified,
        skipped=report.skipped,
        best_confidence=report.best_confidence,
        entries=entries,
    )


@router.post("/conflicts", response_model=ConflictCheckResponse)
async def check_conflicts(
    request: ConflictCheckRequest,
    config=Depends(get_app_config),
    graph_repo=Depends(get_graph_repo),
    _=Depends(require_api_key),
):
    """Check a free-text claim for conflicts against existing knowledge graph."""
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
    from graphbuilder.application.use_cases.text_verification import extract_search_terms
    from graphbuilder.application.use_cases.conflict_detection import KnowledgeConflictDetector
    from graphbuilder.domain.models.graph_models import (
        KnowledgeGraph, GraphRelationship, RelationshipType, Metadata,
    )

    # Build a KnowledgeGraph from existing data
    kg = KnowledgeGraph()
    all_entities = await graph_repo.get_all_entities()
    all_rels = await graph_repo.get_all_relationships()
    kg.entities = dict(all_entities)
    kg.relationships = dict(all_rels)

    # Create a synthetic relationship from the user's claim
    terms = extract_search_terms(request.text)

    # Find entities matching the terms to build candidate pairs
    matched_entity_ids = set()
    for eid, ent in kg.entities.items():
        name_lower = ent.name.lower()
        for term in terms:
            if term.lower() in name_lower:
                matched_entity_ids.add(eid)
                break

    # Build synthetic new relationships between matched entity pairs
    from itertools import combinations
    new_rels: list[GraphRelationship] = []
    matched_list = list(matched_entity_ids)

    for i, src_id in enumerate(matched_list):
        for tgt_id in matched_list[i + 1:]:
            new_rels.append(GraphRelationship(
                source_entity_id=src_id,
                target_entity_id=tgt_id,
                relationship_type=RelationshipType.RELATED_TO,
                description=request.text,
                strength=1.0,
                metadata=Metadata(),
            ))

    detector = KnowledgeConflictDetector(kg)
    report = detector.check_conflicts(
        new_rels,
        use_llm=request.use_llm,
    )

    # Build response entries and auto-queue trust conflicts for review
    from ..review_store import add_review

    conflict_entries = []
    for c in report.conflicts:
        entry = ConflictEntryResponse(
            conflict_type=c.conflict_type,
            severity=c.severity,
            existing_relationship_id=c.existing_relationship_id,
            existing_relationship_type=c.existing_relationship_type,
            existing_description=c.existing_description,
            existing_source_chunk_ids=c.existing_source_chunk_ids,
            existing_source_trust=c.existing_source_trust,
            new_relationship_type=c.new_relationship_type,
            new_description=c.new_description,
            new_source_chunk_ids=c.new_source_chunk_ids,
            new_source_trust=c.new_source_trust,
            source_entity_name=c.source_entity_name,
            target_entity_name=c.target_entity_name,
            reasoning=c.reasoning,
            requires_review=c.requires_review,
        )
        conflict_entries.append(entry)

        # Auto-queue conflicts that need review (lower-trust vs higher-trust)
        if c.requires_review:
            add_review(entry.model_dump())

    return ConflictCheckResponse(
        total_checked=report.total_checked,
        conflicts_found=report.conflicts_found,
        conflicts=conflict_entries,
    )


# ── Pending Review Queue ────────────────────────────────────────────────

@router.get("/reviews", response_model=PendingReviewListResponse)
async def list_pending_reviews(
    status: str = "pending",
    _=Depends(require_api_key),
):
    """List pending review items (trust-conflicted knowledge awaiting user decision)."""
    from ..review_store import get_pending_reviews

    reviews = get_pending_reviews(status=status if status != "all" else None)
    items = [
        PendingReviewItem(
            review_id=r.review_id,
            conflict=ConflictEntryResponse(**r.conflict_data),
            submitted_at=r.submitted_at,
            status=r.status,
        )
        for r in reviews
    ]
    return PendingReviewListResponse(total=len(items), items=items)


@router.post("/reviews/decide")
async def decide_review(
    request: ReviewDecisionRequest,
    config=Depends(get_app_config),
    graph_repo=Depends(get_graph_repo),
    _=Depends(require_api_key),
):
    """Approve or reject a pending review item.

    - **approve**: The new (lower-trust) claim is accepted and injected into the graph.
    - **reject**: The new claim is discarded; existing trusted knowledge is kept.
    """
    from ..review_store import decide_review as _decide

    review = _decide(request.review_id, request.decision, request.notes)
    if review is None:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Review not found")

    return {"review_id": review.review_id, "status": review.status, "notes": review.notes}
