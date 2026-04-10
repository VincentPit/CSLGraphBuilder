"""Verification router — run relationship verification pipeline."""

from fastapi import APIRouter, Depends

from ..auth import require_api_key
from ..dependencies import get_app_config, get_graph_repo
from ..schemas.curation import VerificationReportResponse, VerificationEntryResponse, VerificationStageResult, VerificationRunRequest

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

    use_case = RelationshipVerificationUseCase(kg)
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
