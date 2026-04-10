"""Verification router — run relationship verification pipeline."""

from fastapi import APIRouter, Depends

from ..auth import require_api_key
from ..dependencies import get_app_config, get_graph_repo
from ..schemas.curation import VerificationReportResponse, VerificationEntryResponse, VerificationStageResult, VerificationRunRequest

router = APIRouter(prefix="/verification", tags=["verification"])


@router.post("/run", response_model=VerificationReportResponse)
def run_verification(
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
        use_text_match=request.use_text_match,
        use_embedding=request.use_embedding,
        use_llm=request.use_llm,
        confidence_threshold=request.confidence_threshold,
        early_exit_on_accept=True,
    )
    ver_cfg = VerificationConfig(
        cascading_config=cascading_cfg,
        relationship_ids=request.relationship_ids or None,
    )
    use_case = RelationshipVerificationUseCase(graph_repo, config)
    result = use_case.execute(ver_cfg)

    entries: list[VerificationEntryResponse] = []
    if result.success and result.data:
        for item in result.data.get("results", []):
            stages = [
                VerificationStageResult(
                    stage=s["stage"],
                    status=s["status"],
                    confidence=s.get("confidence"),
                    reason=s.get("reason"),
                )
                for s in item.get("stages", [])
            ]
            entries.append(
                VerificationEntryResponse(
                    relationship_id=item["relationship_id"],
                    final_status=item["final_status"],
                    overall_confidence=item.get("overall_confidence", 0.0),
                    stages=stages,
                    message=item.get("message"),
                )
            )

    return VerificationReportResponse(
        total=len(entries),
        verified=sum(1 for e in entries if e.final_status == "verified"),
        rejected=sum(1 for e in entries if e.final_status == "rejected"),
        unverified=sum(1 for e in entries if e.final_status == "unverified"),
        entries=entries,
    )
