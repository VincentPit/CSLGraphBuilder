"""Health-check router."""

from fastapi import APIRouter, Depends
from typing import Annotated

from ..dependencies import get_app_config, get_graph_repo

router = APIRouter(prefix="/health", tags=["health"])


@router.get("")
async def health():
    return {"status": "ok"}


@router.get("/ready")
async def readiness(config=Depends(get_app_config)):
    """Check that config loaded successfully."""
    return {
        "status": "ready",
        "database_provider": getattr(config.database, "provider", "unknown"),
        "llm_provider": getattr(config.llm, "provider", {}).value
        if hasattr(getattr(config.llm, "provider", None), "value")
        else str(getattr(config.llm, "provider", "unknown")),
    }
