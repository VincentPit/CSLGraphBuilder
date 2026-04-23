"""Health, readiness, and pipeline metrics."""

import os
import sys
from typing import Any, Dict

from fastapi import APIRouter, Depends

from ..dependencies import get_app_config

# Make sure src/ is importable so we can read the metrics singleton.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

router = APIRouter(prefix="/health", tags=["health"])


@router.get("")
async def health() -> Dict[str, str]:
    return {"status": "ok"}


@router.get("/ready")
async def readiness(config=Depends(get_app_config)) -> Dict[str, Any]:
    """Surface configured providers — useful for the dashboard."""
    db_provider = getattr(config.database, "provider", "unknown")
    llm_provider = getattr(config.llm, "provider", "unknown")
    return {
        "status": "ready",
        "database_provider": str(db_provider),
        "llm_provider": llm_provider.value if hasattr(llm_provider, "value") else str(llm_provider),
        "llm_model": getattr(config.llm, "model_name", "unknown"),
    }


@router.get("/metrics")
async def metrics() -> Dict[str, Any]:
    """Process-wide pipeline metrics (LLM calls, cache hits, throughput)."""
    from graphbuilder.infrastructure.services.metrics import get_metrics
    from graphbuilder.infrastructure.services.cache import (
        get_dedup_cache,
        get_embedding_cache,
    )

    snapshot = get_metrics().snapshot()
    snapshot["cache_sizes"] = {
        "dedup_entries": get_dedup_cache().size(),
        "embedding_entries": get_embedding_cache().size(),
    }
    return snapshot
