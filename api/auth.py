"""API key authentication middleware."""

import os
from typing import Optional

from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

_API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)
_REQUIRED_IN_PRODUCTION = True


def _get_configured_key() -> Optional[str]:
    return os.environ.get("API_KEY") or None


def _is_production() -> bool:
    return os.environ.get("ENVIRONMENT", "development").lower() == "production"


async def require_api_key(api_key: Optional[str] = Security(_API_KEY_HEADER)) -> str:
    """
    Dependency that validates the X-API-Key header.

    * Dev / staging: key is optional — if no API_KEY env var is set, all
      requests pass through.
    * Production: key is mandatory and must match API_KEY env var.
    """
    configured_key = _get_configured_key()

    # No key configured → open access (dev / local / test)
    if configured_key is None:
        return "anonymous"

    if api_key is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="X-API-Key header is required.",
        )

    if api_key != configured_key:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key.",
        )

    return api_key
