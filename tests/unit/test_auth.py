"""Tests for api.auth — covers the X-API-Key dependency.

The dependency has three regimes:
 1. No API_KEY env var → open access (returns "anonymous").
 2. API_KEY set, header missing → 401.
 3. API_KEY set, header wrong → 403.
"""

from __future__ import annotations

import pytest
from fastapi import HTTPException

from api.auth import require_api_key


@pytest.mark.asyncio
async def test_open_access_when_api_key_unset(monkeypatch):
    monkeypatch.delenv("API_KEY", raising=False)
    result = await require_api_key(api_key=None)
    assert result == "anonymous"


@pytest.mark.asyncio
async def test_missing_header_raises_401_when_configured(monkeypatch):
    monkeypatch.setenv("API_KEY", "secret-123")
    with pytest.raises(HTTPException) as exc:
        await require_api_key(api_key=None)
    assert exc.value.status_code == 401


@pytest.mark.asyncio
async def test_wrong_header_raises_403(monkeypatch):
    monkeypatch.setenv("API_KEY", "secret-123")
    with pytest.raises(HTTPException) as exc:
        await require_api_key(api_key="wrong")
    assert exc.value.status_code == 403


@pytest.mark.asyncio
async def test_correct_header_returns_key(monkeypatch):
    monkeypatch.setenv("API_KEY", "secret-123")
    result = await require_api_key(api_key="secret-123")
    assert result == "secret-123"


@pytest.mark.asyncio
async def test_empty_string_api_key_treated_as_unset(monkeypatch):
    """`API_KEY=""` should not lock down the API — `_get_configured_key`
    coerces falsy values to None so dev defaults stay open."""
    monkeypatch.setenv("API_KEY", "")
    result = await require_api_key(api_key=None)
    assert result == "anonymous"
