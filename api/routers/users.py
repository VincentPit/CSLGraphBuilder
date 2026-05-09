"""Users router — lightweight browser-identity registration / lookup
for the chatbot (§14.1 of docs/RAG_QA_PLAN.md, revised 2026-05-09).

No password / no email. The browser POSTs ``{display_name}`` once on
first visit, gets back a stable ``user_id``, and stuffs it into
localStorage. Every subsequent /qa/* request carries the id in the
``X-User-Id`` header (resolved server-side by ``get_chat_user_id`` in
api/dependencies_qa.py).
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Path, status

from ..auth import require_api_key
from ..dependencies import get_user_repo
from ..schemas.users import (
    CreateUserRequest,
    UpdateUserRequest,
    UserResponse,
)


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from graphbuilder.domain.models.user_models import User  # noqa: E402


logger = logging.getLogger("graphbuilder.qa.api")

router = APIRouter(prefix="/users", tags=["users"])


def _to_response(user: User) -> UserResponse:
    return UserResponse(
        id=user.id,
        display_name=user.display_name,
        metadata=dict(user.metadata),
        created_at=user.created_at.isoformat(),
        last_seen_at=user.last_seen_at.isoformat(),
    )


@router.post(
    "",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new chatbot user (browser identity)",
)
async def create_user(
    body: CreateUserRequest,
    user_repo=Depends(get_user_repo),
    _=Depends(require_api_key),
) -> UserResponse:
    user = User(display_name=body.display_name.strip())
    saved = await user_repo.create_user(user)
    logger.info("Registered chatbot user id=%s name=%r", saved.id, saved.display_name)
    return _to_response(saved)


@router.get(
    "/{user_id}",
    response_model=UserResponse,
    summary="Fetch a user by id (used to validate localStorage on reload)",
)
async def get_user(
    user_id: Annotated[str, Path(min_length=1, max_length=128)],
    user_repo=Depends(get_user_repo),
    _=Depends(require_api_key),
) -> UserResponse:
    user = await user_repo.get_user(user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="user not found")
    return _to_response(user)


@router.patch(
    "/{user_id}",
    response_model=UserResponse,
    summary="Update a user's display name or metadata",
)
async def update_user(
    user_id: Annotated[str, Path(min_length=1, max_length=128)],
    body: UpdateUserRequest,
    user_repo=Depends(get_user_repo),
    _=Depends(require_api_key),
) -> UserResponse:
    if body.display_name is None and body.metadata is None:
        raise HTTPException(status_code=400, detail="no fields to update")
    name = body.display_name.strip() if body.display_name else None
    user = await user_repo.update_user(
        user_id, display_name=name, metadata=body.metadata,
    )
    if user is None:
        raise HTTPException(status_code=404, detail="user not found")
    return _to_response(user)
