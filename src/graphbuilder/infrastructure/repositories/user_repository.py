"""User repository — persistence for the lightweight browser-identity
flow (§14.1 of docs/RAG_QA_PLAN.md, revised 2026-05-09).

The repo mirrors :mod:`conversation_repository` exactly:

- ``UserRepositoryInterface`` (abstract)
- ``Neo4jUserRepository`` (production, lazy schema init)
- ``InMemoryUserRepository`` (tests / dev)
- ``create_user_repository(config, neo4j_driver)`` factory

Schema:

    (:User {id, display_name, metadata, created_at, last_seen_at})

Sessions reference ``User.id`` via the existing
``ConversationSession.user_id`` property — no foreign-key constraint
yet because ``user_id`` is still allowed to be NULL for anonymous
traffic.
"""

from __future__ import annotations

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from ...domain.models.user_models import User
from ..config.settings import GraphBuilderConfig


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _to_neo4j_props(d: Dict[str, Any]) -> Dict[str, Any]:
    """Same flattening rules as the conversation repo — Neo4j only
    accepts primitives + lists-of-primitives as node properties."""
    out: Dict[str, Any] = {}
    for k, v in d.items():
        if v is None:
            continue
        if isinstance(v, Enum):
            out[k] = v.value
        elif isinstance(v, datetime):
            out[k] = v.isoformat()
        elif isinstance(v, dict):
            if v:
                out[k] = json.dumps(v, default=str)
        elif isinstance(v, list):
            if not v:
                out[k] = []
            elif all(isinstance(item, (str, int, float, bool)) for item in v):
                out[k] = v
            else:
                out[k] = json.dumps(v, default=str)
        elif isinstance(v, (str, int, float, bool)):
            out[k] = v
        else:
            out[k] = str(v)
    return out


def _from_neo4j_user(node_data: Dict[str, Any]) -> User:
    data = dict(node_data)
    md = data.get("metadata")
    if isinstance(md, str):
        try:
            data["metadata"] = json.loads(md)
        except json.JSONDecodeError:
            data["metadata"] = {}
    return User.from_dict(data)


# ----------------------------------------------------------------------
# Interface
# ----------------------------------------------------------------------

class UserRepositoryInterface(ABC):
    """Persistence surface for chatbot users."""

    @abstractmethod
    async def create_user(self, user: User) -> User:
        """Persist a new user. Caller decides ``user.id`` (defaults to
        an auto uuid) so the API can return the canonical id back to
        the browser before localStorage is updated."""

    @abstractmethod
    async def get_user(self, user_id: str) -> Optional[User]:
        ...

    @abstractmethod
    async def update_user(
        self,
        user_id: str,
        *,
        display_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[User]:
        """Patch fields on an existing user; returns the updated record
        or ``None`` if the id is unknown. Always bumps ``last_seen_at``."""

    @abstractmethod
    async def touch_user(self, user_id: str) -> Optional[User]:
        """Bump ``last_seen_at`` only — used on every authenticated
        request so we have a freshness signal without a full update."""

    @abstractmethod
    async def list_users(
        self, limit: int = 50, offset: int = 0
    ) -> List[User]:
        """Most-recently-active first. Used for admin / debug only."""


# ----------------------------------------------------------------------
# Neo4j implementation
# ----------------------------------------------------------------------

class Neo4jUserRepository(UserRepositoryInterface):
    """Neo4j-backed user store. Lazy schema init."""

    def __init__(self, config: GraphBuilderConfig, neo4j_driver):
        self.config = config
        self.driver = neo4j_driver
        self.logger = logging.getLogger(self.__class__.__name__)
        self._schema_task = asyncio.create_task(self._initialize_schema())

    async def _initialize_schema(self) -> None:
        async with self.driver.session() as session:
            for q in (
                "CREATE CONSTRAINT user_id_unique IF NOT EXISTS "
                "FOR (u:User) REQUIRE u.id IS UNIQUE",
                "CREATE INDEX user_last_seen_idx IF NOT EXISTS "
                "FOR (u:User) ON (u.last_seen_at)",
            ):
                try:
                    await session.run(q)
                except Exception as exc:
                    self.logger.debug("User schema step skipped: %s", exc)

    async def create_user(self, user: User) -> User:
        props = user.to_dict()
        props.pop("id", None)
        async with self.driver.session() as s:
            await s.run(
                "MERGE (u:User {id: $id}) SET u += $props",
                {"id": user.id, "props": _to_neo4j_props(props)},
            )
        return user

    async def get_user(self, user_id: str) -> Optional[User]:
        if not user_id:
            return None
        async with self.driver.session() as s:
            result = await s.run(
                "MATCH (u:User {id: $id}) RETURN u",
                {"id": user_id},
            )
            record = await result.single()
            if not record:
                return None
            return _from_neo4j_user(dict(record["u"]))

    async def update_user(
        self,
        user_id: str,
        *,
        display_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[User]:
        existing = await self.get_user(user_id)
        if existing is None:
            return None
        if display_name is not None:
            existing.display_name = display_name
        if metadata is not None:
            existing.metadata.update(metadata)
        existing.touch()
        props = existing.to_dict()
        props.pop("id", None)
        async with self.driver.session() as s:
            await s.run(
                "MATCH (u:User {id: $id}) SET u += $props",
                {"id": user_id, "props": _to_neo4j_props(props)},
            )
        return existing

    async def touch_user(self, user_id: str) -> Optional[User]:
        if not user_id:
            return None
        async with self.driver.session() as s:
            result = await s.run(
                "MATCH (u:User {id: $id}) "
                "SET u.last_seen_at = $ts "
                "RETURN u",
                {"id": user_id, "ts": datetime.now(timezone.utc).isoformat()},
            )
            record = await result.single()
            if not record:
                return None
            return _from_neo4j_user(dict(record["u"]))

    async def list_users(self, limit: int = 50, offset: int = 0) -> List[User]:
        async with self.driver.session() as s:
            result = await s.run(
                "MATCH (u:User) "
                "RETURN u ORDER BY u.last_seen_at DESC SKIP $offset LIMIT $limit",
                {"offset": offset, "limit": limit},
            )
            users: List[User] = []
            async for record in result:
                users.append(_from_neo4j_user(dict(record["u"])))
            return users


# ----------------------------------------------------------------------
# In-memory implementation
# ----------------------------------------------------------------------

class InMemoryUserRepository(UserRepositoryInterface):
    """Dict-backed user store for tests + dev."""

    def __init__(self, config: GraphBuilderConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self._users: Dict[str, User] = {}

    async def create_user(self, user: User) -> User:
        self._users[user.id] = user
        return user

    async def get_user(self, user_id: str) -> Optional[User]:
        return self._users.get(user_id)

    async def update_user(
        self,
        user_id: str,
        *,
        display_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[User]:
        user = self._users.get(user_id)
        if user is None:
            return None
        if display_name is not None:
            user.display_name = display_name
        if metadata is not None:
            user.metadata.update(metadata)
        user.touch()
        return user

    async def touch_user(self, user_id: str) -> Optional[User]:
        user = self._users.get(user_id)
        if user is None:
            return None
        user.touch()
        return user

    async def list_users(self, limit: int = 50, offset: int = 0) -> List[User]:
        items = sorted(
            self._users.values(),
            key=lambda u: u.last_seen_at,
            reverse=True,
        )
        return items[offset : offset + limit]


# ----------------------------------------------------------------------
# Factory
# ----------------------------------------------------------------------

def create_user_repository(
    config: GraphBuilderConfig,
    neo4j_driver=None,
) -> UserRepositoryInterface:
    if config.database.provider == "neo4j" and neo4j_driver is not None:
        return Neo4jUserRepository(config, neo4j_driver)
    return InMemoryUserRepository(config)


__all__ = [
    "UserRepositoryInterface",
    "Neo4jUserRepository",
    "InMemoryUserRepository",
    "create_user_repository",
]
