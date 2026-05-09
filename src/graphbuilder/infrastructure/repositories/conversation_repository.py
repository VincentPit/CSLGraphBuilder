"""
Conversation Repository — persistence for chat sessions and turns.

Mirrors the design of ``document_repository.py``:

- ``ConversationRepositoryInterface`` (abstract)
- ``Neo4jConversationRepository``  (production)
- ``InMemoryConversationRepository`` (tests / dev)
- ``create_conversation_repository(config, neo4j_driver)`` factory

The Neo4j model adds:

    (:ConversationSession {id, user_id, title, summary, turn_count, …})
        -[:HAS_TURN {idx}]->
    (:ConversationTurn {id, idx, user_query, llm_answer, query_embedding, …})

A vector index ``turn_query_vector`` is created on
``ConversationTurn.query_embedding`` so episodic memory recall (§5.3 of
docs/RAG_QA_PLAN.md) can find prior turns relevant to the new question.

References:
- §10 / §12 of docs/RAG_QA_PLAN.md describe the data model and surface.
- The reusable serialisation helper ``_to_neo4j_props`` is duplicated from
  ``document_repository.py`` to keep this module self-contained without a
  cross-module helper import that the existing repositories also avoid.
"""

from __future__ import annotations

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from ...domain.models.conversation_models import ConversationSession, ConversationTurn
from ..config.settings import GraphBuilderConfig


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _to_neo4j_props(d: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten a dict for Neo4j ``SET node += $properties``.

    Same rules as the helper in ``document_repository.py``:
    primitives + lists-of-primitives pass through; dicts and lists of
    complex objects are JSON-serialised; ``None`` keys are dropped.
    """
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


def _from_neo4j_session(node_data: Dict[str, Any]) -> ConversationSession:
    data = dict(node_data)
    md = data.get("metadata")
    if isinstance(md, str):
        try:
            data["metadata"] = json.loads(md)
        except json.JSONDecodeError:
            data["metadata"] = {}
    return ConversationSession.from_dict(data)


def _from_neo4j_turn(node_data: Dict[str, Any]) -> ConversationTurn:
    data = dict(node_data)
    md = data.get("metadata")
    if isinstance(md, str):
        try:
            data["metadata"] = json.loads(md)
        except json.JSONDecodeError:
            data["metadata"] = {}
    return ConversationTurn.from_dict(data)


# ----------------------------------------------------------------------
# Interface
# ----------------------------------------------------------------------

class ConversationRepositoryInterface(ABC):
    """Persistence surface for chat sessions + turns."""

    # ---- sessions -----------------------------------------------------

    @abstractmethod
    async def create_session(self, session: ConversationSession) -> ConversationSession:
        """Persist a new session. Caller decides session.id (defaults auto)."""

    @abstractmethod
    async def get_session(self, session_id: str) -> Optional[ConversationSession]:
        ...

    @abstractmethod
    async def list_sessions(
        self,
        user_id: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[ConversationSession]:
        """Most-recently-active first. ``user_id=None`` matches anonymous sessions."""

    @abstractmethod
    async def delete_session(self, session_id: str) -> bool:
        """Detach-delete the session and all its turns."""

    @abstractmethod
    async def update_session_summary(self, session_id: str, summary: str) -> None:
        ...

    @abstractmethod
    async def update_session_title(self, session_id: str, title: str) -> None:
        ...

    # ---- turns --------------------------------------------------------

    @abstractmethod
    async def append_turn(self, turn: ConversationTurn) -> ConversationTurn:
        """Persist a turn and bump session.turn_count + last_active_at."""

    @abstractmethod
    async def get_turn(self, turn_id: str) -> Optional[ConversationTurn]:
        ...

    @abstractmethod
    async def get_turns_by_session(
        self, session_id: str, limit: int = 100, offset: int = 0
    ) -> List[ConversationTurn]:
        """Oldest-first."""

    @abstractmethod
    async def vector_search_turns(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        min_score: float = 0.6,
        session_id: Optional[str] = None,
    ) -> List[Tuple[ConversationTurn, float]]:
        """Episodic recall — find prior turns whose query is most similar.

        ``session_id`` filters to within-session recall (§5.3); ``None``
        searches across all sessions (used for cross-session semantic recall
        if/when we wire that up in P14).
        """

    @abstractmethod
    async def record_feedback(
        self, turn_id: str, rating: int, comment: Optional[str] = None
    ) -> bool:
        ...


# ----------------------------------------------------------------------
# Neo4j implementation
# ----------------------------------------------------------------------

class Neo4jConversationRepository(ConversationRepositoryInterface):
    """Neo4j-backed conversation store. Lazy schema init."""

    def __init__(self, config: GraphBuilderConfig, neo4j_driver, embedding_dim: int = 768):
        self.config = config
        self.driver = neo4j_driver
        self.logger = logging.getLogger(self.__class__.__name__)
        self._embedding_dim = int(embedding_dim or 768)
        self._schema_task = asyncio.create_task(self._initialize_schema())

    # ---- schema -------------------------------------------------------

    async def _initialize_schema(self) -> None:
        async with self.driver.session() as session:
            constraints_and_indexes = [
                "CREATE CONSTRAINT conversation_session_id_unique IF NOT EXISTS "
                "FOR (s:ConversationSession) REQUIRE s.id IS UNIQUE",
                "CREATE CONSTRAINT conversation_turn_id_unique IF NOT EXISTS "
                "FOR (t:ConversationTurn) REQUIRE t.id IS UNIQUE",
                "CREATE INDEX conversation_session_user_idx IF NOT EXISTS "
                "FOR (s:ConversationSession) ON (s.user_id)",
                "CREATE INDEX conversation_session_active_idx IF NOT EXISTS "
                "FOR (s:ConversationSession) ON (s.last_active_at)",
                "CREATE INDEX conversation_turn_session_idx IF NOT EXISTS "
                "FOR (t:ConversationTurn) ON (t.session_id)",
                "CREATE INDEX conversation_turn_idx_idx IF NOT EXISTS "
                "FOR (t:ConversationTurn) ON (t.idx)",
            ]
            for q in constraints_and_indexes:
                try:
                    await session.run(q)
                except Exception as exc:
                    self.logger.debug("Schema step skipped: %s", exc)

            try:
                await session.run(
                    "CREATE VECTOR INDEX `turn_query_vector` IF NOT EXISTS "
                    "FOR (n:ConversationTurn) ON (n.query_embedding) "
                    "OPTIONS {indexConfig: {"
                    "  `vector.dimensions`: $dim,"
                    "  `vector.similarity_function`: 'cosine'"
                    "}}",
                    {"dim": self._embedding_dim},
                )
            except Exception as exc:
                self.logger.debug("turn_query_vector index step skipped: %s", exc)

    # ---- sessions -----------------------------------------------------

    async def create_session(self, session: ConversationSession) -> ConversationSession:
        props = session.to_dict()
        props.pop("id", None)
        async with self.driver.session() as s:
            await s.run(
                "MERGE (n:ConversationSession {id: $id}) "
                "SET n += $props",
                {"id": session.id, "props": _to_neo4j_props(props)},
            )
        return session

    async def get_session(self, session_id: str) -> Optional[ConversationSession]:
        async with self.driver.session() as s:
            result = await s.run(
                "MATCH (n:ConversationSession {id: $id}) RETURN n",
                {"id": session_id},
            )
            record = await result.single()
            if not record:
                return None
            return _from_neo4j_session(dict(record["n"]))

    async def list_sessions(
        self,
        user_id: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[ConversationSession]:
        async with self.driver.session() as s:
            if user_id is None:
                query = (
                    "MATCH (n:ConversationSession) "
                    "WHERE n.user_id IS NULL "
                    "RETURN n ORDER BY n.last_active_at DESC SKIP $offset LIMIT $limit"
                )
                params: Dict[str, Any] = {"offset": offset, "limit": limit}
            else:
                query = (
                    "MATCH (n:ConversationSession {user_id: $user_id}) "
                    "RETURN n ORDER BY n.last_active_at DESC SKIP $offset LIMIT $limit"
                )
                params = {"user_id": user_id, "offset": offset, "limit": limit}
            result = await s.run(query, params)
            sessions: List[ConversationSession] = []
            async for record in result:
                sessions.append(_from_neo4j_session(dict(record["n"])))
            return sessions

    async def delete_session(self, session_id: str) -> bool:
        async with self.driver.session() as s:
            result = await s.run(
                "MATCH (n:ConversationSession {id: $id}) "
                "OPTIONAL MATCH (n)-[:HAS_TURN]->(t:ConversationTurn) "
                "WITH n, count(t) AS turn_count "
                "DETACH DELETE n "
                "RETURN turn_count",
                {"id": session_id},
            )
            record = await result.single()
            return record is not None

    async def update_session_summary(self, session_id: str, summary: str) -> None:
        async with self.driver.session() as s:
            await s.run(
                "MATCH (n:ConversationSession {id: $id}) "
                "SET n.summary = $summary, n.last_active_at = $ts",
                {"id": session_id, "summary": summary, "ts": datetime.now(timezone.utc).isoformat()},
            )

    async def update_session_title(self, session_id: str, title: str) -> None:
        async with self.driver.session() as s:
            await s.run(
                "MATCH (n:ConversationSession {id: $id}) "
                "SET n.title = $title",
                {"id": session_id, "title": title},
            )

    # ---- turns --------------------------------------------------------

    async def append_turn(self, turn: ConversationTurn) -> ConversationTurn:
        props = turn.to_dict()
        props.pop("id", None)
        idx = int(props.pop("idx", 0))
        async with self.driver.session() as s:
            await s.run(
                "MATCH (sess:ConversationSession {id: $session_id}) "
                "MERGE (t:ConversationTurn {id: $turn_id}) "
                "SET t += $props, t.session_id = $session_id, t.idx = $idx "
                "MERGE (sess)-[r:HAS_TURN]->(t) "
                "SET r.idx = $idx, "
                "    sess.turn_count = coalesce(sess.turn_count, 0) + 1, "
                "    sess.last_active_at = $ts",
                {
                    "session_id": turn.session_id,
                    "turn_id": turn.id,
                    "idx": idx,
                    "props": _to_neo4j_props(props),
                    "ts": datetime.now(timezone.utc).isoformat(),
                },
            )
        return turn

    async def get_turn(self, turn_id: str) -> Optional[ConversationTurn]:
        async with self.driver.session() as s:
            result = await s.run(
                "MATCH (t:ConversationTurn {id: $id}) RETURN t",
                {"id": turn_id},
            )
            record = await result.single()
            if not record:
                return None
            return _from_neo4j_turn(dict(record["t"]))

    async def get_turns_by_session(
        self, session_id: str, limit: int = 100, offset: int = 0
    ) -> List[ConversationTurn]:
        async with self.driver.session() as s:
            result = await s.run(
                "MATCH (:ConversationSession {id: $sid})-[:HAS_TURN]->(t:ConversationTurn) "
                "RETURN t ORDER BY t.idx ASC SKIP $offset LIMIT $limit",
                {"sid": session_id, "offset": offset, "limit": limit},
            )
            turns: List[ConversationTurn] = []
            async for record in result:
                turns.append(_from_neo4j_turn(dict(record["t"])))
            return turns

    async def vector_search_turns(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        min_score: float = 0.6,
        session_id: Optional[str] = None,
    ) -> List[Tuple[ConversationTurn, float]]:
        if not query_embedding:
            return []
        async with self.driver.session() as s:
            # Pull a bit more than top_k so the optional session filter
            # has something to filter from before we trim.
            fetch_k = max(top_k * 4, top_k)
            base_query = (
                "CALL db.index.vector.queryNodes($index, $k, $vec) "
                "YIELD node, score "
                "WHERE score >= $min_score"
            )
            params: Dict[str, Any] = {
                "index": "turn_query_vector",
                "k": fetch_k,
                "vec": query_embedding,
                "min_score": min_score,
            }
            if session_id is not None:
                base_query += " AND node.session_id = $session_id"
                params["session_id"] = session_id
            base_query += " RETURN node, score ORDER BY score DESC LIMIT $top_k"
            params["top_k"] = top_k

            try:
                result = await s.run(base_query, params)
            except Exception as exc:
                self.logger.warning("turn_query_vector search failed: %s", exc)
                return []

            hits: List[Tuple[ConversationTurn, float]] = []
            async for record in result:
                hits.append(
                    (_from_neo4j_turn(dict(record["node"])), float(record["score"]))
                )
            return hits

    async def record_feedback(
        self, turn_id: str, rating: int, comment: Optional[str] = None
    ) -> bool:
        async with self.driver.session() as s:
            result = await s.run(
                "MATCH (t:ConversationTurn {id: $id}) "
                "SET t.feedback_rating = $rating, "
                "    t.feedback_comment = $comment "
                "RETURN t.id AS id",
                {"id": turn_id, "rating": rating, "comment": comment},
            )
            return (await result.single()) is not None


# ----------------------------------------------------------------------
# In-memory implementation
# ----------------------------------------------------------------------

class InMemoryConversationRepository(ConversationRepositoryInterface):
    """Dict-backed store for tests + dev. No real vector index — we compute
    cosine similarity on the fly for ``vector_search_turns``."""

    def __init__(self, config: GraphBuilderConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self._sessions: Dict[str, ConversationSession] = {}
        self._turns: Dict[str, ConversationTurn] = {}

    async def create_session(self, session: ConversationSession) -> ConversationSession:
        self._sessions[session.id] = session
        return session

    async def get_session(self, session_id: str) -> Optional[ConversationSession]:
        return self._sessions.get(session_id)

    async def list_sessions(
        self,
        user_id: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[ConversationSession]:
        items = [
            s for s in self._sessions.values()
            if (user_id is None and s.user_id is None) or s.user_id == user_id
        ]
        items.sort(key=lambda s: s.last_active_at, reverse=True)
        return items[offset : offset + limit]

    async def delete_session(self, session_id: str) -> bool:
        if session_id not in self._sessions:
            return False
        del self._sessions[session_id]
        for tid in [t.id for t in self._turns.values() if t.session_id == session_id]:
            del self._turns[tid]
        return True

    async def update_session_summary(self, session_id: str, summary: str) -> None:
        s = self._sessions.get(session_id)
        if s is not None:
            s.summary = summary
            s.touch()

    async def update_session_title(self, session_id: str, title: str) -> None:
        s = self._sessions.get(session_id)
        if s is not None:
            s.title = title

    async def append_turn(self, turn: ConversationTurn) -> ConversationTurn:
        self._turns[turn.id] = turn
        sess = self._sessions.get(turn.session_id)
        if sess is not None:
            sess.turn_count += 1
            sess.touch()
        return turn

    async def get_turn(self, turn_id: str) -> Optional[ConversationTurn]:
        return self._turns.get(turn_id)

    async def get_turns_by_session(
        self, session_id: str, limit: int = 100, offset: int = 0
    ) -> List[ConversationTurn]:
        items = [t for t in self._turns.values() if t.session_id == session_id]
        items.sort(key=lambda t: t.idx)
        return items[offset : offset + limit]

    async def vector_search_turns(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        min_score: float = 0.6,
        session_id: Optional[str] = None,
    ) -> List[Tuple[ConversationTurn, float]]:
        if not query_embedding:
            return []
        candidates = [
            t for t in self._turns.values()
            if t.query_embedding
            and (session_id is None or t.session_id == session_id)
        ]
        scored: List[Tuple[ConversationTurn, float]] = []
        for t in candidates:
            score = _cosine(query_embedding, t.query_embedding or [])
            if score >= min_score:
                scored.append((t, score))
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:top_k]

    async def record_feedback(
        self, turn_id: str, rating: int, comment: Optional[str] = None
    ) -> bool:
        t = self._turns.get(turn_id)
        if t is None:
            return False
        t.feedback_rating = rating
        t.feedback_comment = comment
        return True


def _cosine(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / ((na ** 0.5) * (nb ** 0.5))


# ----------------------------------------------------------------------
# Factory
# ----------------------------------------------------------------------

def create_conversation_repository(
    config: GraphBuilderConfig,
    neo4j_driver=None,
    embedding_dim: Optional[int] = None,
) -> ConversationRepositoryInterface:
    """Return a Neo4j-backed repo when a driver is available, else in-memory."""
    if config.database.provider == "neo4j" and neo4j_driver is not None:
        dim = int(embedding_dim or 768)
        return Neo4jConversationRepository(config, neo4j_driver, embedding_dim=dim)
    return InMemoryConversationRepository(config)


__all__ = [
    "ConversationRepositoryInterface",
    "Neo4jConversationRepository",
    "InMemoryConversationRepository",
    "create_conversation_repository",
]
