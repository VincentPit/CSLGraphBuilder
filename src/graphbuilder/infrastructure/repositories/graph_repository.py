"""
Graph Repository - Sophisticated data access layer for knowledge graph operations.

This module provides enterprise-grade repository pattern implementation
for graph entities and relationships with advanced querying capabilities.
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any, Set, Tuple, Sequence
from datetime import datetime, timezone
from abc import ABC, abstractmethod

from ...domain.models.graph_models import (
    GraphEntity, GraphRelationship, KnowledgeGraph,
    EntityType, RelationshipType
)
from ..config.settings import GraphBuilderConfig
from .document_repository import _to_neo4j_props  # shared property flattener


def _decode_props(value: Any) -> Dict[str, Any]:
    """Inverse of ``_to_neo4j_props`` for nested-dict fields.

    ``_to_neo4j_props`` JSON-stringifies non-empty maps so Neo4j (which only
    accepts primitives + arrays of primitives) can store them. On read, the
    value comes back as a string — parse it back to a dict. Returns an empty
    dict for None / missing / invalid values so callers can dereference
    safely.
    """
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except (json.JSONDecodeError, ValueError):
            return {}
    return {}


_UNSET: Any = object()  # sentinel: distinguishes "use precomputed" from "compute fresh"


class GraphRepositoryInterface(ABC):
    """Abstract interface for graph repository operations."""

    @abstractmethod
    async def save_entity(self, entity: GraphEntity) -> GraphEntity:
        """Save an entity to the graph."""
        pass

    async def save_entities_batch(
        self, entities: List[GraphEntity]
    ) -> List[GraphEntity]:
        """Save a list of entities, batching expensive work where possible.

        Default implementation falls back to per-entity ``save_entity`` calls;
        the Neo4j implementation overrides this to encode all embeddings in
        a single ``model.encode`` call before persisting.
        """
        out: List[GraphEntity] = []
        for e in entities:
            out.append(await self.save_entity(e))
        return out

    async def save_relationships_batch(
        self,
        relationships: List[GraphRelationship],
        entity_names: Optional[Dict[str, str]] = None,
    ) -> List[GraphRelationship]:
        """Save a list of relationships, batching embedding work where possible.

        ``entity_names`` is an optional ``{entity_id: name}`` map the caller
        can supply to avoid a Neo4j round-trip per relationship — the
        Neo4j implementation uses it to skip the source/target name lookup.
        """
        del entity_names  # unused in fallback
        out: List[GraphRelationship] = []
        for r in relationships:
            out.append(await self.save_relationship(r))
        return out
    
    @abstractmethod
    async def get_entity_by_id(self, entity_id: str) -> Optional[GraphEntity]:
        """Get entity by ID."""
        pass
    
    @abstractmethod
    async def save_relationship(self, relationship: GraphRelationship) -> GraphRelationship:
        """Save a relationship to the graph."""
        pass
    
    @abstractmethod
    async def get_relationship_by_id(self, relationship_id: str) -> Optional[GraphRelationship]:
        """Get relationship by ID."""
        pass
    
    @abstractmethod
    async def find_entities_by_type(self, entity_type: EntityType) -> List[GraphEntity]:
        """Find entities by type."""
        pass
    
    @abstractmethod
    async def find_similar_entities(self, entity: GraphEntity, threshold: float = 0.8) -> List[GraphEntity]:
        """Find similar entities for deduplication."""
        pass
    
    @abstractmethod
    async def get_entity_relationships(self, entity_id: str) -> List[GraphRelationship]:
        """Get all relationships for an entity."""
        pass
    
    @abstractmethod
    async def execute_cypher_query(self, query: str, parameters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Execute custom Cypher query."""
        pass

    async def get_all_entities(self) -> Dict[str, 'GraphEntity']:
        """Return all entities as {id: entity} dict."""
        raise NotImplementedError

    async def get_all_relationships(self) -> Dict[str, 'GraphRelationship']:
        """Return all relationships as {id: relationship} dict."""
        raise NotImplementedError

    async def search_entities_by_text(self, terms: List[str], limit: int = 50) -> Dict[str, 'GraphEntity']:
        """Search entities whose name or description contains any of the given terms."""
        raise NotImplementedError

    async def get_subgraph_slice(
        self,
        seed_types: Sequence[str],
        per_type_limit: int,
        exclude_types: Sequence[str],
        max_neighbors: int,
    ) -> Tuple[Dict[str, 'GraphEntity'], Dict[str, 'GraphRelationship'], Set[str], Dict[str, int], int, int]:
        """Fetch a coherent subgraph slice for the given seed types.

        Returns ``(entities, relationships, seed_ids, seed_per_type,
        total_entities, total_relationships)``:

        * ``entities`` includes every seed plus the other-end entities
          of any returned edge.
        * ``relationships`` are edges where at least one endpoint is a
          seed and neither endpoint's type is in ``exclude_types``.
        * ``seed_ids`` is the subset of ``entities.keys()`` that were
          picked as seeds — the rest are pulled-in neighbours. The
          router needs this to apply per-seed fan-out caps without
          re-deriving seed membership.
        * ``seed_per_type`` is the count of seeds actually picked from
          each requested type (typically equal to ``per_type_limit``,
          less if the type has fewer entities in the graph).
        * ``total_entities`` / ``total_relationships`` are global counts,
          used by the visualisation to show "showing N of M".

        Implementations should use indexed lookups so this stays cheap
        even on large graphs — the goal is to avoid the full-graph
        scan that ``get_all_entities`` / ``get_all_relationships``
        require.
        """
        raise NotImplementedError

    async def vector_search_entities(
        self, query_embedding: List[float], top_k: int = 10, min_score: float = 0.5
    ) -> List[Tuple['GraphEntity', float]]:
        """Find entities by vector similarity. Returns (entity, score) pairs."""
        raise NotImplementedError

    async def vector_search_relationships(
        self, query_embedding: List[float], top_k: int = 10, min_score: float = 0.5
    ) -> List[Tuple['GraphRelationship', float]]:
        """Find relationships by vector similarity on their description embedding. Returns (rel, score) pairs."""
        raise NotImplementedError


class Neo4jGraphRepository(GraphRepositoryInterface):
    """
    Neo4j implementation of graph repository with sophisticated graph operations.
    
    Provides enterprise-grade graph persistence using Neo4j with advanced
    graph algorithms, similarity matching, and complex query capabilities.
    """
    
    def __init__(self, config: GraphBuilderConfig, neo4j_driver):
        self.config = config
        self.driver = neo4j_driver
        self.logger = logging.getLogger(self.__class__.__name__)
        self._embedding_model = None
        self._embedding_dim: int = 384  # default for all-MiniLM-L6-v2
        
        # Initialize graph schema
        asyncio.create_task(self._initialize_schema())
    
    async def _initialize_schema(self) -> None:
        """Initialize graph schema and constraints."""
        
        async with self.driver.session() as session:
            # Create constraints and indexes
            schema_queries = [
                # Entity constraints
                "CREATE CONSTRAINT entity_id_unique IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE",
                "CREATE CONSTRAINT relationship_id_unique IF NOT EXISTS FOR (r:Relationship) REQUIRE r.id IS UNIQUE",
                
                # Entity indexes
                "CREATE INDEX entity_name_idx IF NOT EXISTS FOR (e:Entity) ON (e.name)",
                "CREATE INDEX entity_type_idx IF NOT EXISTS FOR (e:Entity) ON (e.entity_type)",
                "CREATE INDEX entity_hash_idx IF NOT EXISTS FOR (e:Entity) ON (e.content_hash)",
                
                # Relationship indexes
                "CREATE INDEX relationship_type_idx IF NOT EXISTS FOR (r:Relationship) ON (r.relationship_type)",
                "CREATE INDEX relationship_source_idx IF NOT EXISTS FOR (r:Relationship) ON (r.source_entity_id)",
                "CREATE INDEX relationship_target_idx IF NOT EXISTS FOR (r:Relationship) ON (r.target_entity_id)",
                
                # Full-text search indexes
                "CREATE FULLTEXT INDEX entity_search IF NOT EXISTS FOR (e:Entity) ON EACH [e.name, e.description]",
            ]
            
            for query in schema_queries:
                try:
                    await session.run(query)
                except Exception as e:
                    self.logger.debug(f"Schema creation result: {str(e)}")

            # Create vector indexes (separate try since they need dimension param)
            for label, prop, idx_name in [
                ("Entity", "name_embedding", "entity_name_vector"),
                ("Relationship", "desc_embedding", "rel_desc_vector"),
            ]:
                try:
                    await session.run(
                        f"CREATE VECTOR INDEX `{idx_name}` IF NOT EXISTS "
                        f"FOR (n:{label}) ON (n.{prop}) "
                        "OPTIONS {indexConfig: {"
                        "  `vector.dimensions`: $dim,"
                        "  `vector.similarity_function`: 'cosine'"
                        "}}",
                        {"dim": self._embedding_dim},
                    )
                except Exception as e:
                    self.logger.debug(f"Vector index creation ({idx_name}): {e}")

    # ------------------------------------------------------------------
    # Embedding helpers
    # ------------------------------------------------------------------

    def _get_embedding_model(self):
        """Return the shared sentence-transformers model from the factory.

        We delegate to ``embedding_factory`` so this repo, the pipeline,
        the verifier, and the chunker all use the same model instance —
        and so swapping the model (e.g. SapBERT vs MiniLM) only needs
        the ``EMBEDDING_MODEL`` env var, not a code change.
        """
        if self._embedding_model is None:
            from ..services.embedding_factory import get_model, get_embedding_dim
            self._embedding_model = get_model()
            if self._embedding_model is not None:
                self._embedding_dim = get_embedding_dim() or self._embedding_dim
        return self._embedding_model

    async def _embed_text_async(self, text: str) -> Optional[List[float]]:
        """Produce an embedding vector for *text*, or None if unavailable.

        Async wrapper around the model's ``encode`` — runs in the default
        thread executor under a module-level lock so callers don't freeze
        the event loop and concurrent calls don't step on the model.
        """
        model = self._get_embedding_model()
        if model is None or not text:
            return None
        from ..services.embedding_factory import embed_async
        return await embed_async(text)

    def _entity_embedding_text(self, entity: GraphEntity) -> str:
        """Build the string to embed for an entity (name + description)."""
        parts = [entity.name]
        if entity.description:
            parts.append(entity.description)
        return " — ".join(parts)

    def _relationship_embedding_text(self, rel: GraphRelationship, source_name: str = "", target_name: str = "") -> str:
        """Build the string to embed for a relationship."""
        parts = []
        if source_name:
            parts.append(source_name)
        parts.append(rel.relationship_type.value.replace("_", " "))
        if target_name:
            parts.append(target_name)
        if rel.description:
            parts.append(rel.description)
        return " ".join(parts)
    
    async def save_entity(
        self,
        entity: GraphEntity,
        *,
        precomputed_embedding: Any = _UNSET,
    ) -> GraphEntity:
        """Save entity to Neo4j graph database with provenance tracking.

        ``precomputed_embedding`` lets ``save_entities_batch`` skip the
        per-entity ``model.encode`` call by passing in a vector that was
        already computed as part of a batch. Pass ``None`` to explicitly
        skip embedding without recomputation, or omit to compute fresh.
        """

        async with self.driver.session() as session:
            # Check for existing entity with same name and type
            existing_query = """
            MATCH (e:Entity)
            WHERE e.name = $name AND e.entity_type = $entity_type
            RETURN e.id as existing_id, e.source_chunk_ids as existing_chunks, e.source_document_ids as existing_docs
            """
            
            result = await session.run(existing_query, {
                'name': entity.name,
                'entity_type': entity.entity_type.value
            })
            
            existing_record = await result.single()
            
            if existing_record:
                # Update existing entity — merge provenance
                existing_id = existing_record['existing_id']
                existing_chunks = existing_record.get('existing_chunks') or []
                existing_docs = existing_record.get('existing_docs') or []

                # Merge chunk/doc IDs (deduplicated)
                merged_chunks = list(dict.fromkeys(existing_chunks + entity.source_chunk_ids))
                merged_docs = list(dict.fromkeys(existing_docs + entity.source_document_ids))

                update_query = """
                MATCH (e:Entity {id: $existing_id})
                SET e += $properties,
                    e.source_chunk_ids = $source_chunk_ids,
                    e.source_document_ids = $source_document_ids,
                    e.updated_at = datetime(),
                    e.version = e.version + 1
                RETURN e
                """
                
                properties = entity.to_dict()
                properties.pop('id', None)
                properties.pop('source_chunk_ids', None)
                properties.pop('source_document_ids', None)
                properties = _to_neo4j_props(properties)

                # Compute embedding for vector index (added after flattening
                # because the embedding is already a list of primitives).
                if precomputed_embedding is _UNSET:
                    emb = await self._embed_text_async(self._entity_embedding_text(entity))
                else:
                    emb = precomputed_embedding
                if emb is not None:
                    properties['name_embedding'] = emb

                await session.run(update_query, {
                    'existing_id': existing_id,
                    'properties': properties,
                    'source_chunk_ids': merged_chunks,
                    'source_document_ids': merged_docs,
                })

                # Create EXTRACTED_FROM edges for new chunks
                for chunk_id in entity.source_chunk_ids:
                    if chunk_id not in existing_chunks:
                        await session.run(
                            "MATCH (e:Entity {id: $eid}), (c:DocumentChunk {id: $cid}) "
                            "MERGE (e)-[:EXTRACTED_FROM]->(c)",
                            {"eid": existing_id, "cid": chunk_id},
                        )
                
                entity.id = existing_id
                entity.source_chunk_ids = merged_chunks
                entity.source_document_ids = merged_docs
                self.logger.debug(f"Updated existing entity: {entity.id}")
                
            else:
                # Create new entity
                create_query = """
                CREATE (e:Entity {id: $id})
                SET e += $properties,
                    e.source_chunk_ids = $source_chunk_ids,
                    e.source_document_ids = $source_document_ids,
                    e.created_at = datetime(),
                    e.updated_at = datetime(),
                    e.version = 1
                RETURN e
                """
                
                properties = entity.to_dict()
                properties.pop('id', None)
                properties.pop('source_chunk_ids', None)
                properties.pop('source_document_ids', None)
                properties = _to_neo4j_props(properties)
                properties['content_hash'] = entity.get_hash()

                # Compute embedding for vector index
                if precomputed_embedding is _UNSET:
                    emb = await self._embed_text_async(self._entity_embedding_text(entity))
                else:
                    emb = precomputed_embedding
                if emb is not None:
                    properties['name_embedding'] = emb
                
                await session.run(create_query, {
                    'id': entity.id,
                    'properties': properties,
                    'source_chunk_ids': entity.source_chunk_ids,
                    'source_document_ids': entity.source_document_ids,
                })

                # Create EXTRACTED_FROM edges
                for chunk_id in entity.source_chunk_ids:
                    await session.run(
                        "MATCH (e:Entity {id: $eid}), (c:DocumentChunk {id: $cid}) "
                        "MERGE (e)-[:EXTRACTED_FROM]->(c)",
                        {"eid": entity.id, "cid": chunk_id},
                    )
                
                self.logger.debug(f"Created new entity: {entity.id}")
        
        return entity
    
    async def get_entity_by_id(self, entity_id: str) -> Optional[GraphEntity]:
        """Get entity by ID from Neo4j database."""
        
        async with self.driver.session() as session:
            query = """
            MATCH (e:Entity {id: $id})
            RETURN e
            """
            
            result = await session.run(query, {'id': entity_id})
            record = await result.single()
            
            if record:
                entity_data = dict(record['e'])
                return self._create_entity_from_data(entity_data)
            
            return None
    
    async def save_relationship(
        self,
        relationship: GraphRelationship,
        *,
        precomputed_embedding: Any = _UNSET,
        precomputed_endpoint_names: Optional[Tuple[str, str]] = None,
    ) -> GraphRelationship:
        """Save relationship to Neo4j graph database with provenance tracking.

        If a relationship between the same source/target with the same type
        already exists, merges provenance (source chunks) instead of creating
        a duplicate.

        ``precomputed_embedding`` lets ``save_relationships_batch`` skip the
        per-relationship embedding round-trip; ``precomputed_endpoint_names``
        skips the source/target-name MATCH lookup when the caller already
        knows them.
        """

        async with self.driver.session() as session:
            # Resolve source/target names — needed both to validate the
            # endpoints exist and to build the relationship's embedding text.
            # The batch path passes them in to skip this round-trip.
            if precomputed_endpoint_names is not None:
                source_name, target_name = precomputed_endpoint_names
            else:
                entities_query = """
                MATCH (source:Entity {id: $source_id}), (target:Entity {id: $target_id})
                RETURN source, target
                """

                result = await session.run(entities_query, {
                    'source_id': relationship.source_entity_id,
                    'target_id': relationship.target_entity_id
                })

                entities_record = await result.single()
                if not entities_record:
                    raise ValueError(f"Source or target entity not found for relationship {relationship.id}")

                source_name = dict(entities_record['source']).get('name', '')
                target_name = dict(entities_record['target']).get('name', '')

            # Check for existing relationship between same entities with same type
            existing_query = """
            MATCH (source:Entity {id: $source_id})-[r:RELATES]->(target:Entity {id: $target_id})
            WHERE r.relationship_type = $rel_type
            RETURN r.id as existing_id, r.source_chunk_ids as existing_chunks, 
                   r.source_document_ids as existing_docs, r.description as existing_desc
            """
            existing_result = await session.run(existing_query, {
                'source_id': relationship.source_entity_id,
                'target_id': relationship.target_entity_id,
                'rel_type': relationship.relationship_type.value,
            })
            existing_record = await existing_result.single()

            if existing_record:
                # Merge provenance into existing relationship
                existing_id = existing_record['existing_id']
                existing_chunks = existing_record.get('existing_chunks') or []
                existing_docs = existing_record.get('existing_docs') or []
                merged_chunks = list(dict.fromkeys(existing_chunks + relationship.source_chunk_ids))
                merged_docs = list(dict.fromkeys(existing_docs + relationship.source_document_ids))

                # Compute embedding for vector index
                if precomputed_embedding is _UNSET:
                    emb = await self._embed_text_async(self._relationship_embedding_text(
                        relationship, source_name, target_name
                    ))
                else:
                    emb = precomputed_embedding

                # Carry forward any property/metadata updates the caller made
                # (e.g. setting verification_status on an existing rel for the
                # curation queue). ``SET r += $properties`` merges flat keys
                # without disturbing the chunk-id arrays we set explicitly.
                update_props = relationship.to_dict()
                update_props.pop('id', None)
                update_props.pop('source_chunk_ids', None)
                update_props.pop('source_document_ids', None)
                update_props = _to_neo4j_props(update_props)

                update_query = """
                MATCH ()-[r:RELATES {id: $existing_id}]->()
                SET r += $properties,
                    r.source_chunk_ids = $source_chunk_ids,
                    r.source_document_ids = $source_document_ids,
                    r.desc_embedding = $desc_embedding,
                    r.updated_at = datetime(),
                    r.version = r.version + 1
                RETURN r
                """
                await session.run(update_query, {
                    'existing_id': existing_id,
                    'properties': update_props,
                    'source_chunk_ids': merged_chunks,
                    'source_document_ids': merged_docs,
                    'desc_embedding': emb,
                })

                # Create EXTRACTED_FROM edges for new chunks
                for chunk_id in relationship.source_chunk_ids:
                    if chunk_id not in existing_chunks:
                        await session.run(
                            "MATCH ()-[r:RELATES {id: $rid}]->(), (c:DocumentChunk {id: $cid}) "
                            "WITH r, c MATCH (s:Entity {id: r.source_entity_id}) "
                            "MERGE (s)-[:REL_EXTRACTED_FROM {relationship_id: $rid}]->(c)",
                            {"rid": existing_id, "cid": chunk_id},
                        )

                relationship.id = existing_id
                relationship.source_chunk_ids = merged_chunks
                relationship.source_document_ids = merged_docs
                self.logger.debug(f"Merged provenance into existing relationship: {existing_id}")
            else:
                # Create new relationship
                create_query = """
                MATCH (source:Entity {id: $source_id}), (target:Entity {id: $target_id})
                MERGE (source)-[r:RELATES {id: $relationship_id}]->(target)
                SET r += $properties,
                    r.source_chunk_ids = $source_chunk_ids,
                    r.source_document_ids = $source_document_ids,
                    r.created_at = datetime(),
                    r.updated_at = datetime(),
                    r.version = 1
                RETURN r
                """
                
                properties = relationship.to_dict()
                properties.pop('id', None)
                properties.pop('source_chunk_ids', None)
                properties.pop('source_document_ids', None)
                properties = _to_neo4j_props(properties)

                # Compute embedding for vector index
                if precomputed_embedding is _UNSET:
                    emb = await self._embed_text_async(self._relationship_embedding_text(
                        relationship, source_name, target_name
                    ))
                else:
                    emb = precomputed_embedding
                if emb is not None:
                    properties['desc_embedding'] = emb

                await session.run(create_query, {
                    'source_id': relationship.source_entity_id,
                    'target_id': relationship.target_entity_id,
                    'relationship_id': relationship.id,
                    'properties': properties,
                    'source_chunk_ids': relationship.source_chunk_ids,
                    'source_document_ids': relationship.source_document_ids,
                })

                # Create EXTRACTED_FROM edges for chunks
                for chunk_id in relationship.source_chunk_ids:
                    await session.run(
                        "MATCH (e:Entity {id: $eid}), (c:DocumentChunk {id: $cid}) "
                        "MERGE (e)-[:REL_EXTRACTED_FROM {relationship_id: $rid}]->(c)",
                        {"eid": relationship.source_entity_id, "rid": relationship.id, "cid": chunk_id},
                    )
                
                self.logger.debug(f"Created relationship: {relationship.id}")
            return relationship

    # ------------------------------------------------------------------
    # Batched persistence + cascading dedup
    # ------------------------------------------------------------------

    # Cosine threshold for tier-3 (vector) merges. 0.92 is conservative —
    # enough to catch "TNFα" / "Tumor Necrosis Factor Alpha"-style variants
    # but tight enough to avoid merging genuinely distinct concepts whose
    # SapBERT embeddings happen to be neighbors.
    _DEDUP_VECTOR_THRESHOLD: float = 0.92

    async def save_entities_batch(
        self, entities: List[GraphEntity]
    ) -> List[GraphEntity]:
        """Persist a batch with cascading dedup against the existing graph.

        Cascade (cheapest → most expensive, short-circuit on first hit):
          1. **External-ID match** — single Cypher lookup for all new
             entities' external IDs at once. Strongest signal; most OT
             ingests resolve here.
          2. **Name / alias match (case-insensitive)** — single Cypher
             lookup that scans for existing entities whose name or aliases
             match any new entity's name/aliases (same entity_type only).
          3. **SapBERT vector search** — only for the entities still
             unmatched. Embeddings are computed in one batch ``encode``
             call here (deferred from the start of the method so we don't
             encode entities that tiers 1-2 already merged away).

        Each tier applies an external-ID-contradiction gate before
        accepting a match — two entities that share a name but have
        conflicting Ensembl/ChEMBL IDs are *not* the same record.
        """
        if not entities:
            return []

        from ..services.embedding_factory import embed_batch_async

        # ── Tiers 1 + 2: deterministic dedup, no embeddings needed ─────
        matches: Dict[int, Tuple[str, str]] = {}  # idx → (existing_id, tier)
        await self._dedup_tier_external_id(entities, matches)
        await self._dedup_tier_name_alias(entities, matches)

        # ── Tier 3: encode survivors, then per-survivor vector search ──
        survivor_idxs = [i for i in range(len(entities)) if i not in matches]
        survivor_entities = [entities[i] for i in survivor_idxs]

        embeddings: List[Optional[List[float]]] = []
        if survivor_entities:
            texts = [self._entity_embedding_text(e) for e in survivor_entities]
            if self._get_embedding_model() is not None:
                embeddings = await embed_batch_async(texts)
            else:
                embeddings = [None] * len(survivor_entities)

            for idx, entity, emb in zip(survivor_idxs, survivor_entities, embeddings):
                if emb is None:
                    continue
                existing_id = await self._dedup_tier_vector(entity, emb)
                if existing_id is not None:
                    matches[idx] = (existing_id, "vector")

        # ── Persist ─────────────────────────────────────────────────────
        emb_by_idx = dict(zip(survivor_idxs, embeddings))
        out: List[GraphEntity] = []
        for i, entity in enumerate(entities):
            if i in matches:
                existing_id, tier = matches[i]
                await self._merge_into_existing(entity, existing_id, tier)
                out.append(entity)
            else:
                # Unmatched — has a precomputed embedding (or None if no model)
                await self.save_entity(entity, precomputed_embedding=emb_by_idx.get(i))
                out.append(entity)
        return out

    # ── Dedup: tier 1 — external-ID match ─────────────────────────────

    async def _dedup_tier_external_id(
        self,
        entities: List[GraphEntity],
        matches: Dict[int, Tuple[str, str]],
    ) -> None:
        """Look up every new entity's external IDs in one Cypher query.

        ``external_ids`` is JSON-stringified at write time (see
        ``_to_neo4j_props``), so we search via ``CONTAINS '"sys":"id"'``.
        The entity_type filter narrows the scan considerably.
        """
        # Build needles: one per (system, id, type) tuple. Track which new
        # entity each needle came from so we can record the match.
        needles: List[Dict[str, Any]] = []
        for idx, entity in enumerate(entities):
            if idx in matches:
                continue
            for system, ext_id in (entity.external_ids or {}).items():
                if not system or not ext_id:
                    continue
                # Mirror json.dumps' format exactly: "sys": "id" with a
                # space after the colon (json.dumps default separator).
                fragment = f'"{system}": "{ext_id}"'
                needles.append({
                    "idx": idx,
                    "fragment": fragment,
                    "type": entity.entity_type.value,
                    "system": system,
                    "ext_id": ext_id,
                })

        if not needles:
            return

        query = (
            "UNWIND $needles AS n "
            "MATCH (e:Entity) "
            "WHERE e.entity_type = n.type "
            "  AND e.external_ids CONTAINS n.fragment "
            "RETURN n.idx AS idx, e.id AS entity_id "
            "LIMIT 5000"
        )
        async with self.driver.session() as session:
            result = await session.run(query, {"needles": needles})
            async for record in result:
                idx = record["idx"]
                if idx in matches:
                    continue  # First hit wins
                matches[idx] = (record["entity_id"], "external_id")

    # ── Dedup: tier 2 — name + alias match (case-insensitive) ─────────

    async def _dedup_tier_name_alias(
        self,
        entities: List[GraphEntity],
        matches: Dict[int, Tuple[str, str]],
    ) -> None:
        """Look up every new entity's name + aliases against existing names + aliases."""
        # Collect (idx, type, lowercase name/alias) tuples for unmatched entities
        needles: List[Dict[str, Any]] = []
        for idx, entity in enumerate(entities):
            if idx in matches:
                continue
            tokens: Set[str] = set()
            if entity.name:
                tokens.add(entity.name.strip().lower())
            for alias in entity.aliases or []:
                if alias:
                    tokens.add(alias.strip().lower())
            for tok in tokens:
                needles.append({
                    "idx": idx,
                    "token": tok,
                    "type": entity.entity_type.value,
                })

        if not needles:
            return

        query = (
            "UNWIND $needles AS n "
            "MATCH (e:Entity) "
            "WHERE e.entity_type = n.type "
            "  AND (toLower(e.name) = n.token "
            "       OR ANY(a IN coalesce(e.aliases, []) WHERE toLower(a) = n.token)) "
            "RETURN n.idx AS idx, e.id AS entity_id, e.external_ids AS existing_ext "
            "LIMIT 5000"
        )
        async with self.driver.session() as session:
            result = await session.run(query, {"needles": needles})
            async for record in result:
                idx = record["idx"]
                if idx in matches:
                    continue
                # Apply the external-ID-contradiction gate before accepting.
                if self._external_ids_contradict(
                    entities[idx].external_ids, _decode_props(record["existing_ext"])
                ):
                    continue
                matches[idx] = (record["entity_id"], "name_alias")

    # ── Dedup: tier 3 — vector search ─────────────────────────────────

    async def _dedup_tier_vector(
        self, entity: GraphEntity, embedding: List[float]
    ) -> Optional[str]:
        """Vector-search for a candidate of the same type and accept if
        the cosine score clears ``_DEDUP_VECTOR_THRESHOLD`` and the
        candidate's external IDs don't contradict the new entity's."""
        hits = await self.vector_search_entities(
            embedding,
            top_k=5,
            min_score=self._DEDUP_VECTOR_THRESHOLD,
        )
        for candidate, _score in hits:
            if candidate.entity_type != entity.entity_type:
                continue
            if self._external_ids_contradict(
                entity.external_ids, candidate.external_ids
            ):
                continue
            return candidate.id
        return None

    # ── Merge a new entity into an existing graph record ──────────────

    async def _merge_into_existing(
        self, entity: GraphEntity, existing_id: str, tier: str
    ) -> None:
        """Merge a new entity's provenance + aliases + external IDs into
        an existing record, mutate ``entity.id`` to point at it, and
        record an audit annotation so the merge can be reviewed/reversed.
        """
        async with self.driver.session() as session:
            existing = await session.run(
                "MATCH (e:Entity {id: $id}) "
                "RETURN e.source_chunk_ids AS chunks, e.source_document_ids AS docs, "
                "       e.aliases AS aliases, e.external_ids AS ext, "
                "       e.metadata AS meta",
                {"id": existing_id},
            )
            record = await existing.single()
            if record is None:
                # Existing entity vanished between dedup lookup and merge —
                # fall back to creating new (no embedding precomputed here).
                await self.save_entity(entity)
                return

            existing_chunks = record["chunks"] or []
            existing_docs = record["docs"] or []
            existing_aliases = record["aliases"] or []
            existing_ext = _decode_props(record["ext"])

            # Add the new entity's name + aliases as aliases on the existing
            # record so future case-insensitive lookups find it under any name.
            new_alias_pool = [entity.name] + list(entity.aliases or [])
            merged_aliases = list(
                dict.fromkeys(
                    existing_aliases + [a for a in new_alias_pool if a]
                )
            )

            # Merge external IDs (existing wins on conflict — those were
            # presumably curated/saved earlier from a different source).
            merged_ext = dict(existing_ext)
            for sys_key, ext_id in (entity.external_ids or {}).items():
                merged_ext.setdefault(sys_key, ext_id)

            # Audit annotation: record what got merged in and via which tier.
            existing_meta = _decode_props(record["meta"])
            existing_anns = existing_meta.get("annotations") or {}
            dedup_log = list(existing_anns.get("dedup_merged_from") or [])
            dedup_log.append({
                "merged_id": entity.id,
                "tier": tier,
                "merged_at": datetime.now(timezone.utc).isoformat(),
            })

            merged_chunks = list(dict.fromkeys(existing_chunks + (entity.source_chunk_ids or [])))
            merged_docs = list(dict.fromkeys(existing_docs + (entity.source_document_ids or [])))

            # Round-trip the merged annotations through _to_neo4j_props so
            # the ``annotations`` dict gets re-JSON-stringified consistently.
            existing_meta.setdefault("annotations", {})
            existing_meta["annotations"]["dedup_merged_from"] = dedup_log
            meta_props = _to_neo4j_props({"metadata": existing_meta}).get("metadata")

            await session.run(
                "MATCH (e:Entity {id: $id}) "
                "SET e.source_chunk_ids = $chunks, "
                "    e.source_document_ids = $docs, "
                "    e.aliases = $aliases, "
                "    e.external_ids = $ext, "
                "    e.metadata = $meta, "
                "    e.updated_at = datetime(), "
                "    e.version = coalesce(e.version, 1) + 1",
                {
                    "id": existing_id,
                    "chunks": merged_chunks,
                    "docs": merged_docs,
                    "aliases": merged_aliases,
                    "ext": json.dumps(merged_ext, default=str) if merged_ext else None,
                    "meta": meta_props,
                },
            )

            entity.id = existing_id
            entity.source_chunk_ids = merged_chunks
            entity.source_document_ids = merged_docs
            self.logger.debug(
                "Dedup merge (%s): incoming → %s", tier, existing_id
            )

    # ── Helper: detect contradicting external IDs across two entities ──

    @staticmethod
    def _external_ids_contradict(
        a: Optional[Dict[str, str]],
        b: Optional[Dict[str, str]],
    ) -> bool:
        """True iff ``a`` and ``b`` share an external-ID system but disagree
        on the value. Two entities with the same Ensembl key but different
        ENSG IDs are different genes — never merge them.
        """
        if not a or not b:
            return False
        for key, val in a.items():
            other = b.get(key)
            if other and val and other != val:
                return True
        return False

    async def save_relationships_batch(
        self,
        relationships: List[GraphRelationship],
        entity_names: Optional[Dict[str, str]] = None,
    ) -> List[GraphRelationship]:
        """Encode all relationship-description embeddings in one model call, then persist.

        ``entity_names`` should map ``entity_id → entity.name`` for every
        endpoint referenced by the batch. When provided, each
        ``save_relationship`` call skips its source/target MATCH lookup.
        """
        if not relationships:
            return []

        # Resolve endpoint names — prefer the caller-supplied map; for any
        # missing IDs, fall back to one MATCH query upfront so the inner
        # save calls don't each do a lookup.
        names: Dict[str, str] = dict(entity_names or {})
        missing = {
            eid
            for r in relationships
            for eid in (r.source_entity_id, r.target_entity_id)
            if eid not in names
        }
        if missing:
            async with self.driver.session() as session:
                result = await session.run(
                    "MATCH (e:Entity) WHERE e.id IN $ids RETURN e.id AS id, e.name AS name",
                    {"ids": list(missing)},
                )
                async for record in result:
                    names[record["id"]] = record["name"] or ""

        from ..services.embedding_factory import embed_batch_async

        texts = [
            self._relationship_embedding_text(
                r,
                source_name=names.get(r.source_entity_id, ""),
                target_name=names.get(r.target_entity_id, ""),
            )
            for r in relationships
        ]
        embeddings = await embed_batch_async(texts) if self._get_embedding_model() is not None else [None] * len(relationships)

        out: List[GraphRelationship] = []
        for rel, emb in zip(relationships, embeddings):
            endpoints = (
                names.get(rel.source_entity_id, ""),
                names.get(rel.target_entity_id, ""),
            )
            out.append(
                await self.save_relationship(
                    rel,
                    precomputed_embedding=emb,
                    precomputed_endpoint_names=endpoints,
                )
            )
        return out

    async def get_relationship_by_id(self, relationship_id: str) -> Optional[GraphRelationship]:
        """Get relationship by ID from Neo4j database."""
        
        async with self.driver.session() as session:
            query = """
            MATCH ()-[r:RELATES {id: $id}]->()
            RETURN r, startNode(r).id as source_id, endNode(r).id as target_id
            """
            
            result = await session.run(query, {'id': relationship_id})
            record = await result.single()
            
            if record:
                rel_data = dict(record['r'])
                rel_data['source_entity_id'] = record['source_id']
                rel_data['target_entity_id'] = record['target_id']
                return self._create_relationship_from_data(rel_data)
            
            return None
    
    async def find_entities_by_type(self, entity_type: EntityType) -> List[GraphEntity]:
        """Find entities by type."""
        
        async with self.driver.session() as session:
            query = """
            MATCH (e:Entity)
            WHERE e.entity_type = $entity_type
            RETURN e
            ORDER BY e.name
            """
            
            result = await session.run(query, {'entity_type': entity_type.value})
            entities = []
            
            async for record in result:
                entity_data = dict(record['e'])
                entity = self._create_entity_from_data(entity_data)
                entities.append(entity)
            
            return entities
    
    async def find_similar_entities(
        self,
        entity: GraphEntity,
        threshold: float = 0.8
    ) -> List[GraphEntity]:
        """Find similar entities using name similarity and type matching."""
        
        async with self.driver.session() as session:
            # Use fuzzy string matching (simplified version)
            query = """
            MATCH (e:Entity)
            WHERE e.entity_type = $entity_type
            AND e.id <> $entity_id
            AND (
                e.name CONTAINS $name_part
                OR $name CONTAINS substring(e.name, 0, size(e.name)/2)
            )
            RETURN e, 
                   CASE WHEN e.name = $name THEN 1.0
                        WHEN e.name CONTAINS $name OR $name CONTAINS e.name THEN 0.8
                        ELSE 0.6
                   END as similarity_score
            ORDER BY similarity_score DESC
            LIMIT 10
            """
            
            name_part = entity.name[:len(entity.name)//2] if len(entity.name) > 4 else entity.name
            
            result = await session.run(query, {
                'entity_type': entity.entity_type.value,
                'entity_id': entity.id,
                'name': entity.name,
                'name_part': name_part
            })
            
            similar_entities = []
            
            async for record in result:
                similarity_score = record['similarity_score']
                if similarity_score >= threshold:
                    entity_data = dict(record['e'])
                    similar_entity = self._create_entity_from_data(entity_data)
                    similar_entities.append(similar_entity)
            
            return similar_entities
    
    async def get_entity_relationships(self, entity_id: str) -> List[GraphRelationship]:
        """Get all relationships for an entity."""
        
        async with self.driver.session() as session:
            query = """
            MATCH (e:Entity {id: $entity_id})
            MATCH (e)-[r:RELATES]-(other:Entity)
            RETURN r, 
                   CASE WHEN startNode(r).id = $entity_id 
                        THEN endNode(r).id 
                        ELSE startNode(r).id 
                   END as other_entity_id,
                   startNode(r).id as source_id,
                   endNode(r).id as target_id
            """
            
            result = await session.run(query, {'entity_id': entity_id})
            relationships = []
            
            async for record in result:
                rel_data = dict(record['r'])
                rel_data['source_entity_id'] = record['source_id']
                rel_data['target_entity_id'] = record['target_id']
                relationship = self._create_relationship_from_data(rel_data)
                relationships.append(relationship)
            
            return relationships
    
    async def execute_cypher_query(
        self,
        query: str,
        parameters: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Execute custom Cypher query."""
        
        async with self.driver.session() as session:
            result = await session.run(query, parameters)
            records = []
            
            async for record in result:
                records.append(dict(record))
            
            return records
    
    async def get_all_entities(self) -> Dict[str, GraphEntity]:
        """Fetch all entities from Neo4j."""
        async with self.driver.session() as session:
            result = await session.run("MATCH (e:Entity) RETURN e")
            entities = {}
            async for record in result:
                data = dict(record['e'])
                try:
                    entity = self._create_entity_from_data(data)
                    entities[entity.id] = entity
                except Exception as exc:
                    self.logger.debug("Skipping entity: %s", exc)
            return entities

    async def get_all_relationships(self) -> Dict[str, GraphRelationship]:
        """Fetch all relationships from Neo4j."""
        async with self.driver.session() as session:
            result = await session.run(
                "MATCH ()-[r:RELATES]->() RETURN r, startNode(r).id as source_id, endNode(r).id as target_id"
            )
            rels = {}
            async for record in result:
                data = dict(record['r'])
                data['source_entity_id'] = record['source_id']
                data['target_entity_id'] = record['target_id']
                try:
                    rel = self._create_relationship_from_data(data)
                    rels[rel.id] = rel
                except Exception as exc:
                    self.logger.debug("Skipping relationship: %s", exc)
            return rels

    async def get_subgraph_slice(
        self,
        seed_types: Sequence[str],
        per_type_limit: int,
        exclude_types: Sequence[str],
        max_neighbors: int,
    ) -> Tuple[Dict[str, GraphEntity], Dict[str, GraphRelationship], Set[str], Dict[str, int], int, int]:
        """Indexed Cypher implementation of the subgraph-slice fetch.

        Three small queries replace the full-graph scan:
          1. **Per type**: pick the newest ``per_type_limit`` entities
             whose ``entity_type`` matches — uses ``entity_type_idx``,
             so cost scales with ``per_type_limit``, not graph size.
          2. **Edges**: ``UNWIND`` the seed ids and traverse out via
             the ``entity_id_unique`` constraint. Each edge is returned
             with both endpoints; the other-end entity comes back in
             the same record so we don't need a follow-up id→entity
             round trip.
          3. **Counts**: two cheap label/relationship-type counts (O(1)
             via Neo4j's count store) for the response totals.

        The router still applies the per-seed fan-out cap, the
        max-neighbour cap and the orphan drop in Python — those are
        cheap on the smaller in-memory result set and keep the Cypher
        focused on data retrieval.
        """
        seed_types_list = [t for t in (seed_types or []) if t]
        exclude_lower = [t.lower() for t in (exclude_types or []) if t]

        entities: Dict[str, GraphEntity] = {}
        seed_per_type: Dict[str, int] = {}
        seed_ids_list: List[str] = []

        async with self.driver.session() as session:
            # ── Step 1: seeds, one query per type ────────────────
            for seed_type in seed_types_list:
                if seed_type.lower() in exclude_lower:
                    continue
                seed_query = (
                    "MATCH (e:Entity) "
                    "WHERE toLower(e.entity_type) = toLower($type) "
                    "RETURN e "
                    "ORDER BY e.created_at DESC "
                    "LIMIT $limit"
                )
                result = await session.run(
                    seed_query, {"type": seed_type, "limit": per_type_limit}
                )
                picked = 0
                async for record in result:
                    data = dict(record['e'])
                    try:
                        entity = self._create_entity_from_data(data)
                    except Exception as exc:
                        self.logger.debug("Skipping seed entity: %s", exc)
                        continue
                    if entity.id in entities:
                        continue
                    entities[entity.id] = entity
                    seed_ids_list.append(entity.id)
                    picked += 1
                if picked:
                    seed_per_type[seed_type] = picked

            relationships: Dict[str, GraphRelationship] = {}
            seed_ids = set(seed_ids_list)

            # ── Step 2: edges + other-end entities for the seeds ──
            # The undirected ``-[r]-`` traversal returns each edge
            # once per seed endpoint it touches; the dict keyed by
            # rel.id dedupes the seed-seed double-counting that
            # produces. ``startNode(r)`` / ``endNode(r)`` preserve
            # the stored direction so the response keeps the
            # source/target semantics the rest of the pipeline
            # expects.
            if seed_ids_list:
                edges_query = (
                    "UNWIND $ids AS sid "
                    "MATCH (s:Entity {id: sid})-[r:RELATES]-(t:Entity) "
                    "WHERE NOT toLower(t.entity_type) IN $excl "
                    "RETURN r, "
                    "       startNode(r).id AS src_id, "
                    "       endNode(r).id   AS tgt_id, "
                    "       t "
                    "LIMIT $cap"
                )
                # Loose row cap: enough headroom for fan-out caps
                # without flooding the response if a hub has tens
                # of thousands of edges. Tighter caps are applied
                # by the router after dedup.
                row_cap = max(len(seed_ids_list) * 200, max_neighbors * 4, 2000)
                result = await session.run(
                    edges_query,
                    {"ids": seed_ids_list, "excl": exclude_lower, "cap": row_cap},
                )
                neighbor_count = 0
                async for record in result:
                    other_data = dict(record['t'])
                    other_id = other_data.get('id')
                    if not other_id:
                        continue

                    # If the other end isn't already known, try to
                    # add it — bounded by ``max_neighbors`` so a
                    # single hub can't blow up the payload. When the
                    # cap is hit we drop the edge too, since
                    # rendering a half-edge is worse than dropping it.
                    if other_id not in entities:
                        if other_id not in seed_ids and neighbor_count >= max_neighbors:
                            continue
                        try:
                            other = self._create_entity_from_data(other_data)
                        except Exception as exc:
                            self.logger.debug("Skipping neighbour: %s", exc)
                            continue
                        entities[other.id] = other
                        if other.id not in seed_ids:
                            neighbor_count += 1

                    edge_data = dict(record['r'])
                    edge_data['source_entity_id'] = record['src_id']
                    edge_data['target_entity_id'] = record['tgt_id']
                    try:
                        rel = self._create_relationship_from_data(edge_data)
                    except Exception as exc:
                        self.logger.debug("Skipping edge: %s", exc)
                        continue
                    relationships[rel.id] = rel

            # ── Step 3: cheap global counts for the response ─────
            total_entities = 0
            total_relationships = 0
            try:
                r1 = await session.run("MATCH (e:Entity) RETURN count(e) AS c")
                rec = await r1.single()
                if rec is not None:
                    total_entities = int(rec['c'])
                r2 = await session.run("MATCH ()-[r:RELATES]->() RETURN count(r) AS c")
                rec = await r2.single()
                if rec is not None:
                    total_relationships = int(rec['c'])
            except Exception as exc:
                self.logger.debug("Total count query failed: %s", exc)

        return entities, relationships, seed_ids, seed_per_type, total_entities, total_relationships

    async def search_entities_by_text(self, terms: List[str], limit: int = 50) -> Dict[str, GraphEntity]:
        """Search entities using the full-text index on name/description."""
        if not terms:
            return {}

        # Build a Lucene query: term1 OR term2 OR ...
        lucene_query = " OR ".join(terms)
        query = (
            "CALL db.index.fulltext.queryNodes('entity_search', $query, {limit: $limit}) "
            "YIELD node, score "
            "RETURN node"
        )
        async with self.driver.session() as session:
            result = await session.run(query, {"query": lucene_query, "limit": limit})
            entities = {}
            async for record in result:
                data = dict(record['node'])
                try:
                    entity = self._create_entity_from_data(data)
                    entities[entity.id] = entity
                except Exception as exc:
                    self.logger.debug("Skipping entity from search: %s", exc)
            return entities

    async def vector_search_entities(
        self, query_embedding: List[float], top_k: int = 10, min_score: float = 0.5
    ) -> List[Tuple[GraphEntity, float]]:
        """Find entities whose name_embedding is similar to *query_embedding*."""
        query = (
            "CALL db.index.vector.queryNodes('entity_name_vector', $k, $embedding) "
            "YIELD node, score "
            "WHERE score >= $min_score "
            "RETURN node, score "
            "ORDER BY score DESC"
        )
        async with self.driver.session() as session:
            result = await session.run(query, {"k": top_k, "embedding": query_embedding, "min_score": min_score})
            hits: List[Tuple[GraphEntity, float]] = []
            async for record in result:
                data = dict(record['node'])
                try:
                    entity = self._create_entity_from_data(data)
                    hits.append((entity, float(record['score'])))
                except Exception as exc:
                    self.logger.debug("Skipping entity from vector search: %s", exc)
            return hits

    async def vector_search_relationships(
        self, query_embedding: List[float], top_k: int = 10, min_score: float = 0.5
    ) -> List[Tuple[GraphRelationship, float]]:
        """Find relationships whose connected entities have similar embeddings.

        Since Neo4j vector indexes only work on nodes, this performs a vector
        search on Entity nodes first, then returns all relationships between
        the matching entities.
        """
        entity_hits = await self.vector_search_entities(query_embedding, top_k=top_k * 2, min_score=min_score)
        if not entity_hits:
            return []

        entity_ids = [e.id for e, _ in entity_hits]
        entity_scores = {e.id: s for e, s in entity_hits}

        query = (
            "MATCH (src:Entity)-[r:RELATES]->(tgt:Entity) "
            "WHERE src.id IN $ids OR tgt.id IN $ids "
            "RETURN r, src.id as source_id, tgt.id as target_id"
        )
        async with self.driver.session() as session:
            result = await session.run(query, {"ids": entity_ids})
            hits: List[Tuple[GraphRelationship, float]] = []
            async for record in result:
                data = dict(record['r'])
                data['source_entity_id'] = record['source_id']
                data['target_entity_id'] = record['target_id']
                try:
                    rel = self._create_relationship_from_data(data)
                    # Score = max of the two entity match scores
                    score = max(
                        entity_scores.get(record['source_id'], 0.0),
                        entity_scores.get(record['target_id'], 0.0),
                    )
                    hits.append((rel, score))
                except Exception as exc:
                    self.logger.debug("Skipping relationship from vector search: %s", exc)
            hits.sort(key=lambda x: x[1], reverse=True)
            return hits[:top_k]

    async def get_graph_statistics(self) -> Dict[str, Any]:
        """Get comprehensive graph statistics."""
        
        async with self.driver.session() as session:
            stats_query = """
            MATCH (e:Entity)
            OPTIONAL MATCH (e)-[r:RELATES]-()
            RETURN 
                count(DISTINCT e) as total_entities,
                count(DISTINCT r) as total_relationships,
                e.entity_type as entity_type,
                count(e) as entity_count
            """
            
            result = await session.run(stats_query)
            
            statistics = {
                'total_entities': 0,
                'total_relationships': 0,
                'entity_types': {},
                'relationship_types': {},
                'graph_density': 0.0,
                'connected_components': 0
            }
            
            async for record in result:
                statistics['total_entities'] = record['total_entities']
                statistics['total_relationships'] = record['total_relationships']
                
                entity_type = record['entity_type']
                entity_count = record['entity_count']
                if entity_type:
                    statistics['entity_types'][entity_type] = entity_count
            
            # Calculate graph density
            n = statistics['total_entities']
            if n > 1:
                max_edges = n * (n - 1) / 2
                statistics['graph_density'] = statistics['total_relationships'] / max_edges
            
            return statistics
    
    async def merge_entities(
        self,
        primary_entity_id: str,
        duplicate_entity_id: str
    ) -> GraphEntity:
        """Merge duplicate entities and transfer relationships."""
        
        async with self.driver.session() as session:
            merge_query = """
            MATCH (primary:Entity {id: $primary_id})
            MATCH (duplicate:Entity {id: $duplicate_id})
            
            // Transfer relationships from duplicate to primary
            MATCH (duplicate)-[old_rel:RELATES]-(other:Entity)
            WHERE other.id <> $primary_id
            MERGE (primary)-[new_rel:RELATES {
                id: randomUUID(),
                relationship_type: old_rel.relationship_type,
                strength: old_rel.strength,
                created_at: datetime()
            }]-(other)
            SET new_rel += old_rel
            
            // Merge properties from duplicate into primary
            SET primary.aliases = CASE 
                WHEN primary.aliases IS NULL THEN [duplicate.name]
                WHEN duplicate.name IN primary.aliases THEN primary.aliases
                ELSE primary.aliases + [duplicate.name]
            END
            
            // Delete duplicate entity and its relationships
            DETACH DELETE duplicate
            
            RETURN primary
            """
            
            result = await session.run(merge_query, {
                'primary_id': primary_entity_id,
                'duplicate_id': duplicate_entity_id
            })
            
            record = await result.single()
            if record:
                entity_data = dict(record['primary'])
                return self._create_entity_from_data(entity_data)
            else:
                raise RuntimeError(f"Failed to merge entities {primary_entity_id} and {duplicate_entity_id}")
    
    def _create_entity_from_data(self, data: Dict[str, Any]) -> GraphEntity:
        """Create GraphEntity from database data."""

        # Handle enum conversion
        entity_type = EntityType(data.get('entity_type', 'CONCEPT'))

        entity = GraphEntity(
            name=data.get('name', ''),
            entity_type=entity_type,
            description=data.get('description'),
            properties=_decode_props(data.get('properties')),
            aliases=set(data.get('aliases', [])) if data.get('aliases') else set(),
            external_ids=_decode_props(data.get('external_ids')),
            source_chunk_ids=list(data.get('source_chunk_ids') or []),
            source_document_ids=list(data.get('source_document_ids') or []),
        )
        entity.id = data.get('id', entity.id)

        # Restore metadata
        if 'created_at' in data:
            entity.metadata.created_at = self._parse_datetime(data['created_at'])
        if 'updated_at' in data:
            entity.metadata.updated_at = self._parse_datetime(data['updated_at'])
        if 'version' in data:
            entity.metadata.version = data['version']
        if 'confidence_score' in data:
            entity.metadata.confidence_score = data['confidence_score']
        if 'source_trust' in data and data['source_trust']:
            entity.metadata.source_trust = data['source_trust']

        # `metadata` was serialised as a JSON string by ``_to_neo4j_props``
        # because it's a nested dict. Parse it back so annotations + tags
        # survive the round-trip — needed for the Curation queue (which
        # filters by ``annotations.verification_status``).
        meta_blob = _decode_props(data.get('metadata'))
        if meta_blob:
            ann = meta_blob.get('annotations')
            if isinstance(ann, dict):
                entity.metadata.annotations.update(ann)
            tags = meta_blob.get('tags')
            if isinstance(tags, list):
                for t in tags:
                    if t:
                        entity.metadata.tags.add(t)

        return entity
    
    def _create_relationship_from_data(self, data: Dict[str, Any]) -> GraphRelationship:
        """Create GraphRelationship from database data."""
        
        # Handle enum conversion
        relationship_type = RelationshipType(data.get('relationship_type', 'RELATED_TO'))
        
        relationship = GraphRelationship(
            source_entity_id=data.get('source_entity_id', ''),
            target_entity_id=data.get('target_entity_id', ''),
            relationship_type=relationship_type,
            description=data.get('description'),
            properties=_decode_props(data.get('properties')),
            strength=data.get('strength', 1.0),
            source_chunk_ids=list(data.get('source_chunk_ids') or []),
            source_document_ids=list(data.get('source_document_ids') or []),
        )
        relationship.id = data.get('id', relationship.id)

        # Handle temporal validity
        if 'temporal_validity' in data and data['temporal_validity']:
            temporal_data = data['temporal_validity']
            start_date = self._parse_datetime(temporal_data.get('start_date')) if temporal_data.get('start_date') else None
            end_date = self._parse_datetime(temporal_data.get('end_date')) if temporal_data.get('end_date') else None
            relationship.set_temporal_validity(start_date, end_date)

        # Restore metadata
        if 'created_at' in data:
            relationship.metadata.created_at = self._parse_datetime(data['created_at'])
        if 'updated_at' in data:
            relationship.metadata.updated_at = self._parse_datetime(data['updated_at'])
        if 'version' in data:
            relationship.metadata.version = data['version']
        if 'confidence_score' in data:
            relationship.metadata.confidence_score = data['confidence_score']
        if 'source_trust' in data and data['source_trust']:
            relationship.metadata.source_trust = data['source_trust']

        # Restore annotations from the JSON-stringified metadata blob.
        meta_blob = _decode_props(data.get('metadata'))
        if meta_blob:
            ann = meta_blob.get('annotations')
            if isinstance(ann, dict):
                relationship.metadata.annotations.update(ann)
            tags = meta_blob.get('tags')
            if isinstance(tags, list):
                for t in tags:
                    if t:
                        relationship.metadata.tags.add(t)

        return relationship
    
    def _parse_datetime(self, dt_value) -> datetime:
        """Parse datetime from various formats."""
        if isinstance(dt_value, datetime):
            return dt_value
        elif isinstance(dt_value, str):
            return datetime.fromisoformat(dt_value.replace('Z', '+00:00'))
        else:
            return datetime.now(timezone.utc)


class InMemoryGraphRepository(GraphRepositoryInterface):
    """
    In-memory implementation for testing and development.
    
    Provides simple in-memory graph storage for testing
    and development environments.
    """
    
    def __init__(self, config: GraphBuilderConfig):
        self.config = config
        self.entities: Dict[str, GraphEntity] = {}
        self.relationships: Dict[str, GraphRelationship] = {}
        self.logger = logging.getLogger(self.__class__.__name__)
    
    async def save_entity(self, entity: GraphEntity) -> GraphEntity:
        """Save entity to memory."""
        self.entities[entity.id] = entity
        self.logger.debug(f"Saved entity to memory: {entity.id}")
        return entity
    
    async def get_entity_by_id(self, entity_id: str) -> Optional[GraphEntity]:
        """Get entity by ID from memory."""
        return self.entities.get(entity_id)
    
    async def save_relationship(self, relationship: GraphRelationship) -> GraphRelationship:
        """Save relationship to memory."""
        # Validate that entities exist
        if (relationship.source_entity_id not in self.entities or
            relationship.target_entity_id not in self.entities):
            raise ValueError("Source or target entity not found")
        
        self.relationships[relationship.id] = relationship
        self.logger.debug(f"Saved relationship to memory: {relationship.id}")
        return relationship
    
    async def get_relationship_by_id(self, relationship_id: str) -> Optional[GraphRelationship]:
        """Get relationship by ID from memory."""
        return self.relationships.get(relationship_id)
    
    async def find_entities_by_type(self, entity_type: EntityType) -> List[GraphEntity]:
        """Find entities by type in memory."""
        return [
            entity for entity in self.entities.values()
            if entity.entity_type == entity_type
        ]
    
    async def find_similar_entities(
        self,
        entity: GraphEntity,
        threshold: float = 0.8
    ) -> List[GraphEntity]:
        """Find similar entities in memory using simple name matching."""
        similar = []
        
        for other_entity in self.entities.values():
            if (other_entity.id != entity.id and
                other_entity.entity_type == entity.entity_type):
                
                # Simple similarity check
                similarity = self._calculate_name_similarity(entity.name, other_entity.name)
                if similarity >= threshold:
                    similar.append(other_entity)
        
        return similar
    
    async def get_entity_relationships(self, entity_id: str) -> List[GraphRelationship]:
        """Get all relationships for an entity in memory."""
        return [
            rel for rel in self.relationships.values()
            if rel.source_entity_id == entity_id or rel.target_entity_id == entity_id
        ]
    
    async def execute_cypher_query(
        self,
        query: str,
        parameters: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Execute custom query (not supported in memory implementation)."""
        raise NotImplementedError("Custom queries not supported in memory implementation")

    async def get_all_entities(self) -> Dict[str, GraphEntity]:
        return dict(self.entities)

    async def get_all_relationships(self) -> Dict[str, GraphRelationship]:
        return dict(self.relationships)

    async def search_entities_by_text(self, terms: List[str], limit: int = 50) -> Dict[str, GraphEntity]:
        """Search entities by substring match on name/description (in-memory fallback)."""
        if not terms:
            return {}
        lower_terms = [t.lower() for t in terms]
        matches: Dict[str, GraphEntity] = {}
        for eid, ent in self.entities.items():
            text = f"{ent.name} {ent.description or ''}".lower()
            if any(t in text for t in lower_terms):
                matches[eid] = ent
                if len(matches) >= limit:
                    break
        return matches

    async def get_subgraph_slice(
        self,
        seed_types: Sequence[str],
        per_type_limit: int,
        exclude_types: Sequence[str],
        max_neighbors: int,
    ) -> Tuple[Dict[str, GraphEntity], Dict[str, GraphRelationship], Set[str], Dict[str, int], int, int]:
        """In-memory equivalent of the Neo4j fast path.

        The cost is similar to the previous full-scan implementation
        because everything lives in dicts already, but mirroring the
        shape of the Neo4j path keeps the router code uniform.
        """
        seed_types_list = [t for t in (seed_types or []) if t]
        exclude_lower = {t.lower() for t in (exclude_types or []) if t}
        wanted_lower = {t.lower() for t in seed_types_list if t.lower() not in exclude_lower}

        # Group by lowercased type, sort each bucket newest first.
        buckets: Dict[str, List[GraphEntity]] = {}
        for ent in self.entities.values():
            type_lower = ent.entity_type.value.lower()
            if type_lower in exclude_lower or type_lower not in wanted_lower:
                continue
            buckets.setdefault(type_lower, []).append(ent)

        epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
        entities: Dict[str, GraphEntity] = {}
        seed_per_type: Dict[str, int] = {}
        seed_ids: Set[str] = set()

        for seed_type in seed_types_list:
            bucket = buckets.get(seed_type.lower(), [])
            bucket.sort(
                key=lambda e: (e.metadata.created_at if e.metadata and e.metadata.created_at else epoch),
                reverse=True,
            )
            picked = 0
            for ent in bucket[:per_type_limit]:
                if ent.id in entities:
                    continue
                entities[ent.id] = ent
                seed_ids.add(ent.id)
                picked += 1
            if picked:
                seed_per_type[seed_type] = picked

        relationships: Dict[str, GraphRelationship] = {}
        neighbor_count = 0
        for rel in self.relationships.values():
            src = self.entities.get(rel.source_entity_id)
            tgt = self.entities.get(rel.target_entity_id)
            if src is None or tgt is None:
                continue
            if src.entity_type.value.lower() in exclude_lower:
                continue
            if tgt.entity_type.value.lower() in exclude_lower:
                continue
            if rel.source_entity_id not in seed_ids and rel.target_entity_id not in seed_ids:
                continue
            for endpoint in (src, tgt):
                if endpoint.id in entities:
                    continue
                if endpoint.id not in seed_ids and neighbor_count >= max_neighbors:
                    # Cap reached — break out and skip this edge below.
                    break
                entities[endpoint.id] = endpoint
                if endpoint.id not in seed_ids:
                    neighbor_count += 1
            else:
                relationships[rel.id] = rel

        return (
            entities,
            relationships,
            seed_ids,
            seed_per_type,
            len(self.entities),
            len(self.relationships),
        )
    
    def _calculate_name_similarity(self, name1: str, name2: str) -> float:
        """Calculate simple name similarity score."""
        name1_lower = name1.lower()
        name2_lower = name2.lower()
        
        if name1_lower == name2_lower:
            return 1.0
        elif name1_lower in name2_lower or name2_lower in name1_lower:
            return 0.8
        else:
            # Simple character overlap calculation
            common_chars = set(name1_lower) & set(name2_lower)
            total_chars = set(name1_lower) | set(name2_lower)
            return len(common_chars) / len(total_chars) if total_chars else 0.0


# Factory function for creating appropriate repository
def create_graph_repository(config: GraphBuilderConfig, neo4j_driver=None) -> GraphRepositoryInterface:
    """Create graph repository based on configuration."""
    
    import os
    db_provider = os.getenv("DATABASE_PROVIDER", "in_memory")
    if db_provider == "neo4j" and neo4j_driver:
        return Neo4jGraphRepository(config, neo4j_driver)
    else:
        return InMemoryGraphRepository(config)