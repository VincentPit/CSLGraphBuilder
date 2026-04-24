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


class GraphRepositoryInterface(ABC):
    """Abstract interface for graph repository operations."""
    
    @abstractmethod
    async def save_entity(self, entity: GraphEntity) -> GraphEntity:
        """Save an entity to the graph."""
        pass
    
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

    def _embed_text(self, text: str) -> Optional[List[float]]:
        """Produce an embedding vector for *text*, or None if unavailable."""
        model = self._get_embedding_model()
        if model is None or not text:
            return None
        vec = model.encode(text, convert_to_numpy=True)
        return vec.tolist()

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
    
    async def save_entity(self, entity: GraphEntity) -> GraphEntity:
        """Save entity to Neo4j graph database with provenance tracking."""
        
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
                emb = self._embed_text(self._entity_embedding_text(entity))
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
                emb = self._embed_text(self._entity_embedding_text(entity))
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
    
    async def save_relationship(self, relationship: GraphRelationship) -> GraphRelationship:
        """Save relationship to Neo4j graph database with provenance tracking.
        
        If a relationship between the same source/target with the same type
        already exists, merges provenance (source chunks) instead of creating
        a duplicate.
        """
        
        async with self.driver.session() as session:
            # Check if entities exist
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

            # Extract entity names for embedding text
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
                emb = self._embed_text(self._relationship_embedding_text(
                    relationship, source_name, target_name
                ))

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
                emb = self._embed_text(self._relationship_embedding_text(
                    relationship, source_name, target_name
                ))
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