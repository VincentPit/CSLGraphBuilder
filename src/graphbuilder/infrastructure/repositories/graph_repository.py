"""
Graph Repository - Sophisticated data access layer for knowledge graph operations.

This module provides enterprise-grade repository pattern implementation
for graph entities and relationships with advanced querying capabilities.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Set, Tuple
from datetime import datetime, timezone
from abc import ABC, abstractmethod

from ...domain.models.graph_models import (
    GraphEntity, GraphRelationship, KnowledgeGraph,
    EntityType, RelationshipType
)
from ..config.settings import GraphBuilderConfig


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
    
    async def save_entity(self, entity: GraphEntity) -> GraphEntity:
        """Save entity to Neo4j graph database."""
        
        async with self.driver.session() as session:
            # Check for existing entity with same name and type
            existing_query = """
            MATCH (e:Entity)
            WHERE e.name = $name AND e.entity_type = $entity_type
            RETURN e.id as existing_id
            """
            
            result = await session.run(existing_query, {
                'name': entity.name,
                'entity_type': entity.entity_type.value
            })
            
            existing_record = await result.single()
            
            if existing_record:
                # Update existing entity
                existing_id = existing_record['existing_id']
                update_query = """
                MATCH (e:Entity {id: $existing_id})
                SET e += $properties,
                    e.updated_at = datetime(),
                    e.version = e.version + 1
                RETURN e
                """
                
                properties = entity.to_dict()
                properties.pop('id', None)
                
                await session.run(update_query, {
                    'existing_id': existing_id,
                    'properties': properties
                })
                
                entity.id = existing_id
                self.logger.debug(f"Updated existing entity: {entity.id}")
                
            else:
                # Create new entity
                create_query = """
                CREATE (e:Entity {id: $id})
                SET e += $properties,
                    e.created_at = datetime(),
                    e.updated_at = datetime(),
                    e.version = 1
                RETURN e
                """
                
                properties = entity.to_dict()
                properties.pop('id', None)
                properties['content_hash'] = entity.get_hash()
                
                await session.run(create_query, {
                    'id': entity.id,
                    'properties': properties
                })
                
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
        """Save relationship to Neo4j graph database."""
        
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
            
            # Create relationship
            create_query = """
            MATCH (source:Entity {id: $source_id}), (target:Entity {id: $target_id})
            MERGE (source)-[r:RELATES {id: $relationship_id}]->(target)
            SET r += $properties,
                r.created_at = datetime(),
                r.updated_at = datetime(),
                r.version = 1
            RETURN r
            """
            
            properties = relationship.to_dict()
            properties.pop('id', None)
            
            await session.run(create_query, {
                'source_id': relationship.source_entity_id,
                'target_id': relationship.target_entity_id,
                'relationship_id': relationship.id,
                'properties': properties
            })
            
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
            id=data.get('id'),
            name=data.get('name', ''),
            entity_type=entity_type,
            description=data.get('description'),
            properties=data.get('properties', {}),
            aliases=set(data.get('aliases', [])),
            external_ids=data.get('external_ids', {})
        )
        
        # Restore metadata
        if 'created_at' in data:
            entity.metadata.created_at = self._parse_datetime(data['created_at'])
        if 'updated_at' in data:
            entity.metadata.updated_at = self._parse_datetime(data['updated_at'])
        if 'version' in data:
            entity.metadata.version = data['version']
        if 'confidence_score' in data:
            entity.metadata.confidence_score = data['confidence_score']
        
        return entity
    
    def _create_relationship_from_data(self, data: Dict[str, Any]) -> GraphRelationship:
        """Create GraphRelationship from database data."""
        
        # Handle enum conversion
        relationship_type = RelationshipType(data.get('relationship_type', 'RELATED_TO'))
        
        relationship = GraphRelationship(
            id=data.get('id'),
            source_entity_id=data.get('source_entity_id', ''),
            target_entity_id=data.get('target_entity_id', ''),
            relationship_type=relationship_type,
            description=data.get('description'),
            properties=data.get('properties', {}),
            strength=data.get('strength', 1.0)
        )
        
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
    
    if config.database.provider == "neo4j" and neo4j_driver:
        return Neo4jGraphRepository(config, neo4j_driver)
    else:
        return InMemoryGraphRepository(config)