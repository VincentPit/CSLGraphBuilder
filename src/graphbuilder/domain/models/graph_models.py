"""
Domain Models - Core business entities with rich domain logic.

This module contains sophisticated domain models that encapsulate business rules
and provide type-safe, validated entities for the knowledge graph domain.
"""

import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import hashlib
from abc import ABC, abstractmethod


class EntityType(Enum):
    """Standard entity types for knowledge graphs."""
    PERSON = "Person"
    ORGANIZATION = "Organization"
    LOCATION = "Location"
    PRODUCT = "Product"
    TECHNOLOGY = "Technology"
    CONCEPT = "Concept"
    EVENT = "Event"
    DOCUMENT = "Document"
    CATEGORY = "Category"
    BRAND = "Brand"
    FEATURE = "Feature"
    SPECIFICATION = "Specification"


class RelationshipType(Enum):
    """Standard relationship types for knowledge graphs."""
    RELATED_TO = "RELATED_TO"
    PART_OF = "PART_OF"
    CONTAINS = "CONTAINS"
    LOCATED_IN = "LOCATED_IN"
    WORKS_FOR = "WORKS_FOR"
    MANUFACTURES = "MANUFACTURES"
    DEVELOPS = "DEVELOPS"
    HAS_FEATURE = "HAS_FEATURE"
    COMPATIBLE_WITH = "COMPATIBLE_WITH"
    DEPENDS_ON = "DEPENDS_ON"
    INFLUENCES = "INFLUENCES"
    AUTHORED_BY = "AUTHORED_BY"
    PUBLISHED_BY = "PUBLISHED_BY"
    CATEGORIZED_AS = "CATEGORIZED_AS"


class ProcessingStatus(Enum):
    """Processing status for documents and entities."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRY = "retry"


class ConfidenceLevel(Enum):
    """Confidence levels for extracted entities and relationships."""
    LOW = "low"           # 0.0 - 0.4
    MEDIUM = "medium"     # 0.4 - 0.7
    HIGH = "high"         # 0.7 - 0.9
    VERY_HIGH = "very_high"  # 0.9 - 1.0


@dataclass
class Metadata:
    """Rich metadata for all domain entities."""
    
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    created_by: Optional[str] = None
    updated_by: Optional[str] = None
    version: int = 1
    tags: Set[str] = field(default_factory=set)
    annotations: Dict[str, Any] = field(default_factory=dict)
    source_system: Optional[str] = None
    confidence_score: Optional[float] = None
    
    def update(self, updated_by: Optional[str] = None) -> None:
        """Update metadata with new timestamp and version."""
        self.updated_at = datetime.now(timezone.utc)
        self.updated_by = updated_by
        self.version += 1
    
    def add_tag(self, tag: str) -> None:
        """Add a tag to the metadata."""
        self.tags.add(tag.strip().lower())
    
    def add_annotation(self, key: str, value: Any) -> None:
        """Add an annotation to the metadata."""
        self.annotations[key] = value
    
    def get_confidence_level(self) -> Optional[ConfidenceLevel]:
        """Get confidence level based on score."""
        if self.confidence_score is None:
            return None
        
        if self.confidence_score < 0.4:
            return ConfidenceLevel.LOW
        elif self.confidence_score < 0.7:
            return ConfidenceLevel.MEDIUM
        elif self.confidence_score < 0.9:
            return ConfidenceLevel.HIGH
        else:
            return ConfidenceLevel.VERY_HIGH


class DomainEntity(ABC):
    """Abstract base class for all domain entities."""
    
    def __init__(self, id: Optional[str] = None):
        self.id = id or str(uuid.uuid4())
        self.metadata = Metadata()
    
    @abstractmethod
    def validate(self) -> bool:
        """Validate entity state."""
        pass
    
    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Convert entity to dictionary representation."""
        pass
    
    def get_hash(self) -> str:
        """Generate content hash for the entity."""
        content = str(sorted(self.to_dict().items()))
        return hashlib.sha256(content.encode()).hexdigest()


@dataclass
class GraphEntity(DomainEntity):
    """
    Rich graph entity with properties, relationships, and metadata.
    
    Represents nodes in the knowledge graph with sophisticated features
    for entity resolution, relationship management, and provenance tracking.
    """
    
    name: str
    entity_type: EntityType
    description: Optional[str] = None
    properties: Dict[str, Any] = field(default_factory=dict)
    aliases: Set[str] = field(default_factory=set)
    external_ids: Dict[str, str] = field(default_factory=dict)
    
    def __post_init__(self):
        """Post-initialization setup."""
        if not hasattr(self, 'id'):
            super().__init__()
        self.validate()
    
    def validate(self) -> bool:
        """Validate entity state."""
        if not self.name or not self.name.strip():
            raise ValueError("Entity name cannot be empty")
        
        if not isinstance(self.entity_type, EntityType):
            raise ValueError("Invalid entity type")
        
        return True
    
    def add_alias(self, alias: str) -> None:
        """Add an alias for this entity."""
        if alias and alias.strip():
            self.aliases.add(alias.strip())
            self.metadata.update()
    
    def add_property(self, key: str, value: Any) -> None:
        """Add or update a property."""
        self.properties[key] = value
        self.metadata.update()
    
    def add_external_id(self, system: str, external_id: str) -> None:
        """Add external system identifier."""
        self.external_ids[system] = external_id
        self.metadata.update()
    
    def merge_with(self, other: 'GraphEntity') -> 'GraphEntity':
        """Merge this entity with another entity."""
        if self.entity_type != other.entity_type:
            raise ValueError("Cannot merge entities of different types")
        
        # Use the more confident entity as base
        base_entity = self if (self.metadata.confidence_score or 0) >= (other.metadata.confidence_score or 0) else other
        merged_entity = GraphEntity(
            id=base_entity.id,
            name=base_entity.name,
            entity_type=base_entity.entity_type,
            description=base_entity.description or other.description
        )
        
        # Merge properties
        merged_entity.properties.update(self.properties)
        merged_entity.properties.update(other.properties)
        
        # Merge aliases
        merged_entity.aliases.update(self.aliases)
        merged_entity.aliases.update(other.aliases)
        
        # Merge external IDs
        merged_entity.external_ids.update(self.external_ids)
        merged_entity.external_ids.update(other.external_ids)
        
        # Update metadata
        merged_entity.metadata.tags.update(self.metadata.tags)
        merged_entity.metadata.tags.update(other.metadata.tags)
        merged_entity.metadata.annotations.update(self.metadata.annotations)
        merged_entity.metadata.annotations.update(other.metadata.annotations)
        
        return merged_entity
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "name": self.name,
            "entity_type": self.entity_type.value,
            "description": self.description,
            "properties": self.properties,
            "aliases": list(self.aliases),
            "external_ids": self.external_ids,
            "metadata": {
                "created_at": self.metadata.created_at.isoformat(),
                "updated_at": self.metadata.updated_at.isoformat(),
                "version": self.metadata.version,
                "tags": list(self.metadata.tags),
                "annotations": self.metadata.annotations,
                "confidence_score": self.metadata.confidence_score
            }
        }


@dataclass
class GraphRelationship(DomainEntity):
    """
    Rich graph relationship with properties, provenance, and validation.
    
    Represents edges in the knowledge graph with sophisticated features
    for relationship validation, property management, and temporal tracking.
    """
    
    source_entity_id: str
    target_entity_id: str
    relationship_type: RelationshipType
    description: Optional[str] = None
    properties: Dict[str, Any] = field(default_factory=dict)
    strength: float = 1.0  # Relationship strength (0.0 - 1.0)
    temporal_validity: Optional[Dict[str, datetime]] = None  # start_date, end_date
    
    def __post_init__(self):
        """Post-initialization setup."""
        if not hasattr(self, 'id'):
            super().__init__()
        self.validate()
    
    def validate(self) -> bool:
        """Validate relationship state."""
        if not self.source_entity_id:
            raise ValueError("Source entity ID cannot be empty")
        
        if not self.target_entity_id:
            raise ValueError("Target entity ID cannot be empty")
        
        if self.source_entity_id == self.target_entity_id:
            raise ValueError("Self-relationships are not allowed")
        
        if not isinstance(self.relationship_type, RelationshipType):
            raise ValueError("Invalid relationship type")
        
        if not 0.0 <= self.strength <= 1.0:
            raise ValueError("Relationship strength must be between 0.0 and 1.0")
        
        return True
    
    def add_property(self, key: str, value: Any) -> None:
        """Add or update a property."""
        self.properties[key] = value
        self.metadata.update()
    
    def set_temporal_validity(self, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None) -> None:
        """Set temporal validity period."""
        if start_date and end_date and start_date > end_date:
            raise ValueError("Start date cannot be after end date")
        
        self.temporal_validity = {}
        if start_date:
            self.temporal_validity["start_date"] = start_date
        if end_date:
            self.temporal_validity["end_date"] = end_date
        
        self.metadata.update()
    
    def is_valid_at(self, timestamp: datetime) -> bool:
        """Check if relationship is valid at given timestamp."""
        if not self.temporal_validity:
            return True
        
        start_date = self.temporal_validity.get("start_date")
        end_date = self.temporal_validity.get("end_date")
        
        if start_date and timestamp < start_date:
            return False
        
        if end_date and timestamp > end_date:
            return False
        
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        result = {
            "id": self.id,
            "source_entity_id": self.source_entity_id,
            "target_entity_id": self.target_entity_id,
            "relationship_type": self.relationship_type.value,
            "description": self.description,
            "properties": self.properties,
            "strength": self.strength,
            "metadata": {
                "created_at": self.metadata.created_at.isoformat(),
                "updated_at": self.metadata.updated_at.isoformat(),
                "version": self.metadata.version,
                "tags": list(self.metadata.tags),
                "annotations": self.metadata.annotations,
                "confidence_score": self.metadata.confidence_score
            }
        }
        
        if self.temporal_validity:
            result["temporal_validity"] = {
                k: v.isoformat() if isinstance(v, datetime) else v
                for k, v in self.temporal_validity.items()
            }
        
        return result


@dataclass
class DocumentChunk(DomainEntity):
    """
    Rich document chunk with advanced processing metadata.
    
    Represents processed text chunks with sophisticated features for
    content analysis, embedding management, and processing provenance.
    """
    
    content: str
    document_id: str
    chunk_index: int
    token_count: int
    character_count: int
    start_position: int = 0
    end_position: int = 0
    language: Optional[str] = None
    content_type: str = "text/plain"
    processing_metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Post-initialization setup."""
        if not hasattr(self, 'id'):
            super().__init__()
        
        # Auto-calculate metrics if not provided
        if self.character_count == 0:
            self.character_count = len(self.content)
        
        if self.end_position == 0:
            self.end_position = self.start_position + self.character_count
        
        self.validate()
    
    def validate(self) -> bool:
        """Validate chunk state."""
        if not self.content or not self.content.strip():
            raise ValueError("Chunk content cannot be empty")
        
        if not self.document_id:
            raise ValueError("Document ID cannot be empty")
        
        if self.chunk_index < 0:
            raise ValueError("Chunk index must be non-negative")
        
        if self.token_count < 0:
            raise ValueError("Token count must be non-negative")
        
        if self.start_position > self.end_position:
            raise ValueError("Start position cannot be greater than end position")
        
        return True
    
    def get_content_hash(self) -> str:
        """Generate hash of chunk content."""
        return hashlib.sha256(self.content.encode()).hexdigest()
    
    def add_processing_metadata(self, processor: str, metadata: Dict[str, Any]) -> None:
        """Add processing metadata from a specific processor."""
        self.processing_metadata[processor] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metadata": metadata
        }
        self.metadata.update()
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics for the chunk."""
        words = len(self.content.split())
        sentences = self.content.count('.') + self.content.count('!') + self.content.count('?')
        
        return {
            "character_count": self.character_count,
            "token_count": self.token_count,
            "word_count": words,
            "sentence_count": sentences,
            "avg_word_length": self.character_count / words if words > 0 else 0,
            "content_density": self.token_count / self.character_count if self.character_count > 0 else 0
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "content": self.content,
            "document_id": self.document_id,
            "chunk_index": self.chunk_index,
            "token_count": self.token_count,
            "character_count": self.character_count,
            "start_position": self.start_position,
            "end_position": self.end_position,
            "language": self.language,
            "content_type": self.content_type,
            "content_hash": self.get_content_hash(),
            "processing_metadata": self.processing_metadata,
            "summary_stats": self.get_summary_stats(),
            "metadata": {
                "created_at": self.metadata.created_at.isoformat(),
                "updated_at": self.metadata.updated_at.isoformat(),
                "version": self.metadata.version,
                "tags": list(self.metadata.tags),
                "annotations": self.metadata.annotations,
                "confidence_score": self.metadata.confidence_score
            }
        }


@dataclass
class SourceDocument(DomainEntity):
    """
    Rich source document with comprehensive metadata and processing state.
    
    Represents source documents with sophisticated features for content
    management, processing tracking, and relationship management.
    """
    
    title: str
    content_type: str = "text/html"
    source_url: Optional[str] = None
    file_path: Optional[str] = None
    language: Optional[str] = None
    content_length: int = 0
    processing_status: ProcessingStatus = ProcessingStatus.PENDING
    error_message: Optional[str] = None
    
    # Processing metrics
    total_chunks: int = 0
    processed_chunks: int = 0
    extracted_entities: int = 0
    extracted_relationships: int = 0
    
    # Content metadata
    content_metadata: Dict[str, Any] = field(default_factory=dict)
    extraction_metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Post-initialization setup."""
        if not hasattr(self, 'id'):
            super().__init__()
        self.validate()
    
    def validate(self) -> bool:
        """Validate document state."""
        if not self.title or not self.title.strip():
            raise ValueError("Document title cannot be empty")
        
        if not self.source_url and not self.file_path:
            raise ValueError("Document must have either source URL or file path")
        
        if self.content_length < 0:
            raise ValueError("Content length must be non-negative")
        
        if self.processed_chunks > self.total_chunks:
            raise ValueError("Processed chunks cannot exceed total chunks")
        
        return True
    
    def update_processing_status(self, status: ProcessingStatus, error_message: Optional[str] = None) -> None:
        """Update processing status."""
        self.processing_status = status
        self.error_message = error_message
        self.metadata.update()
    
    def increment_processed_chunks(self) -> None:
        """Increment processed chunks counter."""
        self.processed_chunks += 1
        self.metadata.update()
    
    def set_extraction_results(self, entities: int, relationships: int) -> None:
        """Set extraction results."""
        self.extracted_entities = entities
        self.extracted_relationships = relationships
        self.metadata.update()
    
    def get_processing_progress(self) -> float:
        """Get processing progress as percentage."""
        if self.total_chunks == 0:
            return 0.0
        return (self.processed_chunks / self.total_chunks) * 100
    
    def add_content_metadata(self, key: str, value: Any) -> None:
        """Add content metadata."""
        self.content_metadata[key] = value
        self.metadata.update()
    
    def add_extraction_metadata(self, processor: str, metadata: Dict[str, Any]) -> None:
        """Add extraction metadata."""
        self.extraction_metadata[processor] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metadata": metadata
        }
        self.metadata.update()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "title": self.title,
            "content_type": self.content_type,
            "source_url": self.source_url,
            "file_path": self.file_path,
            "language": self.language,
            "content_length": self.content_length,
            "processing_status": self.processing_status.value,
            "error_message": self.error_message,
            "total_chunks": self.total_chunks,
            "processed_chunks": self.processed_chunks,
            "processing_progress": self.get_processing_progress(),
            "extracted_entities": self.extracted_entities,
            "extracted_relationships": self.extracted_relationships,
            "content_metadata": self.content_metadata,
            "extraction_metadata": self.extraction_metadata,
            "metadata": {
                "created_at": self.metadata.created_at.isoformat(),
                "updated_at": self.metadata.updated_at.isoformat(),
                "version": self.metadata.version,
                "tags": list(self.metadata.tags),
                "annotations": self.metadata.annotations,
                "confidence_score": self.metadata.confidence_score
            }
        }


class KnowledgeGraph:
    """
    Rich knowledge graph with advanced querying and analysis capabilities.
    
    Represents the complete knowledge graph with sophisticated features for
    entity resolution, relationship analysis, and graph operations.
    """
    
    def __init__(self, id: Optional[str] = None, name: str = "Knowledge Graph"):
        self.id = id or str(uuid.uuid4())
        self.name = name
        self.entities: Dict[str, GraphEntity] = {}
        self.relationships: Dict[str, GraphRelationship] = {}
        self.metadata = Metadata()
    
    def add_entity(self, entity: GraphEntity) -> None:
        """Add entity to the graph."""
        entity.validate()
        self.entities[entity.id] = entity
        self.metadata.update()
    
    def add_relationship(self, relationship: GraphRelationship) -> None:
        """Add relationship to the graph."""
        relationship.validate()
        
        # Ensure referenced entities exist
        if relationship.source_entity_id not in self.entities:
            raise ValueError(f"Source entity {relationship.source_entity_id} not found")
        if relationship.target_entity_id not in self.entities:
            raise ValueError(f"Target entity {relationship.target_entity_id} not found")
        
        self.relationships[relationship.id] = relationship
        self.metadata.update()
    
    def find_entities_by_type(self, entity_type: EntityType) -> List[GraphEntity]:
        """Find entities by type."""
        return [entity for entity in self.entities.values() if entity.entity_type == entity_type]
    
    def find_relationships_by_type(self, relationship_type: RelationshipType) -> List[GraphRelationship]:
        """Find relationships by type."""
        return [rel for rel in self.relationships.values() if rel.relationship_type == relationship_type]
    
    def get_entity_relationships(self, entity_id: str) -> List[GraphRelationship]:
        """Get all relationships for an entity."""
        return [
            rel for rel in self.relationships.values()
            if rel.source_entity_id == entity_id or rel.target_entity_id == entity_id
        ]
    
    def get_connected_entities(self, entity_id: str) -> List[GraphEntity]:
        """Get entities connected to the given entity."""
        connected_ids = set()
        
        for rel in self.relationships.values():
            if rel.source_entity_id == entity_id:
                connected_ids.add(rel.target_entity_id)
            elif rel.target_entity_id == entity_id:
                connected_ids.add(rel.source_entity_id)
        
        return [self.entities[entity_id] for entity_id in connected_ids if entity_id in self.entities]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get graph statistics."""
        entity_types = {}
        relationship_types = {}
        
        for entity in self.entities.values():
            entity_types[entity.entity_type.value] = entity_types.get(entity.entity_type.value, 0) + 1
        
        for rel in self.relationships.values():
            relationship_types[rel.relationship_type.value] = relationship_types.get(rel.relationship_type.value, 0) + 1
        
        return {
            "total_entities": len(self.entities),
            "total_relationships": len(self.relationships),
            "entity_types": entity_types,
            "relationship_types": relationship_types,
            "average_entity_degree": len(self.relationships) * 2 / len(self.entities) if self.entities else 0,
            "graph_density": len(self.relationships) / (len(self.entities) * (len(self.entities) - 1) / 2) if len(self.entities) > 1 else 0
        }