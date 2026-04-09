"""
Document Repository - Sophisticated data access layer for document operations.

This module provides enterprise-grade repository pattern implementation
for document and chunk persistence with advanced querying capabilities.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, AsyncGenerator
from datetime import datetime, timezone
from abc import ABC, abstractmethod

from ...domain.models.graph_models import SourceDocument, DocumentChunk
from ...domain.models.processing_models import ProcessingStatus
from ..config.settings import GraphBuilderConfig


class DocumentRepositoryInterface(ABC):
    """Abstract interface for document repository operations."""
    
    @abstractmethod
    async def save(self, document: SourceDocument) -> SourceDocument:
        """Save a document."""
        pass
    
    @abstractmethod
    async def get_by_id(self, document_id: str) -> Optional[SourceDocument]:
        """Get document by ID."""
        pass
    
    @abstractmethod
    async def update(self, document: SourceDocument) -> SourceDocument:
        """Update existing document."""
        pass
    
    @abstractmethod
    async def delete(self, document_id: str) -> bool:
        """Delete document."""
        pass
    
    @abstractmethod
    async def find_by_status(self, status: ProcessingStatus) -> List[SourceDocument]:
        """Find documents by processing status."""
        pass
    
    @abstractmethod
    async def save_chunk(self, chunk: DocumentChunk) -> DocumentChunk:
        """Save a document chunk."""
        pass
    
    @abstractmethod
    async def get_chunks_by_document_id(self, document_id: str) -> List[DocumentChunk]:
        """Get all chunks for a document."""
        pass


class Neo4jDocumentRepository(DocumentRepositoryInterface):
    """
    Neo4j implementation of document repository with sophisticated querying.
    
    Provides enterprise-grade document persistence using Neo4j graph database
    with advanced indexing, caching, and transaction management.
    """
    
    def __init__(self, config: GraphBuilderConfig, neo4j_driver):
        self.config = config
        self.driver = neo4j_driver
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize database schema
        asyncio.create_task(self._initialize_schema())
    
    async def _initialize_schema(self) -> None:
        """Initialize database schema and constraints."""
        
        async with self.driver.session() as session:
            # Create constraints and indexes
            constraints = [
                "CREATE CONSTRAINT document_id_unique IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE",
                "CREATE CONSTRAINT chunk_id_unique IF NOT EXISTS FOR (c:DocumentChunk) REQUIRE c.id IS UNIQUE",
                "CREATE INDEX document_status_idx IF NOT EXISTS FOR (d:Document) ON (d.processing_status)",
                "CREATE INDEX document_url_idx IF NOT EXISTS FOR (d:Document) ON (d.source_url)",
                "CREATE INDEX chunk_document_idx IF NOT EXISTS FOR (c:DocumentChunk) ON (c.document_id)",
                "CREATE INDEX chunk_index_idx IF NOT EXISTS FOR (c:DocumentChunk) ON (c.chunk_index)"
            ]
            
            for constraint in constraints:
                try:
                    await session.run(constraint)
                except Exception as e:
                    self.logger.debug(f"Constraint/index creation result: {str(e)}")
    
    async def save(self, document: SourceDocument) -> SourceDocument:
        """Save document to Neo4j database."""
        
        async with self.driver.session() as session:
            query = """
            MERGE (d:Document {id: $id})
            SET d += $properties,
                d.updated_at = datetime()
            RETURN d
            """
            
            properties = document.to_dict()
            properties.pop('id', None)  # Remove ID from properties
            
            result = await session.run(query, {
                'id': document.id,
                'properties': properties
            })
            
            record = await result.single()
            if record:
                self.logger.debug(f"Saved document: {document.id}")
                return document
            else:
                raise RuntimeError(f"Failed to save document: {document.id}")
    
    async def get_by_id(self, document_id: str) -> Optional[SourceDocument]:
        """Get document by ID from Neo4j database."""
        
        async with self.driver.session() as session:
            query = """
            MATCH (d:Document {id: $id})
            RETURN d
            """
            
            result = await session.run(query, {'id': document_id})
            record = await result.single()
            
            if record:
                doc_data = dict(record['d'])
                return self._create_document_from_data(doc_data)
            
            return None
    
    async def update(self, document: SourceDocument) -> SourceDocument:
        """Update existing document in Neo4j database."""
        
        document.metadata.update()
        return await self.save(document)
    
    async def delete(self, document_id: str) -> bool:
        """Delete document and related chunks from Neo4j database."""
        
        async with self.driver.session() as session:
            query = """
            MATCH (d:Document {id: $id})
            OPTIONAL MATCH (d)-[:HAS_CHUNK]->(c:DocumentChunk)
            DETACH DELETE d, c
            RETURN count(d) as deleted_count
            """
            
            result = await session.run(query, {'id': document_id})
            record = await result.single()
            
            deleted_count = record['deleted_count'] if record else 0
            self.logger.debug(f"Deleted document {document_id}: {deleted_count} documents")
            
            return deleted_count > 0
    
    async def find_by_status(self, status: ProcessingStatus) -> List[SourceDocument]:
        """Find documents by processing status."""
        
        async with self.driver.session() as session:
            query = """
            MATCH (d:Document)
            WHERE d.processing_status = $status
            RETURN d
            ORDER BY d.created_at DESC
            """
            
            result = await session.run(query, {'status': status.value})
            documents = []
            
            async for record in result:
                doc_data = dict(record['d'])
                document = self._create_document_from_data(doc_data)
                documents.append(document)
            
            return documents
    
    async def save_chunk(self, chunk: DocumentChunk) -> DocumentChunk:
        """Save document chunk to Neo4j database."""
        
        async with self.driver.session() as session:
            query = """
            MATCH (d:Document {id: $document_id})
            MERGE (c:DocumentChunk {id: $chunk_id})
            SET c += $properties,
                c.updated_at = datetime()
            MERGE (d)-[:HAS_CHUNK]->(c)
            RETURN c
            """
            
            properties = chunk.to_dict()
            properties.pop('id', None)
            
            result = await session.run(query, {
                'document_id': chunk.document_id,
                'chunk_id': chunk.id,
                'properties': properties
            })
            
            record = await result.single()
            if record:
                self.logger.debug(f"Saved chunk: {chunk.id}")
                return chunk
            else:
                raise RuntimeError(f"Failed to save chunk: {chunk.id}")
    
    async def get_chunks_by_document_id(self, document_id: str) -> List[DocumentChunk]:
        """Get all chunks for a document ordered by chunk index."""
        
        async with self.driver.session() as session:
            query = """
            MATCH (d:Document {id: $document_id})-[:HAS_CHUNK]->(c:DocumentChunk)
            RETURN c
            ORDER BY c.chunk_index ASC
            """
            
            result = await session.run(query, {'document_id': document_id})
            chunks = []
            
            async for record in result:
                chunk_data = dict(record['c'])
                chunk = self._create_chunk_from_data(chunk_data)
                chunks.append(chunk)
            
            return chunks
    
    async def find_documents_by_url_pattern(self, url_pattern: str) -> List[SourceDocument]:
        """Find documents matching URL pattern."""
        
        async with self.driver.session() as session:
            query = """
            MATCH (d:Document)
            WHERE d.source_url CONTAINS $pattern
            RETURN d
            ORDER BY d.created_at DESC
            """
            
            result = await session.run(query, {'pattern': url_pattern})
            documents = []
            
            async for record in result:
                doc_data = dict(record['d'])
                document = self._create_document_from_data(doc_data)
                documents.append(document)
            
            return documents
    
    async def get_processing_statistics(self) -> Dict[str, Any]:
        """Get document processing statistics."""
        
        async with self.driver.session() as session:
            query = """
            MATCH (d:Document)
            RETURN 
                d.processing_status as status,
                count(*) as count,
                avg(d.content_length) as avg_length,
                sum(d.extracted_entities) as total_entities,
                sum(d.extracted_relationships) as total_relationships
            """
            
            result = await session.run(query)
            statistics = {
                'by_status': {},
                'totals': {
                    'documents': 0,
                    'entities': 0,
                    'relationships': 0,
                    'avg_content_length': 0
                }
            }
            
            async for record in result:
                status = record['status']
                count = record['count']
                avg_length = record['avg_length'] or 0
                total_entities = record['total_entities'] or 0
                total_relationships = record['total_relationships'] or 0
                
                statistics['by_status'][status] = {
                    'count': count,
                    'avg_content_length': avg_length
                }
                
                statistics['totals']['documents'] += count
                statistics['totals']['entities'] += total_entities
                statistics['totals']['relationships'] += total_relationships
            
            if statistics['totals']['documents'] > 0:
                total_length = sum(
                    stats['count'] * stats['avg_content_length'] 
                    for stats in statistics['by_status'].values()
                )
                statistics['totals']['avg_content_length'] = (
                    total_length / statistics['totals']['documents']
                )
            
            return statistics
    
    def _create_document_from_data(self, data: Dict[str, Any]) -> SourceDocument:
        """Create SourceDocument from database data."""
        
        # Handle datetime conversion
        for field in ['created_at', 'updated_at']:
            if field in data and data[field]:
                if isinstance(data[field], str):
                    data[field] = datetime.fromisoformat(data[field].replace('Z', '+00:00'))
        
        # Handle enum conversion
        if 'processing_status' in data:
            data['processing_status'] = ProcessingStatus(data['processing_status'])
        
        document = SourceDocument(
            id=data.get('id'),
            title=data.get('title', ''),
            content_type=data.get('content_type', 'text/html'),
            source_url=data.get('source_url'),
            file_path=data.get('file_path'),
            language=data.get('language'),
            content_length=data.get('content_length', 0),
            processing_status=data.get('processing_status', ProcessingStatus.PENDING),
            error_message=data.get('error_message'),
            total_chunks=data.get('total_chunks', 0),
            processed_chunks=data.get('processed_chunks', 0),
            extracted_entities=data.get('extracted_entities', 0),
            extracted_relationships=data.get('extracted_relationships', 0),
            content_metadata=data.get('content_metadata', {}),
            extraction_metadata=data.get('extraction_metadata', {})
        )
        
        # Restore metadata if available
        if 'created_at' in data:
            document.metadata.created_at = data['created_at']
        if 'updated_at' in data:
            document.metadata.updated_at = data['updated_at']
        if 'version' in data:
            document.metadata.version = data['version']
        
        return document
    
    def _create_chunk_from_data(self, data: Dict[str, Any]) -> DocumentChunk:
        """Create DocumentChunk from database data."""
        
        # Handle datetime conversion
        for field in ['created_at', 'updated_at']:
            if field in data and data[field]:
                if isinstance(data[field], str):
                    data[field] = datetime.fromisoformat(data[field].replace('Z', '+00:00'))
        
        chunk = DocumentChunk(
            id=data.get('id'),
            content=data.get('content', ''),
            document_id=data.get('document_id', ''),
            chunk_index=data.get('chunk_index', 0),
            token_count=data.get('token_count', 0),
            character_count=data.get('character_count', 0),
            start_position=data.get('start_position', 0),
            end_position=data.get('end_position', 0),
            language=data.get('language'),
            content_type=data.get('content_type', 'text/plain'),
            processing_metadata=data.get('processing_metadata', {})
        )
        
        # Restore metadata if available
        if 'created_at' in data:
            chunk.metadata.created_at = data['created_at']
        if 'updated_at' in data:
            chunk.metadata.updated_at = data['updated_at']
        if 'version' in data:
            chunk.metadata.version = data['version']
        
        return chunk


class InMemoryDocumentRepository(DocumentRepositoryInterface):
    """
    In-memory implementation for testing and development.
    
    Provides simple in-memory document storage for testing
    and development environments.
    """
    
    def __init__(self, config: GraphBuilderConfig):
        self.config = config
        self.documents: Dict[str, SourceDocument] = {}
        self.chunks: Dict[str, List[DocumentChunk]] = {}
        self.logger = logging.getLogger(self.__class__.__name__)
    
    async def save(self, document: SourceDocument) -> SourceDocument:
        """Save document to memory."""
        self.documents[document.id] = document
        self.logger.debug(f"Saved document to memory: {document.id}")
        return document
    
    async def get_by_id(self, document_id: str) -> Optional[SourceDocument]:
        """Get document by ID from memory."""
        return self.documents.get(document_id)
    
    async def update(self, document: SourceDocument) -> SourceDocument:
        """Update document in memory."""
        document.metadata.update()
        self.documents[document.id] = document
        return document
    
    async def delete(self, document_id: str) -> bool:
        """Delete document from memory."""
        if document_id in self.documents:
            del self.documents[document_id]
            if document_id in self.chunks:
                del self.chunks[document_id]
            return True
        return False
    
    async def find_by_status(self, status: ProcessingStatus) -> List[SourceDocument]:
        """Find documents by status in memory."""
        return [
            doc for doc in self.documents.values()
            if doc.processing_status == status
        ]
    
    async def save_chunk(self, chunk: DocumentChunk) -> DocumentChunk:
        """Save chunk to memory."""
        if chunk.document_id not in self.chunks:
            self.chunks[chunk.document_id] = []
        
        # Remove existing chunk with same ID
        self.chunks[chunk.document_id] = [
            c for c in self.chunks[chunk.document_id] if c.id != chunk.id
        ]
        
        self.chunks[chunk.document_id].append(chunk)
        self.logger.debug(f"Saved chunk to memory: {chunk.id}")
        return chunk
    
    async def get_chunks_by_document_id(self, document_id: str) -> List[DocumentChunk]:
        """Get chunks by document ID from memory."""
        chunks = self.chunks.get(document_id, [])
        return sorted(chunks, key=lambda c: c.chunk_index)


# Factory function for creating appropriate repository
def create_document_repository(config: GraphBuilderConfig, neo4j_driver=None) -> DocumentRepositoryInterface:
    """Create document repository based on configuration."""
    
    if config.database.provider == "neo4j" and neo4j_driver:
        return Neo4jDocumentRepository(config, neo4j_driver)
    else:
        return InMemoryDocumentRepository(config)