"""Repository pattern for source node operations."""

from typing import List, Optional, Dict, Any
from langchain_community.graphs import Neo4jGraph
from entities.source_node import SourceNode, SourceStatus
from logger_config import logger
from exceptions import DatabaseError


class SourceNodeRepository:
    """Repository for source node database operations."""
    
    def __init__(self, graph: Neo4jGraph):
        self.graph = graph
    
    def create(self, source_node: SourceNode) -> None:
        """Create a new source node in the database."""
        try:
            logger.info(f"Creating source node: {source_node.file_name}")
            
            query = """
            MERGE (d:Document {fileName: $fileName})
            SET d.fileSize = $fileSize,
                d.fileType = $fileType,
                d.status = $status,
                d.url = $url,
                d.awsAccessKeyId = $awsAccessKeyId,
                d.fileSource = $fileSource,
                d.createdAt = $createdAt,
                d.updatedAt = $updatedAt,
                d.processingTime = $processingTime,
                d.errorMessage = $errorMessage,
                d.nodeCount = $nodeCount,
                d.relationshipCount = $relationshipCount,
                d.model = $model,
                d.language = $language,
                d.is_cancelled = $isCancelled,
                d.total_chunks = $totalChunks,
                d.processed_chunk = $processedChunk
            """
            
            params = source_node.to_dict()
            self.graph.query(query, params)
            logger.info(f"Successfully created source node: {source_node.file_name}")
            
        except Exception as e:
            error_msg = f"Failed to create source node {source_node.file_name}: {e}"
            logger.error(error_msg)
            self.update_status(source_node.file_name, SourceStatus.FAILED, str(e))
            raise DatabaseError(error_msg) from e
    
    def update(self, source_node: SourceNode) -> None:
        """Update an existing source node."""
        try:
            logger.info(f"Updating source node: {source_node.file_name}")
            
            query = """
            MATCH (d:Document {fileName: $fileName})
            SET d.fileSize = $fileSize,
                d.fileType = $fileType,
                d.status = $status,
                d.url = $url,
                d.updatedAt = $updatedAt,
                d.processingTime = $processingTime,
                d.errorMessage = $errorMessage,
                d.nodeCount = $nodeCount,
                d.relationshipCount = $relationshipCount,
                d.model = $model,
                d.language = $language,
                d.is_cancelled = $isCancelled,
                d.total_chunks = $totalChunks,
                d.processed_chunk = $processedChunk
            """
            
            params = source_node.to_dict()
            self.graph.query(query, params)
            logger.info(f"Successfully updated source node: {source_node.file_name}")
            
        except Exception as e:
            error_msg = f"Failed to update source node {source_node.file_name}: {e}"
            logger.error(error_msg)
            raise DatabaseError(error_msg) from e
    
    def get_by_filename(self, file_name: str) -> Optional[Dict[str, Any]]:
        """Get source node by filename."""
        try:
            query = """
            MATCH (d:Document {fileName: $fileName})
            RETURN d
            """
            
            result = self.graph.query(query, {"fileName": file_name})
            return result[0]['d'] if result else None
            
        except Exception as e:
            logger.error(f"Failed to get source node {file_name}: {e}")
            raise DatabaseError(f"Failed to retrieve source node: {e}") from e
    
    def update_status(self, file_name: str, status: SourceStatus, error_message: str = "") -> None:
        """Update the status of a source node."""
        try:
            logger.info(f"Updating status for {file_name} to {status.value}")
            
            query = """
            MATCH (d:Document {fileName: $fileName})
            SET d.status = $status,
                d.errorMessage = $errorMessage,
                d.updatedAt = datetime()
            """
            
            params = {
                "fileName": file_name,
                "status": status.value,
                "errorMessage": error_message
            }
            
            self.graph.query(query, params)
            
        except Exception as e:
            logger.error(f"Failed to update status for {file_name}: {e}")
            raise DatabaseError(f"Failed to update status: {e}") from e
    
    def get_current_status(self, file_name: str) -> Dict[str, Any]:
        """Get current status of a document."""
        try:
            query = """
            MATCH (d:Document {fileName: $fileName})
            RETURN d.status as status, d.is_cancelled as is_cancelled, d.errorMessage as error_message
            """
            
            result = self.graph.query(query, {"fileName": file_name})
            if not result:
                raise DatabaseError(f"Document {file_name} not found")
            
            return result[0]
            
        except Exception as e:
            logger.error(f"Failed to get current status for {file_name}: {e}")
            raise DatabaseError(f"Failed to get current status: {e}") from e
    
    def list_all(self, status: Optional[SourceStatus] = None) -> List[Dict[str, Any]]:
        """List all source nodes, optionally filtered by status."""
        try:
            if status:
                query = """
                MATCH (d:Document {status: $status})
                RETURN d
                ORDER BY d.createdAt DESC
                """
                params = {"status": status.value}
            else:
                query = """
                MATCH (d:Document)
                RETURN d
                ORDER BY d.createdAt DESC
                """
                params = {}
            
            result = self.graph.query(query, params)
            return [row['d'] for row in result]
            
        except Exception as e:
            logger.error(f"Failed to list source nodes: {e}")
            raise DatabaseError(f"Failed to list source nodes: {e}") from e
    
    def delete(self, file_name: str) -> None:
        """Delete a source node and all related data."""
        try:
            logger.info(f"Deleting source node and related data: {file_name}")
            
            query = """
            MATCH (d:Document {fileName: $fileName})
            OPTIONAL MATCH (d)-[:HAS_CHUNK]->(c:Chunk)
            OPTIONAL MATCH (c)-[:HAS_ENTITY]->(e)
            DETACH DELETE d, c, e
            """
            
            self.graph.query(query, {"fileName": file_name})
            logger.info(f"Successfully deleted source node: {file_name}")
            
        except Exception as e:
            logger.error(f"Failed to delete source node {file_name}: {e}")
            raise DatabaseError(f"Failed to delete source node: {e}") from e