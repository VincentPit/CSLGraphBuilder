"""Document processing service for chunking and graph generation."""

import hashlib
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
from langchain_text_splitters import TokenTextSplitter
from langchain.docstore.document import Document
from langchain_community.document_loaders import WebBaseLoader

from config import config
from entities.chunk import ChunkDocument
from entities.source_node import SourceNode, SourceStatus
from exceptions import ProcessingError
from logger_config import logger
from services.llm_service import llm_service


class DocumentProcessor:
    """Service for document processing and chunking."""
    
    def __init__(self):
        self.text_splitter = TokenTextSplitter(
            chunk_size=config.processing.chunk_size,
            chunk_overlap=config.processing.chunk_overlap
        )
    
    def split_documents_into_chunks(self, pages: List[Document]) -> List[ChunkDocument]:
        """
        Split documents into chunks.
        
        Args:
            pages: List of documents to split
            
        Returns:
            List of chunk documents
        """
        try:
            logger.info(f"Splitting {len(pages)} documents into chunks")
            
            chunks = []
            
            # Check if documents have page metadata
            has_page_metadata = pages and 'page' in pages[0].metadata
            
            if has_page_metadata:
                chunks = self._split_with_page_metadata(pages)
            else:
                chunks = self._split_without_page_metadata(pages)
            
            logger.info(f"Created {len(chunks)} chunks")
            return chunks
        
        except Exception as e:
            logger.error(f"Failed to split documents into chunks: {e}")
            raise ProcessingError(f"Document splitting failed: {e}") from e
    
    def _split_with_page_metadata(self, pages: List[Document]) -> List[ChunkDocument]:
        """Split documents that have page metadata."""
        chunks = []
        
        for i, document in enumerate(pages):
            page_number = i + 1
            
            for chunk_doc in self.text_splitter.split_documents([document]):
                chunk = ChunkDocument(
                    content=chunk_doc.page_content,
                    file_name="",  # Will be set later
                    page_number=page_number,
                    metadata=chunk_doc.metadata
                )
                chunks.append(chunk)
        
        return chunks
    
    def _split_without_page_metadata(self, pages: List[Document]) -> List[ChunkDocument]:
        """Split documents without page metadata."""
        chunks = []
        
        chunk_docs = self.text_splitter.split_documents(pages)
        
        for chunk_doc in chunk_docs:
            chunk = ChunkDocument(
                content=chunk_doc.page_content,
                file_name="",  # Will be set later
                metadata=chunk_doc.metadata
            )
            chunks.append(chunk)
        
        return chunks
    
    def create_chunk_relationships(
        self,
        file_name: str,
        chunks: List[ChunkDocument]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Create relationships between chunks and prepare data for database.
        
        Args:
            file_name: Name of the source file
            chunks: List of chunk documents
            
        Returns:
            Tuple of (batch_data, relationships)
        """
        try:
            logger.info(f"Creating chunk relationships for {len(chunks)} chunks")
            
            batch_data = []
            relationships = []
            offset = 0
            previous_chunk_id = ""
            
            for i, chunk in enumerate(chunks):
                # Update chunk metadata
                chunk.file_name = file_name
                chunk.position = i + 1
                chunk.content_offset = offset
                
                # Generate chunk ID if not set
                if not chunk.chunk_id:
                    chunk.chunk_id = self._generate_chunk_id(chunk.content)
                
                # Prepare batch data
                chunk_data = chunk.to_dict()
                chunk_data["previous_id"] = previous_chunk_id
                batch_data.append(chunk_data)
                
                # Create relationships
                if i == 0:
                    relationships.append({
                        "type": "FIRST_CHUNK",
                        "chunk_id": chunk.chunk_id
                    })
                else:
                    relationships.append({
                        "type": "NEXT_CHUNK",
                        "from_chunk_id": previous_chunk_id,
                        "to_chunk_id": chunk.chunk_id
                    })
                
                # Update for next iteration
                previous_chunk_id = chunk.chunk_id
                offset += len(chunk.content)
            
            logger.info(f"Created {len(relationships)} chunk relationships")
            return batch_data, relationships
        
        except Exception as e:
            logger.error(f"Failed to create chunk relationships: {e}")
            raise ProcessingError(f"Chunk relationship creation failed: {e}") from e
    
    def _generate_chunk_id(self, content: str) -> str:
        """Generate unique chunk ID from content."""
        return hashlib.sha1(content.encode()).hexdigest()
    
    def load_web_content(self, url: str) -> List[Document]:
        """
        Load content from a web URL.
        
        Args:
            url: URL to load content from
            
        Returns:
            List of documents
        """
        try:
            logger.info(f"Loading web content from: {url}")
            
            loader = WebBaseLoader(url)
            documents = loader.load()
            
            logger.info(f"Loaded {len(documents)} documents from {url}")
            return documents
        
        except Exception as e:
            logger.error(f"Failed to load web content from {url}: {e}")
            raise ProcessingError(f"Web content loading failed: {e}") from e
    
    def process_url_to_graph(
        self,
        url: str,
        allowed_nodes: Optional[List[str]] = None,
        allowed_relationships: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Process URL and extract graph data.
        
        Args:
            url: URL to process
            allowed_nodes: Allowed node types
            allowed_relationships: Allowed relationship types
            
        Returns:
            Dictionary with processing results
        """
        try:
            logger.info(f"Processing URL to graph: {url}")
            
            # Load web content
            documents = self.load_web_content(url)
            
            if not documents:
                return {"success": False, "error": "No content found"}
            
            # Split into chunks
            chunks = self.split_documents_into_chunks(documents)
            
            if not chunks:
                return {"success": False, "error": "No chunks created"}
            
            # Create chunk documents for LLM processing
            chunk_documents = [
                (chunk.chunk_id, Document(page_content=chunk.content, metadata=chunk.metadata))
                for chunk in chunks
            ]
            
            # Generate graph documents using LLM
            graph_documents = llm_service.generate_graph_documents(
                chunk_documents=chunk_documents,
                allowed_nodes=allowed_nodes,
                allowed_relationships=allowed_relationships
            )
            
            result = {
                "success": True,
                "url": url,
                "chunks_created": len(chunks),
                "graph_documents": len(graph_documents),
                "processed_at": datetime.now().isoformat()
            }
            
            logger.info(f"Successfully processed URL: {url}, Result: {result}")
            return result
        
        except Exception as e:
            logger.error(f"Failed to process URL {url}: {e}")
            return {
                "success": False,
                "url": url,
                "error": str(e),
                "processed_at": datetime.now().isoformat()
            }
    
    def create_source_node_from_url(
        self,
        url: str,
        keyword: str,
        model: str = None
    ) -> Tuple[SourceNode, bool, int, int]:
        """
        Create source node from URL.
        
        Args:
            url: URL to process
            keyword: Keyword for filtering
            model: Model name to use
            
        Returns:
            Tuple of (source_node, success_list, success_count, fail_count)
        """
        try:
            logger.info(f"Creating source node from URL: {url}")
            
            # Create source node
            source_node = SourceNode(
                file_name=url,
                file_source="url",
                url=url,
                model=model or config.llm.model_name,
                status=SourceStatus.NEW
            )
            
            # Process the URL
            documents = self.load_web_content(url)
            
            if documents:
                source_node.file_size = sum(len(doc.page_content) for doc in documents)
                source_node.total_pages = len(documents)
                source_node.update_status(SourceStatus.PROCESSING)
                
                success_count = 1
                fail_count = 0
            else:
                source_node.update_status(SourceStatus.FAILED, "No content found")
                success_count = 0
                fail_count = 1
            
            logger.info(f"Created source node for {url}: Success: {success_count}, Failures: {fail_count}")
            
            return source_node, success_count > 0, success_count, fail_count
        
        except Exception as e:
            logger.error(f"Failed to create source node from URL {url}: {e}")
            
            # Create failed source node
            source_node = SourceNode(
                file_name=url,
                file_source="url",
                url=url,
                model=model or config.llm.model_name,
                status=SourceStatus.FAILED,
                error_message=str(e)
            )
            
            return source_node, False, 0, 1


# Global document processor instance
document_processor = DocumentProcessor()