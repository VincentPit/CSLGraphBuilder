"""LLM service for generating graph documents."""

import os
from typing import List, Optional, Tuple, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from langchain.docstore.document import Document
from langchain_community.graphs import Neo4jGraph

from config import config
from exceptions import LLMError, ConfigurationError
from logger_config import logger
from graphTransformer import LLMGraphTransformer


class LLMService:
    """Service for LLM operations and graph document generation."""
    
    def __init__(self):
        self._llm = None
        self._graph_transformer = None
    
    def get_llm(self, model_version: Optional[str] = None):
        """Get configured LLM instance."""
        if self._llm is not None:
            return self._llm
        
        try:
            model_version = model_version or config.llm.model_name
            logger.info(f"Initializing LLM with model: {model_version}")
            
            if "azure" in model_version.lower():
                self._llm = self._create_azure_llm()
            else:
                self._llm = self._create_openai_llm()
            
            logger.info("LLM initialized successfully")
            return self._llm
            
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            raise LLMError(f"LLM initialization failed: {e}") from e
    
    def _create_azure_llm(self) -> AzureChatOpenAI:
        """Create Azure OpenAI LLM instance."""
        if not all([config.llm.model_name, config.llm.api_endpoint, config.llm.api_key, config.llm.api_version]):
            raise ConfigurationError("Azure LLM configuration incomplete")
        
        return AzureChatOpenAI(
            api_key=config.llm.api_key,
            azure_endpoint=config.llm.api_endpoint,
            azure_deployment=config.llm.model_name,
            api_version=config.llm.api_version,
            temperature=config.llm.temperature,
            max_tokens=config.llm.max_tokens,
        )
    
    def _create_openai_llm(self) -> ChatOpenAI:
        """Create OpenAI LLM instance."""
        if not config.llm.api_key:
            raise ConfigurationError("OpenAI API key not configured")
        
        return ChatOpenAI(
            openai_api_key=config.llm.api_key,
            model_name=config.llm.model_name,
            temperature=config.llm.temperature,
            max_tokens=config.llm.max_tokens,
        )
    
    def get_graph_transformer(self) -> LLMGraphTransformer:
        """Get LLM graph transformer."""
        if self._graph_transformer is None:
            llm = self.get_llm()
            self._graph_transformer = LLMGraphTransformer(llm=llm)
        return self._graph_transformer
    
    def generate_graph_documents(
        self,
        chunk_documents: List[Tuple[str, Document]],
        allowed_nodes: Optional[List[str]] = None,
        allowed_relationships: Optional[List[str]] = None,
        max_workers: int = 3
    ) -> List[Any]:
        """
        Generate graph documents from chunk documents.
        
        Args:
            chunk_documents: List of (chunk_id, document) tuples
            allowed_nodes: List of allowed node types
            allowed_relationships: List of allowed relationship types
            max_workers: Maximum number of worker threads
            
        Returns:
            List of generated graph documents
        """
        try:
            logger.info(f"Generating graph documents for {len(chunk_documents)} chunks")
            
            # Prepare allowed nodes and relationships
            allowed_nodes = allowed_nodes or []
            allowed_relationships = allowed_relationships or []
            
            logger.info(f"Allowed nodes: {allowed_nodes}")
            logger.info(f"Allowed relationships: {allowed_relationships}")
            
            # Get graph documents from LLM
            graph_documents = self._process_chunks_parallel(
                chunk_documents, allowed_nodes, allowed_relationships, max_workers
            )
            
            logger.info(f"Generated {len(graph_documents)} graph documents")
            return graph_documents
            
        except Exception as e:
            logger.error(f"Failed to generate graph documents: {e}")
            raise LLMError(f"Graph document generation failed: {e}") from e
    
    def _process_chunks_parallel(
        self,
        chunk_documents: List[Tuple[str, Document]],
        allowed_nodes: List[str],
        allowed_relationships: List[str],
        max_workers: int
    ) -> List[Any]:
        """Process chunks in parallel to generate graph documents."""
        graph_transformer = self.get_graph_transformer()
        
        # Configure allowed nodes and relationships
        if allowed_nodes:
            graph_transformer.allowed_nodes = allowed_nodes
        if allowed_relationships:
            graph_transformer.allowed_relationships = allowed_relationships
        
        graph_documents = []
        
        # Process chunks in batches
        batch_size = max(1, len(chunk_documents) // max_workers)
        batches = [
            chunk_documents[i:i + batch_size]
            for i in range(0, len(chunk_documents), batch_size)
        ]
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit batch processing tasks
            future_to_batch = {
                executor.submit(self._process_batch, graph_transformer, batch): batch
                for batch in batches
            }
            
            # Collect results
            for future in as_completed(future_to_batch):
                try:
                    batch_results = future.result()
                    graph_documents.extend(batch_results)
                except Exception as e:
                    batch = future_to_batch[future]
                    logger.error(f"Failed to process batch of {len(batch)} chunks: {e}")
                    # Continue processing other batches
        
        return graph_documents
    
    def _process_batch(
        self,
        graph_transformer: LLMGraphTransformer,
        batch: List[Tuple[str, Document]]
    ) -> List[Any]:
        """Process a batch of chunk documents."""
        try:
            documents = [doc for _, doc in batch]
            return graph_transformer.convert_to_graph_documents(documents)
        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            return []
    
    def parse_allowed_items(self, items_str: Optional[str]) -> List[str]:
        """Parse comma-separated string into list."""
        if not items_str or items_str.strip() == "":
            return []
        return [item.strip() for item in items_str.split(',') if item.strip()]


# Global LLM service instance
llm_service = LLMService()