"""Main application orchestrator for GraphBuilder."""

from typing import List, Optional, Dict, Any
from database.connection import db_manager
from database.repositories import SourceNodeRepository
from services.crawler_service import crawler_service
from services.document_service import document_processor
from services.llm_service import llm_service
from entities.source_node import SourceNode, SourceStatus
from config import config
from exceptions import GraphBuilderError
from logger_config import logger


class GraphBuilderApp:
    """Main application orchestrator."""
    
    def __init__(self):
        self.graph = None
        self.source_repo = None
        self._initialize()
    
    def _initialize(self):
        """Initialize the application components."""
        try:
            logger.info("Initializing GraphBuilder application")
            
            # Test database connection
            if not db_manager.test_connection():
                raise GraphBuilderError("Database connection failed")
            
            self.graph = db_manager.get_graph()
            self.source_repo = SourceNodeRepository(self.graph)
            
            logger.info("GraphBuilder application initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize application: {e}")
            raise GraphBuilderError(f"Application initialization failed: {e}") from e
    
    def process_single_url(
        self,
        url: str,
        allowed_nodes: Optional[List[str]] = None,
        allowed_relationships: Optional[List[str]] = None,
        model: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process a single URL and create knowledge graph.
        
        Args:
            url: URL to process
            allowed_nodes: List of allowed node types
            allowed_relationships: List of allowed relationship types  
            model: Model name to use
            
        Returns:
            Processing results dictionary
        """
        try:
            logger.info(f"Processing single URL: {url}")
            
            # Create source node
            source_node, success, success_count, fail_count = document_processor.create_source_node_from_url(
                url=url,
                keyword="dfrobot",  # Default keyword
                model=model
            )
            
            # Save source node to database
            self.source_repo.create(source_node)
            
            if not success:
                return {
                    "success": False,
                    "url": url,
                    "error": source_node.error_message,
                    "source_node_created": success_count,
                    "failures": fail_count
                }
            
            # Process URL to extract graph data
            graph_result = document_processor.process_url_to_graph(
                url=url,
                allowed_nodes=allowed_nodes,
                allowed_relationships=allowed_relationships
            )
            
            # Update source node status
            if graph_result["success"]:
                source_node.update_status(SourceStatus.COMPLETED)
            else:
                source_node.update_status(SourceStatus.FAILED, graph_result.get("error", "Unknown error"))
            
            self.source_repo.update(source_node)
            
            result = {
                "success": graph_result["success"],
                "url": url,
                "source_node_created": success_count,
                "failures": fail_count,
                "chunks_created": graph_result.get("chunks_created", 0),
                "graph_documents": graph_result.get("graph_documents", 0)
            }
            
            if not graph_result["success"]:
                result["error"] = graph_result.get("error")
            
            logger.info(f"Single URL processing completed: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to process single URL {url}: {e}")
            return {
                "success": False,
                "url": url,
                "error": str(e)
            }
    
    def crawl_and_process_urls(
        self,
        start_urls: List[str],
        allowed_nodes: Optional[List[str]] = None,
        allowed_relationships: Optional[List[str]] = None,
        model: Optional[str] = None,
        max_workers: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Crawl multiple URLs and process them in parallel.
        
        Args:
            start_urls: Initial URLs to start crawling
            allowed_nodes: List of allowed node types
            allowed_relationships: List of allowed relationship types
            model: Model name to use
            max_workers: Maximum number of worker threads
            
        Returns:
            Crawling and processing results
        """
        try:
            logger.info(f"Starting crawl and process for {len(start_urls)} URLs")
            
            def process_url_callback(url: str) -> bool:
                """Callback function for processing each crawled URL."""
                result = self.process_single_url(
                    url=url,
                    allowed_nodes=allowed_nodes,
                    allowed_relationships=allowed_relationships,
                    model=model
                )
                return result["success"]
            
            # Start parallel crawling
            crawl_stats = crawler_service.crawl_urls_parallel(
                start_urls=start_urls,
                process_callback=process_url_callback,
                max_workers=max_workers
            )
            
            # Get final statistics
            crawler_stats = crawler_service.get_statistics()
            
            result = {
                "success": True,
                "crawl_stats": crawl_stats,
                "crawler_stats": crawler_stats,
                "total_processed": crawler_stats["processed_count"],
                "crawl_limit": crawler_stats["crawl_limit"]
            }
            
            logger.info(f"Crawling and processing completed: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to crawl and process URLs: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def process_from_json_data(
        self,
        json_data: Dict[str, Any],
        allowed_nodes: Optional[List[str]] = None,
        allowed_relationships: Optional[List[str]] = None,
        model: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process data from JSON input.
        
        Args:
            json_data: JSON data to process
            allowed_nodes: List of allowed node types
            allowed_relationships: List of allowed relationship types
            model: Model name to use
            
        Returns:
            Processing results
        """
        try:
            logger.info("Processing JSON data")
            
            # Extract URLs from JSON data
            urls = []
            if "urls" in json_data:
                urls = json_data["urls"]
            elif "url" in json_data:
                urls = [json_data["url"]]
            
            if not urls:
                return {"success": False, "error": "No URLs found in JSON data"}
            
            # Process URLs
            if len(urls) == 1:
                return self.process_single_url(
                    url=urls[0],
                    allowed_nodes=allowed_nodes,
                    allowed_relationships=allowed_relationships,
                    model=model
                )
            else:
                return self.crawl_and_process_urls(
                    start_urls=urls,
                    allowed_nodes=allowed_nodes,
                    allowed_relationships=allowed_relationships,
                    model=model
                )
            
        except Exception as e:
            logger.error(f"Failed to process JSON data: {e}")
            return {"success": False, "error": str(e)}
    
    def get_processing_status(self, file_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get processing status for documents.
        
        Args:
            file_name: Optional specific file name to check
            
        Returns:
            Status information
        """
        try:
            if file_name:
                status = self.source_repo.get_current_status(file_name)
                return {"success": True, "file_name": file_name, "status": status}
            else:
                # Get all documents
                documents = self.source_repo.list_all()
                return {"success": True, "documents": documents, "count": len(documents)}
        
        except Exception as e:
            logger.error(f"Failed to get processing status: {e}")
            return {"success": False, "error": str(e)}
    
    def reset_crawler(self) -> Dict[str, Any]:
        """Reset crawler state."""
        try:
            crawler_service.reset()
            return {"success": True, "message": "Crawler state reset successfully"}
        except Exception as e:
            logger.error(f"Failed to reset crawler: {e}")
            return {"success": False, "error": str(e)}
    
    def shutdown(self):
        """Shutdown the application gracefully."""
        try:
            logger.info("Shutting down GraphBuilder application")
            db_manager.close()
            logger.info("Application shutdown completed")
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")


# Global application instance
app = GraphBuilderApp()