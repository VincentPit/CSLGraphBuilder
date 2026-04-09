"""
Improved main script for URL processing.

This script demonstrates the restructured GraphBuilder with better error handling,
configuration management, and separation of concerns.
"""

import sys
from typing import List, Optional
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from app import app
from config import config
from logger_config import setup_logging, logger
from exceptions import GraphBuilderError


def main():
    """Main function for URL processing."""
    try:
        # Setup logging
        setup_logging(log_level="INFO")
        logger.info("Starting GraphBuilder URL processing")
        
        # Configuration
        start_urls = [
            "https://www.dfrobot.com",
            "https://www.dfrobot.com/blog"
        ]
        
        # Parse allowed nodes and relationships from environment or use defaults
        allowed_nodes = ["Product", "Company", "Technology", "Person", "Location"]
        allowed_relationships = ["MANUFACTURES", "DEVELOPS", "LOCATED_IN", "WORKS_FOR"]
        
        # Log configuration
        logger.info(f"Configuration loaded:")
        logger.info(f"  - Max crawl limit: {config.crawler.max_crawl_limit}")
        logger.info(f"  - Max workers: {config.crawler.max_workers}")
        logger.info(f"  - LLM model: {config.llm.model_name}")
        logger.info(f"  - Allowed nodes: {allowed_nodes}")
        logger.info(f"  - Allowed relationships: {allowed_relationships}")
        
        # Process URLs
        logger.info(f"Starting to process {len(start_urls)} URLs")
        
        if len(start_urls) == 1:
            # Process single URL
            result = app.process_single_url(
                url=start_urls[0],
                allowed_nodes=allowed_nodes,
                allowed_relationships=allowed_relationships
            )
        else:
            # Crawl and process multiple URLs
            result = app.crawl_and_process_urls(
                start_urls=start_urls,
                allowed_nodes=allowed_nodes,
                allowed_relationships=allowed_relationships
            )
        
        # Log results
        if result["success"]:
            logger.info("Processing completed successfully!")
            logger.info(f"Results: {result}")
        else:
            logger.error(f"Processing failed: {result.get('error', 'Unknown error')}")
            return 1
        
        # Get final status
        status = app.get_processing_status()
        logger.info(f"Final status: {status}")
        
        return 0
        
    except GraphBuilderError as e:
        logger.error(f"GraphBuilder error: {e}")
        return 1
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        return 1
    finally:
        # Clean shutdown
        try:
            app.shutdown()
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")


def process_custom_urls(urls: List[str], **kwargs):
    """
    Process custom URLs with optional parameters.
    
    Args:
        urls: List of URLs to process
        **kwargs: Additional parameters (allowed_nodes, allowed_relationships, model)
    """
    try:
        logger.info(f"Processing custom URLs: {urls}")
        
        if len(urls) == 1:
            result = app.process_single_url(url=urls[0], **kwargs)
        else:
            result = app.crawl_and_process_urls(start_urls=urls, **kwargs)
        
        return result
    
    except Exception as e:
        logger.error(f"Failed to process custom URLs: {e}")
        return {"success": False, "error": str(e)}


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)