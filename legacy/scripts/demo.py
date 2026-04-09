"""
Demo script showing the improved GraphBuilder capabilities.

This script demonstrates the key improvements and new features
of the restructured GraphBuilder.
"""

import sys
import time
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from app import app
from config import config
from logger_config import setup_logging, logger
from services.crawler_service import crawler_service
from utils.helpers import save_json_data, format_timestamp
from utils.validators import URLValidator


def demo_configuration():
    """Demonstrate the configuration system."""
    print("\n" + "="*60)
    print("🔧 CONFIGURATION DEMO")
    print("="*60)
    
    print(f"Database URI: {config.database.neo4j_uri}")
    print(f"LLM Model: {config.llm.model_name}")
    print(f"Max Crawl Limit: {config.crawler.max_crawl_limit}")
    print(f"Chunk Size: {config.processing.chunk_size}")
    print(f"Allowed Domains: {config.crawler.allowed_domains}")


def demo_validation():
    """Demonstrate the validation system."""
    print("\n" + "="*60)
    print("✅ VALIDATION DEMO")
    print("="*60)
    
    test_urls = [
        "https://www.dfrobot.com",
        "invalid-url",
        "https://en.wikipedia.org/wiki/Robot",
        "ftp://example.com"
    ]
    
    for url in test_urls:
        is_valid = URLValidator.is_valid_url(url)
        is_wiki = URLValidator.is_wikipedia_url(url)
        print(f"URL: {url}")
        print(f"  Valid: {is_valid}")
        print(f"  Wikipedia: {is_wiki}")


def demo_logging():
    """Demonstrate the logging system."""
    print("\n" + "="*60)
    print("📝 LOGGING DEMO")
    print("="*60)
    
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    
    print("Check logs/graphbuilder.log for file output")


def demo_crawler_service():
    """Demonstrate the crawler service."""
    print("\n" + "="*60)
    print("🕷️ CRAWLER SERVICE DEMO")
    print("="*60)
    
    stats = crawler_service.get_statistics()
    print(f"Crawler Statistics: {stats}")
    
    # Test URL validation
    test_url = "https://www.dfrobot.com"
    should_process = crawler_service.should_process_url(test_url)
    print(f"Should process {test_url}: {should_process}")


def demo_error_handling():
    """Demonstrate error handling."""
    print("\n" + "="*60)
    print("🚨 ERROR HANDLING DEMO")
    print("="*60)
    
    try:
        # This should fail gracefully
        result = app.process_single_url("invalid-url")
        print(f"Result for invalid URL: {result}")
    except Exception as e:
        logger.error(f"Caught exception: {e}")


def demo_json_processing():
    """Demonstrate JSON processing."""
    print("\n" + "="*60)
    print("📄 JSON PROCESSING DEMO")
    print("="*60)
    
    sample_data = {
        "urls": ["https://www.dfrobot.com"],
        "allowed_nodes": ["Product", "Company"],
        "allowed_relationships": ["MANUFACTURES"],
        "model": "azure_ai_gpt_4o"
    }
    
    # Save sample data
    sample_file = "demo_data.json"
    save_json_data(sample_data, sample_file)
    print(f"Created sample JSON file: {sample_file}")
    
    # Process the data
    result = app.process_from_json_data(sample_data)
    print(f"JSON processing result: {result}")


def main():
    """Main demo function."""
    print("🚀 GraphBuilder Improvements Demo")
    print("="*60)
    
    # Setup logging
    setup_logging(log_level="INFO")
    logger.info("Starting GraphBuilder demo")
    
    try:
        # Run demos
        demo_configuration()
        demo_validation()
        demo_logging()
        demo_crawler_service()
        demo_error_handling()
        demo_json_processing()
        
        print("\n" + "="*60)
        print("✨ DEMO COMPLETED")
        print("="*60)
        print("Key improvements demonstrated:")
        print("  ✅ Configuration management")
        print("  ✅ Input validation")
        print("  ✅ Structured logging")
        print("  ✅ Service architecture")
        print("  ✅ Error handling")
        print("  ✅ JSON processing")
        print("\nFor full functionality, ensure:")
        print("  1. Neo4j database is running")
        print("  2. LLM API credentials are configured")
        print("  3. Environment variables are set")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}", exc_info=True)
        return 1
    
    finally:
        app.shutdown()
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)