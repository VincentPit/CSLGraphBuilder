"""Command Line Interface for GraphBuilder."""

import sys
import argparse
from pathlib import Path
from typing import List, Optional

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from app import app
from config import config
from logger_config import setup_logging, logger
from exceptions import GraphBuilderError
from utils.helpers import save_json_data, format_timestamp
from utils.validators import URLValidator


def setup_cli_parser() -> argparse.ArgumentParser:
    """Setup command line argument parser."""
    parser = argparse.ArgumentParser(
        description="GraphBuilder - Knowledge Graph Builder for Web Content",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s url https://www.dfrobot.com
  %(prog)s crawl https://www.dfrobot.com --max-urls 10
  %(prog)s json data/sample.json
  %(prog)s status --file-name "https://www.dfrobot.com"
  %(prog)s reset-crawler
        """
    )
    
    # Global options
    parser.add_argument(
        '--log-level', 
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
        default='INFO',
        help='Set logging level'
    )
    parser.add_argument(
        '--output',
        help='Output file for results (JSON format)'
    )
    
    # Subcommands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # URL command
    url_parser = subparsers.add_parser('url', help='Process a single URL')
    url_parser.add_argument('url', help='URL to process')
    url_parser.add_argument('--allowed-nodes', help='Comma-separated list of allowed node types')
    url_parser.add_argument('--allowed-relationships', help='Comma-separated list of allowed relationship types')
    url_parser.add_argument('--model', help='LLM model to use')
    
    # Crawl command
    crawl_parser = subparsers.add_parser('crawl', help='Crawl and process multiple URLs')
    crawl_parser.add_argument('start_urls', nargs='+', help='Starting URLs for crawling')
    crawl_parser.add_argument('--max-urls', type=int, help='Maximum number of URLs to process')
    crawl_parser.add_argument('--max-workers', type=int, help='Maximum number of worker threads')
    crawl_parser.add_argument('--allowed-nodes', help='Comma-separated list of allowed node types')
    crawl_parser.add_argument('--allowed-relationships', help='Comma-separated list of allowed relationship types')
    crawl_parser.add_argument('--model', help='LLM model to use')
    
    # JSON command
    json_parser = subparsers.add_parser('json', help='Process data from JSON file')
    json_parser.add_argument('json_file', help='Path to JSON file')
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Get processing status')
    status_parser.add_argument('--file-name', help='Specific file name to check status')
    
    # Reset crawler command
    subparsers.add_parser('reset-crawler', help='Reset crawler state')
    
    # Config command
    config_parser = subparsers.add_parser('config', help='Show configuration')
    
    return parser


def validate_args(args) -> None:
    """Validate command line arguments."""
    if args.command == 'url':
        if not URLValidator.is_valid_url(args.url):
            raise ValueError(f"Invalid URL: {args.url}")
    
    elif args.command == 'crawl':
        for url in args.start_urls:
            if not URLValidator.is_valid_url(url):
                raise ValueError(f"Invalid URL: {url}")
    
    elif args.command == 'json':
        json_file = Path(args.json_file)
        if not json_file.exists():
            raise ValueError(f"JSON file not found: {args.json_file}")


def parse_list_arg(arg_value: Optional[str]) -> Optional[List[str]]:
    """Parse comma-separated argument into list."""
    if not arg_value:
        return None
    return [item.strip() for item in arg_value.split(',') if item.strip()]


def handle_url_command(args) -> dict:
    """Handle URL processing command."""
    logger.info(f"Processing single URL: {args.url}")
    
    return app.process_single_url(
        url=args.url,
        allowed_nodes=parse_list_arg(args.allowed_nodes),
        allowed_relationships=parse_list_arg(args.allowed_relationships),
        model=args.model
    )


def handle_crawl_command(args) -> dict:
    """Handle crawl command."""
    logger.info(f"Starting crawl with {len(args.start_urls)} URLs")
    
    # Update config if provided
    if args.max_urls:
        config.crawler.max_crawl_limit = args.max_urls
    if args.max_workers:
        config.crawler.max_workers = args.max_workers
    
    return app.crawl_and_process_urls(
        start_urls=args.start_urls,
        allowed_nodes=parse_list_arg(args.allowed_nodes),
        allowed_relationships=parse_list_arg(args.allowed_relationships),
        model=args.model,
        max_workers=args.max_workers
    )


def handle_json_command(args) -> dict:
    """Handle JSON processing command."""
    from utils.helpers import load_json_data
    
    logger.info(f"Processing JSON file: {args.json_file}")
    
    json_data = load_json_data(args.json_file)
    
    return app.process_from_json_data(
        json_data=json_data,
        allowed_nodes=json_data.get("allowed_nodes"),
        allowed_relationships=json_data.get("allowed_relationships"),
        model=json_data.get("model")
    )


def handle_status_command(args) -> dict:
    """Handle status command."""
    logger.info("Getting processing status")
    
    return app.get_processing_status(args.file_name)


def handle_reset_crawler_command(args) -> dict:
    """Handle reset crawler command."""
    logger.info("Resetting crawler state")
    
    return app.reset_crawler()


def handle_config_command(args) -> dict:
    """Handle config command."""
    logger.info("Showing configuration")
    
    return {
        "success": True,
        "config": {
            "database": {
                "uri": config.database.neo4j_uri,
                "user": config.database.neo4j_user,
                "database": config.database.neo4j_database
            },
            "llm": {
                "model_name": config.llm.model_name,
                "temperature": config.llm.temperature
            },
            "crawler": {
                "max_crawl_limit": config.crawler.max_crawl_limit,
                "max_workers": config.crawler.max_workers,
                "delay_between_requests": config.crawler.delay_between_requests,
                "allowed_domains": config.crawler.allowed_domains
            },
            "processing": {
                "chunk_size": config.processing.chunk_size,
                "chunk_overlap": config.processing.chunk_overlap,
                "max_chunks_allowed": config.processing.max_chunks_allowed
            }
        }
    }


def save_output(result: dict, output_file: str) -> None:
    """Save results to output file."""
    if output_file:
        try:
            # Add timestamp to result
            result["timestamp"] = format_timestamp()
            
            save_json_data(result, output_file)
            logger.info(f"Results saved to {output_file}")
        except Exception as e:
            logger.error(f"Failed to save results to {output_file}: {e}")


def main():
    """Main CLI function."""
    parser = setup_cli_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    try:
        # Setup logging
        setup_logging(log_level=args.log_level)
        logger.info(f"Starting GraphBuilder CLI - Command: {args.command}")
        
        # Validate arguments
        validate_args(args)
        
        # Execute command
        if args.command == 'url':
            result = handle_url_command(args)
        elif args.command == 'crawl':
            result = handle_crawl_command(args)
        elif args.command == 'json':
            result = handle_json_command(args)
        elif args.command == 'status':
            result = handle_status_command(args)
        elif args.command == 'reset-crawler':
            result = handle_reset_crawler_command(args)
        elif args.command == 'config':
            result = handle_config_command(args)
        else:
            logger.error(f"Unknown command: {args.command}")
            return 1
        
        # Output results
        if result.get("success"):
            logger.info("Command completed successfully!")
            if args.command != 'config':  # Config already shows detailed info
                logger.info(f"Results: {result}")
        else:
            logger.error(f"Command failed: {result.get('error', 'Unknown error')}")
        
        # Save output if requested
        save_output(result, args.output)
        
        return 0 if result.get("success") else 1
        
    except GraphBuilderError as e:
        logger.error(f"GraphBuilder error: {e}")
        return 1
    except ValueError as e:
        logger.error(f"Invalid arguments: {e}")
        return 1
    except KeyboardInterrupt:
        logger.info("Operation interrupted by user")
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


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)