"""
Improved JSON processing script.

This script shows how to process JSON data with the restructured GraphBuilder.
"""

import sys
import json
from typing import Dict, Any
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from app import app
from logger_config import setup_logging, logger
from exceptions import GraphBuilderError


def load_json_data(file_path: str) -> Dict[str, Any]:
    """Load JSON data from file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"Loaded JSON data from {file_path}")
        return data
    except Exception as e:
        logger.error(f"Failed to load JSON data from {file_path}: {e}")
        raise


def main():
    """Main function for JSON processing."""
    try:
        # Setup logging
        setup_logging(log_level="INFO")
        logger.info("Starting GraphBuilder JSON processing")
        
        # Sample JSON data (you can also load from file)
        sample_json = {
            "urls": [
                "https://www.dfrobot.com/product-1.html",
                "https://www.dfrobot.com/product-2.html"
            ],
            "allowed_nodes": ["Product", "Company", "Technology"],
            "allowed_relationships": ["MANUFACTURES", "DEVELOPS"],
            "model": "azure_ai_gpt_4o"
        }
        
        # Or load from file:
        # json_file = "data/sample_data.json"
        # if Path(json_file).exists():
        #     sample_json = load_json_data(json_file)
        
        logger.info(f"Processing JSON data with {len(sample_json.get('urls', []))} URLs")
        
        # Extract parameters
        allowed_nodes = sample_json.get("allowed_nodes")
        allowed_relationships = sample_json.get("allowed_relationships") 
        model = sample_json.get("model")
        
        # Process JSON data
        result = app.process_from_json_data(
            json_data=sample_json,
            allowed_nodes=allowed_nodes,
            allowed_relationships=allowed_relationships,
            model=model
        )
        
        # Log results
        if result["success"]:
            logger.info("JSON processing completed successfully!")
            logger.info(f"Results: {result}")
            
            # Save results to file
            output_file = "results/json_processing_results.json"
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"Results saved to {output_file}")
        else:
            logger.error(f"JSON processing failed: {result.get('error', 'Unknown error')}")
            return 1
        
        return 0
        
    except GraphBuilderError as e:
        logger.error(f"GraphBuilder error: {e}")
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