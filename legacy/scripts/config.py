"""Configuration management for GraphBuilder."""

import os
from typing import Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path


@dataclass
class DatabaseConfig:
    """Database configuration."""
    neo4j_uri: str
    neo4j_user: str
    neo4j_password: str
    neo4j_database: str = "neo4j"


@dataclass
class LLMConfig:
    """LLM configuration."""
    model_name: str
    api_key: str
    api_endpoint: Optional[str] = None
    api_version: Optional[str] = None
    temperature: float = 0.0
    max_tokens: Optional[int] = None


@dataclass
class CrawlerConfig:
    """Web crawler configuration."""
    max_crawl_limit: int = 10
    max_workers: int = 5
    delay_between_requests: int = 1
    allowed_domains: list = None
    visited_urls_file: str = "record/visited_urls.json"
    processed_urls_file: str = "record/processed_urls.json"


@dataclass
class ProcessingConfig:
    """Document processing configuration."""
    chunk_size: int = 200
    chunk_overlap: int = 20
    max_chunks_allowed: int = 1000


class Config:
    """Main configuration class."""
    
    def __init__(self):
        self.database = self._load_database_config()
        self.llm = self._load_llm_config()
        self.crawler = self._load_crawler_config()
        self.processing = self._load_processing_config()
        self._ensure_directories()
    
    def _load_database_config(self) -> DatabaseConfig:
        """Load database configuration from environment variables."""
        return DatabaseConfig(
            neo4j_uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            neo4j_user=os.getenv("NEO4J_USER", "neo4j"),
            neo4j_password=os.getenv("NEO4J_PASSWORD", "password"),
            neo4j_database=os.getenv("NEO4J_DATABASE", "neo4j")
        )
    
    def _load_llm_config(self) -> LLMConfig:
        """Load LLM configuration from environment variables."""
        model_name = os.getenv("LLM_MODEL_NAME", "azure_ai_gpt_4o")
        env_key = f"LLM_MODEL_CONFIG_{model_name}"
        env_value = os.getenv(env_key, "")
        
        if "azure" in model_name.lower() and env_value:
            parts = env_value.split(",")
            if len(parts) >= 4:
                return LLMConfig(
                    model_name=parts[0],
                    api_endpoint=parts[1],
                    api_key=parts[2],
                    api_version=parts[3]
                )
        
        return LLMConfig(
            model_name=model_name,
            api_key=os.getenv("OPENAI_API_KEY", ""),
            api_endpoint=os.getenv("OPENAI_API_ENDPOINT"),
            api_version=os.getenv("OPENAI_API_VERSION")
        )
    
    def _load_crawler_config(self) -> CrawlerConfig:
        """Load crawler configuration."""
        return CrawlerConfig(
            max_crawl_limit=int(os.getenv("MAX_CRAWL_LIMIT", "10")),
            max_workers=int(os.getenv("MAX_WORKERS", "5")),
            delay_between_requests=int(os.getenv("CRAWL_DELAY", "1")),
            allowed_domains=os.getenv("ALLOWED_DOMAINS", "dfrobot").split(","),
            visited_urls_file=os.getenv("VISITED_URLS_FILE", "record/visited_urls.json"),
            processed_urls_file=os.getenv("PROCESSED_URLS_FILE", "record/processed_urls.json")
        )
    
    def _load_processing_config(self) -> ProcessingConfig:
        """Load processing configuration."""
        return ProcessingConfig(
            chunk_size=int(os.getenv("CHUNK_SIZE", "200")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "20")),
            max_chunks_allowed=int(os.getenv("MAX_CHUNKS_ALLOWED", "1000"))
        )
    
    def _ensure_directories(self):
        """Ensure required directories exist."""
        directories = [
            "record",
            "logs",
            Path(self.crawler.visited_urls_file).parent,
            Path(self.crawler.processed_urls_file).parent
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)


# Global configuration instance
config = Config()