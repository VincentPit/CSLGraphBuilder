"""
Enterprise Configuration Management System

Provides a sophisticated, type-safe configuration system with validation,
environment variable support, and configuration profiles.
"""

import os
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum
import json
import yaml
from datetime import datetime


class Environment(Enum):
    """Application environment types."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"


class LogLevel(Enum):
    """Logging levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class ProcessingMode(Enum):
    """Processing execution modes."""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    BATCH = "batch"
    STREAM = "stream"


class LLMProvider(Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    AZURE_OPENAI = "azure_openai"
    ANTHROPIC = "anthropic"
    HUGGINGFACE = "huggingface"
    LOCAL = "local"


@dataclass
class DatabaseConfiguration:
    """Database configuration with connection pooling and optimization settings."""
    
    # Connection settings
    uri: str = field(default_factory=lambda: os.getenv("NEO4J_URI", "bolt://localhost:7687"))
    username: str = field(default_factory=lambda: os.getenv("NEO4J_USER", "neo4j"))
    password: str = field(default_factory=lambda: os.getenv("NEO4J_PASSWORD", ""))
    database_name: str = field(default_factory=lambda: os.getenv("NEO4J_DATABASE", "neo4j"))
    
    # Connection pooling
    max_connection_pool_size: int = field(default_factory=lambda: int(os.getenv("NEO4J_MAX_POOL_SIZE", "50")))
    connection_timeout: int = field(default_factory=lambda: int(os.getenv("NEO4J_CONNECTION_TIMEOUT", "30")))
    max_transaction_retry_time: int = field(default_factory=lambda: int(os.getenv("NEO4J_MAX_RETRY_TIME", "30")))
    
    # Performance tuning
    fetch_size: int = field(default_factory=lambda: int(os.getenv("NEO4J_FETCH_SIZE", "1000")))
    encrypted: bool = field(default_factory=lambda: os.getenv("NEO4J_ENCRYPTED", "false").lower() == "true")
    trust: str = field(default_factory=lambda: os.getenv("NEO4J_TRUST", "TRUST_ALL_CERTIFICATES"))


@dataclass
class LLMConfiguration:
    """Large Language Model configuration with advanced settings."""
    
    # Model settings
    provider: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER", "azure_openai"))
    model_name: str = field(default_factory=lambda: os.getenv("LLM_MODEL_NAME", "gpt-4o"))
    api_key: str = field(default_factory=lambda: os.getenv("LLM_API_KEY", ""))
    api_endpoint: Optional[str] = field(default_factory=lambda: os.getenv("LLM_API_ENDPOINT"))
    api_version: Optional[str] = field(default_factory=lambda: os.getenv("LLM_API_VERSION", "2024-02-01"))
    
    # Generation parameters
    temperature: float = field(default_factory=lambda: float(os.getenv("LLM_TEMPERATURE", "0.1")))
    max_tokens: Optional[int] = field(default_factory=lambda: int(os.getenv("LLM_MAX_TOKENS", "4096")) if os.getenv("LLM_MAX_TOKENS") else None)
    top_p: float = field(default_factory=lambda: float(os.getenv("LLM_TOP_P", "0.95")))
    frequency_penalty: float = field(default_factory=lambda: float(os.getenv("LLM_FREQUENCY_PENALTY", "0.0")))
    presence_penalty: float = field(default_factory=lambda: float(os.getenv("LLM_PRESENCE_PENALTY", "0.0")))
    
    # Rate limiting and retry
    max_retries: int = field(default_factory=lambda: int(os.getenv("LLM_MAX_RETRIES", "3")))
    retry_delay: float = field(default_factory=lambda: float(os.getenv("LLM_RETRY_DELAY", "1.0")))
    requests_per_minute: int = field(default_factory=lambda: int(os.getenv("LLM_REQUESTS_PER_MINUTE", "60")))
    
    # Fallback models
    fallback_models: List[str] = field(default_factory=lambda: os.getenv("LLM_FALLBACK_MODELS", "").split(",") if os.getenv("LLM_FALLBACK_MODELS") else [])


@dataclass
class CrawlerConfiguration:
    """Web crawler configuration with advanced crawling strategies."""
    
    # Basic settings
    max_urls: int = field(default_factory=lambda: int(os.getenv("CRAWLER_MAX_URLS", "100")))
    max_concurrent_workers: int = field(default_factory=lambda: int(os.getenv("CRAWLER_MAX_WORKERS", "10")))
    request_delay: float = field(default_factory=lambda: float(os.getenv("CRAWLER_REQUEST_DELAY", "1.0")))
    
    # Domain and URL filtering
    allowed_domains: List[str] = field(default_factory=lambda: os.getenv("CRAWLER_ALLOWED_DOMAINS", "").split(",") if os.getenv("CRAWLER_ALLOWED_DOMAINS") else [])
    blocked_domains: List[str] = field(default_factory=lambda: os.getenv("CRAWLER_BLOCKED_DOMAINS", "").split(",") if os.getenv("CRAWLER_BLOCKED_DOMAINS") else [])
    url_patterns: List[str] = field(default_factory=lambda: os.getenv("CRAWLER_URL_PATTERNS", "").split(",") if os.getenv("CRAWLER_URL_PATTERNS") else [])
    
    # HTTP settings
    timeout: int = field(default_factory=lambda: int(os.getenv("CRAWLER_TIMEOUT", "30")))
    max_redirects: int = field(default_factory=lambda: int(os.getenv("CRAWLER_MAX_REDIRECTS", "5")))
    user_agent: str = field(default_factory=lambda: os.getenv("CRAWLER_USER_AGENT", "GraphBuilder/2.0 (+https://github.com/VincentPit/GraphBuilder)"))
    
    # Content filtering
    max_file_size: int = field(default_factory=lambda: int(os.getenv("CRAWLER_MAX_FILE_SIZE", "10485760")))  # 10MB
    allowed_content_types: List[str] = field(default_factory=lambda: os.getenv("CRAWLER_ALLOWED_CONTENT_TYPES", "text/html,application/json").split(","))
    
    # Politeness and ethics
    respect_robots_txt: bool = field(default_factory=lambda: os.getenv("CRAWLER_RESPECT_ROBOTS", "true").lower() == "true")
    crawl_delay_from_robots: bool = field(default_factory=lambda: os.getenv("CRAWLER_USE_ROBOTS_DELAY", "true").lower() == "true")


@dataclass
class ProcessingConfiguration:
    """Document processing and NLP configuration."""
    
    # Text processing
    chunk_size: int = field(default_factory=lambda: int(os.getenv("PROCESSING_CHUNK_SIZE", "512")))
    chunk_overlap: int = field(default_factory=lambda: int(os.getenv("PROCESSING_CHUNK_OVERLAP", "50")))
    max_chunks_per_document: int = field(default_factory=lambda: int(os.getenv("PROCESSING_MAX_CHUNKS", "1000")))
    
    # Language detection and processing
    auto_detect_language: bool = field(default_factory=lambda: os.getenv("PROCESSING_AUTO_DETECT_LANG", "true").lower() == "true")
    supported_languages: List[str] = field(default_factory=lambda: os.getenv("PROCESSING_SUPPORTED_LANGS", "en,zh,fr,de,es,ja").split(","))
    
    # Graph extraction
    max_entities_per_chunk: int = field(default_factory=lambda: int(os.getenv("PROCESSING_MAX_ENTITIES", "50")))
    max_relationships_per_chunk: int = field(default_factory=lambda: int(os.getenv("PROCESSING_MAX_RELATIONSHIPS", "50")))
    entity_confidence_threshold: float = field(default_factory=lambda: float(os.getenv("PROCESSING_ENTITY_CONFIDENCE", "0.7")))
    relationship_confidence_threshold: float = field(default_factory=lambda: float(os.getenv("PROCESSING_REL_CONFIDENCE", "0.6")))
    
    # Performance
    processing_mode: ProcessingMode = field(default_factory=lambda: ProcessingMode(os.getenv("PROCESSING_MODE", "batch")))
    batch_size: int = field(default_factory=lambda: int(os.getenv("PROCESSING_BATCH_SIZE", "10")))
    parallel_workers: int = field(default_factory=lambda: int(os.getenv("PROCESSING_PARALLEL_WORKERS", "4")))


@dataclass
class LoggingConfiguration:
    """Advanced logging configuration with multiple handlers and formatters."""
    
    # Basic settings
    level: LogLevel = field(default_factory=lambda: LogLevel(os.getenv("LOG_LEVEL", "INFO")))
    enable_file_logging: bool = field(default_factory=lambda: os.getenv("LOG_ENABLE_FILE", "true").lower() == "true")
    enable_console_logging: bool = field(default_factory=lambda: os.getenv("LOG_ENABLE_CONSOLE", "true").lower() == "true")
    
    # File logging
    log_directory: str = field(default_factory=lambda: os.getenv("LOG_DIRECTORY", "logs"))
    log_filename: str = field(default_factory=lambda: os.getenv("LOG_FILENAME", "graphbuilder.log"))
    max_file_size: int = field(default_factory=lambda: int(os.getenv("LOG_MAX_FILE_SIZE", "52428800")))  # 50MB
    backup_count: int = field(default_factory=lambda: int(os.getenv("LOG_BACKUP_COUNT", "10")))
    
    # Formatting
    log_format: str = field(default_factory=lambda: os.getenv("LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    date_format: str = field(default_factory=lambda: os.getenv("LOG_DATE_FORMAT", "%Y-%m-%d %H:%M:%S"))
    
    # Advanced features
    enable_structured_logging: bool = field(default_factory=lambda: os.getenv("LOG_STRUCTURED", "false").lower() == "true")
    enable_performance_logging: bool = field(default_factory=lambda: os.getenv("LOG_PERFORMANCE", "true").lower() == "true")
    enable_audit_logging: bool = field(default_factory=lambda: os.getenv("LOG_AUDIT", "true").lower() == "true")


@dataclass
class SecurityConfiguration:
    """Security and privacy configuration."""
    
    # API Security
    enable_api_key_validation: bool = field(default_factory=lambda: os.getenv("SECURITY_API_KEY_VALIDATION", "true").lower() == "true")
    api_key_header: str = field(default_factory=lambda: os.getenv("SECURITY_API_KEY_HEADER", "X-API-Key"))
    
    # Data privacy
    enable_data_anonymization: bool = field(default_factory=lambda: os.getenv("SECURITY_ANONYMIZATION", "false").lower() == "true")
    pii_detection_enabled: bool = field(default_factory=lambda: os.getenv("SECURITY_PII_DETECTION", "true").lower() == "true")
    
    # Rate limiting
    enable_rate_limiting: bool = field(default_factory=lambda: os.getenv("SECURITY_RATE_LIMITING", "true").lower() == "true")
    max_requests_per_hour: int = field(default_factory=lambda: int(os.getenv("SECURITY_MAX_REQUESTS_HOUR", "1000")))
    
    # Content filtering
    content_filtering_enabled: bool = field(default_factory=lambda: os.getenv("SECURITY_CONTENT_FILTERING", "true").lower() == "true")
    blocked_content_patterns: List[str] = field(default_factory=lambda: os.getenv("SECURITY_BLOCKED_PATTERNS", "").split(",") if os.getenv("SECURITY_BLOCKED_PATTERNS") else [])


@dataclass
class MonitoringConfiguration:
    """Monitoring and observability configuration."""
    
    # Metrics
    enable_metrics: bool = field(default_factory=lambda: os.getenv("MONITORING_METRICS", "true").lower() == "true")
    metrics_port: int = field(default_factory=lambda: int(os.getenv("MONITORING_METRICS_PORT", "9090")))
    
    # Health checks
    enable_health_checks: bool = field(default_factory=lambda: os.getenv("MONITORING_HEALTH_CHECKS", "true").lower() == "true")
    health_check_interval: int = field(default_factory=lambda: int(os.getenv("MONITORING_HEALTH_INTERVAL", "30")))
    
    # Alerting
    enable_alerting: bool = field(default_factory=lambda: os.getenv("MONITORING_ALERTING", "false").lower() == "true")
    alert_webhook_url: Optional[str] = field(default_factory=lambda: os.getenv("MONITORING_ALERT_WEBHOOK"))
    
    # Performance tracking
    enable_performance_tracking: bool = field(default_factory=lambda: os.getenv("MONITORING_PERFORMANCE", "true").lower() == "true")
    performance_sample_rate: float = field(default_factory=lambda: float(os.getenv("MONITORING_SAMPLE_RATE", "0.1")))


class GraphBuilderConfig:
    """
    Enterprise-grade configuration management system.
    
    Provides centralized configuration with environment variable support,
    validation, profiles, and runtime updates.
    """
    
    def __init__(self, config_file: Optional[str] = None, environment: Optional[Environment] = None):
        """
        Initialize configuration.
        
        Args:
            config_file: Path to configuration file (JSON/YAML)
            environment: Application environment
        """
        self.environment = environment or Environment(os.getenv("APP_ENVIRONMENT", "development"))
        self.config_file = config_file
        self.loaded_at = datetime.now()
        
        # Load configuration
        self._load_configuration()
        
        # Initialize configuration sections
        self.database = DatabaseConfiguration()
        self.llm = LLMConfiguration()
        self.crawler = CrawlerConfiguration()
        self.processing = ProcessingConfiguration()
        self.logging = LoggingConfiguration()
        self.security = SecurityConfiguration()
        self.monitoring = MonitoringConfiguration()
        
        # Validate configuration
        self._validate_configuration()
        
        # Setup directories
        self._setup_directories()
    
    def _load_configuration(self) -> None:
        """Load configuration from file if provided."""
        if not self.config_file:
            return
        
        config_path = Path(self.config_file)
        if not config_path.exists():
            return
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                if config_path.suffix.lower() in ['.yml', '.yaml']:
                    config_data = yaml.safe_load(f)
                else:
                    config_data = json.load(f)
            
            # Update environment variables with file values
            self._update_env_from_config(config_data)
        
        except Exception as e:
            raise ValueError(f"Failed to load configuration file {config_path}: {e}")
    
    def _update_env_from_config(self, config_data: Dict[str, Any]) -> None:
        """Update environment variables from configuration data."""
        def flatten_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '_') -> Dict[str, str]:
            items = []
            for k, v in d.items():
                new_key = f"{parent_key}{sep}{k}" if parent_key else k
                if isinstance(v, dict):
                    items.extend(flatten_dict(v, new_key, sep=sep).items())
                else:
                    items.append((new_key.upper(), str(v)))
            return dict(items)
        
        flattened = flatten_dict(config_data)
        for key, value in flattened.items():
            if key not in os.environ:
                os.environ[key] = value
    
    def _validate_configuration(self) -> None:
        """Validate configuration values."""
        errors = []
        
        # Database validation
        if not self.database.uri:
            errors.append("Database URI is required")
        if not self.database.username:
            errors.append("Database username is required")
        
        # LLM validation
        if not self.llm.api_key:
            errors.append("LLM API key is required")
        if self.llm.temperature < 0 or self.llm.temperature > 2:
            errors.append("LLM temperature must be between 0 and 2")
        
        # Crawler validation
        if self.crawler.max_concurrent_workers <= 0:
            errors.append("Crawler max workers must be positive")
        if self.crawler.request_delay < 0:
            errors.append("Crawler request delay must be non-negative")
        
        if errors:
            raise ValueError(f"Configuration validation failed: {'; '.join(errors)}")
    
    def _setup_directories(self) -> None:
        """Create required directories."""
        directories = [
            self.logging.log_directory,
            "data/cache",
            "data/temp",
            "data/exports",
            "data/imports"
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get configuration summary."""
        return {
            "environment": self.environment.value,
            "loaded_at": self.loaded_at.isoformat(),
            "config_file": self.config_file,
            "database_uri": self.database.uri,
            "llm_provider": self.llm.provider,
            "llm_model": self.llm.model_name,
            "crawler_max_workers": self.crawler.max_concurrent_workers,
            "processing_mode": self.processing.processing_mode.value,
            "log_level": self.logging.level.value
        }
    
    def update_at_runtime(self, updates: Dict[str, Any]) -> None:
        """Update configuration at runtime."""
        for key, value in updates.items():
            if hasattr(self, key):
                section = getattr(self, key)
                if hasattr(section, '__dict__'):
                    for attr, val in value.items():
                        if hasattr(section, attr):
                            setattr(section, attr, val)
    
    def export_config(self, file_path: str) -> None:
        """Export current configuration to file."""
        config_data = {
            "environment": self.environment.value,
            "database": self.database.__dict__,
            "llm": self.llm.__dict__,
            "crawler": self.crawler.__dict__,
            "processing": self.processing.__dict__,
            "logging": self.logging.__dict__,
            "security": self.security.__dict__,
            "monitoring": self.monitoring.__dict__
        }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            if file_path.endswith(('.yml', '.yaml')):
                yaml.dump(config_data, f, default_flow_style=False)
            else:
                json.dump(config_data, f, indent=2, default=str)


# Global configuration instance
_config: Optional[GraphBuilderConfig] = None


def get_config(config_file: Optional[str] = None, environment: Optional[Environment] = None) -> GraphBuilderConfig:
    """Get global configuration instance."""
    global _config
    if _config is None:
        _config = GraphBuilderConfig(config_file=config_file, environment=environment)
    return _config


def reload_config(config_file: Optional[str] = None, environment: Optional[Environment] = None) -> GraphBuilderConfig:
    """Reload configuration."""
    global _config
    _config = GraphBuilderConfig(config_file=config_file, environment=environment)
    return _config