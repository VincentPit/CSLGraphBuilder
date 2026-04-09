"""Validation utilities for GraphBuilder."""

import re
from typing import List, Optional, Dict, Any
from urllib.parse import urlparse
from pathlib import Path

from exceptions import ValidationError
from logger_config import logger


class URLValidator:
    """URL validation utilities."""
    
    WIKIPEDIA_REGEX = r'https?:\/\/(www\.)?([a-zA-Z]{2,3})\.wikipedia\.org\/wiki\/(.*)'
    YOUTUBE_REGEX = r'https?:\/\/(www\.)?(youtube\.com\/watch\?v=|youtu\.be\/)([a-zA-Z0-9_-]+)'
    
    @staticmethod
    def is_valid_url(url: str) -> bool:
        """Check if URL is valid."""
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except Exception:
            return False
    
    @staticmethod
    def is_wikipedia_url(url: str) -> bool:
        """Check if URL is a Wikipedia URL."""
        return bool(re.match(URLValidator.WIKIPEDIA_REGEX, url))
    
    @staticmethod
    def is_youtube_url(url: str) -> bool:
        """Check if URL is a YouTube URL."""
        return bool(re.match(URLValidator.YOUTUBE_REGEX, url))
    
    @staticmethod
    def extract_wikipedia_info(url: str) -> tuple[str, str]:
        """
        Extract language and article ID from Wikipedia URL.
        
        Returns:
            Tuple of (language, article_id)
        """
        match = re.search(URLValidator.WIKIPEDIA_REGEX, url.strip())
        if match:
            language = match.group(2)
            article_id = match.group(3)
            return language, article_id
        else:
            raise ValidationError(f'Not a valid Wikipedia URL: {url}')
    
    @staticmethod
    def validate_domain(url: str, allowed_domains: List[str]) -> bool:
        """Check if URL domain is in allowed list."""
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            return any(allowed_domain in domain for allowed_domain in allowed_domains)
        except Exception:
            return False


class FileValidator:
    """File validation utilities."""
    
    SUPPORTED_FORMATS = {
        '.pdf', '.txt', '.docx', '.doc', '.html', '.htm', '.md', '.json'
    }
    
    @staticmethod
    def is_supported_format(file_path: str) -> bool:
        """Check if file format is supported."""
        suffix = Path(file_path).suffix.lower()
        return suffix in FileValidator.SUPPORTED_FORMATS
    
    @staticmethod
    def validate_file_exists(file_path: str) -> bool:
        """Check if file exists."""
        return Path(file_path).exists()
    
    @staticmethod
    def get_file_size(file_path: str) -> int:
        """Get file size in bytes."""
        try:
            return Path(file_path).stat().st_size
        except Exception as e:
            logger.error(f"Failed to get file size for {file_path}: {e}")
            return 0


class DataValidator:
    """Data validation utilities."""
    
    @staticmethod
    def validate_config_data(data: Dict[str, Any]) -> bool:
        """Validate configuration data."""
        required_fields = ['database', 'llm']
        
        for field in required_fields:
            if field not in data:
                raise ValidationError(f"Missing required config field: {field}")
        
        return True
    
    @staticmethod
    def validate_chunk_data(chunk_data: Dict[str, Any]) -> bool:
        """Validate chunk data."""
        required_fields = ['content', 'chunk_id', 'file_name']
        
        for field in required_fields:
            if field not in chunk_data or not chunk_data[field]:
                raise ValidationError(f"Missing or empty required field: {field}")
        
        return True
    
    @staticmethod
    def validate_source_node_data(data: Dict[str, Any]) -> bool:
        """Validate source node data."""
        if not data.get('file_name'):
            raise ValidationError("file_name is required for source node")
        
        if not data.get('file_source'):
            raise ValidationError("file_source is required for source node")
        
        return True
    
    @staticmethod
    def sanitize_string(value: str, max_length: Optional[int] = None) -> str:
        """Sanitize string input."""
        if not isinstance(value, str):
            return str(value)
        
        # Remove null bytes and control characters
        sanitized = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x84\x86-\x9f]', '', value)
        
        # Trim whitespace
        sanitized = sanitized.strip()
        
        # Limit length if specified
        if max_length and len(sanitized) > max_length:
            sanitized = sanitized[:max_length].rstrip()
        
        return sanitized
    
    @staticmethod
    def validate_list_items(items: List[str], item_type: str = "item") -> List[str]:
        """Validate and clean list of string items."""
        validated_items = []
        
        for item in items:
            if isinstance(item, str) and item.strip():
                cleaned_item = DataValidator.sanitize_string(item)
                if cleaned_item:
                    validated_items.append(cleaned_item)
            else:
                logger.warning(f"Skipping invalid {item_type}: {item}")
        
        return validated_items


class ModelValidator:
    """Model and LLM validation utilities."""
    
    SUPPORTED_MODELS = {
        'azure_ai_gpt_4o', 'azure_ai_gpt_35', 'openai-gpt-4o', 'openai-gpt-3.5',
        'gemini-1.0-pro', 'gemini-1.5-pro', 'groq-llama3', '智谱', '百川',
        '月之暗面', '深度求索', '零一万物', '通义千问', '豆包', 'Ollama'
    }
    
    @staticmethod
    def is_supported_model(model_name: str) -> bool:
        """Check if model is supported."""
        return model_name in ModelValidator.SUPPORTED_MODELS
    
    @staticmethod
    def validate_model_config(model_name: str, config_data: Dict[str, Any]) -> bool:
        """Validate model configuration."""
        if not ModelValidator.is_supported_model(model_name):
            raise ValidationError(f"Unsupported model: {model_name}")
        
        if 'azure' in model_name.lower():
            required_fields = ['api_key', 'api_endpoint', 'api_version']
            for field in required_fields:
                if not config_data.get(field):
                    raise ValidationError(f"Missing Azure config field: {field}")
        
        elif 'openai' in model_name.lower():
            if not config_data.get('api_key'):
                raise ValidationError("Missing OpenAI API key")
        
        return True