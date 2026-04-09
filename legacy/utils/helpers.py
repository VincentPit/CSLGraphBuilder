"""Common helper functions for GraphBuilder."""

import hashlib
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from datetime import datetime

from logger_config import logger


def generate_hash(text: str, algorithm: str = 'sha1') -> str:
    """
    Generate hash for text content.
    
    Args:
        text: Text to hash
        algorithm: Hash algorithm (sha1, md5, sha256)
        
    Returns:
        Hex digest of the hash
    """
    if algorithm == 'md5':
        return hashlib.md5(text.encode()).hexdigest()
    elif algorithm == 'sha256':
        return hashlib.sha256(text.encode()).hexdigest()
    else:  # default to sha1
        return hashlib.sha1(text.encode()).hexdigest()


def ensure_directory(directory: Union[str, Path]) -> Path:
    """
    Ensure directory exists, create if it doesn't.
    
    Args:
        directory: Directory path
        
    Returns:
        Path object of the directory
    """
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json_data(data: Dict[str, Any], file_path: Union[str, Path], indent: int = 2) -> None:
    """
    Save data to JSON file.
    
    Args:
        data: Data to save
        file_path: Path to save file
        indent: JSON indentation
    """
    try:
        file_path = Path(file_path)
        ensure_directory(file_path.parent)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=False, default=str)
        
        logger.debug(f"Saved JSON data to {file_path}")
    
    except Exception as e:
        logger.error(f"Failed to save JSON data to {file_path}: {e}")
        raise


def load_json_data(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load data from JSON file.
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        Loaded data dictionary
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logger.debug(f"Loaded JSON data from {file_path}")
        return data
    
    except Exception as e:
        logger.error(f"Failed to load JSON data from {file_path}: {e}")
        raise


def format_timestamp(dt: Optional[datetime] = None, format_str: str = '%Y-%m-%d %H:%M:%S') -> str:
    """
    Format datetime to string.
    
    Args:
        dt: Datetime object (defaults to now)
        format_str: Format string
        
    Returns:
        Formatted datetime string
    """
    if dt is None:
        dt = datetime.now()
    
    return dt.strftime(format_str)


def parse_comma_separated(value: Optional[str]) -> List[str]:
    """
    Parse comma-separated string into list.
    
    Args:
        value: Comma-separated string
        
    Returns:
        List of trimmed strings
    """
    if not value or value.strip() == "":
        return []
    
    return [item.strip() for item in value.split(',') if item.strip()]


def truncate_string(text: str, max_length: int, suffix: str = '...') -> str:
    """
    Truncate string to maximum length.
    
    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: Suffix to add when truncating
        
    Returns:
        Truncated string
    """
    if len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)] + suffix


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human readable format.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted size string
    """
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    size = float(size_bytes)
    
    while size >= 1024.0 and i < len(size_names) - 1:
        size /= 1024.0
        i += 1
    
    return f"{size:.1f} {size_names[i]}"


def merge_dictionaries(*dicts: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge multiple dictionaries.
    
    Args:
        *dicts: Dictionaries to merge
        
    Returns:
        Merged dictionary
    """
    result = {}
    for d in dicts:
        if isinstance(d, dict):
            result.update(d)
    
    return result


def flatten_list(nested_list: List[List[Any]]) -> List[Any]:
    """
    Flatten a nested list.
    
    Args:
        nested_list: Nested list
        
    Returns:
        Flattened list
    """
    flattened = []
    for item in nested_list:
        if isinstance(item, list):
            flattened.extend(item)
        else:
            flattened.append(item)
    
    return flattened


def chunks(lst: List[Any], chunk_size: int) -> List[List[Any]]:
    """
    Split list into chunks.
    
    Args:
        lst: List to split
        chunk_size: Size of each chunk
        
    Returns:
        List of chunks
    """
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def filter_empty_values(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Filter out empty values from dictionary.
    
    Args:
        data: Dictionary to filter
        
    Returns:
        Filtered dictionary
    """
    return {k: v for k, v in data.items() if v is not None and v != ""}


def safe_get(data: Dict[str, Any], key: str, default: Any = None) -> Any:
    """
    Safely get value from dictionary.
    
    Args:
        data: Dictionary
        key: Key to get
        default: Default value if key not found
        
    Returns:
        Value or default
    """
    try:
        return data.get(key, default)
    except (AttributeError, TypeError):
        return default


def create_backup_filename(original_path: Union[str, Path], suffix: str = None) -> str:
    """
    Create backup filename with timestamp.
    
    Args:
        original_path: Original file path
        suffix: Optional suffix
        
    Returns:
        Backup filename
    """
    path = Path(original_path)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    if suffix:
        backup_name = f"{path.stem}_{suffix}_{timestamp}{path.suffix}"
    else:
        backup_name = f"{path.stem}_backup_{timestamp}{path.suffix}"
    
    return str(path.parent / backup_name)