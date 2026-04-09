"""Chunk entity model."""

from dataclasses import dataclass
from typing import Optional, Dict, Any
import hashlib


@dataclass
class ChunkDocument:
    """
    Document chunk entity.
    
    Attributes:
        content: The text content of the chunk
        chunk_id: Unique identifier for the chunk
        position: Position of the chunk in the original document
        length: Length of the chunk content
        file_name: Name of the source file
        content_offset: Offset position in the original content
        page_number: Page number (if applicable)
        start_time: Start time for audio/video content
        end_time: End time for audio/video content
        metadata: Additional metadata
    """
    
    content: str
    file_name: str
    position: int = 0
    length: int = 0
    content_offset: int = 0
    chunk_id: Optional[str] = None
    page_number: Optional[int] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        """Generate chunk ID if not provided."""
        if self.chunk_id is None:
            self.chunk_id = self._generate_chunk_id()
        
        if self.length == 0:
            self.length = len(self.content)
        
        if self.metadata is None:
            self.metadata = {}
    
    def _generate_chunk_id(self) -> str:
        """Generate a unique chunk ID based on content hash."""
        return hashlib.sha1(self.content.encode()).hexdigest()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database storage."""
        result = {
            "id": self.chunk_id,
            "pg_content": self.content,
            "position": self.position,
            "length": self.length,
            "f_name": self.file_name,
            "content_offset": self.content_offset
        }
        
        if self.page_number is not None:
            result["page_number"] = self.page_number
        
        if self.start_time is not None and self.end_time is not None:
            result["start_time"] = self.start_time
            result["end_time"] = self.end_time
        
        # Add any additional metadata
        if self.metadata:
            result.update(self.metadata)
        
        return result