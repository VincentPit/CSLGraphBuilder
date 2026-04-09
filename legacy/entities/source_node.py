"""Source node entity model."""

from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


class SourceStatus(Enum):
    """Source node processing status."""
    NEW = "New"
    PROCESSING = "Processing" 
    COMPLETED = "Completed"
    FAILED = "Failed"
    CANCELLED = "Cancelled"


class SourceType(Enum):
    """Source node type."""
    FILE = "file"
    URL = "url"
    WIKIPEDIA = "wikipedia"
    YOUTUBE = "youtube"


@dataclass
class SourceNode:
    """
    Source node entity representing a document or URL being processed.
    
    Attributes:
        file_name: Name of the source file or URL identifier
        file_size: Size of the file in bytes
        file_type: Type of the file (pdf, html, etc.)
        file_source: Source type (file, url, etc.)
        status: Processing status
        url: URL of the source (if applicable)
        model: LLM model used for processing
        language: Language of the content
        created_at: Creation timestamp
        updated_at: Last update timestamp
        processing_time: Time taken for processing
        error_message: Error message if processing failed
        node_count: Number of nodes created
        relationship_count: Number of relationships created
        total_pages: Total number of pages
        total_chunks: Total number of chunks
        processed_chunk: Number of processed chunks
        is_cancelled: Whether processing was cancelled
        gcs_bucket: Google Cloud Storage bucket
        gcs_bucket_folder: GCS bucket folder
        gcs_project_id: GCS project ID
        aws_access_key_id: AWS access key ID
        access_token: Access token for authentication
    """
    
    file_name: str
    file_source: SourceType = SourceType.FILE
    status: SourceStatus = SourceStatus.NEW
    
    # Optional attributes with defaults
    file_size: Optional[int] = None
    file_type: Optional[str] = None
    url: Optional[str] = None
    model: Optional[str] = None
    language: str = "en"
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    # Processing metrics
    processing_time: float = 0.0
    node_count: int = 0
    relationship_count: int = 0
    total_pages: int = 0
    total_chunks: int = 0
    processed_chunk: int = 0
    
    # Status tracking
    error_message: str = ""
    is_cancelled: bool = False
    
    # Cloud storage
    gcs_bucket: Optional[str] = None
    gcs_bucket_folder: Optional[str] = None
    gcs_project_id: Optional[str] = None
    aws_access_key_id: Optional[str] = None
    access_token: Optional[str] = None
    
    def __post_init__(self):
        """Post-initialization validation."""
        if not self.file_name:
            raise ValueError("file_name is required")
        
        # Convert string enums to enum instances if needed
        if isinstance(self.status, str):
            self.status = SourceStatus(self.status)
        if isinstance(self.file_source, str):
            self.file_source = SourceType(self.file_source)
    
    def update_status(self, status: SourceStatus, error_message: str = "") -> None:
        """Update the status and timestamp."""
        self.status = status
        self.updated_at = datetime.now()
        if error_message:
            self.error_message = error_message
    
    def increment_processed_chunks(self) -> None:
        """Increment the number of processed chunks."""
        self.processed_chunk += 1
        self.updated_at = datetime.now()
    
    def set_processing_metrics(self, node_count: int, relationship_count: int) -> None:
        """Set processing metrics."""
        self.node_count = node_count
        self.relationship_count = relationship_count
        self.updated_at = datetime.now()
    
    def to_dict(self) -> dict:
        """Convert to dictionary for database storage."""
        return {
            "fileName": self.file_name,
            "fileSize": self.file_size,
            "fileType": self.file_type,
            "fileSource": self.file_source.value if isinstance(self.file_source, SourceType) else self.file_source,
            "status": self.status.value if isinstance(self.status, SourceStatus) else self.status,
            "url": self.url,
            "model": self.model,
            "language": self.language,
            "createdAt": self.created_at,
            "updatedAt": self.updated_at,
            "processingTime": self.processing_time,
            "nodeCount": self.node_count,
            "relationshipCount": self.relationship_count,
            "totalPages": self.total_pages,
            "totalChunks": self.total_chunks,
            "processedChunk": self.processed_chunk,
            "errorMessage": self.error_message,
            "isCancelled": self.is_cancelled,
            "gcsBucket": self.gcs_bucket,
            "gcsBucketFolder": self.gcs_bucket_folder,
            "gcsProjectId": self.gcs_project_id,
            "awsAccessKeyId": self.aws_access_key_id,
            "accessToken": self.access_token,
        }
