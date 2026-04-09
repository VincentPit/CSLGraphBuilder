"""
Processing Models - Sophisticated models for document and content processing.

This module contains advanced models for processing workflows, task management,
and content analysis with rich metadata and state tracking.
"""

from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
import uuid
from graphbuilder.domain.models.graph_models import DomainEntity, Metadata, ProcessingStatus


class TaskType(Enum):
    """Types of processing tasks."""
    URL_CRAWLING = "url_crawling"
    FILE_PROCESSING = "file_processing"
    CONTENT_EXTRACTION = "content_extraction"
    ENTITY_EXTRACTION = "entity_extraction"
    RELATIONSHIP_EXTRACTION = "relationship_extraction"
    GRAPH_CONSTRUCTION = "graph_construction"
    EMBEDDING_GENERATION = "embedding_generation"
    VALIDATION = "validation"
    CLEANUP = "cleanup"


class TaskPriority(Enum):
    """Task priorities for processing queue."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4
    CRITICAL = 5


class ContentType(Enum):
    """Content types for processing."""
    HTML = "text/html"
    PLAIN_TEXT = "text/plain"
    MARKDOWN = "text/markdown"
    JSON = "application/json"
    PDF = "application/pdf"
    XML = "application/xml"
    CSV = "text/csv"
    DOCX = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"


@dataclass
class ProcessingResult:
    """Rich result object for processing operations."""
    
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metrics: Dict[str, Union[int, float, str]] = field(default_factory=dict)
    processing_time: Optional[float] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def add_error(self, error: str) -> None:
        """Add an error message."""
        self.errors.append(error)
        self.success = False
    
    def add_warning(self, warning: str) -> None:
        """Add a warning message."""
        self.warnings.append(warning)
    
    def add_metric(self, key: str, value: Union[int, float, str]) -> None:
        """Add a processing metric."""
        self.metrics[key] = value
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "success": self.success,
            "message": self.message,
            "data": self.data,
            "errors": self.errors,
            "warnings": self.warnings,
            "metrics": self.metrics,
            "processing_time": self.processing_time,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class ProcessingTask(DomainEntity):
    """
    Sophisticated processing task with rich state management.
    
    Represents individual processing tasks with advanced features for
    dependency management, retry logic, and progress tracking.
    """
    
    task_type: TaskType
    name: str
    description: Optional[str] = None
    priority: TaskPriority = TaskPriority.NORMAL
    status: ProcessingStatus = ProcessingStatus.PENDING
    
    # Task configuration
    configuration: Dict[str, Any] = field(default_factory=dict)
    input_data: Optional[Dict[str, Any]] = None
    output_data: Optional[Dict[str, Any]] = None
    
    # Execution metadata
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    processing_duration: Optional[float] = None
    retry_count: int = 0
    max_retries: int = 3
    error_message: Optional[str] = None
    
    # Dependencies and relationships
    depends_on: List[str] = field(default_factory=list)  # Task IDs
    blocks: List[str] = field(default_factory=list)      # Task IDs
    
    # Progress tracking
    progress_percentage: float = 0.0
    progress_message: Optional[str] = None
    
    def __post_init__(self):
        """Post-initialization setup."""
        if not hasattr(self, 'id'):
            super().__init__()
        self.validate()
    
    def validate(self) -> bool:
        """Validate task state."""
        if not self.name or not self.name.strip():
            raise ValueError("Task name cannot be empty")
        
        if not isinstance(self.task_type, TaskType):
            raise ValueError("Invalid task type")
        
        if not isinstance(self.priority, TaskPriority):
            raise ValueError("Invalid task priority")
        
        if not 0.0 <= self.progress_percentage <= 100.0:
            raise ValueError("Progress percentage must be between 0.0 and 100.0")
        
        if self.retry_count < 0:
            raise ValueError("Retry count must be non-negative")
        
        if self.max_retries < 0:
            raise ValueError("Max retries must be non-negative")
        
        return True
    
    def start_execution(self) -> None:
        """Start task execution."""
        self.status = ProcessingStatus.IN_PROGRESS
        self.start_time = datetime.now(timezone.utc)
        self.progress_percentage = 0.0
        self.metadata.update()
    
    def complete_execution(self, output_data: Optional[Dict[str, Any]] = None) -> None:
        """Complete task execution successfully."""
        self.status = ProcessingStatus.COMPLETED
        self.end_time = datetime.now(timezone.utc)
        self.progress_percentage = 100.0
        
        if self.start_time:
            self.processing_duration = (self.end_time - self.start_time).total_seconds()
        
        if output_data:
            self.output_data = output_data
        
        self.metadata.update()
    
    def fail_execution(self, error_message: str) -> None:
        """Fail task execution."""
        self.status = ProcessingStatus.FAILED
        self.end_time = datetime.now(timezone.utc)
        self.error_message = error_message
        
        if self.start_time:
            self.processing_duration = (self.end_time - self.start_time).total_seconds()
        
        self.metadata.update()
    
    def retry_execution(self) -> bool:
        """Attempt to retry task execution."""
        if self.retry_count >= self.max_retries:
            return False
        
        self.retry_count += 1
        self.status = ProcessingStatus.RETRY
        self.error_message = None
        self.start_time = None
        self.end_time = None
        self.processing_duration = None
        self.progress_percentage = 0.0
        self.metadata.update()
        
        return True
    
    def update_progress(self, percentage: float, message: Optional[str] = None) -> None:
        """Update task progress."""
        if not 0.0 <= percentage <= 100.0:
            raise ValueError("Progress percentage must be between 0.0 and 100.0")
        
        self.progress_percentage = percentage
        self.progress_message = message
        self.metadata.update()
    
    def add_dependency(self, task_id: str) -> None:
        """Add task dependency."""
        if task_id not in self.depends_on:
            self.depends_on.append(task_id)
            self.metadata.update()
    
    def add_blocking_task(self, task_id: str) -> None:
        """Add task that this task blocks."""
        if task_id not in self.blocks:
            self.blocks.append(task_id)
            self.metadata.update()
    
    def is_ready_to_execute(self, completed_tasks: set) -> bool:
        """Check if task is ready to execute based on dependencies."""
        return all(dep_id in completed_tasks for dep_id in self.depends_on)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "task_type": self.task_type.value,
            "name": self.name,
            "description": self.description,
            "priority": self.priority.value,
            "status": self.status.value,
            "configuration": self.configuration,
            "input_data": self.input_data,
            "output_data": self.output_data,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "processing_duration": self.processing_duration,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "error_message": self.error_message,
            "depends_on": self.depends_on,
            "blocks": self.blocks,
            "progress_percentage": self.progress_percentage,
            "progress_message": self.progress_message,
            "metadata": {
                "created_at": self.metadata.created_at.isoformat(),
                "updated_at": self.metadata.updated_at.isoformat(),
                "version": self.metadata.version,
                "tags": list(self.metadata.tags),
                "annotations": self.metadata.annotations
            }
        }


@dataclass
class ProcessingPipeline(DomainEntity):
    """
    Sophisticated processing pipeline with workflow orchestration.
    
    Represents complex processing workflows with advanced features for
    task orchestration, error handling, and pipeline optimization.
    """
    
    name: str
    description: Optional[str] = None
    tasks: List[ProcessingTask] = field(default_factory=list)
    configuration: Dict[str, Any] = field(default_factory=dict)
    
    # Pipeline state
    status: ProcessingStatus = ProcessingStatus.PENDING
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    total_duration: Optional[float] = None
    
    # Execution settings
    max_parallel_tasks: int = 5
    continue_on_error: bool = False
    auto_retry_failed: bool = True
    
    # Results tracking
    completed_tasks: set = field(default_factory=set)
    failed_tasks: set = field(default_factory=set)
    results: Dict[str, ProcessingResult] = field(default_factory=dict)
    
    def __post_init__(self):
        """Post-initialization setup."""
        if not hasattr(self, 'id'):
            super().__init__()
        self.validate()
    
    def validate(self) -> bool:
        """Validate pipeline state."""
        if not self.name or not self.name.strip():
            raise ValueError("Pipeline name cannot be empty")
        
        if self.max_parallel_tasks <= 0:
            raise ValueError("Max parallel tasks must be positive")
        
        # Validate task dependencies
        task_ids = {task.id for task in self.tasks}
        for task in self.tasks:
            for dep_id in task.depends_on:
                if dep_id not in task_ids:
                    raise ValueError(f"Task {task.id} depends on non-existent task {dep_id}")
        
        return True
    
    def add_task(self, task: ProcessingTask) -> None:
        """Add task to pipeline."""
        task.validate()
        self.tasks.append(task)
        self.metadata.update()
    
    def remove_task(self, task_id: str) -> bool:
        """Remove task from pipeline."""
        for i, task in enumerate(self.tasks):
            if task.id == task_id:
                # Remove dependencies to this task
                for other_task in self.tasks:
                    if task_id in other_task.depends_on:
                        other_task.depends_on.remove(task_id)
                    if task_id in other_task.blocks:
                        other_task.blocks.remove(task_id)
                
                del self.tasks[i]
                self.metadata.update()
                return True
        return False
    
    def get_ready_tasks(self) -> List[ProcessingTask]:
        """Get tasks ready for execution."""
        ready_tasks = []
        
        for task in self.tasks:
            if (task.status == ProcessingStatus.PENDING and 
                task.is_ready_to_execute(self.completed_tasks)):
                ready_tasks.append(task)
        
        # Sort by priority
        ready_tasks.sort(key=lambda t: t.priority.value, reverse=True)
        return ready_tasks
    
    def get_running_tasks(self) -> List[ProcessingTask]:
        """Get currently running tasks."""
        return [task for task in self.tasks if task.status == ProcessingStatus.IN_PROGRESS]
    
    def get_pipeline_progress(self) -> Dict[str, Any]:
        """Get overall pipeline progress."""
        total_tasks = len(self.tasks)
        if total_tasks == 0:
            return {"percentage": 0.0, "completed": 0, "total": 0, "status": "empty"}
        
        completed = len(self.completed_tasks)
        failed = len(self.failed_tasks)
        in_progress = len(self.get_running_tasks())
        pending = total_tasks - completed - failed - in_progress
        
        percentage = (completed / total_tasks) * 100
        
        return {
            "percentage": percentage,
            "completed": completed,
            "failed": failed,
            "in_progress": in_progress,
            "pending": pending,
            "total": total_tasks,
            "status": self.status.value
        }
    
    def start_pipeline(self) -> None:
        """Start pipeline execution."""
        self.status = ProcessingStatus.IN_PROGRESS
        self.start_time = datetime.now(timezone.utc)
        self.completed_tasks.clear()
        self.failed_tasks.clear()
        self.results.clear()
        self.metadata.update()
    
    def complete_pipeline(self) -> None:
        """Complete pipeline execution."""
        self.status = ProcessingStatus.COMPLETED
        self.end_time = datetime.now(timezone.utc)
        
        if self.start_time:
            self.total_duration = (self.end_time - self.start_time).total_seconds()
        
        self.metadata.update()
    
    def fail_pipeline(self, error_message: str) -> None:
        """Fail pipeline execution."""
        self.status = ProcessingStatus.FAILED
        self.end_time = datetime.now(timezone.utc)
        
        if self.start_time:
            self.total_duration = (self.end_time - self.start_time).total_seconds()
        
        # Add error to metadata
        self.metadata.add_annotation("error_message", error_message)
        self.metadata.update()
    
    def complete_task(self, task_id: str, result: ProcessingResult) -> None:
        """Mark task as completed with result."""
        self.completed_tasks.add(task_id)
        self.results[task_id] = result
        
        # Update task status
        for task in self.tasks:
            if task.id == task_id:
                task.complete_execution(result.data)
                break
        
        self.metadata.update()
    
    def fail_task(self, task_id: str, result: ProcessingResult) -> None:
        """Mark task as failed with result."""
        self.failed_tasks.add(task_id)
        self.results[task_id] = result
        
        # Update task status
        for task in self.tasks:
            if task.id == task_id:
                task.fail_execution(result.message)
                break
        
        self.metadata.update()
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """Get execution summary."""
        progress = self.get_pipeline_progress()
        
        success_rate = 0.0
        if progress["total"] > 0:
            success_rate = (progress["completed"] / progress["total"]) * 100
        
        return {
            "pipeline_id": self.id,
            "name": self.name,
            "status": self.status.value,
            "progress": progress,
            "success_rate": success_rate,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "total_duration": self.total_duration,
            "configuration": self.configuration,
            "results_summary": {
                "successful_tasks": len(self.completed_tasks),
                "failed_tasks": len(self.failed_tasks),
                "total_errors": sum(len(result.errors) for result in self.results.values()),
                "total_warnings": sum(len(result.warnings) for result in self.results.values())
            }
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "configuration": self.configuration,
            "status": self.status.value,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "total_duration": self.total_duration,
            "max_parallel_tasks": self.max_parallel_tasks,
            "continue_on_error": self.continue_on_error,
            "auto_retry_failed": self.auto_retry_failed,
            "tasks": [task.to_dict() for task in self.tasks],
            "progress": self.get_pipeline_progress(),
            "execution_summary": self.get_execution_summary(),
            "metadata": {
                "created_at": self.metadata.created_at.isoformat(),
                "updated_at": self.metadata.updated_at.isoformat(),
                "version": self.metadata.version,
                "tags": list(self.metadata.tags),
                "annotations": self.metadata.annotations
            }
        }


@dataclass 
class ExtractionRule:
    """
    Sophisticated extraction rule with pattern matching and validation.
    
    Represents configurable rules for entity and relationship extraction
    with advanced pattern matching and validation capabilities.
    """
    
    name: str
    description: Optional[str] = None
    entity_type: Optional[str] = None
    relationship_type: Optional[str] = None
    
    # Pattern matching
    patterns: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    regex_patterns: List[str] = field(default_factory=list)
    
    # Validation and filtering
    min_confidence: float = 0.5
    required_properties: List[str] = field(default_factory=list)
    excluded_patterns: List[str] = field(default_factory=list)
    
    # Context requirements
    context_window: int = 100  # Characters
    required_context: List[str] = field(default_factory=list)
    
    # Processing options
    case_sensitive: bool = False
    extract_properties: bool = True
    normalize_text: bool = True
    
    def matches_content(self, content: str, context: Optional[str] = None) -> bool:
        """Check if rule matches given content."""
        if not content:
            return False
        
        search_content = content if self.case_sensitive else content.lower()
        
        # Check patterns
        for pattern in self.patterns:
            search_pattern = pattern if self.case_sensitive else pattern.lower()
            if search_pattern in search_content:
                return True
        
        # Check keywords
        for keyword in self.keywords:
            search_keyword = keyword if self.case_sensitive else keyword.lower()
            if search_keyword in search_content:
                return True
        
        # Check regex patterns (would need re module in real implementation)
        # This is simplified for the example
        
        return False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "description": self.description,
            "entity_type": self.entity_type,
            "relationship_type": self.relationship_type,
            "patterns": self.patterns,
            "keywords": self.keywords,
            "regex_patterns": self.regex_patterns,
            "min_confidence": self.min_confidence,
            "required_properties": self.required_properties,
            "excluded_patterns": self.excluded_patterns,
            "context_window": self.context_window,
            "required_context": self.required_context,
            "case_sensitive": self.case_sensitive,
            "extract_properties": self.extract_properties,
            "normalize_text": self.normalize_text
        }