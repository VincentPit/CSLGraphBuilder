"""Custom exceptions for GraphBuilder."""


class GraphBuilderError(Exception):
    """Base exception class for GraphBuilder."""
    pass


class DatabaseError(GraphBuilderError):
    """Exception raised for database-related errors."""
    pass


class LLMError(GraphBuilderError):
    """Exception raised for LLM-related errors."""
    pass


class CrawlerError(GraphBuilderError):
    """Exception raised for web crawling errors."""
    pass


class ProcessingError(GraphBuilderError):
    """Exception raised for document processing errors."""
    pass


class ConfigurationError(GraphBuilderError):
    """Exception raised for configuration errors."""
    pass


class ValidationError(GraphBuilderError):
    """Exception raised for validation errors."""
    pass