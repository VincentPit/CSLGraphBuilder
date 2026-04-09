"""
GraphBuilder - Enterprise Knowledge Graph Construction Platform

A sophisticated, production-ready system for building knowledge graphs from web content
using state-of-the-art language models and graph databases.

Version: 2.0.0
Author: GraphBuilder Team
License: MIT
"""

__version__ = "2.0.0"
__title__ = "GraphBuilder"
__description__ = "Enterprise Knowledge Graph Construction Platform"
__author__ = "GraphBuilder Team"
__license__ = "MIT"

# Core exports
from graphbuilder.core.entities import *
from graphbuilder.application.use_cases import *
from graphbuilder.infrastructure.config import GraphBuilderConfig

__all__ = [
    "GraphBuilderConfig",
    "__version__",
    "__title__",
    "__description__",
    "__author__",
    "__license__",
]