"""
GraphBuilder CLI - Sophisticated command-line interface with enterprise features.

This module provides a comprehensive CLI with advanced commands, configuration
management, progress tracking, and user-friendly interactions.
"""

import asyncio
import click
import logging
import sys
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

from ..infrastructure.config.settings import GraphBuilderConfig, get_config
from ..infrastructure.repositories.document_repository import create_document_repository
from ..infrastructure.repositories.graph_repository import create_graph_repository
from ..infrastructure.services.llm_service import create_llm_service
from ..infrastructure.services.content_extractor import create_content_extractor_service
from ..application.use_cases.document_processing import (
    ProcessDocumentUseCase, BatchProcessDocumentsUseCase, OptimizeKnowledgeGraphUseCase
)
from ..application.use_cases.open_targets_ingestion import (
    OpenTargetsIngestionUseCase, IngestionConfig
)
from ..application.use_cases.pubmed_ingestion import (
    PubMedIngestionUseCase, PubMedIngestionConfig
)
from ..application.use_cases.curation import (
    CurationUseCase, CurationRequest, CurationAction
)
from ..application.use_cases.graph_visualization import (
    GraphVisualizationUseCase, VisualizationConfig
)
from ..application.use_cases.relationship_verification import (
    RelationshipVerificationUseCase, VerificationConfig
)
from ..core.verification.cascading import CascadingVerifierConfig
from ..core.verification.text_match import TextMatchConfig
from ..core.verification.embedding import EmbeddingConfig
from ..domain.models.graph_models import KnowledgeGraph, SourceDocument, ProcessingStatus
from ..domain.models.processing_models import TaskType, ProcessingPipeline


# Configure rich console for better output formatting
try:
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    console = Console()
    RICH_AVAILABLE = True
except ImportError:
    console = None
    RICH_AVAILABLE = False


class GraphBuilderCLI:
    """
    Sophisticated CLI application with enterprise-grade features.
    
    Provides comprehensive command-line interface with advanced configuration,
    logging, progress tracking, and user-friendly interactions.
    """
    
    def __init__(self):
        self.config: Optional[GraphBuilderConfig] = None
        self.logger = logging.getLogger(__name__)
    
    def setup_logging(self, verbose: bool = False, log_file: Optional[str] = None) -> None:
        """Setup sophisticated logging configuration."""
        
        level = logging.DEBUG if verbose else logging.INFO
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        
        # Root logger configuration
        root_logger = logging.getLogger()
        root_logger.setLevel(level)
        root_logger.addHandler(console_handler)
        
        # File handler if specified
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
    
    def load_configuration(self, config_path: Optional[str] = None) -> GraphBuilderConfig:
        """Load and validate configuration."""
        
        try:
            self.config = get_config(config_path)
            self.logger.info("Configuration loaded successfully")
            return self.config
        except Exception as e:
            self.logger.error(f"Failed to load configuration: {str(e)}")
            raise click.ClickException(f"Configuration error: {str(e)}")
    
    def print_status(self, message: str, status: str = "info") -> None:
        """Print status message with appropriate formatting."""
        
        if RICH_AVAILABLE:
            if status == "success":
                console.print(f"✅ {message}", style="green")
            elif status == "warning":
                console.print(f"⚠️  {message}", style="yellow")
            elif status == "error":
                console.print(f"❌ {message}", style="red")
            else:
                console.print(f"ℹ️  {message}", style="blue")
        else:
            click.echo(f"[{status.upper()}] {message}")
    
    def print_table(self, data: List[Dict[str, Any]], title: str = "Results") -> None:
        """Print data in a formatted table."""
        
        if not data:
            self.print_status("No data to display", "warning")
            return
        
        if RICH_AVAILABLE:
            table = Table(title=title)
            
            # Add columns based on first row
            for key in data[0].keys():
                table.add_column(key.replace('_', ' ').title())
            
            # Add rows
            for row in data:
                table.add_row(*[str(value) for value in row.values()])
            
            console.print(table)
        else:
            # Fallback to simple text output
            click.echo(f"\n{title}:")
            for item in data:
                click.echo("  " + " | ".join([f"{k}: {v}" for k, v in item.items()]))


# CLI Group
@click.group()
@click.option('--config', '-c', help='Configuration file path')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.option('--log-file', help='Log file path')
@click.pass_context
def cli(ctx, config: Optional[str], verbose: bool, log_file: Optional[str]):
    """
    GraphBuilder - Sophisticated Knowledge Graph Builder
    
    A comprehensive tool for building knowledge graphs from various content sources
    with enterprise-grade features and advanced AI-powered extraction capabilities.
    """
    ctx.ensure_object(dict)
    
    # Initialize CLI application
    app = GraphBuilderCLI()
    app.setup_logging(verbose, log_file)
    
    # Store configuration path for lazy loading in commands that need it
    ctx.obj['app'] = app
    ctx.obj['config_path'] = config
    ctx.obj['config_loaded'] = False


def ensure_config_loaded(ctx):
    """Ensure configuration is loaded when needed by a command."""
    if not ctx.obj['config_loaded']:
        try:
            ctx.obj['app'].load_configuration(ctx.obj['config_path'])
            ctx.obj['config'] = ctx.obj['app'].config
            ctx.obj['config_loaded'] = True
        except Exception as e:
            click.echo(f"Configuration error: {str(e)}", err=True)
            sys.exit(1)


@cli.command()
@click.option('--url', '-u', help='URL to process')
@click.option('--file', '-f', 'file_path', help='File path to process')
@click.option('--text', '-t', help='Direct text content to process')
@click.option('--title', help='Title for the document')
@click.option('--output', '-o', help='Output file for results')
@click.option('--extract-entities/--no-extract-entities', default=True, help='Extract entities')
@click.option('--extract-relationships/--no-extract-relationships', default=True, help='Extract relationships')
@click.pass_context
def process(ctx, url: Optional[str], file_path: Optional[str], text: Optional[str], 
           title: Optional[str], output: Optional[str], extract_entities: bool, 
           extract_relationships: bool):
    """
    Process a document and extract knowledge graph entities and relationships.
    
    You can process content from a URL, file, or direct text input. The system
    will automatically extract entities and relationships using advanced AI models.
    """
    
    # Load configuration when needed
    ensure_config_loaded(ctx)
    
    app = ctx.obj['app']
    config = ctx.obj['config']
    
    # Validate input
    if not any([url, file_path, text]):
        raise click.UsageError("Must specify one of --url, --file, or --text")
    
    if sum(bool(x) for x in [url, file_path, text]) > 1:
        raise click.UsageError("Can only specify one input source")
    
    async def process_document():
        """Process document asynchronously."""
        
        try:
            # Initialize services
            doc_repo = create_document_repository(config)
            graph_repo = create_graph_repository(config)
            llm_service = create_llm_service(config)
            content_extractor = create_content_extractor_service(config)
            
            # Create use case
            process_use_case = ProcessDocumentUseCase(
                config, doc_repo, graph_repo, llm_service, content_extractor
            )
            
            # Create document
            if url:
                document = SourceDocument(
                    title=title or f"Document from {url}",
                    source_url=url,
                    content_type="text/html"
                )
            elif file_path:
                document = SourceDocument(
                    title=title or Path(file_path).stem,
                    file_path=file_path,
                    content_type="text/plain"
                )
            else:  # text
                document = SourceDocument(
                    title=title or "Text Document",
                    content_length=len(text),
                    content_type="text/plain"
                )
            
            # Save document
            await doc_repo.save(document)
            
            app.print_status(f"Processing document: {document.title}")
            
            # Configure extraction
            extraction_config = {
                "extract_entities": extract_entities,
                "extract_relationships": extract_relationships,
                "temperature": 0.1,
                "max_tokens": 2000
            }
            
            # Process with progress tracking
            if RICH_AVAILABLE:
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    TimeElapsedColumn(),
                    console=console
                ) as progress:
                    task = progress.add_task("Processing document...", total=None)
                    result = await process_use_case.execute(document.id, extraction_config)
                    progress.update(task, completed=True)
            else:
                app.print_status("Processing document...", "info")
                result = await process_use_case.execute(document.id, extraction_config)
            
            # Display results
            if result.success:
                app.print_status("Document processed successfully!", "success")
                
                # Show metrics
                if result.data:
                    metrics = [
                        {"Metric": "Extracted Entities", "Value": result.data.get("extracted_entities", 0)},
                        {"Metric": "Extracted Relationships", "Value": result.data.get("extracted_relationships", 0)},
                        {"Metric": "Processing Time", "Value": f"{result.processing_time:.2f}s"}
                    ]
                    app.print_table(metrics, "Processing Results")
                
                # Save results if requested
                if output:
                    output_data = {
                        "document_id": document.id,
                        "title": document.title,
                        "processing_result": result.to_dict(),
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    with open(output, 'w') as f:
                        json.dump(output_data, f, indent=2)
                    
                    app.print_status(f"Results saved to {output}", "success")
                
            else:
                app.print_status(f"Processing failed: {result.message}", "error")
                if result.errors:
                    for error in result.errors:
                        app.print_status(f"Error: {error}", "error")
                
                return 1
            
            return 0
            
        except Exception as e:
            app.logger.error(f"Processing error: {str(e)}", exc_info=True)
            app.print_status(f"Unexpected error: {str(e)}", "error")
            return 1
    
    # Run async function
    exit_code = asyncio.run(process_document())
    if exit_code != 0:
        sys.exit(exit_code)


@cli.command()
@click.option('--input-dir', '-i', help='Directory containing documents to process')
@click.option('--pattern', '-p', default='**/*', help='File pattern to match')
@click.option('--output', '-o', help='Output directory for results')
@click.option('--max-concurrent', '-m', default=3, help='Maximum concurrent processing tasks')
@click.option('--extract-entities/--no-extract-entities', default=True, help='Extract entities')
@click.option('--extract-relationships/--no-extract-relationships', default=True, help='Extract relationships')
@click.pass_context
def batch(ctx, input_dir: str, pattern: str, output: Optional[str], 
          max_concurrent: int, extract_entities: bool, extract_relationships: bool):
    """
    Process multiple documents in batch mode.
    
    Efficiently process multiple documents from a directory with advanced
    queue management and parallel processing capabilities.
    """
    
    # Load configuration when needed
    ensure_config_loaded(ctx)
    
    app = ctx.obj['app']
    config = ctx.obj['config']
    
    if not input_dir:
        raise click.UsageError("Must specify --input-dir")
    
    input_path = Path(input_dir)
    if not input_path.exists():
        raise click.UsageError(f"Input directory does not exist: {input_dir}")
    
    async def batch_process():
        """Batch process documents asynchronously."""
        
        try:
            # Find files
            files = list(input_path.glob(pattern))
            if not files:
                app.print_status(f"No files found matching pattern: {pattern}", "warning")
                return 0
            
            app.print_status(f"Found {len(files)} files to process")
            
            # Initialize services
            doc_repo = create_document_repository(config)
            graph_repo = create_graph_repository(config)
            llm_service = create_llm_service(config)
            content_extractor = create_content_extractor_service(config)
            
            # Create use cases
            process_use_case = ProcessDocumentUseCase(
                config, doc_repo, graph_repo, llm_service, content_extractor
            )
            batch_use_case = BatchProcessDocumentsUseCase(config, process_use_case)
            
            # Create documents
            document_ids = []
            for file_path in files:
                document = SourceDocument(
                    title=file_path.stem,
                    file_path=str(file_path),
                    content_type="text/plain"
                )
                await doc_repo.save(document)
                document_ids.append(document.id)
            
            # Configure extraction
            extraction_config = {
                "extract_entities": extract_entities,
                "extract_relationships": extract_relationships,
                "temperature": 0.1,
                "max_tokens": 2000
            }
            
            # Process with progress tracking
            if RICH_AVAILABLE:
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    TimeElapsedColumn(),
                    console=console
                ) as progress:
                    task = progress.add_task(f"Processing {len(files)} documents...", total=None)
                    result = await batch_use_case.execute(
                        document_ids, extraction_config, max_concurrent
                    )
                    progress.update(task, completed=True)
            else:
                app.print_status(f"Processing {len(files)} documents...", "info")
                result = await batch_use_case.execute(
                    document_ids, extraction_config, max_concurrent
                )
            
            # Display results
            if result.success:
                app.print_status("Batch processing completed!", "success")
                
                if result.data:
                    metrics = [
                        {"Metric": "Total Documents", "Value": result.data.get("total_documents", 0)},
                        {"Metric": "Successfully Processed", "Value": result.data.get("processed_documents", 0)},
                        {"Metric": "Failed", "Value": result.data.get("failed_documents", 0)},
                        {"Metric": "Success Rate", "Value": f"{result.data.get('success_rate', 0):.1f}%"},
                        {"Metric": "Total Time", "Value": f"{result.processing_time:.2f}s"}
                    ]
                    app.print_table(metrics, "Batch Processing Results")
                
                # Save detailed results if requested
                if output:
                    output_path = Path(output)
                    output_path.mkdir(parents=True, exist_ok=True)
                    
                    results_file = output_path / "batch_results.json"
                    with open(results_file, 'w') as f:
                        json.dump(result.to_dict(), f, indent=2)
                    
                    app.print_status(f"Detailed results saved to {results_file}", "success")
                
            else:
                app.print_status(f"Batch processing failed: {result.message}", "error")
                return 1
            
            return 0
            
        except Exception as e:
            app.logger.error(f"Batch processing error: {str(e)}", exc_info=True)
            app.print_status(f"Unexpected error: {str(e)}", "error")
            return 1
    
    # Run async function
    exit_code = asyncio.run(batch_process())
    if exit_code != 0:
        sys.exit(exit_code)


@cli.command()
@click.option('--status', help='Filter by processing status')
@click.option('--limit', '-l', default=20, help='Maximum number of documents to show')
@click.option('--output', '-o', help='Output file for results')
@click.pass_context
def list_documents(ctx, status: Optional[str], limit: int, output: Optional[str]):
    """
    List processed documents with filtering and search capabilities.
    
    View all processed documents in your knowledge base with advanced
    filtering, sorting, and export capabilities.
    """
    
    # Load configuration when needed
    ensure_config_loaded(ctx)
    
    app = ctx.obj['app']
    config = ctx.obj['config']
    
    async def list_docs():
        """List documents asynchronously."""
        
        try:
            # Initialize repository
            doc_repo = create_document_repository(config)
            
            # Get documents
            if status:
                try:
                    status_enum = ProcessingStatus(status)
                    documents = await doc_repo.find_by_status(status_enum)
                except ValueError:
                    app.print_status(f"Invalid status: {status}", "error")
                    return 1
            else:
                # Get all documents (this would need to be implemented in repository)
                documents = []
            
            # Limit results
            documents = documents[:limit]
            
            if not documents:
                app.print_status("No documents found", "warning")
                return 0
            
            # Prepare data for display
            doc_data = []
            for doc in documents:
                doc_data.append({
                    "ID": doc.id[:8] + "...",
                    "Title": doc.title[:40] + "..." if len(doc.title) > 40 else doc.title,
                    "Status": doc.processing_status.value,
                    "Entities": doc.extracted_entities,
                    "Relationships": doc.extracted_relationships,
                    "Created": doc.metadata.created_at.strftime("%Y-%m-%d %H:%M")
                })
            
            # Display results
            app.print_table(doc_data, f"Documents ({len(documents)} found)")
            
            # Save to file if requested
            if output:
                full_data = [doc.to_dict() for doc in documents]
                with open(output, 'w') as f:
                    json.dump(full_data, f, indent=2)
                app.print_status(f"Full document data saved to {output}", "success")
            
            return 0
            
        except Exception as e:
            app.logger.error(f"List documents error: {str(e)}", exc_info=True)
            app.print_status(f"Error listing documents: {str(e)}", "error")
            return 1
    
    # Run async function
    exit_code = asyncio.run(list_docs())
    if exit_code != 0:
        sys.exit(exit_code)


@cli.command()
@click.option('--deduplicate/--no-deduplicate', default=True, help='Deduplicate entities')
@click.option('--optimize-relationships/--no-optimize-relationships', default=True, help='Optimize relationships')
@click.option('--analyze-structure/--no-analyze-structure', default=True, help='Analyze graph structure')
@click.pass_context
def optimize(ctx, deduplicate: bool, optimize_relationships: bool, analyze_structure: bool):
    """
    Optimize the knowledge graph with advanced algorithms.
    
    Perform sophisticated graph optimization including entity deduplication,
    relationship optimization, and structural analysis.
    """
    
    # Load configuration when needed
    ensure_config_loaded(ctx)
    
    app = ctx.obj['app']
    config = ctx.obj['config']
    
    async def optimize_graph():
        """Optimize graph asynchronously."""
        
        try:
            # Initialize services
            graph_repo = create_graph_repository(config)
            
            # Create use case
            optimize_use_case = OptimizeKnowledgeGraphUseCase(config, graph_repo)
            
            # Configure optimization
            optimization_config = {
                "deduplicate_entities": deduplicate,
                "optimize_relationships": optimize_relationships,
                "analyze_structure": analyze_structure
            }
            
            app.print_status("Starting graph optimization...")
            
            # Run optimization
            result = await optimize_use_case.execute(optimization_config)
            
            # Display results
            if result.success:
                app.print_status("Graph optimization completed successfully!", "success")
                
                if result.data:
                    metrics = []
                    if "entities_merged" in result.metrics:
                        metrics.append({"Operation": "Entity Deduplication", "Count": result.metrics["entities_merged"]})
                    if "relationships_optimized" in result.metrics:
                        metrics.append({"Operation": "Relationship Optimization", "Count": result.metrics["relationships_optimized"]})
                    
                    if metrics:
                        app.print_table(metrics, "Optimization Results")
                    
                    # Show structure analysis if available
                    if result.data and "total_nodes" in result.data:
                        structure = [
                            {"Metric": "Total Nodes", "Value": result.data["total_nodes"]},
                            {"Metric": "Total Edges", "Value": result.data["total_edges"]},
                            {"Metric": "Connected Components", "Value": result.data.get("connected_components", 0)},
                            {"Metric": "Average Degree", "Value": f"{result.data.get('average_degree', 0):.2f}"}
                        ]
                        app.print_table(structure, "Graph Structure Analysis")
            else:
                app.print_status(f"Optimization failed: {result.message}", "error")
                return 1
            
            return 0
            
        except Exception as e:
            app.logger.error(f"Optimization error: {str(e)}", exc_info=True)
            app.print_status(f"Unexpected error: {str(e)}", "error")
            return 1
    
    # Run async function
    exit_code = asyncio.run(optimize_graph())
    if exit_code != 0:
        sys.exit(exit_code)


@cli.command()
@click.pass_context
def config_info(ctx):
    """
    Display current configuration information.
    
    Show detailed configuration settings including database connection,
    LLM settings, processing parameters, and system information.
    """
    
    # Load configuration when needed
    ensure_config_loaded(ctx)
    
    app = ctx.obj['app']
    config = ctx.obj['config']
    
    if RICH_AVAILABLE:
        # Create configuration panel
        config_text = f"""
Database Provider: {config.database.provider}
Database URI: {config.database.uri[:50]}...
        
LLM Provider: {config.llm.provider.value}
LLM Model: {config.llm.model_name}
        
Processing Chunk Size: {config.processing.chunk_size}
Max Concurrent Tasks: {config.processing.max_concurrent_tasks}
        
Log Level: {config.logging.level}
Log Format: {config.logging.format}
        """
        
        panel = Panel(
            config_text,
            title="GraphBuilder Configuration",
            border_style="blue"
        )
        console.print(panel)
    else:
        click.echo("GraphBuilder Configuration:")
        click.echo(f"  Database Provider: {config.database.provider}")
        click.echo(f"  LLM Provider: {config.llm.provider.value}")
        click.echo(f"  LLM Model: {config.llm.model_name}")
        click.echo(f"  Chunk Size: {config.processing.chunk_size}")


@cli.command()
@click.pass_context  
def version(ctx):
    """Show version information."""
    
    try:
        from .. import __version__
        version_str = __version__
    except ImportError:
        version_str = "unknown"
    
    if RICH_AVAILABLE:
        text = Text(f"GraphBuilder version {version_str}")
        text.stylize("bold blue")
        console.print(text)
    else:
        click.echo(f"GraphBuilder version {version_str}")


@cli.command()
@click.option('--source', required=True,
              type=click.Choice(['open-targets', 'pubmed'], case_sensitive=False),
              help='External data source to ingest from.')
@click.option('--disease-id', default=None,
              help='EFO disease identifier for Open Targets, e.g. EFO_0000275.')
@click.option('--query', default=None,
              help='Search query for PubMed, e.g. "BRCA1 breast cancer".')
@click.option('--max-results', default=500, show_default=True,
              help='Maximum number of records to fetch.')
@click.option('--min-score', default=0.0, show_default=True,
              help='(Open Targets only) Minimum association score (0.0 – 1.0).')
@click.option('--email', default=None,
              help='(PubMed only) Email address for NCBI API identification.')
@click.option('--output', '-o', help='Optional JSON file to write the result summary.')
@click.pass_context
def ingest(
    ctx,
    source: str,
    disease_id: Optional[str],
    query: Optional[str],
    max_results: int,
    min_score: float,
    email: Optional[str],
    output: Optional[str],
):
    """
    Ingest data from an external biomedical source into the knowledge graph.

    Supported sources:

    \b
      open-targets   Open Targets Platform (disease–target associations)
      pubmed         NCBI PubMed (article metadata + MeSH concepts)

    Examples:

    \b
      graphbuilder ingest --source open-targets --disease-id EFO_0000275
      graphbuilder ingest --source pubmed --query "BRCA1 breast cancer" \\
          --max-results 200 --email you@example.com --output results.json
    """
    ensure_config_loaded(ctx)

    app = ctx.obj['app']
    config = ctx.obj['config']

    # Validate source-specific required options
    if source == 'open-targets' and not disease_id:
        raise click.UsageError("--disease-id is required for --source open-targets")
    if source == 'pubmed' and not query:
        raise click.UsageError("--query is required for --source pubmed")

    async def run_ingestion():
        try:
            graph_repo = create_graph_repository(config)

            if source == 'open-targets':
                use_case = OpenTargetsIngestionUseCase(config, graph_repo)
                ingestion_cfg = IngestionConfig(
                    disease_id=disease_id,
                    max_associations=max_results,
                    min_association_score=min_score,
                )
                app.print_status(
                    f"Ingesting '{disease_id}' from Open Targets "
                    f"(max {max_results} associations, min score {min_score})…"
                )
            else:  # pubmed
                use_case = PubMedIngestionUseCase(config, graph_repo)
                ingestion_cfg = PubMedIngestionConfig(
                    query=query,
                    max_articles=max_results,
                    email=email or "graphbuilder@example.com",
                )
                app.print_status(
                    f"Searching PubMed for '{query}' (max {max_results} articles)…"
                )

            if RICH_AVAILABLE:
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    TimeElapsedColumn(),
                    console=console,
                ) as progress:
                    task = progress.add_task("Fetching and persisting…", total=None)
                    result = await use_case.execute(ingestion_cfg)
                    progress.update(task, completed=True)
            else:
                result = await use_case.execute(ingestion_cfg)

            if result.success:
                app.print_status("Ingestion completed successfully!", "success")

                if result.data:
                    if source == 'open-targets':
                        metrics = [
                            {"Metric": "Disease", "Value": result.data.get("disease_name", disease_id)},
                            {"Metric": "Entities Created", "Value": result.data.get("entities_created", 0)},
                            {"Metric": "Relationships Created", "Value": result.data.get("relationships_created", 0)},
                            {"Metric": "Associations Fetched", "Value": result.data.get("associations_fetched", 0)},
                            {"Metric": "Total Available", "Value": result.data.get("total_associations_available", "?")},
                            {"Metric": "Processing Time", "Value": f"{result.processing_time:.2f}s"},
                        ]
                    else:
                        metrics = [
                            {"Metric": "Query", "Value": query},
                            {"Metric": "Articles Fetched", "Value": result.data.get("articles_fetched", 0)},
                            {"Metric": "Entities Created", "Value": result.data.get("entities_created", 0)},
                            {"Metric": "Relationships Created", "Value": result.data.get("relationships_created", 0)},
                            {"Metric": "Processing Time", "Value": f"{result.processing_time:.2f}s"},
                        ]
                    app.print_table(metrics, "Ingestion Results")

                if output:
                    import json as _json
                    with open(output, "w") as fh:
                        _json.dump(result.data, fh, indent=2)
                    app.print_status(f"Result summary written to {output}", "success")

                return 0

            else:
                app.print_status(f"Ingestion failed: {result.message}", "error")
                for err in result.errors:
                    app.print_status(f"  {err}", "error")
                return 1

        except Exception as exc:
            app.logger.error("Ingestion error: %s", exc, exc_info=True)
            app.print_status(f"Unexpected error: {exc}", "error")
            return 1

    exit_code = asyncio.run(run_ingestion())
    if exit_code != 0:
        sys.exit(exit_code)


@cli.command()
@click.option('--input', '-i', 'input_file', required=True,
              help='JSON file containing curation events (see docs for schema).')
@click.option('--curator', required=True,
              help='Identifier of the human curator (username or email).')
@click.option('--output', '-o', help='Optional JSON file to write the audit log.')
@click.option('--dry-run', is_flag=True,
              help='Validate the input file without applying changes.')
@click.pass_context
def curate(ctx, input_file: str, curator: str, output: Optional[str], dry_run: bool):
    """
    Apply manual curation events to the knowledge graph.

    The input JSON must be an array of curation event objects, each with:

    \b
      {
        "action":      "approve_entity" | "reject_entity" | "correct_entity" |
                       "approve_relationship" | "reject_relationship" |
                       "correct_relationship",
        "target_id":   "<entity or relationship UUID>",
        "reason":      "optional human note",
        "corrections": {}   // only for correct_* actions
      }

    Example:

    \b
      graphbuilder curate --input events.json --curator alice@example.com
    """
    ensure_config_loaded(ctx)

    app = ctx.obj['app']
    config = ctx.obj['config']

    # Load and validate input file
    input_path = Path(input_file)
    if not input_path.exists():
        raise click.UsageError(f"Input file not found: {input_file}")

    try:
        with open(input_path) as fh:
            raw_events = json.load(fh)
        if not isinstance(raw_events, list):
            raise ValueError("Expected a JSON array of curation events")
    except (json.JSONDecodeError, ValueError) as exc:
        raise click.UsageError(f"Invalid input file: {exc}")

    app.print_status(f"Loaded {len(raw_events)} curation events from {input_file}")

    if dry_run:
        app.print_status("Dry-run mode — no changes will be applied.", "warning")
        # Validate each event's action field
        valid_actions = {a.value for a in CurationAction}
        errors = []
        for i, ev in enumerate(raw_events):
            if ev.get("action") not in valid_actions:
                errors.append(f"Event {i}: unknown action '{ev.get('action')}'")
            if not ev.get("target_id"):
                errors.append(f"Event {i}: missing 'target_id'")
        if errors:
            for err in errors:
                app.print_status(err, "error")
            sys.exit(1)
        app.print_status("Validation passed.", "success")
        return

    async def run_curation():
        try:
            graph_repo = create_graph_repository(config)
            use_case = CurationUseCase(config, graph_repo)

            # Build CurationRequest from raw JSON
            request = CurationRequest(curator=curator)
            for ev in raw_events:
                action = ev.get("action", "")
                target_id = ev.get("target_id", "")
                reason = ev.get("reason", "")
                corrections = ev.get("corrections", {})

                if action == CurationAction.APPROVE_ENTITY.value:
                    request.approve_entity(target_id, reason)
                elif action == CurationAction.REJECT_ENTITY.value:
                    request.reject_entity(target_id, reason)
                elif action == CurationAction.CORRECT_ENTITY.value:
                    request.correct_entity(target_id, corrections, reason)
                elif action == CurationAction.APPROVE_RELATIONSHIP.value:
                    request.approve_relationship(target_id, reason)
                elif action == CurationAction.REJECT_RELATIONSHIP.value:
                    request.reject_relationship(target_id, reason)
                elif action == CurationAction.CORRECT_RELATIONSHIP.value:
                    request.correct_relationship(target_id, corrections, reason)
                else:
                    app.print_status(f"Unknown action '{action}', skipping", "warning")

            result = await use_case.execute(request)

            if result.success:
                app.print_status("Curation applied successfully!", "success")
                s = result.data.get("summary", {})
                metrics = [
                    {"Metric": "Events Applied", "Value": result.data.get("summary", {}).get("total_events", 0) - s.get("skipped", 0)},
                    {"Metric": "Entities Approved", "Value": s.get("approved_entities", 0)},
                    {"Metric": "Entities Rejected", "Value": s.get("rejected_entities", 0)},
                    {"Metric": "Entities Corrected", "Value": s.get("corrected_entities", 0)},
                    {"Metric": "Rels Approved", "Value": s.get("approved_relationships", 0)},
                    {"Metric": "Rels Rejected", "Value": s.get("rejected_relationships", 0)},
                    {"Metric": "Rels Corrected", "Value": s.get("corrected_relationships", 0)},
                    {"Metric": "Skipped", "Value": s.get("skipped", 0)},
                ]
                app.print_table(metrics, "Curation Results")
            else:
                app.print_status(f"Curation completed with errors: {result.message}", "warning")
                for err in result.errors:
                    app.print_status(f"  {err}", "error")

            if output and result.data.get("audit_log"):
                with open(output, "w") as fh:
                    json.dump(result.data["audit_log"], fh, indent=2)
                app.print_status(f"Audit log written to {output}", "success")

            return 0 if result.success else 1

        except Exception as exc:
            app.logger.error("Curation error: %s", exc, exc_info=True)
            app.print_status(f"Unexpected error: {exc}", "error")
            return 1

    exit_code = asyncio.run(run_curation())
    if exit_code != 0:
        sys.exit(exit_code)


@cli.command()
@click.option('--format', 'fmt', default='html',
              type=click.Choice(['cytoscape', 'graphml', 'html'], case_sensitive=False),
              show_default=True, help='Export format.')
@click.option('--output', '-o', default=None,
              help='Output file path (default: graph.<format-ext>).')
@click.option('--title', default='Knowledge Graph', show_default=True,
              help='Page title for HTML output.')
@click.option('--entity-types', default=None,
              help='Comma-separated entity types to include (default: all).')
@click.option('--relationship-types', default=None,
              help='Comma-separated relationship types to include (default: all).')
@click.option('--include-rejected', is_flag=True, default=False,
              help='Include entities/relationships marked as rejected by curation.')
@click.pass_context
def export(ctx: click.Context, fmt: str, output: Optional[str], title: str,
           entity_types: Optional[str], relationship_types: Optional[str],
           include_rejected: bool) -> None:
    """Export the knowledge graph to a portable format.

    \b
    Examples:
      graphbuilder export --format html --output viz/graph.html
      graphbuilder export --format graphml --output graph.graphml
      graphbuilder export --format cytoscape --output graph.json \\
          --entity-types Concept,Person
    """
    app: CLIApp = ctx.obj

    _EXT = {"cytoscape": "json", "graphml": "graphml", "html": "html"}
    if output is None:
        output = f"graph.{_EXT[fmt.lower()]}"

    entity_filter = [t.strip() for t in entity_types.split(",")] if entity_types else None
    rel_filter = [t.strip() for t in relationship_types.split(",")] if relationship_types else None

    async def run_export() -> int:
        try:
            app.print_status(f"Exporting graph → {output} ({fmt}) …", "info")

            config = app.config

            # Load graph from repository
            graph_repo = create_graph_repository(config)
            raw_graph = await graph_repo.load_graph() if hasattr(graph_repo, "load_graph") else KnowledgeGraph()

            vis_config = VisualizationConfig(
                output_path=output,
                format=fmt,
                title=title,
                include_rejected=include_rejected,
                entity_types=entity_filter,
                relationship_types=rel_filter,
            )

            use_case = GraphVisualizationUseCase(raw_graph)
            result = use_case.execute(vis_config)

            if result.success:
                d = result.data or {}
                app.print_status(
                    f"Exported {d.get('nodes', 0)} nodes, {d.get('edges', 0)} edges → {output}",
                    "success",
                )
                return 0
            else:
                app.print_status(f"Export failed: {result.error}", "error")
                return 1

        except Exception as exc:
            app.logger.error("Export error: %s", exc, exc_info=True)
            app.print_status(f"Unexpected error: {exc}", "error")
            return 1

    exit_code = asyncio.run(run_export())
    if exit_code != 0:
        sys.exit(exit_code)


@cli.command()
@click.option('--output', '-o', default='verification_report.json',
              show_default=True, help='Write JSON report to this file.')
@click.option('--no-embedding', is_flag=True, default=False,
              help='Disable the embedding stage (no sentence-transformers needed).')
@click.option('--no-llm', is_flag=True, default=False,
              help='Disable the LLM stage.')
@click.option('--early-exit-pass', is_flag=True, default=False,
              help='Stop pipeline as soon as a stage passes.')
@click.option('--early-exit-fail', is_flag=True, default=False,
              help='Stop pipeline as soon as a stage fails.')
@click.option('--threshold', default=0.5, show_default=True, type=float,
              help='Cosine similarity threshold for the embedding stage.')
@click.option('--context-file', default=None,
              help='JSON file mapping relationship_id → context string.')
@click.pass_context
def verify(ctx: click.Context, output: str, no_embedding: bool, no_llm: bool,
           early_exit_pass: bool, early_exit_fail: bool,
           threshold: float, context_file: Optional[str]) -> None:
    """Verify relationships in the graph using the cascading pipeline.

    \b
    Runs up to three stages for each relationship:
      1. TextMatch  — lexical term presence check
      2. Embedding  — cosine similarity via sentence-transformers
      3. LLM        — structured LLM prompt with reasoning trace

    Results are written to --output as a JSON report.  Relationships that
    fail are annotated in-memory (use `curate` to persist corrections).

    \b
    Examples:
      graphbuilder verify --no-embedding --no-llm
      graphbuilder verify --threshold 0.6 --output report.json
    """
    app: CLIApp = ctx.obj

    async def run_verify() -> int:
        try:
            context_map: Dict[str, Any] = {}
            if context_file:
                import json as _json
                with open(context_file) as fh:
                    context_map = _json.load(fh)

            text_cfg = TextMatchConfig()
            emb_cfg  = EmbeddingConfig(threshold=threshold)
            cascade_cfg = CascadingVerifierConfig(
                text_match=text_cfg,
                embedding=emb_cfg,
                enable_embedding=not no_embedding,
                enable_llm=not no_llm,
                early_exit_on_pass=early_exit_pass,
                early_exit_on_fail=early_exit_fail,
            )

            config = app.config
            graph_repo = create_graph_repository(config)
            raw_graph = await graph_repo.load_graph() if hasattr(graph_repo, 'load_graph') else KnowledgeGraph()

            llm_svc = None
            if not no_llm:
                try:
                    llm_svc = create_llm_service(config)
                except Exception:
                    app.print_status('LLM service unavailable; LLM stage will be skipped.', 'warning')

            ver_config = VerificationConfig(
                cascading=cascade_cfg,
                context_map=context_map,
            )
            use_case = RelationshipVerificationUseCase(raw_graph, llm_service=llm_svc)
            result = use_case.execute(ver_config)

            if result.success:
                d = result.data or {}
                app.print_status(
                    f"Verified {d.get('total', 0)} relationships: "
                    f"{d.get('passed', 0)} passed, "
                    f"{d.get('failed', 0)} failed, "
                    f"{d.get('skipped', 0)} skipped.",
                    'success',
                )
                import json as _json
                from pathlib import Path as _Path
                _Path(output).parent.mkdir(parents=True, exist_ok=True)
                _Path(output).write_text(_json.dumps(d.get('report', []), indent=2), encoding='utf-8')
                app.print_status(f'Report written to {output}', 'info')
                return 0
            else:
                app.print_status(f'Verification failed: {result.error}', 'error')
                return 1

        except Exception as exc:
            app.logger.error('Verify error: %s', exc, exc_info=True)
            app.print_status(f'Unexpected error: {exc}', 'error')
            return 1

    exit_code = asyncio.run(run_verify())
    if exit_code != 0:
        sys.exit(exit_code)


if __name__ == '__main__':
    cli()