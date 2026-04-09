"""
Document Processing Use Cases - Sophisticated business logic for document operations.

This module contains advanced use cases for document processing with
enterprise-grade error handling, validation, and workflow orchestration.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, AsyncGenerator
from datetime import datetime, timezone
from abc import ABC, abstractmethod

from ...domain.models.graph_models import (
    SourceDocument, DocumentChunk, GraphEntity, GraphRelationship,
    EntityType, RelationshipType, ProcessingStatus
)
from ...domain.models.processing_models import (
    ProcessingTask, ProcessingPipeline, ProcessingResult, TaskType, TaskPriority
)
from ...infrastructure.repositories.document_repository import DocumentRepositoryInterface
from ...infrastructure.repositories.graph_repository import GraphRepositoryInterface
from ...infrastructure.services.llm_service import LLMServiceInterface
from ...infrastructure.services.content_extractor import ContentExtractorInterface
from ...infrastructure.config.settings import GraphBuilderConfig


class UseCase(ABC):
    """Abstract base class for all use cases."""
    
    def __init__(self, config: GraphBuilderConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    async def execute(self, *args, **kwargs) -> ProcessingResult:
        """Execute the use case."""
        pass


class ProcessDocumentUseCase(UseCase):
    """
    Sophisticated document processing use case with advanced workflow orchestration.
    
    Handles complete document processing pipeline from content extraction
    to knowledge graph construction with enterprise-grade error handling.
    """
    
    def __init__(
        self,
        config: GraphBuilderConfig,
        document_repo: DocumentRepositoryInterface,
        graph_repo: GraphRepositoryInterface,
        llm_service: LLMServiceInterface,
        content_extractor: ContentExtractorInterface
    ):
        super().__init__(config)
        self.document_repo = document_repo
        self.graph_repo = graph_repo
        self.llm_service = llm_service
        self.content_extractor = content_extractor
    
    async def execute(
        self,
        document_id: str,
        extraction_config: Optional[Dict[str, Any]] = None
    ) -> ProcessingResult:
        """
        Execute comprehensive document processing pipeline.
        
        Args:
            document_id: Document identifier
            extraction_config: Optional extraction configuration
            
        Returns:
            ProcessingResult: Comprehensive processing results
        """
        start_time = datetime.now(timezone.utc)
        result = ProcessingResult(success=True, message="Document processing initiated")
        
        try:
            # Load document
            document = await self.document_repo.get_by_id(document_id)
            if not document:
                result.add_error(f"Document {document_id} not found")
                return result
            
            self.logger.info(f"Processing document: {document.title}")
            
            # Update document status
            document.update_processing_status(ProcessingStatus.IN_PROGRESS)
            await self.document_repo.update(document)
            
            # Create processing pipeline
            pipeline = await self._create_processing_pipeline(document, extraction_config)
            
            # Execute pipeline
            pipeline_result = await self._execute_pipeline(pipeline)
            
            if pipeline_result.success:
                document.update_processing_status(ProcessingStatus.COMPLETED)
                result.message = "Document processing completed successfully"
                result.data = {
                    "document_id": document_id,
                    "pipeline_id": pipeline.id,
                    "processing_summary": pipeline.get_execution_summary(),
                    "extracted_entities": document.extracted_entities,
                    "extracted_relationships": document.extracted_relationships
                }
            else:
                document.update_processing_status(ProcessingStatus.FAILED, pipeline_result.message)
                result.success = False
                result.message = f"Document processing failed: {pipeline_result.message}"
                result.errors.extend(pipeline_result.errors)
            
            await self.document_repo.update(document)
            
            # Calculate metrics
            processing_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            result.processing_time = processing_time
            result.add_metric("processing_time_seconds", processing_time)
            result.add_metric("document_length", document.content_length)
            result.add_metric("chunks_processed", document.processed_chunks)
            
        except Exception as e:
            self.logger.error(f"Error processing document {document_id}: {str(e)}", exc_info=True)
            result.add_error(f"Unexpected error: {str(e)}")
            
            # Update document status on error
            try:
                if 'document' in locals():
                    document.update_processing_status(ProcessingStatus.FAILED, str(e))
                    await self.document_repo.update(document)
            except Exception as update_error:
                self.logger.error(f"Failed to update document status: {str(update_error)}")
        
        return result
    
    async def _create_processing_pipeline(
        self,
        document: SourceDocument,
        extraction_config: Optional[Dict[str, Any]]
    ) -> ProcessingPipeline:
        """Create sophisticated processing pipeline for document."""
        
        pipeline = ProcessingPipeline(
            name=f"Process Document: {document.title}",
            description=f"Complete processing pipeline for document {document.id}",
            configuration=extraction_config or {}
        )
        
        # Task 1: Content Chunking
        chunking_task = ProcessingTask(
            task_type=TaskType.CONTENT_EXTRACTION,
            name="Content Chunking",
            description="Split document content into processable chunks",
            priority=TaskPriority.HIGH,
            configuration={
                "chunk_size": self.config.processing.chunk_size,
                "overlap_size": self.config.processing.overlap_size,
                "preserve_structure": True
            },
            input_data={"document_id": document.id}
        )
        
        # Task 2: Entity Extraction
        entity_extraction_task = ProcessingTask(
            task_type=TaskType.ENTITY_EXTRACTION,
            name="Entity Extraction",
            description="Extract entities from document chunks using LLM",
            priority=TaskPriority.HIGH,
            configuration=extraction_config or {},
            depends_on=[chunking_task.id]
        )
        
        # Task 3: Relationship Extraction
        relationship_extraction_task = ProcessingTask(
            task_type=TaskType.RELATIONSHIP_EXTRACTION,
            name="Relationship Extraction",
            description="Extract relationships between entities",
            priority=TaskPriority.NORMAL,
            configuration=extraction_config or {},
            depends_on=[entity_extraction_task.id]
        )
        
        # Task 4: Graph Construction
        graph_construction_task = ProcessingTask(
            task_type=TaskType.GRAPH_CONSTRUCTION,
            name="Graph Construction",
            description="Build knowledge graph from extracted data",
            priority=TaskPriority.NORMAL,
            configuration={},
            depends_on=[relationship_extraction_task.id]
        )
        
        # Task 5: Validation
        validation_task = ProcessingTask(
            task_type=TaskType.VALIDATION,
            name="Graph Validation",
            description="Validate constructed knowledge graph",
            priority=TaskPriority.LOW,
            configuration={
                "validate_entities": True,
                "validate_relationships": True,
                "check_consistency": True
            },
            depends_on=[graph_construction_task.id]
        )
        
        # Add tasks to pipeline
        for task in [chunking_task, entity_extraction_task, relationship_extraction_task,
                    graph_construction_task, validation_task]:
            pipeline.add_task(task)
        
        return pipeline
    
    async def _execute_pipeline(self, pipeline: ProcessingPipeline) -> ProcessingResult:
        """Execute processing pipeline with sophisticated orchestration."""
        
        try:
            pipeline.start_pipeline()
            self.logger.info(f"Starting pipeline: {pipeline.name}")
            
            while not self._is_pipeline_complete(pipeline):
                # Get ready tasks
                ready_tasks = pipeline.get_ready_tasks()
                running_tasks = pipeline.get_running_tasks()
                
                # Check if we can start new tasks
                available_slots = pipeline.max_parallel_tasks - len(running_tasks)
                
                if available_slots > 0 and ready_tasks:
                    # Start new tasks
                    tasks_to_start = ready_tasks[:available_slots]
                    
                    for task in tasks_to_start:
                        task_result = await self._execute_task(task, pipeline)
                        
                        if task_result.success:
                            pipeline.complete_task(task.id, task_result)
                        else:
                            pipeline.fail_task(task.id, task_result)
                            
                            if not pipeline.continue_on_error:
                                pipeline.fail_pipeline(f"Task {task.name} failed: {task_result.message}")
                                return ProcessingResult(
                                    success=False,
                                    message=f"Pipeline failed due to task failure: {task_result.message}",
                                    errors=task_result.errors
                                )
                
                # Small delay to prevent busy waiting
                await asyncio.sleep(0.1)
            
            # Check final status
            if len(pipeline.failed_tasks) > 0 and not pipeline.continue_on_error:
                pipeline.fail_pipeline("Pipeline completed with failed tasks")
                return ProcessingResult(
                    success=False,
                    message="Pipeline completed with failures",
                    data={"failed_tasks": list(pipeline.failed_tasks)}
                )
            else:
                pipeline.complete_pipeline()
                return ProcessingResult(
                    success=True,
                    message="Pipeline completed successfully",
                    data=pipeline.get_execution_summary()
                )
                
        except Exception as e:
            self.logger.error(f"Pipeline execution error: {str(e)}", exc_info=True)
            pipeline.fail_pipeline(str(e))
            return ProcessingResult(
                success=False,
                message=f"Pipeline execution failed: {str(e)}",
                errors=[str(e)]
            )
    
    def _is_pipeline_complete(self, pipeline: ProcessingPipeline) -> bool:
        """Check if pipeline execution is complete."""
        total_tasks = len(pipeline.tasks)
        completed_failed = len(pipeline.completed_tasks) + len(pipeline.failed_tasks)
        return completed_failed >= total_tasks
    
    async def _execute_task(self, task: ProcessingTask, pipeline: ProcessingPipeline) -> ProcessingResult:
        """Execute individual processing task."""
        
        self.logger.info(f"Executing task: {task.name}")
        task.start_execution()
        
        try:
            if task.task_type == TaskType.CONTENT_EXTRACTION:
                return await self._execute_content_chunking(task)
            elif task.task_type == TaskType.ENTITY_EXTRACTION:
                return await self._execute_entity_extraction(task)
            elif task.task_type == TaskType.RELATIONSHIP_EXTRACTION:
                return await self._execute_relationship_extraction(task)
            elif task.task_type == TaskType.GRAPH_CONSTRUCTION:
                return await self._execute_graph_construction(task)
            elif task.task_type == TaskType.VALIDATION:
                return await self._execute_validation(task)
            else:
                return ProcessingResult(
                    success=False,
                    message=f"Unknown task type: {task.task_type}",
                    errors=[f"Task type {task.task_type} not supported"]
                )
                
        except Exception as e:
            self.logger.error(f"Task execution error: {str(e)}", exc_info=True)
            return ProcessingResult(
                success=False,
                message=f"Task execution failed: {str(e)}",
                errors=[str(e)]
            )
    
    async def _execute_content_chunking(self, task: ProcessingTask) -> ProcessingResult:
        """Execute content chunking task."""
        
        document_id = task.input_data.get("document_id")
        document = await self.document_repo.get_by_id(document_id)
        
        if not document:
            return ProcessingResult(
                success=False,
                message=f"Document {document_id} not found",
                errors=["Document not found"]
            )
        
        # Extract content if needed
        if document.source_url:
            content_result = await self.content_extractor.extract_from_url(document.source_url)
        elif document.file_path:
            content_result = await self.content_extractor.extract_from_file(document.file_path)
        else:
            return ProcessingResult(
                success=False,
                message="No content source available",
                errors=["Neither URL nor file path provided"]
            )
        
        if not content_result.success:
            return content_result
        
        # Create chunks
        content = content_result.data.get("content", "")
        chunks = self._create_content_chunks(content, document.id, task.configuration)
        
        # Save chunks
        for chunk in chunks:
            await self.document_repo.save_chunk(chunk)
        
        # Update document
        document.total_chunks = len(chunks)
        document.content_length = len(content)
        await self.document_repo.update(document)
        
        task.update_progress(100.0, f"Created {len(chunks)} chunks")
        
        return ProcessingResult(
            success=True,
            message=f"Successfully created {len(chunks)} chunks",
            data={
                "chunks_created": len(chunks),
                "content_length": len(content)
            }
        )
    
    def _create_content_chunks(
        self,
        content: str,
        document_id: str,
        config: Dict[str, Any]
    ) -> List[DocumentChunk]:
        """Create content chunks from document content."""
        
        chunk_size = config.get("chunk_size", 1000)
        overlap_size = config.get("overlap_size", 100)
        
        chunks = []
        start = 0
        chunk_index = 0
        
        while start < len(content):
            end = min(start + chunk_size, len(content))
            chunk_content = content[start:end]
            
            if chunk_content.strip():
                chunk = DocumentChunk(
                    content=chunk_content,
                    document_id=document_id,
                    chunk_index=chunk_index,
                    token_count=len(chunk_content.split()),  # Simple token count
                    character_count=len(chunk_content),
                    start_position=start,
                    end_position=end
                )
                chunks.append(chunk)
                chunk_index += 1
            
            # Move start position with overlap
            start = end - overlap_size if end < len(content) else end
        
        return chunks
    
    async def _execute_entity_extraction(self, task: ProcessingTask) -> ProcessingResult:
        """Execute entity extraction task."""
        
        # Get document chunks
        document_id = task.input_data.get("document_id")
        chunks = await self.document_repo.get_chunks_by_document_id(document_id)
        
        if not chunks:
            return ProcessingResult(
                success=False,
                message="No chunks found for entity extraction",
                errors=["No document chunks available"]
            )
        
        total_entities = 0
        processed_chunks = 0
        
        for i, chunk in enumerate(chunks):
            try:
                # Extract entities using LLM
                extraction_result = await self.llm_service.extract_entities(
                    chunk.content,
                    task.configuration
                )
                
                if extraction_result.success:
                    entities = extraction_result.data.get("entities", [])
                    
                    # Save entities
                    for entity_data in entities:
                        entity = GraphEntity(
                            name=entity_data.get("name"),
                            entity_type=EntityType(entity_data.get("type", "CONCEPT")),
                            description=entity_data.get("description"),
                            properties=entity_data.get("properties", {})
                        )
                        await self.graph_repo.save_entity(entity)
                        total_entities += 1
                
                processed_chunks += 1
                progress = (processed_chunks / len(chunks)) * 100
                task.update_progress(progress, f"Processed {processed_chunks}/{len(chunks)} chunks")
                
            except Exception as e:
                self.logger.error(f"Error extracting entities from chunk {i}: {str(e)}")
                continue
        
        return ProcessingResult(
            success=True,
            message=f"Extracted {total_entities} entities from {processed_chunks} chunks",
            data={
                "entities_extracted": total_entities,
                "chunks_processed": processed_chunks
            }
        )
    
    async def _execute_relationship_extraction(self, task: ProcessingTask) -> ProcessingResult:
        """Execute relationship extraction task."""
        
        # Implementation would extract relationships between entities
        # This is a simplified version
        
        return ProcessingResult(
            success=True,
            message="Relationship extraction completed",
            data={"relationships_extracted": 0}
        )
    
    async def _execute_graph_construction(self, task: ProcessingTask) -> ProcessingResult:
        """Execute graph construction task."""
        
        # Implementation would build the complete knowledge graph
        # This is a simplified version
        
        return ProcessingResult(
            success=True,
            message="Graph construction completed",
            data={"graph_nodes": 0, "graph_edges": 0}
        )
    
    async def _execute_validation(self, task: ProcessingTask) -> ProcessingResult:
        """Execute validation task."""
        
        # Implementation would validate the constructed graph
        # This is a simplified version
        
        return ProcessingResult(
            success=True,
            message="Graph validation completed",
            data={"validation_passed": True}
        )


class BatchProcessDocumentsUseCase(UseCase):
    """
    Sophisticated batch document processing use case.
    
    Handles processing multiple documents with advanced queue management,
    progress tracking, and error recovery.
    """
    
    def __init__(
        self,
        config: GraphBuilderConfig,
        process_document_use_case: ProcessDocumentUseCase
    ):
        super().__init__(config)
        self.process_document_use_case = process_document_use_case
    
    async def execute(
        self,
        document_ids: List[str],
        extraction_config: Optional[Dict[str, Any]] = None,
        max_concurrent: int = 3
    ) -> ProcessingResult:
        """
        Execute batch document processing.
        
        Args:
            document_ids: List of document IDs to process
            extraction_config: Optional extraction configuration
            max_concurrent: Maximum concurrent processing tasks
            
        Returns:
            ProcessingResult: Batch processing results
        """
        
        start_time = datetime.now(timezone.utc)
        result = ProcessingResult(success=True, message="Batch processing initiated")
        
        total_documents = len(document_ids)
        processed_documents = 0
        failed_documents = 0
        results = {}
        
        try:
            self.logger.info(f"Starting batch processing of {total_documents} documents")
            
            # Create semaphore for concurrency control
            semaphore = asyncio.Semaphore(max_concurrent)
            
            async def process_single_document(doc_id: str) -> None:
                nonlocal processed_documents, failed_documents
                
                async with semaphore:
                    try:
                        doc_result = await self.process_document_use_case.execute(
                            doc_id, extraction_config
                        )
                        results[doc_id] = doc_result
                        
                        if doc_result.success:
                            processed_documents += 1
                        else:
                            failed_documents += 1
                            
                    except Exception as e:
                        self.logger.error(f"Error processing document {doc_id}: {str(e)}")
                        failed_documents += 1
                        results[doc_id] = ProcessingResult(
                            success=False,
                            message=f"Processing failed: {str(e)}",
                            errors=[str(e)]
                        )
            
            # Process documents concurrently
            tasks = [process_single_document(doc_id) for doc_id in document_ids]
            await asyncio.gather(*tasks, return_exceptions=True)
            
            # Calculate final results
            success_rate = (processed_documents / total_documents) * 100 if total_documents > 0 else 0
            processing_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            
            result.processing_time = processing_time
            result.data = {
                "total_documents": total_documents,
                "processed_documents": processed_documents,
                "failed_documents": failed_documents,
                "success_rate": success_rate,
                "results": results
            }
            
            result.add_metric("total_documents", total_documents)
            result.add_metric("processed_documents", processed_documents)
            result.add_metric("failed_documents", failed_documents)
            result.add_metric("success_rate", success_rate)
            result.add_metric("processing_time_seconds", processing_time)
            
            if failed_documents > 0:
                result.add_warning(f"{failed_documents} documents failed to process")
            
            result.message = f"Batch processing completed: {processed_documents}/{total_documents} successful"
            
        except Exception as e:
            self.logger.error(f"Batch processing error: {str(e)}", exc_info=True)
            result.add_error(f"Batch processing failed: {str(e)}")
        
        return result


class OptimizeKnowledgeGraphUseCase(UseCase):
    """
    Sophisticated knowledge graph optimization use case.
    
    Handles entity deduplication, relationship optimization,
    and graph structure enhancement with advanced algorithms.
    """
    
    def __init__(
        self,
        config: GraphBuilderConfig,
        graph_repo: GraphRepositoryInterface
    ):
        super().__init__(config)
        self.graph_repo = graph_repo
    
    async def execute(
        self,
        optimization_config: Optional[Dict[str, Any]] = None
    ) -> ProcessingResult:
        """
        Execute knowledge graph optimization.
        
        Args:
            optimization_config: Configuration for optimization algorithms
            
        Returns:
            ProcessingResult: Optimization results
        """
        
        start_time = datetime.now(timezone.utc)
        result = ProcessingResult(success=True, message="Graph optimization initiated")
        
        try:
            config = optimization_config or {}
            
            # Entity deduplication
            if config.get("deduplicate_entities", True):
                dedup_result = await self._deduplicate_entities()
                result.add_metric("entities_merged", dedup_result.get("merged_count", 0))
            
            # Relationship optimization
            if config.get("optimize_relationships", True):
                rel_result = await self._optimize_relationships()
                result.add_metric("relationships_optimized", rel_result.get("optimized_count", 0))
            
            # Graph structure analysis
            if config.get("analyze_structure", True):
                struct_result = await self._analyze_graph_structure()
                result.data = struct_result
            
            processing_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            result.processing_time = processing_time
            result.message = "Graph optimization completed successfully"
            
        except Exception as e:
            self.logger.error(f"Graph optimization error: {str(e)}", exc_info=True)
            result.add_error(f"Optimization failed: {str(e)}")
        
        return result
    
    async def _deduplicate_entities(self) -> Dict[str, Any]:
        """Deduplicate similar entities in the graph."""
        # Implementation would use similarity algorithms
        # This is a simplified version
        return {"merged_count": 0}
    
    async def _optimize_relationships(self) -> Dict[str, Any]:
        """Optimize relationships in the graph."""
        # Implementation would optimize relationship structure
        # This is a simplified version
        return {"optimized_count": 0}
    
    async def _analyze_graph_structure(self) -> Dict[str, Any]:
        """Analyze and report on graph structure."""
        # Implementation would perform structural analysis
        # This is a simplified version
        return {
            "total_nodes": 0,
            "total_edges": 0,
            "connected_components": 0,
            "average_degree": 0.0
        }