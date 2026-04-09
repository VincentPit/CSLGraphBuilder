"""
LLM Service - Sophisticated language model integration for knowledge extraction.

This module provides enterprise-grade LLM integration with advanced
prompt engineering, response validation, and multi-provider support.
"""

import asyncio
import logging
import json
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timezone
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

from ...domain.models.processing_models import ProcessingResult
from ...domain.models.graph_models import EntityType, RelationshipType
from ..config.settings import GraphBuilderConfig, LLMProvider


class PromptType(Enum):
    """Types of prompts for different extraction tasks."""
    ENTITY_EXTRACTION = "entity_extraction"
    RELATIONSHIP_EXTRACTION = "relationship_extraction"
    CONTENT_CLASSIFICATION = "content_classification"
    SUMMARIZATION = "summarization"
    VALIDATION = "validation"


@dataclass
class LLMRequest:
    """Structured LLM request with metadata."""
    
    prompt: str
    content: str
    prompt_type: PromptType
    temperature: float = 0.1
    max_tokens: int = 2000
    model_params: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt": self.prompt,
            "content": self.content,
            "prompt_type": self.prompt_type.value,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "model_params": self.model_params or {}
        }


@dataclass
class LLMResponse:
    """Structured LLM response with metadata."""
    
    content: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    model: str = ""
    finish_reason: str = ""
    processing_time: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "model": self.model,
            "finish_reason": self.finish_reason,
            "processing_time": self.processing_time
        }


class LLMServiceInterface(ABC):
    """Abstract interface for LLM service operations."""
    
    @abstractmethod
    async def extract_entities(
        self,
        content: str,
        config: Dict[str, Any] = None
    ) -> ProcessingResult:
        """Extract entities from content."""
        pass
    
    @abstractmethod
    async def extract_relationships(
        self,
        content: str,
        entities: List[Dict[str, Any]],
        config: Dict[str, Any] = None
    ) -> ProcessingResult:
        """Extract relationships from content."""
        pass
    
    @abstractmethod
    async def classify_content(
        self,
        content: str,
        categories: List[str],
        config: Dict[str, Any] = None
    ) -> ProcessingResult:
        """Classify content into categories."""
        pass
    
    @abstractmethod
    async def summarize_content(
        self,
        content: str,
        config: Dict[str, Any] = None
    ) -> ProcessingResult:
        """Generate content summary."""
        pass


class AdvancedLLMService(LLMServiceInterface):
    """
    Sophisticated LLM service with advanced prompt engineering and validation.
    
    Provides enterprise-grade LLM integration with multiple providers,
    sophisticated prompt templates, response validation, and error handling.
    """
    
    def __init__(self, config: GraphBuilderConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.client = self._initialize_client()
        
        # Load prompt templates
        self.prompts = self._load_prompt_templates()
        
        # Response validators
        self.validators = {
            PromptType.ENTITY_EXTRACTION: self._validate_entity_response,
            PromptType.RELATIONSHIP_EXTRACTION: self._validate_relationship_response,
            PromptType.CONTENT_CLASSIFICATION: self._validate_classification_response,
            PromptType.SUMMARIZATION: self._validate_summary_response
        }
    
    def _initialize_client(self):
        """Initialize LLM client based on configuration."""
        
        if self.config.llm.provider == LLMProvider.OPENAI:
            try:
                import openai
                return openai.AsyncOpenAI(
                    api_key=self.config.llm.api_key,
                    base_url=self.config.llm.base_url,
                    timeout=self.config.llm.timeout
                )
            except ImportError:
                raise RuntimeError("OpenAI package not installed")
        
        elif self.config.llm.provider == LLMProvider.AZURE_OPENAI:
            try:
                import openai
                return openai.AsyncAzureOpenAI(
                    api_key=self.config.llm.api_key,
                    azure_endpoint=self.config.llm.base_url,
                    api_version=self.config.llm.api_version,
                    timeout=self.config.llm.timeout
                )
            except ImportError:
                raise RuntimeError("OpenAI package not installed")
        
        else:
            raise ValueError(f"Unsupported LLM provider: {self.config.llm.provider}")
    
    def _load_prompt_templates(self) -> Dict[PromptType, str]:
        """Load sophisticated prompt templates."""
        
        return {
            PromptType.ENTITY_EXTRACTION: """
You are an expert knowledge graph analyst. Extract entities from the following text with high precision.

INSTRUCTIONS:
1. Identify distinct entities (people, organizations, locations, products, technologies, concepts, events)
2. For each entity, provide: name, type, description, and key properties
3. Ensure entities are specific and meaningful (avoid generic terms)
4. Include confidence scores (0.0-1.0) for each entity
5. Group similar entities together

ENTITY TYPES: {entity_types}

TEXT TO ANALYZE:
{content}

RESPOND WITH VALID JSON:
{{
    "entities": [
        {{
            "name": "Entity Name",
            "type": "ENTITY_TYPE",
            "description": "Brief description",
            "properties": {{"key": "value"}},
            "confidence": 0.95,
            "mentions": ["mention1", "mention2"]
        }}
    ],
    "metadata": {{
        "total_entities": 0,
        "processing_notes": "Any relevant notes"
    }}
}}
""",
            
            PromptType.RELATIONSHIP_EXTRACTION: """
You are an expert knowledge graph analyst. Extract relationships between entities with high precision.

INSTRUCTIONS:
1. Identify meaningful relationships between the provided entities
2. For each relationship, specify: source entity, target entity, relationship type, description
3. Include confidence scores and contextual evidence
4. Focus on factual, verifiable relationships
5. Avoid redundant or trivial relationships

RELATIONSHIP TYPES: {relationship_types}

ENTITIES:
{entities}

TEXT TO ANALYZE:
{content}

RESPOND WITH VALID JSON:
{{
    "relationships": [
        {{
            "source_entity": "Entity A",
            "target_entity": "Entity B", 
            "relationship_type": "RELATIONSHIP_TYPE",
            "description": "Description of relationship",
            "confidence": 0.90,
            "evidence": "Text evidence for relationship",
            "properties": {{"strength": "high", "context": "business"}}
        }}
    ],
    "metadata": {{
        "total_relationships": 0,
        "processing_notes": "Any relevant notes"
    }}
}}
""",
            
            PromptType.CONTENT_CLASSIFICATION: """
You are an expert content classifier. Classify the following content into the most appropriate categories.

INSTRUCTIONS:
1. Analyze the content thoroughly
2. Assign the most relevant categories from the provided list
3. Include confidence scores for each classification
4. Provide reasoning for your classifications
5. Handle ambiguous content appropriately

AVAILABLE CATEGORIES: {categories}

CONTENT TO CLASSIFY:
{content}

RESPOND WITH VALID JSON:
{{
    "classifications": [
        {{
            "category": "Category Name",
            "confidence": 0.95,
            "reasoning": "Explanation for this classification"
        }}
    ],
    "primary_category": "Most relevant category",
    "metadata": {{
        "content_length": 0,
        "processing_notes": "Any relevant notes"
    }}
}}
""",
            
            PromptType.SUMMARIZATION: """
You are an expert content summarizer. Create a comprehensive yet concise summary of the following content.

INSTRUCTIONS:
1. Identify key themes, concepts, and important details
2. Maintain factual accuracy and objectivity
3. Structure the summary logically
4. Include key entities and relationships mentioned
5. Highlight critical insights or conclusions

CONTENT TO SUMMARIZE:
{content}

RESPOND WITH VALID JSON:
{{
    "summary": "Comprehensive summary of the content",
    "key_points": [
        "Important point 1",
        "Important point 2"
    ],
    "entities_mentioned": ["Entity1", "Entity2"],
    "themes": ["Theme1", "Theme2"],
    "word_count": 0,
    "metadata": {{
        "original_length": 0,
        "compression_ratio": 0.0,
        "processing_notes": "Any relevant notes"
    }}
}}
"""
        }
    
    async def extract_entities(
        self,
        content: str,
        config: Dict[str, Any] = None
    ) -> ProcessingResult:
        """Extract entities from content using sophisticated LLM analysis."""
        
        start_time = datetime.now(timezone.utc)
        
        try:
            # Prepare entity types for prompt
            entity_types = [et.value for et in EntityType]
            
            # Create LLM request
            prompt = self.prompts[PromptType.ENTITY_EXTRACTION].format(
                entity_types=", ".join(entity_types),
                content=content[:4000]  # Limit content length
            )
            
            llm_request = LLMRequest(
                prompt=prompt,
                content=content,
                prompt_type=PromptType.ENTITY_EXTRACTION,
                temperature=config.get("temperature", 0.1) if config else 0.1,
                max_tokens=config.get("max_tokens", 2000) if config else 2000
            )
            
            # Execute LLM call
            llm_response = await self._execute_llm_call(llm_request)
            
            # Parse and validate response
            entities_data = await self._parse_json_response(llm_response.content)
            validation_result = self.validators[PromptType.ENTITY_EXTRACTION](entities_data)
            
            if not validation_result.success:
                return validation_result
            
            # Calculate processing metrics
            processing_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            
            result = ProcessingResult(
                success=True,
                message=f"Extracted {len(entities_data.get('entities', []))} entities",
                data={
                    "entities": entities_data.get("entities", []),
                    "metadata": entities_data.get("metadata", {}),
                    "llm_response": llm_response.to_dict()
                },
                processing_time=processing_time
            )
            
            result.add_metric("entities_extracted", len(entities_data.get("entities", [])))
            result.add_metric("tokens_used", llm_response.total_tokens)
            result.add_metric("processing_time", processing_time)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Entity extraction error: {str(e)}", exc_info=True)
            return ProcessingResult(
                success=False,
                message=f"Entity extraction failed: {str(e)}",
                errors=[str(e)]
            )
    
    async def extract_relationships(
        self,
        content: str,
        entities: List[Dict[str, Any]],
        config: Dict[str, Any] = None
    ) -> ProcessingResult:
        """Extract relationships from content using sophisticated LLM analysis."""
        
        start_time = datetime.now(timezone.utc)
        
        try:
            # Prepare relationship types and entities for prompt
            relationship_types = [rt.value for rt in RelationshipType]
            entities_text = "\n".join([
                f"- {entity.get('name', 'Unknown')} ({entity.get('type', 'Unknown')}): {entity.get('description', '')}"
                for entity in entities
            ])
            
            # Create LLM request
            prompt = self.prompts[PromptType.RELATIONSHIP_EXTRACTION].format(
                relationship_types=", ".join(relationship_types),
                entities=entities_text,
                content=content[:3000]  # Leave room for entities and prompt
            )
            
            llm_request = LLMRequest(
                prompt=prompt,
                content=content,
                prompt_type=PromptType.RELATIONSHIP_EXTRACTION,
                temperature=config.get("temperature", 0.1) if config else 0.1,
                max_tokens=config.get("max_tokens", 2000) if config else 2000
            )
            
            # Execute LLM call
            llm_response = await self._execute_llm_call(llm_request)
            
            # Parse and validate response
            relationships_data = await self._parse_json_response(llm_response.content)
            validation_result = self.validators[PromptType.RELATIONSHIP_EXTRACTION](relationships_data)
            
            if not validation_result.success:
                return validation_result
            
            # Calculate processing metrics
            processing_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            
            result = ProcessingResult(
                success=True,
                message=f"Extracted {len(relationships_data.get('relationships', []))} relationships",
                data={
                    "relationships": relationships_data.get("relationships", []),
                    "metadata": relationships_data.get("metadata", {}),
                    "llm_response": llm_response.to_dict()
                },
                processing_time=processing_time
            )
            
            result.add_metric("relationships_extracted", len(relationships_data.get("relationships", [])))
            result.add_metric("tokens_used", llm_response.total_tokens)
            result.add_metric("processing_time", processing_time)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Relationship extraction error: {str(e)}", exc_info=True)
            return ProcessingResult(
                success=False,
                message=f"Relationship extraction failed: {str(e)}",
                errors=[str(e)]
            )
    
    async def classify_content(
        self,
        content: str,
        categories: List[str],
        config: Dict[str, Any] = None
    ) -> ProcessingResult:
        """Classify content into categories using sophisticated LLM analysis."""
        
        start_time = datetime.now(timezone.utc)
        
        try:
            # Create LLM request
            prompt = self.prompts[PromptType.CONTENT_CLASSIFICATION].format(
                categories=", ".join(categories),
                content=content[:4000]
            )
            
            llm_request = LLMRequest(
                prompt=prompt,
                content=content,
                prompt_type=PromptType.CONTENT_CLASSIFICATION,
                temperature=config.get("temperature", 0.1) if config else 0.1,
                max_tokens=config.get("max_tokens", 1000) if config else 1000
            )
            
            # Execute LLM call
            llm_response = await self._execute_llm_call(llm_request)
            
            # Parse and validate response
            classification_data = await self._parse_json_response(llm_response.content)
            validation_result = self.validators[PromptType.CONTENT_CLASSIFICATION](classification_data)
            
            if not validation_result.success:
                return validation_result
            
            # Calculate processing metrics
            processing_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            
            result = ProcessingResult(
                success=True,
                message=f"Content classified into {len(classification_data.get('classifications', []))} categories",
                data={
                    "classifications": classification_data.get("classifications", []),
                    "primary_category": classification_data.get("primary_category"),
                    "metadata": classification_data.get("metadata", {}),
                    "llm_response": llm_response.to_dict()
                },
                processing_time=processing_time
            )
            
            result.add_metric("categories_assigned", len(classification_data.get("classifications", [])))
            result.add_metric("tokens_used", llm_response.total_tokens)
            result.add_metric("processing_time", processing_time)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Content classification error: {str(e)}", exc_info=True)
            return ProcessingResult(
                success=False,
                message=f"Content classification failed: {str(e)}",
                errors=[str(e)]
            )
    
    async def summarize_content(
        self,
        content: str,
        config: Dict[str, Any] = None
    ) -> ProcessingResult:
        """Generate content summary using sophisticated LLM analysis."""
        
        start_time = datetime.now(timezone.utc)
        
        try:
            # Create LLM request
            prompt = self.prompts[PromptType.SUMMARIZATION].format(
                content=content[:4000]
            )
            
            llm_request = LLMRequest(
                prompt=prompt,
                content=content,
                prompt_type=PromptType.SUMMARIZATION,
                temperature=config.get("temperature", 0.3) if config else 0.3,
                max_tokens=config.get("max_tokens", 1500) if config else 1500
            )
            
            # Execute LLM call
            llm_response = await self._execute_llm_call(llm_request)
            
            # Parse and validate response
            summary_data = await self._parse_json_response(llm_response.content)
            validation_result = self.validators[PromptType.SUMMARIZATION](summary_data)
            
            if not validation_result.success:
                return validation_result
            
            # Calculate processing metrics
            processing_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            
            result = ProcessingResult(
                success=True,
                message="Content summary generated successfully",
                data={
                    "summary": summary_data.get("summary"),
                    "key_points": summary_data.get("key_points", []),
                    "entities_mentioned": summary_data.get("entities_mentioned", []),
                    "themes": summary_data.get("themes", []),
                    "metadata": summary_data.get("metadata", {}),
                    "llm_response": llm_response.to_dict()
                },
                processing_time=processing_time
            )
            
            result.add_metric("summary_length", len(summary_data.get("summary", "")))
            result.add_metric("key_points_count", len(summary_data.get("key_points", [])))
            result.add_metric("tokens_used", llm_response.total_tokens)
            result.add_metric("processing_time", processing_time)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Content summarization error: {str(e)}", exc_info=True)
            return ProcessingResult(
                success=False,
                message=f"Content summarization failed: {str(e)}",
                errors=[str(e)]
            )
    
    async def _execute_llm_call(self, request: LLMRequest) -> LLMResponse:
        """Execute LLM API call with error handling and retries."""
        
        start_time = datetime.now(timezone.utc)
        
        try:
            # Prepare messages
            messages = [
                {"role": "system", "content": "You are a helpful assistant that provides accurate, structured responses in JSON format."},
                {"role": "user", "content": request.prompt}
            ]
            
            # Execute API call
            response = await self.client.chat.completions.create(
                model=self.config.llm.model_name,
                messages=messages,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                response_format={"type": "json_object"} if self.config.llm.provider in [LLMProvider.OPENAI, LLMProvider.AZURE_OPENAI] else None
            )
            
            # Calculate processing time
            processing_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            
            # Create response object
            llm_response = LLMResponse(
                content=response.choices[0].message.content,
                prompt_tokens=response.usage.prompt_tokens if response.usage else 0,
                completion_tokens=response.usage.completion_tokens if response.usage else 0,
                total_tokens=response.usage.total_tokens if response.usage else 0,
                model=response.model,
                finish_reason=response.choices[0].finish_reason,
                processing_time=processing_time
            )
            
            self.logger.debug(f"LLM call completed: {llm_response.total_tokens} tokens, {processing_time:.2f}s")
            
            return llm_response
            
        except Exception as e:
            self.logger.error(f"LLM API call error: {str(e)}", exc_info=True)
            raise RuntimeError(f"LLM API call failed: {str(e)}")
    
    async def _parse_json_response(self, content: str) -> Dict[str, Any]:
        """Parse and validate JSON response from LLM."""
        
        try:
            # Clean up content (remove markdown code blocks if present)
            cleaned_content = content.strip()
            if cleaned_content.startswith("```json"):
                cleaned_content = cleaned_content[7:]
            if cleaned_content.endswith("```"):
                cleaned_content = cleaned_content[:-3]
            
            # Parse JSON
            parsed_data = json.loads(cleaned_content.strip())
            return parsed_data
            
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON parsing error: {str(e)}\nContent: {content[:500]}")
            raise ValueError(f"Invalid JSON response: {str(e)}")
    
    def _validate_entity_response(self, data: Dict[str, Any]) -> ProcessingResult:
        """Validate entity extraction response."""
        
        if "entities" not in data:
            return ProcessingResult(
                success=False,
                message="Missing 'entities' field in response",
                errors=["Response validation failed"]
            )
        
        entities = data["entities"]
        if not isinstance(entities, list):
            return ProcessingResult(
                success=False,
                message="'entities' must be a list",
                errors=["Response validation failed"]
            )
        
        # Validate each entity
        for i, entity in enumerate(entities):
            if not isinstance(entity, dict):
                return ProcessingResult(
                    success=False,
                    message=f"Entity {i} must be a dictionary",
                    errors=["Response validation failed"]
                )
            
            required_fields = ["name", "type"]
            for field in required_fields:
                if field not in entity:
                    return ProcessingResult(
                        success=False,
                        message=f"Entity {i} missing required field: {field}",
                        errors=["Response validation failed"]
                    )
        
        return ProcessingResult(success=True, message="Entity response validation passed")
    
    def _validate_relationship_response(self, data: Dict[str, Any]) -> ProcessingResult:
        """Validate relationship extraction response."""
        
        if "relationships" not in data:
            return ProcessingResult(
                success=False,
                message="Missing 'relationships' field in response",
                errors=["Response validation failed"]
            )
        
        relationships = data["relationships"]
        if not isinstance(relationships, list):
            return ProcessingResult(
                success=False,
                message="'relationships' must be a list",
                errors=["Response validation failed"]
            )
        
        # Validate each relationship
        for i, relationship in enumerate(relationships):
            if not isinstance(relationship, dict):
                return ProcessingResult(
                    success=False,
                    message=f"Relationship {i} must be a dictionary",
                    errors=["Response validation failed"]
                )
            
            required_fields = ["source_entity", "target_entity", "relationship_type"]
            for field in required_fields:
                if field not in relationship:
                    return ProcessingResult(
                        success=False,
                        message=f"Relationship {i} missing required field: {field}",
                        errors=["Response validation failed"]
                    )
        
        return ProcessingResult(success=True, message="Relationship response validation passed")
    
    def _validate_classification_response(self, data: Dict[str, Any]) -> ProcessingResult:
        """Validate classification response."""
        
        if "classifications" not in data:
            return ProcessingResult(
                success=False,
                message="Missing 'classifications' field in response",
                errors=["Response validation failed"]
            )
        
        return ProcessingResult(success=True, message="Classification response validation passed")
    
    def _validate_summary_response(self, data: Dict[str, Any]) -> ProcessingResult:
        """Validate summary response."""
        
        if "summary" not in data:
            return ProcessingResult(
                success=False,
                message="Missing 'summary' field in response",
                errors=["Response validation failed"]
            )
        
        return ProcessingResult(success=True, message="Summary response validation passed")


# Factory function for creating LLM service
def create_llm_service(config: GraphBuilderConfig) -> LLMServiceInterface:
    """Create LLM service based on configuration."""
    return AdvancedLLMService(config)