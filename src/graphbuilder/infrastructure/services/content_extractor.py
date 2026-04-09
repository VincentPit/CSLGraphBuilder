"""
Content Extractor Service - Sophisticated content extraction from multiple sources.

This module provides enterprise-grade content extraction capabilities
with support for multiple formats, intelligent preprocessing, and metadata extraction.
"""

import asyncio
import logging
import re
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timezone
from abc import ABC, abstractmethod
from urllib.parse import urljoin, urlparse
import hashlib

from ...domain.models.processing_models import ProcessingResult, ContentType
from ..config.settings import GraphBuilderConfig


class ContentExtractorInterface(ABC):
    """Abstract interface for content extraction operations."""
    
    @abstractmethod
    async def extract_from_url(self, url: str, config: Dict[str, Any] = None) -> ProcessingResult:
        """Extract content from URL."""
        pass
    
    @abstractmethod
    async def extract_from_file(self, file_path: str, config: Dict[str, Any] = None) -> ProcessingResult:
        """Extract content from file."""
        pass
    
    @abstractmethod
    async def extract_from_text(self, text: str, config: Dict[str, Any] = None) -> ProcessingResult:
        """Process raw text content."""
        pass


class AdvancedContentExtractorService(ContentExtractorInterface):
    """
    Sophisticated content extractor with multi-format support and intelligent processing.
    
    Provides enterprise-grade content extraction with advanced features for
    web scraping, file processing, text cleaning, and metadata extraction.
    """
    
    def __init__(self, config: GraphBuilderConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize HTTP session for web requests
        self.session = self._create_http_session()
        
        # Content processors for different formats
        self.processors = {
            ContentType.HTML: self._process_html_content,
            ContentType.PLAIN_TEXT: self._process_text_content,
            ContentType.MARKDOWN: self._process_markdown_content,
            ContentType.JSON: self._process_json_content,
            ContentType.PDF: self._process_pdf_content,
            ContentType.XML: self._process_xml_content,
            ContentType.CSV: self._process_csv_content,
            ContentType.DOCX: self._process_docx_content
        }
    
    def _create_http_session(self):
        """Create HTTP session with appropriate headers and settings."""
        
        try:
            import aiohttp
            
            headers = {
                'User-Agent': self.config.crawler.user_agent,
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1'
            }
            
            timeout = aiohttp.ClientTimeout(total=self.config.crawler.timeout)
            
            return aiohttp.ClientSession(
                headers=headers,
                timeout=timeout,
                connector=aiohttp.TCPConnector(limit=self.config.crawler.max_concurrent_requests)
            )
            
        except ImportError:
            self.logger.warning("aiohttp not available, web extraction will be limited")
            return None
    
    async def extract_from_url(self, url: str, config: Dict[str, Any] = None) -> ProcessingResult:
        """Extract content from URL with sophisticated web scraping."""
        
        start_time = datetime.now(timezone.utc)
        
        try:
            if not self.session:
                return ProcessingResult(
                    success=False,
                    message="HTTP session not available",
                    errors=["aiohttp not installed"]
                )
            
            self.logger.info(f"Extracting content from URL: {url}")
            
            # Validate URL
            parsed_url = urlparse(url)
            if not parsed_url.scheme or not parsed_url.netloc:
                return ProcessingResult(
                    success=False,
                    message="Invalid URL format",
                    errors=[f"Invalid URL: {url}"]
                )
            
            # Fetch content
            async with self.session.get(url) as response:
                if response.status != 200:
                    return ProcessingResult(
                        success=False,
                        message=f"HTTP error: {response.status}",
                        errors=[f"Failed to fetch URL: {response.status} {response.reason}"]
                    )
                
                # Get content and metadata
                raw_content = await response.text()
                content_type = response.headers.get('Content-Type', 'text/html')
                content_length = len(raw_content)
                
                # Extract base content type
                base_content_type = self._normalize_content_type(content_type)
                
                # Process content based on type
                if base_content_type in self.processors:
                    processing_result = await self.processors[base_content_type](
                        raw_content, config or {}
                    )
                else:
                    # Default to text processing
                    processing_result = await self._process_text_content(
                        raw_content, config or {}
                    )
                
                if not processing_result.success:
                    return processing_result
                
                # Calculate metrics
                processing_time = (datetime.now(timezone.utc) - start_time).total_seconds()
                
                # Create comprehensive result
                result = ProcessingResult(
                    success=True,
                    message=f"Successfully extracted content from {url}",
                    data={
                        "content": processing_result.data.get("content", ""),
                        "title": processing_result.data.get("title", ""),
                        "metadata": {
                            "url": url,
                            "content_type": content_type,
                            "base_content_type": base_content_type.value,
                            "content_length": content_length,
                            "processed_length": len(processing_result.data.get("content", "")),
                            "response_headers": dict(response.headers),
                            "extraction_metadata": processing_result.data.get("metadata", {}),
                            "processing_time": processing_time
                        }
                    },
                    processing_time=processing_time
                )
                
                # Add metrics
                result.add_metric("content_length", content_length)
                result.add_metric("processed_length", len(processing_result.data.get("content", "")))
                result.add_metric("compression_ratio", len(processing_result.data.get("content", "")) / content_length if content_length > 0 else 0)
                result.add_metric("processing_time", processing_time)
                
                return result
                
        except Exception as e:
            self.logger.error(f"URL extraction error: {str(e)}", exc_info=True)
            return ProcessingResult(
                success=False,
                message=f"URL extraction failed: {str(e)}",
                errors=[str(e)]
            )
    
    async def extract_from_file(self, file_path: str, config: Dict[str, Any] = None) -> ProcessingResult:
        """Extract content from file with format-specific processing."""
        
        start_time = datetime.now(timezone.utc)
        
        try:
            import os
            from pathlib import Path
            
            self.logger.info(f"Extracting content from file: {file_path}")
            
            # Validate file
            if not os.path.exists(file_path):
                return ProcessingResult(
                    success=False,
                    message="File not found",
                    errors=[f"File does not exist: {file_path}"]
                )
            
            # Get file info
            file_info = os.stat(file_path)
            file_size = file_info.st_size
            file_extension = Path(file_path).suffix.lower()
            
            # Determine content type from extension
            content_type = self._get_content_type_from_extension(file_extension)
            
            # Read file content
            if content_type in [ContentType.PDF, ContentType.DOCX]:
                # Binary file processing
                raw_content = await self._read_binary_file(file_path)
            else:
                # Text file processing
                raw_content = await self._read_text_file(file_path)
            
            # Process content based on type
            if content_type in self.processors:
                processing_result = await self.processors[content_type](
                    raw_content, config or {}
                )
            else:
                # Default to text processing
                processing_result = await self._process_text_content(
                    raw_content, config or {}
                )
            
            if not processing_result.success:
                return processing_result
            
            # Calculate metrics
            processing_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            
            # Create comprehensive result
            result = ProcessingResult(
                success=True,
                message=f"Successfully extracted content from {file_path}",
                data={
                    "content": processing_result.data.get("content", ""),
                    "title": processing_result.data.get("title", Path(file_path).stem),
                    "metadata": {
                        "file_path": file_path,
                        "file_size": file_size,
                        "file_extension": file_extension,
                        "content_type": content_type.value,
                        "processed_length": len(processing_result.data.get("content", "")),
                        "extraction_metadata": processing_result.data.get("metadata", {}),
                        "processing_time": processing_time
                    }
                },
                processing_time=processing_time
            )
            
            # Add metrics
            result.add_metric("file_size", file_size)
            result.add_metric("processed_length", len(processing_result.data.get("content", "")))
            result.add_metric("compression_ratio", len(processing_result.data.get("content", "")) / file_size if file_size > 0 else 0)
            result.add_metric("processing_time", processing_time)
            
            return result
            
        except Exception as e:
            self.logger.error(f"File extraction error: {str(e)}", exc_info=True)
            return ProcessingResult(
                success=False,
                message=f"File extraction failed: {str(e)}",
                errors=[str(e)]
            )
    
    async def extract_from_text(self, text: str, config: Dict[str, Any] = None) -> ProcessingResult:
        """Process raw text content with intelligent preprocessing."""
        
        start_time = datetime.now(timezone.utc)
        
        try:
            # Process text content
            processing_result = await self._process_text_content(text, config or {})
            
            if not processing_result.success:
                return processing_result
            
            # Calculate metrics
            processing_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            
            result = ProcessingResult(
                success=True,
                message="Successfully processed text content",
                data={
                    "content": processing_result.data.get("content", ""),
                    "title": processing_result.data.get("title", "Text Content"),
                    "metadata": {
                        "original_length": len(text),
                        "processed_length": len(processing_result.data.get("content", "")),
                        "extraction_metadata": processing_result.data.get("metadata", {}),
                        "processing_time": processing_time
                    }
                },
                processing_time=processing_time
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Text processing error: {str(e)}", exc_info=True)
            return ProcessingResult(
                success=False,
                message=f"Text processing failed: {str(e)}",
                errors=[str(e)]
            )
    
    async def _process_html_content(self, html: str, config: Dict[str, Any]) -> ProcessingResult:
        """Process HTML content with sophisticated extraction."""
        
        try:
            from bs4 import BeautifulSoup
            
            # Parse HTML
            soup = BeautifulSoup(html, 'html.parser')
            
            # Extract title
            title = ""
            title_tag = soup.find('title')
            if title_tag:
                title = title_tag.get_text().strip()
            
            # Remove unwanted elements
            for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside']):
                element.decompose()
            
            # Extract main content
            main_content = ""
            
            # Try to find main content area
            content_selectors = [
                'main', 'article', '[role="main"]', '.main-content',
                '.content', '.post-content', '.entry-content'
            ]
            
            for selector in content_selectors:
                content_element = soup.select_one(selector)
                if content_element:
                    main_content = content_element.get_text()
                    break
            
            # Fallback to body content
            if not main_content:
                body = soup.find('body')
                if body:
                    main_content = body.get_text()
                else:
                    main_content = soup.get_text()
            
            # Clean up text
            cleaned_content = self._clean_text(main_content)
            
            # Extract metadata
            metadata = {
                "title_length": len(title),
                "original_html_length": len(html),
                "cleaned_content_length": len(cleaned_content),
                "extraction_method": "html_parsing"
            }
            
            return ProcessingResult(
                success=True,
                message="HTML content processed successfully",
                data={
                    "content": cleaned_content,
                    "title": title,
                    "metadata": metadata
                }
            )
            
        except ImportError:
            self.logger.warning("BeautifulSoup not available, using basic HTML processing")
            # Fallback to regex-based processing
            return await self._process_html_basic(html, config)
        except Exception as e:
            self.logger.error(f"HTML processing error: {str(e)}")
            return ProcessingResult(
                success=False,
                message=f"HTML processing failed: {str(e)}",
                errors=[str(e)]
            )
    
    async def _process_html_basic(self, html: str, config: Dict[str, Any]) -> ProcessingResult:
        """Basic HTML processing using regex (fallback)."""
        
        # Extract title
        title_match = re.search(r'<title[^>]*>(.*?)</title>', html, re.IGNORECASE | re.DOTALL)
        title = title_match.group(1).strip() if title_match else ""
        
        # Remove script and style tags
        html_clean = re.sub(r'<(script|style)[^>]*>.*?</\1>', '', html, flags=re.IGNORECASE | re.DOTALL)
        
        # Remove HTML tags
        text_content = re.sub(r'<[^>]+>', ' ', html_clean)
        
        # Clean up text
        cleaned_content = self._clean_text(text_content)
        
        metadata = {
            "title_length": len(title),
            "original_html_length": len(html),
            "cleaned_content_length": len(cleaned_content),
            "extraction_method": "regex_parsing"
        }
        
        return ProcessingResult(
            success=True,
            message="HTML content processed with basic method",
            data={
                "content": cleaned_content,
                "title": title,
                "metadata": metadata
            }
        )
    
    async def _process_text_content(self, text: str, config: Dict[str, Any]) -> ProcessingResult:
        """Process plain text content with intelligent cleaning."""
        
        # Clean text
        cleaned_content = self._clean_text(text)
        
        # Extract potential title (first line if it looks like a title)
        lines = cleaned_content.split('\n')
        title = ""
        
        if lines:
            first_line = lines[0].strip()
            if len(first_line) < 100 and not first_line.endswith('.'):
                title = first_line
                # Remove title from content
                cleaned_content = '\n'.join(lines[1:]).strip()
        
        metadata = {
            "original_length": len(text),
            "cleaned_length": len(cleaned_content),
            "line_count": len(lines),
            "extraction_method": "text_processing"
        }
        
        return ProcessingResult(
            success=True,
            message="Text content processed successfully",
            data={
                "content": cleaned_content,
                "title": title,
                "metadata": metadata
            }
        )
    
    async def _process_markdown_content(self, markdown: str, config: Dict[str, Any]) -> ProcessingResult:
        """Process Markdown content."""
        
        # Extract title from first heading
        title_match = re.search(r'^#\s+(.+)$', markdown, re.MULTILINE)
        title = title_match.group(1).strip() if title_match else ""
        
        # Remove markdown syntax for plain text extraction
        text_content = re.sub(r'[#*_`\[\]()]+', '', markdown)
        
        # Clean text
        cleaned_content = self._clean_text(text_content)
        
        metadata = {
            "original_length": len(markdown),
            "cleaned_length": len(cleaned_content),
            "title_length": len(title),
            "extraction_method": "markdown_processing"
        }
        
        return ProcessingResult(
            success=True,
            message="Markdown content processed successfully",
            data={
                "content": cleaned_content,
                "title": title,
                "metadata": metadata
            }
        )
    
    async def _process_json_content(self, json_str: str, config: Dict[str, Any]) -> ProcessingResult:
        """Process JSON content."""
        
        import json
        
        try:
            data = json.loads(json_str)
            
            # Extract text content from JSON
            text_content = self._extract_text_from_json(data)
            
            # Clean text
            cleaned_content = self._clean_text(text_content)
            
            # Try to find title
            title = ""
            if isinstance(data, dict):
                for key in ['title', 'name', 'subject', 'heading']:
                    if key in data and isinstance(data[key], str):
                        title = data[key]
                        break
            
            metadata = {
                "original_length": len(json_str),
                "cleaned_length": len(cleaned_content),
                "json_type": type(data).__name__,
                "extraction_method": "json_processing"
            }
            
            return ProcessingResult(
                success=True,
                message="JSON content processed successfully",
                data={
                    "content": cleaned_content,
                    "title": title,
                    "metadata": metadata
                }
            )
            
        except json.JSONDecodeError as e:
            return ProcessingResult(
                success=False,
                message=f"Invalid JSON: {str(e)}",
                errors=[str(e)]
            )
    
    async def _process_pdf_content(self, pdf_data: bytes, config: Dict[str, Any]) -> ProcessingResult:
        """Process PDF content (placeholder - requires PyPDF2 or similar)."""
        
        return ProcessingResult(
            success=False,
            message="PDF processing not implemented",
            errors=["PDF processing requires additional dependencies"]
        )
    
    async def _process_xml_content(self, xml: str, config: Dict[str, Any]) -> ProcessingResult:
        """Process XML content."""
        
        try:
            import xml.etree.ElementTree as ET
            
            root = ET.fromstring(xml)
            
            # Extract text content
            text_content = ET.tostring(root, encoding='unicode', method='text')
            
            # Clean text
            cleaned_content = self._clean_text(text_content)
            
            # Try to find title
            title = ""
            title_element = root.find('.//title') or root.find('.//name')
            if title_element is not None:
                title = title_element.text or ""
            
            metadata = {
                "original_length": len(xml),
                "cleaned_length": len(cleaned_content),
                "root_tag": root.tag,
                "extraction_method": "xml_processing"
            }
            
            return ProcessingResult(
                success=True,
                message="XML content processed successfully",
                data={
                    "content": cleaned_content,
                    "title": title,
                    "metadata": metadata
                }
            )
            
        except ET.ParseError as e:
            return ProcessingResult(
                success=False,
                message=f"Invalid XML: {str(e)}",
                errors=[str(e)]
            )
    
    async def _process_csv_content(self, csv_str: str, config: Dict[str, Any]) -> ProcessingResult:
        """Process CSV content."""
        
        import csv
        from io import StringIO
        
        try:
            reader = csv.reader(StringIO(csv_str))
            rows = list(reader)
            
            # Convert to text
            text_lines = []
            for row in rows:
                text_lines.append(' | '.join(row))
            
            text_content = '\n'.join(text_lines)
            
            # Clean text
            cleaned_content = self._clean_text(text_content)
            
            # Use first row as title if it looks like headers
            title = ""
            if rows and len(rows[0]) > 0:
                title = "CSV Data: " + " | ".join(rows[0][:3])
            
            metadata = {
                "original_length": len(csv_str),
                "cleaned_length": len(cleaned_content),
                "row_count": len(rows),
                "column_count": len(rows[0]) if rows else 0,
                "extraction_method": "csv_processing"
            }
            
            return ProcessingResult(
                success=True,
                message="CSV content processed successfully",
                data={
                    "content": cleaned_content,
                    "title": title,
                    "metadata": metadata
                }
            )
            
        except Exception as e:
            return ProcessingResult(
                success=False,
                message=f"CSV processing error: {str(e)}",
                errors=[str(e)]
            )
    
    async def _process_docx_content(self, docx_data: bytes, config: Dict[str, Any]) -> ProcessingResult:
        """Process DOCX content (placeholder - requires python-docx)."""
        
        return ProcessingResult(
            success=False,
            message="DOCX processing not implemented",
            errors=["DOCX processing requires additional dependencies"]
        )
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text content."""
        
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove control characters
        text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
        
        # Normalize line breaks
        text = re.sub(r'\r\n|\r|\n', '\n', text)
        
        # Remove excessive line breaks
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        
        # Strip whitespace
        text = text.strip()
        
        return text
    
    def _normalize_content_type(self, content_type: str) -> ContentType:
        """Normalize content type string to ContentType enum."""
        
        content_type_lower = content_type.lower()
        
        if 'html' in content_type_lower:
            return ContentType.HTML
        elif 'json' in content_type_lower:
            return ContentType.JSON
        elif 'xml' in content_type_lower:
            return ContentType.XML
        elif 'pdf' in content_type_lower:
            return ContentType.PDF
        elif 'csv' in content_type_lower:
            return ContentType.CSV
        elif 'markdown' in content_type_lower:
            return ContentType.MARKDOWN
        else:
            return ContentType.PLAIN_TEXT
    
    def _get_content_type_from_extension(self, extension: str) -> ContentType:
        """Get content type from file extension."""
        
        extension_map = {
            '.html': ContentType.HTML,
            '.htm': ContentType.HTML,
            '.txt': ContentType.PLAIN_TEXT,
            '.md': ContentType.MARKDOWN,
            '.json': ContentType.JSON,
            '.pdf': ContentType.PDF,
            '.xml': ContentType.XML,
            '.csv': ContentType.CSV,
            '.docx': ContentType.DOCX
        }
        
        return extension_map.get(extension, ContentType.PLAIN_TEXT)
    
    async def _read_text_file(self, file_path: str) -> str:
        """Read text file with encoding detection."""
        
        import aiofiles
        
        # Try different encodings
        encodings = ['utf-8', 'utf-16', 'latin-1', 'cp1252']
        
        for encoding in encodings:
            try:
                async with aiofiles.open(file_path, 'r', encoding=encoding) as f:
                    return await f.read()
            except UnicodeDecodeError:
                continue
        
        # Fallback to binary read with error handling
        async with aiofiles.open(file_path, 'rb') as f:
            content = await f.read()
            return content.decode('utf-8', errors='ignore')
    
    async def _read_binary_file(self, file_path: str) -> bytes:
        """Read binary file."""
        
        import aiofiles
        
        async with aiofiles.open(file_path, 'rb') as f:
            return await f.read()
    
    def _extract_text_from_json(self, data: Any, max_depth: int = 5) -> str:
        """Recursively extract text content from JSON data."""
        
        if max_depth <= 0:
            return ""
        
        if isinstance(data, str):
            return data + " "
        elif isinstance(data, (int, float, bool)):
            return str(data) + " "
        elif isinstance(data, list):
            return "".join([self._extract_text_from_json(item, max_depth - 1) for item in data])
        elif isinstance(data, dict):
            return "".join([self._extract_text_from_json(value, max_depth - 1) for value in data.values()])
        else:
            return ""


# Factory function for creating content extractor service
def create_content_extractor_service(config: GraphBuilderConfig) -> ContentExtractorInterface:
    """Create content extractor service based on configuration."""
    return AdvancedContentExtractorService(config)