"""Web crawler service for extracting content from URLs."""

import json
import time
from queue import Queue
from threading import Lock
from typing import Set, List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urljoin, urlparse
import requests
from bs4 import BeautifulSoup

from config import config
from exceptions import CrawlerError
from logger_config import logger


class WebCrawlerService:
    """Service for web crawling and URL processing."""
    
    def __init__(self):
        self.visited_lock = Lock()
        self.visited: Set[str] = set()
        self.processed_urls: Set[str] = set()
        self.url_queue = Queue()
        self.session = requests.Session()
        
        # Configure session
        self.session.headers.update({
            'User-Agent': 'GraphBuilder/1.0 (+https://github.com/VincentPit/GraphBuilder)'
        })
        
        # Load existing data
        self._load_url_data()
    
    def _load_url_data(self) -> None:
        """Load visited and processed URLs from files."""
        try:
            visited_file = Path(config.crawler.visited_urls_file)
            if visited_file.exists():
                with open(visited_file, 'r', encoding='utf-8') as f:
                    self.visited = set(json.load(f))
                    logger.info(f"Loaded {len(self.visited)} visited URLs")
            
            processed_file = Path(config.crawler.processed_urls_file)
            if processed_file.exists():
                with open(processed_file, 'r', encoding='utf-8') as f:
                    self.processed_urls = set(json.load(f))
                    logger.info(f"Loaded {len(self.processed_urls)} processed URLs")
        
        except Exception as e:
            logger.error(f"Failed to load URL data: {e}")
            # Continue with empty sets
    
    def _save_url_data(self) -> None:
        """Save visited and processed URLs to files."""
        try:
            # Ensure directories exist
            visited_file = Path(config.crawler.visited_urls_file)
            processed_file = Path(config.crawler.processed_urls_file)
            
            visited_file.parent.mkdir(parents=True, exist_ok=True)
            processed_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Save visited URLs
            with open(visited_file, 'w', encoding='utf-8') as f:
                json.dump(list(self.visited), f, ensure_ascii=False, indent=2)
            
            # Save processed URLs
            with open(processed_file, 'w', encoding='utf-8') as f:
                json.dump(list(self.processed_urls), f, ensure_ascii=False, indent=2)
            
            logger.debug("URL data saved successfully")
        
        except Exception as e:
            logger.error(f"Failed to save URL data: {e}")
    
    def extract_links(self, url: str) -> List[str]:
        """
        Extract all links from a webpage.
        
        Args:
            url: URL to extract links from
            
        Returns:
            List of extracted URLs
        """
        try:
            logger.debug(f"Extracting links from: {url}")
            
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            links = []
            
            for a_tag in soup.find_all('a', href=True):
                href = a_tag.get('href')
                if href:
                    absolute_url = urljoin(url, href)
                    if self._is_valid_url(absolute_url):
                        links.append(absolute_url)
            
            logger.debug(f"Extracted {len(links)} links from {url}")
            return links
        
        except requests.exceptions.RequestException as e:
            logger.error(f"HTTP error extracting links from {url}: {e}")
            return []
        except Exception as e:
            logger.error(f"Error extracting links from {url}: {e}")
            return []
    
    def _is_valid_url(self, url: str) -> bool:
        """Check if URL is valid and should be processed."""
        try:
            parsed = urlparse(url)
            
            # Must have scheme and netloc
            if not parsed.scheme or not parsed.netloc:
                return False
            
            # Check allowed domains
            domain = parsed.netloc.lower()
            if config.crawler.allowed_domains:
                return any(allowed_domain in domain for allowed_domain in config.crawler.allowed_domains)
            
            return True
        
        except Exception:
            return False
    
    def should_process_url(self, url: str) -> bool:
        """Check if URL should be processed."""
        with self.visited_lock:
            # Skip if already visited or processed
            if url in self.visited or url in self.processed_urls:
                return False
            
            # Check crawl limit
            if len(self.processed_urls) >= config.crawler.max_crawl_limit:
                logger.info(f"Crawl limit reached: {config.crawler.max_crawl_limit}")
                return False
            
            # Check if URL matches allowed domains
            if not self._is_valid_url(url):
                logger.debug(f"URL not valid for processing: {url}")
                return False
            
            return True
    
    def mark_url_visited(self, url: str) -> None:
        """Mark URL as visited."""
        with self.visited_lock:
            self.visited.add(url)
    
    def mark_url_processed(self, url: str) -> None:
        """Mark URL as processed."""
        with self.visited_lock:
            self.processed_urls.add(url)
        self._save_url_data()
    
    def add_urls_to_queue(self, urls: List[str]) -> int:
        """Add URLs to processing queue."""
        added_count = 0
        
        for url in urls:
            if self.should_process_url(url):
                self.url_queue.put(url)
                added_count += 1
        
        return added_count
    
    def crawl_urls_parallel(
        self,
        start_urls: List[str],
        process_callback: callable,
        max_workers: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Crawl URLs in parallel.
        
        Args:
            start_urls: Initial URLs to start crawling from
            process_callback: Function to process each URL
            max_workers: Maximum number of worker threads
            
        Returns:
            Dictionary with crawling statistics
        """
        max_workers = max_workers or config.crawler.max_workers
        
        logger.info(f"Starting parallel crawling with {max_workers} workers")
        
        # Add start URLs to queue
        initial_added = self.add_urls_to_queue(start_urls)
        logger.info(f"Added {initial_added} initial URLs to queue")
        
        stats = {
            'processed_count': 0,
            'failed_count': 0,
            'skipped_count': 0
        }
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Process URLs from queue
            while not self.url_queue.empty() and len(self.processed_urls) < config.crawler.max_crawl_limit:
                # Get batch of URLs
                batch_urls = []
                batch_size = min(max_workers, self.url_queue.qsize())
                
                for _ in range(batch_size):
                    if not self.url_queue.empty():
                        batch_urls.append(self.url_queue.get())
                
                if not batch_urls:
                    break
                
                # Submit processing tasks
                future_to_url = {
                    executor.submit(self._process_url_with_callback, url, process_callback): url
                    for url in batch_urls
                }
                
                # Collect results and extract new links
                for future in as_completed(future_to_url):
                    url = future_to_url[future]
                    
                    try:
                        success = future.result()
                        
                        if success:
                            stats['processed_count'] += 1
                            
                            # Extract new links and add to queue
                            if len(self.processed_urls) < config.crawler.max_crawl_limit:
                                new_links = self.extract_links(url)
                                added_count = self.add_urls_to_queue(new_links)
                                if added_count > 0:
                                    logger.debug(f"Added {added_count} new URLs from {url}")
                        else:
                            stats['failed_count'] += 1
                    
                    except Exception as e:
                        logger.error(f"Error processing URL {url}: {e}")
                        stats['failed_count'] += 1
                
                # Rate limiting
                if config.crawler.delay_between_requests > 0:
                    time.sleep(config.crawler.delay_between_requests)
        
        logger.info(f"Crawling completed. Stats: {stats}")
        return stats
    
    def _process_url_with_callback(self, url: str, process_callback: callable) -> bool:
        """Process URL with callback function."""
        try:
            if not self.should_process_url(url):
                return False
            
            self.mark_url_visited(url)
            logger.info(f"Processing URL: {url}")
            
            # Call the processing callback
            result = process_callback(url)
            
            self.mark_url_processed(url)
            logger.info(f"Successfully processed: {url}")
            
            return True
        
        except Exception as e:
            logger.error(f"Failed to process URL {url}: {e}")
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get crawling statistics."""
        return {
            'visited_count': len(self.visited),
            'processed_count': len(self.processed_urls),
            'queue_size': self.url_queue.qsize(),
            'crawl_limit': config.crawler.max_crawl_limit
        }
    
    def reset(self) -> None:
        """Reset crawler state."""
        with self.visited_lock:
            self.visited.clear()
            self.processed_urls.clear()
            
            # Clear queue
            while not self.url_queue.empty():
                try:
                    self.url_queue.get_nowait()
                except:
                    break
        
        self._save_url_data()
        logger.info("Crawler state reset")


# Global crawler service instance
crawler_service = WebCrawlerService()