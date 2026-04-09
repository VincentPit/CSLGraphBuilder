"""
Legacy Url Main - Migrated from main_url.py

This module has been migrated to the new GraphBuilder enterprise structure.
Original functionality is preserved with improved organization and standards.

Migration Date: 2025-10-31
Original File: main_url.py
New Location: src/graphbuilder/application/cli/legacy_url_main.py
"""

import os
import json
import logging
import time
from queue import Queue
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from graphbuilder.infrastructure.database.neo4j_client import graphDBdataAccess
from graphbuilder.core.processing.processor import create_source_node_graph_dfrobot_url, extract_graph_from_web_page
from graphbuilder.core.utils.common_functions import create_graph_database_connection
from concurrent.futures import wait

# Setup logging
logging.basicConfig(level=logging.INFO)

# Global variables
visited_lock = Lock()
visited = set()
processed_urls = set()
queue = Queue()

# Settings
MAX_CRAWL_LIMIT = 2  # Limit the number of URLs to crawl
VISITED_FILE = 'record/visited_urls.json'
PROCESSED_FILE = 'record/processed_urls.json'

# Load visited and processed URLs from file
def load_visited_and_processed():
    global visited, processed_urls
    if os.path.exists(VISITED_FILE):
        with open(VISITED_FILE, 'r') as f:
            visited = set(json.load(f))
    if os.path.exists(PROCESSED_FILE):
        with open(PROCESSED_FILE, 'r') as f:
            processed_urls = set(json.load(f))

# Save visited and processed URLs to file
def save_visited_and_processed():
    with open(VISITED_FILE, 'w') as f:
        json.dump(list(visited), f)
    with open(PROCESSED_FILE, 'w') as f:
        json.dump(list(processed_urls), f)

def extract_links(url):
    """Extract all links from the page."""
    try:
        response = requests.get(url)
        response.raise_for_status()
        html = response.text
        soup = BeautifulSoup(html, 'html.parser')
        return [urljoin(url, a.get('href')) for a in soup.find_all('a') if a.get('href')]
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching {url}: {e}")
        return []

def process_url(graph, model, allowed_nodes, allowed_relationship, url):
    """Crawl and process a single URL."""
    global visited, processed_urls

    
    if url in visited or url in processed_urls:
        return
    visited.add(url)

    if 'dfrobot' not in url:
        logging.info(f"Skipping URL without keyword: {url}")
        return

    logging.info(f"Processing: {url}")

    # First, process the source node graph for this URL
    lst_file, success_count, fail_count = create_source_node_graph_dfrobot_url(graph, model, url, "dfrobot")
    logging.info(f"Processed source node for {url}: Success: {success_count}, Failures: {fail_count}")

    # Then, extract the graph from the page
    result_dic = extract_graph_from_web_page(graph, model, url, allowed_nodes, allowed_relationship)
    logging.info(f"Extracted graph data from {url}: {result_dic}")

    # Add to processed URLs
    #with visited_lock:
    processed_urls.add(url)

    # Save after processing each URL to avoid losing progress
    save_visited_and_processed()

    # Extract and queue the new links for crawling
    links = extract_links(url)
    for link in links:
        if len(processed_urls) < MAX_CRAWL_LIMIT:
            queue.put(link)  # Add the new link to the queue
        else:
            logging.info(f"URL limit reached: {MAX_CRAWL_LIMIT}, here are the processed  {processed_urls}")

def crawl_urls_in_parallel(graph, model, allowed_nodes, allowed_relationship, delay=1, max_workers=5):
    """Crawl URLs in parallel and process them."""
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        while not queue.empty() and len(processed_urls) < MAX_CRAWL_LIMIT:
            url = queue.get()
            future = executor.submit(process_url, graph, model, allowed_nodes, allowed_relationship, url)
            futures.append(future)
            time.sleep(delay)  # Be nice to the server
        
        # Wait for all submitted tasks to finish
        wait(futures)
        
        if len(processed_urls) >= MAX_CRAWL_LIMIT:
            logging.info(f"URL limit reached: {MAX_CRAWL_LIMIT}, here are the processed {processed_urls}")
        else:
            logging.info("Queue is empty")

def main(start_urls, graph, model, allowed_nodes, allowed_relationship):
    # Enqueue all the starting URLs
    for url in start_urls:
        queue.put(url)

    # Start crawling and processing URLs in parallel
    crawl_urls_in_parallel(graph, model, allowed_nodes, allowed_relationship)

if __name__ == "__main__":
    # To Be Modified
    uri = "bolt://localhost:7687"
    userName = "neo4j"
    password = "neo4j"
    database = "neo4j"
    model = "azure_ai_gpt_4o"
    allowedNodes = "controller,sensor,board,actor,module,software"
    allowedRelationships = "兼容"
    
    graph = create_graph_database_connection(uri, userName, password, database)
    graphDb_data_Access = graphDBdataAccess(graph)

    start_urls = [
        "https://www.dfrobot.com.cn/", 
        "https://makelog.dfrobot.com.cn/", 
        "https://mc.dfrobot.com.cn/", 
        "https://www.dfrobot.com.cn/index.php", 
        "https://wiki.dfrobot.com.cn/"
    ]

    logging.info(f"Starting crawl with initial URLs: {start_urls}")

    # Load visited and processed URLs from files
    load_visited_and_processed()

    main(start_urls, graph, model, allowedNodes, allowedRelationships)
    logging.info(f"Done Crawling. Starting to save. visited: {visited}; processed: {processed_urls}")
    # Save visited and processed URLs when finished
    save_visited_and_processed()
