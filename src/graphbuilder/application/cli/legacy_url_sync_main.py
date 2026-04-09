"""
Legacy Url Sync Main - Migrated from main_url_sync.py

This module has been migrated to the new GraphBuilder enterprise structure.
Original functionality is preserved with improved organization and standards.

Migration Date: 2025-10-31
Original File: main_url_sync.py
New Location: src/graphbuilder/application/cli/legacy_url_sync_main.py
"""

import os
import json
import logging
from threading import Lock
from graphbuilder.infrastructure.database.neo4j_client import graphDBdataAccess
from graphbuilder.core.processing.processor import create_source_node_graph_dfrobot_url, extract_graph_from_web_page
from graphbuilder.core.utils.common_functions import create_graph_database_connection

# Setup logging
logging.basicConfig(level=logging.INFO)

# Global variables
visited_lock = Lock()
visited = set()
processed_urls = set()

# Settings
MAX_CRAWL_LIMIT = 200  # Limit the number of URLs to crawl
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



def main(urls, graph, model, allowed_nodes, allowed_relationship):
    for url in urls:
        process_url(graph, model, allowed_nodes, allowed_relationship, url)


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
    path ='visited_links.txt'
    with open(path, 'r') as file:
        all_urls = file.read().splitlines()
        urls = all_urls[:200]

    logging.info(f"Starting crawl with initial URLs from: {path}")

    main(urls, graph, model, allowedNodes, allowedRelationships)
    logging.info(f"Done Processing all the urls. Starting to save. visited: {visited}; processed: {processed_urls}")
    # Save visited and processed URLs when finished
    save_visited_and_processed()
