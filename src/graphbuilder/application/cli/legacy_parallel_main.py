"""
Legacy Parallel Main - Migrated from main_para.py

This module has been migrated to the new GraphBuilder enterprise structure.
Original functionality is preserved with improved organization and standards.

Migration Date: 2025-10-31
Original File: main_para.py
New Location: src/graphbuilder/application/cli/legacy_parallel_main.py
"""

from multiprocessing import Manager, Pool
import logging
import json
import os
from graphbuilder.infrastructure.database.neo4j_client import graphDBdataAccess
from graphbuilder.core.processing.processor import create_source_node_graph_dfrobot_url, extract_graph_from_web_page
from graphbuilder.core.utils.common_functions import create_graph_database_connection

# Setup logging
logging.basicConfig(level=logging.INFO)

# Global variables
VISITED_FILE = 'record/visited_urls.json'
PROCESSED_FILE = 'record/processed_urls.json'

# Settings
MAX_CRAWL_LIMIT = 200  # Limit the number of URLs to crawl

# Load visited and processed URLs from file
def load_visited_and_processed():
    visited = set()
    processed_urls = set()
    if os.path.exists(VISITED_FILE):
        with open(VISITED_FILE, 'r') as f:
            visited = set(json.load(f))
    if os.path.exists(PROCESSED_FILE):
        with open(PROCESSED_FILE, 'r') as f:
            processed_urls = set(json.load(f))
    return visited, processed_urls

# Save visited and processed URLs to file
def save_visited_and_processed(visited, processed_urls):
    with open(VISITED_FILE, 'w') as f:
        json.dump(list(visited), f)
    with open(PROCESSED_FILE, 'w') as f:
        json.dump(list(processed_urls), f)

def process_url(model, allowed_nodes, allowed_relationship, url, visited, processed_urls):
    """Crawl and process a single URL."""
    if url in visited or url in processed_urls:
        return
    visited.add(url)

    if 'dfrobot' not in url:
        logging.info(f"Skipping URL without keyword: {url}")
        return

    logging.info(f"Processing: {url}")
    
    # Create a new graph connection in the worker
    uri = "bolt://localhost:7687"
    userName = "neo4j"
    password = "neo4j"
    database = "neo4j"
    graph = create_graph_database_connection(uri, userName, password, database)
    graphDb_data_Access = graphDBdataAccess(graph)

    # First, process the source node graph for this URL
    lst_file, success_count, fail_count = create_source_node_graph_dfrobot_url(graph, model, url, "dfrobot")
    logging.info(f"Processed source node for {url}: Success: {success_count}, Failures: {fail_count}")

    # Then, extract the graph from the page
    result_dic = extract_graph_from_web_page(graph, model, url, allowed_nodes, allowed_relationship)
    logging.info(f"Extracted graph data from {url}: {result_dic}")

    # Add to processed URLs
    processed_urls.add(url)

    # Save visited and processed after processing each URL to avoid losing progress
    save_visited_and_processed(visited, processed_urls)

def worker(model, allowed_nodes, allowed_relationship, urls_chunk, visited, processed_urls):
    """Worker function to process a chunk of URLs."""
    for url in urls_chunk:
        process_url(model, allowed_nodes, allowed_relationship, url, visited, processed_urls)

def main(urls, model, allowed_nodes, allowed_relationship):
    # Split the URLs into chunks for parallel processing
    num_workers = 50
    
    
    chunk_size = len(urls) // num_workers
    url_chunks = [urls[i:i + chunk_size] for i in range(0, len(urls), chunk_size)]

    # Create a multiprocessing Manager to share visited and processed sets
    with Manager() as manager:
        visited = set(manager.list()) 
        processed_urls = set(manager.list())  

        # Load previously visited and processed URLs if they exist
        loaded_visited, loaded_processed = load_visited_and_processed()
        visited.update(loaded_visited)  # Use update to add elements to a set
        processed_urls.update(loaded_processed)  # Use update to add elements to a set

        # Create a pool of workers
        with Pool(processes=num_workers) as pool:
            pool.starmap(worker, [(model, allowed_nodes, allowed_relationship, chunk, visited, processed_urls) for chunk in url_chunks])

        # After processing all URLs, save the visited and processed data
        save_visited_and_processed(visited, processed_urls)

if __name__ == "__main__":
    model = "azure_ai_gpt_4o"
    allowedNodes = "controller,sensor,board,actor,module,software"
    allowedRelationships = "兼容"
    
    path ='visited_links.txt'
    
    with open(path, 'r') as file:
        all_urls = file.read().splitlines()
        urls = all_urls[:200]  # Limit the number of URLs to crawl
    
    logging.info(f"Starting crawl with initial URLs from: {path}")

    main(urls, model, allowedNodes, allowedRelationships)
    logging.info(f"Done Processing all the URLs. Starting to save.")
