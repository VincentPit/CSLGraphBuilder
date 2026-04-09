"""
Web Crawler - Migrated from webpage_retriever.py

This module has been migrated to the new GraphBuilder enterprise structure.
Original functionality is preserved with improved organization and standards.

Migration Date: 2025-10-31
Original File: webpage_retriever.py
New Location: src/graphbuilder/infrastructure/crawlers/web_crawler.py
"""

import asyncio
import aiohttp
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO)

# File to store visited links
VISITED_FILE = 'visited_links.txt'

# Set up visited set (this will store visited URLs)
visited = set()

async def extract_links(url):
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')
                return [urljoin(url, a.get('href')) for a in soup.find_all('a') if a.get('href')]
    except Exception as e:
        logging.error(f"Error fetching {url}: {e}")
        return []

async def recursive_crawl(url, limit, delay=0.05):  # Be nice to the server
    if url in visited:
        return

    if len(visited) >= limit:
        logging.info(f"Reached the limit of {limit} links. Stopping crawl.")
        return

    if 'dfrobot' not in url:
        logging.info(f"Skipping URL without keyword: {url}")
        return

    visited.add(url)
    logging.info(f"Processing: {url} (Visited: {len(visited)})")
    links = await extract_links(url)
    for link in links:
        if len(visited) >= limit:
            break  # Stop if the limit is reached
        await asyncio.sleep(delay)  # Be nice to the server
        await recursive_crawl(link, limit)

def load_visited_links():
    """Load previously visited links from a file."""
    if os.path.exists(VISITED_FILE):
        with open(VISITED_FILE, 'r') as f:
            return set(f.read().splitlines())
    return set()

def save_visited_links():
    """Save visited links to a file."""
    with open(VISITED_FILE, 'w') as f:
        for link in visited:
            f.write(link + '\n')

if __name__ == "__main__":
    start_url = "https://www.dfrobot.com.cn/"
    crawl_limit = 1000 # Set a limit on how many links you want to crawl
    start_urls = [
        "https://www.dfrobot.com.cn/", 
        "https://makelog.dfrobot.com.cn/", 
        "https://mc.dfrobot.com.cn/", 
        "https://www.dfrobot.com.cn/index.php", 
        "https://wiki.dfrobot.com.cn/"
    ]
    
    
    # Load previously visited links
    visited = load_visited_links()
    logging.info(f"Loaded {len(visited)} previously visited links.")
    for start_url in start_urls:
        logging.info(f"Starting crawl at {start_url}")
        asyncio.run(recursive_crawl(start_url, crawl_limit))

    # Save the visited links to a file
    save_visited_links()
    
    logging.info(f"Crawl completed. {len(visited)} links found.")
