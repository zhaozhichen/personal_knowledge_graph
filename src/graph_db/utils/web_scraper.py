"""
Web Scraper Module

This module provides functionality to scrape content from web pages.
"""

import logging
import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

def scrape_url(url: str) -> str:
    """
    Scrape content from a URL.
    
    Args:
        url (str): URL to scrape
        
    Returns:
        str: Extracted text content
    """
    logger.info(f"Scraping URL: {url}")
    
    try:
        response = requests.get(url, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.extract()
            
        # Get text
        text = soup.get_text()
        
        # Break into lines and remove leading and trailing space on each
        lines = (line.strip() for line in text.splitlines())
        
        # Break multi-headlines into a line each
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        
        # Drop blank lines
        text = '\n'.join(chunk for chunk in chunks if chunk)
        
        logger.info(f"Successfully scraped {len(text)} characters from {url}")
        return text
        
    except Exception as e:
        logger.error(f"Error scraping URL {url}: {str(e)}")
        return f"Error scraping URL: {str(e)}" 