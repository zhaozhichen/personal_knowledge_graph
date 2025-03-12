"""
Input Processor Module

This module handles various input types and formats for the graph database system.
It supports processing:
- Text chunks
- Lists of text chunks
- Text files
- Lists of text files
- URLs
- Lists of URLs
- Any combination of the above
"""

import os
import logging
import uuid
from typing import List, Dict, Any, Union, Optional
from pathlib import Path

from src.graph_db.utils.web_scraper import scrape_url

# Configure logging
logger = logging.getLogger(__name__)

# Constants for chunking
MAX_CHUNK_SIZE = 8000  # Maximum characters per chunk
CHUNK_OVERLAP = 1000   # Overlap between chunks to maintain context

class InputProcessor:
    """
    Handles processing of various input types and formats.
    """
    
    def __init__(self, 
                 input_dir: str = "input",
                 chunk_size: int = MAX_CHUNK_SIZE,
                 chunk_overlap: int = CHUNK_OVERLAP):
        """
        Initialize the input processor.
        
        Args:
            input_dir (str): Directory to save processed input files
            chunk_size (int): Maximum size of text chunks
            chunk_overlap (int): Overlap between chunks
        """
        self.input_dir = input_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Create input directory if it doesn't exist
        os.makedirs(self.input_dir, exist_ok=True)
    
    def process_input(self, 
                     text: Optional[Union[str, List[str]]] = None,
                     files: Optional[Union[str, List[str]]] = None,
                     urls: Optional[Union[str, List[str]]] = None) -> Dict[str, Any]:
        """
        Process various input types and merge them into a single text.
        
        Args:
            text (str or List[str]): Text chunk(s) to process
            files (str or List[str]): File path(s) to process
            urls (str or List[str]): URL(s) to process
            
        Returns:
            Dict[str, Any]: Dictionary containing processed text and metadata
        """
        all_texts = []
        sources = []
        
        # Process text input
        if text:
            text_chunks = [text] if isinstance(text, str) else text
            for i, chunk in enumerate(text_chunks):
                all_texts.append(chunk)
                sources.append(f"Text input {i+1}")
                logger.info(f"Processed text input {i+1} ({len(chunk)} characters)")
        
        # Process file input
        if files:
            file_paths = [files] if isinstance(files, str) else files
            for file_path in file_paths:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        file_content = f.read()
                    all_texts.append(file_content)
                    sources.append(f"File: {file_path}")
                    logger.info(f"Processed file: {file_path} ({len(file_content)} characters)")
                except Exception as e:
                    logger.error(f"Error reading file {file_path}: {str(e)}")
        
        # Process URL input
        if urls:
            url_list = [urls] if isinstance(urls, str) else urls
            for url in url_list:
                url_content = scrape_url(url)
                if not url_content.startswith("Error"):
                    all_texts.append(url_content)
                    sources.append(f"URL: {url}")
                    logger.info(f"Processed URL: {url} ({len(url_content)} characters)")
                else:
                    logger.error(f"Error processing URL {url}: {url_content}")
        
        # Merge all texts
        if not all_texts:
            logger.error("No valid input provided")
            return {"success": False, "error": "No valid input provided"}
        
        # Create a merged text with source headers
        merged_text = ""
        for i, (text, source) in enumerate(zip(all_texts, sources)):
            merged_text += f"# Source: {source}\n\n{text}\n\n"
            if i < len(all_texts) - 1:
                merged_text += "---\n\n"
        
        # Generate a unique filename for the merged text
        output_filename = f"merged_{uuid.uuid4().hex[:8]}.txt"
        output_path = os.path.join(self.input_dir, output_filename)
        
        # Save the merged text
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(merged_text)
        
        logger.info(f"Merged text saved to {output_path} ({len(merged_text)} characters)")
        
        # Check if the merged text needs to be chunked
        chunks = []
        if len(merged_text) > self.chunk_size:
            logger.info(f"Merged text is large ({len(merged_text)} characters), splitting into chunks")
            chunks = self._chunk_text(merged_text)
            logger.info(f"Split merged text into {len(chunks)} chunks")
        else:
            chunks = [merged_text]
        
        return {
            "success": True,
            "merged_text": merged_text,
            "merged_file": output_path,
            "chunks": chunks,
            "sources": sources,
            "num_sources": len(sources)
        }
    
    def _chunk_text(self, text: str) -> List[str]:
        """
        Split text into chunks of maximum size with overlap.
        
        Args:
            text (str): Text to split
            
        Returns:
            List[str]: List of text chunks
        """
        if len(text) <= self.chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            # Calculate end position
            end = start + self.chunk_size
            
            # If this is not the last chunk, try to find a good break point
            if end < len(text):
                # Look for paragraph break
                paragraph_break = text.rfind('\n\n', start, end)
                if paragraph_break != -1 and paragraph_break > start + self.chunk_size // 2:
                    end = paragraph_break + 2  # Include the newlines
                else:
                    # Look for sentence break
                    sentence_break = text.rfind('. ', start, end)
                    if sentence_break != -1 and sentence_break > start + self.chunk_size // 2:
                        end = sentence_break + 2  # Include the period and space
            
            # Add the chunk
            chunks.append(text[start:end])
            
            # Move start position for next chunk, accounting for overlap
            start = end - self.chunk_overlap if end < len(text) else len(text)
        
        return chunks
    
    def process_directory(self, directory_path: str) -> Dict[str, Any]:
        """
        Process all files in a directory.
        
        Args:
            directory_path (str): Path to the directory containing files to process
            
        Returns:
            Dict[str, Any]: Dictionary containing processed text and metadata
        """
        # Get all text and markdown files in the directory
        file_paths = []
        for ext in ['.txt', '.md', '.text', '.markdown']:
            file_paths.extend(list(Path(directory_path).glob(f'**/*{ext}')))
        
        if not file_paths:
            logger.error(f"No text or markdown files found in directory: {directory_path}")
            return {"success": False, "error": f"No text or markdown files found in directory: {directory_path}"}
        
        # Process the files
        return self.process_input(files=[str(path) for path in file_paths]) 