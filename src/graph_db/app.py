"""
Application Module

This module provides the main application interface for the graph database system.
"""

import argparse
import logging
import os
import sys
import glob
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Use absolute imports with the updated path
from src.graph_db.core.graph_builder import GraphBuilder
from src.graph_db.visualization.graph_visualizer import GraphVisualizer
from src.graph_db.utils.web_scraper import scrape_url

# Import ConnectionError for exception handling
from neo4j.exceptions import ServiceUnavailable, AuthError

# Constants for chunking
MAX_CHUNK_SIZE = 8000  # Maximum characters per chunk
CHUNK_OVERLAP = 1000   # Overlap between chunks to maintain context

def process_text(text: str, 
                db_uri: str = "bolt://localhost:7687",
                db_username: str = "neo4j",
                db_password: str = "password",
                output_path: str = "graph.html",
                clear_existing: bool = False,
                llm_model: str = "gemini-1.5-pro",
                verbose: bool = False) -> bool:
    """
    Process text input and generate a graph.
    
    Args:
        text (str): Text to process
        db_uri (str): Neo4j connection URI
        db_username (str): Neo4j username
        db_password (str): Neo4j password
        output_path (str): Path to save the visualization
        clear_existing (bool): Whether to clear existing data
        llm_model (str): LLM model to use for entity extraction
        verbose (bool): Whether to display detailed information
        
    Returns:
        bool: True if processing successful, False otherwise
    """
    try:
        # Try to build graph with Neo4j
        try:
            # Build graph
            graph_builder = GraphBuilder(
                db_uri=db_uri,
                db_username=db_username,
                db_password=db_password,
                llm_model=llm_model
            )
            
            entities, relations = graph_builder.build_graph_from_text(text, clear_existing)
            
            if not entities:
                logger.warning("No entities extracted from text")
                return False
                
            # Create visualization
            visualizer = GraphVisualizer()
            
            # Prepare data for visualization
            nodes = []
            for entity in entities:
                nodes.append({
                    "id": entity.get("id", hash(entity["name"])),
                    "label": entity["name"],
                    "group": entity["type"],
                    "properties": entity.get("properties", {})
                })
            
            edges = []
            for relation in relations:
                edges.append({
                    "from": relation["from_entity"]["id"],
                    "to": relation["to_entity"]["id"],
                    "label": relation["relation"],
                    "properties": relation.get("properties", {})
                })
            
            # Create visualization
            success = visualizer.create_visualization_from_data(
                entities=entities,
                relations=relations,
                output_path=output_path,
                title="Text Graph",
                raw_text=text
            )
            
            if success:
                logger.info(f"Graph visualization saved to {output_path}")
                return True
            else:
                logger.error("Failed to create visualization")
                return False
                
        except ConnectionError as e:
            logger.error(f"Neo4j connection error: {str(e)}")
            logger.info("Falling back to visualization-only mode due to database connection failure")
            
            # Use the LLMEntityExtractor directly for visualization-only mode
            from src.graph_db.nlp.llm_entity_extractor import LLMEntityExtractor
            extractor = LLMEntityExtractor(model=llm_model)
            
            # Extract entities and relations
            entities, relations = extractor.process_text(text)
            
            # Display extracted entities and relations if verbose
            if verbose:
                logger.info(f"Extracted {len(entities)} entities:")
                for entity in entities:
                    logger.info(f"  - {entity['name']} ({entity['type']})")
                
                logger.info(f"Extracted {len(relations)} relations:")
                for relation in relations:
                    logger.info(f"  - {relation['from_entity']['name']} --[{relation['relation']}]--> {relation['to_entity']['name']} (confidence: {relation.get('confidence', 1.0):.2f})")
            
            # Create visualization
            visualizer = GraphVisualizer()
            success = visualizer.create_visualization_from_data(
                entities=entities,
                relations=relations,
                output_path=output_path,
                title="Text Graph",
                raw_text=text
            )
            
            if success:
                logger.info(f"Graph visualization saved to {output_path}")
                return True
            else:
                logger.error("Failed to create visualization")
                return False
        except Exception as e:
            logger.error(f"Error building graph: {str(e)}")
            logger.info("Falling back to visualization-only mode due to error")
            
            # Use the LLMEntityExtractor directly for visualization-only mode
            from src.graph_db.nlp.llm_entity_extractor import LLMEntityExtractor
            extractor = LLMEntityExtractor(model=llm_model)
            
            # Extract entities and relations
            entities, relations = extractor.process_text(text)
            
            # Display extracted entities and relations if verbose
            if verbose:
                logger.info(f"Extracted {len(entities)} entities:")
                for entity in entities:
                    logger.info(f"  - {entity['name']} ({entity['type']})")
                
                logger.info(f"Extracted {len(relations)} relations:")
                for relation in relations:
                    logger.info(f"  - {relation['from_entity']['name']} --[{relation['relation']}]--> {relation['to_entity']['name']} (confidence: {relation.get('confidence', 1.0):.2f})")
            
            # Create visualization
            visualizer = GraphVisualizer()
            success = visualizer.create_visualization_from_data(
                entities=entities,
                relations=relations,
                output_path=output_path,
                title="Text Graph",
                raw_text=text
            )
            
            if success:
                logger.info(f"Graph visualization saved to {output_path}")
                return True
            else:
                logger.error("Failed to create visualization")
                return False
            
    except Exception as e:
        logger.error(f"Error processing text: {str(e)}")
        return False

def chunk_text(text: str, max_size: int = MAX_CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Split text into chunks of maximum size with overlap.
    
    Args:
        text (str): Text to split
        max_size (int): Maximum chunk size
        overlap (int): Overlap between chunks
        
    Returns:
        List[str]: List of text chunks
    """
    if len(text) <= max_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        # Calculate end position
        end = start + max_size
        
        # If this is not the last chunk, try to find a good break point
        if end < len(text):
            # Look for paragraph break
            paragraph_break = text.rfind('\n\n', start, end)
            if paragraph_break != -1 and paragraph_break > start + max_size // 2:
                end = paragraph_break + 2  # Include the newlines
            else:
                # Look for sentence break
                sentence_break = text.rfind('. ', start, end)
                if sentence_break != -1 and sentence_break > start + max_size // 2:
                    end = sentence_break + 2  # Include the period and space
        
        # Add the chunk
        chunks.append(text[start:end])
        
        # Move start position for next chunk, accounting for overlap
        start = end - overlap if end < len(text) else len(text)
    
    return chunks

def deduplicate_entities(entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Deduplicate entities based on name and type.
    
    Args:
        entities (List[Dict[str, Any]]): List of entities
        
    Returns:
        List[Dict[str, Any]]: Deduplicated list of entities
    """
    unique_entities = {}
    
    for entity in entities:
        key = (entity["name"], entity["type"])
        
        if key not in unique_entities:
            unique_entities[key] = entity
        else:
            # Merge properties if they exist
            if "properties" in entity and entity["properties"]:
                if "properties" not in unique_entities[key]:
                    unique_entities[key]["properties"] = {}
                
                for prop_key, prop_value in entity["properties"].items():
                    if prop_key not in unique_entities[key]["properties"]:
                        unique_entities[key]["properties"][prop_key] = prop_value
    
    return list(unique_entities.values())

def deduplicate_relations(relations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Deduplicate relations based on source, target, and relation type.
    
    Args:
        relations (List[Dict[str, Any]]): List of relations
        
    Returns:
        List[Dict[str, Any]]: Deduplicated list of relations
    """
    unique_relations = {}
    
    for relation in relations:
        from_entity_name = relation["from_entity"]["name"]
        to_entity_name = relation["to_entity"]["name"]
        relation_type = relation["relation"]
        
        key = (from_entity_name, to_entity_name, relation_type)
        
        if key not in unique_relations:
            unique_relations[key] = relation
        else:
            # Keep the relation with the highest confidence
            if relation.get("confidence", 0) > unique_relations[key].get("confidence", 0):
                unique_relations[key] = relation
    
    return list(unique_relations.values())

def process_url(url: str,
               db_uri: str = "bolt://localhost:7687",
               db_username: str = "neo4j",
               db_password: str = "password",
               output_path: str = "graph.html",
               clear_existing: bool = False,
               llm_model: str = "gemini-1.5-pro",
               verbose: bool = False) -> bool:
    """
    Process a URL input and generate a graph.
    
    Args:
        url (str): URL to process
        db_uri (str): Neo4j connection URI
        db_username (str): Neo4j username
        db_password (str): Neo4j password
        output_path (str): Path to save the visualization
        clear_existing (bool): Whether to clear existing data
        llm_model (str): LLM model to use for entity extraction
        verbose (bool): Whether to display detailed information
        
    Returns:
        bool: True if processing successful, False otherwise
    """
    try:
        # Scrape URL
        text = scrape_url(url)
        if text.startswith("Error"):
            logger.error(text)
            return False
        
        logger.info(f"Successfully scraped URL: {url} ({len(text)} characters)")
        
        # Check if text is too large and needs chunking
        if len(text) > MAX_CHUNK_SIZE:
            logger.info(f"Text is large ({len(text)} characters), processing in chunks")
            return process_url_in_chunks(
                url, text, db_uri, db_username, db_password, 
                output_path, clear_existing, llm_model, verbose
            )
        
        # Try to build graph with Neo4j
        try:
            # Build graph
            graph_builder = GraphBuilder(
                db_uri=db_uri,
                db_username=db_username,
                db_password=db_password,
                llm_model=llm_model
            )
            
            entities, relations = graph_builder.build_graph_from_text(text, clear_existing)
            
            if not entities:
                logger.warning("No entities extracted from text")
                return False
                
            # Display detailed entities and relations if verbose
            if verbose:
                logger.info(f"Detailed entities ({len(entities)}):")
                for entity in entities:
                    logger.info(f"  - {entity['name']} ({entity['type']})")
                
                logger.info(f"Detailed relations ({len(relations)}):")
                for relation in relations:
                    logger.info(f"  - {relation['from_entity']['name']} --[{relation['relation']}]--> {relation['to_entity']['name']} (confidence: {relation.get('confidence', 1.0):.2f})")
            
            # Create visualization
            visualizer = GraphVisualizer()
            
            # Prepare data for visualization
            nodes = []
            for entity in entities:
                nodes.append({
                    "id": entity.get("id", hash(entity["name"])),
                    "label": entity["name"],
                    "group": entity["type"],
                    "properties": entity.get("properties", {})
                })
            
            edges = []
            for relation in relations:
                edges.append({
                    "from": relation["from_entity"]["id"],
                    "to": relation["to_entity"]["id"],
                    "label": relation["relation"],
                    "properties": relation.get("properties", {})
                })
            
            # Create visualization
            success = visualizer.create_visualization_from_data(
                entities=entities,
                relations=relations,
                output_path=output_path,
                title=f"URL Graph: {url}",
                raw_text=text[:500] + "..." if len(text) > 500 else text
            )
            
            if success:
                logger.info(f"Graph visualization saved to {output_path}")
                return True
            else:
                logger.error("Failed to create visualization")
                return False
                
        except ConnectionError as e:
            logger.error(f"Neo4j connection error: {str(e)}")
            logger.info("Falling back to visualization-only mode due to database connection failure")
            
            # Use the LLMEntityExtractor directly for visualization-only mode
            from src.graph_db.nlp.llm_entity_extractor import LLMEntityExtractor
            extractor = LLMEntityExtractor(model=llm_model)
            
            # Extract entities and relations
            entities, relations = extractor.process_text(text)
            
            # Display detailed entities and relations if verbose
            if verbose:
                logger.info(f"Detailed entities ({len(entities)}):")
                for entity in entities:
                    logger.info(f"  - {entity['name']} ({entity['type']})")
                
                logger.info(f"Detailed relations ({len(relations)}):")
                for relation in relations:
                    logger.info(f"  - {relation['from_entity']['name']} --[{relation['relation']}]--> {relation['to_entity']['name']} (confidence: {relation.get('confidence', 1.0):.2f})")
            
            # Create visualization
            visualizer = GraphVisualizer()
            success = visualizer.create_visualization_from_data(
                entities=entities,
                relations=relations,
                output_path=output_path,
                title=f"URL Graph: {url}",
                raw_text=text[:500] + "..." if len(text) > 500 else text
            )
            
            if success:
                logger.info(f"Graph visualization saved to {output_path}")
                return True
            else:
                logger.error("Failed to create visualization")
                return False
        except Exception as e:
            logger.error(f"Error building graph: {str(e)}")
            logger.info("Falling back to visualization-only mode due to error")
            
            # Use the LLMEntityExtractor directly for visualization-only mode
            from src.graph_db.nlp.llm_entity_extractor import LLMEntityExtractor
            extractor = LLMEntityExtractor(model=llm_model)
            
            # Extract entities and relations
            entities, relations = extractor.process_text(text)
            
            # Display detailed entities and relations if verbose
            if verbose:
                logger.info(f"Detailed entities ({len(entities)}):")
                for entity in entities:
                    logger.info(f"  - {entity['name']} ({entity['type']})")
                
                logger.info(f"Detailed relations ({len(relations)}):")
                for relation in relations:
                    logger.info(f"  - {relation['from_entity']['name']} --[{relation['relation']}]--> {relation['to_entity']['name']} (confidence: {relation.get('confidence', 1.0):.2f})")
            
            # Create visualization
            visualizer = GraphVisualizer()
            success = visualizer.create_visualization_from_data(
                entities=entities,
                relations=relations,
                output_path=output_path,
                title=f"URL Graph: {url}",
                raw_text=text[:500] + "..." if len(text) > 500 else text
            )
            
            if success:
                logger.info(f"Graph visualization saved to {output_path}")
                return True
            else:
                logger.error("Failed to create visualization")
                return False
            
    except Exception as e:
        logger.error(f"Error processing URL: {str(e)}")
        return False

def process_url_in_chunks(url: str, 
                         text: str,
                         db_uri: str = "bolt://localhost:7687",
                         db_username: str = "neo4j",
                         db_password: str = "password",
                         output_path: str = "graph.html",
                         clear_existing: bool = False,
                         llm_model: str = "gemini-1.5-pro",
                         verbose: bool = False) -> bool:
    """
    Process a URL by splitting the text into chunks and processing each chunk.
    
    Args:
        url (str): URL being processed
        text (str): Text content from the URL
        db_uri (str): Neo4j connection URI
        db_username (str): Neo4j username
        db_password (str): Neo4j password
        output_path (str): Path to save the visualization
        clear_existing (bool): Whether to clear existing data
        llm_model (str): LLM model to use for entity extraction
        verbose (bool): Whether to display detailed information
        
    Returns:
        bool: True if processing successful, False otherwise
    """
    try:
        # Split text into chunks
        chunks = chunk_text(text)
        logger.info(f"Split text into {len(chunks)} chunks")
        
        # Initialize lists to store entities and relations from all chunks
        all_entities = []
        all_relations = []
        
        # Initialize entity extractor
        from src.graph_db.nlp.llm_entity_extractor import LLMEntityExtractor
        extractor = LLMEntityExtractor(model=llm_model)
        
        # Process each chunk
        for i, chunk in enumerate(chunks):
            logger.info(f"Processing chunk {i+1}/{len(chunks)} ({len(chunk)} characters)")
            
            # Extract entities and relations from this chunk
            chunk_entities, chunk_relations = extractor.process_text(chunk)
            
            # If no entities were found, skip to next chunk
            if not chunk_entities:
                logger.warning(f"No entities found in chunk {i+1}")
                continue
                
            # Add to overall lists
            all_entities.extend(chunk_entities)
            all_relations.extend(chunk_relations)
            
            logger.info(f"Chunk {i+1}: Found {len(chunk_entities)} entities and {len(chunk_relations)} relations")
        
        # Deduplicate entities and relations
        unique_entities = deduplicate_entities(all_entities)
        
        # Create a map of entity names for relation deduplication
        entity_map = {entity["name"]: entity for entity in unique_entities}
        
        # Update relations to use the deduplicated entities
        for relation in all_relations:
            from_name = relation["from_entity"]["name"]
            to_name = relation["to_entity"]["name"]
            
            if from_name in entity_map and to_name in entity_map:
                relation["from_entity"] = entity_map[from_name]
                relation["to_entity"] = entity_map[to_name]
        
        # Deduplicate relations
        unique_relations = deduplicate_relations(all_relations)
        
        logger.info(f"After deduplication: {len(unique_entities)} entities and {len(unique_relations)} relations")
        
        # Display detailed entities and relations if verbose
        if verbose:
            logger.info(f"Detailed entities ({len(unique_entities)}):")
            for entity in unique_entities:
                logger.info(f"  - {entity['name']} ({entity['type']})")
            
            logger.info(f"Detailed relations ({len(unique_relations)}):")
            for relation in unique_relations:
                logger.info(f"  - {relation['from_entity']['name']} --[{relation['relation']}]--> {relation['to_entity']['name']} (confidence: {relation.get('confidence', 1.0):.2f})")
        
        # Try to store in Neo4j if requested
        if not clear_existing:
            try:
                # Build graph
                graph_builder = GraphBuilder(
                    db_uri=db_uri,
                    db_username=db_username,
                    db_password=db_password,
                    llm_model=llm_model
                )
                
                # Store entities and relations in Neo4j
                graph_builder.store_entities_and_relations(unique_entities, unique_relations)
                logger.info("Stored entities and relations in Neo4j")
                
            except Exception as e:
                logger.error(f"Error storing in Neo4j: {str(e)}")
                logger.info("Continuing with visualization only")
        
        # Create visualization
        visualizer = GraphVisualizer()
        success = visualizer.create_visualization_from_data(
            entities=unique_entities,
            relations=unique_relations,
            output_path=output_path,
            title=f"URL Graph (Chunked): {url}",
            raw_text=text[:500] + "..." if len(text) > 500 else text
        )
        
        if success:
            logger.info(f"Graph visualization saved to {output_path}")
            return True
        else:
            logger.error("Failed to create visualization")
            return False
            
    except Exception as e:
        logger.error(f"Error processing URL in chunks: {str(e)}")
        return False

def process_multiple_urls(urls: List[str],
                     db_uri: str = "bolt://localhost:7687",
                     db_username: str = "neo4j",
                     db_password: str = "password",
                     output_path: str = "graph.html",
                     clear_existing: bool = False,
                     llm_model: str = "gemini-1.5-pro",
                     verbose: bool = False) -> bool:
    """
    Process multiple URL inputs and generate a combined graph.
    
    Args:
        urls (List[str]): URLs to process
        db_uri (str): Neo4j connection URI
        db_username (str): Neo4j username
        db_password (str): Neo4j password
        output_path (str): Path to save the visualization
        clear_existing (bool): Whether to clear existing data
        llm_model (str): LLM model to use for entity extraction
        verbose (bool): Whether to display detailed information
        
    Returns:
        bool: True if processing successful, False otherwise
    """
    try:
        # Initialize lists to store all entities and relations
        all_entities = []
        all_relations = []
        all_raw_texts = []
        
        # Process each URL
        for url in urls:
            logger.info(f"Processing URL: {url}")
            
            # Scrape URL
            text = scrape_url(url)
            if text.startswith("Error"):
                logger.error(text)
                continue
            
            logger.info(f"Successfully scraped {len(text)} characters from URL: {url}")
            
            # Add to raw texts for visualization
            all_raw_texts.append(f"From {url}:\n{text[:500]}..." if len(text) > 500 else text)
            
            # Check if text is too large and needs chunking
            if len(text) > MAX_CHUNK_SIZE:
                logger.info(f"Text from {url} is large ({len(text)} characters), processing in chunks")
                
                # Split text into chunks
                chunks = chunk_text(text)
                logger.info(f"Split text from {url} into {len(chunks)} chunks")
                
                # Initialize entity extractor
                from src.graph_db.nlp.llm_entity_extractor import LLMEntityExtractor
                extractor = LLMEntityExtractor(model=llm_model)
                
                # Process each chunk
                for i, chunk in enumerate(chunks):
                    logger.info(f"Processing chunk {i+1}/{len(chunks)} from {url} ({len(chunk)} characters)")
                    
                    # Extract entities and relations from this chunk
                    chunk_entities, chunk_relations = extractor.process_text(chunk)
                    
                    # If no entities were found, skip to next chunk
                    if not chunk_entities:
                        logger.warning(f"No entities found in chunk {i+1} from {url}")
                        continue
                        
                    # Add to overall lists
                    all_entities.extend(chunk_entities)
                    all_relations.extend(chunk_relations)
                    
                    logger.info(f"Chunk {i+1} from {url}: Found {len(chunk_entities)} entities and {len(chunk_relations)} relations")
            else:
                # Process the entire text at once
                logger.info(f"Processing entire text from {url} ({len(text)} characters)")
                
                # Initialize entity extractor
                from src.graph_db.nlp.llm_entity_extractor import LLMEntityExtractor
                extractor = LLMEntityExtractor(model=llm_model)
                
                # Extract entities and relations
                url_entities, url_relations = extractor.process_text(text)
                
                # If no entities were found, skip to next URL
                if not url_entities:
                    logger.warning(f"No entities found in text from {url}")
                    continue
                    
                # Add to overall lists
                all_entities.extend(url_entities)
                all_relations.extend(url_relations)
                
                logger.info(f"Found {len(url_entities)} entities and {len(url_relations)} relations from {url}")
        
        if not all_entities:
            logger.error("No entities were extracted from any of the provided URLs")
            return False
        
        # Deduplicate entities and relations
        unique_entities = deduplicate_entities(all_entities)
        
        # Create a map of entity names for relation deduplication
        entity_map = {entity["name"]: entity for entity in unique_entities}
        
        # Update relations to use the deduplicated entities
        for relation in all_relations:
            from_name = relation["from_entity"]["name"]
            to_name = relation["to_entity"]["name"]
            
            if from_name in entity_map and to_name in entity_map:
                relation["from_entity"] = entity_map[from_name]
                relation["to_entity"] = entity_map[to_name]
        
        # Deduplicate relations
        unique_relations = deduplicate_relations(all_relations)
        
        logger.info(f"After deduplication: {len(unique_entities)} entities and {len(unique_relations)} relations from {len(urls)} URLs")
        
        # Display detailed entities and relations if verbose
        if verbose:
            logger.info(f"Detailed entities ({len(unique_entities)}):")
            for entity in unique_entities:
                logger.info(f"  - {entity['name']} ({entity['type']})")
            
            logger.info(f"Detailed relations ({len(unique_relations)}):")
            for relation in unique_relations:
                logger.info(f"  - {relation['from_entity']['name']} --[{relation['relation']}]--> {relation['to_entity']['name']} (confidence: {relation.get('confidence', 1.0):.2f})")
        
        # Try to store in Neo4j if requested
        if not clear_existing:
            try:
                # Build graph
                graph_builder = GraphBuilder(
                    db_uri=db_uri,
                    db_username=db_username,
                    db_password=db_password,
                    llm_model=llm_model
                )
                
                # Store entities and relations in Neo4j
                graph_builder.store_entities_and_relations(unique_entities, unique_relations)
                logger.info("Stored entities and relations in Neo4j")
                
            except Exception as e:
                logger.error(f"Error storing in Neo4j: {str(e)}")
                logger.info("Continuing with visualization only")
        
        # Create visualization
        visualizer = GraphVisualizer()
        success = visualizer.create_visualization_from_data(
            entities=unique_entities,
            relations=unique_relations,
            output_path=output_path,
            title=f"Combined Graph from {len(urls)} URLs",
            raw_text="\n\n".join(all_raw_texts)
        )
        
        if success:
            logger.info(f"Combined graph visualization saved to {output_path}")
            return True
        else:
            logger.error("Failed to create visualization")
            return False
            
    except Exception as e:
        logger.error(f"Error processing URLs: {str(e)}")
        return False

def process_directory(input_dir: str, output_file: str, visualization_only: bool, llm_model: str = "gemini", verbose: bool = False) -> None:
    """Process all files in a directory and generate a merged graph visualization."""
    # Find all files in the directory
    files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
    logger.info(f"Found {len(files)} files to process in {input_dir}")
    
    # Initialize the graph builder
    if visualization_only:
        # For visualization-only mode, we don't need to connect to Neo4j
        graph_builder = GraphBuilder(db_uri="", db_username="", db_password="", llm_model=llm_model)
    else:
        graph_builder = GraphBuilder(llm_model=llm_model)
    
    # Initialize the entity extractor
    # Use the real LLMEntityExtractor
    from src.graph_db.nlp.llm_entity_extractor import LLMEntityExtractor
    from src.graph_db.nlp.mock_entity_extractor import MockEntityExtractor
    entity_extractor = LLMEntityExtractor(model=llm_model)
    
    # Initialize the visualizer
    visualizer = GraphVisualizer()
    
    # Lists to store all entities, relations, and texts
    all_entities = []
    all_relations = []
    all_texts = []
    
    # Process each file
    for file_path in files:
        try:
            # Read the file
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            try:
                logger.info(f"Attempting to extract entities and relations from {file_path} using LLMEntityExtractor")
                # Use the process_text method which handles both entity and relation extraction
                entities, relations = entity_extractor.process_text(text)
                logger.info(f"Successfully extracted {len(entities)} entities and {len(relations)} relations from {file_path}")
                
            except Exception as e:
                logger.error(f"LLMEntityExtractor failed with error: {str(e)}")
                logger.error(f"Error type: {type(e).__name__}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
                logger.warning(f"Falling back to MockEntityExtractor.")
                
                # Fall back to MockEntityExtractor
                mock_extractor = MockEntityExtractor()
                entities, relations = mock_extractor.process_text(text)
                logger.info(f"Successfully extracted {len(entities)} entities and {len(relations)} relations using MockEntityExtractor")
            
            logger.info(f"Processing file: {file_path}")
            
            # Log extracted entities and relations
            if verbose:
                logger.info(f"Extracted {len(entities)} entities from {file_path}:")
                for entity in entities:
                    logger.info(f"  - {entity['name']} ({entity['type']})")
                
                logger.info(f"Extracted {len(relations)} relations from {file_path}:")
                for relation in relations:
                    logger.info(f"  - {relation['from_entity']['name']} --[{relation['relation']}]--> {relation['to_entity']['name']} (confidence: {relation.get('confidence', 1.0):.2f})")
            
            # Add to the combined lists
            all_entities.extend(entities)
            all_relations.extend(relations)
            all_texts.append(f"# {os.path.basename(file_path)}\n\n{text}")
            
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}")
    
    # Deduplicate entities and relations
    unique_entities = deduplicate_entities(all_entities)
    unique_relations = deduplicate_relations(all_relations)
    
    # Create visualization
    combined_text = "\n\n---\n\n".join(all_texts)
    
    success = visualizer.create_visualization_from_data(
        entities=unique_entities,
        relations=unique_relations,
        output_path=output_file,
        title=f"Combined Graph: {input_dir}",
        raw_text=combined_text
    )
    
    if success:
        logger.info(f"Combined graph visualization saved to {output_file}")
    else:
        logger.error("Failed to create visualization")

def main():
    """Main entry point for the application."""
    # Define global variables
    global MAX_CHUNK_SIZE, CHUNK_OVERLAP
    
    parser = argparse.ArgumentParser(description="Graph Database Application")
    parser.add_argument("--text", help="Text to process")
    parser.add_argument("--file", help="File to process")
    parser.add_argument("--url", action="append", help="URL to process (can be specified multiple times)")
    parser.add_argument("--input-dir", help="Directory containing files to process")
    parser.add_argument("--output", default="graph.html", help="Output file path")
    parser.add_argument("--db-uri", default="bolt://localhost:7687", help="Neo4j URI")
    parser.add_argument("--db-user", default="neo4j", help="Neo4j username")
    parser.add_argument("--db-pass", default="password", help="Neo4j password")
    parser.add_argument("--clear", action="store_true", help="Clear existing data")
    parser.add_argument("--llm-model", default="gpt-4o", help="LLM model to use")
    parser.add_argument("--visualization-only", action="store_true", help="Skip Neo4j connection and only create visualization")
    parser.add_argument("--verbose", action="store_true", help="Display detailed information about extracted entities and relations")
    parser.add_argument("--chunk-size", type=int, default=MAX_CHUNK_SIZE, help=f"Maximum chunk size for processing large texts (default: {MAX_CHUNK_SIZE})")
    parser.add_argument("--chunk-overlap", type=int, default=CHUNK_OVERLAP, help=f"Overlap between chunks (default: {CHUNK_OVERLAP})")
    
    args = parser.parse_args()
    
    # Update chunk size and overlap if specified
    if args.chunk_size:
        MAX_CHUNK_SIZE = args.chunk_size
    if args.chunk_overlap:
        CHUNK_OVERLAP = args.chunk_overlap
    
    if not args.text and not args.file and not args.url and not args.input_dir:
        parser.print_help()
        return False
    
    # Process input directory
    if args.input_dir:
        if args.visualization_only:
            return process_directory(
                args.input_dir,
                args.output,
                args.visualization_only,
                args.llm_model,
                args.verbose
            )
        else:
            # For non-visualization-only mode, we would need to implement Neo4j integration
            # For now, we'll just use the visualization-only mode
            logger.info("Processing directory with Neo4j integration is not implemented yet. Using visualization-only mode.")
            return process_directory(
                args.input_dir,
                args.output,
                args.visualization_only,
                args.llm_model,
                args.verbose
            )
    
    if args.text:
        if args.visualization_only:
            # Use the LLMEntityExtractor directly for visualization-only mode
            from src.graph_db.nlp.llm_entity_extractor import LLMEntityExtractor
            extractor = LLMEntityExtractor(model=args.llm_model)
            
            # Extract entities and relations
            entities, relations = extractor.process_text(args.text)
            
            # Display extracted entities and relations if verbose
            if args.verbose:
                logger.info(f"Detailed entities ({len(entities)}):")
                for entity in entities:
                    logger.info(f"  - {entity['name']} ({entity['type']})")
                
                logger.info(f"Detailed relations ({len(relations)}):")
                for relation in relations:
                    logger.info(f"  - {relation['from_entity']['name']} --[{relation['relation']}]--> {relation['to_entity']['name']} (confidence: {relation.get('confidence', 1.0):.2f})")
            
            # Create visualization
            visualizer = GraphVisualizer()
            success = visualizer.create_visualization_from_data(
                entities=entities,
                relations=relations,
                output_path=args.output,
                title="Text Graph",
                raw_text=args.text
            )
            
            if success:
                logger.info(f"Graph visualization saved to {args.output}")
                return True
            else:
                logger.error("Failed to create visualization")
                return False
        else:
            return process_text(
                args.text, 
                args.db_uri, 
                args.db_user, 
                args.db_pass, 
                args.output,
                args.clear,
                args.llm_model,
                args.verbose
            )
    
    if args.file:
        try:
            with open(args.file, 'r', encoding='utf-8') as f:
                text = f.read()
            
            # Check if text is too large and needs chunking
            if len(text) > MAX_CHUNK_SIZE and not args.visualization_only:
                logger.info(f"Text from file is large ({len(text)} characters), processing in chunks")
                
                # Create a temporary URL-like identifier for the file
                file_url = f"file://{os.path.abspath(args.file)}"
                
                # Process the file as if it were a URL
                return process_url_in_chunks(
                    file_url, 
                    text, 
                    args.db_uri, 
                    args.db_user, 
                    args.db_pass, 
                    args.output,
                    args.clear,
                    args.llm_model,
                    args.verbose
                )
            
            if args.visualization_only:
                # Use the LLMEntityExtractor directly for visualization-only mode
                from src.graph_db.nlp.llm_entity_extractor import LLMEntityExtractor
                extractor = LLMEntityExtractor(model=args.llm_model)
                
                # Extract entities and relations
                entities, relations = extractor.process_text(text)
                
                # Display extracted entities and relations if verbose
                if args.verbose:
                    logger.info(f"Detailed entities ({len(entities)}):")
                    for entity in entities:
                        logger.info(f"  - {entity['name']} ({entity['type']})")
                    
                    logger.info(f"Detailed relations ({len(relations)}):")
                    for relation in relations:
                        logger.info(f"  - {relation['from_entity']['name']} --[{relation['relation']}]--> {relation['to_entity']['name']} (confidence: {relation.get('confidence', 1.0):.2f})")
                
                # Create visualization
                visualizer = GraphVisualizer()
                success = visualizer.create_visualization_from_data(
                    entities=entities,
                    relations=relations,
                    output_path=args.output,
                    title=f"File Graph: {args.file}",
                    raw_text=text
                )
                
                if success:
                    logger.info(f"Graph visualization saved to {args.output}")
                    return True
                else:
                    logger.error("Failed to create visualization")
                    return False
            else:
                return process_text(
                    text, 
                    args.db_uri, 
                    args.db_user, 
                    args.db_pass, 
                    args.output,
                    args.clear,
                    args.llm_model,
                    args.verbose
                )
        except Exception as e:
            logger.error(f"Error reading file: {str(e)}")
            return False
    
    if args.url:
        if len(args.url) > 1:
            # Process multiple URLs
            return process_multiple_urls(
                args.url,
                args.db_uri,
                args.db_user,
                args.db_pass,
                args.output,
                args.clear,
                args.llm_model,
                args.verbose
            )
        else:
            # Process single URL
            return process_url(
                args.url[0], 
                args.db_uri, 
                args.db_user, 
                args.db_pass, 
                args.output,
                args.clear,
                args.llm_model,
                args.verbose
            )

if __name__ == "__main__":
    main() 