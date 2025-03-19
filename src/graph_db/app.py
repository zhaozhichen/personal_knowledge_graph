"""
Application Module

This module provides the main application interface for the graph database system.
"""

import argparse
import logging
import os
import sys
import glob
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
import uuid
from dotenv import load_dotenv, find_dotenv

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
from src.graph_db.utils.node_deduplicator import NodeDeduplicator
from src.graph_db.input.input_processor import InputProcessor
from src.graph_db.nlp.llm_entity_extractor import LLMEntityExtractor
from src.graph_db.nlp.mock_entity_extractor import MockEntityExtractor
from src.graph_db.utils.graph_qa import GraphQA

# Import ConnectionError for exception handling
from neo4j.exceptions import ServiceUnavailable, AuthError

# Import embedding functions from tools.llm_api
try:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from tools.llm_api import get_embedding
except ImportError:
    print(f"Warning: Could not import embedding functions from tools.llm_api. Edge embeddings will be disabled.", file=sys.stderr)
    get_embedding = None

# Constants for chunking
MAX_CHUNK_SIZE = 1000  # Maximum characters per chunk
CHUNK_OVERLAP = 50   # Overlap between chunks to maintain context

# Add import for the API server
from src.graph_db.api_server import start_api_server

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
                
        except Exception as e:
            logger.error(f"Error building graph: {str(e)}")
            logger.info("Falling back to visualization-only mode due to error")
            
            # Use the LLMEntityExtractor directly for visualization-only mode
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

def process_with_entity_extractor(text: str, 
                                 llm_model: str = "gpt-4o", 
                                 verbose: bool = False,
                                 use_mock: bool = False) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Process text with entity extractor, with fallback to mock extractor if needed.
    
    Args:
        text (str): Text to process
        llm_model (str): LLM model to use
        verbose (bool): Whether to display detailed information
        use_mock (bool): Whether to use MockEntityExtractor instead of LLMEntityExtractor
        
    Returns:
        Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]: Tuple of (entities, relations)
    """
    if use_mock:
        logger.info(f"Using MockEntityExtractor for UI debugging")
        mock_extractor = MockEntityExtractor()
        entities, relations = mock_extractor.process_text(text)
        logger.info(f"Successfully extracted {len(entities)} entities and {len(relations)} relations using MockEntityExtractor")
    else:
        try:
            logger.info(f"Attempting to extract entities and relations using LLMEntityExtractor with model {llm_model}")
            # Use the process_text method which handles both entity and relation extraction
            entity_extractor = LLMEntityExtractor(model=llm_model)
            entities, relations = entity_extractor.process_text(text)
            
            # Check if extraction was successful
            if not entities and not relations:
                logger.warning("LLMEntityExtractor returned empty results, trying with a different model")
                # Try with a different model
                fallback_model = "claude-3-5-sonnet-20241022" if llm_model != "claude-3-5-sonnet-20241022" else "gpt-4o"
                logger.info(f"Trying with fallback model: {fallback_model}")
                entity_extractor = LLMEntityExtractor(model=fallback_model)
                entities, relations = entity_extractor.process_text(text)
            
            logger.info(f"Successfully extracted {len(entities)} entities and {len(relations)} relations")
            
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
    
    # Log extracted entities and relations
    if verbose:
        logger.info(f"Extracted {len(entities)} entities:")
        for entity in entities:
            logger.info(f"  - {entity['name']} ({entity['type']})")
        
        logger.info(f"Extracted {len(relations)} relations:")
        for relation in relations:
            logger.info(f"  - {relation['from_entity']['name']} --[{relation['relation']}]--> {relation['to_entity']['name']} (confidence: {relation.get('confidence', 1.0):.2f})")
    
    return entities, relations

def process_input_sources(
    text: Optional[str] = None,
    files: Optional[List[str]] = None,
    urls: Optional[List[str]] = None,
    input_dir: Optional[str] = None,
    output_path: str = "graph.html",
    db_uri: str = "bolt://localhost:7687",
    db_username: str = "neo4j",
    db_password: str = "password",
    clear_existing: bool = False,
    visualization_only: bool = False,
    llm_model: str = "gpt-4o",
    verbose: bool = False,
    chunk_size: int = MAX_CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    use_mock: bool = False,
    deduplicate_nodes: bool = False,
    similarity_threshold: float = 0.85,
    use_embeddings: bool = True,
) -> bool:
    """
    Process multiple input sources and generate a combined graph.
    
    Args:
        text (Optional[str]): Text to process
        files (Optional[List[str]]): List of files to process
        urls (Optional[List[str]]): List of URLs to process
        input_dir (Optional[str]): Directory containing files to process
        output_path (str): Output file path
        db_uri (str): Neo4j URI
        db_username (str): Neo4j username
        db_password (str): Neo4j password
        clear_existing (bool): Clear existing data
        visualization_only (bool): Skip Neo4j connection and only create visualization
        llm_model (str): LLM model to use
        verbose (bool): Display detailed information about extracted entities and relations
        chunk_size (int): Maximum chunk size for processing large texts
        chunk_overlap (int): Overlap between chunks
        use_mock (bool): Use MockEntityExtractor instead of LLMEntityExtractor
        deduplicate_nodes (bool): Whether to deduplicate similar nodes (default: False)
        similarity_threshold (float): Threshold for string similarity (0.0 to 1.0)
        use_embeddings (bool): Whether to use LLM embeddings for similarity calculation
        
    Returns:
        bool: True if processing was successful, False otherwise
    """
    # Initialize the graph visualizer
    visualizer = GraphVisualizer()
    
    # Step 1: Check if corresponding JSON file exists, if not generate it from input sources
    json_output_path = output_path.replace('.html', '.json')
    
    # Check if the JSON file already exists
    if os.path.exists(json_output_path):
        logger.info(f"Found existing JSON file at {json_output_path}, using it for visualization")
        try:
            with open(json_output_path, 'r') as f:
                graph_data = json.load(f)
                
            # Extract entities and relations from the existing JSON file
            unique_entities = graph_data["data"]["entities"]
            unique_relations = graph_data["data"]["relations"]
            raw_text = graph_data.get("raw_text", "")
            
            logger.info(f"Loaded {len(unique_entities)} entities and {len(unique_relations)} relations from existing JSON file")
            if raw_text:
                logger.info(f"Source text found in JSON file ({len(raw_text)} characters)")
            else:
                logger.info("No source text found in JSON file")
                # If raw text is empty but we have input files or text, try to recover the raw text
                if files and len(files) == 1:
                    try:
                        logger.info(f"Attempting to load raw text from original file: {files[0]}")
                        with open(files[0], 'r', encoding='utf-8') as f:
                            raw_text = f.read()
                        logger.info(f"Successfully loaded raw text from file ({len(raw_text)} characters)")
                    except Exception as e:
                        logger.error(f"Failed to load raw text from file: {str(e)}")
                elif text:
                    raw_text = text
                    logger.info(f"Using provided text as raw text ({len(raw_text)} characters)")
            
        except Exception as e:
            logger.error(f"Error loading existing JSON file: {str(e)}")
            logger.info("Processing input sources to generate new JSON file")
            # Fall through to process input sources
            unique_entities, unique_relations, raw_text = process_input_and_get_graph_data(
                text, files, urls, input_dir, 
                db_uri, db_username, db_password, 
                clear_existing, visualization_only, 
                llm_model, verbose, chunk_size, chunk_overlap, 
                use_mock, deduplicate_nodes, similarity_threshold, use_embeddings,
                json_output_path
            )
    else:
        logger.info(f"No existing JSON file found at {json_output_path}, processing input sources")
        # Process input sources to generate graph data
        unique_entities, unique_relations, raw_text = process_input_and_get_graph_data(
            text, files, urls, input_dir, 
            db_uri, db_username, db_password, 
            clear_existing, visualization_only, 
            llm_model, verbose, chunk_size, chunk_overlap, 
            use_mock, deduplicate_nodes, similarity_threshold, use_embeddings,
            json_output_path
        )
    
    # Step 2: Generate HTML visualization from the graph data
    if unique_entities is None or unique_relations is None:
        logger.error("Failed to generate or load graph data")
        return False
    
    # Create visualization
    success = visualizer.create_visualization_from_data(
        entities=unique_entities,
        relations=unique_relations,
        output_path=output_path,
        title=f"Combined Graph",
        raw_text=raw_text
    )
    
    if success:
        logger.info(f"Graph visualization saved to {output_path}")
        return True
    else:
        logger.error("Failed to create visualization")
        return False

def get_relation_representation(relation: Dict[str, Any]) -> str:
    """
    Create a comprehensive text representation of a relation including entity properties.
    
    Args:
        relation (Dict[str, Any]): The relation dictionary
        
    Returns:
        str: Text representation of the relation
    """
    from_entity = relation["from_entity"]
    to_entity = relation["to_entity"]
    relation_type = relation["relation"]
    
    # Start with basic relation representation
    representation = f"{from_entity['name']} ({from_entity.get('type', 'UNKNOWN')}) --[{relation_type}]--> {to_entity['name']} ({to_entity.get('type', 'UNKNOWN')})"
    
    # Add source entity properties if they exist
    if "properties" in from_entity and from_entity["properties"]:
        from_props_str = "; ".join([f"{k}: {v}" for k, v in from_entity["properties"].items()])
        representation += f"\nSource properties: {from_props_str}"
    
    # Add target entity properties if they exist
    if "properties" in to_entity and to_entity["properties"]:
        to_props_str = "; ".join([f"{k}: {v}" for k, v in to_entity["properties"].items()])
        representation += f"\nTarget properties: {to_props_str}"
    
    # Add relation properties if they exist
    if "properties" in relation and relation["properties"]:
        rel_props_str = "; ".join([f"{k}: {v}" for k, v in relation["properties"].items()])
        representation += f"\nRelation properties: {rel_props_str}"
    
    return representation

def process_input_and_get_graph_data(
    text: Optional[str] = None,
    files: Optional[List[str]] = None,
    urls: Optional[List[str]] = None,
    input_dir: Optional[str] = None,
    db_uri: str = "bolt://localhost:7687",
    db_username: str = "neo4j",
    db_password: str = "password",
    clear_existing: bool = False,
    visualization_only: bool = False,
    llm_model: str = "gpt-4o",
    verbose: bool = False,
    chunk_size: int = MAX_CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    use_mock: bool = False,
    deduplicate_nodes: bool = False,
    similarity_threshold: float = 0.85,
    use_embeddings: bool = True,
    json_output_path: str = "graph.json"
) -> Tuple[Optional[List[Dict[str, Any]]], Optional[List[Dict[str, Any]]], str]:
    """
    Process input sources and generate graph data.
    
    Args:
        text (Optional[str]): Text to process
        files (Optional[List[str]]): List of files to process
        urls (Optional[List[str]]): List of URLs to process
        input_dir (Optional[str]): Directory containing files to process
        db_uri (str): Neo4j URI
        db_username (str): Neo4j username
        db_password (str): Neo4j password
        clear_existing (bool): Clear existing data
        visualization_only (bool): Skip Neo4j connection and only create visualization
        llm_model (str): LLM model to use
        verbose (bool): Display detailed information about extracted entities and relations
        chunk_size (int): Maximum chunk size for processing large texts
        chunk_overlap (int): Overlap between chunks
        use_mock (bool): Use MockEntityExtractor instead of LLMEntityExtractor
        deduplicate_nodes (bool): Whether to deduplicate similar nodes (default: False)
        similarity_threshold (float): Threshold for string similarity (0.0 to 1.0)
        use_embeddings (bool): Whether to use LLM embeddings for similarity calculation
        json_output_path (str): Path to save the JSON output
        
    Returns:
        Tuple[Optional[List[Dict[str, Any]]], Optional[List[Dict[str, Any]]], str]: 
            Tuple of (entities, relations, raw_text) or (None, None, "") if processing fails
    """
    # Initialize the input processor
    input_processor = InputProcessor(
        input_dir="input",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    # Process input sources
    if input_dir:
        # Process all files in the directory
        result = input_processor.process_directory(input_dir)
    else:
        # Process other input sources
        result = input_processor.process_input(text=text, files=files, urls=urls)
    
    if not result["success"]:
        logger.error("Failed to process input sources")
        return None, None, ""
    
    # Initialize the entity extractor
    if use_mock:
        logger.info("Using MockEntityExtractor for UI debugging")
        entity_extractor = MockEntityExtractor()
    else:
        entity_extractor = LLMEntityExtractor(model=llm_model)
    
    # Initialize the graph builder
    if not visualization_only:
        try:
            graph_builder = GraphBuilder(
                db_uri=db_uri,
                db_username=db_username,
                db_password=db_password,
                llm_model=llm_model
            )
            
            # Note: We'll use clear_existing parameter when calling build_graph_from_text
            
        except (ServiceUnavailable, AuthError) as e:
            logger.error(f"Failed to connect to Neo4j: {str(e)}")
            logger.info("Falling back to visualization-only mode")
            visualization_only = True
    
    # Process each chunk
    all_entities = []
    all_relations = []
    
    for i, chunk in enumerate(result["chunks"]):
        logger.info(f"Processing chunk {i+1}/{len(result['chunks'])} ({len(chunk)} characters)")
        
        if visualization_only:
            # Extract entities and relations without Neo4j
            if use_mock:
                entities, relations = entity_extractor.process_text(chunk)
            else:
                entities, relations = entity_extractor.process_text(chunk)
            all_entities.extend(entities)
            all_relations.extend(relations)
        else:
            # Process the chunk with Neo4j
            # Pass the clear_existing flag only for the first chunk
            should_clear = clear_existing and i == 0
            entities, relations = graph_builder.build_graph_from_text(chunk, clear_existing=should_clear)
            all_entities.extend(entities)
            all_relations.extend(relations)
    
    # Log the extracted entities and relations
    if verbose:
        logger.info(f"Extracted {len(all_entities)} entities:")
        for entity in all_entities:
            logger.info(f"  - {entity['name']} ({entity['type']})")
            
        logger.info(f"Extracted {len(all_relations)} relations:")
        for relation in all_relations:
            logger.info(f"  - {relation['from_entity']['name']} --[{relation['relation']}]--> {relation['to_entity']['name']} (confidence: {relation['confidence']:.2f})")
    else:
        logger.info(f"Successfully extracted {len(all_entities)} entities and {len(all_relations)} relations using {'MockEntityExtractor' if use_mock else 'LLMEntityExtractor'}")
    
    # Deduplicate entities and relations
    unique_entities = deduplicate_entities(all_entities)
    unique_relations = deduplicate_relations(all_relations)
    
    logger.info(f"After deduplication: {len(unique_entities)} entities and {len(unique_relations)} relations")
    
    # Apply node deduplication if enabled
    if deduplicate_nodes:
        logger.info(f"Applying node deduplication with similarity threshold: {similarity_threshold}")
        deduplicator = NodeDeduplicator(
            similarity_threshold=similarity_threshold,
            use_embeddings=use_embeddings
        )
        deduplicated_graph = deduplicator.deduplicate_graph(graph_data)
        unique_entities = deduplicated_graph["data"]["entities"]
        unique_relations = deduplicated_graph["data"]["relations"]
        logger.info(f"After node deduplication: {len(unique_entities)} entities and {len(unique_relations)} relations")
    
    # Add embeddings to relations if get_embedding is available
    if get_embedding is not None:
        logger.info(f"Generating embeddings for {len(unique_relations)} relations")
        embedding_cache = {}  # Cache to avoid redundant API calls
        
        for relation in unique_relations:
            # Generate a textual representation of the relation
            relation_text = get_relation_representation(relation)
            
            # Use cache to avoid redundant API calls
            cache_key = relation_text.lower().strip()
            if cache_key in embedding_cache:
                relation["embedding"] = embedding_cache[cache_key]
            else:
                # Generate embedding for the relation
                embedding = get_embedding(relation_text)
                if embedding:
                    relation["embedding"] = embedding
                    embedding_cache[cache_key] = embedding
                    if verbose:
                        logger.info(f"Generated embedding for relation: {relation['from_entity']['name']} --[{relation['relation']}]--> {relation['to_entity']['name']}")
                else:
                    logger.warning(f"Failed to generate embedding for relation: {relation['from_entity']['name']} --[{relation['relation']}]--> {relation['to_entity']['name']}")
        
        logger.info(f"Successfully generated embeddings for relations")
    else:
        logger.warning("Embedding generation is disabled because get_embedding function is not available")
    
    # Create graph data structure
    raw_text_for_json = result["merged_text"]
    
    # If raw text is very large, truncate it for JSON storage to avoid memory issues
    # but keep enough to make the Show Text button functional
    max_raw_text_length = 100000  # Limit to ~100KB of text
    if len(raw_text_for_json) > max_raw_text_length:
        logger.warning(f"Raw text is very large ({len(raw_text_for_json)} characters). Truncating to {max_raw_text_length} characters for JSON storage.")
        truncation_message = f"\n\n... Text truncated due to size ({len(raw_text_for_json)} total characters) ...\n\n"
        # Keep the beginning and some of the end
        beginning_size = int(max_raw_text_length * 0.7)  # 70% from beginning
        end_size = max_raw_text_length - beginning_size - len(truncation_message)
        raw_text_for_json = raw_text_for_json[:beginning_size] + truncation_message + raw_text_for_json[-end_size:]
    
    graph_data = {
        "schema": {
            "entity_types": list(set(entity["type"] for entity in unique_entities)),
            "relation_types": list(set(relation["relation"] for relation in unique_relations))
        },
        "data": {
            "entities": unique_entities,
            "relations": unique_relations
        },
        "raw_text": raw_text_for_json
    }
    
    # Save the extracted data to a JSON file
    try:
        with open(json_output_path, 'w') as f:
            json.dump(graph_data, f, indent=2)
        logger.info(f"Graph data saved to {json_output_path}")
    except Exception as e:
        logger.error(f"Error saving graph data to JSON: {str(e)}")
    
    return unique_entities, unique_relations, result["merged_text"]

def main():
    parser = argparse.ArgumentParser(description='Graph Construction and Visualization CLI')
    
    # Input sources
    input_group = parser.add_argument_group('Input Sources')
    input_group.add_argument('--text', type=str, help='Input text to process')
    input_group.add_argument('--file', type=str, help='Input file path to process')
    input_group.add_argument('--url', type=str, help='URL to scrape and process')
    
    # Output options
    output_group = parser.add_argument_group('Output Options')
    output_group.add_argument('--output', type=str, default='output.html', 
                         help='Output path for visualization (default: output.html)')
    output_group.add_argument('--json-output', type=str, 
                         help='Output path for graph data in JSON format (optional with --visualization-only, will try to find JSON file with same name as output)')
    output_group.add_argument('--visualization-only', action='store_true', 
                         help='Skip graph construction and only visualize from existing JSON data')
    output_group.add_argument('--verbose', action='store_true', 
                         help='Enable verbose output')
    
    # Question answering options
    qa_group = parser.add_argument_group('Question Answering')
    qa_group.add_argument('--qa', type=str, 
                      help='Question to answer based on the constructed graph')
    qa_group.add_argument('--qa-json', type=str, 
                      help='Path to JSON file containing graph data for question answering')
    qa_group.add_argument('--qa-include-raw-text', action='store_true', 
                      help='Include raw text in context for QA')
    
    # Neo4j options
    neo4j_group = parser.add_argument_group('Neo4j Database Options')
    neo4j_group.add_argument('--neo4j-uri', type=str, default='bolt://localhost:7687', 
                         help='URI for Neo4j connection')
    neo4j_group.add_argument('--neo4j-user', type=str, default='neo4j', 
                         help='Username for Neo4j connection')
    neo4j_group.add_argument('--neo4j-password', type=str, default='password', 
                         help='Password for Neo4j connection')
    
    # API server options
    api_group = parser.add_argument_group('API Server Options')
    api_group.add_argument('--api-server', action='store_true', 
                       help='Start API server for question answering')
    api_group.add_argument('--api-host', type=str, default='localhost', 
                       help='Host for API server (default: localhost)')
    api_group.add_argument('--api-port', type=int, default=8000, 
                       help='Port for API server (default: 8000)')
    api_group.add_argument('--with-visualization', action='store_true',
                       help='Generate visualization in addition to starting API server')
    
    args = parser.parse_args()
    
    # Initialize logger
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Removed validation that requires --json-output with --visualization-only
    # The run_visualization_only function will now auto-detect JSON files
    
    # Handle API server
    if args.api_server:
        logging.info(f"Starting API server on {args.api_host}:{args.api_port}")
        if args.with_visualization and (args.file or args.text or args.url or args.visualization_only):
            # Run both API server and visualization
            logging.info("Generating visualization before starting API server")
            
            # Process the visualization first
            if args.visualization_only:
                run_visualization_only(args)
            else:
                run_graph_construction(args)
            
            # Then start the API server
            start_api_server(args.api_host, args.api_port)
        else:
            # Just run the API server
            start_api_server(args.api_host, args.api_port)
        return

    # Handle Question Answering
    if args.qa:
        run_graph_qa(args.qa, args.qa_json, args.qa_include_raw_text, args.verbose)
        return
    
    # Handle standard graph construction and visualization
    if args.visualization_only:
        run_visualization_only(args)
    else:
        run_graph_construction(args)

def run_graph_qa(question, json_path, include_raw_text=False, verbose=False):
    """Run question answering on a graph from a JSON file.
    
    Args:
        question (str): The question to answer
        json_path (str): Path to the JSON file containing the graph data
        include_raw_text (bool): Whether to include raw text in the context
        verbose (bool): Whether to enable verbose output
    """
    if not question:
        logging.error("No question provided for QA mode")
        return
    
    if not json_path:
        logging.error("No JSON file provided for QA mode")
        return
    
    try:
        # Initialize QA module
        qa = GraphQA(
            json_file_path=json_path
        )
        
        # Run QA
        result = qa.answer_question(
            question=question,
            include_raw_text=include_raw_text
        )
        
        # Print result
        print("\nQuestion:", question)
        print("\nAnswer:", result["answer"])
        
        # Check if we have relations information
        if "relations" in result:
            print(f"\n{len(result['relations'])} relations used to generate this answer.")
            
            if verbose and "relations" in result:
                print("\nTop relations used:")
                for rel in result["relations"][:10]:  # Print top 10 relations
                    print(f"- {rel['source_entity']} --[{rel['relation_type']}]--> {rel['target_entity']} (Score: {rel['relevance_score']:.4f})")
        else:
            print("\nNo specific relations data available for this answer.")
        
        return result
    
    except Exception as e:
        logging.error(f"Error in QA mode: {e}")
        if verbose:
            logging.exception("Detailed error:")
        return None

def run_visualization_only(args):
    """Run visualization from existing JSON data without constructing a new graph.
    
    Args:
        args: Command line arguments
    """
    # Check if JSON file path is provided, otherwise derive it from output path
    if args.json_output:
        json_path = args.json_output
    else:
        # Auto-derive JSON path from output path (replace .html with .json)
        base_output_path = os.path.splitext(args.output)[0]
        possible_json_paths = [
            f"{base_output_path}.json",
            args.output.replace('.html', '.json')
        ]
        
        # Try to find an existing JSON file
        json_path = None
        for path in possible_json_paths:
            if os.path.exists(path):
                json_path = path
                logging.info(f"Found JSON file at {json_path}")
                break
        
        if not json_path:
            logging.error(f"No JSON file found at {possible_json_paths[0]} or {possible_json_paths[1]}. Please provide --json-output parameter.")
            return None
    
    output_path = args.output
    
    try:
        # Check if JSON file exists
        if not os.path.exists(json_path):
            logging.error(f"JSON file not found: {json_path}")
            return None
            
        # Load graph data from JSON
        with open(json_path, 'r', encoding='utf-8') as f:
            graph_data = json.load(f)
        
        # Check if raw_text is empty and try to load from original file if needed
        if 'raw_text' in graph_data and not graph_data['raw_text'] and args.file:
            try:
                with open(args.file, 'r', encoding='utf-8') as f:
                    graph_data['raw_text'] = f.read()
                logging.info(f"Loaded raw text from {args.file}")
            except Exception as e:
                logging.warning(f"Could not load raw text from {args.file}: {e}")
        
        # Create visualization
        visualizer = GraphVisualizer()
        html_path = visualizer.visualize(
            graph_data=graph_data,
            output_path=output_path,
            title="Knowledge Graph Visualization",
            json_path=json_path  # Pass JSON path for QA functionality
        )
        
        logging.info(f"Graph visualization created at {html_path}")
        return html_path
    
    except Exception as e:
        logging.error(f"Error in visualization-only mode: {e}")
        if args.verbose:
            logging.exception("Detailed error:")
        return None

def run_graph_construction(args):
    """Run graph construction and visualization.
    
    Args:
        args: Command line arguments
    """
    # Validate input args
    if not args.text and not args.file and not args.url:
        logging.error("No input source provided (--text, --file, or --url)")
        return
    
    try:
        # Process input
        input_processor = InputProcessor()
        
        if args.text:
            input_data = args.text
            raw_text = args.text
        elif args.file:
            with open(args.file, 'r', encoding='utf-8') as f:
                input_data = f.read()
                raw_text = input_data
        elif args.url:
            input_data = input_processor.process_url(args.url)
            raw_text = input_data
        else:
            logging.error("No valid input source provided")
            return
        
        # Build graph
        builder = GraphBuilder(
            db_uri=args.neo4j_uri,
            db_username=args.neo4j_user,
            db_password=args.neo4j_password,
            llm_model="gpt-4o"  # Use GPT-4o for best extraction
        )
        
        # Extract entities and relations
        entities, relations = builder.build_graph_from_text(input_data)
        
        # Prepare graph data
        schema = builder.schema_generator.get_schema()
        graph_data = {
            "schema": schema,
            "entities": entities,
            "relations": relations,
            "raw_text": raw_text
        }
        
        # Save graph data to JSON if specified
        if args.json_output:
            with open(args.json_output, 'w', encoding='utf-8') as f:
                json.dump(graph_data, f, indent=2)
            logging.info(f"Graph data saved to {args.json_output}")
        
        # Create visualization
        visualizer = GraphVisualizer()
        html_path = visualizer.visualize(
            graph_data=graph_data,
            output_path=args.output,
            title="Knowledge Graph Visualization",
            json_path=args.json_output if args.json_output else None  # Pass JSON path for QA functionality
        )
        
        logging.info(f"Graph visualization created at {html_path}")
        return html_path
    
    except Exception as e:
        logging.error(f"Error in graph construction: {e}")
        if args.verbose:
            logging.exception("Detailed error:")
        return None

def print_help_and_examples():
    """Print help text and examples for the application."""
    parser = argparse.ArgumentParser(description="Graph Database Application")
    parser.print_help()
    print("\nExamples:")
    print("  Process a single file:")
    print("    python -m src.graph_db.app --file input/example.txt --output graph.html")
    print("\n  Process multiple files:")
    print("    python -m src.graph_db.app --file input/file1.txt --output graph.html")
    print("\n  Process all files in a directory:")
    print("    python -m src.graph_db.app --input-dir input --output graph.html")
    print("\n  Visualize an existing JSON graph without building a new one:")
    print("    python -m src.graph_db.app --visualization-only --output example/graph.html")
    print("    # Note: Will automatically look for example/graph.json")
    print("\n  Explicitly specify JSON file for visualization:")
    print("    python -m src.graph_db.app --visualization-only --json-output example/custom_graph.json --output graph.html")
    print("\n  Ask a question using an existing graph:")
    print("    python -m src.graph_db.app --qa \"Who is Elon Musk?\" --qa-json example/musk_graph.json")
    print("\n  Start the API server for handling QA requests in the visualization:")
    print("    python -m src.graph_db.app --api-server --api-host localhost --api-port 8000")
    print("\n  Start API server and generate visualization at the same time:")
    print("    python -m src.graph_db.app --file input/text.md --output output.html --api-server --with-visualization")

if __name__ == '__main__':
    # Load environment variables
    logging.info("Loading environment variables from .env and .env.example files")
    load_dotenv(find_dotenv())
    load_dotenv(find_dotenv('.env.example'))
    
    # Example usage for --help
    if len(sys.argv) == 1:
        print("""
Graph Construction and Visualization CLI

Examples:
  # Process text from file and visualize
  python -m src.graph_db.app --file input/text.md --output output.html
  
  # Process URL content and visualize
  python -m src.graph_db.app --url https://example.com --output output.html
  
  # Only visualize existing JSON data (will look for output.json automatically)
  python -m src.graph_db.app --visualization-only --output output.html
  
  # Explicitly specify JSON file for visualization
  python -m src.graph_db.app --visualization-only --json-output graph_data.json --output vis.html
  
  # Question answering based on graph
  python -m src.graph_db.app --qa "Who is Frodo?" --qa-json example/lotr_graph.json
  
  # Start API server for question answering
  python -m src.graph_db.app --api-server --api-host localhost --api-port 8000
  
  # Start API server and generate visualization at the same time
  python -m src.graph_db.app --file input/text.md --output output.html --api-server --with-visualization
        """)
        sys.exit(0)
    
    main() 