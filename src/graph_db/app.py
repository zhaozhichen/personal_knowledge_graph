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
                uri=db_uri,
                username=db_username,
                password=db_password,
                entity_extractor=entity_extractor
            )
            
            # Clear existing data if requested
            if clear_existing:
                graph_builder.clear_database()
                
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
            entities, relations = graph_builder.process_text(chunk)
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
    graph_data = {
        "schema": {
            "entity_types": list(set(entity["type"] for entity in unique_entities)),
            "relation_types": list(set(relation["relation"] for relation in unique_relations))
        },
        "data": {
            "entities": unique_entities,
            "relations": unique_relations
        },
        "raw_text": result["merged_text"]
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
    """Main entry point for the application."""
    # Define global variables
    global MAX_CHUNK_SIZE, CHUNK_OVERLAP
    
    parser = argparse.ArgumentParser(description="Graph Database Application")
    
    # Input source arguments
    parser.add_argument("--text", help="Text to process")
    parser.add_argument("--file", action="append", help="File to process (can be specified multiple times, e.g., --file file1.txt --file file2.txt)")
    parser.add_argument("--url", action="append", help="URL to process (can be specified multiple times, e.g., --url url1 --url url2)")
    parser.add_argument("--input-dir", help="Directory containing files to process")
    
    # Output arguments
    parser.add_argument("--output", default="graph.html", help="Output file path")
    
    # Neo4j connection arguments
    parser.add_argument("--db-uri", default="bolt://localhost:7687", help="Neo4j URI")
    parser.add_argument("--db-user", default="neo4j", help="Neo4j username")
    parser.add_argument("--db-pass", default="password", help="Neo4j password")
    parser.add_argument("--clear", action="store_true", help="Clear existing data")
    
    # Processing arguments
    parser.add_argument("--llm-model", default="gpt-4o", help="LLM model to use")
    parser.add_argument("--visualization-only", action="store_true", help="Skip Neo4j connection and only create visualization")
    parser.add_argument("--verbose", action="store_true", help="Display detailed information about extracted entities and relations")
    parser.add_argument("--chunk-size", type=int, default=MAX_CHUNK_SIZE, help=f"Maximum chunk size for processing large texts (default: {MAX_CHUNK_SIZE})")
    parser.add_argument("--chunk-overlap", type=int, default=CHUNK_OVERLAP, help=f"Overlap between chunks (default: {CHUNK_OVERLAP})")
    parser.add_argument("--use-mock", action="store_true", help="Use MockEntityExtractor instead of LLMEntityExtractor")
    parser.add_argument("--dedup", action="store_true", help="Enable node deduplication")
    parser.add_argument("--similarity-threshold", type=float, default=0.85, help="Threshold for string similarity in node deduplication (0.0 to 1.0)")
    parser.add_argument("--use-embeddings", action="store_true", help="Use LLM embeddings for similarity calculation in node deduplication")
    
    # QA-specific arguments
    qa_group = parser.add_argument_group('Question Answering')
    qa_group.add_argument("--qa", help="Enable question answering mode and specify the question")
    qa_group.add_argument("--qa-json", help="JSON file to use for question answering (defaults to the JSON version of the output file)")
    qa_group.add_argument("--qa-top-n", type=int, default=10, help="Number of top relations to include in each expansion iteration")
    qa_group.add_argument("--qa-depth", type=int, default=3, help="Number of relation expansion iterations")
    qa_group.add_argument("--qa-include-raw-text", action="store_true", help="Include raw text in QA context")
    qa_group.add_argument("--qa-model", help="LLM model to use for question answering (defaults to --llm-model)")
    qa_group.add_argument("--qa-provider", default="openai", help="LLM provider to use for question answering")
    
    args = parser.parse_args()
    
    # Update chunk size and overlap if specified
    if args.chunk_size:
        MAX_CHUNK_SIZE = args.chunk_size
    if args.chunk_overlap:
        CHUNK_OVERLAP = args.chunk_overlap
    
    # If in QA mode, perform question answering
    if args.qa:
        return run_graph_qa(args)
    
    # Otherwise, perform graph construction
    return run_graph_construction(args)

def run_graph_qa(args):
    """
    Run the graph-based question answering functionality.
    
    Args:
        args: Command line arguments
        
    Returns:
        bool: True if successful, False otherwise
    """
    # Determine which JSON file to use
    json_file_path = args.qa_json
    if not json_file_path:
        # Default to the JSON version of the output file
        json_file_path = args.output.replace('.html', '.json')
        
    if not os.path.exists(json_file_path):
        logger.error(f"JSON file not found: {json_file_path}")
        print(f"Error: JSON file not found: {json_file_path}")
        print("You must first generate a graph before using the QA functionality.")
        return False
        
    try:
        # Initialize QA module
        qa_model = args.qa_model if args.qa_model else args.llm_model
        qa = GraphQA(
            json_file_path=json_file_path,
            llm_model=qa_model,
            llm_provider=args.qa_provider
        )
        
        # Get answer
        result = qa.answer_question(
            question=args.qa,
            top_n=args.qa_top_n,
            depth=args.qa_depth,
            include_raw_text=args.qa_include_raw_text
        )
        
        # Print answer
        print("\n" + "="*80)
        print(f"Question: {result['question']}")
        print("="*80)
        print(f"Answer: {result['answer']}")
        print("="*80)
        print(f"Used {result['metadata']['relations_used']} relations to generate the answer")
        
        if args.verbose:
            print("\nContext used:")
            print(result['metadata']['context'])
            
        return True
    except Exception as e:
        logger.error(f"Error in QA mode: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        print(f"Error: {str(e)}")
        return False

def run_graph_construction(args):
    """
    Run the graph construction and visualization functionality.
    
    Args:
        args: Command line arguments
        
    Returns:
        bool: True if successful, False otherwise
    """
    if not args.text and not args.file and not args.url and not args.input_dir:
        print_help_and_examples()
        return False
    
    # Process all input sources using the unified function
    return process_input_sources(
        text=args.text,
        files=args.file,
        urls=args.url,
        input_dir=args.input_dir,
        output_path=args.output,
        db_uri=args.db_uri,
        db_username=args.db_user,
        db_password=args.db_pass,
        clear_existing=args.clear,
        visualization_only=args.visualization_only,
        llm_model=args.llm_model,
        verbose=args.verbose,
        chunk_size=MAX_CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        use_mock=args.use_mock,
        deduplicate_nodes=args.dedup,
        similarity_threshold=args.similarity_threshold,
        use_embeddings=args.use_embeddings,
    )

def print_help_and_examples():
    """Print help text and examples for the application."""
    parser = argparse.ArgumentParser(description="Graph Database Application")
    parser.print_help()
    print("\nExamples:")
    print("  Process a single file:")
    print("    python -m src.graph_db.app --file input/example.txt --output graph.html --visualization-only")
    print("\n  Process multiple files:")
    print("    python -m src.graph_db.app --file input/file1.txt --file input/file2.txt --output graph.html --visualization-only")
    print("\n  Process all files in a directory:")
    print("    python -m src.graph_db.app --input-dir input --output graph.html --visualization-only")
    print("\n  Ask a question using an existing graph:")
    print("    python -m src.graph_db.app --qa \"Who is Elon Musk?\" --output graph.html")

if __name__ == "__main__":
    main() 