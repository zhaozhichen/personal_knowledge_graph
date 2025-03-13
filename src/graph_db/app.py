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
from src.graph_db.input.input_processor import InputProcessor
from src.graph_db.nlp.llm_entity_extractor import LLMEntityExtractor
from src.graph_db.nlp.mock_entity_extractor import MockEntityExtractor

# Import ConnectionError for exception handling
from neo4j.exceptions import ServiceUnavailable, AuthError

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

def detect_historical_content(text: str) -> bool:
    """
    Detect if the text is about historical events.
    
    Args:
        text (str): Text to analyze
        
    Returns:
        bool: True if the text is about historical events, False otherwise
    """
    historical_keywords = [
        "World War", "WWII", "World War 2", "World War II",
        "Nazi Germany", "Adolf Hitler", "Winston Churchill",
        "Franklin D. Roosevelt", "Joseph Stalin", "Axis Powers", 
        "Allied Powers", "Pearl Harbor", "D-Day", "Holocaust",
        "Normandy", "Hiroshima", "Nagasaki", "atomic bomb",
        "Stalingrad", "Operation Barbarossa", "Blitzkrieg",
        "Cold War", "Soviet Union", "Iron Curtain", "Berlin Wall",
        "Great Depression", "American Revolution", "Civil War",
        "French Revolution", "Industrial Revolution", "Ancient Rome",
        "Medieval", "Renaissance", "Enlightenment", "Ottoman Empire",
        "Byzantine Empire", "Mongol Empire", "British Empire",
        "colonization", "imperialism", "monarchy", "dynasty"
    ]
    
    # Count the number of historical keywords in the text
    keyword_count = sum(1 for keyword in historical_keywords if keyword.lower() in text.lower())
    
    # If more than 3 keywords are found, consider it historical content
    return keyword_count >= 3

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

def process_input_sources(text: Optional[Union[str, List[str]]] = None,
                         files: Optional[Union[str, List[str]]] = None,
                         urls: Optional[Union[str, List[str]]] = None,
                         input_dir: Optional[str] = None,
                         output_path: str = "graph.html",
                         visualization_only: bool = True,
                         llm_model: str = "gpt-4o",
                         verbose: bool = False,
                         chunk_size: int = MAX_CHUNK_SIZE,
                         chunk_overlap: int = CHUNK_OVERLAP,
                         use_mock: bool = False) -> bool:
    """
    Process various input sources and generate a graph.
    
    Args:
        text (str or List[str]): Text chunk(s) to process
        files (str or List[str]): File path(s) to process
        urls (str or List[str]): URL(s) to process
        input_dir (str): Directory containing files to process
        output_path (str): Path to save the visualization
        visualization_only (bool): Whether to skip Neo4j connection
        llm_model (str): LLM model to use
        verbose (bool): Whether to display detailed information
        chunk_size (int): Maximum chunk size
        chunk_overlap (int): Overlap between chunks
        use_mock (bool): Whether to use MockEntityExtractor instead of LLMEntityExtractor
        
    Returns:
        bool: True if processing successful, False otherwise
    """
    # Initialize the input processor
    processor = InputProcessor(
        input_dir="input",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    # Log input information
    if files:
        if isinstance(files, list):
            logger.info(f"Processing {len(files)} files: {', '.join(files)}")
        else:
            logger.info(f"Processing file: {files}")
    
    if urls:
        if isinstance(urls, list):
            logger.info(f"Processing {len(urls)} URLs: {', '.join(urls)}")
        else:
            logger.info(f"Processing URL: {urls}")
    
    # Process input directory if specified
    if input_dir:
        logger.info(f"Processing all files in directory: {input_dir}")
        result = processor.process_directory(input_dir)
    else:
        # Process other input sources
        result = processor.process_input(text=text, files=files, urls=urls)
    
    if not result["success"]:
        logger.error(f"Error processing input: {result.get('error', 'Unknown error')}")
        return False
    
    # Check if the content is about historical events
    is_historical = detect_historical_content(result["merged_text"])
    if is_historical:
        logger.info("Detected historical content, using specialized historical prompts")
    
    # Initialize the visualizer
    visualizer = GraphVisualizer()
    
    # Lists to store all entities and relations
    all_entities = []
    all_relations = []
    
    # Process each chunk
    for i, chunk in enumerate(result["chunks"]):
        logger.info(f"Processing chunk {i+1}/{len(result['chunks'])} ({len(chunk)} characters)")
        
        # Extract entities and relations
        entities, relations = process_with_entity_extractor(
            chunk, 
            llm_model, 
            verbose, 
            use_mock
        )
        
        # Add to the combined lists
        all_entities.extend(entities)
        all_relations.extend(relations)
    
    # Deduplicate entities and relations
    unique_entities = deduplicate_entities(all_entities)
    unique_relations = deduplicate_relations(all_relations)
    
    logger.info(f"After deduplication: {len(unique_entities)} entities and {len(unique_relations)} relations")
    
    # Save the extracted data to a JSON file
    json_output_path = output_path.replace('.html', '.json')
    try:
        with open(json_output_path, 'w') as f:
            json.dump({
                "entities": unique_entities,
                "relations": unique_relations
            }, f, indent=2)
        logger.info(f"Graph data saved to {json_output_path}")
    except Exception as e:
        logger.error(f"Error saving graph data to JSON: {str(e)}")
    
    # Create visualization
    success = visualizer.create_visualization_from_data(
        entities=unique_entities,
        relations=unique_relations,
        output_path=output_path,
        title=f"Combined Graph",
        raw_text=result["merged_text"]
    )
    
    if success:
        logger.info(f"Graph visualization saved to {output_path}")
        return True
    else:
        logger.error("Failed to create visualization")
        return False

def main():
    """Main entry point for the application."""
    # Define global variables
    global MAX_CHUNK_SIZE, CHUNK_OVERLAP
    
    parser = argparse.ArgumentParser(description="Graph Database Application")
    parser.add_argument("--text", help="Text to process")
    parser.add_argument("--file", action="append", help="File to process (can be specified multiple times, e.g., --file file1.txt --file file2.txt)")
    parser.add_argument("--url", action="append", help="URL to process (can be specified multiple times, e.g., --url url1 --url url2)")
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
    parser.add_argument("--use-mock", action="store_true", help="Use MockEntityExtractor instead of LLMEntityExtractor")
    
    args = parser.parse_args()
    
    # Update chunk size and overlap if specified
    if args.chunk_size:
        MAX_CHUNK_SIZE = args.chunk_size
    if args.chunk_overlap:
        CHUNK_OVERLAP = args.chunk_overlap
    
    if not args.text and not args.file and not args.url and not args.input_dir:
        parser.print_help()
        print("\nExamples:")
        print("  Process a single file:")
        print("    python -m src.graph_db.app --file input/example.txt --output graph.html --visualization-only")
        print("\n  Process multiple files:")
        print("    python -m src.graph_db.app --file input/file1.txt --file input/file2.txt --output graph.html --visualization-only")
        print("\n  Process all files in a directory:")
        print("    python -m src.graph_db.app --input-dir input --output graph.html --visualization-only")
        return False
    
    # Process all input sources using the new unified function
    return process_input_sources(
        text=args.text,
        files=args.file,
        urls=args.url,
        input_dir=args.input_dir,
        output_path=args.output,
        visualization_only=args.visualization_only,
        llm_model=args.llm_model,
        verbose=args.verbose,
        chunk_size=MAX_CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        use_mock=args.use_mock
    )

if __name__ == "__main__":
    main() 