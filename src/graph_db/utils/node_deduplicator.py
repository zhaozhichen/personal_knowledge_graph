"""
Node Deduplicator Module

This module provides functionality to deduplicate nodes and edges in a graph.
It merges similar nodes and updates the corresponding edges.
"""

import json
import logging
import re
from typing import Dict, List, Any, Tuple, Set, Optional, Union
from difflib import SequenceMatcher
import os
import sys
from pathlib import Path

# Add the project root to the Python path to allow importing from tools
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

# Import embedding functions from tools
try:
    from tools.llm_api import get_embedding, cosine_similarity
except ImportError:
    print(f"Warning: Could not import embedding functions from tools.llm_api. Falling back to SequenceMatcher.", file=sys.stderr)
    get_embedding = None
    cosine_similarity = None

logger = logging.getLogger(__name__)

class NodeDeduplicator:
    def __init__(self, similarity_threshold: float = 0.95, use_embeddings: bool = True):
        """
        Initialize the node deduplicator.
        
        Args:
            similarity_threshold (float): Threshold for string similarity (0.0 to 1.0)
            use_embeddings (bool): Whether to use LLM embeddings for similarity calculation
        """
        self.similarity_threshold = similarity_threshold
        self.use_embeddings = use_embeddings and get_embedding is not None
        self.logger = logging.getLogger(__name__)
        
        # Cache for embeddings to avoid redundant API calls
        self.embedding_cache = {}
        
        if self.use_embeddings:
            self.logger.info("Using LLM embeddings for similarity calculation")
        else:
            self.logger.info("Using SequenceMatcher for similarity calculation")
    
    def get_text_embedding(self, text: str) -> Optional[List[float]]:
        """
        Get embedding for a text string, using cache if available.
        
        Args:
            text (str): Text to get embedding for
            
        Returns:
            Optional[List[float]]: Embedding vector or None if embedding failed
        """
        if not self.use_embeddings:
            return None
            
        # Normalize text for cache key
        cache_key = text.lower().strip()
        
        # Return cached embedding if available
        if cache_key in self.embedding_cache:
            return self.embedding_cache[cache_key]
        
        # Get embedding from API
        embedding = get_embedding(text)
        
        # Cache the embedding
        if embedding:
            self.embedding_cache[cache_key] = embedding
        
        return embedding
    
    def calculate_similarity(self, str1: str, str2: str) -> float:
        """
        Calculate string similarity using LLM embeddings or SequenceMatcher.
        
        Args:
            str1 (str): First string
            str2 (str): Second string
            
        Returns:
            float: Similarity score between 0.0 and 1.0
        """
        # Normalize strings for better comparison
        str1_norm = str1.lower().strip()
        str2_norm = str2.lower().strip()
        
        # If strings are identical after normalization, return 1.0
        if str1_norm == str2_norm:
            return 1.0
        
        # Use LLM embeddings if enabled
        if self.use_embeddings:
            # Get embeddings
            embedding1 = self.get_text_embedding(str1)
            embedding2 = self.get_text_embedding(str2)
            
            # Calculate cosine similarity if embeddings are available
            if embedding1 and embedding2:
                # Cosine similarity ranges from -1 to 1, so normalize to 0 to 1
                similarity = (cosine_similarity(embedding1, embedding2) + 1) / 2
                return similarity
            else:
                self.logger.warning(f"Failed to get embeddings for similarity calculation. Falling back to SequenceMatcher.")
        
        # Fall back to SequenceMatcher if embeddings are not enabled or failed
        return SequenceMatcher(None, str1_norm, str2_norm).ratio()
    
    def get_node_representation(self, entity: Dict[str, Any]) -> str:
        """
        Create a comprehensive text representation of a node including name, type, and properties.
        
        Args:
            entity (Dict[str, Any]): The entity dictionary
            
        Returns:
            str: Text representation of the entity
        """
        # Start with the entity name and type
        representation = f"{entity['name']} ({entity['type']})"
        
        # Add properties if they exist
        if "properties" in entity and entity["properties"]:
            properties_str = "; ".join([f"{k}: {v}" for k, v in entity["properties"].items()])
            representation += f" | Properties: {properties_str}"
        
        return representation
    
    def find_similar_nodes(self, entities: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """
        Find groups of similar nodes based on comprehensive node similarity.
        
        Args:
            entities (List[Dict[str, Any]]): List of entity dictionaries
            
        Returns:
            Dict[str, List[str]]: Dictionary mapping primary node IDs to lists of similar node IDs
        """
        # Create a dictionary to store groups of similar nodes
        similar_nodes = {}
        
        # Create a dictionary of node IDs to node names for quick lookup
        node_id_to_name = {entity["id"]: entity["name"] for entity in entities}
        
        # Create a dictionary of node IDs to node types for type checking
        node_id_to_type = {entity["id"]: entity["type"] for entity in entities}
        
        # Create a dictionary of node IDs to entity objects for representation
        node_id_to_entity = {entity["id"]: entity for entity in entities}
        
        # Create a set to track processed nodes
        processed_nodes = set()
        
        # Iterate through all entities
        for i, entity in enumerate(entities):
            entity_id = entity["id"]
            
            # Skip if this node has already been processed
            if entity_id in processed_nodes:
                continue
            
            # Create a new group with this entity as the primary
            similar_nodes[entity_id] = []
            processed_nodes.add(entity_id)
            
            # Compare with all other entities
            for j, other_entity in enumerate(entities):
                if i == j:
                    continue
                    
                other_id = other_entity["id"]
                
                # Skip if already processed
                if other_id in processed_nodes:
                    continue
                
                # Only compare entities of the same type
                if node_id_to_type[entity_id] != node_id_to_type[other_id]:
                    continue
                
                # Get comprehensive representations of both entities
                entity_repr = self.get_node_representation(node_id_to_entity[entity_id])
                other_repr = self.get_node_representation(node_id_to_entity[other_id])
                
                # Calculate similarity using the comprehensive representations
                similarity = self.calculate_similarity(entity_repr, other_repr)
                
                # If similarity is above threshold, add to group
                if similarity >= self.similarity_threshold:
                    similar_nodes[entity_id].append(other_id)
                    processed_nodes.add(other_id)
                    self.logger.info(f"Merging similar nodes: '{node_id_to_name[entity_id]}' ({node_id_to_type[entity_id]}) and '{node_id_to_name[other_id]}' ({node_id_to_type[other_id]}) (similarity: {similarity:.2f})")
        
        # Remove empty groups
        similar_nodes = {k: v for k, v in similar_nodes.items() if v}
        
        # Log the total number of node groups that will be merged
        if similar_nodes:
            self.logger.info(f"Found {len(similar_nodes)} groups of similar nodes to merge")
            for primary_id, similar_ids in similar_nodes.items():
                self.logger.info(f"  Group with primary node '{node_id_to_name[primary_id]}' ({node_id_to_type[primary_id]}) will merge {len(similar_ids)} similar nodes:")
                for similar_id in similar_ids:
                    self.logger.info(f"    - '{node_id_to_name[similar_id]}' ({node_id_to_type[similar_id]})")
        else:
            self.logger.info("No similar nodes found to merge")
        
        return similar_nodes
    
    def merge_node_properties(self, primary_entity: Dict[str, Any], similar_entities: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Merge properties of similar nodes into the primary node.
        
        Args:
            primary_entity (Dict[str, Any]): The primary entity
            similar_entities (List[Dict[str, Any]]): List of similar entities to merge
            
        Returns:
            Dict[str, Any]: Updated primary entity with merged properties
        """
        # Create a copy of the primary entity
        merged_entity = primary_entity.copy()
        
        # Ensure properties dictionary exists
        if "properties" not in merged_entity:
            merged_entity["properties"] = {}
        
        # Merge properties from similar entities
        for entity in similar_entities:
            if "properties" in entity:
                for key, value in entity["properties"].items():
                    # If property doesn't exist in primary, add it
                    if key not in merged_entity["properties"]:
                        merged_entity["properties"][key] = value
                    # If property exists but has a different value, create a combined value
                    elif merged_entity["properties"][key] != value:
                        # For list properties, extend the list
                        if isinstance(merged_entity["properties"][key], list):
                            if isinstance(value, list):
                                merged_entity["properties"][key].extend(value)
                            else:
                                merged_entity["properties"][key].append(value)
                        # For string properties, combine with separator
                        elif isinstance(merged_entity["properties"][key], str) and isinstance(value, str):
                            merged_entity["properties"][key] = f"{merged_entity['properties'][key]}; {value}"
                        # For other types, keep the primary value
        
        return merged_entity
    
    def update_relations(self, relations: List[Dict[str, Any]], node_mapping: Dict[str, str]) -> List[Dict[str, Any]]:
        """
        Update relations to use the new node IDs after deduplication.
        
        Args:
            relations (List[Dict[str, Any]]): List of relation dictionaries
            node_mapping (Dict[str, str]): Mapping from old node IDs to new node IDs
            
        Returns:
            List[Dict[str, Any]]: Updated relations
        """
        updated_relations = []
        
        # Track unique relation signatures to avoid duplicates
        unique_relations = set()
        
        # Track updated and deduplicated relations for logging
        updated_relation_count = 0
        deduplicated_relation_count = 0
        
        for relation in relations:
            # Get the from and to entity IDs
            from_id = relation["from_entity"]["id"]
            to_id = relation["to_entity"]["id"]
            
            # Map to new IDs if they exist in the mapping
            new_from_id = node_mapping.get(from_id, from_id)
            new_to_id = node_mapping.get(to_id, to_id)
            
            # Check if this relation is being updated
            if from_id != new_from_id or to_id != new_to_id:
                updated_relation_count += 1
                self.logger.info(f"Updating relation: '{relation['from_entity']['name']}' --[{relation['relation']}]--> '{relation['to_entity']['name']}'")
                self.logger.info(f"  From ID: {from_id} -> {new_from_id}")
                self.logger.info(f"  To ID: {to_id} -> {new_to_id}")
            
            # Update the entity IDs in the relation
            updated_relation = relation.copy()
            updated_relation["from_entity"]["id"] = new_from_id
            updated_relation["to_entity"]["id"] = new_to_id
            
            # Create a unique signature for this relation
            relation_signature = (
                new_from_id, 
                new_to_id, 
                relation["relation"]
            )
            
            # Only add if this is a unique relation
            if relation_signature not in unique_relations:
                unique_relations.add(relation_signature)
                updated_relations.append(updated_relation)
            else:
                deduplicated_relation_count += 1
                self.logger.info(f"Deduplicating relation: '{updated_relation['from_entity']['name']}' --[{updated_relation['relation']}]--> '{updated_relation['to_entity']['name']}'")
        
        # Log summary of relation updates
        if updated_relation_count > 0:
            self.logger.info(f"Updated {updated_relation_count} relations with new entity IDs")
        if deduplicated_relation_count > 0:
            self.logger.info(f"Deduplicated {deduplicated_relation_count} duplicate relations")
        
        return updated_relations
    
    def deduplicate_graph(self, graph_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deduplicate nodes and update relations in a graph.
        
        Args:
            graph_data (Dict[str, Any]): Graph data containing entities and relations
            
        Returns:
            Dict[str, Any]: Updated graph data with deduplicated nodes and relations
        """
        # Extract entities and relations
        entities = graph_data["data"]["entities"]
        relations = graph_data["data"]["relations"]
        
        # Create a dictionary for quick entity lookup by ID
        entity_dict = {entity["id"]: entity for entity in entities}
        
        # Find similar nodes
        similar_nodes = self.find_similar_nodes(entities)
        
        # Create a mapping from old node IDs to new node IDs
        node_mapping = {}
        for primary_id, similar_ids in similar_nodes.items():
            for similar_id in similar_ids:
                node_mapping[similar_id] = primary_id
        
        # Log the node mapping
        if node_mapping:
            self.logger.info(f"Created mapping for {len(node_mapping)} nodes to be merged")
        
        # Merge properties of similar nodes
        for primary_id, similar_ids in similar_nodes.items():
            primary_entity = entity_dict[primary_id]
            similar_entities = [entity_dict[similar_id] for similar_id in similar_ids]
            merged_entity = self.merge_node_properties(primary_entity, similar_entities)
            entity_dict[primary_id] = merged_entity
        
        # Create a new list of entities, excluding the merged ones
        deduplicated_entities = [
            entity for entity_id, entity in entity_dict.items()
            if entity_id not in node_mapping
        ]
        
        # Update relations to use the new node IDs
        deduplicated_relations = self.update_relations(relations, node_mapping)
        
        # Create the updated graph data
        deduplicated_graph = graph_data.copy()
        deduplicated_graph["data"]["entities"] = deduplicated_entities
        deduplicated_graph["data"]["relations"] = deduplicated_relations
        
        # Log deduplication statistics
        self.logger.info(f"Deduplication complete: {len(entities)} entities -> {len(deduplicated_entities)} entities")
        self.logger.info(f"Deduplication complete: {len(relations)} relations -> {len(deduplicated_relations)} relations")
        
        # Log the entities that were kept after deduplication
        if len(deduplicated_entities) > 0:
            self.logger.info("Entities after deduplication:")
            for entity in deduplicated_entities:
                self.logger.info(f"  - {entity['name']} ({entity['type']})")
        
        return deduplicated_graph

def deduplicate_graph_file(input_file: str, output_file: str = None, similarity_threshold: float = 0.85, use_embeddings: bool = True, deduplicate: bool = False) -> bool:
    """
    Deduplicate nodes and relations in a graph file.
    
    Args:
        input_file (str): Path to the input JSON file
        output_file (str, optional): Path to the output JSON file. If None, overwrites the input file.
        similarity_threshold (float, optional): Threshold for string similarity (0.0 to 1.0)
        use_embeddings (bool, optional): Whether to use LLM embeddings for similarity calculation
        deduplicate (bool, optional): Whether to perform deduplication. If False, just copies the input file to output.
        
    Returns:
        bool: True if deduplication was successful, False otherwise
    """
    try:
        # Set output file to input file if not specified
        if output_file is None:
            output_file = input_file
        
        # Read the input file
        with open(input_file, 'r', encoding='utf-8') as f:
            graph_data = json.load(f)
        
        # If deduplication is disabled, just write the input to the output
        if not deduplicate:
            logger.info(f"Deduplication is disabled. Copying input to output without changes.")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(graph_data, f, indent=2, ensure_ascii=False)
            logger.info(f"Graph data saved to {output_file}")
            return True
        
        # Create a deduplicator and deduplicate the graph
        deduplicator = NodeDeduplicator(
            similarity_threshold=similarity_threshold,
            use_embeddings=use_embeddings
        )
        deduplicated_graph = deduplicator.deduplicate_graph(graph_data)
        
        # Write the deduplicated graph to the output file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(deduplicated_graph, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Deduplicated graph saved to {output_file}")
        return True
        
    except Exception as e:
        logger.error(f"Error deduplicating graph: {str(e)}")
        return False

if __name__ == "__main__":
    import argparse
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Deduplicate nodes and relations in a graph file')
    parser.add_argument('input_file', help='Path to the input JSON file')
    parser.add_argument('--output-file', '-o', help='Path to the output JSON file (default: overwrite input)')
    parser.add_argument('--threshold', '-t', type=float, default=0.85, help='Similarity threshold (0.0 to 1.0)')
    parser.add_argument('--use-embeddings', action='store_true', help='Use LLM embeddings for similarity calculation')
    parser.add_argument('--deduplicate', '-d', action='store_true', help='Enable deduplication (default: disabled)')
    
    args = parser.parse_args()
    
    # Deduplicate the graph
    success = deduplicate_graph_file(
        args.input_file, 
        args.output_file, 
        args.threshold,
        args.use_embeddings,
        args.deduplicate
    )
    
    if success:
        print(f"Processing complete. Output saved to {args.output_file or args.input_file}")
    else:
        print("Processing failed. See log for details.") 