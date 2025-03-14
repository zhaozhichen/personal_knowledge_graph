"""
Node Deduplicator Module

This module provides functionality to deduplicate nodes and edges in a graph.
It merges similar nodes and updates the corresponding edges.
"""

import json
import logging
import re
from typing import Dict, List, Any, Tuple, Set
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)

class NodeDeduplicator:
    def __init__(self, similarity_threshold: float = 0.85):
        """
        Initialize the node deduplicator.
        
        Args:
            similarity_threshold (float): Threshold for string similarity (0.0 to 1.0)
        """
        self.similarity_threshold = similarity_threshold
        self.logger = logging.getLogger(__name__)
    
    def calculate_similarity(self, str1: str, str2: str) -> float:
        """
        Calculate string similarity using SequenceMatcher.
        
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
        
        # Use SequenceMatcher for similarity calculation
        return SequenceMatcher(None, str1_norm, str2_norm).ratio()
    
    def find_similar_nodes(self, entities: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """
        Find groups of similar nodes based on name similarity.
        
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
                
                # Calculate similarity
                similarity = self.calculate_similarity(
                    node_id_to_name[entity_id], 
                    node_id_to_name[other_id]
                )
                
                # If similarity is above threshold, add to group
                if similarity >= self.similarity_threshold:
                    similar_nodes[entity_id].append(other_id)
                    processed_nodes.add(other_id)
                    self.logger.info(f"Merging similar nodes: '{node_id_to_name[entity_id]}' and '{node_id_to_name[other_id]}' (similarity: {similarity:.2f})")
        
        # Remove empty groups
        similar_nodes = {k: v for k, v in similar_nodes.items() if v}
        
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
        
        for relation in relations:
            # Get the from and to entity IDs
            from_id = relation["from_entity"]["id"]
            to_id = relation["to_entity"]["id"]
            
            # Map to new IDs if they exist in the mapping
            new_from_id = node_mapping.get(from_id, from_id)
            new_to_id = node_mapping.get(to_id, to_id)
            
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
        
        return deduplicated_graph

def deduplicate_graph_file(input_file: str, output_file: str = None, similarity_threshold: float = 0.85) -> bool:
    """
    Deduplicate nodes and relations in a graph file.
    
    Args:
        input_file (str): Path to the input JSON file
        output_file (str, optional): Path to the output JSON file. If None, overwrites the input file.
        similarity_threshold (float, optional): Threshold for string similarity (0.0 to 1.0)
        
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
        
        # Create a deduplicator and deduplicate the graph
        deduplicator = NodeDeduplicator(similarity_threshold=similarity_threshold)
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
    
    args = parser.parse_args()
    
    # Deduplicate the graph
    success = deduplicate_graph_file(args.input_file, args.output_file, args.threshold)
    
    if success:
        print(f"Deduplication complete. Output saved to {args.output_file or args.input_file}")
    else:
        print("Deduplication failed. See log for details.") 