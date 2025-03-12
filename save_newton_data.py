import json
import re
from bs4 import BeautifulSoup
import os

def extract_data_from_html(html_file):
    """Extract entity and relation data from the HTML visualization file."""
    with open(html_file, 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    # Extract nodes data using regex
    nodes_match = re.search(r'var nodes = new vis.DataSet\(\[(.*?)\]\);', html_content, re.DOTALL)
    edges_match = re.search(r'var edges = new vis.DataSet\(\[(.*?)\]\);', html_content, re.DOTALL)
    
    nodes_data = []
    edges_data = []
    
    if nodes_match:
        # Process the nodes data
        nodes_json_str = '[' + nodes_match.group(1) + ']'
        # Fix JavaScript object format to valid JSON
        nodes_json_str = re.sub(r'(\w+):', r'"\1":', nodes_json_str)
        nodes_json_str = re.sub(r',\s*}', '}', nodes_json_str)
        
        try:
            nodes_data = json.loads(nodes_json_str)
            print(f"Successfully extracted {len(nodes_data)} nodes")
        except json.JSONDecodeError as e:
            print(f"Error parsing nodes data: {e}")
    else:
        print("No nodes data found in the HTML file")
    
    if edges_match:
        # Process the edges data
        edges_json_str = '[' + edges_match.group(1) + ']'
        # Fix JavaScript object format to valid JSON
        edges_json_str = re.sub(r'(\w+):', r'"\1":', edges_json_str)
        edges_json_str = re.sub(r',\s*}', '}', edges_json_str)
        
        try:
            edges_data = json.loads(edges_json_str)
            print(f"Successfully extracted {len(edges_data)} edges")
        except json.JSONDecodeError as e:
            print(f"Error parsing edges data: {e}")
    else:
        print("No edges data found in the HTML file")
    
    return nodes_data, edges_data

def convert_to_entities_relations(nodes, edges):
    """Convert nodes and edges to entities and relations format."""
    entities = []
    relations = []
    
    # Create a map of node IDs to entities
    node_map = {}
    
    # Process nodes to create entities
    for node in nodes:
        entity = {
            "id": node.get("id"),
            "name": node.get("label"),
            "type": node.get("group")
        }
        entities.append(entity)
        node_map[node.get("id")] = entity
    
    # Process edges to create relations
    for edge in edges:
        from_id = edge.get("from")
        to_id = edge.get("to")
        
        if from_id in node_map and to_id in node_map:
            relation = {
                "from_entity": node_map[from_id],
                "to_entity": node_map[to_id],
                "relation": edge.get("label"),
                "confidence": 1.0  # Default confidence
            }
            relations.append(relation)
    
    return entities, relations

def save_to_json(data, output_file):
    """Save data to a JSON file."""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    print(f"Saved data to {output_file}")

def extract_entity_types(entities):
    """Extract and count entity types."""
    entity_types = {}
    for entity in entities:
        entity_type = entity.get("type")
        if entity_type in entity_types:
            entity_types[entity_type] += 1
        else:
            entity_types[entity_type] = 1
    return entity_types

def extract_relation_types(relations):
    """Extract and count relation types."""
    relation_types = {}
    for relation in relations:
        relation_type = relation.get("relation")
        if relation_type in relation_types:
            relation_types[relation_type] += 1
        else:
            relation_types[relation_type] = 1
    return relation_types

def find_central_entities(relations, top_n=20):
    """Find entities with the most connections."""
    entity_connections = {}
    for relation in relations:
        from_entity = relation["from_entity"]["name"]
        to_entity = relation["to_entity"]["name"]
        
        if from_entity in entity_connections:
            entity_connections[from_entity] += 1
        else:
            entity_connections[from_entity] = 1
            
        if to_entity in entity_connections:
            entity_connections[to_entity] += 1
        else:
            entity_connections[to_entity] = 1
    
    # Sort by number of connections
    sorted_entities = sorted(entity_connections.items(), key=lambda x: x[1], reverse=True)
    return sorted_entities[:top_n]

def main():
    # Input HTML file
    html_file = "output/newton_wiki_chunked.html"
    
    # Output JSON files
    entities_file = "output/newton_wiki_chunked_entities.json"
    relations_file = "output/newton_wiki_chunked_relations.json"
    stats_file = "output/newton_wiki_chunked_stats.json"
    
    # Extract data from HTML
    nodes, edges = extract_data_from_html(html_file)
    
    if nodes and edges:
        print(f"Extracted {len(nodes)} nodes and {len(edges)} edges from HTML")
        
        # Convert to entities and relations
        entities, relations = convert_to_entities_relations(nodes, edges)
        
        print(f"Converted to {len(entities)} entities and {len(relations)} relations")
        
        # Save to JSON files
        save_to_json(entities, entities_file)
        save_to_json(relations, relations_file)
        
        # Extract and save statistics
        entity_types = extract_entity_types(entities)
        relation_types = extract_relation_types(relations)
        central_entities = find_central_entities(relations)
        
        stats = {
            "total_entities": len(entities),
            "total_relations": len(relations),
            "entity_types": entity_types,
            "relation_types": relation_types,
            "central_entities": dict(central_entities)
        }
        
        save_to_json(stats, stats_file)
        
        # Print some statistics
        print("\nEntity types distribution:")
        for entity_type, count in sorted(entity_types.items(), key=lambda x: x[1], reverse=True):
            print(f"  {entity_type}: {count}")
        
        print("\nTop 20 relation types:")
        for relation_type, count in sorted(relation_types.items(), key=lambda x: x[1], reverse=True)[:20]:
            print(f"  {relation_type}: {count}")
        
        print("\nTop 20 central entities:")
        for entity, count in central_entities:
            print(f"  {entity}: {count} connections")
    else:
        print("Failed to extract data from HTML")

if __name__ == "__main__":
    main() 