"""
Graph Builder Module

This module coordinates schema generation, entity extraction, and database operations
to build a complete graph from text.
"""

import logging
from typing import Dict, List, Set, Tuple, Any, Optional

from .schema_generator import SchemaGenerator
from .db_manager import Neo4jManager
from ..nlp.llm_entity_extractor import LLMEntityExtractor

class GraphBuilder:
    def __init__(self, 
                 db_uri: str = "bolt://localhost:7687", 
                 db_username: str = "neo4j", 
                 db_password: str = "password",
                 llm_model: str = "gemini-1.5-pro"):
        """
        Initialize the graph builder.
        
        Args:
            db_uri (str): Neo4j connection URI
            db_username (str): Neo4j username
            db_password (str): Neo4j password
            llm_model (str): LLM model to use for entity extraction
        """
        self.logger = logging.getLogger(__name__)
        self.schema_generator = SchemaGenerator()
        self.entity_extractor = LLMEntityExtractor(model=llm_model)
        self.logger.info(f"Using LLM-based entity extractor with model {llm_model}")
        self.db_manager = Neo4jManager(db_uri, db_username, db_password)
        
    def connect_to_db(self) -> bool:
        """
        Connect to the Neo4j database.
        
        Returns:
            bool: True if connection successful, False otherwise
            
        Raises:
            ConnectionError: If unable to connect to the Neo4j database
        """
        success = self.db_manager.connect()
        if not success:
            raise ConnectionError("Failed to connect to Neo4j database")
        return success
    
    def close_db_connection(self):
        """Close the database connection."""
        self.db_manager.close()
        
    def build_graph_from_text(self, text: str, clear_existing: bool = False) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Build a graph from text.
        
        Args:
            text (str): Input text to analyze
            clear_existing (bool): Whether to clear existing data
            
        Returns:
            Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]: Extracted entities and relations
        """
        try:
            # Connect to database if not already connected
            if not self.db_manager.driver:
                try:
                    if not self.connect_to_db():
                        self.logger.error("Failed to connect to database")
                        # Fall back to direct extraction without Neo4j
                        return self._extract_without_neo4j(text)
                except Exception as e:
                    self.logger.error(f"Error connecting to database: {str(e)}")
                    # Fall back to direct extraction without Neo4j
                    return self._extract_without_neo4j(text)
                
            # Clear existing data if requested
            if clear_existing:
                self.db_manager.clear_database()
                
            # Generate schema
            entity_types, relation_types = self.schema_generator.generate_schema(text)
            
            # Create schema in Neo4j
            self.db_manager.create_schema(entity_types, relation_types)
            
            # Extract entities and relations
            entities, relations = self.entity_extractor.process_text(text)
            
            # Create entities in Neo4j
            for entity in entities:
                entity_id = self.db_manager.create_entity(
                    entity_type=entity["type"],
                    name=entity["name"],
                    properties=entity.get("properties", {})
                )
                entity["id"] = entity_id
            
            # Create relations in Neo4j
            for relation in relations:
                from_entity = (relation["from_entity"]["type"], relation["from_entity"]["name"])
                to_entity = (relation["to_entity"]["type"], relation["to_entity"]["name"])
                
                # Include any properties from the relation
                properties = relation.get("properties", {}).copy()
                properties["confidence"] = relation.get("confidence", 1.0)
                
                relation_id = self.db_manager.create_relation(
                    from_entity=from_entity,
                    to_entity=to_entity,
                    relation_type=relation["relation"],
                    properties=properties
                )
                relation["id"] = relation_id
            
            return entities, relations
            
        except Exception as e:
            self.logger.error(f"Error building graph: {str(e)}")
            # Fall back to direct extraction without Neo4j
            return self._extract_without_neo4j(text)
    
    def store_entities_and_relations(self, entities: List[Dict[str, Any]], relations: List[Dict[str, Any]]) -> bool:
        """
        Store pre-extracted entities and relations in Neo4j.
        
        Args:
            entities (List[Dict[str, Any]]): List of entities to store
            relations (List[Dict[str, Any]]): List of relations to store
            
        Returns:
            bool: True if storage successful, False otherwise
        """
        try:
            # Connect to database if not already connected
            if not self.db_manager.driver:
                try:
                    if not self.connect_to_db():
                        self.logger.error("Failed to connect to database")
                        return False
                except Exception as e:
                    self.logger.error(f"Error connecting to database: {str(e)}")
                    return False
            
            # Extract unique entity and relation types
            entity_types = set()
            relation_types = set()
            
            for entity in entities:
                entity_types.add(entity["type"])
                
            for relation in relations:
                relation_types.add(relation["relation"])
            
            # Create schema in Neo4j
            self.db_manager.create_schema(list(entity_types), list(relation_types))
            
            # Create entities in Neo4j
            for entity in entities:
                entity_id = self.db_manager.create_entity(
                    entity_type=entity["type"],
                    name=entity["name"],
                    properties=entity.get("properties", {})
                )
                entity["id"] = entity_id
            
            # Create relations in Neo4j
            for relation in relations:
                from_entity = (relation["from_entity"]["type"], relation["from_entity"]["name"])
                to_entity = (relation["to_entity"]["type"], relation["to_entity"]["name"])
                
                relation_id = self.db_manager.create_relation(
                    from_entity=from_entity,
                    to_entity=to_entity,
                    relation_type=relation["relation"],
                    properties={"confidence": relation.get("confidence", 1.0)}
                )
                relation["id"] = relation_id
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error storing entities and relations: {str(e)}")
            return False
            
    def _extract_without_neo4j(self, text: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Extract entities and relations directly without using Neo4j.
        
        Args:
            text (str): Input text to analyze
            
        Returns:
            Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]: Extracted entities and relations
        """
        self.logger.info("Extracting entities and relations without Neo4j")
        
        # Extract entities and relations
        entities, relations = self.entity_extractor.process_text(text)
        self.logger.info(f"Extracted {len(entities)} entities and {len(relations)} relations")
        
        # Add IDs to entities
        for i, entity in enumerate(entities):
            entity["id"] = f"e{i}"
        
        # Add IDs to relations
        for i, relation in enumerate(relations):
            relation["id"] = f"r{i}"
        
        return entities, relations
            
    def get_graph_data(self) -> Dict[str, Any]:
        """
        Get all graph data from the database.
        
        Returns:
            Dict[str, Any]: Dictionary containing entities and relations
        """
        entities = self.db_manager.get_all_entities()
        relations = self.db_manager.get_all_relations()
        
        return {
            "entities": entities,
            "relations": relations
        } 