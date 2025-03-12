"""
Neo4j Database Manager Module

This module is responsible for managing the Neo4j database connection,
creating schemas, and handling graph operations.
"""

import logging
from typing import Dict, List, Set, Tuple, Optional, Any
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable, AuthError

class Neo4jManager:
    def __init__(self, uri: str = "bolt://localhost:7687", 
                 username: str = "neo4j", 
                 password: str = "password"):
        """
        Initialize the Neo4j database manager.
        
        Args:
            uri (str): Neo4j connection URI
            username (str): Neo4j username
            password (str): Neo4j password
        """
        self.uri = uri
        self.username = username
        self.password = password
        self.driver = None
        self.logger = logging.getLogger(__name__)
        
    def connect(self) -> bool:
        """
        Connect to the Neo4j database.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            self.driver = GraphDatabase.driver(
                self.uri, 
                auth=(self.username, self.password)
            )
            # Verify connection
            self.driver.verify_connectivity()
            self.logger.info(f"Connected to Neo4j database at {self.uri}")
            return True
        except (ServiceUnavailable, AuthError) as e:
            self.logger.error(f"Failed to connect to Neo4j: {str(e)}")
            return False
            
    def close(self):
        """Close the database connection."""
        if self.driver:
            self.driver.close()
            self.logger.info("Neo4j connection closed")
            
    def create_schema(self, entity_types: Set[str], relation_types: Set[str]) -> bool:
        """
        Create schema constraints and indexes for entity and relation types.
        
        Args:
            entity_types (Set[str]): Set of entity types to create
            relation_types (Set[str]): Set of relation types to create
            
        Returns:
            bool: True if schema creation successful, False otherwise
        """
        if not self.driver:
            self.logger.error("Not connected to Neo4j database")
            return False
            
        try:
            with self.driver.session() as session:
                # Create constraints for entity types
                for entity_type in entity_types:
                    # Create constraint to ensure entity names are unique within their type
                    query = (
                        f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:{entity_type}) "
                        f"REQUIRE n.name IS UNIQUE"
                    )
                    session.run(query)
                    self.logger.info(f"Created constraint for {entity_type}")
                    
                # Create indexes for relation types if needed
                for relation_type in relation_types:
                    # Create index on relationship type for faster queries
                    query = (
                        f"CREATE INDEX IF NOT EXISTS FOR ()-[r:{relation_type}]-() "
                        f"ON (r.timestamp)"
                    )
                    session.run(query)
                    self.logger.info(f"Created index for {relation_type}")
                    
                return True
        except Exception as e:
            self.logger.error(f"Error creating schema: {str(e)}")
            return False
            
    def create_entity(self, entity_type: str, name: str, properties: Dict[str, Any] = None) -> bool:
        """
        Create a new entity node in the graph.
        
        Args:
            entity_type (str): Type of entity
            name (str): Name of the entity
            properties (Dict[str, Any], optional): Additional properties
            
        Returns:
            bool: True if entity creation successful, False otherwise
        """
        if not self.driver:
            self.logger.error("Not connected to Neo4j database")
            return False
            
        if properties is None:
            properties = {}
            
        try:
            with self.driver.session() as session:
                # Merge ensures we don't create duplicates
                query = (
                    f"MERGE (n:{entity_type} {{name: $name}}) "
                    f"SET n += $properties "
                    f"RETURN n"
                )
                result = session.run(
                    query, 
                    name=name, 
                    properties=properties
                )
                return result.single() is not None
        except Exception as e:
            self.logger.error(f"Error creating entity: {str(e)}")
            return False
            
    def create_relation(self, from_entity: Tuple[str, str], 
                        to_entity: Tuple[str, str], 
                        relation_type: str,
                        properties: Dict[str, Any] = None) -> bool:
        """
        Create a relationship between two entities.
        
        Args:
            from_entity (Tuple[str, str]): (entity_type, name) of source entity
            to_entity (Tuple[str, str]): (entity_type, name) of target entity
            relation_type (str): Type of relationship
            properties (Dict[str, Any], optional): Additional properties
            
        Returns:
            bool: True if relation creation successful, False otherwise
        """
        if not self.driver:
            self.logger.error("Not connected to Neo4j database")
            return False
            
        if properties is None:
            properties = {}
            
        from_type, from_name = from_entity
        to_type, to_name = to_entity
        
        try:
            with self.driver.session() as session:
                query = (
                    f"MATCH (a:{from_type} {{name: $from_name}}), "
                    f"(b:{to_type} {{name: $to_name}}) "
                    f"MERGE (a)-[r:{relation_type}]->(b) "
                    f"SET r += $properties "
                    f"RETURN r"
                )
                result = session.run(
                    query,
                    from_name=from_name,
                    to_name=to_name,
                    properties=properties
                )
                return result.single() is not None
        except Exception as e:
            self.logger.error(f"Error creating relation: {str(e)}")
            return False
            
    def clear_database(self) -> bool:
        """
        Clear all data from the database.
        
        Returns:
            bool: True if database cleared successfully, False otherwise
        """
        if not self.driver:
            self.logger.error("Not connected to Neo4j database")
            return False
            
        try:
            with self.driver.session() as session:
                # Delete all nodes and relationships
                query = "MATCH (n) DETACH DELETE n"
                session.run(query)
                self.logger.info("Database cleared")
                return True
        except Exception as e:
            self.logger.error(f"Error clearing database: {str(e)}")
            return False
            
    def get_all_entities(self) -> List[Dict[str, Any]]:
        """
        Get all entities from the database.
        
        Returns:
            List[Dict[str, Any]]: List of entities with their properties
        """
        if not self.driver:
            self.logger.error("Not connected to Neo4j database")
            return []
            
        try:
            with self.driver.session() as session:
                query = (
                    "MATCH (n) "
                    "RETURN n, labels(n) as types"
                )
                result = session.run(query)
                
                entities = []
                for record in result:
                    node = record["n"]
                    types = record["types"]
                    entity = dict(node.items())
                    entity["types"] = types
                    entities.append(entity)
                    
                return entities
        except Exception as e:
            self.logger.error(f"Error getting entities: {str(e)}")
            return []
            
    def get_all_relations(self) -> List[Dict[str, Any]]:
        """
        Get all relationships from the database.
        
        Returns:
            List[Dict[str, Any]]: List of relationships with their properties
        """
        if not self.driver:
            self.logger.error("Not connected to Neo4j database")
            return []
            
        try:
            with self.driver.session() as session:
                query = (
                    "MATCH (a)-[r]->(b) "
                    "RETURN a.name as from_name, labels(a) as from_type, "
                    "type(r) as relation, r as properties, "
                    "b.name as to_name, labels(b) as to_type"
                )
                result = session.run(query)
                
                relations = []
                for record in result:
                    relation = {
                        "from_name": record["from_name"],
                        "from_type": record["from_type"][0],  # Get first label
                        "relation": record["relation"],
                        "to_name": record["to_name"],
                        "to_type": record["to_type"][0],  # Get first label
                        "properties": dict(record["properties"].items())
                    }
                    relations.append(relation)
                    
                return relations
        except Exception as e:
            self.logger.error(f"Error getting relations: {str(e)}")
            return [] 