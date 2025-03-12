"""
Schema Generator Module

This module is responsible for dynamically generating Neo4j schema based on text input.
It uses LLM techniques to identify entity types and relationships.
"""

import logging
from typing import Dict, List, Tuple, Set, Any

from ..nlp.llm_entity_extractor import LLMEntityExtractor

class SchemaGenerator:
    def __init__(self, llm_model: str = "gemini-1.5-pro"):
        """
        Initialize the schema generator with specified LLM model.
        
        Args:
            llm_model (str): Name of the LLM model to use
        """
        self.logger = logging.getLogger(__name__)
        self.extractor = LLMEntityExtractor(model=llm_model)
        
    def extract_entity_types(self, text: str) -> Set[str]:
        """
        Extract potential entity types from the text using LLM.
        
        Args:
            text (str): Input text to analyze
            
        Returns:
            Set[str]: Set of identified entity types
        """
        entities, _ = self.extractor.process_text(text)
        entity_types = {entity["type"] for entity in entities}
        self.logger.info(f"Extracted {len(entity_types)} entity types: {', '.join(entity_types)}")
        return entity_types
    
    def extract_relation_types(self, text: str) -> Set[str]:
        """
        Extract potential relation types from the text using LLM.
        
        Args:
            text (str): Input text to analyze
            
        Returns:
            Set[str]: Set of identified relation types
        """
        # Extract entities and relations
        _, relations = self.extractor.process_text(text)
        
        relation_types = {relation["relation"] for relation in relations}
        self.logger.info(f"Extracted {len(relation_types)} relation types: {', '.join(relation_types)}")
        return relation_types
    
    def generate_schema(self, text: str) -> Tuple[Set[str], Set[str]]:
        """
        Generate Neo4j schema based on the input text.
        
        Args:
            text (str): Input text to analyze
            
        Returns:
            Tuple[Set[str], Set[str]]: Tuple of (entity_types, relation_types)
        """
        entity_types = self.extract_entity_types(text)
        relation_types = self.extract_relation_types(text)
        
        return entity_types, relation_types 