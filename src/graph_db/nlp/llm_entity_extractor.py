"""
LLM-based Entity Extractor Module

This module uses Google's Gemini model to extract entities and relationships from text.
"""

import os
import json
import logging
from typing import Dict, List, Set, Tuple, Any, Optional
from dotenv import load_dotenv
import google.generativeai as genai

# Import prompts
from .prompts import (
    ENTITY_EXTRACTION_PROMPT,
    ENTITY_EXTRACTION_SYSTEM_MESSAGE,
    RELATION_EXTRACTION_PROMPT,
    RELATION_EXTRACTION_SYSTEM_MESSAGE
)

# Load environment variables
load_dotenv()

class LLMEntityExtractor:
    def __init__(self, model: str = "gemini-1.5-pro"):
        """
        Initialize the LLM-based entity extractor.
        
        Args:
            model (str): Gemini model to use
        """
        self.model = model
        self.api_key = os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set")
        
        # Configure the Gemini API
        genai.configure(api_key=self.api_key)
        self.logger = logging.getLogger(__name__)
        
    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract entity instances from text using Google's Gemini model.
        
        Args:
            text (str): Input text to analyze
            
        Returns:
            List[Dict[str, Any]]: List of extracted entities with their types and properties
        """
        try:
            # Format the prompt with the text
            prompt = ENTITY_EXTRACTION_PROMPT.format(text=text)
            
            # Call the Gemini API
            model = genai.GenerativeModel(self.model)
            response = model.generate_content(
                [
                    ENTITY_EXTRACTION_SYSTEM_MESSAGE,
                    prompt
                ]
            )
            
            # Parse the response
            content = response.text
            
            # Try to extract JSON from the response
            try:
                # Look for JSON object in the response
                json_start = content.find('{')
                json_end = content.rfind('}') + 1
                if json_start >= 0 and json_end > json_start:
                    json_str = content[json_start:json_end]
                    result = json.loads(json_str)
                else:
                    # If no JSON object found, try to parse the entire response
                    result = json.loads(content)
            except json.JSONDecodeError:
                self.logger.warning(f"Failed to parse JSON from response: {content}")
                # Attempt to extract entities from text response
                result = {"entities": []}
            
            # Extract entities from the result
            entities = result.get("entities", [])
            
            # Filter out DATE and TIME entities
            filtered_entities = [entity for entity in entities if entity.get("type") not in ["DATE", "TIME"]]
            
            # Add source information
            for entity in filtered_entities:
                entity["source"] = "llm"
                
            return filtered_entities
            
        except Exception as e:
            self.logger.error(f"Error extracting entities with LLM: {str(e)}")
            return []
    
    def extract_relations(self, text: str, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract relationships between entities using Google's Gemini model.
        
        Args:
            text (str): Input text to analyze
            entities (List[Dict[str, Any]]): List of extracted entities
            
        Returns:
            List[Dict[str, Any]]: List of extracted relationships
        """
        try:
            # If no entities, return empty list
            if not entities:
                return []
                
            # Create a map of entity names for quick lookup
            entity_map = {entity["name"]: entity for entity in entities}
            
            # Create the entity list for the prompt
            entity_list = ", ".join([f"{entity['name']} ({entity['type']})" for entity in entities])
            
            # Format the prompt with the entity list and text
            prompt = RELATION_EXTRACTION_PROMPT.format(entity_list=entity_list, text=text)
            
            # Call the Gemini API
            model = genai.GenerativeModel(self.model)
            response = model.generate_content(
                [
                    RELATION_EXTRACTION_SYSTEM_MESSAGE,
                    prompt
                ]
            )
            
            # Parse the response
            content = response.text
            
            # Try to extract JSON from the response
            try:
                # Look for JSON object in the response
                json_start = content.find('{')
                json_end = content.rfind('}') + 1
                if json_start >= 0 and json_end > json_start:
                    json_str = content[json_start:json_end]
                    result = json.loads(json_str)
                else:
                    # If no JSON object found, try to parse the entire response
                    result = json.loads(content)
            except json.JSONDecodeError:
                self.logger.warning(f"Failed to parse JSON from response: {content}")
                # Attempt to extract relationships from text response
                result = {"relationships": []}
            
            # Extract relations from the result
            raw_relations = result.get("relationships", [])
            
            # Convert to the expected format
            relations = []
            for rel in raw_relations:
                from_name = rel.get("from_entity")
                to_name = rel.get("to_entity")
                
                # Skip if entities not found
                if from_name not in entity_map or to_name not in entity_map:
                    continue
                    
                relation = {
                    "from_entity": entity_map[from_name],
                    "to_entity": entity_map[to_name],
                    "relation": rel.get("relation", "").upper(),
                    "confidence": rel.get("confidence", 0.5),
                    "properties": rel.get("properties", {})
                }
                relations.append(relation)
            
            return relations
            
        except Exception as e:
            self.logger.error(f"Error extracting relations with LLM: {str(e)}")
            return []
            
    def process_text(self, text: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Process text to extract entities and relations, ensuring no orphan nodes.
        
        Args:
            text (str): Input text to analyze
            
        Returns:
            Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]: Tuple of (entities, relations)
        """
        # Extract entities
        entities = self.extract_entities(text)
        
        if not entities:
            return [], []
        
        # Extract relations
        relations = self.extract_relations(text, entities)
        
        # Track which entities are connected
        connected_entities = set()
        for relation in relations:
            connected_entities.add(relation["from_entity"]["name"])
            connected_entities.add(relation["to_entity"]["name"])
        
        # Filter out orphan entities
        connected_entity_objects = [entity for entity in entities if entity["name"] in connected_entities]
        
        # If we filtered out entities, log it
        if len(connected_entity_objects) < len(entities):
            self.logger.info(f"Filtered out {len(entities) - len(connected_entity_objects)} orphan entities")
        
        return connected_entity_objects, relations 