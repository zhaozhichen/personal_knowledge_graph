"""
LLM-based Entity Extractor Module

This module uses LLM models to extract entities and relationships from text.
Supports multiple LLM providers with a waterfall approach: Gemini -> Deepseek.
OpenAI is currently disabled.
"""

import os
import json
import logging
import re
from typing import Dict, List, Set, Tuple, Any, Optional
from dotenv import load_dotenv
import google.generativeai as genai
# from openai import OpenAI  # Commented out OpenAI
import requests

# Import prompts
from .prompts import (
    ENTITY_EXTRACTION_PROMPT,
    ENTITY_EXTRACTION_SYSTEM_MESSAGE,
    RELATION_EXTRACTION_PROMPT,
    RELATION_EXTRACTION_SYSTEM_MESSAGE,
    HISTORICAL_ENTITY_EXTRACTION_PROMPT,
    HISTORICAL_RELATION_EXTRACTION_PROMPT
)

# Load environment variables
load_dotenv()

class LLMEntityExtractor:
    def __init__(self, model: str = "gemini-1.5-pro"):
        """
        Initialize the LLM-based entity extractor.
        
        Args:
            model (str): LLM model to use (e.g., "gemini-1.5-pro", "gpt-4o")
        """
        self.model = model
        # Initialize logger first before using it in other methods
        self.logger = logging.getLogger(__name__)
        self.provider = self._determine_provider(model)
        
        # Configure API keys for all providers
        self.gemini_api_key = os.getenv("GOOGLE_API_KEY")
        # self.openai_api_key = os.getenv("OPENAI_API_KEY")  # Commented out OpenAI
        self.deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
        
        # Configure the primary provider
        self._configure_provider(self.provider)
        
        # Define fallback order - try Gemini first, then Deepseek
        # OpenAI is commented out
        self.fallback_providers = ["gemini", "deepseek"]
        
        # Remove the primary provider from fallbacks to avoid duplication
        if self.provider in self.fallback_providers:
            self.fallback_providers.remove(self.provider)
            
        # Ensure the primary provider is tried first, followed by the fallbacks in the specified order
        self.provider_queue = [self.provider]
        
        # Add the fallbacks in the specified order
        for provider in self.fallback_providers:
            if provider not in self.provider_queue:  # Avoid duplicates
                self.provider_queue.append(provider)
    
    def _configure_provider(self, provider: str) -> None:
        """Configure the specified provider with API keys"""
        if provider == "gemini":
            if not self.gemini_api_key:
                self.logger.warning("GOOGLE_API_KEY environment variable not set")
            else:
                genai.configure(api_key=self.gemini_api_key)
        # elif provider == "openai":  # Commented out OpenAI
        #     if not self.openai_api_key:
        #         self.logger.warning("OPENAI_API_KEY environment variable not set")
        #     else:
        #         # Create a new OpenAI client instance
        #         self.openai_client = OpenAI(api_key=self.openai_api_key)
        elif provider == "deepseek":
            if not self.deepseek_api_key:
                self.logger.warning("DEEPSEEK_API_KEY environment variable not set")
    
    def _determine_provider(self, model: str) -> str:
        """
        Determine the provider based on the model name.
        
        Args:
            model (str): Model name
            
        Returns:
            str: Provider name ("gemini", "deepseek", or "openai")
        """
        if model.startswith("gemini"):
            return "gemini"
        elif model.startswith("deepseek"):
            return "deepseek"
        elif model.startswith("gpt") or model.startswith("text-davinci") or model.startswith("claude"):
            # Return "gemini" as default since OpenAI is disabled
            self.logger.warning("OpenAI/Claude is currently disabled for direct API calls. Using Gemini instead.")
            return "gemini"
        else:
            # Default to gemini
            return "gemini"
    
    def _is_historical_content(self, text: str) -> bool:
        """
        Detect if the text is about historical events like World War II.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            bool: True if the text is about historical events, False otherwise
        """
        historical_keywords = [
            "World War", "WWII", "Nazi", "Hitler", "Churchill",
            "Roosevelt", "Stalin", "Axis Powers", "Allied Powers",
            "Pearl Harbor", "D-Day", "Holocaust", "Normandy", 
            "Hiroshima", "Nagasaki", "atomic bomb", "Stalingrad",
            "Operation Barbarossa", "Blitzkrieg", "Cold War",
            "Soviet Union", "Iron Curtain", "Berlin Wall",
            "Great Depression", "American Revolution", "Civil War",
            "French Revolution", "Industrial Revolution"
        ]
        
        # Count the number of historical keywords in the text
        keyword_count = sum(1 for keyword in historical_keywords if keyword.lower() in text.lower())
        
        # If more than 2 keywords are found, consider it historical content
        return keyword_count >= 2
        
    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract entity instances from text using the configured LLM model with fallbacks.
        
        Args:
            text (str): Input text to analyze
            
        Returns:
            List[Dict[str, Any]]: List of extracted entities with their types and properties
        """
        # Check if the text is about historical events
        is_historical = self._is_historical_content(text)
        
        # Format the prompt with the text
        if is_historical:
            self.logger.info("Detected historical content, using specialized historical prompt")
            prompt = HISTORICAL_ENTITY_EXTRACTION_PROMPT.format(text=text)
        else:
            prompt = ENTITY_EXTRACTION_PROMPT.format(text=text)
        
        # Try each provider in the queue until one succeeds
        for provider in self.provider_queue:
            try:
                self.logger.info(f"Attempting to extract entities using {provider} provider")
                self._configure_provider(provider)
                
                if provider == "gemini":
                    return self._extract_entities_gemini(prompt, text)
                elif provider == "deepseek":
                    return self._extract_entities_deepseek(prompt, text)
                # elif provider == "openai":  # Commented out OpenAI
                #     return self._extract_entities_openai(prompt, text)
            except Exception as e:
                self.logger.warning(f"Failed to extract entities with {provider}: {str(e)}")
                continue
        
        # If all providers fail, return empty list
        self.logger.error("All providers failed to extract entities")
        return []
    
    def _extract_json_from_text(self, text: str) -> Dict[str, Any]:
        """
        Extract JSON from text with improved error handling.
        
        Args:
            text (str): Text containing JSON
            
        Returns:
            Dict[str, Any]: Extracted JSON object or empty dict if extraction fails
        """
        # Try to find JSON object in the text
        try:
            # First, try to find JSON object with regex
            json_pattern = r'(\{[\s\S]*\})'
            json_matches = re.findall(json_pattern, text)
            
            if json_matches:
                for json_str in json_matches:
                    try:
                        return json.loads(json_str)
                    except json.JSONDecodeError:
                        continue
            
            # If regex fails, try to find JSON object with string methods
            json_start = text.find('{')
            json_end = text.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = text[json_start:json_end]
                return json.loads(json_str)
            
            # If no JSON object found, try to parse the entire text
            return json.loads(text)
            
        except json.JSONDecodeError:
            self.logger.warning(f"Failed to parse JSON from text: {text[:100]}...")
            
            # Try to fix common JSON formatting issues
            try:
                # Replace single quotes with double quotes
                fixed_text = text.replace("'", '"')
                return json.loads(fixed_text)
            except json.JSONDecodeError:
                pass
                
            # Try to extract JSON with a more lenient approach
            try:
                # Look for "entities" or "relationships" key
                entities_start = text.find('"entities"')
                relationships_start = text.find('"relationships"')
                
                if entities_start >= 0:
                    # Construct a valid JSON object
                    return {"entities": self._extract_list_from_text(text, "entities")}
                elif relationships_start >= 0:
                    # Construct a valid JSON object
                    return {"relationships": self._extract_list_from_text(text, "relationships")}
            except Exception:
                pass
                
            # Return empty dict if all extraction methods fail
            return {}
    
    def _extract_list_from_text(self, text: str, key: str) -> List[Dict[str, Any]]:
        """
        Extract a list from text based on a key.
        
        Args:
            text (str): Text containing a list
            key (str): Key to look for (e.g., "entities" or "relationships")
            
        Returns:
            List[Dict[str, Any]]: Extracted list or empty list if extraction fails
        """
        try:
            # Find the start of the list
            list_start = text.find(f'"{key}"') + len(f'"{key}"')
            list_start = text.find('[', list_start)
            
            # Find the end of the list
            list_end = text.find(']', list_start) + 1
            
            if list_start >= 0 and list_end > list_start:
                list_str = text[list_start:list_end]
                return json.loads(list_str)
            
            return []
        except Exception:
            return []
    
    def _extract_entities_gemini(self, prompt: str, text: str) -> List[Dict[str, Any]]:
        """Extract entities using Gemini API"""
        # Call the Gemini API
        model = genai.GenerativeModel("gemini-1.5-pro")
        response = model.generate_content(
            [
                ENTITY_EXTRACTION_SYSTEM_MESSAGE,
                prompt
            ]
        )
        
        # Parse the response
        content = response.text
        
        # Try to extract JSON from the response
        result = self._extract_json_from_text(content)
        
        # Extract entities from the result
        entities = result.get("entities", [])
        
        # Filter out DATE and TIME entities
        filtered_entities = [entity for entity in entities if entity.get("type") not in ["DATE", "TIME"]]
        
        # Add source information
        for entity in filtered_entities:
            entity["source"] = "llm"
            # Ensure entity has an ID
            if "id" not in entity:
                entity["id"] = str(hash(entity.get("name", "") + entity.get("type", "")))
            
        return filtered_entities
    
    def _extract_entities_deepseek(self, prompt: str, text: str) -> List[Dict[str, Any]]:
        """Extract entities using Deepseek API"""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.deepseek_api_key}"
        }
        
        payload = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": ENTITY_EXTRACTION_SYSTEM_MESSAGE},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.2
        }
        
        response = requests.post(
            "https://api.deepseek.com/v1/chat/completions",
            headers=headers,
            json=payload
        )
        
        if response.status_code != 200:
            raise Exception(f"Deepseek API error: {response.status_code} {response.text}")
        
        # Parse the response
        response_data = response.json()
        content = response_data["choices"][0]["message"]["content"]
        
        # Try to extract JSON from the response
        result = self._extract_json_from_text(content)
        
        # Extract entities from the result
        entities = result.get("entities", [])
        
        # Filter out DATE and TIME entities
        filtered_entities = [entity for entity in entities if entity.get("type") not in ["DATE", "TIME"]]
        
        # Add source information
        for entity in filtered_entities:
            entity["source"] = "llm"
            # Ensure entity has an ID
            if "id" not in entity:
                entity["id"] = str(hash(entity.get("name", "") + entity.get("type", "")))
            
        return filtered_entities
    
    # def _extract_entities_openai(self, prompt: str, text: str) -> List[Dict[str, Any]]:
    #     """Extract entities using OpenAI API"""
    #     # Call the OpenAI API using the client instance
    #     response = self.openai_client.chat.completions.create(
    #         model="gpt-4o",
    #         messages=[
    #             {"role": "system", "content": ENTITY_EXTRACTION_SYSTEM_MESSAGE},
    #             {"role": "user", "content": prompt}
    #         ],
    #         temperature=0.2
    #     )
    #     
    #     # Parse the response
    #     content = response.choices[0].message.content
    #     
    #     # Try to extract JSON from the response
    #     result = self._extract_json_from_text(content)
    #     
    #     # Extract entities from the result
    #     entities = result.get("entities", [])
    #     
    #     # Filter out DATE and TIME entities
    #     filtered_entities = [entity for entity in entities if entity.get("type") not in ["DATE", "TIME"]]
    #     
    #     # Add source information
    #     for entity in filtered_entities:
    #         entity["source"] = "llm"
    #         # Ensure entity has an ID
    #         if "id" not in entity:
    #             entity["id"] = str(hash(entity.get("name", "") + entity.get("type", "")))
    #         
    #     return filtered_entities
    
    def extract_relations(self, text: str, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract relationships between entities using the configured LLM model with fallbacks.
        
        Args:
            text (str): Input text to analyze
            entities (List[Dict[str, Any]]): List of extracted entities
            
        Returns:
            List[Dict[str, Any]]: List of extracted relationships
        """
        # If no entities, return empty list
        if not entities:
            return []
            
        # Create a map of entity names for quick lookup
        entity_map = {entity["name"]: entity for entity in entities}
        
        # Create the entity list for the prompt
        entity_list = ", ".join([f"{entity['name']} ({entity['type']})" for entity in entities])
        
        # Check if the text is about historical events
        is_historical = self._is_historical_content(text)
        
        # Format the prompt with the entity list and text
        if is_historical:
            self.logger.info("Detected historical content, using specialized historical prompt for relations")
            prompt = HISTORICAL_RELATION_EXTRACTION_PROMPT.format(entity_list=entity_list, text=text)
        else:
            prompt = RELATION_EXTRACTION_PROMPT.format(entity_list=entity_list, text=text)
        
        # Try each provider in the queue until one succeeds
        for provider in self.provider_queue:
            try:
                self.logger.info(f"Attempting to extract relations using {provider} provider")
                self._configure_provider(provider)
                
                if provider == "gemini":
                    return self._extract_relations_gemini(prompt, text, entity_map)
                elif provider == "deepseek":
                    return self._extract_relations_deepseek(prompt, text, entity_map)
                # elif provider == "openai":  # Commented out OpenAI
                #     return self._extract_relations_openai(prompt, text, entity_map)
            except Exception as e:
                self.logger.warning(f"Failed to extract relations with {provider}: {str(e)}")
                continue
        
        # If all providers fail, return empty list
        self.logger.error("All providers failed to extract relations")
        return []
    
    def _extract_relations_gemini(self, prompt: str, text: str, entity_map: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract relations using Gemini API"""
        # Call the Gemini API
        model = genai.GenerativeModel("gemini-1.5-pro")
        response = model.generate_content(
            [
                RELATION_EXTRACTION_SYSTEM_MESSAGE,
                prompt
            ]
        )
        
        # Parse the response
        content = response.text
        
        # Try to extract JSON from the response
        result = self._extract_json_from_text(content)
        
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
    
    def _extract_relations_deepseek(self, prompt: str, text: str, entity_map: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract relations using Deepseek API"""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.deepseek_api_key}"
        }
        
        payload = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": RELATION_EXTRACTION_SYSTEM_MESSAGE},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.2
        }
        
        response = requests.post(
            "https://api.deepseek.com/v1/chat/completions",
            headers=headers,
            json=payload
        )
        
        if response.status_code != 200:
            raise Exception(f"Deepseek API error: {response.status_code} {response.text}")
        
        # Parse the response
        response_data = response.json()
        content = response_data["choices"][0]["message"]["content"]
        
        # Try to extract JSON from the response
        result = self._extract_json_from_text(content)
        
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
    
    def _extract_relations_openai(self, prompt: str, text: str, entity_map: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract relations using OpenAI API"""
        # Call the OpenAI API using the client instance
        response = self.openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": RELATION_EXTRACTION_SYSTEM_MESSAGE},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )
        
        # Parse the response
        content = response.choices[0].message.content
        
        # Try to extract JSON from the response
        result = self._extract_json_from_text(content)
        
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
            self.logger.warning("No entities extracted from text")
            return [], []
        
        # Extract relations
        relations = self.extract_relations(text, entities)
        
        if not relations:
            self.logger.warning("No relations extracted from text")
            # Instead of returning empty lists, return all entities without filtering
            return entities, []
        
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