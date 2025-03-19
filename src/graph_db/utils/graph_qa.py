"""
Graph-based Question Answering Module

This module provides functionality for answering questions using the knowledge graph
by finding relevant relations based on embedding similarity.
"""

import json
import logging
import os
from typing import Dict, List, Any, Optional, Set, Tuple
import heapq

# Try to import embedding functions from tools.llm_api
try:
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
    from tools.llm_api import get_embedding, cosine_similarity, query_llm
except ImportError:
    print(f"Warning: Could not import embedding functions from tools.llm_api. QA will be disabled.", file=sys.stderr)
    get_embedding = None
    cosine_similarity = None
    query_llm = None

logger = logging.getLogger(__name__)

class GraphQA:
    """Class for performing question answering using the knowledge graph."""
    
    def __init__(self, 
                 json_file_path: str,
                 llm_model: str = "gpt-4o",
                 llm_provider: str = "openai"):
        """
        Initialize the Graph QA module.
        
        Args:
            json_file_path (str): Path to the JSON file containing the graph data
            llm_model (str): LLM model to use for question answering
            llm_provider (str): LLM provider to use
        """
        self.json_file_path = json_file_path
        self.graph_data = self._load_graph_data(json_file_path)
        self.llm_model = llm_model
        self.llm_provider = llm_provider
        
        # Verify that embedding functions are available
        if get_embedding is None or cosine_similarity is None or query_llm is None:
            logger.error("Embedding functions are not available. QA will not work.")
            raise ImportError("Required embedding functions not available")
    
    def _load_graph_data(self, json_file_path: str) -> Dict[str, Any]:
        """
        Load graph data from a JSON file.
        
        Args:
            json_file_path (str): Path to the JSON file
            
        Returns:
            Dict[str, Any]: Loaded graph data
        """
        try:
            with open(json_file_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading graph data from JSON: {str(e)}")
            raise
    
    def _embed_query(self, query: str) -> List[float]:
        """
        Generate an embedding for the query.
        
        Args:
            query (str): The question to embed
            
        Returns:
            List[float]: Embedding vector for the query
        """
        embedding = get_embedding(query)
        if embedding is None:
            logger.error("Failed to generate embedding for query")
            raise ValueError("Failed to generate embedding for query")
        return embedding
    
    def _calculate_similarity(self, 
                             query_embedding: List[float], 
                             relation_embedding: List[float]) -> float:
        """
        Calculate the similarity between the query and a relation.
        
        Args:
            query_embedding (List[float]): Query embedding
            relation_embedding (List[float]): Relation embedding
            
        Returns:
            float: Similarity score (0.0 to 1.0)
        """
        # Convert from -1...1 range to 0...1 range
        if query_embedding and relation_embedding:
            raw_similarity = cosine_similarity(query_embedding, relation_embedding)
            # Normalize from [-1, 1] to [0, 1]
            return (raw_similarity + 1) / 2
        return 0.0
    
    def _get_relation_by_id(self, relation_id: int) -> Dict[str, Any]:
        """
        Get a relation by its index in the relations list.
        
        Args:
            relation_id (int): Index of the relation
            
        Returns:
            Dict[str, Any]: The relation
        """
        return self.graph_data["data"]["relations"][relation_id]
    
    def _find_connected_relations(self, 
                                 relation_ids: Set[int],
                                 excluded_ids: Set[int]) -> List[Tuple[int, Dict[str, Any]]]:
        """
        Find relations that connect to any entity in the given relations.
        
        Args:
            relation_ids (Set[int]): Set of relation IDs to find connections for
            excluded_ids (Set[int]): Set of relation IDs to exclude
            
        Returns:
            List[Tuple[int, Dict[str, Any]]]: List of (relation_id, relation) tuples
        """
        # Get all entities mentioned in the given relations
        entities = set()
        for rel_id in relation_ids:
            relation = self._get_relation_by_id(rel_id)
            entities.add(relation["from_entity"]["name"])
            entities.add(relation["to_entity"]["name"])
        
        # Find relations that connect to these entities
        connected_relations = []
        for i, relation in enumerate(self.graph_data["data"]["relations"]):
            # Skip if this relation is already in the set
            if i in relation_ids or i in excluded_ids:
                continue
                
            # Check if this relation connects to any entity in our set
            if (relation["from_entity"]["name"] in entities or 
                relation["to_entity"]["name"] in entities):
                connected_relations.append((i, relation))
                
        return connected_relations
    
    def _create_context_from_relations(self, relation_ids: Set[int], raw_text: Optional[str] = None) -> str:
        """
        Create a context string from the selected relations.
        
        Args:
            relation_ids (Set[int]): Set of relation IDs to include in the context
            raw_text (Optional[str]): The raw text to include at the end of the context
            
        Returns:
            str: Context string for question answering
        """
        context = "Knowledge Graph Context:\n\n"
        
        # Add relations to context
        for i, rel_id in enumerate(relation_ids):
            relation = self._get_relation_by_id(rel_id)
            from src.graph_db.app import get_relation_representation
            relation_text = get_relation_representation(relation)
            context += f"{i+1}. {relation_text}\n\n"
        
        return context
    
    def answer_question(self, 
                        question: str, 
                        top_n: int = 10, 
                        depth: int = 3,
                        include_raw_text: bool = False) -> Dict[str, Any]:
        """
        Answer a question using the knowledge graph.
        
        Args:
            question (str): The question to answer
            top_n (int): Number of top relations to select in each iteration
            depth (int): Number of expansion iterations
            include_raw_text (bool): Whether to include the raw text in the context
            
        Returns:
            Dict[str, Any]: Dictionary containing the answer and metadata
        """
        if not get_embedding or not cosine_similarity or not query_llm:
            raise ImportError("Required embedding functions not available")
            
        # Step 1: Embed the query
        logger.info(f"Generating embedding for question: {question}")
        query_embedding = self._embed_query(question)
        
        # Step 2: Calculate similarity with all relations
        relation_similarities = []
        for i, relation in enumerate(self.graph_data["data"]["relations"]):
            # Skip relations without embeddings
            if "embedding" not in relation:
                logger.warning(f"Relation {i} does not have an embedding, skipping")
                continue
                
            similarity = self._calculate_similarity(query_embedding, relation["embedding"])
            relation_similarities.append((i, similarity))
        
        # Sort by similarity (descending)
        relation_similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Step 3: Pick top N relations
        context_relation_ids = set()
        added_in_iteration = set()
        
        # Add top N most similar relations
        for i, (rel_id, similarity) in enumerate(relation_similarities):
            if i >= top_n:
                break
            context_relation_ids.add(rel_id)
            added_in_iteration.add(rel_id)
            
        logger.info(f"Added {len(added_in_iteration)} initial relations to context")
        
        # Step 4-6: Expand context by adding connected relations
        for d in range(depth):
            # Find connected relations
            connected_relations = self._find_connected_relations(
                added_in_iteration, 
                context_relation_ids - added_in_iteration
            )
            
            # Calculate similarity for connected relations
            connected_similarities = []
            for rel_id, relation in connected_relations:
                # Skip relations without embeddings
                if "embedding" not in relation:
                    continue
                    
                similarity = self._calculate_similarity(query_embedding, relation["embedding"])
                connected_similarities.append((rel_id, relation, similarity))
            
            # Sort by similarity
            connected_similarities.sort(key=lambda x: x[2], reverse=True)
            
            # Reset the added in this iteration set
            added_in_iteration = set()
            
            # Add top N to context
            for i, (rel_id, relation, similarity) in enumerate(connected_similarities):
                if i >= top_n:
                    break
                context_relation_ids.add(rel_id)
                added_in_iteration.add(rel_id)
            
            logger.info(f"Iteration {d+1}: Added {len(added_in_iteration)} new relations to context")
            
            # Break if no new relations were added
            if not added_in_iteration:
                logger.info(f"No new relations added in iteration {d+1}, stopping expansion")
                break
        
        # Step 7: Build context string
        context = self._create_context_from_relations(
            context_relation_ids,
            self.graph_data.get("raw_text") if include_raw_text else None
        )
        
        # Step 8: Generate answer using LLM
        prompt = f"""You are a helpful question answering system that has access to a knowledge graph. 
Answer the question based on the provided context from the knowledge graph.

{context}

Question: {question}

Please provide a comprehensive answer that directly addresses the question based ONLY on the information in the context.
If you don't know the answer or the context doesn't contain relevant information, simply state that you don't know based on the available information.
"""
        
        logger.info(f"Generating answer using LLM ({self.llm_model})")
        answer = query_llm(prompt, model=self.llm_model, provider=self.llm_provider)
        
        # Return answer and metadata
        result = {
            "question": question,
            "answer": answer,
            "metadata": {
                "relations_used": len(context_relation_ids),
                "context_size": len(context),
                "context": context  # Include context for debugging
            }
        }
        
        return result 