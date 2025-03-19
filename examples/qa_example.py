#!/usr/bin/env python3
"""
Graph-based Question Answering Example

This script demonstrates how to use the graph-based question answering
functionality in the personal graph database system.
"""

import os
import sys
import logging
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Import QA module
from src.graph_db.utils.graph_qa import GraphQA

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

def show_separator(title: str = ""):
    """Print a separator line with an optional title."""
    print("\n" + "="*80)
    if title:
        print(title)
        print("="*80)

def qa_example_with_embeddings():
    """Example using a graph with embeddings."""
    show_separator("Example 1: Question answering with an embedding-enhanced graph (LOTR)")
    
    # Find the most recent json file in the test_output directory with "edge_embeddings" in the name
    json_files = [f for f in os.listdir() if f.endswith('.json') and 'edge_embeddings' in f]
    
    if not json_files:
        print("No embedding-enhanced graph files found. Please generate one first.")
        return
    
    json_file = json_files[0]
    print(f"Using graph file: {json_file}")
    
    # Initialize the QA system
    qa = GraphQA(
        json_file_path=json_file,
        llm_model="gpt-4o",
        llm_provider="openai"
    )
    
    # Ask a question
    questions = [
        "What happened to Gollum at the end of the story?",
        "Who created the One Ring?",
        "What is the relationship between Frodo and Sam?"
    ]
    
    for question in questions:
        result = qa.answer_question(
            question=question,
            top_n=10,
            depth=3,
            include_raw_text=False
        )
        
        print(f"\nQuestion: {result['question']}")
        print(f"Answer: {result['answer']}")
        print(f"Used {result['metadata']['relations_used']} relations to generate the answer")

def qa_example_without_embeddings():
    """Example using a graph without embeddings (using text similarity fallback)."""
    show_separator("Example 2: Question answering with text similarity fallback (Harry Potter)")
    
    json_file = "example/harry_potter.json"
    
    if not os.path.exists(json_file):
        print(f"JSON file not found: {json_file}")
        return
    
    # Initialize the QA system
    qa = GraphQA(
        json_file_path=json_file,
        llm_model="gpt-4o",
        llm_provider="openai"
    )
    
    # Ask a question
    questions = [
        "Tell me about Sirius Black's relationship with Harry Potter",
        "What role does Dumbledore play in Harry's life?",
        "What events happen in Harry Potter and the Deathly Hallows?"
    ]
    
    for question in questions:
        result = qa.answer_question(
            question=question,
            top_n=10,
            depth=3,
            include_raw_text=False
        )
        
        print(f"\nQuestion: {result['question']}")
        print(f"Answer: {result['answer']}")
        print(f"Used {result['metadata']['relations_used']} relations to generate the answer")

def main():
    """Main entry point for the example script."""
    show_separator("Graph-based Question Answering Examples")
    
    # Try the embedding-based example
    try:
        qa_example_with_embeddings()
    except Exception as e:
        logger.error(f"Error in embedding-based example: {str(e)}")
    
    # Try the text similarity example
    try:
        qa_example_without_embeddings()
    except Exception as e:
        logger.error(f"Error in text similarity example: {str(e)}")
    
    show_separator("End of examples")

if __name__ == "__main__":
    main() 