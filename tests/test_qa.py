#!/usr/bin/env python3
"""
Test script for graph-based question answering functionality.
"""

import os
import sys
import logging
import argparse
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

def main():
    """Run question answering on a graph JSON file."""
    parser = argparse.ArgumentParser(description="Test graph-based question answering")
    parser.add_argument("--json", required=True, help="Path to the graph JSON file")
    parser.add_argument("--question", required=True, help="Question to answer")
    parser.add_argument("--model", default="gpt-4o", help="LLM model to use")
    parser.add_argument("--provider", default="openai", help="LLM provider (openai, anthropic, etc.)")
    parser.add_argument("--top-n", type=int, default=10, help="Number of top relations to include")
    parser.add_argument("--depth", type=int, default=3, help="Number of relation expansion iterations")
    parser.add_argument("--verbose", action="store_true", help="Display verbose information")
    
    args = parser.parse_args()
    
    # Check if the JSON file exists
    if not os.path.exists(args.json):
        print(f"Error: JSON file not found: {args.json}")
        return False
    
    try:
        # Initialize QA module
        qa = GraphQA(
            json_file_path=args.json,
            llm_model=args.model,
            llm_provider=args.provider
        )
        
        # Get answer
        result = qa.answer_question(
            question=args.question,
            top_n=args.top_n,
            depth=args.depth,
            include_raw_text=False
        )
        
        # Print answer
        print("\n" + "="*80)
        print(f"Question: {result['question']}")
        print("="*80)
        print(f"Answer: {result['answer']}")
        print("="*80)
        print(f"Used {result['metadata']['relations_used']} relations to generate the answer")
        
        if args.verbose:
            print("\nContext used:")
            print(result['metadata']['context'])
        
        return True
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        print(f"Error: {str(e)}")
        return False

if __name__ == "__main__":
    main() 