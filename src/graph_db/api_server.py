"""
API Server for Graph QA

This module provides a simple API server to handle question answering requests from the HTML visualization.
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from flask import Flask, request, jsonify
from flask_cors import CORS

# Import the GraphQA class
from src.graph_db.utils.graph_qa import GraphQA

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests

@app.route('/api/ask', methods=['POST'])
def ask_question():
    """API endpoint for asking questions about the graph."""
    try:
        data = request.get_json()
        question = data.get('question')
        json_path = data.get('json_path')
        include_raw_text = data.get('include_raw_text', False)
        
        if not question:
            return jsonify({"error": "No question provided"}), 400
        
        if not json_path:
            return jsonify({"error": "No JSON path provided"}), 400
        
        # Initialize QA module
        qa = GraphQA(
            json_file_path=json_path,
            verbose=True
        )
        
        # Run QA
        result = qa.answer_question(
            question=question,
            include_raw_text=include_raw_text
        )
        
        # Prepare response
        response = {
            "answer": result["answer"],
            "question": question,
            "context": None
        }
        
        # Add relations to context if available
        if "relations" in result:
            context_lines = [
                f"- {rel['source_entity']} --[{rel['relation_type']}]--> {rel['target_entity']} (Score: {rel['relevance_score']:.4f})"
                for rel in result["relations"][:20]  # Include top 20 relations
            ]
            response["context"] = "\n".join(context_lines)
        
        return jsonify(response)
    
    except Exception as e:
        logging.error(f"Error in API: {e}")
        logging.exception("Detailed error:")
        return jsonify({"error": str(e)}), 500

def start_api_server(host='localhost', port=8000):
    """Start the API server.
    
    Args:
        host (str): Host to bind to
        port (int): Port to bind to
    """
    logging.info(f"Starting API server on {host}:{port}")
    app.run(host=host, port=port, debug=False, threaded=True)

if __name__ == '__main__':
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Start server
    start_api_server() 