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

# Simple test endpoint to check if the server is reachable
@app.route('/api/test', methods=['GET'])
def test_endpoint():
    logger.info("Test endpoint reached")
    return jsonify({"status": "ok", "message": "API server is running"}), 200

# Debug endpoint for file path issues
@app.route('/api/debug', methods=['POST'])
def debug_endpoint():
    try:
        logger.info("Debug endpoint reached")
        data = request.json
        file_path = data.get('file_path')
        
        if not file_path:
            return jsonify({"error": "No file_path provided"}), 400
        
        # Check if the file exists
        file_exists = os.path.exists(file_path)
        is_file = os.path.isfile(file_path) if file_exists else False
        is_readable = os.access(file_path, os.R_OK) if file_exists else False
        
        # Get directory of the file and check permissions
        dir_path = os.path.dirname(file_path)
        dir_exists = os.path.exists(dir_path)
        dir_readable = os.access(dir_path, os.R_OK) if dir_exists else False
        
        # Get file size if exists
        file_size = os.path.getsize(file_path) if file_exists and is_file else 0
        
        # Try to read the first few bytes of the file
        file_preview = ""
        if file_exists and is_file and is_readable:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    file_preview = f.read(100)  # Read first 100 characters
            except Exception as e:
                file_preview = f"Error reading file: {str(e)}"
        
        response = {
            "file_path": file_path,
            "exists": file_exists,
            "is_file": is_file,
            "is_readable": is_readable,
            "file_size": file_size,
            "file_preview": file_preview if file_exists and is_file and is_readable else None,
            "directory": {
                "path": dir_path,
                "exists": dir_exists,
                "readable": dir_readable
            },
            "server_info": {
                "cwd": os.getcwd(),
                "python_path": sys.path,
                "platform": sys.platform,
                "user": os.getenv('USER')
            }
        }
        
        return jsonify(response), 200
    except Exception as e:
        logger.error(f"Error in debug endpoint: {e}")
        logger.exception("Detailed error:")
        return jsonify({"error": str(e)}), 500

# Add OPTIONS method handler for CORS preflight requests
@app.route('/api/qa', methods=['OPTIONS'])
@app.route('/api/debug', methods=['OPTIONS'])
def handle_options():
    response = app.make_default_options_response()
    response.headers.add('Access-Control-Allow-Methods', 'POST, OPTIONS')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
    response.headers.add('Access-Control-Max-Age', '3600')
    return response

@app.route('/api/qa', methods=['POST'])
def api_qa():
    """
    API endpoint for question answering.
    
    Expected JSON payload:
    {
        "question": "Who is Frodo?",
        "json_path": "path/to/graph.json",
        "include_raw_text": false,
        "llm_model": "gpt-4o",  // Optional
        "llm_provider": "openai"  // Optional
    }
    """
    try:
        logger.info("Received API request to /api/qa")
        if request.method == 'POST':
            data = request.json
            logger.info(f"Request data: {data}")
            question = data.get('question')
            json_path = data.get('json_path')
            include_raw_text = data.get('include_raw_text', False)
            llm_model = data.get('llm_model')
            llm_provider = data.get('llm_provider')
            
            logger.info(f"Processing question: '{question}'")
            logger.info(f"JSON path: {json_path}")
            logger.info(f"LLM provider: {llm_provider}, model: {llm_model}")
            
            if not question:
                logger.warning("No question provided in request")
                return jsonify({"error": "No question provided"}), 400
            
            if not json_path:
                logger.warning("No JSON path provided in request")
                return jsonify({"error": "No JSON path provided"}), 400
            
            # Initialize QA module with specified LLM model and provider if provided
            qa_params = {"json_file_path": json_path}
            if llm_model:
                qa_params["llm_model"] = llm_model
            if llm_provider:
                qa_params["llm_provider"] = llm_provider
                
            logger.info(f"Initializing QA with params: {qa_params}")
            qa = GraphQA(**qa_params)
            
            # Run QA
            logger.info("Running QA process...")
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
            if "relations" in result and result["relations"]:
                context_lines = [
                    f"- {rel['source_entity']} --[{rel['relation_type']}]--> {rel['target_entity']} (Score: {rel['relevance_score']:.4f})"
                    for rel in result["relations"][:20]  # Include top 20 relations
                ]
                response["context"] = "\n".join(context_lines)
            
            logger.info("Sending successful response")
            return jsonify(response)
            
    except Exception as e:
        logger.error(f"Error in API: {e}")
        logger.exception("Detailed error:")
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