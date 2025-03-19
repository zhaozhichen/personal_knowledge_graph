"""
Graph Visualizer Module

This module is responsible for visualizing the graph in HTML format.
"""

import os
import json
import logging
import html
from typing import Dict, List, Any, Optional
import networkx as nx
from pyvis.network import Network
import textwrap
import re
import uuid

class GraphVisualizer:
    def __init__(self, height: str = "800px", width: str = "100%"):
        """
        Initialize the graph visualizer.
        
        Args:
            height (str): Height of the visualization
            width (str): Width of the visualization
        """
        self.height = height
        self.width = width
        self.logger = logging.getLogger(__name__)
        
        # Define entity colors - updated to match exact entity types from LLM
        self.entity_colors = {
            "PERSON": "#FF5733",        # Red-orange
            "ORGANIZATION": "#33A8FF",  # Blue
            "LOCATION": "#33FF57",      # Green
            "DATE": "#FF33A8",          # Pink
            "FACILITY": "#6A5ACD",      # Slate blue
            "PRODUCT": "#00CED1",       # Turquoise
            "MONEY": "#32CD32",         # Lime green
            "EVENT": "#FFD700",         # Gold
            "WORK_OF_ART": "#FF8C00",   # Dark orange
            "LAW": "#8B4513",           # Brown
            "LANGUAGE": "#708090"       # Slate gray
        }
        
        self.default_color = "#AAAAAA"  # Gray for unknown types
        
    def visualize(self, graph_data: dict, output_path: str, title: str = "Knowledge Graph", raw_text: str = None, json_path: str = None) -> str:
        """
        Visualize graph data and save the visualization to an HTML file.
        
        Args:
            graph_data (dict): Graph data containing entities and relations.
            output_path (str): Path to save the HTML file.
            title (str, optional): Title of the visualization. Defaults to "Knowledge Graph".
            raw_text (str, optional): Raw text to display in the visualization.
            json_path (str, optional): Path to the JSON file containing graph data for QA functionality.
            
        Returns:
            str: Path to the saved HTML file.
        """
        try:
            import traceback
            # Create a directed graph
            G = nx.DiGraph()
            
            # Figure out where entities and relations are stored
            entities = []
            relations = []
            
            # Check if entities and relations are at the top level
            if "entities" in graph_data:
                entities = graph_data["entities"]
            # Check if they're inside a "data" field
            elif "data" in graph_data and "entities" in graph_data["data"]:
                entities = graph_data["data"]["entities"]
            
            # Same for relations
            if "relations" in graph_data:
                relations = graph_data["relations"]
            elif "data" in graph_data and "relations" in graph_data["data"]:
                relations = graph_data["data"]["relations"]
            
            # Extract raw text from graph_data if not provided as parameter
            if raw_text is None and "raw_text" in graph_data:
                raw_text = graph_data["raw_text"]
            
            # For debugging
            self.logger.info(f"Found {len(entities)} entities and {len(relations)} relations for visualization")
            
            return self._create_visualization(G, entities, relations, output_path, title, raw_text, json_path)
            
        except Exception as e:
            error_msg = f"Error in visualize method: {str(e)}"
            stack_trace = traceback.format_exc()
            self.logger.error(f"{error_msg}\n{stack_trace}")
            print(f"DETAILED ERROR: {error_msg}\n{stack_trace}")
            return None
    
    def _create_visualization(self, G, entities, relations, output_path, title, raw_text=None, json_path=None):
        """
        Create and save the actual visualization.
        
        Args:
            G: The NetworkX DiGraph object
            entities: List of entity dictionaries
            relations: List of relation dictionaries
            output_path: Path to save the HTML file
            title: Title of the visualization
            raw_text: Optional raw text to display
            json_path: Optional path to JSON for QA functionality
            
        Returns:
            str: Path to the saved HTML file
        """
        print("Debug: Starting _create_visualization")
        print(f"Debug: raw_text type = {type(raw_text)}")
        print(f"Debug: json_path type = {type(json_path)}")
        print(f"Debug: html module = {html}")
        print(f"Debug: html module id = {id(html)}")
        
        try:
            import traceback
            import uuid
            
            print("Debug: Starting _create_visualization")
            print(f"Debug: raw_text type = {type(raw_text)}")
            print(f"Debug: json_path type = {type(json_path)}")
            
            # Add nodes with attributes
            for entity in entities:
                # Support both new and old entity formats
                if "entity_name" in entity and "entity_type" in entity and "entity_id" in entity:
                    # New format
                    entity_type = entity["entity_type"]
                    entity_name = entity["entity_name"]
                    entity_id = entity["entity_id"]
                    properties = entity.get("properties", {})
                else:
                    # Old format (lotr_graph.json)
                    entity_type = entity.get("type", "UNKNOWN")
                    entity_name = entity.get("name", "Unnamed")
                    entity_id = entity.get("id", str(uuid.uuid4()))
                    properties = entity.get("properties", {})
                
                # Get the color for this entity type
                color = self._get_entity_color(entity_type)
                
                # Create a formatted title with all properties
                title_text = f"{entity_name} ({entity_type})"
                if properties:
                    title_text += "\n\nProperties:"
                    for prop_key, prop_value in properties.items():
                        title_text += f"\n• {prop_key}: {prop_value}"
                
                # Add node with attributes
                G.add_node(
                    entity_id, 
                    title=title_text, 
                    label=entity_name, 
                    color=color,
                    shape="dot",
                    size=8,
                    entity_type=entity_type,
                    font={'color': color}  # Add font color to match node color
                )
            
            # Add edges with attributes
            for relation in relations:
                # Support both new and old relation formats
                if "source_id" in relation and "target_id" in relation and "relation_type" in relation:
                    # New format
                    source_id = relation["source_id"]
                    target_id = relation["target_id"]
                    relation_type = relation["relation_type"]
                    properties = relation.get("properties", {})
                else:
                    # Old format (lotr_graph.json)
                    source_id = relation.get("from_entity", {}).get("id", "")
                    target_id = relation.get("to_entity", {}).get("id", "")
                    relation_type = relation.get("relation", "UNKNOWN")
                    properties = relation.get("properties", {})
                
                # Create a formatted title with all properties
                title_text = relation_type
                if properties:
                    title_text += "\n\nProperties:"
                    for prop_key, prop_value in properties.items():
                        title_text += f"\n• {prop_key}: {prop_value}"
                
                # Add edge with attributes
                G.add_edge(
                    source_id, 
                    target_id, 
                    title=title_text, 
                    label=relation_type,
                    arrows="to",
                    color={
                        "color": "#848484",
                        "highlight": "#848484",
                        "hover": "#848484",
                        "inherit": False
                    }
                )
            
            # Get raw text from graph_data if not provided as parameter
            if raw_text is None and hasattr(self, 'graph_data') and "raw_text" in self.graph_data:
                raw_text = self.graph_data["raw_text"]
            
            # Create NetworkX graph visualization using PyVis
            from pyvis.network import Network
            net = Network(
                height="800px", 
                width="100%", 
                directed=True, 
                notebook=False,
                heading=title
            )
            
            # Copy nodes and edges from NetworkX graph to PyVis network
            net.from_nx(G)
            
            # Configure physics
            net.toggle_physics(True)
            physics_options = """
            var options = {
                "nodes": {
                    "font": {
                        "size": 8,
                        "face": "arial",
                        "color": "inherit"
                    },
                    "borderWidth": 2,
                    "borderWidthSelected": 4
                },
                "edges": {
                    "smooth": {
                        "type": "continuous",
                        "forceDirection": "none"
                    },
                    "font": {
                        "size": 4,
                        "align": "middle"
                    },
                    "color": {
                        "color": "#848484",
                        "highlight": "#848484",
                        "hover": "#848484",
                        "inherit": false
                    },
                    "arrows": {
                        "to": {
                            "enabled": true,
                            "type": "arrow",
                            "scaleFactor": 1
                        },
                        "from": {
                            "enabled": false
                        },
                        "middle": {
                            "enabled": false
                        }
                    }
                },
                "physics": {
                    "forceAtlas2Based": {
                        "gravitationalConstant": -50,
                        "centralGravity": 0.01,
                        "springLength": 150,
                        "springConstant": 0.08
                    },
                    "maxVelocity": 50,
                    "minVelocity": 0.1,
                    "solver": "forceAtlas2Based"
                },
                "interaction": {
                    "hover": true,
                    "tooltipDelay": 0,
                    "hideEdgesOnDrag": false,
                    "multiselect": true,
                    "hoverConnectedEdges": true
                },
                "tooltip": {
                    "delay": 0,
                    "fontColor": "black",
                    "fontSize": 14,
                    "fontFace": "arial",
                    "color": {
                        "border": "#666",
                        "background": "#fff"
                    }
                }
            }
            """
            net.set_options(physics_options)
            
            # Generate HTML for color legend
            legend_html = self._generate_legend_html()
            
            # Generate HTML for the button controls and expandable sections
            control_sections_html = self._generate_control_sections_html(raw_text, json_path)
            
            # Save the visualization to file with custom HTML
            html_path = os.path.abspath(output_path)
            
            # Get the path to the directory containing the HTML file
            html_dir = os.path.dirname(html_path)
            
            # Create the directory if it doesn't exist
            os.makedirs(html_dir, exist_ok=True)
            
            # Generate HTML with the custom elements
            net.save_graph(html_path)
            
            # Read the generated HTML file
            with open(html_path, "r", encoding="utf-8") as f:
                html_content = f.read()
            
            # Remove duplicate Knowledge Graph Visualization title
            html_content = html_content.replace("<center>\n<h1>Knowledge Graph Visualization</h1>\n</center>", "")
            
            # Define all custom HTML as a string without using f-string for improved stability
            legend_div = '<div style="margin: 20px; padding: 20px; border-top: 1px solid #ddd;">'
            legend_div += '<div style="display: flex; flex-wrap: wrap; gap: 10px;">'
            legend_div += legend_html
            legend_div += '</div></div>'
            
            # Script for text size adjustment and dropdown synchronization
            script_section = '<script>'
            script_section += 'function adjustTextSize(val) {'
            script_section += '    document.querySelectorAll(".vis-network .vis-node text").forEach(function(textElement) {'
            script_section += '        textElement.setAttribute("font-size", val);'
            script_section += '    });'
            script_section += '    document.querySelectorAll(".vis-network .vis-edge text").forEach(function(textElement) {'
            script_section += '        textElement.setAttribute("font-size", val);'
            script_section += '    });'
            script_section += '}'
            
            script_section += 'function syncModelProvider() {'
            script_section += '    var modelSelect = document.getElementById("llmModel");'
            script_section += '    var providerSelect = document.getElementById("llmProvider");'
            script_section += '    if (!modelSelect || !providerSelect) return;'
            
            script_section += '    modelSelect.addEventListener("change", function() {'
            script_section += '        var model = modelSelect.value;'
            script_section += '        if (model === "gpt-4o") {'
            script_section += '            providerSelect.value = "openai";'
            script_section += '        } else if (model === "deepseek-chat") {'
            script_section += '            providerSelect.value = "deepseek";'
            script_section += '        } else if (model === "claude-3-5-sonnet-20241022") {'
            script_section += '            providerSelect.value = "anthropic";'
            script_section += '        }'
            script_section += '    });'
            
            script_section += '    providerSelect.addEventListener("change", function() {'
            script_section += '        var provider = providerSelect.value;'
            script_section += '        if (provider === "openai" && modelSelect.value !== "gpt-4o") {'
            script_section += '            modelSelect.value = "gpt-4o";'
            script_section += '        } else if (provider === "deepseek" && modelSelect.value !== "deepseek-chat") {'
            script_section += '            modelSelect.value = "deepseek-chat";'
            script_section += '        } else if (provider === "anthropic" && modelSelect.value !== "claude-3-5-sonnet-20241022") {'
            script_section += '            modelSelect.value = "claude-3-5-sonnet-20241022";'
            script_section += '        }'
            script_section += '    });'
            script_section += '}'
            
            script_section += 'window.addEventListener("load", function() {'
            script_section += '    syncModelProvider();'
            script_section += '});'
            
            # Add the missing toggle functions
            script_section += '// Functions to toggle sections\n'
            script_section += 'function toggleTextSection() {\n'
            script_section += '    console.log("Toggling text section");\n'
            script_section += '    const textSection = document.getElementById("textSection");\n'
            script_section += '    const textBtn = document.getElementById("toggleTextBtn");\n'
            script_section += '    const qaSection = document.getElementById("qaSection");\n'
            script_section += '    const qaBtn = document.getElementById("toggleQABtn");\n'
            script_section += '    \n'
            script_section += '    if (!textSection) {\n'
            script_section += '        console.error("Text section not found");\n'
            script_section += '        return;\n'
            script_section += '    }\n'
            script_section += '    \n'
            script_section += '    // Check if section is visible\n'
            script_section += '    const isVisible = textSection.style.display === "block";\n'
            script_section += '    \n'
            script_section += '    // Hide QA section if visible\n'
            script_section += '    if (qaSection && qaSection.style.display === "block") {\n'
            script_section += '        qaSection.style.display = "none";\n'
            script_section += '        if (qaBtn) qaBtn.style.backgroundColor = "#2196F3";\n'
            script_section += '    }\n'
            script_section += '    \n'
            script_section += '    // Toggle text section\n'
            script_section += '    textSection.style.display = isVisible ? "none" : "block";\n'
            script_section += '    if (textBtn) {\n'
            script_section += '        textBtn.style.backgroundColor = isVisible ? "#4CAF50" : "#f44336";\n'
            script_section += '    }\n'
            script_section += '    \n'
            script_section += '    // Adjust network height\n'
            script_section += '    const networkContainer = document.getElementById("mynetwork");\n'
            script_section += '    if (networkContainer) {\n'
            script_section += '        if (isVisible) {\n'
            script_section += '            // Maximize height when section is hidden\n'
            script_section += '            const windowHeight = window.innerHeight;\n'
            script_section += '            const networkTop = networkContainer.getBoundingClientRect().top;\n'
            script_section += '            const newHeight = windowHeight - networkTop - 80; // Allow for buttons at bottom\n'
            script_section += '            networkContainer.style.height = newHeight + "px";\n'
            script_section += '        } else {\n'
            script_section += '            // Fixed height when a section is visible\n'
            script_section += '            networkContainer.style.height = "600px";\n'
            script_section += '        }\n'
            script_section += '        // Redraw the network\n'
            script_section += '        if (typeof network !== "undefined") {\n'
            script_section += '            network.fit();\n'
            script_section += '        }\n'
            script_section += '    }\n'
            script_section += '}\n'
            
            script_section += '\n'
            script_section += 'function toggleQASection() {\n'
            script_section += '    console.log("Toggling QA section");\n'
            script_section += '    const qaSection = document.getElementById("qaSection");\n'
            script_section += '    const qaBtn = document.getElementById("toggleQABtn");\n'
            script_section += '    const textSection = document.getElementById("textSection");\n'
            script_section += '    const textBtn = document.getElementById("toggleTextBtn");\n'
            script_section += '    \n'
            script_section += '    if (!qaSection) {\n'
            script_section += '        console.error("QA section not found");\n'
            script_section += '        return;\n'
            script_section += '    }\n'
            script_section += '    \n'
            script_section += '    // Check if section is visible\n'
            script_section += '    const isVisible = qaSection.style.display === "block";\n'
            script_section += '    \n'
            script_section += '    // Hide text section if visible\n'
            script_section += '    if (textSection && textSection.style.display === "block") {\n'
            script_section += '        textSection.style.display = "none";\n'
            script_section += '        if (textBtn) textBtn.style.backgroundColor = "#4CAF50";\n'
            script_section += '    }\n'
            script_section += '    \n'
            script_section += '    // Toggle QA section\n'
            script_section += '    qaSection.style.display = isVisible ? "none" : "block";\n'
            script_section += '    if (qaBtn) {\n'
            script_section += '        qaBtn.style.backgroundColor = isVisible ? "#2196F3" : "#f44336";\n'
            script_section += '    }\n'
            script_section += '    \n'
            script_section += '    // Adjust network height\n'
            script_section += '    const networkContainer = document.getElementById("mynetwork");\n'
            script_section += '    if (networkContainer) {\n'
            script_section += '        if (isVisible) {\n'
            script_section += '            // Maximize height when section is hidden\n'
            script_section += '            const windowHeight = window.innerHeight;\n'
            script_section += '            const networkTop = networkContainer.getBoundingClientRect().top;\n'
            script_section += '            const newHeight = windowHeight - networkTop - 80; // Allow for buttons at bottom\n'
            script_section += '            networkContainer.style.height = newHeight + "px";\n'
            script_section += '        } else {\n'
            script_section += '            // Fixed height when a section is visible\n'
            script_section += '            networkContainer.style.height = "600px";\n'
            script_section += '        }\n'
            script_section += '        // Redraw the network\n'
            script_section += '        if (typeof network !== "undefined") {\n'
            script_section += '            network.fit();\n'
            script_section += '        }\n'
            script_section += '    }\n'
            script_section += '}\n'
            
            script_section += '</script>'
            
            # Create the sections container
            sections_html = f"""
            <div id="expandableSections" style="position: relative; width: 100%; clear: both; padding: 0 20px; z-index: 10; display: block;">
                <div id="textSection" class="expandable-section" style="position: relative; width: 100%; margin: 20px 0; padding: 20px; display: none; border: 1px solid #eee; border-radius: 4px; background-color: #f9f9f9;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                        <h3 style="margin: 0;">Source Text</h3>
                    </div>
                    <div id="sourceTextContainer" style="max-height: 300px; overflow-y: auto; border: 1px solid #eee; padding: 10px; margin-top: 10px; background-color: white;">
                        <p style="white-space: pre-wrap;">{raw_text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;').replace("'", '&#39;') if raw_text else "No source text available."}</p>
                    </div>
                </div>
            """
            
            # Create the QA section if json_path is available
            if json_path:
                # Get available LLM models, order is important to set defaults
                llm_models = [
                    "gpt-4o",
                    "deepseek-chat",
                    "claude-3-5-sonnet-20241022",
                    "gemini-pro",
                    "qwen-local-model"
                ]
                
                # Available LLM providers
                llm_providers = [
                    "openai",
                    "deepseek", 
                    "anthropic",
                    "gemini",
                    "local"
                ]
                
                # Default select values
                default_model = "gpt-4o"
                default_provider = "openai"
                
                # Create the model and provider selection dropdowns
                model_options = ""
                for model in llm_models:
                    selected = 'selected="selected"' if model == default_model else ''
                    model_options += f'<option value="{model}" {selected}>{model}</option>'
                    
                provider_options = ""
                for provider in llm_providers:
                    selected = 'selected="selected"' if provider == default_provider else ''
                    provider_options += f'<option value="{provider}" {selected}>{provider}</option>'

                sections_html += f"""
                <div id="qaSection" class="expandable-section" style="position: relative; width: 100%; margin: 20px 0; padding: 20px; display: none; border: 1px solid #eee; border-radius: 4px; background-color: #f9f9f9;">
                    <h3>Ask Questions About This Knowledge Graph</h3>
                    
                    <div style="margin-bottom: 15px; padding: 10px; background-color: #f0f8ff; border-left: 4px solid #2196F3; font-size: 14px;">
                        <strong>Note:</strong> To use this feature, you need to start the API server in a separate terminal window:
                        <pre style="margin-top: 5px; background-color: #f5f5f5; padding: 8px; border-radius: 4px; overflow-x: auto;">python src/graph_db/app.py --api-server --api-host localhost --api-port 8000</pre>
                    </div>
                    
                    <div style="margin: 20px 0;">
                        <div>
                            <label for="model-select">LLM Model:</label>
                            <select id="model-select" name="model">
                                {model_options}
                            </select>
                            
                            <label for="provider-select" style="margin-left: 15px;">LLM Provider:</label>
                            <select id="provider-select" name="provider">
                                {provider_options}
                            </select>
                        </div>
                        
                        <div style="margin-top: 10px;">
                            <textarea id="questionInput" rows="3" style="width: 100%; padding: 8px; border: 1px solid #ddd; border-radius: 4px;" placeholder="Type your question here..."></textarea>
                        </div>
                        
                        <div style="margin-top: 10px;">
                            <button id="askButton" class="btn" style="background-color: #2196F3; color: white; border: none; padding: 10px 20px; text-align: center; text-decoration: none; display: inline-block; font-size: 16px; cursor: pointer; border-radius: 4px;">
                                Ask Question
                            </button>
                        </div>
                    </div>
                    
                    <div id="loadingIndicator" style="display: none; margin: 20px 0;">
                        <p>Processing your question... <span class="loading-spinner">⏳</span></p>
                    </div>
                    
                    <div id="answerContainer" style="display: none; margin: 20px 0; padding: 15px; background-color: #fff; border-radius: 4px; border: 1px solid #ddd;">
                        <h4>Answer:</h4>
                        <div id="answerContent"></div>
                        
                        <div id="expandButtonContainer" style="margin-top: 15px; display: none;">
                            <button id="expandContextButton" class="btn" style="background-color: #999; color: white; border: none; padding: 8px 15px; text-align: center; text-decoration: none; display: inline-block; font-size: 14px; cursor: pointer; border-radius: 4px;">
                                Show Context
                            </button>
                        </div>
                        
                        <div id="contextContainer" style="display: none; margin-top: 10px; padding: 10px; background-color: #f1f1f1; border-radius: 4px;">
                            <h4>Context Information:</h4>
                            <div id="contextContent"></div>
                        </div>
                    </div>
                    
                    <div id="errorContainer" style="display: none; margin: 20px 0; padding: 15px; background-color: #ffe6e6; border-radius: 4px;">
                        <h4>Error:</h4>
                        <div id="errorContent"></div>
                    </div>
                </div>
                """

            # Close the sections container
            sections_html += "</div>"
            
            # Combine all elements
            combined_html = legend_div + control_sections_html + sections_html + script_section
            
            # Insert custom HTML before closing body tag
            html_content = html_content.replace("</body>", combined_html + "</body>")
            
            # Write the modified HTML back to the file
            with open(html_path, "w", encoding="utf-8") as f:
                f.write(html_content)
            
            return html_path
            
        except Exception as e:
            self.logger.error(f"Error creating visualization: {str(e)}")
            return None
    
    def _get_entity_color(self, entity_type: str) -> str:
        """Get the color for a given entity type."""
        return self.entity_colors.get(entity_type.upper(), self.default_color)

    def create_visualization_from_data(self, 
                                      entities: List[Dict[str, Any]], 
                                      relations: List[Dict[str, Any]],
                                      output_path: str = "graph.html",
                                      title: str = "Graph Visualization",
                                      raw_text: str = "") -> bool:
        """
        Create a visualization from entities and relations.
        
        Args:
            entities (List[Dict[str, Any]]): List of entities
            relations (List[Dict[str, Any]]): List of relations
            output_path (str): Path to save the visualization
            title (str): Title of the visualization
            raw_text (str): Raw text used to generate the graph
            
        Returns:
            bool: True if visualization created successfully, False otherwise
        """
        try:
            # Create a networkx graph
            G = nx.DiGraph()
            
            # Add nodes
            for entity in entities:
                node_id = entity.get("id", hash(entity["name"]))
                entity_type = entity["type"].upper()  # Ensure uppercase for color matching
                color = self.entity_colors.get(entity_type, self.default_color)
                
                # Create a formatted title with all properties
                properties = entity.get("properties", {})
                title_text = f"{entity['name']} ({entity['type']})"
                if properties:
                    title_text += "\n\nProperties:"
                    for prop_key, prop_value in properties.items():
                        title_text += f"\n• {prop_key}: {prop_value}"
                
                G.add_node(
                    node_id, 
                    label=entity["name"], 
                    title=title_text,  # Use plain text title
                    color=color,  # Explicitly set color instead of group
                    font={'color': color},  # Set font color to match node color
                    properties=entity.get("properties", {})
                )
            
            # Add edges
            for relation in relations:
                from_id = relation["from_entity"].get("id", hash(relation["from_entity"]["name"]))
                to_id = relation["to_entity"].get("id", hash(relation["to_entity"]["name"]))
                
                # Create a formatted title with all properties
                properties = relation.get("properties", {})
                title_text = relation['relation']
                if properties:
                    title_text += "\n\nProperties:"
                    for prop_key, prop_value in properties.items():
                        title_text += f"\n• {prop_key}: {prop_value}"
                
                G.add_edge(
                    from_id, 
                    to_id, 
                    label=relation["relation"],
                    title=title_text,  # Use plain text title
                    properties=relation.get("properties", {}),
                    color={
                        "color": "#000000",
                        "highlight": "#000000",
                        "hover": "#000000",
                        "inherit": False
                    },
                    arrows={
                        "to": {
                            "enabled": True
                        }
                    }
                )
            
            # Create a pyvis network
            net = Network(notebook=False, directed=True, height="750px", width="100%")
            
            # Set options for create_visualization_from_data method
            net.set_options("""
            {
                "nodes": {
                    "shape": "dot",
                    "size": 10,
                    "font": {
                        "size": 8,
                        "face": "Tahoma",
                        "color": "inherit"
                    }
                },
                "edges": {
                    "font": {
                        "size": 4,
                        "align": "middle"
                    },
                    "color": {
                        "color": "#000000",
                        "inherit": false
                    },
                    "arrows": {
                        "to": {
                            "enabled": true,
                            "type": "arrow",
                            "scaleFactor": 1
                        },
                        "from": {
                            "enabled": false
                        },
                        "middle": {
                            "enabled": false
                        }
                    },
                    "smooth": {
                        "type": "continuous",
                        "forceDirection": "none"
                    }
                },
                "physics": {
                    "barnesHut": {
                        "gravitationalConstant": -80000,
                        "springLength": 250,
                        "springConstant": 0.001
                    },
                    "minVelocity": 0.75
                },
                "interaction": {
                    "hover": true,
                    "tooltipDelay": 0,
                    "navigationButtons": true,
                    "keyboard": true,
                    "hideEdgesOnDrag": false,
                    "multiselect": true,
                    "hoverConnectedEdges": true
                },
                "tooltip": {
                    "delay": 0,
                    "fontColor": "black",
                    "fontSize": 14,
                    "fontFace": "Tahoma",
                    "color": {
                        "border": "#666",
                        "background": "#fff"
                    }
                }
            }
            """)
            
            # Add the networkx graph
            net.from_nx(G)
            
            # Save the visualization
            net.save_graph(output_path)
            
            # Generate HTML for the legend and raw text
            used_entity_types = set([entity["type"] for entity in entities])
            legend_html = self._generate_legend_html()
            raw_text_html = self._generate_raw_text_html(raw_text)
            
            # Generate HTML for the QA panel
            json_path = os.path.splitext(output_path)[0] + '.json'
            qa_html = self._generate_qa_html(json_path)
            
            # Add the legend, raw text, and QA panel to the HTML file
            self._add_html_content(output_path, legend_html, raw_text_html, qa_html)
            
            # Save the graph data as JSON
            self._save_graph_data_as_json(entities, relations, output_path, raw_text)
            
            self.logger.info(f"Graph visualization saved to {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error creating visualization: {str(e)}")
            return False
            
    def _save_graph_data_as_json(self, entities: List[Dict[str, Any]], relations: List[Dict[str, Any]], html_path: str, raw_text: str = "") -> None:
        """
        Save the graph data (entities and relations) as a JSON file.
        
        Args:
            entities (List[Dict[str, Any]]): List of entities
            relations (List[Dict[str, Any]]): List of relations
            html_path (str): Path to the HTML visualization file
            raw_text (str): Raw text used to generate the graph
        """
        try:
            # Create the JSON path by replacing the extension
            json_path = os.path.splitext(html_path)[0] + '.json'
            
            # Extract entity types (schema)
            entity_types = list(set(entity["type"] for entity in entities))
            
            # Extract relation types (schema)
            relation_types = list(set(relation["relation"] for relation in relations))
            
            # Prepare the data
            graph_data = {
                "schema": {
                    "entity_types": entity_types,
                    "relation_types": relation_types
                },
                "data": {
                    "entities": entities,
                    "relations": relations
                },
                "raw_text": raw_text
            }
            
            # Save the data
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(graph_data, f, indent=2, ensure_ascii=False)
                
            self.logger.info(f"Graph data saved to {json_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving graph data as JSON: {str(e)}")
            
    def _add_html_content(self, html_path: str, legend_html: str, raw_text_html: str, qa_html: str) -> None:
        """
        Add custom HTML content to the visualization file.
        
        Args:
            html_path (str): Path to the HTML visualization file
            legend_html (str): HTML for the legend
            raw_text_html (str): HTML for the raw text
            qa_html (str): HTML for the QA functionality
        """
        try:
            # Read the HTML file
            with open(html_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            # Create the expandable sections container
            expandable_sections = f"""
            <div id="expandableSections" style="position: fixed; bottom: 80px; left: 0; right: 0; z-index: 900;">
                <div id="textSection" style="background-color: white; border-top: 1px solid #ddd; padding: 20px; display: none; height: 300px; overflow-y: auto;">
                    <h3>Source Text</h3>
                    {raw_text_html}
                </div>
                <div id="qaSection" style="background-color: white; border-top: 1px solid #ddd; padding: 20px; display: none; height: 300px; overflow-y: auto;">
                    <h3>Ask Questions About This Knowledge Graph</h3>
                    <div style="display: flex; gap: 10px; margin-bottom: 10px;">
                        <input type="text" id="questionInput" placeholder="Ask a question about the entities or relationships..." style="flex-grow: 1; padding: 8px; border: 1px solid #ddd; border-radius: 4px;">
                        <button id="askButton" style="background-color: #4CAF50; color: white; border: none; padding: 8px 15px; border-radius: 4px; cursor: pointer;">Ask</button>
                    </div>
                    <div style="margin-top: 5px; font-size: 0.8em;">
                        <label for="provider-select">LLM Provider:</label>
                        <select id="provider-select" style="margin-right: 10px;">
                            <option value="openai" selected>OpenAI</option>
                            <option value="anthropic">Anthropic</option>
                            <option value="deepseek">DeepSeek</option>
                        </select>
                        <label for="model-select">Model:</label>
                        <select id="model-select">
                            <option value="gpt-4o" selected>GPT-4o</option>
                            <option value="claude-3-5-sonnet-20241022">Claude 3.5 Sonnet</option>
                            <option value="deepseek-chat">DeepSeek Chat</option>
                        </select>
                    </div>
                    <div id="loadingIndicator" style="display: none; margin-top: 10px; text-align: center;">
                        <p>Processing your question...</p>
                    </div>
                    <div id="answerContainer" style="display: none; margin-top: 10px; border: 1px solid #ddd; padding: 10px; border-radius: 4px;">
                        <h4>Answer:</h4>
                        <p id="answerContent" style="white-space: pre-line;"></p>
                        <div id="expandButtonContainer" style="display: none; margin-top: 10px;">
                            <button id="expandContextButton" style="background-color: #2196F3; color: white; border: none; padding: 4px 8px; border-radius: 4px; cursor: pointer; font-size: 0.8em;">Show Context</button>
                            <div id="contextContainer" style="display: none; margin-top: 10px; padding: 10px; background-color: #f5f5f5; border-radius: 4px;">
                                <h5>Supporting Context:</h5>
                                <div id="contextContent"></div>
                            </div>
                        </div>
                    </div>
                    <div id="errorContainer" style="display: none; margin-top: 10px; border: 1px solid #f44336; padding: 10px; border-radius: 4px; background-color: #ffebee;">
                        <h4>Error:</h4>
                        <p id="errorContent" style="color: #d32f2f;"></p>
                    </div>
                </div>
            </div>
            """
            
            # Create the control buttons
            control_buttons = """
            <div style="position: fixed; bottom: 20px; right: 20px; z-index: 1000; display: flex; gap: 10px;">
                <button id="toggleTextBtn" class="control-btn" style="padding: 8px 15px; cursor: pointer; background-color: #4CAF50; color: white; border: none; border-radius: 4px; font-weight: bold; box-shadow: 0 2px 5px rgba(0,0,0,0.2);">Show Text</button>
                <button id="toggleQABtn" class="control-btn" style="padding: 8px 15px; cursor: pointer; background-color: #2196F3; color: white; border: none; border-radius: 4px; font-weight: bold; box-shadow: 0 2px 5px rgba(0,0,0,0.2);">Ask Questions</button>
            </div>
            """
            
            # Create JavaScript for the QA functionality
            json_path_rel = os.path.basename(os.path.splitext(html_path)[0] + '.json')
            
            qa_js = f"""
            <script>
                // Store the JSON path for the QA functionality
                const jsonPath = '{json_path_rel}';
                console.log('JSON Path:', jsonPath);
                
                document.addEventListener('DOMContentLoaded', function() {{
                    console.log('DOM loaded for QA functionality');
                    
                    // Handle button clicks
                    const textBtn = document.getElementById('toggleTextBtn');
                    const textSection = document.getElementById('textSection');
                    const qaBtn = document.getElementById('toggleQABtn');
                    const qaSection = document.getElementById('qaSection');
                    const networkContainer = document.getElementById('mynetwork');
                    
                    // Ensure sections are initially hidden
                    if (textSection) {{
                        textSection.style.display = 'none';
                        console.log('Initialized textSection to hidden');
                    }}
                    
                    if (qaSection) {{
                        qaSection.style.display = 'none';
                        console.log('Initialized qaSection to hidden');
                    }}
                    
                    // Text button click handler
                    if (textBtn && textSection) {{
                        textBtn.addEventListener('click', function() {{
                            console.log('Text button clicked');
                            // Check if section is visible
                            const isVisible = textSection.style.display === 'block';
                            
                            // Hide QA section if visible
                            if (qaSection && qaSection.style.display === 'block') {{
                                qaSection.style.display = 'none';
                                if (qaBtn) qaBtn.style.backgroundColor = '#2196F3';
                            }}
                            
                            // Toggle text section
                            textSection.style.display = isVisible ? 'none' : 'block';
                            textBtn.style.backgroundColor = isVisible ? '#4CAF50' : '#f44336';
                            
                            // Adjust network height
                            adjustNetworkHeight();
                        }});
                    }}
                    
                    // QA button click handler
                    if (qaBtn && qaSection) {{
                        qaBtn.addEventListener('click', function() {{
                            console.log('QA button clicked');
                            // Check if section is visible
                            const isVisible = qaSection.style.display === 'block';
                            
                            // Hide text section if visible
                            if (textSection && textSection.style.display === 'block') {{
                                textSection.style.display = 'none';
                                if (textBtn) textBtn.style.backgroundColor = '#4CAF50';
                            }}
                            
                            // Toggle QA section
                            qaSection.style.display = isVisible ? 'none' : 'block';
                            qaBtn.style.backgroundColor = isVisible ? '#2196F3' : '#f44336';
                            
                            // Adjust network height
                            adjustNetworkHeight();
                        }});
                    }}
                    
                    // Handle question submission
                    const askButton = document.getElementById('askButton');
                    const questionInput = document.getElementById('questionInput');
                    const loadingIndicator = document.getElementById('loadingIndicator');
                    const answerContainer = document.getElementById('answerContainer');
                    const answerContent = document.getElementById('answerContent');
                    const errorContainer = document.getElementById('errorContainer');
                    const errorContent = document.getElementById('errorContent');
                    const providerSelect = document.getElementById('provider-select');
                    const modelSelect = document.getElementById('model-select');
                    
                    if (askButton && questionInput) {{
                        askButton.addEventListener('click', function() {{
                            const question = questionInput.value.trim();
                            if (!question) {{
                                alert('Please enter a question');
                                return;
                            }}
                            
                            // Hide any previous results
                            if (answerContainer) answerContainer.style.display = 'none';
                            if (errorContainer) errorContainer.style.display = 'none';
                            
                            // Show loading indicator
                            if (loadingIndicator) loadingIndicator.style.display = 'block';
                            
                            // Get selected provider and model
                            const provider = providerSelect ? providerSelect.value : 'openai';
                            const model = modelSelect ? modelSelect.value : 'gpt-4o';
                            
                            console.log('Submitting question:', question);
                            console.log('Provider:', provider);
                            console.log('Model:', model);
                            console.log('JSON Path:', jsonPath);
                            
                            // Get the current window location to determine API endpoint
                            const currentHost = window.location.hostname || 'localhost';
                            const apiPort = 8000; // Default API port
                            const apiUrl = `http://${{currentHost}}:${{apiPort}}/api/qa`;
                            
                            console.log('API URL:', apiUrl);
                            
                            // Make API request
                            fetch(apiUrl, {{
                                method: 'POST',
                                headers: {{
                                    'Content-Type': 'application/json'
                                }},
                                body: JSON.stringify({{
                                    question: question,
                                    llm_provider: provider,
                                    llm_model: model,
                                    include_raw_text: true,
                                    json_path: jsonPath
                                }})
                            }})
                            .then(response => {{
                                if (!response.ok) {{
                                    return response.text().then(text => {{
                                        throw new Error(`HTTP error! Status: ${{response.status}}, Response: ${{text}}`);
                                    }});
                                }}
                                return response.json();
                            }})
                            .then(data => {{
                                console.log('API response:', data);
                                
                                // Hide loading indicator
                                if (loadingIndicator) loadingIndicator.style.display = 'none';
                                
                                // Show answer container
                                if (answerContainer) answerContainer.style.display = 'block';
                                
                                // Update answer content
                                if (answerContent) answerContent.textContent = data.answer;
                                
                                // Handle context if available
                                const expandButtonContainer = document.getElementById('expandButtonContainer');
                                const expandContextButton = document.getElementById('expandContextButton');
                                const contextContainer = document.getElementById('contextContainer');
                                const contextContent = document.getElementById('contextContent');
                                
                                if (data.context && contextContent) {{
                                    // Format the context string as HTML
                                    contextContent.innerHTML = `<pre>${data.context}</pre>`;
                                    
                                    // Show expand button
                                    if (expandButtonContainer) expandButtonContainer.style.display = 'block';
                                }}
                            }})
                            .catch(error => {{
                                console.error('Error:', error);
                                
                                // Hide loading indicator
                                if (loadingIndicator) loadingIndicator.style.display = 'none';
                                
                                // Show error message
                                if (errorContent) errorContent.textContent = error.message || 'An error occurred while processing your question.';
                                
                                // Show error container
                                if (errorContainer) errorContainer.style.display = 'block';
                            }});
                        }});
                    }}
                    
                    // Handle context expansion
                    const expandContextButton = document.getElementById('expandContextButton');
                    const contextContainer = document.getElementById('contextContainer');
                    
                    if (expandContextButton && contextContainer) {{
                        expandContextButton.addEventListener('click', function() {{
                            const isVisible = contextContainer.style.display === 'block';
                            contextContainer.style.display = isVisible ? 'none' : 'block';
                            expandContextButton.textContent = isVisible ? 'Show Context' : 'Hide Context';
                        }});
                    }}
                    
                    // Initialize network height
                    adjustNetworkHeight();
                }});
                
                // Function to adjust network height based on visible sections
                function adjustNetworkHeight() {{
                    const networkContainer = document.getElementById('mynetwork');
                    if (!networkContainer) return;
                    
                    const textSection = document.getElementById('textSection');
                    const qaSection = document.getElementById('qaSection');
                    
                    const textVisible = textSection && textSection.style.display === 'block';
                    const qaVisible = qaSection && qaSection.style.display === 'block';
                    
                    if (!textVisible && !qaVisible) {{
                        // Maximize height when no sections are visible
                        const windowHeight = window.innerHeight;
                        const networkTop = networkContainer.getBoundingClientRect().top;
                        const newHeight = windowHeight - networkTop - 80; // Allow for buttons at bottom
                        networkContainer.style.height = newHeight + 'px';
                    }} else {{
                        // Fixed height when a section is visible
                        networkContainer.style.height = '600px';
                    }}
                    
                    // Redraw the network
                    if (typeof network !== 'undefined') {{
                        network.fit();
                    }}
                }}
                
                // Add provider-model synchronization
                const modelSelect = document.getElementById('model-select');
                const providerSelect = document.getElementById('provider-select');
                
                if (modelSelect && providerSelect) {{
                    modelSelect.addEventListener('change', function() {{
                        const model = modelSelect.value;
                        if (model === 'gpt-4o') {{
                            providerSelect.value = 'openai';
                        }} else if (model === 'deepseek-chat') {{
                            providerSelect.value = 'deepseek';
                        }} else if (model === 'claude-3-5-sonnet-20241022') {{
                            providerSelect.value = 'anthropic';
                        }}
                    }});
                    
                    providerSelect.addEventListener('change', function() {{
                        const provider = providerSelect.value;
                        if (provider === 'openai' && modelSelect.value !== 'gpt-4o') {{
                            modelSelect.value = 'gpt-4o';
                        }} else if (provider === 'deepseek' && modelSelect.value !== 'deepseek-chat') {{
                            modelSelect.value = 'deepseek-chat';
                        }} else if (provider === 'anthropic' && modelSelect.value !== 'claude-3-5-sonnet-20241022') {{
                            modelSelect.value = 'claude-3-5-sonnet-20241022';
                        }}
                    }});
                }}
            </script>
            """
            
            # Combine all the components
            combined_html = f"""
            <div style="position: absolute; top: 10px; right: 10px; background-color: white; border: 1px solid #ddd; border-radius: 5px; padding: 10px; max-width: 300px; z-index: 1000;">
                {legend_html}
            </div>
            {expandable_sections}
            {control_buttons}
            {qa_js}
            """
            
            # Insert the combined HTML before the closing body tag
            html_content = html_content.replace("</body>", combined_html + "</body>")
            
            # Write the modified HTML back to the file
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
                
        except Exception as e:
            self.logger.error(f"Error adding HTML content: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())

    def _generate_legend_html(self):
        """Generate HTML for the entity type legend."""
        legend_html = """
        <div style="position: absolute; top: 10px; right: 10px; background-color: white; 
                    border: 1px solid #ccc; padding: 10px; border-radius: 5px; max-width: 250px;">
            <table style="width: 100%; border-collapse: collapse;" id="entityTypeTable">
                <tr>
                    <th style="text-align: left; padding: 5px;">Type</th>
                    <th style="text-align: left; padding: 5px;">Color</th>
                </tr>
        """
        
        for entity_type, color in self.entity_colors.items():
            legend_html += f"""
                <tr class="entity-type-row" data-entity-type="{entity_type}" style="cursor: pointer;">
                    <td style="padding: 5px;">{entity_type}</td>
                    <td style="padding: 5px;"><div style="width: 20px; height: 20px; background-color: {color}; 
                                                border-radius: 50%; display: inline-block;"></div></td>
                </tr>
            """
            
        legend_html += """
            </table>
            <div style="margin-top: 10px; text-align: center;">
                <button id="resetHighlightBtn" style="padding: 5px 10px; background-color: #f0f0f0; border: 1px solid #ccc; border-radius: 3px; cursor: pointer;">Reset Highlight</button>
            </div>
            <div style="margin-top: 15px; border-top: 1px solid #eee; padding-top: 10px;">
                <h4 style="margin-top: 0; text-align: center;">Text Size</h4>
                <div style="display: flex; align-items: center; margin-top: 5px;">
                    <span style="margin-right: 5px; font-size: 12px;">A</span>
                    <input type="range" id="textSizeSlider" min="2" max="32" value="8" style="flex-grow: 1;">
                    <span style="margin-left: 5px; font-size: 18px;">A</span>
                </div>
                <div style="text-align: center; margin-top: 5px;">
                    <span id="currentTextSize">8px</span>
                </div>
            </div>
            <script>
                // Add event listener to the slider
                document.addEventListener('DOMContentLoaded', function() {
                    var slider = document.getElementById('textSizeSlider');
                    var sizeDisplay = document.getElementById('currentTextSize');
                    
                    // Function to update text size
                    function updateTextSize(size) {
                        // Update the display
                        sizeDisplay.textContent = size + 'px';
                        
                        // Make sure network is defined
                        if (typeof network === 'undefined') {
                            console.warn('Network variable not found. Text size control may not work properly.');
                            return;
                        }
                        
                        // Update node font size
                        var nodeOptions = {
                            font: {
                                size: size,
                                color: 'inherit'
                            }
                        };
                        network.setOptions({ nodes: nodeOptions });
                        
                        // Update edge font size to half of node size
                        var edgeOptions = {
                            font: {
                                size: Math.round(size / 2)
                            }
                        };
                        network.setOptions({ edges: edgeOptions });
                    }
                    
                    // Add event listener for slider changes
                    slider.addEventListener('input', function() {
                        updateTextSize(parseInt(this.value));
                    });
                    
                    // Initialize with default values
                    updateTextSize(parseInt(slider.value));
                    
                    // Set global variable to always enable tooltips
                    window.tooltipsEnabled = true;
                    
                    // Track selected entity types
                    var selectedEntityTypes = [];
                    
                    // Add event listeners for entity type highlighting
                    var entityTypeRows = document.querySelectorAll('.entity-type-row');
                    entityTypeRows.forEach(function(row) {
                        row.addEventListener('click', function(event) {
                            var entityType = this.getAttribute('data-entity-type');
                            
                            // Always allow multi-select by default
                            // Toggle selection
                            var index = selectedEntityTypes.indexOf(entityType);
                            if (index === -1) {
                                // Add to selection
                                selectedEntityTypes.push(entityType);
                                this.style.backgroundColor = '#e0e0e0';
                                this.style.fontWeight = 'bold';
                            } else {
                                // Remove from selection
                                selectedEntityTypes.splice(index, 1);
                                this.style.backgroundColor = '';
                                this.style.fontWeight = 'normal';
                            }
                            
                            console.log('Selected entity types:', selectedEntityTypes);
                            
                            // Apply highlighting based on current selection
                            if (selectedEntityTypes.length > 0) {
                                highlightNodesByTypes(selectedEntityTypes);
                            } else {
                                resetHighlighting();
                            }
                        });
                    });
                    
                    // Add a message about multi-select
                    var entityTypeTable = document.getElementById('entityTypeTable');
                    if (entityTypeTable) {
                        var helpRow = document.createElement('tr');
                        helpRow.innerHTML = '<td colspan="2" style="padding: 5px; font-size: 11px; color: #666; text-align: center;">Click to select/unselect multiple types</td>';
                        entityTypeTable.appendChild(helpRow);
                    }
                    
                    // Add event listener for reset button
                    var resetBtn = document.getElementById('resetHighlightBtn');
                    if (resetBtn) {
                        resetBtn.addEventListener('click', function() {
                            resetHighlighting();
                            
                            // Reset visual feedback in the table and clear selection
                            entityTypeRows.forEach(function(r) {
                                r.style.backgroundColor = '';
                                r.style.fontWeight = 'normal';
                            });
                            selectedEntityTypes = [];
                        });
                    }
                    
                    // Function to highlight nodes by multiple types
                    function highlightNodesByTypes(entityTypes) {
                        if (typeof network === 'undefined') {
                            console.warn('Network variable not found. Highlighting may not work properly.');
                            return;
                        }
                        
                        var allNodes = network.body.nodes;
                        var allEdges = network.body.edges;
                        
                        // First pass: save original values if not already saved
                        Object.values(allNodes).forEach(function(node) {
                            if (node.options) {
                                if (!node.options._originalColor) {
                                    node.options._originalColor = node.options.color;
                                    node.options._originalFont = JSON.parse(JSON.stringify(node.options.font));
                                    node.options._originalSize = node.options.size;
                                    node.options._originalLabel = node.options.label;
                                }
                            }
                        });
                        
                        Object.values(allEdges).forEach(function(edge) {
                            if (edge.options) {
                                if (!edge.options._originalColor) {
                                    edge.options._originalColor = edge.options.color;
                                    edge.options._originalWidth = edge.options.width;
                                    edge.options._originalFont = JSON.parse(JSON.stringify(edge.options.font));
                                    edge.options._originalLabel = edge.options.label;
                                }
                            }
                        });
                        
                        // Second pass: reset all nodes to lowlighted state
                        Object.values(allNodes).forEach(function(node) {
                            if (node.options) {
                                // Hide label for all nodes initially
                                node.options.label = undefined;
                                
                                // Keep original size but make less visible
                                node.options.color = {
                                    background: '#f0f0f0',
                                    border: '#e0e0e0'
                                };
                                node.options.size = node.options._originalSize;
                            }
                        });
                        
                        // Hide labels for all edges
                        Object.values(allEdges).forEach(function(edge) {
                            if (edge.options) {
                                // Hide label for all edges
                                edge.options.label = undefined;
                                
                                // Make edges less visible but preserve color structure
                                // Important: maintain the original color structure to avoid changing arrow colors
                                if (typeof edge.options._originalColor === 'object') {
                                    // If original color was an object with color properties, maintain structure
                                    var originalColorObj = edge.options._originalColor;
                                    edge.options.color = {
                                        color: '#e0e0e0',
                                        highlight: originalColorObj.highlight || '#e0e0e0',
                                        hover: originalColorObj.hover || '#e0e0e0',
                                        inherit: false,
                                        opacity: originalColorObj.opacity || 1.0
                                    };
                                    
                                    // Preserve any arrow color settings
                                    if (originalColorObj.to) edge.options.color.to = originalColorObj.to;
                                    if (originalColorObj.from) edge.options.color.from = originalColorObj.from;
                                    if (originalColorObj.middle) edge.options.color.middle = originalColorObj.middle;
                                } else {
                                    // Simple color case
                                    edge.options.color = {
                                        color: '#e0e0e0',
                                        highlight: '#e0e0e0',
                                        hover: '#e0e0e0',
                                        inherit: false
                                    };
                                }
                            }
                        });
                        
                        // Third pass: highlight nodes of the selected types
                        var highlightedNodeIds = [];
                        Object.values(allNodes).forEach(function(node) {
                            if (node.options && node.options.title) {
                                // Check if node belongs to any of the selected types
                                var nodeMatches = entityTypes.some(function(entityType) {
                                    return node.options.title.includes('(' + entityType + ')');
                                });
                                
                                if (nodeMatches) {
                                    // Make node 2X larger and restore label
                                    node.options.size = node.options._originalSize * 2;
                                    node.options.label = node.options._originalLabel;
                                    
                                    // Restore original color
                                    node.options.color = node.options._originalColor;
                                    
                                    // Ensure font color matches node color
                                    if (typeof node.options.color === 'object' && node.options.color.background) {
                                        node.options.font.color = node.options.color.background;
                                    } else if (typeof node.options.color === 'string') {
                                        node.options.font.color = node.options.color;
                                    }
                                    
                                    // Make font bold and slightly larger
                                    node.options.font.bold = true;
                                    node.options.font.size = parseInt(node.options.font.size) * 1.2;
                                    
                                    highlightedNodeIds.push(node.id);
                                }
                            }
                        });
                        
                        // Refresh the network
                        network.redraw();
                    }
                    
                    // Function to reset highlighting
                    function resetHighlighting() {
                        if (typeof network === 'undefined') {
                            console.warn('Network variable not found. Reset may not work properly.');
                            return;
                        }
                        
                        // Restore all nodes and edges to original state
                        Object.values(network.body.nodes).forEach(function(node) {
                            if (node.options && node.options._originalColor) {
                                node.options.color = node.options._originalColor;
                                node.options.font = JSON.parse(JSON.stringify(node.options._originalFont));
                                node.options.size = node.options._originalSize;
                                node.options.label = node.options._originalLabel;
                                
                                // Clear saved original values
                                delete node.options._originalColor;
                                delete node.options._originalFont;
                                delete node.options._originalSize;
                                delete node.options._originalLabel;
                            }
                        });
                        
                        Object.values(network.body.edges).forEach(function(edge) {
                            if (edge.options && edge.options._originalColor) {
                                // Ensure we properly restore the original color structure
                                edge.options.color = edge.options._originalColor;
                                edge.options.width = edge.options._originalWidth;
                                edge.options.font = JSON.parse(JSON.stringify(edge.options._originalFont));
                                edge.options.label = edge.options._originalLabel;
                                
                                // Clear saved original values
                                delete edge.options._originalColor;
                                delete edge.options._originalWidth;
                                delete edge.options._originalFont;
                                delete edge.options._originalLabel;
                            }
                        });
                        
                        // Refresh the network
                        network.redraw();
                    }
                });
            </script>
        </div>
        """
        
        return legend_html

    def _generate_raw_text_html(self, raw_text: str) -> str:
        """Generate HTML for raw text display."""
        try:
            print(f"Debug: In _generate_raw_text_html, raw_text type = {type(raw_text)}")
            print(f"Debug: html module type = {type(html)}")
            print(f"Debug: html.escape type = {type(html.escape)}")
            return f"""
            <div id="sourceTextContainer" style="max-height: 300px; overflow-y: auto; border: 1px solid #eee; padding: 10px; margin-top: 10px; background-color: white;">
                <p style="white-space: pre-wrap;">{raw_text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;').replace("'", '&#39;') if raw_text else "No source text available."}</p>
            </div>
            """
        except Exception as e:
            import traceback
            print(f"Debug: Error in _generate_raw_text_html: {str(e)}")
            print(f"Debug: Traceback: {traceback.format_exc()}")
            # Fallback in case html.escape fails
            if raw_text:
                # Replace problematic characters manually
                safe_text = raw_text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;').replace("'", '&#39;')
            else:
                safe_text = "No source text available."
            return f"""
            <div id="sourceTextContainer" style="max-height: 300px; overflow-y: auto; border: 1px solid #eee; padding: 10px; margin-top: 10px; background-color: white;">
                <p style="white-space: pre-wrap;">{safe_text}</p>
            </div>
            """

    def _generate_qa_html(self, json_path: str) -> str:
        """Generate HTML for QA functionality."""
        # Create JavaScript to toggle sections
        toggle_js = """
        <script>
            document.addEventListener('DOMContentLoaded', function() {
                // Get DOM elements
                const textBtn = document.getElementById('toggleTextBtn');
                const textSection = document.getElementById('textSection');
                const qaBtn = document.getElementById('toggleQABtn');
                const qaSection = document.getElementById('qaSection');
                const networkContainer = document.getElementById('mynetwork');
                const expandableSections = document.getElementById('expandableSections');
                
                console.log('DOM Content Loaded');
                console.log('textBtn exists:', !!textBtn);
                console.log('textSection exists:', !!textSection);
                console.log('qaBtn exists:', !!qaBtn);
                console.log('qaSection exists:', !!qaSection);
                console.log('expandableSections exists:', !!expandableSections);
                
                // Store JSON path for QA functionality
                const jsonPath = '""" + (json_path if json_path else "") + """';
                console.log('JSON Path:', jsonPath);
                
                // Ensure sections are initially hidden
                if (textSection) {
                    textSection.style.display = 'none';
                    console.log('Initialized textSection to hidden');
                }
                
                if (qaSection) {
                    qaSection.style.display = 'none';
                    console.log('Initialized qaSection to hidden');
                }
                
                // Make sure expandableSections is visible
                if (expandableSections) {
                    expandableSections.style.display = 'block';
                    console.log('Made expandableSections visible');
                }
                
                // Text button click handler
                if (textBtn && textSection) {
                    textBtn.addEventListener('click', function() {
                        console.log('Text button clicked');
                        console.log('Current textSection display:', textSection.style.display);
                        
                        // Check if section is visible (could be empty string or 'none')
                        const isVisible = textSection.style.display === 'block';
                        
                        // Hide QA section if visible
                        if (qaSection && qaSection.style.display === 'block') {
                            qaSection.style.display = 'none';
                            if (qaBtn) qaBtn.style.backgroundColor = '#2196F3';
                        }
                        
                        // Toggle text section
                        textSection.style.display = isVisible ? 'none' : 'block';
                        textBtn.style.backgroundColor = isVisible ? '#4CAF50' : '#f44336';
                        
                        console.log('New textSection display:', textSection.style.display);
                        
                        // Adjust network height
                        adjustNetworkHeight();
                    });
                }
                
                // QA button click handler
                if (qaBtn && qaSection) {
                    qaBtn.addEventListener('click', function() {
                        console.log('QA button clicked');
                        console.log('Current qaSection display:', qaSection.style.display);
                        
                        // Check if section is visible (could be empty string or 'none')
                        const isVisible = qaSection.style.display === 'block';
                        
                        // Hide text section if visible
                        if (textSection && textSection.style.display === 'block') {
                            textSection.style.display = 'none';
                            if (textBtn) textBtn.style.backgroundColor = '#4CAF50';
                        }
                        
                        // Toggle QA section
                        qaSection.style.display = isVisible ? 'none' : 'block';
                        qaBtn.style.backgroundColor = isVisible ? '#2196F3' : '#f44336';
                        
                        console.log('New qaSection display:', qaSection.style.display);
                        
                        // Adjust network height
                        adjustNetworkHeight();
                    });
                }
                
                // Initialize network height
                adjustNetworkHeight();
                
                // Function to adjust network height based on visible sections
                function adjustNetworkHeight() {
                    if (!networkContainer) return;
                    
                    const textVisible = textSection && textSection.style.display === 'block';
                    const qaVisible = qaSection && qaSection.style.display === 'block';
                    
                    if (!textVisible && !qaVisible) {
                        // Maximize height when no sections are visible
                        const windowHeight = window.innerHeight;
                        const networkTop = networkContainer.getBoundingClientRect().top;
                        const newHeight = windowHeight - networkTop - 80; // Allow for buttons at bottom
                        networkContainer.style.height = newHeight + 'px';
                    } else {
                        // Fixed height when a section is visible
                        networkContainer.style.height = '600px';
                    }
                    
                    // Redraw the network
                    if (typeof network !== 'undefined') {
                        network.fit();
                    }
                }
                
                // Add window resize handler to adjust heights
                window.addEventListener('resize', adjustNetworkHeight);
                
                // Setup QA functionality if applicable
                setupQAFunctionality();
            });
            
            function setupQAFunctionality() {
                // Get QA elements
                const askButton = document.getElementById('askButton');
                const questionInput = document.getElementById('questionInput');
                const loadingIndicator = document.getElementById('loadingIndicator');
                const answerContainer = document.getElementById('answerContainer');
                const answerContent = document.getElementById('answerContent');
                const expandButtonContainer = document.getElementById('expandButtonContainer');
                const expandContextButton = document.getElementById('expandContextButton');
                const contextContainer = document.getElementById('contextContainer');
                const contextContent = document.getElementById('contextContent');
                const errorContainer = document.getElementById('errorContainer');
                const errorContent = document.getElementById('errorContent');
                const llmProviderSelect = document.getElementById('provider-select');
                const llmModelSelect = document.getElementById('model-select');
                
                // If no QA elements, exit
                if (!askButton || !questionInput) return;
                
                // Handle question submission
                askButton.addEventListener('click', function() {
                    const question = questionInput.value.trim();
                    if (!question) {
                        alert('Please enter a question');
                        return;
                    }
                    
                    // Hide any previous results
                    if (answerContainer) answerContainer.style.display = 'none';
                    if (errorContainer) errorContainer.style.display = 'none';
                    
                    // Show loading indicator
                    if (loadingIndicator) loadingIndicator.style.display = 'block';
                    
                    // Get selected provider and model
                    const provider = llmProviderSelect ? llmProviderSelect.value : 'openai';
                    const model = llmModelSelect ? llmModelSelect.value : 'gpt-4o';
                    
                    // Get the current window location to determine API endpoint
                    const currentHost = window.location.hostname || 'localhost';
                    const apiPort = 8000; // Default API port
                    const apiUrl = `http://${currentHost}:${apiPort}/api/qa`;
                    
                    console.log('Sending request to:', apiUrl);
                    console.log('Question:', question);
                    console.log('Provider:', provider);
                    console.log('Model:', model);
                    console.log('JSON Path:', jsonPath);
                    
                    // Make API request with error handling
                    fetch(apiUrl, {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({
                            question: question,
                            llm_provider: provider,
                            llm_model: model,
                            include_raw_text: true,
                            json_path: jsonPath
                        })
                    })
                    .then(response => {
                        if (!response.ok) {
                            return response.text().then(text => {
                                throw new Error(`HTTP error! Status: ${response.status}, Response: ${text}`);
                            });
                        }
                        return response.json();
                    })
                    .then(data => {
                        console.log('API response:', data);
                        
                        // Hide loading indicator
                        document.getElementById('loadingIndicator').style.display = 'none';
                        
                        // Show answer container
                        const answerContainer = document.getElementById('answerContainer');
                        answerContainer.style.display = 'block';
                        
                        // Update answer content with markdown support
                        const answerContent = document.getElementById('answerContent');
                        answerContent.textContent = data.answer;
                        
                        // Handle context if available
                        if (data.context && contextContent) {
                            // Format the context string as HTML
                            contextContent.innerHTML = `<pre>${data.context}</pre>`;
                            
                            // Show expand button
                            document.getElementById('expandButtonContainer').style.display = 'block';
                        }
                    })
                    .catch(error => {
                        console.error('Error submitting question:', error);
                        // Show error message
                        const errorContent = document.getElementById('errorContent');
                        errorContent.textContent = 'An error occurred while submitting the question. Please try again later.';
                        
                        // Show error container
                        document.getElementById('errorContainer').style.display = 'block';
                    });
                });
                
                // Handle context expansion
                if (expandContextButton && contextContainer) {
                    expandContextButton.addEventListener('click', function() {
                        const isVisible = contextContainer.style.display === 'block';
                        contextContainer.style.display = isVisible ? 'none' : 'block';
                        expandContextButton.textContent = isVisible ? 'Show Context' : 'Hide Context';
                    });
                }
                
                // Setup model-provider sync
                syncModelProvider();
            }
            
            function syncModelProvider() {
                const modelSelect = document.getElementById('model-select');
                const providerSelect = document.getElementById('provider-select');
                if (!modelSelect || !providerSelect) return;
                
                modelSelect.addEventListener('change', function() {
                    const model = modelSelect.value;
                    if (model === 'gpt-4o') {
                        providerSelect.value = 'openai';
                    } else if (model === 'deepseek-chat') {
                        providerSelect.value = 'deepseek';
                    } else if (model === 'claude-3-5-sonnet-20241022') {
                        providerSelect.value = 'anthropic';
                    }
                });
                
                providerSelect.addEventListener('change', function() {
                    const provider = providerSelect.value;
                    if (provider === 'openai' && modelSelect.value !== 'gpt-4o') {
                        modelSelect.value = 'gpt-4o';
                    } else if (provider === 'deepseek' && modelSelect.value !== 'deepseek-chat') {
                        modelSelect.value = 'deepseek-chat';
                    } else if (provider === 'anthropic' && modelSelect.value !== 'claude-3-5-sonnet-20241022') {
                        modelSelect.value = 'claude-3-5-sonnet-20241022';
                    }
                });
            }
        </script>
        """
        
        # Return the combined HTML
        return toggle_js

    def _generate_control_sections_html(self, raw_text: str = None, json_path: str = None) -> str:
        """
        Generate HTML for the button control panel and expandable sections.
        
        Args:
            raw_text (str, optional): Raw text to display in the text section.
            json_path (str, optional): Path to the JSON file for QA functionality.
                
        Returns:
            str: HTML for the control panel and expandable sections.
        """
        # Create buttons for text and QA sections
        buttons_html = """
        <div style="position: fixed; bottom: 20px; right: 20px; z-index: 1000; display: flex; gap: 10px;">
            <button id="toggleTextBtn" class="control-btn" style="padding: 8px 15px; cursor: pointer; background-color: #4CAF50; color: white; border: none; border-radius: 4px; font-weight: bold; box-shadow: 0 2px 5px rgba(0,0,0,0.2);">Show Text</button>
            <button id="toggleQABtn" class="control-btn" style="padding: 8px 15px; cursor: pointer; background-color: #2196F3; color: white; border: none; border-radius: 4px; font-weight: bold; box-shadow: 0 2px 5px rgba(0,0,0,0.2);">Ask Questions</button>
        </div>
        """
        
        # Create JavaScript to toggle sections
        toggle_js = """
        <script>
            document.addEventListener('DOMContentLoaded', function() {
                // Get DOM elements
                const textBtn = document.getElementById('toggleTextBtn');
                const textSection = document.getElementById('textSection');
                const qaBtn = document.getElementById('toggleQABtn');
                const qaSection = document.getElementById('qaSection');
                const networkContainer = document.getElementById('mynetwork');
                const expandableSections = document.getElementById('expandableSections');
                
                console.log('DOM Content Loaded');
                console.log('textBtn exists:', !!textBtn);
                console.log('textSection exists:', !!textSection);
                console.log('qaBtn exists:', !!qaBtn);
                console.log('qaSection exists:', !!qaSection);
                console.log('expandableSections exists:', !!expandableSections);
                
                // Store JSON path for QA functionality
                const jsonPath = '""" + (json_path if json_path else "") + """';
                console.log('JSON Path:', jsonPath);
                
                // Ensure sections are initially hidden
                if (textSection) {
                    textSection.style.display = 'none';
                    console.log('Initialized textSection to hidden');
                }
                
                if (qaSection) {
                    qaSection.style.display = 'none';
                    console.log('Initialized qaSection to hidden');
                }
                
                // Make sure expandableSections is visible
                if (expandableSections) {
                    expandableSections.style.display = 'block';
                    console.log('Made expandableSections visible');
                }
                
                // Text button click handler
                if (textBtn && textSection) {
                    textBtn.addEventListener('click', function() {
                        console.log('Text button clicked');
                        console.log('Current textSection display:', textSection.style.display);
                        
                        // Check if section is visible (could be empty string or 'none')
                        const isVisible = textSection.style.display === 'block';
                        
                        // Hide QA section if visible
                        if (qaSection && qaSection.style.display === 'block') {
                            qaSection.style.display = 'none';
                            if (qaBtn) qaBtn.style.backgroundColor = '#2196F3';
                        }
                        
                        // Toggle text section
                        textSection.style.display = isVisible ? 'none' : 'block';
                        textBtn.style.backgroundColor = isVisible ? '#4CAF50' : '#f44336';
                        
                        console.log('New textSection display:', textSection.style.display);
                        
                        // Adjust network height
                        adjustNetworkHeight();
                    });
                }
                
                // QA button click handler
                if (qaBtn && qaSection) {
                    qaBtn.addEventListener('click', function() {
                        console.log('QA button clicked');
                        console.log('Current qaSection display:', qaSection.style.display);
                        
                        // Check if section is visible (could be empty string or 'none')
                        const isVisible = qaSection.style.display === 'block';
                        
                        // Hide text section if visible
                        if (textSection && textSection.style.display === 'block') {
                            textSection.style.display = 'none';
                            if (textBtn) textBtn.style.backgroundColor = '#4CAF50';
                        }
                        
                        // Toggle QA section
                        qaSection.style.display = isVisible ? 'none' : 'block';
                        qaBtn.style.backgroundColor = isVisible ? '#2196F3' : '#f44336';
                        
                        console.log('New qaSection display:', qaSection.style.display);
                        
                        // Adjust network height
                        adjustNetworkHeight();
                    });
                }
                
                // Initialize network height
                adjustNetworkHeight();
                
                // Function to adjust network height based on visible sections
                function adjustNetworkHeight() {
                    if (!networkContainer) return;
                    
                    const textVisible = textSection && textSection.style.display === 'block';
                    const qaVisible = qaSection && qaSection.style.display === 'block';
                    
                    if (!textVisible && !qaVisible) {
                        // Maximize height when no sections are visible
                        const windowHeight = window.innerHeight;
                        const networkTop = networkContainer.getBoundingClientRect().top;
                        const newHeight = windowHeight - networkTop - 80; // Allow for buttons at bottom
                        networkContainer.style.height = newHeight + 'px';
                    } else {
                        // Fixed height when a section is visible
                        networkContainer.style.height = '600px';
                    }
                    
                    // Redraw the network
                    if (typeof network !== 'undefined') {
                        network.fit();
                    }
                }
                
                // Add window resize handler to adjust heights
                window.addEventListener('resize', adjustNetworkHeight);
                
                // Setup QA functionality if applicable
                setupQAFunctionality();
            });
            
            function setupQAFunctionality() {
                // Get QA elements
                const askButton = document.getElementById('askButton');
                const questionInput = document.getElementById('questionInput');
                const loadingIndicator = document.getElementById('loadingIndicator');
                const answerContainer = document.getElementById('answerContainer');
                const answerContent = document.getElementById('answerContent');
                const expandButtonContainer = document.getElementById('expandButtonContainer');
                const expandContextButton = document.getElementById('expandContextButton');
                const contextContainer = document.getElementById('contextContainer');
                const contextContent = document.getElementById('contextContent');
                const errorContainer = document.getElementById('errorContainer');
                const errorContent = document.getElementById('errorContent');
                const llmProviderSelect = document.getElementById('provider-select');
                const llmModelSelect = document.getElementById('model-select');
                
                // If no QA elements, exit
                if (!askButton || !questionInput) return;
                
                // Handle question submission
                askButton.addEventListener('click', function() {
                    const question = questionInput.value.trim();
                    if (!question) {
                        alert('Please enter a question');
                        return;
                    }
                    
                    // Hide any previous results
                    if (answerContainer) answerContainer.style.display = 'none';
                    if (errorContainer) errorContainer.style.display = 'none';
                    
                    // Show loading indicator
                    if (loadingIndicator) loadingIndicator.style.display = 'block';
                    
                    // Get selected provider and model
                    const provider = llmProviderSelect ? llmProviderSelect.value : 'openai';
                    const model = llmModelSelect ? llmModelSelect.value : 'gpt-4o';
                    
                    // Get the current window location to determine API endpoint
                    const currentHost = window.location.hostname || 'localhost';
                    const apiPort = 8000; // Default API port
                    const apiUrl = `http://${currentHost}:${apiPort}/api/qa`;
                    
                    console.log('Sending request to:', apiUrl);
                    console.log('Question:', question);
                    console.log('Provider:', provider);
                    console.log('Model:', model);
                    console.log('JSON Path:', jsonPath);
                    
                    // Make API request with error handling
                    fetch(apiUrl, {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({
                            question: question,
                            llm_provider: provider,
                            llm_model: model,
                            include_raw_text: true,
                            json_path: jsonPath
                        })
                    })
                    .then(response => {
                        if (!response.ok) {
                            return response.text().then(text => {
                                throw new Error(`HTTP error! Status: ${response.status}, Response: ${text}`);
                            });
                        }
                        return response.json();
                    })
                    .then(data => {
                        console.log('API response:', data);
                        
                        // Hide loading indicator
                        document.getElementById('loadingIndicator').style.display = 'none';
                        
                        // Show answer container
                        const answerContainer = document.getElementById('answerContainer');
                        answerContainer.style.display = 'block';
                        
                        // Update answer content with markdown support
                        const answerContent = document.getElementById('answerContent');
                        answerContent.textContent = data.answer;
                        
                        // Handle context if available
                        if (data.context && contextContent) {
                            // Format the context string as HTML
                            contextContent.innerHTML = `<pre>${data.context}</pre>`;
                            
                            // Show expand button
                            document.getElementById('expandButtonContainer').style.display = 'block';
                        }
                    })
                    .catch(error => {
                        console.error('Error submitting question:', error);
                        // Show error message
                        const errorContent = document.getElementById('errorContent');
                        errorContent.textContent = 'An error occurred while submitting the question. Please try again later.';
                        
                        // Show error container
                        document.getElementById('errorContainer').style.display = 'block';
                    });
                });
                
                // Handle context expansion
                if (expandContextButton && contextContainer) {
                    expandContextButton.addEventListener('click', function() {
                        const isVisible = contextContainer.style.display === 'block';
                        contextContainer.style.display = isVisible ? 'none' : 'block';
                        expandContextButton.textContent = isVisible ? 'Show Context' : 'Hide Context';
                    });
                }
                
                // Setup model-provider sync
                syncModelProvider();
            }
            
            function syncModelProvider() {
                const modelSelect = document.getElementById('model-select');
                const providerSelect = document.getElementById('provider-select');
                if (!modelSelect || !providerSelect) return;
                
                modelSelect.addEventListener('change', function() {
                    const model = modelSelect.value;
                    if (model === 'gpt-4o') {
                        providerSelect.value = 'openai';
                    } else if (model === 'deepseek-chat') {
                        providerSelect.value = 'deepseek';
                    } else if (model === 'claude-3-5-sonnet-20241022') {
                        providerSelect.value = 'anthropic';
                    }
                });
                
                providerSelect.addEventListener('change', function() {
                    const provider = providerSelect.value;
                    if (provider === 'openai' && modelSelect.value !== 'gpt-4o') {
                        modelSelect.value = 'gpt-4o';
                    } else if (provider === 'deepseek' && modelSelect.value !== 'deepseek-chat') {
                        modelSelect.value = 'deepseek-chat';
                    } else if (provider === 'anthropic' && modelSelect.value !== 'claude-3-5-sonnet-20241022') {
                        modelSelect.value = 'claude-3-5-sonnet-20241022';
                    }
                });
            }
        </script>
        """
        
        # Return the combined HTML
        return buttons_html + toggle_js 