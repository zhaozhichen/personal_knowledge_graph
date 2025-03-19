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
            
        # Log what we found
        self.logger.info(f"Found {len(entities)} entities and {len(relations)} relations")
        
        # Add nodes with attributes
        for entity in entities:
            # Support both new and old entity formats
            if "entity_name" in entity and "entity_type" in entity and "entity_id" in entity:
                # New format
                entity_type = entity["entity_type"]
                entity_name = entity["entity_name"]
                entity_id = entity["entity_id"]
            else:
                # Old format (lotr_graph.json)
                entity_type = entity.get("type", "UNKNOWN")
                entity_name = entity.get("name", "Unnamed")
                entity_id = entity.get("id", str(uuid.uuid4()))
            
            # Get the color for this entity type
            color = self._get_entity_color(entity_type)
            
            # Add node with attributes
            G.add_node(
                entity_id, 
                title=f"{entity_name} ({entity_type})", 
                label=entity_name, 
                color=color,
                shape="dot",
                size=15,
                entity_type=entity_type
            )
        
        # Add edges with attributes
        for relation in relations:
            # Support both new and old relation formats
            if "source_id" in relation and "target_id" in relation and "relation_type" in relation:
                # New format
                source_id = relation["source_id"]
                target_id = relation["target_id"]
                relation_type = relation["relation_type"]
            else:
                # Old format (lotr_graph.json)
                source_id = relation.get("from_entity", {}).get("id", "")
                target_id = relation.get("to_entity", {}).get("id", "")
                relation_type = relation.get("relation", "UNKNOWN")
            
            # Add edge with attributes
            G.add_edge(
                source_id, 
                target_id, 
                title=relation_type, 
                label=relation_type,
                arrows="to"
            )
        
        # Get raw text from graph_data if not provided
        if raw_text is None and "raw_text" in graph_data:
            raw_text = graph_data["raw_text"]
        
        # Create NetworkX graph visualization using PyVis
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
        net.set_options("""
        var options = {
            "nodes": {
                "font": {
                    "size": 14,
                    "face": "arial"
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
                    "size": 12,
                    "align": "middle"
                },
                "color": {
                    "color": "#848484",
                    "highlight": "#848484",
                    "hover": "#848484"
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
            }
        }
        """)
        
        # Generate HTML for color legend
        legend_html = self._generate_legend_html()
        
        # Generate HTML for raw text display
        raw_text_html = self._generate_raw_text_html(raw_text) if raw_text else ""
        
        # Generate HTML for the QA panel
        qa_html = self._generate_qa_html(json_path) if json_path else ""
        
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
            html = f.read()
        
        # Insert custom HTML before the closing body tag
        custom_html = f"""
        <div style="margin: 20px; padding: 20px; border-top: 1px solid #ddd;">
            <h2 style="color: #333;">Entity Types</h2>
            <div style="display: flex; flex-wrap: wrap; gap: 10px;">
                {legend_html}
            </div>
        </div>
        
        {raw_text_html}
        
        {qa_html}
        
        <script>
            // Function to adjust text size based on slider value
            function adjustTextSize(val) {{
                document.querySelectorAll('.vis-network .vis-node text').forEach(function(textElement) {{
                    textElement.setAttribute('font-size', val);
                }});
                
                document.querySelectorAll('.vis-network .vis-edge text').forEach(function(textElement) {{
                    textElement.setAttribute('font-size', val);
                }});
            }}
        </script>
        """
        
        html = html.replace("</body>", custom_html + "</body>")
        
        # Write the modified HTML back to the file
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html)
        
        return html_path
    
    def _generate_legend_html(self):
        """Generate HTML for the entity type legend."""
        legend_html = """
        <div style="position: absolute; top: 10px; right: 10px; background-color: white; 
                    border: 1px solid #ccc; padding: 10px; border-radius: 5px; max-width: 250px;">
            <h3 style="margin-top: 0; text-align: center;">Entity Types</h3>
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
                    <input type="range" id="textSizeSlider" min="8" max="128" value="64" style="flex-grow: 1;">
                    <span style="margin-left: 5px; font-size: 18px;">A</span>
                </div>
                <div style="text-align: center; margin-top: 5px;">
                    <span id="currentTextSize">64px</span>
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
                                
                                // Make edges less visible
                                edge.options.color = {
                                    color: '#e0e0e0',
                                    highlight: '#e0e0e0'
                                };
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
                                    // Make node 3X larger and restore label
                                    node.options.size = node.options._originalSize * 3;
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
        """Generate HTML for displaying the raw text with a show/hide button."""
        if not raw_text:
            return ""
            
        # Create a fixed position button that's always visible
        raw_text_html = f"""
        <div id="textControlSection" style="position: fixed; bottom: 20px; right: 20px; z-index: 1000;">
            <button id="toggleTextBtn" onclick="toggleSourceText()" style="padding: 8px 15px; cursor: pointer; background-color: #4CAF50; color: white; border: none; border-radius: 4px; font-weight: bold; box-shadow: 0 2px 5px rgba(0,0,0,0.2);">Show Text</button>
        </div>
        <div id="textSection" style="position: relative; margin-top: 20px; padding-top: 10px; display: none;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                <h3 style="margin: 0;">Source Text</h3>
            </div>
            <div id="sourceTextContainer" style="max-height: 300px; overflow-y: auto; border: 1px solid #eee; padding: 10px; margin-top: 10px;">
                <p style="white-space: pre-wrap;">{html.escape(raw_text)}</p>
            </div>
        </div>
        <script>
            // Make sure this script runs after the network is initialized
            document.addEventListener('DOMContentLoaded', function() {{
                // Ensure network variable is accessible
                if (typeof network === 'undefined') {{
                    console.warn('Network variable not found. Some features may not work properly.');
                }}
                
                // Initially maximize the network container height since text is hidden
                var networkContainer = document.getElementById('mynetwork');
                var windowHeight = window.innerHeight;
                var networkTop = networkContainer.getBoundingClientRect().top;
                var newHeight = windowHeight - networkTop - 20; // 20px padding
                networkContainer.style.height = newHeight + 'px';
                
                // Redraw the network to fit the new container size
                if (typeof network !== 'undefined') {{
                    network.fit();
                }}
            }});
            
            function toggleSourceText() {{
                var textSection = document.getElementById('textSection');
                var btn = document.getElementById('toggleTextBtn');
                var networkContainer = document.getElementById('mynetwork');
                
                if (textSection.style.display === 'none') {{
                    // Show text
                    textSection.style.display = 'block';
                    btn.innerText = 'Hide Text';
                    btn.style.backgroundColor = '#f44336'; // Red color for hide button
                    
                    // Restore original network container height
                    networkContainer.style.height = '750px';
                }} else {{
                    // Hide text
                    textSection.style.display = 'none';
                    btn.innerText = 'Show Text';
                    btn.style.backgroundColor = '#4CAF50'; // Green color for show button
                    
                    // Maximize network container height
                    var windowHeight = window.innerHeight;
                    var networkTop = networkContainer.getBoundingClientRect().top;
                    var newHeight = windowHeight - networkTop - 20; // 20px padding
                    networkContainer.style.height = newHeight + 'px';
                }}
                
                // Redraw the network to fit the new container size
                if (typeof network !== 'undefined') {{
                    network.fit();
                }}
            }}
            
            // Add window resize event listener to adjust the graph size when window is resized
            window.addEventListener('resize', function() {{
                var textSection = document.getElementById('textSection');
                var networkContainer = document.getElementById('mynetwork');
                
                // Only adjust if text is hidden
                if (textSection.style.display === 'none') {{
                    var windowHeight = window.innerHeight;
                    var networkTop = networkContainer.getBoundingClientRect().top;
                    var newHeight = windowHeight - networkTop - 20; // 20px padding
                    networkContainer.style.height = newHeight + 'px';
                    
                    // Redraw the network to fit the new container size
                    if (typeof network !== 'undefined') {{
                        network.fit();
                    }}
                }}
            }});
        </script>
        """
        
        return raw_text_html
    
    def _generate_qa_html(self, json_path: str = None) -> str:
        """Generate HTML for the Question Answering panel.
        
        Args:
            json_path (str, optional): Path to the JSON file containing graph data.
            
        Returns:
            str: HTML for the QA panel.
        """
        if not json_path:
            return ""
            
        # Create the HTML with the JSON path directly embedded
        qa_html = f"""
        <div id="qaPanel" style="margin: 20px; padding: 20px; border-top: 1px solid #ddd;">
            <h2 style="color: #333;">Question Answering</h2>
            <p>Ask questions about the knowledge graph:</p>
            
            <div style="margin: 20px 0;">
                <input type="text" id="questionInput" placeholder="Enter your question..." 
                       style="width: 70%; padding: 10px; font-size: 16px; border: 1px solid #ccc; border-radius: 4px;">
                <button onclick="askQuestion()" 
                        style="padding: 10px 20px; font-size: 16px; background-color: #4CAF50; color: white; border: none; border-radius: 4px; cursor: pointer;">
                    Ask
                </button>
                <div style="margin-top: 10px;">
                    <input type="checkbox" id="includeRawText" checked>
                    <label for="includeRawText">Include original text in context</label>
                </div>
            </div>
            
            <div id="loadingIndicator" style="display: none; margin: 20px 0;">
                <p>Processing your question...</p>
            </div>
            
            <div id="answerContainer" style="display: none; margin: 20px 0; padding: 15px; background-color: #f9f9f9; border-radius: 4px;">
                <h3>Answer:</h3>
                <p id="answerText" style="font-size: 16px; line-height: 1.5;"></p>
                
                <div id="expandButtonContainer" style="margin-top: 15px; display: none;">
                    <button onclick="toggleContext()" id="toggleContextButton" 
                            style="padding: 5px 10px; font-size: 14px; background-color: #ddd; border: none; border-radius: 4px; cursor: pointer;">
                        Show Context
                    </button>
                    <div id="contextContainer" style="display: none; margin-top: 10px; padding: 10px; background-color: #f1f1f1; border-radius: 4px;">
                        <h4>Relations used:</h4>
                        <pre id="contextText" style="font-family: monospace; white-space: pre-wrap; font-size: 14px;"></pre>
                    </div>
                </div>
            </div>
            
            <div id="errorContainer" style="display: none; margin: 20px 0; padding: 15px; background-color: #ffe6e6; border-radius: 4px;">
                <h3>Error:</h3>
                <pre id="errorText" style="font-family: monospace; white-space: pre-wrap; font-size: 14px;"></pre>
            </div>
        </div>
        
        <script>
            function toggleContext() {{
                var contextContainer = document.getElementById('contextContainer');
                var toggleButton = document.getElementById('toggleContextButton');
                
                if (contextContainer.style.display === 'none') {{
                    contextContainer.style.display = 'block';
                    toggleButton.textContent = 'Hide Context';
                }} else {{
                    contextContainer.style.display = 'none';
                    toggleButton.textContent = 'Show Context';
                }}
            }}
            
            function askQuestion() {{
                var question = document.getElementById('questionInput').value.trim();
                if (!question) {{
                    alert("Please enter a question.");
                    return;
                }}
                
                var includeRawText = document.getElementById('includeRawText').checked;
                var jsonPath = "{json_path}";
                
                // Show loading indicator
                document.getElementById('loadingIndicator').style.display = 'block';
                document.getElementById('answerContainer').style.display = 'none';
                document.getElementById('errorContainer').style.display = 'none';
                
                // Try to use the API server first
                fetch('/api/ask', {{
                    method: 'POST',
                    headers: {{
                        'Content-Type': 'application/json',
                    }},
                    body: JSON.stringify({{
                        question: question,
                        json_path: jsonPath,
                        include_raw_text: includeRawText
                    }})
                }})
                .then(response => {{
                    if (!response.ok) {{
                        throw new Error('Network response was not ok');
                    }}
                    return response.json();
                }})
                .then(data => {{
                    // Hide loading indicator
                    document.getElementById('loadingIndicator').style.display = 'none';
                    
                    // Show answer
                    document.getElementById('answerContainer').style.display = 'block';
                    document.getElementById('answerText').textContent = data.answer;
                    
                    // Show context if available
                    if (data.context) {{
                        document.getElementById('expandButtonContainer').style.display = 'block';
                        document.getElementById('contextText').textContent = data.context;
                    }} else {{
                        document.getElementById('expandButtonContainer').style.display = 'none';
                    }}
                }})
                .catch(error => {{
                    // If the API server fails, show a message with instructions for running the app with --qa
                    document.getElementById('loadingIndicator').style.display = 'none';
                    document.getElementById('errorContainer').style.display = 'block';
                    
                    // Provide detailed instructions for the fallback mechanism
                    var fallbackInstructions = 'Error: ' + error.message + 
                        '\\n\\nTo get answers directly in this visualization, please:' +
                        '\\n1. Close this HTML file if it is open in your browser' +
                        '\\n2. Start the API server: python -m src.graph_db.app --api-server' +
                        '\\n3. Open this HTML file in your browser while the server is running' +
                        '\\n\\nAlternatively, you can run the QA functionality directly from the command line:' +
                        '\\npython -m src.graph_db.app --qa "' + question + '" --qa-json ' + jsonPath + (includeRawText ? ' --qa-include-raw-text' : '');
                    
                    document.getElementById('errorText').textContent = fallbackInstructions;
                }});
            }}
        </script>
        """
        
        return qa_html
    
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
                    color="#000000"  # Set edge color to black
                )
            
            # Create a pyvis network
            net = Network(notebook=False, directed=True, height="750px", width="100%")
            
            # Set options for create_visualization_from_data method
            net.set_options("""
            {
                "nodes": {
                    "shape": "dot",
                    "size": 25,
                    "font": {
                        "size": 64,
                        "face": "Tahoma",
                        "color": "inherit"
                    }
                },
                "edges": {
                    "font": {
                        "size": 32,
                        "align": "middle"
                    },
                    "color": {
                        "color": "#000000",
                        "inherit": false
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
                    "hover": {
                        "enabled": true
                    },
                    "navigationButtons": true,
                    "keyboard": true,
                    "tooltipDelay": 200,
                    "hideEdgesOnDrag": false,
                    "multiselect": false,
                    "hoverConnectedEdges": true
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