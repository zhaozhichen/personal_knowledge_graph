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
        
    def visualize(self, 
                  graph_data: Dict[str, Any], 
                  output_path: str = "graph.html",
                  title: str = "Graph Visualization",
                  raw_text: str = None) -> bool:
        """
        Visualize the graph data in HTML format.
        
        Args:
            graph_data (Dict[str, Any]): Graph data containing entities and relations
            output_path (str): Path to save the HTML file
            title (str): Title of the visualization
            raw_text (str): Original text used to generate the graph
            
        Returns:
            bool: True if visualization successful, False otherwise
        """
        try:
            # Create a networkx graph
            G = nx.DiGraph()
            
            # Track entity types used in this visualization
            used_entity_types = set()
            
            # Add nodes (entities)
            for entity in graph_data["entities"]:
                entity_type = entity.get("types", ["UNKNOWN"])[0].upper()  # Ensure uppercase
                used_entity_types.add(entity_type)
                color = self.entity_colors.get(entity_type, self.default_color)
                
                # Create a formatted title with all properties
                properties = entity.get("properties", {})
                title_text = f"{entity['name']} ({entity_type})"
                if properties:
                    title_text += "\n\nProperties:"
                    for prop_key, prop_value in properties.items():
                        title_text += f"\n• {prop_key}: {prop_value}"
                
                G.add_node(
                    entity["name"],
                    label=entity["name"],
                    title=title_text,  # Use plain text title
                    color=color,
                    font={'color': color}  # Set font color to match node color
                )
                
            # Add edges (relations)
            for relation in graph_data["relations"]:
                # Create a formatted title with all properties
                properties = relation.get("properties", {})
                title_text = relation['relation']
                if properties:
                    title_text += "\n\nProperties:"
                    for prop_key, prop_value in properties.items():
                        title_text += f"\n• {prop_key}: {prop_value}"
                
                G.add_edge(
                    relation["from_name"],
                    relation["to_name"],
                    title=title_text,  # Use plain text title
                    label=relation["relation"],
                    arrows="to",
                    color="#000000"  # Set edge color to black
                )
                
            # Create a pyvis network from the networkx graph
            net = Network(
                height=self.height,
                width=self.width,
                directed=True,
                notebook=False,
                heading=title
            )
            
            # Configure physics
            net.barnes_hut(
                gravity=-80000,
                central_gravity=0.3,
                spring_length=250,
                spring_strength=0.001,
                damping=0.09,
                overlap=0
            )
            
            # Configure other options
            net.set_options("""
            var options = {
                "nodes": {
                    "font": {
                        "size": 32,
                        "face": "Tahoma",
                        "color": "inherit"
                    },
                    "borderWidth": 2,
                    "borderWidthSelected": 4,
                    "size": 30
                },
                "edges": {
                    "font": {
                        "size": 32,
                        "face": "Tahoma"
                    },
                    "width": 2,
                    "color": {
                        "color": "#000000",
                        "inherit": false
                    },
                    "smooth": {
                        "type": "continuous",
                        "forceDirection": "none"
                    }
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
                },
                "physics": {
                    "stabilization": {
                        "iterations": 1000
                    }
                }
            }
            """)
            
            # Add the networkx graph to the pyvis network
            net.from_nx(G)
            
            # Generate HTML for the legend and raw text
            legend_html = self._generate_legend_html(used_entity_types)
            raw_text_html = self._generate_raw_text_html(raw_text)
            
            # Save the visualization to an HTML file
            net.save_graph(output_path)
            
            # Add the legend and raw text to the HTML file
            self._add_html_content(output_path, legend_html, raw_text_html)
            
            # Save the graph data as JSON
            self._save_graph_data_as_json(graph_data["entities"], graph_data["relations"], output_path)
            
            self.logger.info(f"Graph visualization saved to {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error visualizing graph: {str(e)}")
            return False
    
    def _generate_legend_html(self, used_entity_types: set) -> str:
        """Generate HTML for the entity type legend."""
        if not used_entity_types:
            return ""
            
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
        
        for entity_type in sorted(used_entity_types):
            color = self.entity_colors.get(entity_type, self.default_color)
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
                    <input type="range" id="textSizeSlider" min="8" max="64" value="32" style="flex-grow: 1;">
                    <span style="margin-left: 5px; font-size: 18px;">A</span>
                </div>
                <div style="text-align: center; margin-top: 5px;">
                    <span id="currentTextSize">32px</span>
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
                        
                        // Update edge font size
                        var edgeOptions = {
                            font: {
                                size: size
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
                    
                    // Add event listeners for entity type highlighting
                    var entityTypeRows = document.querySelectorAll('.entity-type-row');
                    entityTypeRows.forEach(function(row) {
                        row.addEventListener('click', function() {
                            var entityType = this.getAttribute('data-entity-type');
                            highlightNodesByType(entityType);
                            
                            // Update visual feedback in the table
                            entityTypeRows.forEach(function(r) {
                                r.style.backgroundColor = '';
                                r.style.fontWeight = 'normal';
                            });
                            this.style.backgroundColor = '#f0f0f0';
                            this.style.fontWeight = 'bold';
                        });
                    });
                    
                    // Add event listener for reset button
                    var resetBtn = document.getElementById('resetHighlightBtn');
                    if (resetBtn) {
                        resetBtn.addEventListener('click', function() {
                            resetHighlighting();
                            
                            // Reset visual feedback in the table
                            entityTypeRows.forEach(function(r) {
                                r.style.backgroundColor = '';
                                r.style.fontWeight = 'normal';
                            });
                        });
                    }
                    
                    // Function to highlight nodes by type
                    function highlightNodesByType(entityType) {
                        if (typeof network === 'undefined') {
                            console.warn('Network variable not found. Highlighting may not work properly.');
                            return;
                        }
                        
                        var allNodes = network.body.nodes;
                        var allEdges = network.body.edges;
                        
                        // Lowlight all nodes and edges
                        Object.values(allNodes).forEach(function(node) {
                            if (node.options) {
                                // Save original values if not already saved
                                if (!node.options._originalColor) {
                                    node.options._originalColor = node.options.color;
                                    node.options._originalFont = JSON.parse(JSON.stringify(node.options.font));
                                    node.options._originalSize = node.options.size;
                                }
                                
                                // Apply lowlight effect
                                node.options.color = {
                                    background: '#f0f0f0',
                                    border: '#e0e0e0'
                                };
                                node.options.font.color = '#aaaaaa';
                                node.options.size = node.options.size * 0.8;
                            }
                        });
                        
                        Object.values(allEdges).forEach(function(edge) {
                            if (edge.options) {
                                // Save original values if not already saved
                                if (!edge.options._originalColor) {
                                    edge.options._originalColor = edge.options.color;
                                    edge.options._originalWidth = edge.options.width;
                                    edge.options._originalFont = JSON.parse(JSON.stringify(edge.options.font));
                                }
                                
                                // Apply lowlight effect
                                edge.options.color = {
                                    color: '#e0e0e0',
                                    highlight: '#e0e0e0'
                                };
                                edge.options.width = edge.options.width * 0.5;
                                edge.options.font.color = '#aaaaaa';
                            }
                        });
                        
                        // Highlight nodes of the selected type and their connections
                        var highlightedNodeIds = [];
                        Object.values(allNodes).forEach(function(node) {
                            if (node.options && node.options.title && node.options.title.includes('(' + entityType + ')')) {
                                // Restore original values for this node
                                if (node.options._originalColor) {
                                    node.options.color = node.options._originalColor;
                                    node.options.font = JSON.parse(JSON.stringify(node.options._originalFont));
                                    node.options.size = node.options._originalSize * 1.2; // Make slightly larger
                                }
                                highlightedNodeIds.push(node.id);
                            }
                        });
                        
                        // Highlight edges connected to highlighted nodes
                        Object.values(allEdges).forEach(function(edge) {
                            if (edge.options && 
                                (highlightedNodeIds.includes(edge.from) || highlightedNodeIds.includes(edge.to))) {
                                // Restore original values for this edge
                                if (edge.options._originalColor) {
                                    edge.options.color = edge.options._originalColor;
                                    edge.options.width = edge.options._originalWidth;
                                    edge.options.font = JSON.parse(JSON.stringify(edge.options._originalFont));
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
                                
                                // Clear saved original values
                                delete node.options._originalColor;
                                delete node.options._originalFont;
                                delete node.options._originalSize;
                            }
                        });
                        
                        Object.values(network.body.edges).forEach(function(edge) {
                            if (edge.options && edge.options._originalColor) {
                                edge.options.color = edge.options._originalColor;
                                edge.options.width = edge.options._originalWidth;
                                edge.options.font = JSON.parse(JSON.stringify(edge.options._originalFont));
                                
                                // Clear saved original values
                                delete edge.options._originalColor;
                                delete edge.options._originalWidth;
                                delete edge.options._originalFont;
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
    
    def _add_html_content(self, html_file: str, legend_html: str, raw_text_html: str) -> None:
        """Add the legend and raw text HTML to the visualization file."""
        with open(html_file, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Add custom tooltip CSS and JavaScript
        tooltip_js = """
        <style>
        .custom-tooltip {
            position: absolute;
            display: none;
            padding: 10px;
            font-family: Arial, sans-serif;
            font-size: 14px;
            color: #333;
            background-color: white;
            border: 1px solid #ccc;
            border-radius: 4px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            z-index: 1000;
            max-width: 300px;
            word-wrap: break-word;
            white-space: pre-wrap;
            line-height: 1.5;
        }
        </style>
        <div id="custom-tooltip" class="custom-tooltip"></div>
        <script>
        // Initialize global variable for tooltip control - always enabled
        window.tooltipsEnabled = true;
        
        // Add the tooltip functionality after the page has fully loaded
        window.addEventListener('load', function() {
            // Make sure the network variable exists
            if (typeof network === 'undefined') {
                console.warn('Network variable not found');
                return;
            }
            
            var tooltip = document.getElementById('custom-tooltip');
            
            // Function to disable default tooltips
            function disableDefaultTooltips() {
                // Remove title attributes from all nodes and edges to prevent default browser tooltips
                if (network && network.body) {
                    // For nodes
                    Object.values(network.body.nodes).forEach(function(node) {
                        if (node.element) {
                            node.element.removeAttribute('title');
                        }
                    });
                    
                    // For edges
                    Object.values(network.body.edges).forEach(function(edge) {
                        if (edge.element) {
                            edge.element.removeAttribute('title');
                        }
                    });
                }
            }
            
            // Call once on load
            disableDefaultTooltips();
            
            // Function to show tooltip
            function showTooltip(text, x, y) {
                tooltip.textContent = text;
                tooltip.style.display = 'block';
                tooltip.style.left = (x + 10) + 'px';
                tooltip.style.top = (y + 10) + 'px';
            }
            
            // Function to hide tooltip
            function hideTooltip() {
                tooltip.style.display = 'none';
            }
            
            // Add event listeners to the network
            network.on('hoverNode', function(params) {
                var nodeId = params.node;
                var node = network.body.nodes[nodeId];
                if (node && node.options && node.options.title) {
                    showTooltip(node.options.title, params.pointer.DOM.x, params.pointer.DOM.y);
                }
            });
            
            network.on('hoverEdge', function(params) {
                var edgeId = params.edge;
                var edge = network.body.edges[edgeId];
                if (edge && edge.options && edge.options.title) {
                    showTooltip(edge.options.title, params.pointer.DOM.x, params.pointer.DOM.y);
                }
            });
            
            network.on('blurNode', function() {
                hideTooltip();
            });
            
            network.on('blurEdge', function() {
                hideTooltip();
            });
            
            // Hide tooltip during drag operations
            network.on('dragStart', function() {
                hideTooltip();
            });
            
            // Hide tooltip during zoom operations
            network.on('zoom', function() {
                hideTooltip();
            });
            
            // Ensure hover is enabled for tooltips
            network.setOptions({
                interaction: {
                    hover: {
                        enabled: true
                    }
                }
            });
        });
        </script>
        """
            
        # Insert the legend before the closing body tag
        if legend_html:
            content = content.replace('</body>', f'{legend_html}</body>')
            
        # Insert the raw text before the closing body tag
        if raw_text_html:
            # Extract the script part from raw_text_html
            script_start = raw_text_html.find('<script>')
            script_end = raw_text_html.find('</script>') + 9  # Include the </script> tag
            
            if script_start != -1 and script_end != -1:
                script_part = raw_text_html[script_start:script_end]
                html_part = raw_text_html[:script_start] + raw_text_html[script_end:]
                
                # Insert the HTML part before the closing body tag
                content = content.replace('</body>', f'{html_part}</body>')
                
                # Insert the script part right before the closing body tag to ensure it runs after the network is initialized
                content = content.replace('</body>', f'{script_part}</body>')
            else:
                content = content.replace('</body>', f'{raw_text_html}</body>')
        
        # Add the tooltip implementation right before the closing body tag
        content = content.replace('</body>', f'{tooltip_js}</body>')
            
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(content)
            
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
                        "size": 32,
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
            legend_html = self._generate_legend_html(used_entity_types)
            raw_text_html = self._generate_raw_text_html(raw_text)
            
            # Add the legend and raw text to the HTML file
            self._add_html_content(output_path, legend_html, raw_text_html)
            
            # Save the graph data as JSON
            self._save_graph_data_as_json(entities, relations, output_path)
            
            self.logger.info(f"Graph visualization saved to {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error creating visualization: {str(e)}")
            return False
            
    def _save_graph_data_as_json(self, entities: List[Dict[str, Any]], relations: List[Dict[str, Any]], html_path: str) -> None:
        """
        Save the graph data (entities and relations) as a JSON file.
        
        Args:
            entities (List[Dict[str, Any]]): List of entities
            relations (List[Dict[str, Any]]): List of relations
            html_path (str): Path to the HTML visualization file
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
                }
            }
            
            # Save the data
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(graph_data, f, indent=2, ensure_ascii=False)
                
            self.logger.info(f"Graph data saved to {json_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving graph data as JSON: {str(e)}") 