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
                title = f"<b>{entity['name']} ({entity_type})</b>"
                if properties:
                    title += "<br><br><b>Properties:</b><br>"
                    for prop_key, prop_value in properties.items():
                        title += f"• {prop_key}: {prop_value}<br>"
                
                G.add_node(
                    entity["name"],
                    label=entity["name"],
                    title=title,
                    color=color,
                    font={'color': color}  # Set font color to match node color
                )
                
            # Add edges (relations)
            for relation in graph_data["relations"]:
                # Create a formatted title with all properties
                properties = relation.get("properties", {})
                title = f"<b>{relation['relation']}</b>"
                if properties:
                    title += "<br><br><b>Properties:</b><br>"
                    for prop_key, prop_value in properties.items():
                        title += f"• {prop_key}: {prop_value}<br>"
                
                G.add_edge(
                    relation["from_name"],
                    relation["to_name"],
                    title=title,
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
                        "size": 14,
                        "face": "Tahoma",
                        "color": "inherit"
                    },
                    "borderWidth": 2,
                    "borderWidthSelected": 4,
                    "size": 30
                },
                "edges": {
                    "font": {
                        "size": 14,
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
                        "enabled": true,
                        "title": false
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
            <table style="width: 100%; border-collapse: collapse;">
                <tr>
                    <th style="text-align: left; padding: 5px;">Type</th>
                    <th style="text-align: left; padding: 5px;">Color</th>
                </tr>
        """
        
        for entity_type in sorted(used_entity_types):
            color = self.entity_colors.get(entity_type, self.default_color)
            legend_html += f"""
                <tr>
                    <td style="padding: 5px;">{entity_type}</td>
                    <td style="padding: 5px;"><div style="width: 20px; height: 20px; background-color: {color}; 
                                                border-radius: 50%; display: inline-block;"></div></td>
                </tr>
            """
            
        legend_html += """
            </table>
            <div style="margin-top: 15px; border-top: 1px solid #eee; padding-top: 10px;">
                <h4 style="margin-top: 0; text-align: center;">Text Size</h4>
                <div style="display: flex; align-items: center; margin-top: 5px;">
                    <span style="margin-right: 5px; font-size: 12px;">A</span>
                    <input type="range" id="textSizeSlider" min="8" max="64" value="14" style="flex-grow: 1;">
                    <span style="margin-left: 5px; font-size: 18px;">A</span>
                </div>
                <div style="text-align: center; margin-top: 5px;">
                    <span id="currentTextSize">14px</span>
                </div>
            </div>
            <div style="margin-top: 15px; border-top: 1px solid #eee; padding-top: 10px;">
                <h4 style="margin-top: 0; text-align: center;">Display Options</h4>
                <div style="display: flex; align-items: center; justify-content: space-between; margin-top: 10px;">
                    <span>Show Tooltips:</span>
                    <label class="switch" style="position: relative; display: inline-block; width: 40px; height: 20px;">
                        <input type="checkbox" id="tooltipToggle" checked style="opacity: 0; width: 0; height: 0;">
                        <span style="position: absolute; cursor: pointer; top: 0; left: 0; right: 0; bottom: 0; background-color: #ccc; transition: .4s; border-radius: 34px;"></span>
                        <span style="position: absolute; cursor: pointer; top: 2px; left: 2px; height: 16px; width: 16px; background-color: white; transition: .4s; border-radius: 50%;"></span>
                    </label>
                </div>
            </div>
            <script>
                // Add event listener to the slider
                document.addEventListener('DOMContentLoaded', function() {
                    var slider = document.getElementById('textSizeSlider');
                    var sizeDisplay = document.getElementById('currentTextSize');
                    var tooltipToggle = document.getElementById('tooltipToggle');
                    
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
                    
                    // Function to toggle tooltips
                    function toggleTooltips(enabled) {
                        // Update the toggle switch appearance
                        var toggleSwitch = tooltipToggle.nextElementSibling.nextElementSibling;
                        if (enabled) {
                            toggleSwitch.style.transform = 'translateX(20px)';
                            tooltipToggle.nextElementSibling.style.backgroundColor = '#4CAF50';
                        } else {
                            toggleSwitch.style.transform = 'translateX(0)';
                            tooltipToggle.nextElementSibling.style.backgroundColor = '#ccc';
                        }
                        
                        // Set global variable to control tooltip display
                        window.tooltipsEnabled = enabled;
                        
                        // Update network interaction options
                        if (typeof network !== 'undefined') {
                            network.setOptions({
                                interaction: {
                                    hover: {
                                        enabled: enabled,
                                        title: false
                                    }
                                }
                            });
                            
                            // Hide any visible tooltip when disabled
                            if (!enabled) {
                                var customTooltip = document.getElementById('custom-tooltip');
                                if (customTooltip) {
                                    customTooltip.style.display = 'none';
                                }
                            }
                            
                            // Re-apply the title attribute removal
                            if (typeof disableDefaultTooltips === 'function') {
                                disableDefaultTooltips();
                            }
                        }
                    }
                    
                    // Add event listener for slider changes
                    slider.addEventListener('input', function() {
                        updateTextSize(parseInt(this.value));
                    });
                    
                    // Add event listener for tooltip toggle
                    tooltipToggle.addEventListener('change', function() {
                        toggleTooltips(this.checked);
                    });
                    
                    // Initialize with default values
                    updateTextSize(parseInt(slider.value));
                    toggleTooltips(tooltipToggle.checked);
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
            
        # Remove all title attributes from the HTML content
        content = re.sub(r'title="[^"]*"', '', content)
            
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
        }
        
        /* Hide default browser tooltips */
        #mynetwork [title] {
            position: relative;
        }
        #mynetwork [title]:hover::after {
            content: none !important;
        }
        
        /* Additional CSS to prevent tooltips */
        [title] {
            position: relative;
        }
        [title]:hover::after {
            content: none !important;
        }
        </style>
        <div id="custom-tooltip" class="custom-tooltip"></div>
        <script>
        // Initialize global variable for tooltip control
        window.tooltipsEnabled = true;
        
        // Store node and edge data for custom tooltips
        window.nodeTooltips = {};
        window.edgeTooltips = {};
        
        // Add the tooltip functionality after the page has fully loaded
        window.addEventListener('load', function() {
            // Make sure the network variable exists
            if (typeof network === 'undefined') {
                console.warn('Network variable not found');
                return;
            }
            
            var tooltip = document.getElementById('custom-tooltip');
            
            // Store tooltip data from nodes and edges
            function storeTooltipData() {
                // For nodes
                var nodeIds = network.body.data.nodes.getIds();
                nodeIds.forEach(function(nodeId) {
                    var node = network.body.data.nodes.get(nodeId);
                    if (node && node.title) {
                        window.nodeTooltips[nodeId] = node.title;
                        // Remove the title attribute
                        delete node.title;
                    }
                });
                
                // For edges
                var edgeIds = network.body.data.edges.getIds();
                edgeIds.forEach(function(edgeId) {
                    var edge = network.body.data.edges.get(edgeId);
                    if (edge && edge.title) {
                        window.edgeTooltips[edgeId] = edge.title;
                        // Remove the title attribute
                        delete edge.title;
                    }
                });
                
                // Update the network with the modified data
                network.setData({
                    nodes: network.body.data.nodes,
                    edges: network.body.data.edges
                });
            }
            
            // Completely disable default tooltips by removing title attributes and preventing default behavior
            function disableDefaultTooltips() {
                // Remove all title attributes from the DOM
                document.querySelectorAll('[title]').forEach(function(element) {
                    element.removeAttribute('title');
                });
                
                // For nodes
                var nodeIds = network.body.data.nodes.getIds();
                nodeIds.forEach(function(nodeId) {
                    var nodeElement = network.body.nodes[nodeId].element;
                    if (nodeElement) {
                        nodeElement.removeAttribute('title');
                    }
                });
                
                // For edges
                var edgeIds = network.body.data.edges.getIds();
                edgeIds.forEach(function(edgeId) {
                    var edgeElement = network.body.edges[edgeId].element;
                    if (edgeElement) {
                        edgeElement.removeAttribute('title');
                    }
                });
                
                // Also remove titles from DOM elements
                var canvasElements = document.querySelectorAll('canvas');
                canvasElements.forEach(function(canvas) {
                    canvas.removeAttribute('title');
                });
                
                // Remove titles from all elements in the network container
                var networkContainer = document.getElementById('mynetwork');
                if (networkContainer) {
                    var allElements = networkContainer.querySelectorAll('*');
                    allElements.forEach(function(element) {
                        element.removeAttribute('title');
                    });
                }
            }
            
            // Store tooltip data
            storeTooltipData();
            
            // Call once at initialization
            disableDefaultTooltips();
            
            // Also call when network is redrawn
            network.on("afterDrawing", disableDefaultTooltips);
            
            // Prevent default title behavior for the entire network container
            var networkContainer = document.getElementById('mynetwork');
            if (networkContainer) {
                // Prevent default title behavior
                networkContainer.addEventListener('mouseover', function(event) {
                    if (event.target.hasAttribute('title')) {
                        event.target.dataset.originalTitle = event.target.getAttribute('title');
                        event.target.removeAttribute('title');
                    }
                }, true);
            }
            
            // Function to show tooltip
            function showTooltip(html, x, y) {
                // Only show tooltip if tooltips are enabled
                if (!window.tooltipsEnabled) {
                    return;
                }
                
                tooltip.innerHTML = html;
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
                var tooltipHtml = window.nodeTooltips[nodeId];
                
                if (tooltipHtml) {
                    // Create a div to parse the HTML and extract text
                    var div = document.createElement('div');
                    div.innerHTML = tooltipHtml;
                    
                    // Format the tooltip content properly
                    var entityName = div.querySelector('b').textContent;
                    var tooltipContent = '<div><strong>' + entityName + '</strong>';
                    
                    // Check if there are properties
                    if (div.innerHTML.includes('Properties:')) {
                        tooltipContent += '<br><br><strong>Properties:</strong><ul>';
                        var propertyItems = div.innerHTML.split('•').slice(1);
                        propertyItems.forEach(function(item) {
                            if (item.trim()) {
                                var propText = item.split('<br>')[0].trim();
                                tooltipContent += '<li>' + propText + '</li>';
                            }
                        });
                        tooltipContent += '</ul>';
                    }
                    
                    tooltipContent += '</div>';
                    showTooltip(tooltipContent, params.pointer.DOM.x, params.pointer.DOM.y);
                }
            });
            
            network.on('hoverEdge', function(params) {
                var edgeId = params.edge;
                var tooltipHtml = window.edgeTooltips[edgeId];
                
                if (tooltipHtml) {
                    // Create a div to parse the HTML and extract text
                    var div = document.createElement('div');
                    div.innerHTML = tooltipHtml;
                    
                    // Format the tooltip content properly
                    var relationName = div.querySelector('b').textContent;
                    var tooltipContent = '<div><strong>' + relationName + '</strong>';
                    
                    // Check if there are properties
                    if (div.innerHTML.includes('Properties:')) {
                        tooltipContent += '<br><br><strong>Properties:</strong><ul>';
                        var propertyItems = div.innerHTML.split('•').slice(1);
                        propertyItems.forEach(function(item) {
                            if (item.trim()) {
                                var propText = item.split('<br>')[0].trim();
                                tooltipContent += '<li>' + propText + '</li>';
                            }
                        });
                        tooltipContent += '</ul>';
                    }
                    
                    tooltipContent += '</div>';
                    showTooltip(tooltipContent, params.pointer.DOM.x, params.pointer.DOM.y);
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
                title = f"<b>{entity['name']} ({entity['type']})</b>"
                if properties:
                    title += "<br><br><b>Properties:</b><br>"
                    for prop_key, prop_value in properties.items():
                        title += f"• {prop_key}: {prop_value}<br>"
                
                G.add_node(
                    node_id, 
                    label=entity["name"], 
                    title=title,
                    color=color,  # Explicitly set color instead of group
                    font={'color': color},  # Set font color to match node color
                    properties=properties
                )
            
            # Add edges
            for relation in relations:
                from_id = relation["from_entity"].get("id", hash(relation["from_entity"]["name"]))
                to_id = relation["to_entity"].get("id", hash(relation["to_entity"]["name"]))
                
                # Create a formatted title with all properties
                properties = relation.get("properties", {})
                title = f"<b>{relation['relation']}</b>"
                if properties:
                    title += "<br><br><b>Properties:</b><br>"
                    for prop_key, prop_value in properties.items():
                        title += f"• {prop_key}: {prop_value}<br>"
                
                G.add_edge(
                    from_id, 
                    to_id, 
                    label=relation["relation"],
                    title=title,
                    properties=properties,
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
                        "size": 14,
                        "face": "Tahoma",
                        "color": "inherit"
                    }
                },
                "edges": {
                    "font": {
                        "size": 14,
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
                        "enabled": true,
                        "title": false
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