# Personal Knowledge Graph

A tool for automatically extracting entities and relationships from text to build a personal knowledge graph.

## Features

- Extract entities and relationships from text or URLs using LLM-based extraction
- Dynamically generate schema based on input content
- Store extracted data in Neo4j graph database
- Visualize the graph in interactive HTML format
- Export graph data to JSON

## Installation

### Basic Requirements

1. Clone this repository
2. Install the required Python packages:
   ```
   pip install -r requirements.txt
   ```

### Neo4j Database (Optional)

For full functionality including persistent graph storage:

1. Install [Neo4j Desktop](https://neo4j.com/download/) or use a Neo4j cloud instance
2. Create a new database or use an existing one
3. Update connection details in your environment or command arguments

### API Server Requirements (Optional)

For interactive question answering in the HTML visualization:

1. Install the additional API server requirements:
   ```
   pip install -r requirements-api.txt
   ```

## Environment Setup

Create a `.env` file in the project root with:

```
OPENAI_API_KEY=your_openai_api_key
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password
```

## Usage

### Basic Graph Generation

Process text from a file:
```
python -m src.graph_db.app --file input/sample.txt --output graph.html
```

Process text from a URL:
```
python -m src.graph_db.app --url https://example.com/article --output graph.html
```

### Visualization Only

Visualize an existing JSON graph without rebuilding:
```
python -m src.graph_db.app --visualization-only --json-output graph_data.json --output graph.html
```

### Question Answering

Ask questions about a constructed graph:
```
python -m src.graph_db.app --qa "Who is Frodo Baggins?" --qa-json example/lotr_graph.json
```

Include raw text in context for better answers:
```
python -m src.graph_db.app --qa "What happened at Mount Doom?" --qa-json example/lotr_graph.json --qa-include-raw-text
```

### API Server

Start the API server for interactive QA in the visualization:
```
python -m src.graph_db.app --api-server --api-host localhost --api-port 8000
```

Generate visualization and start API server:
```
python -m src.graph_db.app --file input/sample.txt --output graph.html --api-server --with-visualization
```

### Neo4j Integration

To use Neo4j for graph storage:
```
python -m src.graph_db.app --file input/sample.txt --output graph.html --neo4j-uri bolt://localhost:7687 --neo4j-user neo4j --neo4j-password password
```

## Visualization Features

The HTML visualization provides:

- Interactive graph with draggable nodes
- Color coding for different entity types
- Hover information for nodes and edges
- Zooming and panning capabilities
- Legend showing entity types
- "Show Text" button to view the original source text
- Question Answering panel for asking questions about the graph

## Examples

The `/example` directory contains sample graph visualizations:

- `lotr_graph.html` - A graph visualization of entities and relationships from Lord of the Rings, extracted using LLM-based entity extraction
- `mock_graph.html` - A sample graph visualization using mock data for UI testing

To view these examples, open the HTML files in a web browser.

## Features of the Visualization

- Interactive graph with draggable nodes
- Color-coded entity types
- Tooltips showing entity and relationship details
- Adjustable text size
- Option to show/hide the source text
- Zoom and pan controls

## Architecture

The system consists of several components:

1. **NLP Module**: Extracts entities and relationships from text using LLM-based extraction
2. **Graph Database Module**: Manages Neo4j database operations
3. **Visualization Module**: Creates interactive HTML visualizations of the graph

## License

MIT
