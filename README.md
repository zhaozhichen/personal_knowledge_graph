# Personal Knowledge Graph

A tool for automatically extracting entities and relationships from text to build a personal knowledge graph.

## Features

- Extract entities and relationships from text or URLs using LLM-based extraction
- Dynamically generate schema based on input content
- Store extracted data in Neo4j graph database
- Visualize the graph in interactive HTML format
- Export graph data to JSON

## Installation

### Prerequisites

- Python 3.9+
- Neo4j (optional, for database storage)

### Setup

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/personal_knowledge_graph.git
   cd personal_knowledge_graph
   ```

2. Create a virtual environment and install dependencies:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Set up environment variables (optional, for Neo4j):
   ```
   export NEO4J_URI="bolt://localhost:7687"
   export NEO4J_USER="neo4j"
   export NEO4J_PASSWORD="password"
   ```

## Usage

### Process Text

```bash
python -m src.graph_db.app --text "Your text here" --output graph.html
```

### Process File

```bash
python -m src.graph_db.app --file path/to/file.txt --output graph.html
```

### Process URL

```bash
python -m src.graph_db.app --url "https://example.com" --output graph.html
```

### Visualization Only (No Neo4j)

Add the `--visualization-only` flag to skip Neo4j database operations:

```bash
python -m src.graph_db.app --text "Your text here" --visualization-only --output graph.html
```

## Architecture

The system consists of several components:

1. **NLP Module**: Extracts entities and relationships from text using LLM-based extraction
2. **Graph Database Module**: Manages Neo4j database operations
3. **Visualization Module**: Creates interactive HTML visualizations of the graph

## License

MIT
