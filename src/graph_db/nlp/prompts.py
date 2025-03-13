"""
Prompts Module

This module contains all the prompts used in the application.
"""

# Entity extraction prompt
ENTITY_EXTRACTION_PROMPT = """
Extract all named entities from the following text. For each entity, provide:
1. The entity name (exactly as it appears in the text)
2. The entity type (PERSON, ORGANIZATION, LOCATION, FACILITY, PRODUCT, EVENT, etc.)
3. Any relevant attributes or properties

IMPORTANT RULES:
- DO NOT extract DATE or TIME as separate entities. Instead, include date/time information as properties of relevant entities or relationships.
- Only extract entities that have clear relationships with other entities in the text.
- Focus on extracting meaningful entities that form a connected graph.

Format the output as a JSON object with an "entities" key containing an array of objects, where each object has the following structure:
{{
    "name": "entity name",
    "type": "ENTITY_TYPE",
    "properties": {{"attribute1": "value1", "attribute2": "value2"}}
}}

Text to analyze:
{text}
"""

# Historical entity extraction prompt
HISTORICAL_ENTITY_EXTRACTION_PROMPT = """
Extract all named entities from the following historical text. For each entity, provide:
1. The entity name (exactly as it appears in the text)
2. The entity type (PERSON, ORGANIZATION, LOCATION, FACILITY, PRODUCT, EVENT, CONCEPT, etc.)
3. Any relevant attributes or properties

For historical texts, pay special attention to:
- Political entities (countries, alliances, empires, etc.)
- Key historical figures (leaders, generals, etc.)
- Significant events (wars, battles, treaties, etc.)
- Important locations (countries, cities, regions, battle sites)
- Concepts and ideologies (democracy, fascism, communism, etc.)

IMPORTANT RULES:
- DO NOT extract DATE or TIME as separate entities. Instead, include date/time information as properties of relevant entities or relationships.
- Only extract entities that have clear relationships with other entities in the text.
- Focus on extracting meaningful entities that form a connected graph.
- For historical figures, include their roles, titles, or positions as properties.
- For events, include when they occurred as properties.
- For locations, include their significance in the historical context as properties.

Format the output as a JSON object with an "entities" key containing an array of objects, where each object has the following structure:
{{
    "name": "entity name",
    "type": "ENTITY_TYPE",
    "properties": {{"attribute1": "value1", "attribute2": "value2"}}
}}

Text to analyze:
{text}
"""

# Entity extraction system message
ENTITY_EXTRACTION_SYSTEM_MESSAGE = "You are an expert in named entity recognition. Extract entities precisely as they appear in the text, following the specified rules."

# Relation extraction prompt
RELATION_EXTRACTION_PROMPT = """
Identify relationships between entities in the following text. The entities are:
{entity_list}

For each relationship, provide:
1. The source entity (exactly as listed above)
2. The target entity (exactly as listed above)
3. The relationship type (a single verb or phrase describing the relationship)
4. A confidence score (0.0 to 1.0)

IMPORTANT RULES:
- Include any relevant date/time information as properties of the relationship, not as separate entities.
- For example, instead of creating a relationship to a date entity, add a "date" or "time" property to the relationship.
- Ensure all entities are connected in the graph - every entity should have at least one relationship.

Format the output as a JSON object with a "relationships" key containing an array of objects, where each object has the following structure:
{{
    "from_entity": "source entity name",
    "to_entity": "target entity name",
    "relation": "RELATIONSHIP_TYPE",
    "confidence": 0.9,
    "properties": {{"date": "1921", "location": "Stockholm", "other_attribute": "value"}}
}}

Only include relationships that are explicitly stated in the text. Do not infer relationships that aren't directly mentioned.

Text to analyze:
{text}
"""

# Historical relation extraction prompt
HISTORICAL_RELATION_EXTRACTION_PROMPT = """
Identify relationships between historical entities in the following text. The entities are:
{entity_list}

For each relationship, provide:
1. The source entity (exactly as listed above)
2. The target entity (exactly as listed above)
3. The relationship type (a single verb or phrase describing the relationship)
4. A confidence score (0.0 to 1.0)

For historical texts, pay special attention to these relationship types:
- Political relationships (ALLIED_WITH, ENEMY_OF, INVADED, OCCUPIED, etc.)
- Leadership relationships (LED, RULED, GOVERNED, etc.)
- Participation relationships (PARTICIPATED_IN, FOUGHT_IN, SIGNED, etc.)
- Temporal relationships (PRECEDED, FOLLOWED, DURING, etc.)
- Causal relationships (CAUSED, RESULTED_IN, LED_TO, etc.)
- Membership relationships (MEMBER_OF, PART_OF, etc.)

IMPORTANT RULES:
- Include any relevant date/time information as properties of the relationship, not as separate entities.
- For example, instead of creating a relationship to a date entity, add a "date" or "time" property to the relationship.
- Ensure all entities are connected in the graph - every entity should have at least one relationship.
- For military conflicts, specify who fought whom, when, and the outcome if mentioned.
- For political alliances, specify the time period and context.
- For treaties and agreements, specify the signatories and date.

Format the output as a JSON object with a "relationships" key containing an array of objects, where each object has the following structure:
{{
    "from_entity": "source entity name",
    "to_entity": "target entity name",
    "relation": "RELATIONSHIP_TYPE",
    "confidence": 0.9,
    "properties": {{"date": "1939-1945", "location": "Europe", "outcome": "Allied victory"}}
}}

Only include relationships that are explicitly stated in the text. Do not infer relationships that aren't directly mentioned.

Text to analyze:
{text}
"""

# Relation extraction system message
RELATION_EXTRACTION_SYSTEM_MESSAGE = "You are an expert in relationship extraction from text. Extract relationships precisely as they appear in the text, following the specified rules." 