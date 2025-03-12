"""
Tests for the Schema Generator module.
"""

import unittest
from src.graph_db.core.schema_generator import SchemaGenerator

class TestSchemaGenerator(unittest.TestCase):
    def setUp(self):
        """Set up test cases."""
        self.generator = SchemaGenerator()
        
    def test_entity_extraction(self):
        """Test entity type extraction from sample text."""
        test_text = """
        John Smith works at Apple Inc. in California. 
        He visited New York last summer for a tech conference.
        """
        
        entity_types = self.generator.extract_entity_types(test_text)
        
        # Check if common entity types are detected
        expected_types = {'PERSON', 'ORG', 'GPE'}  # GPE = Geo-Political Entity
        self.assertTrue(all(t in entity_types for t in expected_types))
        
    def test_relation_extraction(self):
        """Test relation type extraction from sample text."""
        test_text = """
        John Smith works at Apple Inc.
        He visited New York and attended a conference.
        """
        
        relation_types = self.generator.extract_relation_types(test_text)
        
        # Check if verbs are extracted as relations
        expected_relations = {'WORK', 'VISIT', 'ATTEND'}
        self.assertTrue(all(r in relation_types for r in expected_relations))
        
    def test_schema_generation(self):
        """Test complete schema generation."""
        test_text = """
        John Smith works at Apple Inc. in California.
        He visited New York last summer for a tech conference.
        The conference was organized by Microsoft.
        """
        
        entity_types, relation_types = self.generator.generate_schema(test_text)
        
        # Check if both entities and relations are extracted
        self.assertTrue(len(entity_types) >= 3)  # Should at least find PERSON, ORG, GPE
        self.assertTrue(len(relation_types) >= 2)  # Should at least find WORK, VISIT

if __name__ == '__main__':
    unittest.main() 