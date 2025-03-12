"""
Test script to verify schema generator functionality.
"""

from core.schema_generator import SchemaGenerator

def main():
    # Initialize schema generator
    generator = SchemaGenerator()
    
    # Test text
    test_text = """
    John Smith is the CEO of Apple Inc., based in Cupertino, California.
    He previously worked at Microsoft and Google before joining Apple in 2020.
    During his tenure at Microsoft, he led the development of Windows 11.
    """
    
    print("Analyzing text...")
    print("-" * 50)
    print(test_text)
    print("-" * 50)
    
    # Generate schema
    entity_types, relation_types = generator.generate_schema(test_text)
    
    # Print results
    print("\nIdentified Entity Types:")
    print("-" * 50)
    for entity_type in sorted(entity_types):
        print(f"- {entity_type}")
        
    print("\nIdentified Relation Types:")
    print("-" * 50)
    for relation_type in sorted(relation_types):
        print(f"- {relation_type}")

if __name__ == "__main__":
    main() 