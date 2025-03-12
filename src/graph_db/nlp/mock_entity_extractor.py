"""
Mock Entity Extractor Module

This module provides a mock implementation of the entity extractor for testing purposes.
"""

import logging
import uuid
from typing import Dict, List, Tuple, Any

class MockEntityExtractor:
    def __init__(self, model: str = "mock"):
        """
        Initialize the mock entity extractor.
        
        Args:
            model (str): Model name (ignored)
        """
        self.logger = logging.getLogger(__name__)
        
    def process_text(self, text: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Process text to extract entities and relations.
        
        Args:
            text (str): Input text to analyze
            
        Returns:
            Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]: Tuple of (entities, relations)
        """
        # Generate mock entities based on the text content
        entities = []
        relations = []
        
        # Check for keywords to determine which entities to create
        if "Newton" in text:
            entities, relations = self._generate_newton_data()
        elif "Einstein" in text:
            entities, relations = self._generate_einstein_data()
        elif "Descartes" in text:
            entities, relations = self._generate_descartes_data()
        
        return entities, relations
    
    def _generate_newton_data(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Generate mock data for Newton"""
        # Create entities
        newton = self._create_entity("Isaac Newton", "PERSON", {"birth_year": "1642", "death_year": "1727"})
        woolsthorpe = self._create_entity("Woolsthorpe", "LOCATION", {"country": "England"})
        trinity_college = self._create_entity("Trinity College, Cambridge", "ORGANIZATION", {"founded": "1546"})
        calculus = self._create_entity("Calculus", "WORK_OF_ART", {"field": "Mathematics"})
        leibniz = self._create_entity("Gottfried Wilhelm Leibniz", "PERSON", {"birth_year": "1646", "death_year": "1716"})
        principia = self._create_entity("Principia Mathematica", "WORK_OF_ART", {"published": "1687", "field": "Physics"})
        laws_of_motion = self._create_entity("Three Laws of Motion", "WORK_OF_ART", {"field": "Physics"})
        universal_gravitation = self._create_entity("Law of Universal Gravitation", "WORK_OF_ART", {"field": "Physics"})
        reflecting_telescope = self._create_entity("Reflecting Telescope", "PRODUCT", {"invented": "1668"})
        prism = self._create_entity("Prism", "PRODUCT", {})
        royal_mint = self._create_entity("Royal Mint", "ORGANIZATION", {"country": "England"})
        royal_society = self._create_entity("Royal Society", "ORGANIZATION", {"founded": "1660"})
        queen_anne = self._create_entity("Queen Anne", "PERSON", {"reign": "1702-1714"})
        
        # Create relations
        relations = [
            self._create_relation(newton, woolsthorpe, "BORN IN", {"year": "1642"}),
            self._create_relation(newton, trinity_college, "STUDIED AT", {"year": "1661"}),
            self._create_relation(newton, calculus, "DEVELOPED", {"year": "1665-1666"}),
            self._create_relation(leibniz, calculus, "DEVELOPED", {"year": "1675-1676"}),
            self._create_relation(newton, principia, "WROTE", {"year": "1687"}),
            self._create_relation(newton, laws_of_motion, "FORMULATED", {}),
            self._create_relation(newton, universal_gravitation, "DISCOVERED", {}),
            self._create_relation(newton, reflecting_telescope, "INVENTED", {"year": "1668"}),
            self._create_relation(newton, prism, "EXPERIMENTED WITH", {"discovery": "Light Spectrum"}),
            self._create_relation(newton, royal_mint, "SERVED AS WARDEN OF", {"period": "1696-1699"}),
            self._create_relation(newton, royal_mint, "SERVED AS MASTER OF", {"period": "1699-1727"}),
            self._create_relation(newton, royal_society, "BECAME PRESIDENT OF", {"year": "1703"}),
            self._create_relation(queen_anne, newton, "KNIGHTED", {"year": "1705"})
        ]
        
        entities = [newton, woolsthorpe, trinity_college, calculus, leibniz, principia, 
                   laws_of_motion, universal_gravitation, reflecting_telescope, prism, 
                   royal_mint, royal_society, queen_anne]
        
        return entities, relations
    
    def _generate_einstein_data(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Generate mock data for Einstein"""
        # Create entities
        einstein = self._create_entity("Albert Einstein", "PERSON", {"birth_year": "1879", "death_year": "1955"})
        ulm = self._create_entity("Ulm", "LOCATION", {"country": "Germany"})
        germany = self._create_entity("Germany", "LOCATION", {})
        usa = self._create_entity("United States", "LOCATION", {})
        swiss_polytechnic = self._create_entity("Swiss Federal Polytechnic School", "ORGANIZATION", {})
        special_relativity = self._create_entity("Theory of Special Relativity", "WORK_OF_ART", {"year": "1905"})
        general_relativity = self._create_entity("General Theory of Relativity", "WORK_OF_ART", {"year": "1915"})
        photoelectric_effect = self._create_entity("Photoelectric Effect", "WORK_OF_ART", {"year": "1905"})
        brownian_motion = self._create_entity("Brownian Motion", "WORK_OF_ART", {"year": "1905"})
        mass_energy = self._create_entity("Mass-Energy Equivalence", "WORK_OF_ART", {"equation": "E=mc²"})
        nobel_prize = self._create_entity("1921 Nobel Prize in Physics", "EVENT", {})
        princeton = self._create_entity("Institute for Advanced Study in Princeton", "ORGANIZATION", {})
        maric = self._create_entity("Mileva Marić", "PERSON", {})
        lowenthal = self._create_entity("Elsa Löwenthal", "PERSON", {})
        hitler = self._create_entity("Adolf Hitler", "PERSON", {})
        
        # Create relations
        relations = [
            self._create_relation(einstein, ulm, "BORN IN", {"year": "1879"}),
            self._create_relation(ulm, germany, "LOCATED IN", {}),
            self._create_relation(einstein, swiss_polytechnic, "STUDIED AT", {}),
            self._create_relation(einstein, special_relativity, "INTRODUCED", {"year": "1905"}),
            self._create_relation(einstein, general_relativity, "DEVELOPED", {"year": "1915"}),
            self._create_relation(einstein, photoelectric_effect, "EXPLAINED", {"year": "1905"}),
            self._create_relation(einstein, brownian_motion, "EXPLAINED", {"year": "1905"}),
            self._create_relation(einstein, mass_energy, "FORMULATED", {"year": "1905"}),
            self._create_relation(einstein, nobel_prize, "EARNED", {"year": "1921"}),
            self._create_relation(einstein, maric, "MARRIED", {"period": "1903-1919"}),
            self._create_relation(einstein, lowenthal, "MARRIED", {"period": "1919-1936"}),
            self._create_relation(einstein, germany, "FLED", {"year": "1933"}),
            self._create_relation(einstein, usa, "MOVED TO", {"year": "1933"}),
            self._create_relation(einstein, usa, "BECAME CITIZEN OF", {"year": "1940"}),
            self._create_relation(einstein, princeton, "BECAME PROFESSOR AT", {"year": "1933"}),
            self._create_relation(hitler, germany, "ROSE TO POWER IN", {"year": "1933"})
        ]
        
        entities = [einstein, ulm, germany, usa, swiss_polytechnic, special_relativity, 
                   general_relativity, photoelectric_effect, brownian_motion, mass_energy, 
                   nobel_prize, princeton, maric, lowenthal, hitler]
        
        return entities, relations
    
    def _generate_descartes_data(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Generate mock data for Descartes"""
        # Create entities
        descartes = self._create_entity("René Descartes", "PERSON", {"birth_year": "1596", "death_year": "1650"})
        la_haye = self._create_entity("La Haye en Touraine", "LOCATION", {"country": "France"})
        france = self._create_entity("France", "LOCATION", {})
        netherlands = self._create_entity("Netherlands", "LOCATION", {})
        sweden = self._create_entity("Sweden", "LOCATION", {})
        jesuit_college = self._create_entity("Jesuit College of La Flèche", "ORGANIZATION", {})
        university_of_poitiers = self._create_entity("University of Poitiers", "ORGANIZATION", {})
        cogito = self._create_entity("Cogito, ergo sum", "WORK_OF_ART", {"translation": "I think, therefore I am"})
        discourse_on_method = self._create_entity("Discourse on the Method", "WORK_OF_ART", {"year": "1637"})
        meditations = self._create_entity("Meditations on First Philosophy", "WORK_OF_ART", {"year": "1641"})
        cartesian_dualism = self._create_entity("Cartesian Dualism", "WORK_OF_ART", {})
        analytical_geometry = self._create_entity("Analytical Geometry", "WORK_OF_ART", {})
        queen_christina = self._create_entity("Queen Christina", "PERSON", {"country": "Sweden"})
        
        # Create relations
        relations = [
            self._create_relation(descartes, la_haye, "BORN IN", {"year": "1596"}),
            self._create_relation(la_haye, france, "LOCATED IN", {}),
            self._create_relation(descartes, jesuit_college, "EDUCATED AT", {"period": "1607-1614"}),
            self._create_relation(descartes, university_of_poitiers, "GRADUATED FROM", {"year": "1616", "degree": "Law"}),
            self._create_relation(descartes, netherlands, "LIVED IN", {"period": "1628-1649"}),
            self._create_relation(descartes, cogito, "FORMULATED", {}),
            self._create_relation(descartes, discourse_on_method, "WROTE", {"year": "1637"}),
            self._create_relation(descartes, meditations, "WROTE", {"year": "1641"}),
            self._create_relation(descartes, cartesian_dualism, "DEVELOPED", {}),
            self._create_relation(descartes, analytical_geometry, "INVENTED", {}),
            self._create_relation(queen_christina, descartes, "INVITED TO COURT", {"year": "1649"}),
            self._create_relation(descartes, sweden, "DIED IN", {"year": "1650", "cause": "Pneumonia"})
        ]
        
        entities = [descartes, la_haye, france, netherlands, sweden, jesuit_college, 
                   university_of_poitiers, cogito, discourse_on_method, meditations, 
                   cartesian_dualism, analytical_geometry, queen_christina]
        
        return entities, relations
    
    def _create_entity(self, name: str, entity_type: str, properties: Dict[str, str]) -> Dict[str, Any]:
        """Create an entity with a unique ID"""
        return {
            "id": str(uuid.uuid4()),
            "name": name,
            "type": entity_type,
            "properties": properties
        }
    
    def _create_relation(self, from_entity: Dict[str, Any], to_entity: Dict[str, Any], 
                         relation_type: str, properties: Dict[str, str]) -> Dict[str, Any]:
        """Create a relation between two entities"""
        return {
            "from_entity": from_entity,
            "to_entity": to_entity,
            "relation": relation_type,
            "confidence": 1.0,
            "properties": properties
        } 