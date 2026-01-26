"""
Knowledge Extractor for Aura's Knowledge Graph
Extracts entities and relationships from text using LLM.

Author: Aura Development Team
Created: 2025-01-26
"""

import json
import re
from typing import Dict, List, Any, Optional

# Support both relative and absolute imports
try:
    from .knowledge_graph import NODE_TYPES, EDGE_TYPES
except ImportError:
    from knowledge_graph import NODE_TYPES, EDGE_TYPES


class KnowledgeExtractor:
    """
    Extracts structured knowledge from text using LLM.

    Identifies entities, concepts, and relationships from
    conversations and stores them in the knowledge graph.
    """

    def __init__(self, llm=None):
        """
        Initialize the extractor.

        Args:
            llm: LLM interface (OllamaBrain) for extraction
        """
        self.llm = llm

        # Common entity patterns for rule-based extraction
        self._patterns = {
            "project": [
                r"(?:project|repo|repository)\s+['\"]?(\w+)['\"]?",
                r"(?:working on|building|developing)\s+(\w+)",
            ],
            "tool": [
                r"(?:using|use|run|execute)\s+(\w+_?\w*)\s+(?:tool)?",
                r"(?:web_search|code_executor|browser|vision|fluxmind)",
            ],
            "person": [
                r"(?:my\s+)?(dad|mom|brother|sister|friend|boss|colleague)",
                r"(?:@|user\s+)?([A-Z][a-z]+)\s+(?:said|asked|wants)",
            ],
            "file": [
                r"(?:file|document)\s+['\"]?([^\s'\"]+\.\w+)['\"]?",
                r"([^\s]+\.(?:py|js|ts|md|txt|json|yaml|csv|pdf))",
            ],
            "concept": [
                r"(?:about|regarding|concerning)\s+(\w+(?:\s+\w+)?)",
            ],
        }

    def extract(self, text: str, use_llm: bool = True) -> Dict[str, Any]:
        """
        Extract structured knowledge from text.

        Args:
            text: Text to extract from
            use_llm: Whether to use LLM for extraction (slower but better)

        Returns:
            Dict with 'entities' and 'relationships' lists
        """
        if use_llm and self.llm:
            return self._extract_with_llm(text)
        else:
            return self._extract_with_rules(text)

    def _extract_with_llm(self, text: str) -> Dict[str, Any]:
        """Extract using LLM for better understanding."""
        node_types_str = ", ".join(NODE_TYPES.keys())
        edge_types_str = ", ".join(EDGE_TYPES.keys())

        prompt = f"""Extract entities and relationships from this text.

Text: {text}

Valid entity types: {node_types_str}
Valid relationship types: {edge_types_str}

Return ONLY valid JSON (no markdown, no explanation):
{{
    "entities": [
        {{"label": "name", "type": "concept|entity|person|project|tool|event|skill|location|file", "properties": {{}}}}
    ],
    "relationships": [
        {{"source": "entity1", "target": "entity2", "type": "relationship_type", "evidence": "why this relationship"}}
    ]
}}

Rules:
- Only extract CLEAR, FACTUAL information
- Be conservative - don't infer too much
- Use snake_case for relationship types
- Maximum 5 entities and 5 relationships
- If nothing clear to extract, return empty arrays"""

        try:
            response = self.llm.generate(prompt, model="qwen2:1.5b")

            # Try to parse JSON from response
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                result = json.loads(json_match.group())
                return self._validate_extraction(result)
        except Exception as e:
            print(f"[KG Extractor] LLM extraction failed: {e}")

        # Fallback to rules
        return self._extract_with_rules(text)

    def _extract_with_rules(self, text: str) -> Dict[str, Any]:
        """Extract using pattern matching (faster, less accurate)."""
        entities = []
        relationships = []
        seen_labels = set()

        text_lower = text.lower()

        for entity_type, patterns in self._patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, text_lower, re.IGNORECASE)
                for match in matches:
                    label = match.strip() if isinstance(match, str) else match[0].strip()
                    if label and label not in seen_labels and len(label) > 2:
                        entities.append({
                            "label": label,
                            "type": entity_type,
                            "properties": {}
                        })
                        seen_labels.add(label)

        # Simple relationship extraction
        # Pattern: "X uses Y", "X creates Y", etc.
        rel_patterns = [
            (r"(\w+)\s+uses\s+(\w+)", "uses"),
            (r"(\w+)\s+creates?\s+(\w+)", "created_by"),
            (r"(\w+)\s+is\s+(?:a|an)\s+(\w+)", "is_a"),
            (r"(\w+)\s+(?:causes?|leads?\s+to)\s+(\w+)", "causes"),
            (r"(\w+)\s+(?:solves?|fixes?)\s+(\w+)", "solves"),
            (r"(\w+)\s+(?:part\s+of|belongs?\s+to)\s+(\w+)", "part_of"),
        ]

        for pattern, rel_type in rel_patterns:
            matches = re.findall(pattern, text_lower)
            for match in matches:
                if len(match) == 2:
                    source, target = match
                    if source != target:
                        relationships.append({
                            "source": source,
                            "target": target,
                            "type": rel_type,
                            "evidence": f"Pattern match: {pattern}"
                        })

        return {
            "entities": entities[:10],  # Limit
            "relationships": relationships[:10]
        }

    def _validate_extraction(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and clean extracted data."""
        valid_entities = []
        valid_relationships = []

        # Validate entities
        for entity in result.get("entities", []):
            if isinstance(entity, dict):
                label = entity.get("label", "").strip()
                entity_type = entity.get("type", "concept")

                if label and entity_type in NODE_TYPES:
                    valid_entities.append({
                        "label": label,
                        "type": entity_type,
                        "properties": entity.get("properties", {})
                    })

        # Validate relationships
        for rel in result.get("relationships", []):
            if isinstance(rel, dict):
                source = rel.get("source", "").strip()
                target = rel.get("target", "").strip()
                rel_type = rel.get("type", "relates_to")

                if source and target and rel_type in EDGE_TYPES:
                    valid_relationships.append({
                        "source": source,
                        "target": target,
                        "type": rel_type,
                        "evidence": rel.get("evidence", "")
                    })

        return {
            "entities": valid_entities,
            "relationships": valid_relationships
        }

    def extract_from_dialogue(
        self,
        user_msg: str,
        aura_response: str
    ) -> Dict[str, Any]:
        """
        Extract knowledge from a conversation turn.

        Args:
            user_msg: What the user said
            aura_response: Aura's response

        Returns:
            Extracted entities and relationships
        """
        # Combine for context
        combined = f"User: {user_msg}\nAura: {aura_response}"
        return self.extract(combined)

    def extract_topics(self, text: str) -> List[str]:
        """
        Extract main topics from text for context retrieval.

        Returns list of topic strings.
        """
        topics = []

        # Remove common words
        stopwords = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
            'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'could', 'should', 'may', 'might', 'must', 'can',
            'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she',
            'it', 'we', 'they', 'what', 'which', 'who', 'whom', 'how',
            'when', 'where', 'why', 'and', 'or', 'but', 'if', 'then',
            'so', 'as', 'of', 'at', 'by', 'for', 'with', 'about', 'to',
            'from', 'in', 'on', 'into', 'through', 'during', 'before',
            'after', 'above', 'below', 'between', 'under', 'again',
            'further', 'once', 'here', 'there', 'all', 'each', 'few',
            'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not',
            'only', 'own', 'same', 'than', 'too', 'very', 'just', 'now',
            'also', 'me', 'my', 'your', 'please', 'help', 'want', 'need',
            'tell', 'show', 'find', 'search', 'look', 'get', 'make',
        }

        # Extract words
        words = re.findall(r'\b[a-zA-Z][a-zA-Z0-9_]*\b', text.lower())

        for word in words:
            if word not in stopwords and len(word) > 2:
                topics.append(word)

        # Deduplicate while preserving order
        seen = set()
        unique_topics = []
        for topic in topics:
            if topic not in seen:
                seen.add(topic)
                unique_topics.append(topic)

        return unique_topics[:10]  # Limit to top 10


def create_extractor(llm=None) -> KnowledgeExtractor:
    """Create a knowledge extractor instance."""
    return KnowledgeExtractor(llm)


# Export
__all__ = ["KnowledgeExtractor", "create_extractor"]
