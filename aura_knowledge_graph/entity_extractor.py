"""
Entity and relationship extraction using local LLM.

Uses AURA's existing Ollama integration to extract:
- Named entities (people, projects, technologies, etc.)
- Relationships between entities
- Entity descriptions and properties
"""

import json
import logging
import re
from dataclasses import dataclass
from typing import Callable, List, Optional

from .graph_database import Entity, Relationship
from .schema import EntityType, get_schema_prompt

logger = logging.getLogger(__name__)


@dataclass
class ExtractionResult:
    """Results from entity extraction."""
    entities: List[Entity]
    relationships: List[Relationship]
    raw_response: str
    success: bool
    error: Optional[str] = None


class EntityExtractor:
    """
    LLM-based entity and relationship extractor.

    Uses structured prompting to extract entities that conform
    to AURA's knowledge graph schema.
    """

    def __init__(self, llm_func: Callable[[str], str]):
        """
        Initialize extractor with LLM function.

        Args:
            llm_func: Function that takes prompt string, returns response string.
                      This should be AURA's existing LocalLLM.generate() method.
        """
        self.llm = llm_func
        self.schema_prompt = get_schema_prompt()

    def extract(self, text: str, context: Optional[str] = None) -> ExtractionResult:
        """
        Extract entities and relationships from text.

        Args:
            text: The text to extract from (conversation, document, etc.)
            context: Optional context about what this text is (e.g., "user conversation")

        Returns:
            ExtractionResult with entities and relationships
        """
        prompt = self._build_extraction_prompt(text, context)

        try:
            response = self.llm(prompt)
            return self._parse_response(response, text)
        except Exception as e:
            logger.error(f"[EntityExtractor] Extraction failed: {e}")
            return ExtractionResult(
                entities=[],
                relationships=[],
                raw_response="",
                success=False,
                error=str(e)
            )

    def _build_extraction_prompt(self, text: str, context: Optional[str] = None) -> str:
        """Build the extraction prompt for the LLM."""
        context_str = f"\nContext: {context}" if context else ""

        return f"""You are an entity extraction system. Extract entities and relationships from the given text.

{self.schema_prompt}

RULES:
1. Only extract entities that match the allowed types
2. Only create relationships that match the allowed patterns
3. Entity names should be normalized (e.g., "AURA" not "aura" or "Aura system")
4. Include brief descriptions for each entity
5. Return ONLY valid JSON, no other text
6. If no entities are found, return empty arrays

TEXT TO ANALYZE:{context_str}
\"\"\"
{text}
\"\"\"

Return a JSON object with this EXACT structure:
{{
    "entities": [
        {{
            "name": "Entity Name",
            "type": "Person|Project|Technology|Company|Concept|Task|Location|Event|Document|Skill",
            "description": "Brief description of this entity"
        }}
    ],
    "relationships": [
        {{
            "source": "Source Entity Name",
            "relationship": "RELATIONSHIP_TYPE",
            "target": "Target Entity Name",
            "evidence": "Quote from text supporting this relationship"
        }}
    ]
}}

JSON Output:"""

    def _parse_response(self, response: str, original_text: str) -> ExtractionResult:
        """Parse LLM response into entities and relationships."""
        # Try to extract JSON from response
        json_match = re.search(r'\{[\s\S]*\}', response)
        if not json_match:
            logger.warning("[EntityExtractor] No JSON found in response")
            return ExtractionResult(
                entities=[],
                relationships=[],
                raw_response=response,
                success=False,
                error="No JSON found in response"
            )

        try:
            data = json.loads(json_match.group())
        except json.JSONDecodeError as e:
            logger.warning(f"[EntityExtractor] JSON parse error: {e}")
            return ExtractionResult(
                entities=[],
                relationships=[],
                raw_response=response,
                success=False,
                error=f"JSON parse error: {e}"
            )

        # Parse entities
        entities = []
        entity_name_to_id = {}  # Map names to IDs for relationship resolution

        for e_data in data.get("entities", []):
            try:
                entity_type_str = e_data.get("type", "Concept")
                entity_type = EntityType(entity_type_str)
            except ValueError:
                logger.debug(f"[EntityExtractor] Unknown entity type: {entity_type_str}, defaulting to Concept")
                entity_type = EntityType.CONCEPT

            name = e_data.get("name", "").strip()
            if not name:
                continue

            entity = Entity(
                name=name,
                entity_type=entity_type,
                description=e_data.get("description", "")
            )
            entities.append(entity)
            entity_name_to_id[entity.name.lower()] = entity.id

        # Parse relationships
        relationships = []
        for r_data in data.get("relationships", []):
            source_name = r_data.get("source", "").strip().lower()
            target_name = r_data.get("target", "").strip().lower()

            source_id = entity_name_to_id.get(source_name)
            target_id = entity_name_to_id.get(target_name)

            if source_id and target_id:
                relationships.append(Relationship(
                    source_id=source_id,
                    target_id=target_id,
                    relationship_type=r_data.get("relationship", "RELATES_TO"),
                    evidence=r_data.get("evidence", "")[:500]  # Limit evidence length
                ))
            else:
                logger.debug(
                    f"[EntityExtractor] Skipping relationship - missing entity: "
                    f"{source_name} -> {target_name}"
                )

        logger.info(
            f"[EntityExtractor] Extracted {len(entities)} entities, "
            f"{len(relationships)} relationships"
        )

        return ExtractionResult(
            entities=entities,
            relationships=relationships,
            raw_response=response,
            success=True
        )

    def extract_incremental(
        self,
        text: str,
        existing_entities: List[str]
    ) -> ExtractionResult:
        """
        Extract entities with awareness of existing graph.
        Helps with entity resolution (avoiding duplicates).
        """
        existing_list = ", ".join(existing_entities[:50])  # Limit context size

        prompt = f"""You are an entity extraction system. Extract NEW entities and relationships from the given text.

{self.schema_prompt}

EXISTING ENTITIES IN KNOWLEDGE GRAPH:
{existing_list}

RULES:
1. If an entity matches an existing one, use the EXACT same name
2. Only extract genuinely NEW entities not in the existing list
3. Create relationships between new AND existing entities
4. Return ONLY valid JSON
5. If no new entities are found, return empty arrays

TEXT TO ANALYZE:
\"\"\"
{text}
\"\"\"

Return JSON with entities and relationships:"""

        try:
            response = self.llm(prompt)
            return self._parse_response(response, text)
        except Exception as e:
            logger.error(f"[EntityExtractor] Incremental extraction failed: {e}")
            return ExtractionResult(
                entities=[],
                relationships=[],
                raw_response="",
                success=False,
                error=str(e)
            )

    def extract_from_conversation(
        self,
        messages: List[dict],
        existing_entities: List[str] = None
    ) -> ExtractionResult:
        """
        Extract entities from a conversation history.

        Args:
            messages: List of message dicts with 'role' and 'content'
            existing_entities: Optional list of existing entity names
        """
        # Format conversation
        conversation_text = "\n".join([
            f"{msg.get('role', 'user').upper()}: {msg.get('content', '')}"
            for msg in messages[-10:]  # Limit to last 10 messages
        ])

        if existing_entities:
            return self.extract_incremental(conversation_text, existing_entities)
        else:
            return self.extract(conversation_text, context="conversation")
