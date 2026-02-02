"""
Define the ontology for AURA's Knowledge Graph.
This constrains what entities and relationships can be extracted.
"""

from enum import Enum
from typing import List, Tuple


class EntityType(Enum):
    """Types of entities that can be stored in the knowledge graph."""
    PERSON = "Person"
    PROJECT = "Project"
    TECHNOLOGY = "Technology"
    COMPANY = "Company"
    CONCEPT = "Concept"
    TASK = "Task"
    LOCATION = "Location"
    EVENT = "Event"
    DOCUMENT = "Document"
    SKILL = "Skill"


# Allowed relationships: (source_type, relationship_name, target_type)
ALLOWED_RELATIONSHIPS: List[Tuple[EntityType, str, EntityType]] = [
    (EntityType.PERSON, "WORKS_ON", EntityType.PROJECT),
    (EntityType.PERSON, "KNOWS", EntityType.PERSON),
    (EntityType.PERSON, "HAS_SKILL", EntityType.SKILL),
    (EntityType.PERSON, "WORKS_AT", EntityType.COMPANY),
    (EntityType.PROJECT, "USES", EntityType.TECHNOLOGY),
    (EntityType.PROJECT, "OWNED_BY", EntityType.COMPANY),
    (EntityType.TASK, "BELONGS_TO", EntityType.PROJECT),
    (EntityType.TASK, "ASSIGNED_TO", EntityType.PERSON),
    (EntityType.CONCEPT, "RELATES_TO", EntityType.CONCEPT),
    (EntityType.TECHNOLOGY, "IMPLEMENTS", EntityType.CONCEPT),
    (EntityType.EVENT, "INVOLVES", EntityType.PERSON),
    (EntityType.EVENT, "LOCATED_AT", EntityType.LOCATION),
    (EntityType.DOCUMENT, "MENTIONS", EntityType.PERSON),
    (EntityType.DOCUMENT, "ABOUT", EntityType.PROJECT),
]


def get_schema_prompt() -> str:
    """Generate schema description for LLM extraction prompt."""
    entity_types = ", ".join([e.value for e in EntityType])
    relationships = "\n".join([
        f"  - ({s.value})-[{r}]->({t.value})"
        for s, r, t in ALLOWED_RELATIONSHIPS
    ])
    return f"""Entity Types: {entity_types}

Allowed Relationships:
{relationships}"""


def validate_relationship(
    source_type: EntityType,
    relationship: str,
    target_type: EntityType
) -> bool:
    """Check if a relationship is valid according to the schema."""
    # Check exact match first
    for s, r, t in ALLOWED_RELATIONSHIPS:
        if s == source_type and r == relationship and t == target_type:
            return True

    # Allow generic RELATES_TO between any concepts
    if relationship == "RELATES_TO":
        return True

    # Allow CO_OCCURS between any entities (created by bridge)
    if relationship == "CO_OCCURS":
        return True

    return False
