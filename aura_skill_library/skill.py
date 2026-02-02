"""
AURA Skill Library - Data Models

Defines the core Skill structure and related types for storing
procedural knowledge and reusable workflows.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Any
from enum import Enum
import uuid
import re


class SkillCategory(Enum):
    """Categories for organizing skills."""
    CODING = "coding"
    WRITING = "writing"
    RESEARCH = "research"
    AUTOMATION = "automation"
    ANALYSIS = "analysis"
    COMMUNICATION = "communication"
    LEARNING = "learning"
    CUSTOM = "custom"


@dataclass
class SkillExample:
    """A concrete example of skill application."""
    input_context: str          # What triggered this skill
    input_data: Optional[str]   # Any input data provided
    output: str                 # What the skill produced
    success: bool               # Did it work?
    timestamp: datetime = field(default_factory=datetime.utcnow)
    feedback: Optional[str] = None  # User feedback if any

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "input_context": self.input_context,
            "input_data": self.input_data,
            "output": self.output,
            "success": self.success,
            "timestamp": self.timestamp.isoformat(),
            "feedback": self.feedback
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SkillExample":
        """Create from dictionary."""
        return cls(
            input_context=data["input_context"],
            input_data=data.get("input_data"),
            output=data["output"],
            success=data["success"],
            timestamp=datetime.fromisoformat(data["timestamp"]) if data.get("timestamp") else datetime.utcnow(),
            feedback=data.get("feedback")
        )


@dataclass
class SkillMetadata:
    """Tracking metadata for skill evolution."""
    version: str = "1.0"
    success_count: int = 0
    failure_count: int = 0
    total_uses: int = 0
    avg_execution_time_ms: float = 0.0
    last_used: Optional[datetime] = None
    last_modified: Optional[datetime] = None
    parent_skill_id: Optional[str] = None  # If evolved from another skill
    tags: List[str] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_uses == 0:
            return 0.0
        return self.success_count / self.total_uses

    def record_use(self, success: bool, execution_time_ms: float = 0.0):
        """Record a skill usage."""
        self.total_uses += 1
        if success:
            self.success_count += 1
        else:
            self.failure_count += 1
        self.last_used = datetime.utcnow()

        # Running average for execution time
        if self.avg_execution_time_ms == 0:
            self.avg_execution_time_ms = execution_time_ms
        else:
            self.avg_execution_time_ms = (
                self.avg_execution_time_ms * 0.9 + execution_time_ms * 0.1
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "version": self.version,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "total_uses": self.total_uses,
            "avg_execution_time_ms": self.avg_execution_time_ms,
            "last_used": self.last_used.isoformat() if self.last_used else None,
            "last_modified": self.last_modified.isoformat() if self.last_modified else None,
            "parent_skill_id": self.parent_skill_id,
            "tags": self.tags
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SkillMetadata":
        """Create from dictionary."""
        return cls(
            version=data.get("version", "1.0"),
            success_count=data.get("success_count", 0),
            failure_count=data.get("failure_count", 0),
            total_uses=data.get("total_uses", 0),
            avg_execution_time_ms=data.get("avg_execution_time_ms", 0.0),
            last_used=datetime.fromisoformat(data["last_used"]) if data.get("last_used") else None,
            last_modified=datetime.fromisoformat(data["last_modified"]) if data.get("last_modified") else None,
            parent_skill_id=data.get("parent_skill_id"),
            tags=data.get("tags", [])
        )


@dataclass
class Skill:
    """A reusable procedural pattern."""
    id: str
    name: str
    description: str
    category: SkillCategory
    trigger_patterns: List[str]
    procedure: str
    examples: List[SkillExample] = field(default_factory=list)
    metadata: SkillMetadata = field(default_factory=SkillMetadata)
    related_skills: List[str] = field(default_factory=list)  # skill IDs
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    @classmethod
    def create(
        cls,
        name: str,
        description: str,
        category: SkillCategory,
        trigger_patterns: List[str],
        procedure: str,
        tags: Optional[List[str]] = None
    ) -> "Skill":
        """Factory method to create a new skill."""
        return cls(
            id=f"skill_{uuid.uuid4().hex[:12]}",
            name=name,
            description=description,
            category=category,
            trigger_patterns=trigger_patterns,
            procedure=procedure,
            metadata=SkillMetadata(tags=tags or [])
        )

    def add_example(self, example: SkillExample):
        """Add an example, keeping max 10 most recent."""
        self.examples.append(example)
        if len(self.examples) > 10:
            # Keep most recent, prioritize successful ones
            successful = [e for e in self.examples if e.success]
            failed = [e for e in self.examples if not e.success]
            # Keep 8 successful + 2 failed for learning
            self.examples = successful[-8:] + failed[-2:]
        self.updated_at = datetime.utcnow()

    def to_markdown(self) -> str:
        """Serialize skill to SKILL.md format."""
        # Build YAML frontmatter
        trigger_yaml = "\n".join(f'  - "{p}"' for p in self.trigger_patterns)
        tags_yaml = str(self.metadata.tags)

        frontmatter = f"""---
id: {self.id}
name: {self.name}
version: {self.metadata.version}
category: {self.category.value}
tags: {tags_yaml}
trigger_patterns:
{trigger_yaml}
success_count: {self.metadata.success_count}
failure_count: {self.metadata.failure_count}
total_uses: {self.metadata.total_uses}
last_used: {self.metadata.last_used.isoformat() if self.metadata.last_used else 'null'}
created_at: {self.created_at.isoformat()}
updated_at: {self.updated_at.isoformat()}
---"""

        # Build examples section
        examples_md = ""
        if self.examples:
            examples_md = "\n\n## Examples\n"
            for i, ex in enumerate(self.examples[:5], 1):  # Max 5 in output
                status = "Success" if ex.success else "Failure"
                examples_md += f"\n### Example {i} ({status})\n"
                examples_md += f"**Context:** {ex.input_context}\n"
                if ex.input_data:
                    examples_md += f"**Input:** {ex.input_data}\n"
                examples_md += f"**Output:** {ex.output}\n"

        # Build related skills
        related_md = ""
        if self.related_skills:
            related_md = "\n\n## Related Skills\n\n"
            related_md += "\n".join(f"- {sid}" for sid in self.related_skills)

        return f"""{frontmatter}

# {self.name}

## Description

{self.description}

## Procedure

{self.procedure}
{examples_md}{related_md}
"""

    @classmethod
    def from_markdown(cls, content: str) -> "Skill":
        """Parse a SKILL.md file into a Skill object."""
        import yaml

        def parse_datetime(value):
            """Parse datetime from YAML (may be string or datetime object)."""
            if value is None or value == 'null':
                return None
            if isinstance(value, datetime):
                return value
            if isinstance(value, str):
                if value == 'null':
                    return None
                return datetime.fromisoformat(value)
            return None

        # Split frontmatter and body
        match = re.match(r'^---\n(.*?)\n---\n(.*)$', content, re.DOTALL)
        if not match:
            raise ValueError("Invalid SKILL.md format: missing frontmatter")

        frontmatter = yaml.safe_load(match.group(1))
        body = match.group(2)

        # Extract procedure section
        proc_match = re.search(
            r'## Procedure\n\n(.*?)(?=\n## |\Z)',
            body,
            re.DOTALL
        )
        procedure = proc_match.group(1).strip() if proc_match else ""

        # Extract description
        desc_match = re.search(
            r'## Description\n\n(.*?)(?=\n## |\Z)',
            body,
            re.DOTALL
        )
        description = desc_match.group(1).strip() if desc_match else ""

        metadata = SkillMetadata(
            version=str(frontmatter.get('version', '1.0')),
            success_count=frontmatter.get('success_count', 0),
            failure_count=frontmatter.get('failure_count', 0),
            total_uses=frontmatter.get('total_uses', 0),
            tags=frontmatter.get('tags', []),
            last_used=parse_datetime(frontmatter.get('last_used'))
        )

        # Parse category
        try:
            category = SkillCategory(frontmatter.get('category', 'custom'))
        except ValueError:
            category = SkillCategory.CUSTOM

        return cls(
            id=frontmatter['id'],
            name=frontmatter['name'],
            description=description,
            category=category,
            trigger_patterns=frontmatter.get('trigger_patterns', []),
            procedure=procedure,
            metadata=metadata,
            created_at=parse_datetime(frontmatter.get('created_at')) or datetime.now(),
            updated_at=parse_datetime(frontmatter.get('updated_at')) or datetime.now()
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "category": self.category.value,
            "trigger_patterns": self.trigger_patterns,
            "procedure": self.procedure,
            "examples": [e.to_dict() for e in self.examples],
            "metadata": self.metadata.to_dict(),
            "related_skills": self.related_skills,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Skill":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            name=data["name"],
            description=data["description"],
            category=SkillCategory(data["category"]),
            trigger_patterns=data["trigger_patterns"],
            procedure=data["procedure"],
            examples=[SkillExample.from_dict(e) for e in data.get("examples", [])],
            metadata=SkillMetadata.from_dict(data.get("metadata", {})),
            related_skills=data.get("related_skills", []),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.utcnow(),
            updated_at=datetime.fromisoformat(data["updated_at"]) if data.get("updated_at") else datetime.utcnow()
        )
