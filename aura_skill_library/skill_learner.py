"""
AURA Skill Library - Skill Learner

Extracts new skills from successful interactions and suggests improvements.
"""

import json
import logging
import re
from typing import List, Optional, Dict, Any, Callable
from datetime import datetime, timezone

from .skill import Skill, SkillCategory, SkillExample, SkillMetadata
from .skill_store import SkillStore

logger = logging.getLogger(__name__)


class SkillLearner:
    """
    Learns new skills from successful interactions.
    Integrates with Self-Evolution Engine.
    """

    def __init__(
        self,
        store: SkillStore,
        llm_func: Optional[Callable[[str], str]] = None,
        min_examples_to_learn: int = 3,
        similarity_threshold: float = 0.85
    ):
        """
        Initialize skill learner.

        Args:
            store: SkillStore instance
            llm_func: Function that takes a prompt and returns LLM response
            min_examples_to_learn: Minimum examples before creating a skill
            similarity_threshold: Threshold for grouping similar interactions
        """
        self.store = store
        self.llm_func = llm_func
        self.min_examples = min_examples_to_learn
        self.similarity_threshold = similarity_threshold

        # Buffer for potential skills (patterns seen but not yet skills)
        self.pattern_buffer: Dict[str, List[SkillExample]] = {}

        # Statistics
        self._stats = {
            "interactions_recorded": 0,
            "skills_learned": 0,
            "skills_updated": 0,
            "improvements_applied": 0
        }

    def record_interaction(
        self,
        user_input: str,
        aura_output: str,
        success: bool,
        context: Optional[Dict] = None,
        feedback: Optional[str] = None
    ) -> Optional[str]:
        """
        Record an interaction for potential skill learning.

        Args:
            user_input: What the user asked
            aura_output: What AURA produced
            success: Whether it was successful
            context: Optional context data
            feedback: Optional user feedback

        Returns:
            skill_id if a new skill was created or existing updated, None otherwise
        """
        self._stats["interactions_recorded"] += 1

        example = SkillExample(
            input_context=user_input,
            input_data=json.dumps(context) if context else None,
            output=aura_output,
            success=success,
            feedback=feedback
        )

        # Check if this matches an existing skill
        matching_skills = self.store.search_by_trigger(user_input, threshold=0.8)

        if matching_skills:
            # Update existing skill with new example
            skill = self.store.load(matching_skills[0][0])
            if skill:
                skill.add_example(example)
                skill.metadata.record_use(success)
                self.store.save(skill)
                self._stats["skills_updated"] += 1
                logger.debug(f"Updated skill {skill.name} with new example")
                return skill.id

        # No matching skill - add to pattern buffer
        pattern_key = self._extract_pattern_key(user_input)
        if pattern_key not in self.pattern_buffer:
            self.pattern_buffer[pattern_key] = []

        self.pattern_buffer[pattern_key].append(example)

        # Check if we have enough examples to create a skill
        if len(self.pattern_buffer[pattern_key]) >= self.min_examples:
            successful = [e for e in self.pattern_buffer[pattern_key] if e.success]
            if len(successful) >= self.min_examples - 1:  # Allow 1 failure
                new_skill = self._synthesize_skill(
                    pattern_key,
                    self.pattern_buffer[pattern_key]
                )
                if new_skill:
                    skill_id = self.store.save(new_skill)
                    del self.pattern_buffer[pattern_key]
                    self._stats["skills_learned"] += 1
                    logger.info(f"Learned new skill: {new_skill.name}")
                    return skill_id

        return None

    def _extract_pattern_key(self, text: str) -> str:
        """
        Extract a canonical pattern key from input text.
        Groups similar requests together.
        """
        # Normalize
        text = text.lower().strip()

        # Remove specific values, keep structure
        # e.g., "convert 5 USD to EUR" -> "convert NUM CODE to CODE"
        text = re.sub(r'\b\d+(\.\d+)?\b', 'NUM', text)
        text = re.sub(r'\b[A-Z]{2,4}\b', 'CODE', text.upper())
        text = re.sub(r'"[^"]*"', 'STRING', text)
        text = re.sub(r"'[^']*'", 'STRING', text)

        # Extract key verbs/nouns
        key_words = []
        action_verbs = [
            'create', 'make', 'generate', 'write', 'build', 'convert',
            'find', 'search', 'analyze', 'explain', 'summarize', 'fix',
            'debug', 'translate', 'format', 'parse', 'extract', 'help',
            'show', 'list', 'get', 'set', 'update', 'delete', 'add'
        ]

        words = text.lower().split()
        for word in words:
            word_clean = re.sub(r'[^\w]', '', word)
            if word_clean in action_verbs or len(word_clean) > 4:
                key_words.append(word_clean)

        return " ".join(key_words[:5])  # Max 5 key words

    def _synthesize_skill(
        self,
        pattern_key: str,
        examples: List[SkillExample]
    ) -> Optional[Skill]:
        """
        Use LLM to synthesize a skill from examples.
        """
        if not self.llm_func:
            logger.warning("No LLM function provided, cannot synthesize skill")
            return None

        # Prepare examples for LLM
        examples_text = ""
        for i, ex in enumerate(examples[:5], 1):
            examples_text += f"""
Example {i}:
Input: {ex.input_context[:500]}
Output: {ex.output[:500]}
Success: {ex.success}
"""

        prompt = f"""Analyze these successful interactions and extract a reusable skill.

Pattern: {pattern_key}

{examples_text}

Create a skill definition with:
1. A clear, concise name (2-4 words)
2. A description of what this skill does
3. 3-5 trigger phrases that would activate this skill
4. A step-by-step procedure that generalizes from the examples
5. The best category: coding, writing, research, automation, analysis, communication, learning

Respond in this exact JSON format:
{{
  "name": "Skill Name",
  "description": "What this skill does...",
  "trigger_patterns": ["phrase 1", "phrase 2", "phrase 3"],
  "procedure": "Step 1: ...\\nStep 2: ...\\nStep 3: ...",
  "category": "coding",
  "tags": ["tag1", "tag2"]
}}

Respond ONLY with the JSON, no other text."""

        try:
            response = self.llm_func(prompt)

            # Try to extract JSON from response
            json_match = re.search(r'\{[\s\S]*\}', response)
            if not json_match:
                logger.error("No JSON found in LLM response")
                return None

            skill_data = json.loads(json_match.group())

            # Validate required fields
            required = ["name", "description", "trigger_patterns", "procedure"]
            for field in required:
                if field not in skill_data:
                    logger.error(f"Missing required field: {field}")
                    return None

            # Create skill
            category_str = skill_data.get("category", "custom").lower()
            try:
                category = SkillCategory(category_str)
            except ValueError:
                category = SkillCategory.CUSTOM

            skill = Skill.create(
                name=skill_data["name"],
                description=skill_data["description"],
                category=category,
                trigger_patterns=skill_data["trigger_patterns"],
                procedure=skill_data["procedure"],
                tags=skill_data.get("tags", [])
            )

            # Mark as learned
            skill.id = f"learned_{skill.id}"

            # Add examples
            for ex in examples:
                skill.add_example(ex)

            return skill

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse skill JSON: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to synthesize skill: {e}")
            return None

    def suggest_improvements(self, skill_id: str) -> Optional[Dict]:
        """
        Analyze a skill's performance and suggest improvements.
        For integration with Self-Evolution Engine.

        Args:
            skill_id: Skill to analyze

        Returns:
            Improvement suggestions or None
        """
        skill = self.store.load(skill_id)
        if not skill:
            return None

        # Check if skill needs improvement
        if skill.metadata.success_rate >= 0.9 and skill.metadata.total_uses >= 10:
            return None  # Skill is performing well

        # Gather failure cases
        failures = [ex for ex in skill.examples if not ex.success]
        if not failures:
            return None

        if not self.llm_func:
            return {
                "analysis": "LLM not available for analysis",
                "skill_id": skill_id,
                "success_rate": skill.metadata.success_rate
            }

        failures_text = "\n".join([
            f"Input: {f.input_context[:200]}\nOutput: {f.output[:200]}\nFeedback: {f.feedback or 'None'}"
            for f in failures[:3]
        ])

        prompt = f"""Analyze this skill's failures and suggest improvements.

Skill: {skill.name}
Description: {skill.description}
Current Procedure:
{skill.procedure}

Success Rate: {skill.metadata.success_rate:.1%}
Total Uses: {skill.metadata.total_uses}

Failure Cases:
{failures_text}

Suggest specific improvements to the procedure that would prevent these failures.
Respond in JSON format:
{{
  "analysis": "What's going wrong...",
  "improved_procedure": "Step 1: ...\\nStep 2: ...",
  "new_trigger_patterns": ["additional pattern 1"],
  "confidence": 0.8
}}

Respond ONLY with the JSON, no other text."""

        try:
            response = self.llm_func(prompt)

            json_match = re.search(r'\{[\s\S]*\}', response)
            if not json_match:
                return None

            return json.loads(json_match.group())

        except Exception as e:
            logger.error(f"Failed to analyze skill: {e}")
            return None

    def apply_improvement(self, skill_id: str, improvement: Dict) -> bool:
        """
        Apply suggested improvements to a skill.

        Args:
            skill_id: Skill to improve
            improvement: Improvement suggestions dict

        Returns:
            True if applied successfully
        """
        skill = self.store.load(skill_id)
        if not skill:
            return False

        # Update procedure
        if "improved_procedure" in improvement:
            skill.procedure = improvement["improved_procedure"]

        # Add new trigger patterns
        if "new_trigger_patterns" in improvement:
            for pattern in improvement["new_trigger_patterns"]:
                if pattern not in skill.trigger_patterns:
                    skill.trigger_patterns.append(pattern)

        # Increment version
        try:
            current_version = float(skill.metadata.version)
            skill.metadata.version = f"{current_version + 0.1:.1f}"
        except ValueError:
            skill.metadata.version = "1.1"

        skill.metadata.last_modified = datetime.now(timezone.utc)
        skill.updated_at = datetime.now(timezone.utc)

        self.store.save(skill)
        self._stats["improvements_applied"] += 1

        logger.info(f"Applied improvements to skill: {skill.name}")
        return True

    def get_statistics(self) -> Dict[str, Any]:
        """Get learner statistics."""
        return {
            **self._stats,
            "patterns_in_buffer": len(self.pattern_buffer),
            "buffer_sizes": {k: len(v) for k, v in self.pattern_buffer.items()}
        }

    def clear_buffer(self):
        """Clear the pattern buffer."""
        self.pattern_buffer.clear()
