"""
AURA Skill Library - Skill Executor

Retrieves and applies relevant skills to user requests.
Tracks execution for learning feedback loop.
"""

import logging
import time
from typing import List, Optional, Dict, Any, Tuple, Callable

from .skill import Skill, SkillExample
from .skill_store import SkillStore
from .skill_learner import SkillLearner

logger = logging.getLogger(__name__)


class SkillExecutor:
    """
    Retrieves and applies relevant skills to user requests.
    Tracks execution for learning feedback loop.
    """

    def __init__(
        self,
        store: SkillStore,
        learner: SkillLearner,
        llm_func: Optional[Callable[[str], str]] = None
    ):
        """
        Initialize skill executor.

        Args:
            store: SkillStore instance
            learner: SkillLearner instance
            llm_func: Function that takes a prompt and returns LLM response
        """
        self.store = store
        self.learner = learner
        self.llm_func = llm_func

        # Statistics
        self._stats = {
            "executions": 0,
            "with_skill": 0,
            "without_skill": 0,
            "successful": 0,
            "failed": 0
        }

    def find_applicable_skills(
        self,
        user_input: str,
        context: Optional[Dict] = None,
        max_skills: int = 3
    ) -> List[Tuple[Skill, float]]:
        """
        Find skills applicable to the user's request.

        Args:
            user_input: User's request
            context: Optional context
            max_skills: Maximum skills to return

        Returns:
            List of (skill, relevance_score) tuples
        """
        # First, check trigger patterns (fast, exact)
        trigger_matches = self.store.search_by_trigger(user_input, threshold=0.75)

        # Then, semantic search (slower, fuzzy)
        semantic_matches = self.store.search(user_input, limit=max_skills * 2)

        # Combine and deduplicate
        seen = set()
        results = []

        for skill_id, score in trigger_matches[:max_skills]:
            if skill_id not in seen:
                skill = self.store.load(skill_id)
                if skill:
                    results.append((skill, 1.0))  # Trigger match = high confidence
                    seen.add(skill_id)

        for skill_id, score in semantic_matches:
            if skill_id not in seen and len(results) < max_skills:
                skill = self.store.load(skill_id)
                if skill:
                    results.append((skill, score))
                    seen.add(skill_id)

        return results

    def format_skill_context(
        self,
        skills: List[Tuple[Skill, float]],
        include_examples: bool = True
    ) -> str:
        """
        Format skills into context for LLM injection.

        Args:
            skills: List of (skill, score) tuples
            include_examples: Whether to include examples

        Returns:
            Formatted context string
        """
        if not skills:
            return ""

        context_parts = ["## Available Skills\n"]

        for skill, score in skills:
            context_parts.append(f"### {skill.name} (relevance: {score:.0%})\n")
            context_parts.append(f"{skill.description}\n\n")
            context_parts.append("**Procedure:**\n")
            context_parts.append(f"{skill.procedure}\n")

            if include_examples and skill.examples:
                successful = [e for e in skill.examples if e.success][:2]
                if successful:
                    context_parts.append("\n**Examples:**\n")
                    for ex in successful:
                        context_parts.append(f"- Input: {ex.input_context[:100]}...\n")
                        context_parts.append(f"  Output: {ex.output[:100]}...\n")

            context_parts.append("\n---\n")

        return "\n".join(context_parts)

    def execute_with_skill(
        self,
        user_input: str,
        skill: Skill,
        context: Optional[Dict] = None
    ) -> Tuple[str, bool, float]:
        """
        Execute a request using a specific skill.

        Args:
            user_input: User's request
            skill: Skill to apply
            context: Optional context

        Returns:
            Tuple of (output, success, execution_time_ms)
        """
        if not self.llm_func:
            return "Error: No LLM function configured", False, 0.0

        start_time = time.time()

        # Build prompt with skill guidance
        prompt = f"""You have access to this skill:

{skill.name}: {skill.description}

Procedure to follow:
{skill.procedure}

User request: {user_input}

Apply the skill's procedure to handle this request. Follow the steps carefully."""

        try:
            output = self.llm_func(prompt)
            execution_time = (time.time() - start_time) * 1000

            # We assume success for now - feedback loop will correct
            return output, True, execution_time

        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            logger.error(f"Error executing skill: {e}")
            return f"Error executing skill: {e}", False, execution_time

    def execute_and_learn(
        self,
        user_input: str,
        context: Optional[Dict] = None,
        feedback_callback: Optional[Callable[[Dict], Dict]] = None
    ) -> Dict[str, Any]:
        """
        Full execution pipeline with learning feedback.

        1. Find applicable skills
        2. Execute with best skill (or without if none found)
        3. Collect feedback
        4. Update skill/learn new skill

        Args:
            user_input: User's request
            context: Optional context
            feedback_callback: Optional function to collect feedback

        Returns:
            Execution result with metadata
        """
        self._stats["executions"] += 1

        result = {
            "input": user_input,
            "output": None,
            "skill_used": None,
            "execution_time_ms": 0,
            "success": False,
            "learned_skill": None
        }

        # Find skills
        applicable = self.find_applicable_skills(user_input, context)

        if applicable:
            best_skill, score = applicable[0]
            result["skill_used"] = {
                "id": best_skill.id,
                "name": best_skill.name,
                "relevance": score
            }

            output, success, exec_time = self.execute_with_skill(
                user_input, best_skill, context
            )

            result["output"] = output
            result["execution_time_ms"] = exec_time
            result["success"] = success

            self._stats["with_skill"] += 1

        else:
            # No skill found - execute raw and potentially learn
            self._stats["without_skill"] += 1

            if self.llm_func:
                start_time = time.time()
                try:
                    result["output"] = self.llm_func(user_input)
                    result["success"] = True
                except Exception as e:
                    result["output"] = f"Error: {e}"
                    result["success"] = False

                result["execution_time_ms"] = (time.time() - start_time) * 1000
            else:
                result["output"] = "No LLM function configured and no applicable skill found"
                result["success"] = False

        # Update statistics
        if result["success"]:
            self._stats["successful"] += 1
        else:
            self._stats["failed"] += 1

        # Collect feedback if callback provided
        feedback = None
        if feedback_callback:
            try:
                feedback = feedback_callback(result)
                if isinstance(feedback, dict):
                    result["success"] = feedback.get("success", result["success"])
            except Exception as e:
                logger.warning(f"Feedback callback error: {e}")

        # Record for learning
        if result["output"]:
            learned_id = self.learner.record_interaction(
                user_input=user_input,
                aura_output=result["output"],
                success=result["success"],
                context=context,
                feedback=feedback.get("comment") if isinstance(feedback, dict) else None
            )

            if learned_id and learned_id.startswith("learned_"):
                result["learned_skill"] = learned_id

        return result

    def get_skill_for_context(
        self,
        context: Dict[str, Any]
    ) -> Optional[Tuple[Skill, float]]:
        """
        Given context, find the most relevant skill.

        Args:
            context: Context including recent inputs, goals, etc.

        Returns:
            (Skill, relevance_score) or None
        """
        query_parts = []

        if context.get("current_goal"):
            query_parts.append(context["current_goal"])

        if context.get("recent_inputs"):
            recent = context["recent_inputs"]
            if isinstance(recent, list):
                query_parts.extend(recent[-2:])
            else:
                query_parts.append(str(recent))

        if not query_parts:
            return None

        query = " ".join(query_parts)

        # Search with higher bar since we're proactively suggesting
        matches = self.store.search(query, limit=1, min_success_rate=0.7)

        if matches:
            skill = self.store.load(matches[0][0])
            if skill:
                return (skill, matches[0][1])

        return None

    def get_statistics(self) -> Dict[str, Any]:
        """Get executor statistics."""
        return {
            **self._stats,
            "skill_usage_rate": self._stats["with_skill"] / max(self._stats["executions"], 1),
            "success_rate": self._stats["successful"] / max(self._stats["executions"], 1)
        }
