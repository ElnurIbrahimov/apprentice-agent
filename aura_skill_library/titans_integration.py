"""
AURA Skill Library - Titans Memory Integration

Bridges the Skill Library with Titans Memory for:
- Automatic skill suggestion based on memory context
- Skill-guided memory formation
- Coordinated learning loops
"""

import logging
from typing import List, Optional, Dict, Any, Tuple, Callable

from .skill import Skill, SkillExample
from .skill_store import SkillStore
from .skill_learner import SkillLearner
from .skill_executor import SkillExecutor

logger = logging.getLogger(__name__)


class TitansSkillBridge:
    """
    Bridges Titans Memory with the Skill Library.
    Enables context-aware skill suggestion and coordinated learning.
    """

    def __init__(
        self,
        store: SkillStore,
        learner: SkillLearner,
        executor: SkillExecutor,
        titans_memory: Optional[Any] = None,
        episodic_memory: Optional[Any] = None,
        kg_brain: Optional[Any] = None
    ):
        """
        Initialize Titans-Skill bridge.

        Args:
            store: SkillStore instance
            learner: SkillLearner instance
            executor: SkillExecutor instance
            titans_memory: Optional TitansMemory instance
            episodic_memory: Optional EpisodicMemory instance
            kg_brain: Optional KGBrain instance
        """
        self.store = store
        self.learner = learner
        self.executor = executor
        self.titans_memory = titans_memory
        self.episodic_memory = episodic_memory
        self.kg_brain = kg_brain

        # Statistics
        self._stats = {
            "suggestions_made": 0,
            "skills_applied": 0,
            "memory_integrations": 0
        }

    def suggest_skill_from_context(
        self,
        current_input: str,
        memory_context: Optional[Dict] = None
    ) -> Optional[Tuple[Skill, float, str]]:
        """
        Suggest a skill based on current input and memory context.

        Args:
            current_input: Current user input
            memory_context: Optional context from Titans Memory

        Returns:
            Tuple of (skill, confidence, reason) or None
        """
        # Build rich context from memory systems
        context_parts = [current_input]

        if memory_context:
            # Add recent context from Titans Memory
            if memory_context.get("recent_interactions"):
                recent = memory_context["recent_interactions"][-3:]
                context_parts.extend([str(r) for r in recent])

            # Add current goal/task context
            if memory_context.get("current_goal"):
                context_parts.append(f"Goal: {memory_context['current_goal']}")

        # Get context from episodic memory if available
        if self.episodic_memory:
            try:
                # Recall similar past episodes
                similar_episodes = self._recall_similar_episodes(current_input)
                if similar_episodes:
                    for ep in similar_episodes[:2]:
                        if ep.get("content"):
                            context_parts.append(f"Past: {ep['content'][:100]}")
            except Exception as e:
                logger.debug(f"Could not get episodic context: {e}")

        # Get context from KG brain if available
        if self.kg_brain:
            try:
                # Get related entities
                related = self._get_related_knowledge(current_input)
                if related:
                    context_parts.append(f"Knowledge: {related}")
            except Exception as e:
                logger.debug(f"Could not get KG context: {e}")

        # Search for applicable skills with enriched context
        query = " ".join(context_parts)
        results = self.executor.find_applicable_skills(query, max_skills=3)

        if not results:
            return None

        best_skill, score = results[0]

        # Determine reason for suggestion
        reason = self._determine_suggestion_reason(
            current_input, best_skill, score, memory_context
        )

        self._stats["suggestions_made"] += 1

        return (best_skill, score, reason)

    def _recall_similar_episodes(self, query: str, limit: int = 3) -> List[Dict]:
        """Recall similar episodes from episodic memory."""
        if not self.episodic_memory:
            return []

        try:
            # Use episodic memory's recall method
            if hasattr(self.episodic_memory, 'recall'):
                return self.episodic_memory.recall(query, limit=limit)
            elif hasattr(self.episodic_memory, 'search'):
                return self.episodic_memory.search(query, limit=limit)
        except Exception as e:
            logger.debug(f"Episodic recall failed: {e}")

        return []

    def _get_related_knowledge(self, query: str) -> Optional[str]:
        """Get related knowledge from KG brain."""
        if not self.kg_brain:
            return None

        try:
            # Search for related entities
            if hasattr(self.kg_brain, 'search_entities'):
                entities = self.kg_brain.search_entities(query, limit=3)
                if entities:
                    return ", ".join([e.get("name", str(e)) for e in entities])
            elif hasattr(self.kg_brain, 'query'):
                result = self.kg_brain.query(query)
                if result:
                    return str(result)[:200]
        except Exception as e:
            logger.debug(f"KG query failed: {e}")

        return None

    def _determine_suggestion_reason(
        self,
        input_text: str,
        skill: Skill,
        score: float,
        context: Optional[Dict]
    ) -> str:
        """Determine why a skill is being suggested."""
        reasons = []

        # Check trigger pattern match
        input_lower = input_text.lower()
        for pattern in skill.trigger_patterns:
            if pattern.lower() in input_lower:
                reasons.append(f"matches pattern '{pattern}'")
                break

        # Check semantic similarity
        if score >= 0.9:
            reasons.append("high semantic match")
        elif score >= 0.7:
            reasons.append("good semantic match")

        # Check success rate
        if skill.metadata.success_rate >= 0.9:
            reasons.append(f"proven skill ({skill.metadata.success_rate:.0%} success)")
        elif skill.metadata.total_uses >= 10:
            reasons.append(f"well-tested ({skill.metadata.total_uses} uses)")

        # Check context alignment
        if context and context.get("current_goal"):
            if any(tag in context["current_goal"].lower() for tag in skill.metadata.tags):
                reasons.append("aligns with current goal")

        if not reasons:
            reasons.append("potential match")

        return "; ".join(reasons)

    def apply_skill_with_memory(
        self,
        user_input: str,
        skill: Skill,
        context: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Apply a skill and integrate with memory systems.

        Args:
            user_input: User's request
            skill: Skill to apply
            context: Optional context

        Returns:
            Execution result with memory integration info
        """
        # Execute the skill
        output, success, exec_time = self.executor.execute_with_skill(
            user_input, skill, context
        )

        result = {
            "output": output,
            "success": success,
            "execution_time_ms": exec_time,
            "skill_used": skill.name,
            "memory_updated": False
        }

        # Record to episodic memory
        if self.episodic_memory and success:
            try:
                self._record_to_episodic(user_input, output, skill)
                result["memory_updated"] = True
            except Exception as e:
                logger.debug(f"Could not record to episodic: {e}")

        # Update KG brain with skill usage
        if self.kg_brain and success:
            try:
                self._update_knowledge_graph(user_input, skill)
            except Exception as e:
                logger.debug(f"Could not update KG: {e}")

        self._stats["skills_applied"] += 1
        self._stats["memory_integrations"] += 1 if result["memory_updated"] else 0

        return result

    def _record_to_episodic(
        self,
        input_text: str,
        output: str,
        skill: Skill
    ):
        """Record skill usage to episodic memory."""
        if not self.episodic_memory:
            return

        try:
            # Create an episode for the skill usage
            if hasattr(self.episodic_memory, 'record'):
                self.episodic_memory.record(
                    content=f"Used skill '{skill.name}': {input_text[:100]}",
                    episode_type="skill_usage",
                    metadata={
                        "skill_id": skill.id,
                        "skill_name": skill.name,
                        "input": input_text[:200],
                        "output": output[:200]
                    }
                )
            elif hasattr(self.episodic_memory, 'store'):
                self.episodic_memory.store({
                    "type": "skill_usage",
                    "skill": skill.name,
                    "content": f"{input_text[:100]} -> {output[:100]}"
                })
        except Exception as e:
            logger.debug(f"Episodic record failed: {e}")

    def _update_knowledge_graph(self, input_text: str, skill: Skill):
        """Update knowledge graph with skill usage."""
        if not self.kg_brain:
            return

        try:
            # Add skill entity if not exists
            if hasattr(self.kg_brain, 'add_entity'):
                self.kg_brain.add_entity(
                    name=skill.name,
                    entity_type="skill",
                    properties={
                        "category": skill.category.value,
                        "uses": skill.metadata.total_uses,
                        "success_rate": skill.metadata.success_rate
                    }
                )
            elif hasattr(self.kg_brain, 'store'):
                self.kg_brain.store({
                    "type": "skill",
                    "name": skill.name,
                    "category": skill.category.value
                })
        except Exception as e:
            logger.debug(f"KG update failed: {e}")

    def learn_from_memory(
        self,
        lookback_episodes: int = 50
    ) -> List[str]:
        """
        Analyze recent memory episodes to learn new skills.

        Args:
            lookback_episodes: Number of recent episodes to analyze

        Returns:
            List of newly learned skill IDs
        """
        learned_skills = []

        if not self.episodic_memory:
            return learned_skills

        try:
            # Get recent successful interactions from episodic memory
            recent = self._get_recent_successes(lookback_episodes)

            for interaction in recent:
                # Record to learner for potential skill synthesis
                skill_id = self.learner.record_interaction(
                    user_input=interaction.get("input", ""),
                    aura_output=interaction.get("output", ""),
                    success=interaction.get("success", True),
                    context=interaction.get("context"),
                    feedback=interaction.get("feedback")
                )

                if skill_id and skill_id.startswith("learned_"):
                    learned_skills.append(skill_id)

        except Exception as e:
            logger.error(f"Failed to learn from memory: {e}")

        return learned_skills

    def _get_recent_successes(self, limit: int) -> List[Dict]:
        """Get recent successful interactions from memory."""
        if not self.episodic_memory:
            return []

        try:
            if hasattr(self.episodic_memory, 'get_recent'):
                episodes = self.episodic_memory.get_recent(limit=limit)
                return [
                    ep for ep in episodes
                    if ep.get("success", True) and ep.get("input")
                ]
            elif hasattr(self.episodic_memory, 'recall'):
                episodes = self.episodic_memory.recall("successful interaction", limit=limit)
                return episodes
        except Exception as e:
            logger.debug(f"Could not get recent successes: {e}")

        return []

    def get_skill_context_for_input(
        self,
        user_input: str,
        include_examples: bool = True
    ) -> str:
        """
        Get formatted skill context for LLM injection.

        Args:
            user_input: Current user input
            include_examples: Whether to include examples

        Returns:
            Formatted context string for LLM
        """
        # Find applicable skills
        skills = self.executor.find_applicable_skills(user_input, max_skills=3)

        if not skills:
            return ""

        # Format for injection
        return self.executor.format_skill_context(skills, include_examples)

    def sync_with_titans(self, titans_state: Dict) -> Dict[str, Any]:
        """
        Synchronize skill library state with Titans Memory.

        Args:
            titans_state: Current Titans Memory state

        Returns:
            Sync result with statistics
        """
        sync_result = {
            "skills_suggested": 0,
            "patterns_identified": 0,
            "memory_correlations": 0
        }

        # Get current context from Titans
        if titans_state.get("active_context"):
            # Find skills relevant to current context
            context_skills = self.executor.get_skill_for_context(titans_state)
            if context_skills:
                sync_result["skills_suggested"] += 1

        # Check for repeating patterns in memory
        if titans_state.get("pattern_buffer"):
            for pattern, examples in self.learner.pattern_buffer.items():
                if len(examples) >= self.learner.min_examples:
                    sync_result["patterns_identified"] += 1

        return sync_result

    def get_statistics(self) -> Dict[str, Any]:
        """Get bridge statistics."""
        return {
            **self._stats,
            "connected_systems": {
                "titans_memory": self.titans_memory is not None,
                "episodic_memory": self.episodic_memory is not None,
                "kg_brain": self.kg_brain is not None
            }
        }
