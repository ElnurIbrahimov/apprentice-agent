"""Skill Library mixin — skill CRUD, search, context, learning.

Extracted from agent.py (2026-03-23) to reduce class size.
All methods assume self has: skill_library, skill_bridge.
"""

import logging

logger = logging.getLogger(__name__)


class SkillManagerMixin:
    """Mixin providing Skill Library methods for ApprenticeAgent."""

    def get_skill_library_stats(self) -> dict:
        """Get Skill Library statistics.

        Returns:
            dict with skill library stats, or empty dict if not available
        """
        if self.skill_library is None:
            return {"available": False, "reason": "Skill Library not initialized"}

        try:
            return self.skill_library.get_stats()
        except (AttributeError, KeyError, TypeError, OSError) as e:
            return {"available": False, "error": str(e)}

    def skill_search(self, query: str, limit: int = 5, category: str | None = None) -> list:
        """Search for relevant skills.

        Args:
            query: Search query
            limit: Maximum results
            category: Optional category filter

        Returns:
            list of (skill_id, score) tuples
        """
        if self.skill_library is None:
            return []

        try:
            return self.skill_library.search(query, limit=limit, category=category)
        except (AttributeError, KeyError, TypeError, ValueError, OSError) as e:
            logger.error(f"Skill search error: {e}")
            return []

    def skill_get(self, skill_id: str) -> dict:
        """Get a skill by ID.

        Args:
            skill_id: ID of the skill to retrieve

        Returns:
            dict with skill data, or error dict
        """
        if self.skill_library is None:
            return {"error": "Skill Library not available"}

        try:
            skill = self.skill_library.get_skill(skill_id)
            if skill:
                return skill.to_dict()
            return {"error": f"Skill not found: {skill_id}"}
        except (AttributeError, KeyError, TypeError, OSError) as e:
            return {"error": str(e)}

    def skill_create(
        self,
        name: str,
        description: str,
        category: str,
        trigger_patterns: list,
        procedure: str,
        tags: list | None = None
    ) -> dict:
        """Create a new skill.

        Args:
            name: Skill name (2-4 words)
            description: What the skill does
            category: coding, writing, research, automation, analysis, communication, learning, custom
            trigger_patterns: Phrases that trigger this skill
            procedure: Step-by-step procedure
            tags: Optional tags

        Returns:
            dict with created skill ID
        """
        if self.skill_library is None:
            return {"success": False, "error": "Skill Library not available"}

        try:
            skill_id = self.skill_library.create_skill(
                name=name,
                description=description,
                category=category,
                trigger_patterns=trigger_patterns,
                procedure=procedure,
                tags=tags
            )
            return {"success": True, "skill_id": skill_id}
        except (AttributeError, KeyError, TypeError, ValueError, OSError) as e:
            return {"success": False, "error": str(e)}

    def skill_record_use(
        self,
        skill_id: str,
        input_context: str,
        output: str,
        success: bool,
        feedback: str | None = None
    ) -> dict:
        """Record usage of a skill for learning.

        Args:
            skill_id: ID of the skill used
            input_context: What triggered the skill
            output: What the skill produced
            success: Whether it worked
            feedback: Optional user feedback

        Returns:
            dict with success status
        """
        if self.skill_library is None:
            return {"success": False, "error": "Skill Library not available"}

        try:
            result = self.skill_library.record_use(
                skill_id=skill_id,
                input_context=input_context,
                output=output,
                success=success,
                feedback=feedback
            )
            return {"success": result}
        except (AttributeError, KeyError, TypeError, ValueError, OSError) as e:
            return {"success": False, "error": str(e)}

    def skill_find_applicable(self, user_input: str, max_skills: int = 3) -> list:
        """Find skills applicable to a user request.

        Args:
            user_input: User's request
            max_skills: Maximum skills to return

        Returns:
            list of (skill, score) tuples
        """
        if self.skill_library is None:
            return []

        try:
            return self.skill_library.find_applicable(user_input, max_skills=max_skills)
        except (AttributeError, KeyError, TypeError, ValueError, OSError) as e:
            logger.error(f"Skill find error: {e}")
            return []

    def skill_get_context(self, user_input: str) -> str:
        """Get skill context for LLM prompting.

        Args:
            user_input: User's request

        Returns:
            Formatted context string for LLM injection
        """
        if self.skill_library is None:
            return ""

        try:
            return self.skill_library.get_skill_context(user_input)
        except (AttributeError, KeyError, TypeError, ValueError, OSError) as e:
            logger.error(f"Skill context error: {e}")
            return ""

    def skill_record_interaction(
        self,
        user_input: str,
        output: str,
        success: bool,
        context: dict | None = None,
        feedback: str | None = None
    ) -> str:
        """Record an interaction for potential skill learning.

        Args:
            user_input: What the user asked
            output: What was produced
            success: Whether successful
            context: Optional context
            feedback: Optional feedback

        Returns:
            skill_id if a skill was learned/updated, None otherwise
        """
        if self.skill_library is None:
            return None

        try:
            return self.skill_library.record_interaction(
                user_input=user_input,
                output=output,
                success=success,
                context=context,
                feedback=feedback
            )
        except (AttributeError, KeyError, TypeError, ValueError, OSError) as e:
            logger.error(f"Skill record error: {e}")
            return None

    def skill_list_summaries(self) -> list:
        """Return lightweight summaries of all skills (name + description + category).

        Used for progressive skill loading: inject these into the system prompt
        so the LLM knows what skills exist without loading full procedures.

        Returns:
            list of dicts with keys: id, name, description, category
        """
        if self.skill_library is None:
            return []

        try:
            all_skills = self.skill_library.store.list_all(sort_by="name")
            return [
                {
                    "id": s["id"],
                    "name": s["name"],
                    "description": s.get("description", ""),
                    "category": s.get("category", "custom"),
                }
                for s in all_skills
            ]
        except (AttributeError, KeyError, TypeError, OSError) as e:
            logger.error(f"Skill list summaries error: {e}")
            return []

    def skill_list(self, category: str | None = None, sort_by: str = "success_rate") -> list:
        """List all skills.

        Args:
            category: Optional category filter
            sort_by: Sort order (success_rate, uses, name, updated)

        Returns:
            list of skill info dicts
        """
        if self.skill_library is None:
            return []

        try:
            return self.skill_library.list_skills(category=category, sort_by=sort_by)
        except (AttributeError, KeyError, TypeError, ValueError, OSError) as e:
            logger.error(f"Skill list error: {e}")
            return []

    def skill_improve(self, skill_id: str, apply: bool = False) -> dict:
        """Analyze and optionally improve a skill.

        Args:
            skill_id: Skill to improve
            apply: Whether to apply improvements

        Returns:
            Improvement suggestions
        """
        if self.skill_library is None:
            return {"error": "Skill Library not available"}

        try:
            return self.skill_library.improve_skill(skill_id, apply=apply)
        except (AttributeError, KeyError, TypeError, ValueError, OSError) as e:
            return {"error": str(e)}
