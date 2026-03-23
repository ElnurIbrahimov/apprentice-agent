"""Load Skill tool — on-demand retrieval of full skill procedures.

Part of the progressive skill loading system: the system prompt contains
skill names and descriptions only. When the LLM needs the full procedure
for a skill, it calls this tool to load it.
"""

import logging

logger = logging.getLogger(__name__)

# Will be set by the agent after initialization
_skill_library = None


def set_skill_library(library):
    """Wire the SkillLibrary instance so execute() can use it."""
    global _skill_library
    _skill_library = library


class LoadSkillTool:
    """Load the full procedure and examples for a skill by name."""

    name = "load_skill"
    description = (
        "Load the full procedure and examples for a skill by name. "
        "Use this when you need the detailed instructions for a specific skill."
    )

    def execute(self, action: str, **kwargs) -> dict:
        """Load a skill's full content.

        Args:
            action: The skill name (or 'load <name>') to retrieve.
            **kwargs: Optional 'skill_name' or 'skill_id' overrides.

        Returns:
            dict with skill procedure, examples, and metadata — or error with suggestions.
        """
        if _skill_library is None:
            return {"success": False, "error": "Skill Library not available"}

        # Extract skill name from action or kwargs
        skill_name = kwargs.get("skill_name") or kwargs.get("name") or ""
        skill_id = kwargs.get("skill_id") or kwargs.get("id") or ""

        if not skill_name and not skill_id:
            # Parse from the action string
            skill_name = self._extract_skill_name(action)

        # --- Try loading by ID first ---
        if skill_id:
            skill = _skill_library.store.load(skill_id)
            if skill:
                return self._format_skill(skill)

        # --- Search by name ---
        if skill_name:
            # Exact match in index
            for sid, info in _skill_library.store.index.items():
                if info["name"].lower() == skill_name.lower():
                    skill = _skill_library.store.load(sid)
                    if skill:
                        return self._format_skill(skill)

            # Fuzzy: search by name as query
            results = _skill_library.store.search(skill_name, limit=5)
            if results:
                # Check if top result is close enough
                top_id, top_score = results[0]
                top_info = _skill_library.store.index.get(top_id, {})
                if top_score > 0.6 or top_info.get("name", "").lower() == skill_name.lower():
                    skill = _skill_library.store.load(top_id)
                    if skill:
                        return self._format_skill(skill)

                # Return suggestions
                suggestions = [
                    {"id": sid, "name": _skill_library.store.index.get(sid, {}).get("name", sid), "score": round(sc, 3)}
                    for sid, sc in results
                ]
                return {
                    "success": False,
                    "error": f"Skill not found: '{skill_name}'",
                    "suggestions": suggestions,
                    "hint": "Try one of the suggested skill names or IDs.",
                }

        return {
            "success": False,
            "error": "No skill name or ID provided. Pass the skill name in the action.",
        }

    def _extract_skill_name(self, action: str) -> str:
        """Extract the skill name from various action formats."""
        if not action:
            return ""
        a = action.strip()
        # Remove common prefixes
        for prefix in ["load skill", "load_skill", "load", "get skill", "get"]:
            if a.lower().startswith(prefix):
                a = a[len(prefix):].strip().strip(":").strip()
                break
        # Remove surrounding quotes
        a = a.strip("\"'")
        return a

    def _format_skill(self, skill) -> dict:
        """Format a loaded skill into a clean response."""
        examples_text = ""
        if skill.examples:
            examples_text = "\n\n## Examples\n"
            for i, ex in enumerate(skill.examples[:5], 1):
                status = "Success" if ex.success else "Failure"
                examples_text += f"\n### Example {i} ({status})\n"
                examples_text += f"Context: {ex.input_context}\n"
                if ex.input_data:
                    examples_text += f"Input: {ex.input_data}\n"
                examples_text += f"Output: {ex.output}\n"

        full_text = f"# {skill.name}\n\n## Procedure\n\n{skill.procedure}{examples_text}"

        return {
            "success": True,
            "skill_id": skill.id,
            "name": skill.name,
            "category": skill.category.value,
            "procedure": skill.procedure,
            "full_text": full_text,
            "metadata": {
                "version": skill.metadata.version,
                "success_rate": skill.metadata.success_rate,
                "total_uses": skill.metadata.total_uses,
                "tags": skill.metadata.tags,
            },
        }
