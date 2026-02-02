"""
AURA Skill Library - MCP Tools

Exposes skill library functionality as MCP tools for agent integration.
"""

import logging
from typing import List, Optional, Dict, Any

from .skill import Skill, SkillCategory, SkillExample
from .skill_store import SkillStore
from .skill_learner import SkillLearner
from .skill_executor import SkillExecutor

logger = logging.getLogger(__name__)


class SkillLibraryTools:
    """
    MCP tool interface for AURA Skill Library.
    Provides 7 tools for skill management and execution.
    """

    def __init__(
        self,
        store: SkillStore,
        learner: SkillLearner,
        executor: SkillExecutor
    ):
        """
        Initialize skill library tools.

        Args:
            store: SkillStore instance
            learner: SkillLearner instance
            executor: SkillExecutor instance
        """
        self.store = store
        self.learner = learner
        self.executor = executor

    def get_tools(self) -> List[Dict[str, Any]]:
        """
        Get MCP tool definitions.

        Returns:
            List of tool definitions in MCP format
        """
        return [
            {
                "name": "skill_search",
                "description": "Search for relevant skills based on a query. Returns skills that match semantically or by trigger patterns.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query describing what you want to do"
                        },
                        "category": {
                            "type": "string",
                            "description": "Optional category filter (coding, writing, research, automation, analysis, communication, learning, custom)",
                            "enum": ["coding", "writing", "research", "automation", "analysis", "communication", "learning", "custom"]
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of results (default: 5)",
                            "default": 5
                        }
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "skill_get",
                "description": "Get detailed information about a specific skill by ID.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "skill_id": {
                            "type": "string",
                            "description": "The skill ID to retrieve"
                        }
                    },
                    "required": ["skill_id"]
                }
            },
            {
                "name": "skill_create",
                "description": "Create a new skill from a successful interaction or workflow.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Short, descriptive name for the skill (2-4 words)"
                        },
                        "description": {
                            "type": "string",
                            "description": "What this skill does and when to use it"
                        },
                        "category": {
                            "type": "string",
                            "description": "Skill category",
                            "enum": ["coding", "writing", "research", "automation", "analysis", "communication", "learning", "custom"]
                        },
                        "trigger_patterns": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Phrases that should trigger this skill"
                        },
                        "procedure": {
                            "type": "string",
                            "description": "Step-by-step procedure to follow"
                        },
                        "tags": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional tags for categorization"
                        }
                    },
                    "required": ["name", "description", "category", "trigger_patterns", "procedure"]
                }
            },
            {
                "name": "skill_record_use",
                "description": "Record the usage of a skill for learning and improvement.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "skill_id": {
                            "type": "string",
                            "description": "The skill ID that was used"
                        },
                        "input_context": {
                            "type": "string",
                            "description": "What triggered the skill usage"
                        },
                        "output": {
                            "type": "string",
                            "description": "What the skill produced"
                        },
                        "success": {
                            "type": "boolean",
                            "description": "Whether the skill application was successful"
                        },
                        "feedback": {
                            "type": "string",
                            "description": "Optional user feedback"
                        }
                    },
                    "required": ["skill_id", "input_context", "output", "success"]
                }
            },
            {
                "name": "skill_list",
                "description": "List all skills, optionally filtered by category.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "category": {
                            "type": "string",
                            "description": "Optional category filter",
                            "enum": ["coding", "writing", "research", "automation", "analysis", "communication", "learning", "custom"]
                        },
                        "sort_by": {
                            "type": "string",
                            "description": "Sort order (success_rate, uses, name, updated)",
                            "enum": ["success_rate", "uses", "name", "updated"],
                            "default": "success_rate"
                        }
                    }
                }
            },
            {
                "name": "skill_improve",
                "description": "Analyze a skill and suggest improvements based on failure cases.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "skill_id": {
                            "type": "string",
                            "description": "The skill ID to analyze and improve"
                        },
                        "apply": {
                            "type": "boolean",
                            "description": "Whether to automatically apply suggested improvements",
                            "default": False
                        }
                    },
                    "required": ["skill_id"]
                }
            },
            {
                "name": "skill_stats",
                "description": "Get statistics about the skill library.",
                "parameters": {
                    "type": "object",
                    "properties": {}
                }
            }
        ]

    def handle_tool_call(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle an MCP tool call.

        Args:
            tool_name: Name of the tool to call
            arguments: Tool arguments

        Returns:
            Tool result
        """
        handlers = {
            "skill_search": self._handle_search,
            "skill_get": self._handle_get,
            "skill_create": self._handle_create,
            "skill_record_use": self._handle_record_use,
            "skill_list": self._handle_list,
            "skill_improve": self._handle_improve,
            "skill_stats": self._handle_stats
        }

        handler = handlers.get(tool_name)
        if not handler:
            return {"error": f"Unknown tool: {tool_name}"}

        try:
            return handler(arguments)
        except Exception as e:
            logger.error(f"Tool {tool_name} failed: {e}")
            return {"error": str(e)}

    def _handle_search(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle skill_search tool."""
        query = args["query"]
        category = args.get("category")
        limit = args.get("limit", 5)

        # Parse category if provided
        cat_filter = None
        if category:
            try:
                cat_filter = SkillCategory(category)
            except ValueError:
                pass

        # Search using executor for combined results
        results = self.executor.find_applicable_skills(
            query,
            max_skills=limit
        )

        # Apply category filter if needed
        if cat_filter:
            results = [(s, score) for s, score in results if s.category == cat_filter]

        return {
            "skills": [
                {
                    "id": skill.id,
                    "name": skill.name,
                    "description": skill.description,
                    "category": skill.category.value,
                    "relevance": score,
                    "success_rate": skill.metadata.success_rate,
                    "total_uses": skill.metadata.total_uses
                }
                for skill, score in results
            ],
            "count": len(results)
        }

    def _handle_get(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle skill_get tool."""
        skill_id = args["skill_id"]
        skill = self.store.load(skill_id)

        if not skill:
            return {"error": f"Skill not found: {skill_id}"}

        return {
            "skill": {
                "id": skill.id,
                "name": skill.name,
                "description": skill.description,
                "category": skill.category.value,
                "trigger_patterns": skill.trigger_patterns,
                "procedure": skill.procedure,
                "metadata": {
                    "version": skill.metadata.version,
                    "success_rate": skill.metadata.success_rate,
                    "total_uses": skill.metadata.total_uses,
                    "tags": skill.metadata.tags
                },
                "examples": [
                    {
                        "input": ex.input_context[:200],
                        "output": ex.output[:200],
                        "success": ex.success
                    }
                    for ex in skill.examples[:3]
                ]
            }
        }

    def _handle_create(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle skill_create tool."""
        try:
            category = SkillCategory(args["category"])
        except ValueError:
            category = SkillCategory.CUSTOM

        skill = Skill.create(
            name=args["name"],
            description=args["description"],
            category=category,
            trigger_patterns=args["trigger_patterns"],
            procedure=args["procedure"],
            tags=args.get("tags", [])
        )

        skill_id = self.store.save(skill)

        return {
            "skill_id": skill_id,
            "name": skill.name,
            "message": f"Created skill: {skill.name}"
        }

    def _handle_record_use(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle skill_record_use tool."""
        skill_id = args["skill_id"]
        skill = self.store.load(skill_id)

        if not skill:
            return {"error": f"Skill not found: {skill_id}"}

        example = SkillExample(
            input_context=args["input_context"],
            input_data=None,
            output=args["output"],
            success=args["success"],
            feedback=args.get("feedback")
        )

        skill.add_example(example)
        skill.metadata.record_use(args["success"])
        self.store.save(skill)

        return {
            "recorded": True,
            "skill_id": skill_id,
            "new_success_rate": skill.metadata.success_rate,
            "total_uses": skill.metadata.total_uses
        }

    def _handle_list(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle skill_list tool."""
        category = args.get("category")
        sort_by = args.get("sort_by", "success_rate")

        cat_filter = None
        if category:
            try:
                cat_filter = SkillCategory(category)
            except ValueError:
                pass

        skills = self.store.list_all(category=cat_filter, sort_by=sort_by)

        return {
            "skills": [
                {
                    "id": s["id"],
                    "name": s["name"],
                    "category": s["category"],
                    "success_rate": s.get("success_rate", 0),
                    "total_uses": s.get("total_uses", 0)
                }
                for s in skills
            ],
            "count": len(skills)
        }

    def _handle_improve(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle skill_improve tool."""
        skill_id = args["skill_id"]
        apply = args.get("apply", False)

        suggestions = self.learner.suggest_improvements(skill_id)

        if not suggestions:
            return {
                "skill_id": skill_id,
                "message": "No improvements needed or skill not found"
            }

        result = {
            "skill_id": skill_id,
            "suggestions": suggestions
        }

        if apply and "improved_procedure" in suggestions:
            success = self.learner.apply_improvement(skill_id, suggestions)
            result["applied"] = success
            if success:
                result["message"] = "Improvements applied successfully"

        return result

    def _handle_stats(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle skill_stats tool."""
        store_stats = self.store.get_stats()
        learner_stats = self.learner.get_statistics()
        executor_stats = self.executor.get_statistics()

        return {
            "library": store_stats,
            "learner": learner_stats,
            "executor": executor_stats
        }
