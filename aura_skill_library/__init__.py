"""
AURA Skill Library

Procedural knowledge storage and retrieval system for AURA.
Stores successful patterns, workflows, and techniques as reusable skills.

Features:
- SKILL.md format for human-readable skill storage
- Semantic search with sentence-transformers (CPU only, zero VRAM)
- Automatic skill learning from successful interactions
- Self-improvement through failure analysis
- Integration with Titans Memory and other AURA components

Usage:
    from aura_skill_library import SkillLibrary

    # Initialize
    library = SkillLibrary(storage_path="./aura_skills")

    # Search for skills
    skills = library.search("help me write a Python function")

    # Create a skill manually
    skill_id = library.create_skill(
        name="Python Function Writer",
        description="Creates well-documented Python functions",
        category="coding",
        trigger_patterns=["write a python function", "create a function"],
        procedure="1. Understand the requirements\\n2. Design the signature\\n3. Implement with docstring\\n4. Add type hints"
    )

    # Record skill usage for learning
    library.record_use(skill_id, input_ctx, output, success=True)
"""

__version__ = "0.1.0"

from .skill import Skill, SkillCategory, SkillExample, SkillMetadata
from .skill_store import SkillStore
from .skill_learner import SkillLearner
from .skill_executor import SkillExecutor
from .mcp_tools import SkillLibraryTools
from .titans_integration import TitansSkillBridge
from .skill_md_importer import (
    import_skill_md_dir,
    import_skill_md_file,
    import_many,
)

# Lazy-check: don't import sentence_transformers at module load time
EMBEDDINGS_AVAILABLE = True  # assume available; SkillStore will handle gracefully


class SkillLibrary:
    """
    High-level interface to the AURA Skill Library.
    Combines storage, learning, and execution in a simple API.
    """

    def __init__(
        self,
        storage_path: str = "./aura_skills",
        embedding_model: str = "all-MiniLM-L6-v2",
        llm_func=None,
        min_examples_to_learn: int = 3
    ):
        """
        Initialize the Skill Library.

        Args:
            storage_path: Path to skill storage directory
            embedding_model: Sentence transformer model for embeddings
            llm_func: Optional function(prompt) -> response for LLM calls
            min_examples_to_learn: Min examples before auto-learning a skill
        """
        # Initialize components
        self.store = SkillStore(
            storage_path=storage_path,
            embedding_model=embedding_model
        )

        self.learner = SkillLearner(
            store=self.store,
            llm_func=llm_func,
            min_examples_to_learn=min_examples_to_learn
        )

        self.executor = SkillExecutor(
            store=self.store,
            learner=self.learner,
            llm_func=llm_func
        )

        self.tools = SkillLibraryTools(
            store=self.store,
            learner=self.learner,
            executor=self.executor
        )

        self._bridge = None

    def search(self, query: str, limit: int = 5, category: str = None) -> list:
        """
        Search for relevant skills.

        Args:
            query: Search query
            limit: Maximum results
            category: Optional category filter

        Returns:
            List of (skill, score) tuples
        """
        cat_filter = None
        if category:
            try:
                cat_filter = SkillCategory(category)
            except ValueError:
                pass

        return self.store.search(query, limit=limit, category=cat_filter)

    def get_skill(self, skill_id: str) -> Skill:
        """Get a skill by ID."""
        return self.store.load(skill_id)

    def create_skill(
        self,
        name: str,
        description: str,
        category: str,
        trigger_patterns: list,
        procedure: str,
        tags: list = None
    ) -> str:
        """
        Create a new skill.

        Args:
            name: Skill name
            description: What the skill does
            category: Category (coding, writing, research, etc.)
            trigger_patterns: Phrases that trigger this skill
            procedure: Step-by-step procedure
            tags: Optional tags

        Returns:
            Created skill ID
        """
        try:
            cat = SkillCategory(category)
        except ValueError:
            cat = SkillCategory.CUSTOM

        skill = Skill.create(
            name=name,
            description=description,
            category=cat,
            trigger_patterns=trigger_patterns,
            procedure=procedure,
            tags=tags
        )

        return self.store.save(skill)

    def record_use(
        self,
        skill_id: str,
        input_context: str,
        output: str,
        success: bool,
        feedback: str = None
    ) -> bool:
        """
        Record skill usage for learning.

        Args:
            skill_id: Skill that was used
            input_context: What triggered the usage
            output: What was produced
            success: Whether it worked
            feedback: Optional user feedback

        Returns:
            True if recorded successfully
        """
        skill = self.store.load(skill_id)
        if not skill:
            return False

        example = SkillExample(
            input_context=input_context,
            input_data=None,
            output=output,
            success=success,
            feedback=feedback
        )

        skill.add_example(example)
        skill.metadata.record_use(success)
        self.store.save(skill)

        return True

    def record_interaction(
        self,
        user_input: str,
        output: str,
        success: bool,
        context: dict = None,
        feedback: str = None
    ) -> str:
        """
        Record an interaction for potential skill learning.

        Args:
            user_input: What the user asked
            output: What was produced
            success: Whether it was successful
            context: Optional context
            feedback: Optional feedback

        Returns:
            Skill ID if a skill was learned or updated, None otherwise
        """
        return self.learner.record_interaction(
            user_input=user_input,
            aura_output=output,
            success=success,
            context=context,
            feedback=feedback
        )

    def find_applicable(self, user_input: str, max_skills: int = 3) -> list:
        """
        Find skills applicable to a user request.

        Args:
            user_input: User's request
            max_skills: Maximum skills to return

        Returns:
            List of (skill, relevance_score) tuples
        """
        return self.executor.find_applicable_skills(
            user_input=user_input,
            max_skills=max_skills
        )

    def get_skill_context(self, user_input: str) -> str:
        """
        Get formatted skill context for LLM injection.

        Args:
            user_input: User's request

        Returns:
            Formatted context string
        """
        skills = self.find_applicable(user_input)
        return self.executor.format_skill_context(skills)

    def improve_skill(self, skill_id: str, apply: bool = False) -> dict:
        """
        Analyze and optionally improve a skill.

        Args:
            skill_id: Skill to improve
            apply: Whether to apply improvements

        Returns:
            Improvement suggestions
        """
        suggestions = self.learner.suggest_improvements(skill_id)

        if suggestions and apply:
            self.learner.apply_improvement(skill_id, suggestions)

        return suggestions

    def list_skills(self, category: str = None, sort_by: str = "success_rate") -> list:
        """
        List all skills.

        Args:
            category: Optional category filter
            sort_by: Sort order (success_rate, uses, name, updated)

        Returns:
            List of skill info dicts
        """
        cat_filter = None
        if category:
            try:
                cat_filter = SkillCategory(category)
            except ValueError:
                pass

        return self.store.list_all(category=cat_filter, sort_by=sort_by)

    def get_stats(self) -> dict:
        """Get library statistics."""
        return {
            "store": self.store.get_stats(),
            "learner": self.learner.get_statistics(),
            "executor": self.executor.get_statistics(),
            "embeddings_available": EMBEDDINGS_AVAILABLE
        }

    def get_mcp_tools(self) -> list:
        """Get MCP tool definitions."""
        return self.tools.get_tools()

    def handle_mcp_call(self, tool_name: str, arguments: dict) -> dict:
        """Handle an MCP tool call."""
        return self.tools.handle_tool_call(tool_name, arguments)

    def connect_bridge(
        self,
        titans_memory=None,
        episodic_memory=None,
        kg_brain=None
    ) -> TitansSkillBridge:
        """
        Connect to other AURA memory systems.

        Args:
            titans_memory: TitansMemory instance
            episodic_memory: EpisodicMemory instance
            kg_brain: KGBrain instance

        Returns:
            TitansSkillBridge instance
        """
        self._bridge = TitansSkillBridge(
            store=self.store,
            learner=self.learner,
            executor=self.executor,
            titans_memory=titans_memory,
            episodic_memory=episodic_memory,
            kg_brain=kg_brain
        )
        return self._bridge

    @property
    def bridge(self) -> TitansSkillBridge:
        """Get the connected bridge."""
        return self._bridge

    def shutdown(self):
        """Clean shutdown."""
        # Save any pending index updates
        self.store._save_index(self.store.index)


__all__ = [
    # Main interface
    "SkillLibrary",

    # Core classes
    "Skill",
    "SkillCategory",
    "SkillExample",
    "SkillMetadata",

    # Components
    "SkillStore",
    "SkillLearner",
    "SkillExecutor",

    # Integration
    "SkillLibraryTools",
    "TitansSkillBridge",

    # SKILL.md importer
    "import_skill_md_dir",
    "import_skill_md_file",
    "import_many",

    # Constants
    "EMBEDDINGS_AVAILABLE",
    "__version__"
]
