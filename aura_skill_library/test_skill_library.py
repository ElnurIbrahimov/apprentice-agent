"""
Tests for AURA Skill Library.
"""

import unittest
import tempfile
import shutil
import sys
from pathlib import Path
from datetime import datetime

# Support both direct execution and package import
try:
    from .skill import Skill, SkillCategory, SkillExample, SkillMetadata
    from .skill_store import SkillStore
    from .skill_learner import SkillLearner
    from .skill_executor import SkillExecutor
    from .mcp_tools import SkillLibraryTools
    from .titans_integration import TitansSkillBridge
    from . import SkillLibrary
except ImportError:
    # Direct execution - add parent to path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from aura_skill_library.skill import Skill, SkillCategory, SkillExample, SkillMetadata
    from aura_skill_library.skill_store import SkillStore
    from aura_skill_library.skill_learner import SkillLearner
    from aura_skill_library.skill_executor import SkillExecutor
    from aura_skill_library.mcp_tools import SkillLibraryTools
    from aura_skill_library.titans_integration import TitansSkillBridge
    from aura_skill_library import SkillLibrary


class TestSkillDataModels(unittest.TestCase):
    """Test Skill data models."""

    def test_skill_example_creation(self):
        """Test SkillExample creation and serialization."""
        example = SkillExample(
            input_context="Write a Python function to add numbers",
            input_data='{"a": 1, "b": 2}',
            output="def add(a, b): return a + b",
            success=True,
            feedback="Works great!"
        )

        self.assertEqual(example.input_context, "Write a Python function to add numbers")
        self.assertTrue(example.success)
        self.assertEqual(example.feedback, "Works great!")

        # Test serialization
        data = example.to_dict()
        restored = SkillExample.from_dict(data)
        self.assertEqual(restored.input_context, example.input_context)
        self.assertEqual(restored.success, example.success)

    def test_skill_metadata(self):
        """Test SkillMetadata tracking."""
        meta = SkillMetadata()

        # Initial state
        self.assertEqual(meta.success_rate, 0.0)
        self.assertEqual(meta.total_uses, 0)

        # Record uses
        meta.record_use(success=True, execution_time_ms=100)
        meta.record_use(success=True, execution_time_ms=120)
        meta.record_use(success=False, execution_time_ms=50)

        self.assertEqual(meta.total_uses, 3)
        self.assertEqual(meta.success_count, 2)
        self.assertEqual(meta.failure_count, 1)
        self.assertAlmostEqual(meta.success_rate, 0.667, places=2)
        self.assertIsNotNone(meta.last_used)

    def test_skill_creation(self):
        """Test Skill creation."""
        skill = Skill.create(
            name="Python Function Writer",
            description="Creates Python functions with proper documentation",
            category=SkillCategory.CODING,
            trigger_patterns=["write a python function", "create a function"],
            procedure="1. Understand requirements\n2. Design signature\n3. Implement",
            tags=["python", "coding"]
        )

        self.assertTrue(skill.id.startswith("skill_"))
        self.assertEqual(skill.name, "Python Function Writer")
        self.assertEqual(skill.category, SkillCategory.CODING)
        self.assertEqual(len(skill.trigger_patterns), 2)

    def test_skill_markdown_serialization(self):
        """Test Skill to/from markdown."""
        skill = Skill.create(
            name="Test Skill",
            description="A test skill for unit testing",
            category=SkillCategory.CODING,
            trigger_patterns=["test pattern"],
            procedure="Step 1: Do something\nStep 2: Do something else",
            tags=["test"]
        )

        # Add an example
        skill.add_example(SkillExample(
            input_context="test input",
            input_data=None,
            output="test output",
            success=True
        ))

        # Serialize to markdown
        md = skill.to_markdown()
        self.assertIn("---", md)
        self.assertIn("# Test Skill", md)
        self.assertIn("## Description", md)
        self.assertIn("## Procedure", md)

        # Deserialize
        restored = Skill.from_markdown(md)
        self.assertEqual(restored.id, skill.id)
        self.assertEqual(restored.name, skill.name)
        self.assertEqual(restored.category, skill.category)

    def test_skill_example_limit(self):
        """Test that skills keep limited examples."""
        skill = Skill.create(
            name="Test",
            description="Test",
            category=SkillCategory.CUSTOM,
            trigger_patterns=["test"],
            procedure="test"
        )

        # Add 15 examples
        for i in range(15):
            skill.add_example(SkillExample(
                input_context=f"input {i}",
                input_data=None,
                output=f"output {i}",
                success=(i % 3 != 0)  # 2/3 success rate
            ))

        # Should be capped at 10
        self.assertLessEqual(len(skill.examples), 10)


class TestSkillStore(unittest.TestCase):
    """Test SkillStore functionality."""

    def setUp(self):
        """Create temporary storage directory."""
        self.test_dir = tempfile.mkdtemp()
        self.store = SkillStore(storage_path=self.test_dir)

    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.test_dir)

    def test_save_and_load(self):
        """Test saving and loading skills."""
        skill = Skill.create(
            name="Test Skill",
            description="For testing",
            category=SkillCategory.CODING,
            trigger_patterns=["test this"],
            procedure="Do the test"
        )

        # Save
        skill_id = self.store.save(skill)
        self.assertEqual(skill_id, skill.id)
        self.assertIn(skill_id, self.store.index)

        # Load
        loaded = self.store.load(skill_id)
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.name, skill.name)
        self.assertEqual(loaded.description, skill.description)

    def test_delete(self):
        """Test deleting skills."""
        skill = Skill.create(
            name="Delete Me",
            description="To be deleted",
            category=SkillCategory.CUSTOM,
            trigger_patterns=["delete"],
            procedure="N/A"
        )

        skill_id = self.store.save(skill)
        self.assertIn(skill_id, self.store.index)

        # Delete
        result = self.store.delete(skill_id)
        self.assertTrue(result)
        self.assertNotIn(skill_id, self.store.index)

        # Try to load deleted
        loaded = self.store.load(skill_id)
        self.assertIsNone(loaded)

    def test_search_by_trigger(self):
        """Test trigger-based search."""
        # Create skills with different triggers
        skill1 = Skill.create(
            name="Python Helper",
            description="Helps with Python",
            category=SkillCategory.CODING,
            trigger_patterns=["write python code", "python function"],
            procedure="..."
        )
        skill2 = Skill.create(
            name="JavaScript Helper",
            description="Helps with JavaScript",
            category=SkillCategory.CODING,
            trigger_patterns=["write javascript", "js function"],
            procedure="..."
        )

        self.store.save(skill1)
        self.store.save(skill2)

        # Search for Python
        results = self.store.search_by_trigger("I want to write python code", threshold=0.5)
        self.assertGreater(len(results), 0)
        # Should find Python skill with exact trigger match
        skill_ids = [r[0] for r in results]
        self.assertIn(skill1.id, skill_ids)

    def test_list_all(self):
        """Test listing all skills."""
        # Create multiple skills
        for i in range(3):
            skill = Skill.create(
                name=f"Skill {i}",
                description=f"Description {i}",
                category=SkillCategory.CODING if i < 2 else SkillCategory.WRITING,
                trigger_patterns=[f"trigger {i}"],
                procedure=f"Procedure {i}"
            )
            self.store.save(skill)

        # List all
        all_skills = self.store.list_all()
        self.assertEqual(len(all_skills), 3)

        # List by category
        coding_skills = self.store.list_all(category=SkillCategory.CODING)
        self.assertEqual(len(coding_skills), 2)

    def test_get_stats(self):
        """Test statistics."""
        skill = Skill.create(
            name="Stats Test",
            description="Test",
            category=SkillCategory.CODING,
            trigger_patterns=["test"],
            procedure="test"
        )
        self.store.save(skill)

        stats = self.store.get_stats()
        self.assertEqual(stats["total_skills"], 1)
        self.assertIn("coding", stats["by_category"])


class TestSkillLearner(unittest.TestCase):
    """Test SkillLearner functionality."""

    def setUp(self):
        """Create temporary storage and learner."""
        self.test_dir = tempfile.mkdtemp()
        self.store = SkillStore(storage_path=self.test_dir)
        self.learner = SkillLearner(
            store=self.store,
            llm_func=None,  # No LLM for testing
            min_examples_to_learn=3
        )

    def tearDown(self):
        """Clean up."""
        shutil.rmtree(self.test_dir)

    def test_pattern_key_extraction(self):
        """Test pattern key extraction."""
        key1 = self.learner._extract_pattern_key("Convert 100 USD to EUR")
        key2 = self.learner._extract_pattern_key("Convert 50 GBP to JPY")

        # Both should produce similar pattern keys
        self.assertIn("convert", key1)
        self.assertIn("convert", key2)

    def test_record_interaction_existing_skill(self):
        """Test recording interaction updates existing skill."""
        # Create a skill first
        skill = Skill.create(
            name="Currency Converter",
            description="Converts currencies",
            category=SkillCategory.AUTOMATION,
            trigger_patterns=["convert currency", "exchange rate"],
            procedure="1. Get rates\n2. Calculate"
        )
        self.store.save(skill)

        # Record interaction that matches
        result = self.learner.record_interaction(
            user_input="convert currency from USD to EUR",
            aura_output="100 USD = 92 EUR",
            success=True
        )

        # Should return the existing skill ID
        self.assertEqual(result, skill.id)

        # Skill should have updated stats
        updated = self.store.load(skill.id)
        self.assertEqual(updated.metadata.total_uses, 1)

    def test_record_interaction_pattern_buffer(self):
        """Test recording interactions builds pattern buffer."""
        # Record similar interactions (not enough to create skill)
        self.learner.record_interaction(
            user_input="Write a Python function for sorting",
            aura_output="def sort(lst): return sorted(lst)",
            success=True
        )
        self.learner.record_interaction(
            user_input="Write a Python function for filtering",
            aura_output="def filter_list(lst, pred): return [x for x in lst if pred(x)]",
            success=True
        )

        # Should be in pattern buffer
        self.assertGreater(len(self.learner.pattern_buffer), 0)
        self.assertEqual(self.learner._stats["interactions_recorded"], 2)

    def test_statistics(self):
        """Test learner statistics."""
        self.learner.record_interaction(
            user_input="test input",
            aura_output="test output",
            success=True
        )

        stats = self.learner.get_statistics()
        self.assertEqual(stats["interactions_recorded"], 1)
        self.assertIn("patterns_in_buffer", stats)


class TestSkillExecutor(unittest.TestCase):
    """Test SkillExecutor functionality."""

    def setUp(self):
        """Create executor with test components."""
        self.test_dir = tempfile.mkdtemp()
        self.store = SkillStore(storage_path=self.test_dir)
        self.learner = SkillLearner(store=self.store)
        self.executor = SkillExecutor(
            store=self.store,
            learner=self.learner,
            llm_func=None
        )

    def tearDown(self):
        """Clean up."""
        shutil.rmtree(self.test_dir)

    def test_find_applicable_skills(self):
        """Test finding applicable skills."""
        # Create some skills
        skill1 = Skill.create(
            name="Python Coder",
            description="Writes Python code",
            category=SkillCategory.CODING,
            trigger_patterns=["write python", "python code"],
            procedure="..."
        )
        skill2 = Skill.create(
            name="Essay Writer",
            description="Writes essays",
            category=SkillCategory.WRITING,
            trigger_patterns=["write essay", "compose text"],
            procedure="..."
        )

        self.store.save(skill1)
        self.store.save(skill2)

        # Find skills for Python request
        results = self.executor.find_applicable_skills(
            "Can you write python code for me?",
            max_skills=3
        )

        self.assertGreater(len(results), 0)
        # Python skill should be found
        skill_names = [s.name for s, score in results]
        self.assertIn("Python Coder", skill_names)

    def test_format_skill_context(self):
        """Test skill context formatting."""
        skill = Skill.create(
            name="Test Skill",
            description="Test description",
            category=SkillCategory.CUSTOM,
            trigger_patterns=["test"],
            procedure="Step 1\nStep 2"
        )

        context = self.executor.format_skill_context([(skill, 0.9)])

        self.assertIn("## Available Skills", context)
        self.assertIn("Test Skill", context)
        self.assertIn("90%", context)
        self.assertIn("Step 1", context)

    def test_statistics(self):
        """Test executor statistics."""
        stats = self.executor.get_statistics()

        self.assertEqual(stats["executions"], 0)
        self.assertIn("skill_usage_rate", stats)
        self.assertIn("success_rate", stats)


class TestSkillLibraryTools(unittest.TestCase):
    """Test MCP tools interface."""

    def setUp(self):
        """Create tools instance."""
        self.test_dir = tempfile.mkdtemp()
        self.store = SkillStore(storage_path=self.test_dir)
        self.learner = SkillLearner(store=self.store)
        self.executor = SkillExecutor(
            store=self.store,
            learner=self.learner
        )
        self.tools = SkillLibraryTools(
            store=self.store,
            learner=self.learner,
            executor=self.executor
        )

    def tearDown(self):
        """Clean up."""
        shutil.rmtree(self.test_dir)

    def test_get_tools(self):
        """Test tool definitions."""
        tool_defs = self.tools.get_tools()

        self.assertEqual(len(tool_defs), 7)

        tool_names = [t["name"] for t in tool_defs]
        self.assertIn("skill_search", tool_names)
        self.assertIn("skill_get", tool_names)
        self.assertIn("skill_create", tool_names)
        self.assertIn("skill_record_use", tool_names)
        self.assertIn("skill_list", tool_names)
        self.assertIn("skill_improve", tool_names)
        self.assertIn("skill_stats", tool_names)

    def test_handle_create(self):
        """Test skill_create tool."""
        result = self.tools.handle_tool_call("skill_create", {
            "name": "MCP Test Skill",
            "description": "Created via MCP",
            "category": "coding",
            "trigger_patterns": ["mcp test"],
            "procedure": "Test procedure"
        })

        self.assertIn("skill_id", result)
        self.assertEqual(result["name"], "MCP Test Skill")

    def test_handle_list(self):
        """Test skill_list tool."""
        # Create a skill first
        self.tools.handle_tool_call("skill_create", {
            "name": "List Test",
            "description": "For listing",
            "category": "writing",
            "trigger_patterns": ["list"],
            "procedure": "..."
        })

        result = self.tools.handle_tool_call("skill_list", {})

        self.assertEqual(result["count"], 1)
        self.assertEqual(len(result["skills"]), 1)

    def test_handle_stats(self):
        """Test skill_stats tool."""
        result = self.tools.handle_tool_call("skill_stats", {})

        self.assertIn("library", result)
        self.assertIn("learner", result)
        self.assertIn("executor", result)


class TestTitansSkillBridge(unittest.TestCase):
    """Test Titans-Skill bridge."""

    def setUp(self):
        """Create bridge instance."""
        self.test_dir = tempfile.mkdtemp()
        self.store = SkillStore(storage_path=self.test_dir)
        self.learner = SkillLearner(store=self.store)
        self.executor = SkillExecutor(
            store=self.store,
            learner=self.learner
        )
        self.bridge = TitansSkillBridge(
            store=self.store,
            learner=self.learner,
            executor=self.executor
        )

    def tearDown(self):
        """Clean up."""
        shutil.rmtree(self.test_dir)

    def test_suggest_skill_from_context(self):
        """Test skill suggestion."""
        # Create a skill
        skill = Skill.create(
            name="Context Test Skill",
            description="For context testing",
            category=SkillCategory.CODING,
            trigger_patterns=["context test", "test context"],
            procedure="..."
        )
        self.store.save(skill)

        # Request suggestion
        result = self.bridge.suggest_skill_from_context(
            current_input="I need to test context",
            memory_context={"current_goal": "testing"}
        )

        self.assertIsNotNone(result)
        suggested_skill, score, reason = result
        self.assertEqual(suggested_skill.id, skill.id)
        self.assertIn("match", reason.lower())

    def test_get_skill_context_for_input(self):
        """Test getting skill context for LLM."""
        skill = Skill.create(
            name="LLM Context Skill",
            description="For LLM context",
            category=SkillCategory.CODING,
            trigger_patterns=["llm context"],
            procedure="Step 1: Do this"
        )
        self.store.save(skill)

        context = self.bridge.get_skill_context_for_input("I need llm context help")

        self.assertIn("LLM Context Skill", context)
        self.assertIn("Step 1", context)

    def test_statistics(self):
        """Test bridge statistics."""
        stats = self.bridge.get_statistics()

        self.assertEqual(stats["suggestions_made"], 0)
        self.assertIn("connected_systems", stats)
        self.assertFalse(stats["connected_systems"]["titans_memory"])


class TestSkillLibraryHighLevel(unittest.TestCase):
    """Test high-level SkillLibrary interface."""

    def setUp(self):
        """Create library instance."""
        self.test_dir = tempfile.mkdtemp()
        self.library = SkillLibrary(storage_path=self.test_dir)

    def tearDown(self):
        """Clean up."""
        self.library.shutdown()
        shutil.rmtree(self.test_dir)

    def test_create_and_get(self):
        """Test creating and getting skills."""
        skill_id = self.library.create_skill(
            name="High Level Test",
            description="Testing high-level API",
            category="coding",
            trigger_patterns=["high level test"],
            procedure="Test procedure",
            tags=["test"]
        )

        self.assertIsNotNone(skill_id)

        skill = self.library.get_skill(skill_id)
        self.assertEqual(skill.name, "High Level Test")

    def test_search(self):
        """Test searching skills."""
        self.library.create_skill(
            name="Searchable Skill",
            description="Can be searched",
            category="writing",
            trigger_patterns=["search me"],
            procedure="..."
        )

        results = self.library.search("search me", limit=5)
        self.assertGreater(len(results), 0)

    def test_record_use(self):
        """Test recording skill usage."""
        skill_id = self.library.create_skill(
            name="Usage Test",
            description="Test usage tracking",
            category="custom",
            trigger_patterns=["usage"],
            procedure="..."
        )

        success = self.library.record_use(
            skill_id=skill_id,
            input_context="Testing usage",
            output="Usage tested",
            success=True
        )

        self.assertTrue(success)

        skill = self.library.get_skill(skill_id)
        self.assertEqual(skill.metadata.total_uses, 1)

    def test_find_applicable(self):
        """Test finding applicable skills."""
        self.library.create_skill(
            name="Applicable Skill",
            description="Should be found",
            category="coding",
            trigger_patterns=["find this skill"],
            procedure="..."
        )

        results = self.library.find_applicable("Can you find this skill?")
        self.assertGreater(len(results), 0)

    def test_get_skill_context(self):
        """Test getting skill context."""
        self.library.create_skill(
            name="Context Provider",
            description="Provides context",
            category="coding",
            trigger_patterns=["provide context"],
            procedure="Step 1: Provide\nStep 2: Context"
        )

        context = self.library.get_skill_context("please provide context")
        self.assertIn("Context Provider", context)

    def test_list_skills(self):
        """Test listing skills."""
        self.library.create_skill(
            name="List Test 1",
            description="Test 1",
            category="coding",
            trigger_patterns=["t1"],
            procedure="..."
        )
        self.library.create_skill(
            name="List Test 2",
            description="Test 2",
            category="writing",
            trigger_patterns=["t2"],
            procedure="..."
        )

        all_skills = self.library.list_skills()
        self.assertEqual(len(all_skills), 2)

        coding_only = self.library.list_skills(category="coding")
        self.assertEqual(len(coding_only), 1)

    def test_get_stats(self):
        """Test getting statistics."""
        stats = self.library.get_stats()

        self.assertIn("store", stats)
        self.assertIn("learner", stats)
        self.assertIn("executor", stats)
        self.assertIn("embeddings_available", stats)

    def test_mcp_tools(self):
        """Test MCP tool interface."""
        tools = self.library.get_mcp_tools()
        self.assertEqual(len(tools), 7)

        # Test a tool call
        result = self.library.handle_mcp_call("skill_stats", {})
        self.assertIn("library", result)

    def test_connect_bridge(self):
        """Test connecting bridge."""
        bridge = self.library.connect_bridge()

        self.assertIsNotNone(bridge)
        self.assertEqual(self.library.bridge, bridge)


if __name__ == "__main__":
    unittest.main()
