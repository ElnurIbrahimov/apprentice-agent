"""Tests for agent mixin classes — KGBrainMixin, SkillManagerMixin, NarrativeMixin, DirectHandlersMixin."""
import pytest
from unittest.mock import MagicMock, patch
import inspect

from aura.core.kg_brain import KGBrainMixin
from aura.core.skill_manager import SkillManagerMixin
from aura.core.narrative import NarrativeMixin
from aura.core.direct_handlers import DirectHandlersMixin


# ── ApprenticeAgent inherits all mixins ──────────────────────────────

def test_apprentice_agent_inherits_all_mixins():
    from aura.agent import ApprenticeAgent
    assert issubclass(ApprenticeAgent, KGBrainMixin)
    assert issubclass(ApprenticeAgent, SkillManagerMixin)
    assert issubclass(ApprenticeAgent, NarrativeMixin)
    assert issubclass(ApprenticeAgent, DirectHandlersMixin)


# ── KGBrainMixin ─────────────────────────────────────────────────────

class TestKGBrainMixin:

    def test_methods_exist(self):
        expected = [
            "_handle_knowledge_graph_command",
            "get_kg_brain_stats",
            "kg_brain_query",
            "kg_brain_add_knowledge",
            "kg_brain_consolidate",
            "get_episodic_memory_stats",
            "episodic_recall",
            "episodic_record",
        ]
        for method_name in expected:
            assert hasattr(KGBrainMixin, method_name), f"Missing method: {method_name}"
            assert callable(getattr(KGBrainMixin, method_name))

    def test_get_kg_brain_stats_no_brain(self):
        obj = KGBrainMixin()
        obj.kg_brain = None
        result = obj.get_kg_brain_stats()
        assert result["available"] is False

    def test_get_kg_brain_stats_with_brain(self):
        obj = KGBrainMixin()
        obj.kg_brain = MagicMock()
        obj.kg_brain.get_statistics.return_value = {
            "total_entities": 10,
            "total_relationships": 5,
            "entity_type_distribution": {"person": 3},
            "average_importance": 0.7,
        }
        obj.kg_bridge = MagicMock()
        obj.kg_bridge.get_statistics.return_value = {
            "total_entities_extracted": 8,
            "total_extractions_triggered": 4,
            "queue_size": 0,
        }
        result = obj.get_kg_brain_stats()
        assert result["available"] is True
        assert result["total_entities"] == 10

    def test_kg_brain_query_no_engine(self):
        obj = KGBrainMixin()
        obj.kg_query_engine = None
        result = obj.kg_brain_query("test")
        assert "not available" in result.lower()

    def test_kg_brain_add_knowledge_no_bridge(self):
        obj = KGBrainMixin()
        obj.kg_bridge = None
        result = obj.kg_brain_add_knowledge("some text")
        assert result["success"] is False

    def test_kg_brain_consolidate_no_brain(self):
        obj = KGBrainMixin()
        obj.kg_brain = None
        result = obj.kg_brain_consolidate()
        assert result["success"] is False

    def test_handle_kg_command_returns_none_for_non_kg(self):
        obj = KGBrainMixin()
        obj.kg_brain = None
        obj.kg_bridge = None
        obj.kg_query_engine = None
        obj.tools = {}
        result = obj._handle_knowledge_graph_command("hello world")
        assert result is None

    def test_episodic_time_travel_deprecated(self):
        obj = KGBrainMixin()
        result = obj.episodic_time_travel("yesterday")
        assert result["success"] is False

    def test_episodic_get_health(self):
        obj = KGBrainMixin()
        result = obj.episodic_get_health()
        assert "consolidated" in result["status"]


# ── SkillManagerMixin ────────────────────────────────────────────────

class TestSkillManagerMixin:

    def test_methods_exist(self):
        expected = [
            "get_skill_library_stats",
            "skill_search",
            "skill_get",
            "skill_create",
            "skill_record_use",
            "skill_find_applicable",
            "skill_get_context",
            "skill_record_interaction",
            "skill_list",
            "skill_improve",
        ]
        for method_name in expected:
            assert hasattr(SkillManagerMixin, method_name), f"Missing method: {method_name}"
            assert callable(getattr(SkillManagerMixin, method_name))

    def test_stats_no_library(self):
        obj = SkillManagerMixin()
        obj.skill_library = None
        result = obj.get_skill_library_stats()
        assert result["available"] is False

    def test_search_no_library(self):
        obj = SkillManagerMixin()
        obj.skill_library = None
        assert obj.skill_search("test") == []

    def test_get_no_library(self):
        obj = SkillManagerMixin()
        obj.skill_library = None
        result = obj.skill_get("some-id")
        assert "error" in result

    def test_create_no_library(self):
        obj = SkillManagerMixin()
        obj.skill_library = None
        result = obj.skill_create("name", "desc", "coding", ["trigger"], "proc")
        assert result["success"] is False

    def test_find_applicable_no_library(self):
        obj = SkillManagerMixin()
        obj.skill_library = None
        assert obj.skill_find_applicable("test") == []

    def test_get_context_no_library(self):
        obj = SkillManagerMixin()
        obj.skill_library = None
        assert obj.skill_get_context("test") == ""

    def test_record_interaction_no_library(self):
        obj = SkillManagerMixin()
        obj.skill_library = None
        assert obj.skill_record_interaction("input", "output", True) is None

    def test_list_no_library(self):
        obj = SkillManagerMixin()
        obj.skill_library = None
        assert obj.skill_list() == []

    def test_improve_no_library(self):
        obj = SkillManagerMixin()
        obj.skill_library = None
        result = obj.skill_improve("id")
        assert "error" in result

    def test_skill_search_with_library(self):
        obj = SkillManagerMixin()
        obj.skill_library = MagicMock()
        obj.skill_library.search.return_value = [("skill-1", 0.9)]
        result = obj.skill_search("query")
        assert result == [("skill-1", 0.9)]

    def test_skill_get_context_with_library(self):
        obj = SkillManagerMixin()
        obj.skill_library = MagicMock()
        obj.skill_library.get_skill_context.return_value = "Context string"
        assert obj.skill_get_context("test") == "Context string"


# ── NarrativeMixin ───────────────────────────────────────────────────

class TestNarrativeMixin:

    def test_methods_exist(self):
        expected = [
            "_analyze_emotion",
            "get_current_mood",
            "get_mood_emoji",
            "_get_soul_prompt",
            "_temporal_grounding",
            "_build_aura_context",
            "_pre_response_appraisal",
            "_post_response_feedback",
            "_handle_aura_command",
            "_handle_evoemo_command",
        ]
        for method_name in expected:
            assert hasattr(NarrativeMixin, method_name), f"Missing method: {method_name}"
            assert callable(getattr(NarrativeMixin, method_name))

    def test_get_mood_emoji_no_tool(self):
        obj = NarrativeMixin()
        obj.tools = {}
        assert obj.get_mood_emoji() == "\U0001f610"  # neutral face

    def test_get_current_mood_no_tool(self):
        obj = NarrativeMixin()
        obj.tools = {}
        assert obj.get_current_mood() is None

    def test_analyze_emotion_no_tool(self):
        obj = NarrativeMixin()
        obj.tools = {}
        assert obj._analyze_emotion("I'm happy") is None

    def test_get_soul_prompt_no_soul(self):
        obj = NarrativeMixin()
        obj._soul = None
        assert obj._get_soul_prompt() == ""

    def test_get_soul_prompt_with_soul(self):
        obj = NarrativeMixin()
        obj._soul = MagicMock()
        obj._soul.get_system_prompt_addition.return_value = "Be creative."
        assert obj._get_soul_prompt() == "Be creative."

    def test_handle_aura_command_non_aura(self):
        obj = NarrativeMixin()
        obj.tools = {}
        obj.identity = {"name": "AURA"}
        result = obj._handle_aura_command("what's the weather?")
        assert result is None

    def test_handle_aura_status(self):
        obj = NarrativeMixin()
        obj.tools = {"tool1": MagicMock()}
        obj.identity = {"name": "AURA"}
        result = obj._handle_aura_command("aura status")
        assert result is not None
        assert "AURA" in result
        assert "Online" in result

    def test_handle_evoemo_command_non_evoemo(self):
        obj = NarrativeMixin()
        obj.tools = {}
        result = obj._handle_evoemo_command("hello")
        assert result is None

    def test_post_response_feedback_no_previous(self):
        """Should be a no-op when no previous exchange exists."""
        obj = NarrativeMixin()
        obj._prev_message = None
        obj._prev_response = None
        # Should not raise
        obj._post_response_feedback("new message")


# ── DirectHandlersMixin ──────────────────────────────────────────────

class TestDirectHandlersMixin:

    def test_methods_exist(self):
        expected = [
            "_handle_monologue_command",
            "_handle_neurodream_command",
            "_handle_git_command",
            "_handle_direct_search",
            "_handle_direct_crypto",
            "_handle_direct_code",
        ]
        for method_name in expected:
            assert hasattr(DirectHandlersMixin, method_name), f"Missing method: {method_name}"
            assert callable(getattr(DirectHandlersMixin, method_name))

    def test_monologue_non_monologue(self):
        obj = DirectHandlersMixin()
        obj.tools = {}
        result = obj._handle_monologue_command("what's for lunch?")
        assert result is None

    def test_monologue_no_tool(self):
        obj = DirectHandlersMixin()
        obj.tools = {}
        result = obj._handle_monologue_command("show thoughts")
        assert result == "Inner monologue not available."

    def test_neurodream_non_command(self):
        obj = DirectHandlersMixin()
        result = obj._handle_neurodream_command("hello")
        assert result is None

    def test_neurodream_no_engine(self):
        obj = DirectHandlersMixin()
        obj.neurodream = None
        result = obj._handle_neurodream_command("dream status")
        assert result == "NeuroDream not available."

    def test_git_non_command(self):
        obj = DirectHandlersMixin()
        obj.tools = {}
        result = obj._handle_git_command("hello world")
        assert result is None

    def test_git_no_tool(self):
        obj = DirectHandlersMixin()
        obj.tools = {}
        result = obj._handle_git_command("git status")
        assert result == "Git tool is not available."

    def test_git_status_with_tool(self):
        obj = DirectHandlersMixin()
        mock_git = MagicMock()
        mock_git.status.return_value = {"success": True, "output": "On branch main\nnothing to commit"}
        obj.tools = {"git": mock_git}
        result = obj._handle_git_command("git status")
        assert "On branch main" in result

    def test_direct_crypto_non_crypto(self):
        obj = DirectHandlersMixin()
        obj.tools = {}
        result = obj._handle_direct_crypto("hello")
        assert result is None

    def test_direct_code_non_code(self):
        obj = DirectHandlersMixin()
        obj.tools = {}
        result = obj._handle_direct_code("hello")
        assert result is None

    def test_direct_search_non_search(self):
        obj = DirectHandlersMixin()
        obj.tools = {}
        result = obj._handle_direct_search("tell me about cats")
        assert result is None
