"""
Tests for ADV-02 Phase 2: LLM-Powered State Extraction Pipeline.

All LLM calls are mocked — no real Ollama/brain interaction needed.
Tests cover: JSON parsing, conversation formatting, should_extract logic,
apply helpers (project/goal/belief/relationship/environment), and the
full process_conversation pipeline.
"""

import json
import os
import sys
import tempfile
import time
import unittest
from unittest.mock import patch, MagicMock

# Ensure the project root is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from aura.consciousness.state_extractor import StateExtractor, EXTRACTION_PROMPT
from aura.consciousness.world_model import (
    WorldModel,
    ProjectStatus,
    GoalHorizon,
    BeliefCategory,
)


def _make_wm(enabled=True):
    """Create a WorldModel with a temp DB for testing."""
    tmpdir = tempfile.mkdtemp()
    tmp = os.path.join(tmpdir, "world.db")
    snap = os.path.join(tmpdir, "snapshot.json")
    return WorldModel(db_path=tmp, snapshot_path=snap, enabled=enabled)


# ============================================================================
# JSON Parsing Tests
# ============================================================================

class TestJsonParsing(unittest.TestCase):
    """Test StateExtractor._parse_json resilience."""

    def setUp(self):
        self.extractor = StateExtractor()

    def test_clean_json(self):
        """Clean JSON parses correctly."""
        data = {"projects": [], "goals": [], "beliefs": []}
        result = self.extractor._parse_json(json.dumps(data))
        self.assertEqual(result, data)

    def test_markdown_wrapped_json(self):
        """JSON wrapped in markdown code fences parses correctly."""
        data = {"projects": [{"name": "test"}]}
        text = f"```json\n{json.dumps(data)}\n```"
        result = self.extractor._parse_json(text)
        self.assertEqual(result, data)

    def test_preamble_text(self):
        """JSON preceded by preamble text parses via fallback."""
        data = {"projects": [], "beliefs": []}
        text = f"Here is the extraction:\n{json.dumps(data)}"
        result = self.extractor._parse_json(text)
        self.assertEqual(result, data)

    def test_invalid_json(self):
        """Invalid JSON returns None."""
        result = self.extractor._parse_json("this is not json at all")
        self.assertIsNone(result)

    def test_empty_input(self):
        """Empty input returns None."""
        self.assertIsNone(self.extractor._parse_json(""))
        self.assertIsNone(self.extractor._parse_json(None))


# ============================================================================
# Conversation Formatting Tests
# ============================================================================

class TestConversationFormatting(unittest.TestCase):
    """Test StateExtractor._format_conversation."""

    def setUp(self):
        self.extractor = StateExtractor()

    def test_basic_formatting(self):
        """Messages are formatted as ROLE: content."""
        messages = [
            {"role": "user", "content": "Hello there"},
            {"role": "assistant", "content": "Hi! How can I help?"},
        ]
        result = self.extractor._format_conversation(messages)
        self.assertIn("USER: Hello there", result)
        self.assertIn("ASSISTANT: Hi! How can I help?", result)

    def test_truncation(self):
        """Long messages are truncated at MAX_MESSAGE_LENGTH."""
        long_content = "x" * 3000
        messages = [{"role": "user", "content": long_content}]
        result = self.extractor._format_conversation(messages)
        self.assertIn("... [truncated]", result)
        # Should be truncated to MAX_MESSAGE_LENGTH + role prefix + truncation marker
        self.assertLess(len(result), 3000)


# ============================================================================
# should_extract Tests
# ============================================================================

class TestShouldExtract(unittest.TestCase):
    """Test StateExtractor.should_extract debounce and filtering."""

    def setUp(self):
        self.extractor = StateExtractor()
        # Reset last extract time to allow extraction
        self.extractor._last_extract_time = 0.0

    @patch("aura.consciousness.state_extractor.time")
    def test_trivial_message_skip(self, mock_time):
        """Messages with < 3 words are skipped."""
        mock_time.monotonic.return_value = 100.0
        messages = [{"role": "user", "content": "hi"}]
        self.assertFalse(self.extractor.should_extract(messages))

    @patch("aura.consciousness.state_extractor.time")
    def test_normal_message_pass(self, mock_time):
        """Normal messages pass the filter."""
        mock_time.monotonic.return_value = 100.0
        messages = [{"role": "user", "content": "I'm working on a new project today"}]
        self.assertTrue(self.extractor.should_extract(messages))

    def test_debounce(self):
        """Extractions within MIN_EXTRACT_INTERVAL are blocked."""
        self.extractor._last_extract_time = time.monotonic()
        messages = [{"role": "user", "content": "I started a big new project"}]
        self.assertFalse(self.extractor.should_extract(messages))


# ============================================================================
# Apply Project Tests
# ============================================================================

class TestApplyProject(unittest.TestCase):
    """Test WorldModel._apply_project."""

    def setUp(self):
        self.wm = _make_wm()

    def test_new_project_created(self):
        """New project action creates a project."""
        data = {
            "name": "MyApp",
            "action": "new",
            "technologies_mentioned": ["Python", "FastAPI"],
            "progress_notes": "Just started",
        }
        result = self.wm._apply_project(data, "conv1")
        self.assertEqual(result, 1)
        projects = self.wm.get_all_projects()
        self.assertEqual(len(projects), 1)
        self.assertEqual(projects[0].name, "MyApp")
        self.assertIn("Python", projects[0].technologies)

    def test_existing_project_updated(self):
        """Updating an existing project merges technologies."""
        self.wm.add_project("MyApp", technologies=["Python"])
        data = {
            "name": "MyApp",
            "action": "update",
            "technologies_mentioned": ["Docker"],
            "progress_notes": "Added containerization",
        }
        result = self.wm._apply_project(data, "conv2")
        self.assertEqual(result, 1)
        proj = self.wm.get_all_projects()[0]
        self.assertIn("Python", proj.technologies)
        self.assertIn("Docker", proj.technologies)

    def test_fuzzy_match(self):
        """Substring matching finds existing project."""
        self.wm.add_project("AURA Apprentice Agent")
        data = {"name": "Apprentice Agent", "action": "mention"}
        result = self.wm._apply_project(data, "conv3")
        self.assertEqual(result, 1)
        # Should not create a new project
        self.assertEqual(len(self.wm.get_all_projects()), 1)

    def test_blocker_handling(self):
        """New blockers are added and resolved blockers are resolved."""
        proj = self.wm.add_project("TestProj")
        self.wm.add_blocker(proj.id, "API key missing")
        data = {
            "name": "TestProj",
            "action": "update",
            "new_blockers": ["Database not provisioned"],
            "resolved_blockers": ["API key missing"],
        }
        result = self.wm._apply_project(data, "conv4")
        self.assertEqual(result, 1)
        blockers = self.wm.get_project_blockers(proj.id)
        ongoing = [b for b in blockers if b["status"] == "ongoing"]
        resolved = [b for b in blockers if b["status"] == "resolved"]
        self.assertEqual(len(ongoing), 1)
        self.assertEqual(ongoing[0]["description"], "Database not provisioned")
        self.assertEqual(len(resolved), 1)


# ============================================================================
# Apply Belief Tests
# ============================================================================

class TestApplyBelief(unittest.TestCase):
    """Test WorldModel._apply_belief."""

    def setUp(self):
        self.wm = _make_wm()

    def test_new_belief_created(self):
        """A new belief statement creates a belief."""
        data = {
            "statement": "User prefers Python over JavaScript",
            "category": "preference",
            "confidence": 0.8,
        }
        updated, contras = self.wm._apply_belief(data, "conv1")
        self.assertEqual(updated, 1)
        self.assertEqual(contras, 0)
        beliefs = self.wm.get_current_beliefs()
        self.assertEqual(len(beliefs), 1)
        self.assertEqual(beliefs[0].statement, "User prefers Python over JavaScript")

    def test_reinforce_existing(self):
        """A similar statement reinforces existing belief."""
        self.wm.add_belief(
            "User prefers Python over JavaScript",
            category=BeliefCategory.PREFERENCE,
            confidence=0.7,
        )
        data = {
            "statement": "User prefers Python over JavaScript for backend",
            "category": "preference",
            "confidence": 0.8,
        }
        updated, contras = self.wm._apply_belief(data, "conv2")
        self.assertEqual(updated, 1)
        beliefs = self.wm.get_current_beliefs()
        # Should still be 1 belief (reinforced, not new)
        self.assertEqual(len(beliefs), 1)
        self.assertGreater(beliefs[0].confidence, 0.7)

    def test_contradiction_detected_by_llm(self):
        """LLM-flagged contradiction supersedes old belief."""
        self.wm.add_belief(
            "User prefers tabs for indentation",
            category=BeliefCategory.PREFERENCE,
            confidence=0.8,
        )
        data = {
            "statement": "User prefers spaces for indentation",
            "category": "preference",
            "confidence": 0.9,
            "contradicts_existing": "Previously preferred tabs, now spaces",
        }
        updated, contras = self.wm._apply_belief(data, "conv3")
        self.assertEqual(updated, 1)
        self.assertEqual(contras, 1)
        beliefs = self.wm.get_current_beliefs()
        # Old belief superseded, new one active
        self.assertEqual(len(beliefs), 1)
        self.assertIn("spaces", beliefs[0].statement)
        # Contradiction logged
        self.assertEqual(len(self.wm.get_unresolved_contradictions()), 1)

    def test_local_contradiction_check(self):
        """Local heuristic detects negation contradictions."""
        self.wm.add_belief(
            "User uses vim as editor",
            category=BeliefCategory.PREFERENCE,
            confidence=0.7,
        )
        data = {
            "statement": "User no longer uses vim as editor",
            "category": "preference",
            "confidence": 0.8,
        }
        updated, contras = self.wm._apply_belief(data, "conv4")
        self.assertEqual(updated, 1)
        self.assertGreaterEqual(contras, 1)


# ============================================================================
# Apply Relationship Tests
# ============================================================================

class TestApplyRelationship(unittest.TestCase):
    """Test WorldModel._apply_relationship."""

    def setUp(self):
        self.wm = _make_wm()

    def test_new_person(self):
        """A new person creates a relationship."""
        data = {
            "name": "Alice",
            "role": "coworker",
            "context": "Working on project X",
            "sentiment": "positive",
        }
        result = self.wm._apply_relationship(data, "conv1")
        self.assertEqual(result, 1)
        rel = self.wm.get_relationship("Alice")
        self.assertIsNotNone(rel)
        self.assertEqual(rel.role, "coworker")
        self.assertEqual(rel.sentiment, "positive")

    def test_existing_person_updated(self):
        """An existing person gets updated mention count and context."""
        self.wm.add_relationship("Bob", role="friend")
        data = {
            "name": "Bob",
            "role": "colleague",
            "context": "Started working together",
            "sentiment": "positive",
        }
        result = self.wm._apply_relationship(data, "conv2")
        self.assertEqual(result, 1)
        rel = self.wm.get_relationship("Bob")
        self.assertEqual(rel.role, "colleague")
        self.assertEqual(rel.mention_count, 2)


# ============================================================================
# Apply Environment Tests
# ============================================================================

class TestApplyEnvironment(unittest.TestCase):
    """Test WorldModel._apply_environment."""

    def setUp(self):
        self.wm = _make_wm()

    def test_new_entry(self):
        """A new environment entry is created."""
        data = {"key": "os", "category": "hardware", "value": "Windows 11"}
        result = self.wm._apply_environment(data, "conv1")
        self.assertEqual(result, 1)
        env = self.wm.get_environment("os")
        self.assertIsNotNone(env)
        self.assertEqual(env["value"], "Windows 11")

    def test_update_existing(self):
        """An existing environment entry is updated."""
        self.wm.set_environment("editor", "tool", "vim")
        data = {"key": "editor", "category": "tool", "value": "vscode"}
        result = self.wm._apply_environment(data, "conv2")
        self.assertEqual(result, 1)
        env = self.wm.get_environment("editor")
        self.assertEqual(env["value"], "vscode")
        self.assertEqual(env["observation_count"], 2)


# ============================================================================
# Full Pipeline Tests
# ============================================================================

class TestProcessConversation(unittest.TestCase):
    """Test WorldModel.process_conversation end-to-end."""

    def setUp(self):
        self.wm = _make_wm()

    @patch("aura.consciousness.state_extractor.get_state_extractor")
    def test_full_pipeline_with_mock(self, mock_get_extractor):
        """Full pipeline applies extraction results to world model."""
        extraction_result = {
            "projects": [
                {
                    "name": "WebApp",
                    "action": "new",
                    "technologies_mentioned": ["React"],
                    "progress_notes": "Building frontend",
                    "new_blockers": [],
                    "resolved_blockers": [],
                }
            ],
            "goals": [
                {
                    "description": "Launch MVP by March",
                    "action": "new",
                    "horizon": "short_term",
                    "progress_delta": 0.0,
                    "evidence": "User mentioned deadline",
                }
            ],
            "beliefs": [
                {
                    "statement": "User wants fast iteration",
                    "category": "user_intent",
                    "confidence": 0.8,
                }
            ],
            "people_mentioned": [
                {
                    "name": "Charlie",
                    "role": "designer",
                    "context": "UI mockups",
                    "sentiment": "positive",
                }
            ],
            "environment_changes": [
                {"key": "framework", "category": "tool", "value": "React"}
            ],
        }

        mock_extractor = MagicMock()
        mock_extractor.should_extract.return_value = True
        mock_extractor.extract.return_value = extraction_result
        mock_get_extractor.return_value = mock_extractor

        messages = [
            {"role": "user", "content": "I'm building a WebApp with React"},
            {"role": "assistant", "content": "Great! Let me help you get started."},
        ]

        counts = self.wm.process_conversation("conv1", messages)

        self.assertEqual(counts["projects_updated"], 1)
        self.assertEqual(counts["goals_updated"], 1)
        self.assertEqual(counts["beliefs_updated"], 1)
        self.assertEqual(counts["relationships_updated"], 1)
        self.assertEqual(counts["environment_updated"], 1)

        # Verify entities were created
        self.assertEqual(len(self.wm.get_all_projects()), 1)
        self.assertEqual(self.wm.get_all_projects()[0].name, "WebApp")
        self.assertEqual(len(self.wm.get_active_goals()), 1)
        self.assertEqual(len(self.wm.get_current_beliefs()), 1)
        self.assertIsNotNone(self.wm.get_relationship("Charlie"))
        self.assertIsNotNone(self.wm.get_environment("framework"))

    def test_disabled_skips(self):
        """Disabled world model returns zero counts."""
        wm = _make_wm(enabled=False)
        counts = wm.process_conversation("conv1", [{"role": "user", "content": "test"}])
        self.assertEqual(sum(counts.values()), 0)

    @patch("aura.consciousness.state_extractor.get_state_extractor")
    def test_empty_extraction_skips(self, mock_get_extractor):
        """Empty extraction result causes no changes."""
        mock_extractor = MagicMock()
        mock_extractor.should_extract.return_value = True
        mock_extractor.extract.return_value = {}
        mock_get_extractor.return_value = mock_extractor

        messages = [{"role": "user", "content": "Hello, how are you today?"}]
        counts = self.wm.process_conversation("conv1", messages)
        self.assertEqual(sum(counts.values()), 0)

    @patch("aura.consciousness.state_extractor.get_state_extractor")
    def test_extraction_failure_graceful(self, mock_get_extractor):
        """Extraction failure is handled gracefully."""
        mock_extractor = MagicMock()
        mock_extractor.should_extract.return_value = True
        mock_extractor.extract.side_effect = Exception("LLM crashed")
        mock_get_extractor.return_value = mock_extractor

        messages = [{"role": "user", "content": "Working on important stuff"}]
        counts = self.wm.process_conversation("conv1", messages)
        self.assertEqual(sum(counts.values()), 0)
        # World model should be unchanged
        self.assertEqual(len(self.wm.get_all_projects()), 0)


# ============================================================================
# Run
# ============================================================================

if __name__ == "__main__":
    unittest.main()
