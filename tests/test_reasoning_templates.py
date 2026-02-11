"""Tests for Reasoning Template Library — Phase 3 of ADV-01."""

import json
import math
import shutil
import struct
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from apprentice_agent.consciousness.reasoning_templates import (
    ReasoningTemplate,
    ReasoningTemplateLibrary,
    ReasoningTrace,
    TemplateMatch,
    _EmbeddingHelper,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def temp_data_dir():
    """Create a temporary data directory for test databases."""
    tmpdir = tempfile.mkdtemp(prefix="aura_test_templates_")
    yield Path(tmpdir)
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture
def library(temp_data_dir):
    """Create a ReasoningTemplateLibrary with test database."""
    db_path = str(temp_data_dir / "test_meta.db")
    lib = ReasoningTemplateLibrary(db_path=db_path, enabled=True)
    return lib


@pytest.fixture
def disabled_library(temp_data_dir):
    """Create a disabled ReasoningTemplateLibrary."""
    db_path = str(temp_data_dir / "test_meta_disabled.db")
    return ReasoningTemplateLibrary(db_path=db_path, enabled=False)


def _mock_embed(text):
    """Deterministic mock embedding: hash text into a 384-dim vector."""
    h = hash(text) % (2**32)
    vec = [(h + i) % 100 / 100.0 for i in range(384)]
    # Normalize
    norm = math.sqrt(sum(x * x for x in vec))
    if norm > 0:
        vec = [x / norm for x in vec]
    return vec


def _similar_embed(text):
    """Return a nearly-identical vector for any input (simulates high similarity)."""
    base = [0.1] * 384
    norm = math.sqrt(sum(x * x for x in base))
    return [x / norm for x in base]


# ============================================================================
# TestTraceCollection
# ============================================================================


class TestTraceCollection:
    """Test trace collection with reward thresholds."""

    @patch.object(_EmbeddingHelper, "embed", side_effect=_mock_embed)
    @patch.object(_EmbeddingHelper, "available", new_callable=PropertyMock, return_value=True)
    def test_collect_high_reward_trace(self, mock_avail, mock_embed, library):
        """High-reward trace should be stored."""
        result = library.collect_trace(
            request_id="req_001",
            problem="How do I sort a list in Python?",
            category="code",
            strategy="chain_of_thought",
            full_trace=json.dumps([{"step": "understand"}, {"step": "solve"}]),
            reward=0.9,
        )
        assert result is True

        stats = library.get_stats()
        assert stats["trace_count"] == 1

    @patch.object(_EmbeddingHelper, "embed", side_effect=_mock_embed)
    def test_skip_low_reward_trace(self, mock_embed, library):
        """Traces below threshold should be skipped."""
        result = library.collect_trace(
            request_id="req_002",
            problem="Simple question",
            category="analysis",
            strategy="chain_of_thought",
            full_trace=json.dumps([{"step": "answer"}]),
            reward=0.5,
        )
        assert result is False

        stats = library.get_stats()
        assert stats["trace_count"] == 0

    @patch.object(_EmbeddingHelper, "embed", side_effect=_mock_embed)
    def test_skip_empty_problem(self, mock_embed, library):
        """Empty problem text should be skipped."""
        result = library.collect_trace(
            request_id="req_003",
            problem="",
            category="code",
            strategy="chain_of_thought",
            full_trace=json.dumps([]),
            reward=0.95,
        )
        assert result is False

    def test_skip_when_disabled(self, disabled_library):
        """Disabled library should skip all traces."""
        result = disabled_library.collect_trace(
            request_id="req_004",
            problem="Test problem",
            category="code",
            strategy="chain_of_thought",
            full_trace=json.dumps([]),
            reward=0.95,
        )
        assert result is False


# ============================================================================
# TestEmbeddingHelper
# ============================================================================


class TestEmbeddingHelper:
    """Test embedding helper utilities (cosine sim, serialization)."""

    def test_cosine_similarity_identical(self):
        """Identical vectors should have similarity ~1.0."""
        vec = [0.1, 0.2, 0.3, 0.4]
        sim = _EmbeddingHelper.cosine_similarity(vec, vec)
        assert sim == pytest.approx(1.0, abs=0.001)

    def test_cosine_similarity_orthogonal(self):
        """Orthogonal vectors should have similarity ~0.0."""
        a = [1.0, 0.0, 0.0]
        b = [0.0, 1.0, 0.0]
        sim = _EmbeddingHelper.cosine_similarity(a, b)
        assert sim == pytest.approx(0.0, abs=0.001)

    def test_cosine_similarity_zero_vector(self):
        """Zero vector should return 0.0 similarity."""
        a = [0.0, 0.0, 0.0]
        b = [1.0, 2.0, 3.0]
        sim = _EmbeddingHelper.cosine_similarity(a, b)
        assert sim == 0.0

    def test_serialize_deserialize_roundtrip(self):
        """Serialized embeddings should roundtrip correctly."""
        vec = [0.1, 0.2, 0.3, 0.4, 0.5]
        blob = _EmbeddingHelper.serialize_embedding(vec)
        recovered = _EmbeddingHelper.deserialize_embedding(blob)
        assert len(recovered) == len(vec)
        for a, b in zip(vec, recovered):
            assert a == pytest.approx(b, abs=1e-6)


# ============================================================================
# TestTemplateRetrieval
# ============================================================================


class TestTemplateRetrieval:
    """Test template retrieval by similarity."""

    def test_no_templates_returns_none(self, library):
        """Empty library should return None."""
        with patch.object(_EmbeddingHelper, "embed", side_effect=_mock_embed):
            result = library.retrieve_template("How do I sort a list?")
        assert result is None

    @patch.object(_EmbeddingHelper, "embed", side_effect=_similar_embed)
    @patch.object(_EmbeddingHelper, "available", new_callable=PropertyMock, return_value=True)
    def test_matches_by_similarity(self, mock_avail, mock_embed, library):
        """Should match a template with high similarity."""
        import sqlite3

        # Insert a template directly
        vec = _similar_embed("test")
        blob = _EmbeddingHelper.serialize_embedding(vec)
        conn = sqlite3.connect(library._db_path)
        conn.execute(
            """INSERT INTO reasoning_templates
               (template_id, name, description, abstract_steps,
                applicable_categories, source_trace_ids, embedding,
                times_used, avg_reward_when_used, avg_reward_baseline,
                status, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                "tmpl_test_001", "Sort Pattern", "Pattern for sorting problems",
                json.dumps(["Understand input", "Choose algorithm", "Implement"]),
                json.dumps(["code"]), json.dumps(["trace_1"]),
                blob, 5, 0.85, 0.80, "active", "2025-01-01T00:00:00",
            ),
        )
        conn.commit()
        conn.close()

        result = library.retrieve_template("How do I sort a list?", category="code")
        assert result is not None
        assert isinstance(result, TemplateMatch)
        assert result.template.name == "Sort Pattern"
        assert result.similarity_score > 0.0
        assert "Sort Pattern" in result.guidance_text

    @patch.object(_EmbeddingHelper, "embed", side_effect=_similar_embed)
    @patch.object(_EmbeddingHelper, "available", new_callable=PropertyMock, return_value=True)
    def test_skips_deprecated_templates(self, mock_avail, mock_embed, library):
        """Deprecated templates should not be returned."""
        import sqlite3

        vec = _similar_embed("test")
        blob = _EmbeddingHelper.serialize_embedding(vec)
        conn = sqlite3.connect(library._db_path)
        conn.execute(
            """INSERT INTO reasoning_templates
               (template_id, name, description, abstract_steps,
                applicable_categories, source_trace_ids, embedding,
                times_used, avg_reward_when_used, avg_reward_baseline,
                status, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                "tmpl_dep_001", "Old Pattern", "Deprecated pattern",
                json.dumps(["step1"]), json.dumps(["code"]),
                json.dumps([]), blob, 30, 0.4, 0.8,
                "deprecated", "2025-01-01T00:00:00",
            ),
        )
        conn.commit()
        conn.close()

        result = library.retrieve_template("Test query", category="code")
        assert result is None


# ============================================================================
# TestTemplateUsageRecording
# ============================================================================


class TestTemplateUsageRecording:
    """Test recording template usage and deprecation."""

    def test_updates_stats(self, library):
        """Usage recording should update times_used and avg_reward."""
        import sqlite3

        # Insert a template
        conn = sqlite3.connect(library._db_path)
        conn.execute(
            """INSERT INTO reasoning_templates
               (template_id, name, description, abstract_steps,
                applicable_categories, source_trace_ids,
                times_used, avg_reward_when_used, avg_reward_baseline,
                status, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                "tmpl_usage_001", "Test Template", "Test",
                json.dumps(["step1"]), json.dumps(["code"]),
                json.dumps([]), 0, 0.0, 0.8, "active", "2025-01-01T00:00:00",
            ),
        )
        conn.commit()
        conn.close()

        # Record usage
        library.record_template_usage("tmpl_usage_001", 0.9)
        library.record_template_usage("tmpl_usage_001", 0.85)

        conn = sqlite3.connect(library._db_path)
        row = conn.execute(
            "SELECT times_used, avg_reward_when_used FROM reasoning_templates "
            "WHERE template_id = 'tmpl_usage_001'"
        ).fetchone()
        conn.close()

        assert row[0] == 2  # times_used
        assert row[1] == pytest.approx(0.875, abs=0.01)  # (0.9 + 0.85) / 2

    def test_triggers_deprecation(self, library):
        """Template should be deprecated after enough low-reward uses."""
        import sqlite3

        # Insert a template with high baseline and enough uses to be near threshold
        conn = sqlite3.connect(library._db_path)
        conn.execute(
            """INSERT INTO reasoning_templates
               (template_id, name, description, abstract_steps,
                applicable_categories, source_trace_ids,
                times_used, avg_reward_when_used, avg_reward_baseline,
                status, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                "tmpl_dep_test", "Bad Template", "Underperforming",
                json.dumps(["step1"]), json.dumps(["code"]),
                json.dumps([]), 19, 0.4, 0.8, "active", "2025-01-01T00:00:00",
            ),
        )
        conn.commit()
        conn.close()

        # This brings it to 20 uses with low reward, should trigger deprecation
        library.record_template_usage("tmpl_dep_test", 0.4)

        conn = sqlite3.connect(library._db_path)
        row = conn.execute(
            "SELECT status FROM reasoning_templates WHERE template_id = 'tmpl_dep_test'"
        ).fetchone()
        conn.close()

        assert row[0] == "deprecated"


# ============================================================================
# TestTemplateAbstraction
# ============================================================================


class TestTemplateAbstraction:
    """Test batch abstraction logic."""

    def test_parse_template_json(self, library):
        """JSON parsing should handle raw and markdown-fenced responses."""
        # Raw JSON
        raw = '{"name": "Test", "description": "desc", "abstract_steps": ["s1"], "applicable_categories": ["code"]}'
        result = library._parse_template_json(raw)
        assert result is not None
        assert result["name"] == "Test"

        # Markdown-fenced JSON
        fenced = '```json\n{"name": "Test2", "description": "desc2", "abstract_steps": ["s1"]}\n```'
        result2 = library._parse_template_json(fenced)
        assert result2 is not None
        assert result2["name"] == "Test2"

        # Invalid
        result3 = library._parse_template_json("not json at all")
        assert result3 is None

    @patch.object(_EmbeddingHelper, "embed", side_effect=_mock_embed)
    @patch.object(_EmbeddingHelper, "available", new_callable=PropertyMock, return_value=True)
    def test_batch_grouping_by_category(self, mock_avail, mock_embed, library):
        """Traces should be grouped by category before abstraction."""
        import sqlite3

        # Insert traces in two categories
        conn = sqlite3.connect(library._db_path)
        for i in range(5):
            conn.execute(
                """INSERT INTO reasoning_traces
                   (trace_id, problem, problem_category, strategy_used,
                    full_trace, composite_reward, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    f"trace_code_{i}", f"Code problem {i}", "code",
                    "chain_of_thought", json.dumps([{"step": "solve"}]),
                    0.9, "2025-01-01T00:00:00",
                ),
            )
        for i in range(5):
            conn.execute(
                """INSERT INTO reasoning_traces
                   (trace_id, problem, problem_category, strategy_used,
                    full_trace, composite_reward, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    f"trace_math_{i}", f"Math problem {i}", "math",
                    "chain_of_thought", json.dumps([{"step": "compute"}]),
                    0.85, "2025-01-01T00:00:00",
                ),
            )
        conn.commit()
        conn.close()

        # Mock the Brain to capture calls
        mock_brain = MagicMock()
        mock_brain.think.return_value = json.dumps({
            "name": "Mock Pattern",
            "description": "A mock pattern",
            "abstract_steps": ["Step 1", "Step 2"],
            "applicable_categories": ["code"],
        })

        with patch.dict("sys.modules", {"apprentice_agent.brain": MagicMock(Brain=MagicMock(return_value=mock_brain))}):
            library._run_abstraction_batch()

        # Brain should have been called for both categories (code + math)
        assert mock_brain.think.call_count == 2


# ============================================================================
# TestStats
# ============================================================================


class TestStats:
    """Test monitoring/debugging stats."""

    def test_get_stats_returns_expected_keys(self, library):
        """Stats should include all expected monitoring keys."""
        stats = library.get_stats()
        expected_keys = {
            "enabled", "trace_count", "template_count",
            "active_templates", "deprecated_templates",
            "total_template_uses", "avg_trace_reward",
            "traces_since_abstraction", "category_trace_counts",
        }
        assert expected_keys.issubset(set(stats.keys()))
        assert stats["enabled"] is True
        assert stats["trace_count"] == 0
        assert stats["template_count"] == 0


# ============================================================================
# TestFormatGuidance
# ============================================================================


class TestFormatGuidance:
    """Test guidance text formatting."""

    def test_guidance_text_formatting(self, library):
        """Guidance text should include template name and steps."""
        template = ReasoningTemplate(
            template_id="tmpl_fmt_001",
            name="Debug Pattern",
            description="Use when debugging runtime errors",
            abstract_steps=json.dumps(["Reproduce the error", "Check stack trace", "Fix root cause"]),
            applicable_categories=json.dumps(["debug"]),
            source_trace_ids=json.dumps([]),
            status="active",
            created_at="2025-01-01T00:00:00",
        )
        guidance = library._format_guidance(template)

        assert "Debug Pattern" in guidance
        assert "Reproduce the error" in guidance
        assert "Check stack trace" in guidance
        assert "Fix root cause" in guidance
        assert "1." in guidance
        assert "2." in guidance
        assert "3." in guidance
