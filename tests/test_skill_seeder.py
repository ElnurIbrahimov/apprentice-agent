"""Tests for scripts/seed_skill_store.py."""

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# The seeder script lives at scripts/seed_skill_store.py — make it importable.
_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(_SCRIPTS))


def test_parse_candidates_handles_plain_json_array():
    from seed_skill_store import _parse_candidates
    raw = '[{"name": "x"}, {"name": "y"}]'
    assert _parse_candidates(raw) == [{"name": "x"}, {"name": "y"}]


def test_parse_candidates_strips_code_fences():
    from seed_skill_store import _parse_candidates
    raw = '```json\n[{"name": "x"}]\n```'
    assert _parse_candidates(raw) == [{"name": "x"}]


def test_parse_candidates_finds_array_amid_commentary():
    from seed_skill_store import _parse_candidates
    raw = 'Sure! Here are the skills I found:\n[{"name": "x"}]\nHope this helps.'
    assert _parse_candidates(raw) == [{"name": "x"}]


def test_parse_candidates_returns_empty_on_invalid_json():
    from seed_skill_store import _parse_candidates
    assert _parse_candidates("not json at all") == []


def test_validate_candidate_accepts_correct_shape():
    from seed_skill_store import _validate_candidate
    cand = {
        "name": "Summarize a paper",
        "description": "Extract key findings from a research paper.",
        "category": "research",
        "trigger_patterns": ["summarize paper", "tl;dr of pdf"],
        "procedure": "1. Fetch paper.\n2. Extract abstract.\n3. Distill to 3 bullets.",
    }
    ok, reason = _validate_candidate(cand)
    assert ok, f"expected valid, got: {reason}"


def test_validate_rejects_missing_fields():
    from seed_skill_store import _validate_candidate
    ok, reason = _validate_candidate({"name": "x"})
    assert not ok
    assert "missing fields" in reason


def test_validate_rejects_bad_category():
    from seed_skill_store import _validate_candidate
    cand = {
        "name": "Some valid name",
        "description": "y",
        "category": "bogus",
        "trigger_patterns": ["a"],
        "procedure": "a long enough procedure string here",
    }
    ok, reason = _validate_candidate(cand)
    assert not ok
    assert "category" in reason


def test_validate_rejects_empty_trigger_patterns():
    from seed_skill_store import _validate_candidate
    cand = {
        "name": "Some valid name",
        "description": "y",
        "category": "research",
        "trigger_patterns": [],
        "procedure": "a long enough procedure string here",
    }
    ok, reason = _validate_candidate(cand)
    assert not ok
    assert "trigger_patterns" in reason


def test_seed_dry_run_does_not_save(tmp_path):
    """Dry run should report accepted/rejected but NOT write to the store."""
    from seed_skill_store import seed

    fake_interactions = [
        {"content": f"user asked to summarize a paper about topic {i}"}
        for i in range(5)
    ]
    fake_llm_output = json.dumps([
        {
            "name": "Summarize a paper",
            "description": "Extract key findings from a paper.",
            "category": "research",
            "trigger_patterns": ["summarize paper", "tl;dr pdf"],
            "procedure": "1. Fetch.\n2. Extract abstract.\n3. Distill.",
        }
    ])

    with patch("seed_skill_store._fetch_recent_interactions", return_value=fake_interactions), \
         patch("seed_skill_store._call_llm", return_value=fake_llm_output):
        result = seed(model="fake", limit=5, dry_run=True, storage_path=str(tmp_path))

    assert result["status"] == "dry_run"
    assert len(result["accepted"]) == 1
    assert result["skills_added"] == 0
    # No index file should have been created in the storage path (the SkillStore
    # init does create its dirs, but no skill should have been saved).
    index_path = tmp_path / "index.json"
    if index_path.exists():
        assert json.loads(index_path.read_text()) == {}


def test_seed_no_signal_short_circuits(tmp_path):
    from seed_skill_store import seed
    with patch("seed_skill_store._fetch_recent_interactions", return_value=[]):
        result = seed(model="fake", limit=5, dry_run=False, storage_path=str(tmp_path))
    assert result["status"] == "no_signal"
    assert result["skills_added"] == 0


def test_seed_no_candidates_when_llm_returns_garbage(tmp_path):
    from seed_skill_store import seed
    with patch("seed_skill_store._fetch_recent_interactions", return_value=[{"content": "x"}]), \
         patch("seed_skill_store._call_llm", return_value="garbage not json"):
        result = seed(model="fake", limit=5, dry_run=True, storage_path=str(tmp_path))
    assert result["status"] == "no_candidates"
    assert result["skills_added"] == 0
