"""Tests for /why — Intent-to-Code Ledger v1."""
from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

from aura.cli.commands.why_commands import (
    _group_and_enrich,
    _normalize_for_match,
    _read_edit_records,
    _resolve_path,
    _split_line_suffix,
    handle_why,
)


# ── line-suffix parser ──────────────────────────────────────────────────────

def test_split_line_suffix_no_colon():
    assert _split_line_suffix("foo.py") == ("foo.py", None)


def test_split_line_suffix_with_line():
    assert _split_line_suffix("foo.py:42") == ("foo.py", 42)


def test_split_line_suffix_with_non_numeric_suffix():
    # "foo.py:bar" → no integer line, treat as plain path.
    assert _split_line_suffix("foo.py:bar") == ("foo.py:bar", None)


def test_split_line_suffix_windows_drive_only():
    assert _split_line_suffix("C:\\src\\foo.py") == ("C:\\src\\foo.py", None)


def test_split_line_suffix_windows_drive_and_line():
    path, line = _split_line_suffix("C:\\src\\foo.py:42")
    assert path == "C:\\src\\foo.py"
    assert line == 42


# ── record reader ───────────────────────────────────────────────────────────

def test_read_edit_records_no_log_file(tmp_path):
    with patch.dict(os.environ, {"AURA_DATA_DIR": str(tmp_path)}):
        assert _read_edit_records("/anything.py") == []


def test_read_edit_records_filters_by_path(tmp_path):
    log_dir = tmp_path / "events"
    log_dir.mkdir()
    # Write 3 records: 2 for foo.py, 1 for bar.py.
    foo = str((tmp_path / "foo.py").absolute())
    bar = str((tmp_path / "bar.py").absolute())
    with (log_dir / "edits.jsonl").open("w", encoding="utf-8") as f:
        for rec in (
            {"ts": 1.0, "session_id": "s1", "tool": "edit_file", "path": foo, "prompt": "a", "model": "m", "iteration": 1},
            {"ts": 2.0, "session_id": "s1", "tool": "edit_file", "path": bar, "prompt": "b", "model": "m", "iteration": 2},
            {"ts": 3.0, "session_id": "s2", "tool": "write_file", "path": foo, "prompt": "c", "model": "m", "iteration": 1},
        ):
            f.write(json.dumps(rec) + "\n")

    with patch.dict(os.environ, {"AURA_DATA_DIR": str(tmp_path)}):
        foo_records = _read_edit_records(foo)
        bar_records = _read_edit_records(bar)

    assert len(foo_records) == 2
    assert len(bar_records) == 1
    assert all(r["path"] == foo for r in foo_records)


def test_read_edit_records_skips_malformed_lines(tmp_path):
    log_dir = tmp_path / "events"
    log_dir.mkdir()
    foo = str((tmp_path / "foo.py").absolute())
    with (log_dir / "edits.jsonl").open("w", encoding="utf-8") as f:
        f.write("not json\n")
        f.write("\n")
        f.write(json.dumps({
            "ts": 1.0, "session_id": "s1", "tool": "edit_file",
            "path": foo, "prompt": "a", "model": "m", "iteration": 1,
        }) + "\n")

    with patch.dict(os.environ, {"AURA_DATA_DIR": str(tmp_path)}):
        records = _read_edit_records(foo)
    assert len(records) == 1


# ── grouping / enrichment ───────────────────────────────────────────────────

def test_group_and_enrich_collapses_same_session():
    records = [
        {"ts": 1.0, "session_id": "s1", "model": "m1", "prompt": "do the thing", "iteration": 1, "path": "x"},
        {"ts": 2.0, "session_id": "s1", "model": "m1", "prompt": "do the thing", "iteration": 2, "path": "x"},
        {"ts": 3.0, "session_id": "s1", "model": "m1", "prompt": "do the thing", "iteration": 3, "path": "x"},
    ]
    groups = _group_and_enrich(records)
    assert len(groups) == 1
    g = groups[0]
    assert g["edit_count"] == 3
    assert g["iterations"] == {1, 2, 3}
    assert g["first_ts"] == 1.0
    assert g["last_ts"] == 3.0
    assert g["prompt"] == "do the thing"


def test_group_and_enrich_orders_newest_first():
    records = [
        {"ts": 1.0, "session_id": "s1", "model": "m", "prompt": "older", "iteration": 1, "path": "x"},
        {"ts": 5.0, "session_id": "s2", "model": "m", "prompt": "newer", "iteration": 1, "path": "x"},
        {"ts": 3.0, "session_id": "s3", "model": "m", "prompt": "middle", "iteration": 1, "path": "x"},
    ]
    groups = _group_and_enrich(records)
    assert [g["session_id"] for g in groups] == ["s2", "s3", "s1"]


def test_group_and_enrich_prefers_longest_prompt():
    """Multiple records in one session may have differently-truncated prompts."""
    records = [
        {"ts": 1.0, "session_id": "s1", "model": "m", "prompt": "short", "iteration": 1, "path": "x"},
        {"ts": 2.0, "session_id": "s1", "model": "m", "prompt": "this is a longer prompt", "iteration": 2, "path": "x"},
    ]
    groups = _group_and_enrich(records)
    assert groups[0]["prompt"] == "this is a longer prompt"


# ── handle_why — end-to-end ────────────────────────────────────────────────

def test_handle_why_empty_arg_shows_usage(capsys):
    handle_why(MagicMock(), "", {})
    assert "Usage" in capsys.readouterr().out


def test_handle_why_no_records_prints_graceful_message(tmp_path, capsys):
    (tmp_path / "events").mkdir()
    with patch.dict(os.environ, {"AURA_DATA_DIR": str(tmp_path)}), \
         patch("aura.cli.commands.why_commands._git_blame_summary", return_value=None):
        handle_why(MagicMock(), "does_not_exist.py", {})
    assert "No Aura edit history" in capsys.readouterr().out


def test_handle_why_renders_records(tmp_path, capsys):
    log_dir = tmp_path / "events"
    log_dir.mkdir()
    foo = str((tmp_path / "foo.py").absolute())
    with (log_dir / "edits.jsonl").open("w", encoding="utf-8") as f:
        f.write(json.dumps({
            "ts": 1700000000.0, "session_id": "sess123abc",
            "tool": "edit_file", "path": foo,
            "prompt": "refactor the login flow to use JWT",
            "model": "kimi-k2.6:cloud", "iteration": 3,
        }) + "\n")

    with patch.dict(os.environ, {"AURA_DATA_DIR": str(tmp_path)}), \
         patch("aura.cli.commands.why_commands._git_blame_summary", return_value=None), \
         patch("aura.cli.commands.why_commands._lookup_session_prompt", return_value=""):
        handle_why(MagicMock(), foo, {})

    output = capsys.readouterr().out
    assert "Edit history" in output
    assert "1 session" in output
    assert "kimi-k2.6" in output
    assert "refactor the login flow" in output


def test_handle_why_line_suffix_shows_v2_note(tmp_path, capsys):
    (tmp_path / "events").mkdir()
    with patch.dict(os.environ, {"AURA_DATA_DIR": str(tmp_path)}), \
         patch("aura.cli.commands.why_commands._git_blame_summary", return_value=None):
        handle_why(MagicMock(), "foo.py:42", {})
    # v2 note should appear even when no records exist.
    assert "line-level /why is v2" in capsys.readouterr().out


# ── path normalization ────────────────────────────────────────────────────

def test_normalize_for_match_case_folds_on_windows(tmp_path):
    if os.name != "nt":
        return  # Only matters on Windows.
    a = str((tmp_path / "Foo.py").absolute())
    b = str((tmp_path / "foo.py").absolute())
    assert _normalize_for_match(a) == _normalize_for_match(b)


def test_normalize_for_match_uses_forward_slashes():
    normalized = _normalize_for_match("a/b/c.py")
    assert "\\" not in normalized
