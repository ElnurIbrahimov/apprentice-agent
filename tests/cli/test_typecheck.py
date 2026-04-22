"""Tests for aura/tools/typecheck.py."""
from __future__ import annotations

import os
from unittest.mock import patch

from aura.tools.typecheck import (
    Diagnostic,
    TypecheckResult,
    _parse_mypy,
    _parse_pyright,
    _parse_tsc,
    detect_typechecker,
    typecheck_changed_files,
)


# ── Detection ─────────────────────────────────────────────────────────────

def test_detect_no_files_returns_none(tmp_path):
    assert detect_typechecker(str(tmp_path), []) == "none"


def test_detect_typescript_with_tsconfig(tmp_path):
    (tmp_path / "tsconfig.json").write_text("{}", encoding="utf-8")
    with patch("aura.tools.typecheck._has_tool", side_effect=lambda n: n in ("tsc",)):
        runner = detect_typechecker(str(tmp_path), ["src/foo.ts"])
    assert runner == "tsc"


def test_detect_typescript_needs_tsconfig(tmp_path):
    # No tsconfig.json means tsc is not used.
    runner = detect_typechecker(str(tmp_path), ["src/foo.ts"])
    assert runner == "none"


def test_detect_mypy_via_pyproject(tmp_path):
    (tmp_path / "pyproject.toml").write_text("[tool.mypy]\nstrict = true\n", encoding="utf-8")
    with patch("aura.tools.typecheck._has_tool", side_effect=lambda n: n == "mypy"):
        runner = detect_typechecker(str(tmp_path), ["src/foo.py"])
    assert runner == "mypy"


def test_detect_pyright_preferred_over_mypy(tmp_path):
    (tmp_path / "pyrightconfig.json").write_text("{}", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text("[tool.mypy]\n", encoding="utf-8")
    with patch("aura.tools.typecheck._has_tool", side_effect=lambda n: n in ("pyright", "mypy")):
        runner = detect_typechecker(str(tmp_path), ["src/foo.py"])
    assert runner == "pyright"


def test_detect_no_py_or_ts_returns_none(tmp_path):
    (tmp_path / "pyproject.toml").write_text("[tool.mypy]\n", encoding="utf-8")
    with patch("aura.tools.typecheck._has_tool", return_value=True):
        runner = detect_typechecker(str(tmp_path), ["README.md", "config.yaml"])
    # No .py/.ts files → nothing to check
    assert runner == "none"


# ── Parsers ───────────────────────────────────────────────────────────────

def test_parse_tsc_error():
    out = "src/foo.ts(10,5): error TS2322: Type 'string' is not assignable to type 'number'."
    diags = _parse_tsc(out)
    assert len(diags) == 1
    d = diags[0]
    assert d.file == "src/foo.ts"
    assert d.line == 10
    assert d.col == 5
    assert d.severity == "error"
    assert d.code == "TS2322"
    assert "not assignable" in d.message


def test_parse_mypy_error_with_code():
    out = "src/foo.py:42:10: error: Argument 1 has incompatible type [arg-type]"
    diags = _parse_mypy(out)
    assert len(diags) == 1
    d = diags[0]
    assert d.file == "src/foo.py"
    assert d.line == 42
    assert d.col == 10
    assert d.severity == "error"
    assert d.code == "arg-type"


def test_parse_mypy_ignores_notes():
    out = "src/foo.py:42: note: this is not an error"
    assert _parse_mypy(out) == []


def test_parse_pyright_error():
    out = "  src/foo.py:15:3 - error: Unknown identifier (reportUndefinedVariable)"
    diags = _parse_pyright(out)
    assert len(diags) == 1
    d = diags[0]
    assert d.file == "src/foo.py"
    assert d.line == 15
    assert d.col == 3
    assert d.severity == "error"


# ── Integration ───────────────────────────────────────────────────────────

def test_typecheck_no_changed_files_returns_success(tmp_path):
    r = typecheck_changed_files(str(tmp_path), [])
    assert r.runner == "none"
    assert r.success is True
    assert "no changed files" in r.skipped_reason


def test_typecheck_no_checker_returns_success(tmp_path):
    # No tsconfig, no pyproject — nothing detected.
    r = typecheck_changed_files(str(tmp_path), ["foo.py"])
    assert r.runner == "none"
    assert r.success is True


def test_typecheck_override_cmd_used(tmp_path):
    with patch("aura.tools.typecheck._run", return_value=(0, "all good", "", 0.05)) as m:
        r = typecheck_changed_files(
            str(tmp_path), ["foo.py"],
            override_cmd="echo checking",
        )
    assert r.runner == "custom"
    assert r.success is True
    # Override command should have received the files.
    args, _kwargs = m.call_args
    cmd = args[0]
    assert any("foo.py" in str(a) for a in cmd)
