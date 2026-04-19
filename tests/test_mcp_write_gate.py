"""Tests for MCP server write-tool opt-in and resources."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from aura.core import mcp_server
from aura.core.mcp_server import (
    AuraMCPServer,
    _allow_writes,
    _current_excluded_tools,
    _path_allowed,
    _write_allowlist,
    _WRITE_TOOLS,
)


def test_allow_writes_env_parsing(monkeypatch):
    monkeypatch.delenv("AURA_MCP_ALLOW_WRITES", raising=False)
    assert not _allow_writes()
    for truthy in ("1", "true", "TRUE", "yes", "on"):
        monkeypatch.setenv("AURA_MCP_ALLOW_WRITES", truthy)
        assert _allow_writes()
    for falsy in ("0", "false", "no", "", "random"):
        monkeypatch.setenv("AURA_MCP_ALLOW_WRITES", falsy)
        assert not _allow_writes()


def test_excluded_tools_default_blocks_writes(monkeypatch):
    monkeypatch.delenv("AURA_MCP_ALLOW_WRITES", raising=False)
    excluded = _current_excluded_tools()
    assert _WRITE_TOOLS.issubset(excluded)


def test_excluded_tools_opt_in_allows_writes(monkeypatch):
    monkeypatch.setenv("AURA_MCP_ALLOW_WRITES", "true")
    excluded = _current_excluded_tools()
    assert not _WRITE_TOOLS.issubset(excluded)


def test_always_excluded_never_exposed_even_with_opt_in(monkeypatch):
    monkeypatch.setenv("AURA_MCP_ALLOW_WRITES", "true")
    excluded = _current_excluded_tools()
    assert "shell" in excluded
    assert "spawn_agent" in excluded
    assert "git_push" in excluded


def test_write_allowlist_empty_allows_all(monkeypatch, tmp_path):
    monkeypatch.delenv("AURA_MCP_WRITE_ALLOWLIST", raising=False)
    allowlist = _write_allowlist()
    assert allowlist == []
    # Empty allowlist = _path_allowed returns True for anything
    assert _path_allowed(str(tmp_path / "anywhere.txt"), allowlist)


def test_write_allowlist_restricts(monkeypatch, tmp_path):
    allowed_dir = tmp_path / "project"
    allowed_dir.mkdir()
    monkeypatch.setenv("AURA_MCP_WRITE_ALLOWLIST", str(allowed_dir))
    allowlist = _write_allowlist()

    # File inside the allowed dir
    assert _path_allowed(str(allowed_dir / "file.txt"), allowlist)
    # File outside
    outside = tmp_path / "other.txt"
    assert not _path_allowed(str(outside), allowlist)


def test_server_tools_list_excludes_writes_by_default(monkeypatch, tmp_path):
    monkeypatch.delenv("AURA_MCP_ALLOW_WRITES", raising=False)
    server = AuraMCPServer(project_root=str(tmp_path))
    tool_names = {t["name"] for t in server.tools}
    assert "write_file" not in tool_names
    assert "edit_file" not in tool_names
    assert "shell" not in tool_names  # always excluded


def test_server_tools_list_includes_writes_when_enabled(monkeypatch, tmp_path):
    monkeypatch.setenv("AURA_MCP_ALLOW_WRITES", "true")
    server = AuraMCPServer(project_root=str(tmp_path))
    tool_names = {t["name"] for t in server.tools}
    # At least one of the write tools should appear when opt-in
    assert any(name in tool_names for name in _WRITE_TOOLS)


def test_resources_list_skips_ignored_dirs(monkeypatch, tmp_path):
    # Set up a fake project with a .git dir (should be ignored)
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("x", encoding="utf-8")
    (tmp_path / "readme.md").write_text("hello", encoding="utf-8")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "pkg.json").write_text("{}", encoding="utf-8")

    monkeypatch.delenv("AURA_MCP_ALLOW_WRITES", raising=False)
    server = AuraMCPServer(project_root=str(tmp_path))

    captured = []
    server._write = lambda data: captured.append(data)
    server._handle_resources_list(1, {})
    resources = captured[0]["result"]["resources"]
    names = {r["name"] for r in resources}
    assert "readme.md" in names
    assert not any(".git" in n or "node_modules" in n for n in names)


def test_resources_read_rejects_outside_project(monkeypatch, tmp_path):
    (tmp_path / "a.txt").write_text("hi", encoding="utf-8")
    server = AuraMCPServer(project_root=str(tmp_path))

    captured = []
    server._write = lambda data: captured.append(data)
    # Absolute path outside project root
    other = tmp_path.parent / "outside.txt"
    other.write_text("x", encoding="utf-8")
    server._handle_resources_read(1, {"uri": f"file://{other.as_posix()}"})
    assert "error" in captured[0]


def test_resources_read_allows_inside_project(monkeypatch, tmp_path):
    target = tmp_path / "a.txt"
    target.write_text("hello world", encoding="utf-8")
    server = AuraMCPServer(project_root=str(tmp_path))

    captured = []
    server._write = lambda data: captured.append(data)
    server._handle_resources_read(1, {"uri": f"file://{target.as_posix()}"})
    assert "result" in captured[0]
    assert "hello world" in captured[0]["result"]["contents"][0]["text"]
