"""Core unit tests for router, code_edit, mcp_client, repo_map."""

import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure aura package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ── router.py ──────────────────────────────────────────────────

from aura.core.router import classify_task, ModelRouter


def test_classify_task_code_gen():
    category, confidence = classify_task("implement a login feature")
    assert category == "code_gen"
    assert confidence > 0.0


def test_classify_task_reasoning():
    category, confidence = classify_task("explain how auth works")
    assert category == "reasoning"
    assert confidence > 0.0


def test_classify_task_fallback():
    category, confidence = classify_task("hello")
    # With embeddings available, "hello" may be classified as any category;
    # without embeddings, it falls back to "orchestrator" with confidence 0.0.
    # Both paths are valid — the key invariant is that a result is returned.
    assert isinstance(category, str) and len(category) > 0
    assert isinstance(confidence, float) and 0.0 <= confidence <= 1.0


def test_select_model_per_tier():
    router = ModelRouter("fast")
    model = router.select("code_gen")
    assert model == "qwen3-coder-next:cloud"


def test_select_agentic_with_prompt():
    router = ModelRouter("balanced")
    model = router.select_agentic("fix the bug in auth.py")
    # "fix" -> small_edit category, balanced tier
    assert model is not None
    assert isinstance(model, str)
    assert len(model) > 0


# ── code_edit.py ───────────────────────────────────────────────

from aura.tools.code_edit import CodeEditTool, _is_safe_path


def test_safe_path_blocks_system(tmp_path):
    import platform
    if platform.system() == "Windows":
        p = Path("C:/Windows/System32/test.py")
    else:
        p = Path("/etc/passwd")
    safe, reason = _is_safe_path(p)
    assert not safe
    assert "blocked" in reason.lower() or "System" in reason


def test_safe_path_blocks_node_modules(tmp_path):
    p = tmp_path / "node_modules" / "pkg" / "index.js"
    safe, reason = _is_safe_path(p)
    assert not safe
    assert "node_modules" in reason


def test_safe_path_allows_normal(tmp_path):
    p = tmp_path / "src" / "app.py"
    safe, _ = _is_safe_path(p)
    assert safe


def test_edit_exact_match(tmp_path):
    f = tmp_path / "hello.py"
    f.write_text("print('hello')\n", encoding="utf-8")
    tool = CodeEditTool(backup_enabled=False)
    result = tool.edit(str(f), "hello", "world")
    assert result["success"]
    assert f.read_text(encoding="utf-8") == "print('world')\n"


def test_edit_fuzzy_match(tmp_path):
    f = tmp_path / "app.py"
    f.write_text("def calculate_total(items):\n    return sum(items)\n", encoding="utf-8")
    tool = CodeEditTool(backup_enabled=False)
    # Slightly different old_string — fuzzy should kick in
    result = tool.edit(str(f), "def calculate_totl(items):\n    return sum(items)", "def calc_total(items):\n    return sum(items)")
    assert result["success"]
    content = f.read_text(encoding="utf-8")
    assert "calc_total" in content


def test_edit_dry_run(tmp_path):
    f = tmp_path / "app.py"
    original = "x = 1\n"
    f.write_text(original, encoding="utf-8")
    tool = CodeEditTool(backup_enabled=False)
    result = tool.edit(str(f), "x = 1", "x = 2", dry_run=True)
    assert result["success"]
    assert result.get("preview") is True
    assert f.read_text(encoding="utf-8") == original  # unchanged


def test_rollback_restores(tmp_path):
    f = tmp_path / "app.py"
    original = "a = 1\n"
    f.write_text(original, encoding="utf-8")
    tool = CodeEditTool(backup_enabled=True)
    tool.edit(str(f), "a = 1", "a = 2")
    assert "a = 2" in f.read_text(encoding="utf-8")
    result = tool.rollback(str(f))
    assert result["success"]
    assert f.read_text(encoding="utf-8") == original


# ── mcp_client.py ──────────────────────────────────────────────

from aura.core.mcp_client import MCPClientManager, MCPClientConnection


def test_list_all_tools_prefixes():
    mgr = MCPClientManager()
    conn = MagicMock(spec=MCPClientConnection)
    conn.tools = [
        {"name": "read", "description": "Read a file"},
        {"name": "write", "description": "Write a file"},
    ]
    mgr.connections["fs"] = conn
    tools = mgr.list_all_tools()
    assert len(tools) == 2
    assert tools[0]["name"] == "mcp_fs__read"
    assert tools[1]["name"] == "mcp_fs__write"


def test_call_tool_routes_correctly():
    mgr = MCPClientManager()
    conn = MagicMock(spec=MCPClientConnection)
    conn.call_tool.return_value = "ok"
    mgr.connections["myserver"] = conn
    result = mgr.call_tool("mcp_myserver__do_thing", {"arg": 1})
    conn.call_tool.assert_called_once_with("do_thing", {"arg": 1})
    assert result == "ok"


def test_call_tool_invalid_prefix():
    mgr = MCPClientManager()
    result = mgr.call_tool("bad_name", {})
    parsed = json.loads(result)
    assert "error" in parsed


def test_call_tool_missing_server():
    mgr = MCPClientManager()
    result = mgr.call_tool("mcp_ghost__tool", {})
    parsed = json.loads(result)
    assert "not connected" in parsed["error"]


def test_load_from_config_empty():
    mgr = MCPClientManager()
    mgr.load_from_config({})  # no crash
    mgr.load_from_config({"mcp_servers": {}})  # no crash
    assert len(mgr.connections) == 0


# ── repo_map.py ────────────────────────────────────────────────

from aura.core.repo_map import extract_symbols, _format_map, _collect_files, LANG_PATTERNS


def test_extract_symbols_python(tmp_path):
    f = tmp_path / "sample.py"
    f.write_text("class Foo:\n    def bar(self):\n        pass\n\ndef baz():\n    pass\n", encoding="utf-8")
    syms = extract_symbols(str(f))
    names = {s["name"] for s in syms}
    assert "Foo" in names
    assert "baz" in names


def test_format_map_token_budget():
    file_symbols = {
        f"file_{i}.py": [{"name": f"Cls{i}", "kind": "class", "line": 1}]
        for i in range(50)
    }
    full = _format_map(file_symbols, max_tokens=50000)
    small = _format_map(file_symbols, max_tokens=20)
    # Small budget should produce significantly less output
    assert len(small) < len(full)
    assert small.count("class") < 50


def test_collect_files_filters_extensions(tmp_path):
    # Create .py and .txt files
    (tmp_path / "app.py").write_text("x = 1", encoding="utf-8")
    (tmp_path / "lib.js").write_text("const x = 1", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("hello", encoding="utf-8")
    (tmp_path / "data.csv").write_text("a,b", encoding="utf-8")
    files = _collect_files(str(tmp_path))
    extensions = {os.path.splitext(f)[1] for f in files}
    assert ".txt" not in extensions
    assert ".csv" not in extensions
    # Should have at least the .py and .js files
    assert any(f.endswith(".py") for f in files)
    assert any(f.endswith(".js") for f in files)
