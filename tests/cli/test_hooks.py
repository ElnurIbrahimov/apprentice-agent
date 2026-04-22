"""Tests for hooks system."""
import pytest
import os
from aura.cli.hooks import Hook, HookEvent, HookManager, render_hooks_table
from rich.console import Console
from io import StringIO

def test_hook_events():
    assert HookEvent.PRE_TOOL_CALL == "pre_tool_call"
    assert len(HookEvent.ALL) >= 8

def test_add_hook():
    mgr = HookManager()
    h = mgr.add("post_edit", "echo edited", name="test_hook")
    assert h.name == "test_hook"
    assert h.event == "post_edit"

def test_add_invalid_event():
    mgr = HookManager()
    with pytest.raises(ValueError):
        mgr.add("invalid_event", "echo")

def test_remove_hook():
    mgr = HookManager()
    mgr.add("post_edit", "echo", name="removeme")
    assert mgr.remove("removeme")
    assert not mgr.remove("nonexistent")

def test_list_hooks():
    mgr = HookManager()
    mgr.add("post_edit", "echo a", name="h1")
    mgr.add("post_edit", "echo b", name="h2")
    assert len(mgr.list_hooks()) == 2

def test_fire_hook():
    mgr = HookManager()
    # Use a cross-platform command
    cmd = "echo hello" if os.name != "nt" else "echo hello"
    mgr.add("post_edit", cmd, name="echo_test")
    # wait=True — default is async now (returns empty list) so results
    # can be inspected synchronously when the caller actually needs them.
    results = mgr.fire("post_edit", {"tool_name": "edit_file"}, wait=True)
    assert len(results) == 1
    assert results[0]["success"]
    assert "hello" in results[0]["stdout"]

def test_fire_no_hooks():
    mgr = HookManager()
    results = mgr.fire("post_edit")
    assert results == []

def test_fire_timeout():
    mgr = HookManager()
    # Use python to sleep, works on all platforms
    cmd = "python -c \"import time; time.sleep(60)\""
    h = mgr.add("post_edit", cmd, name="slow")
    h.timeout = 1
    results = mgr.fire("post_edit", wait=True)
    assert len(results) == 1
    assert not results[0]["success"]
    assert "Timeout" in results[0].get("error", "")

def test_load_from_config():
    mgr = HookManager()
    config = {
        "hooks": [
            {"event": "post_edit", "command": "echo lint", "name": "linter"},
            {"event": "post_response", "command": "echo done"},
        ]
    }
    count = mgr.load_from_config(config)
    assert count == 2

def test_load_builtin_hooks():
    mgr = HookManager()
    mgr.load_builtin_hooks({"lint_cmd": "ruff check .", "test_cmd": "pytest", "auto_test": True})
    hooks = mgr.list_hooks()
    assert any(h.name == "auto_lint" for h in hooks)
    assert any(h.name == "auto_test" for h in hooks)

def test_clear():
    mgr = HookManager()
    mgr.add("post_edit", "echo", name="test")
    mgr.clear()
    assert len(mgr.list_hooks()) == 0

def test_render_hooks_table():
    console = Console(file=StringIO(), force_terminal=True, width=100)
    hooks = [Hook(event="post_edit", command="echo lint", name="linter")]
    render_hooks_table(console, hooks)
    assert "linter" in console.file.getvalue()
