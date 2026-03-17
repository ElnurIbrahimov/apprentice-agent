"""Tests for progressive disclosure."""
import pytest
from aura.cli.disclosure import (
    CollapsibleSection, DisclosureManager,
    render_collapsed, render_expanded, render_section,
    create_section_from_tool_call,
)
from rich.console import Console
from io import StringIO

def test_section_defaults():
    s = CollapsibleSection(id="s1", title="test", content="full content")
    assert not s.expanded
    assert s.display == "test"

def test_section_expanded():
    s = CollapsibleSection(id="s1", title="test", content="full", expanded=True)
    assert s.display == "full"

def test_manager_add_and_get():
    mgr = DisclosureManager()
    s = mgr.add_section("s1", "title", "content", tool_name="shell")
    assert mgr.get_section("s1") is s
    assert mgr.section_count == 1

def test_manager_toggle():
    mgr = DisclosureManager()
    mgr.add_section("s1", "title", "content")
    assert mgr.toggle("s1") == True  # was False, now True
    assert mgr.toggle("s1") == False  # back to False

def test_manager_verbose():
    mgr = DisclosureManager()
    mgr.set_verbose(True)
    s = mgr.add_section("s1", "title", "content")
    assert s.expanded  # verbose = default expanded

def test_manager_expand_collapse_all():
    mgr = DisclosureManager()
    mgr.add_section("s1", "a", "aa")
    mgr.add_section("s2", "b", "bb")
    mgr.expand_all()
    assert all(s.expanded for s in mgr.get_recent())
    mgr.collapse_all()
    assert all(not s.expanded for s in mgr.get_recent())

def test_manager_max_sections():
    mgr = DisclosureManager()
    for i in range(60):
        mgr.add_section(f"s{i}", f"title{i}", f"content{i}")
    assert mgr.section_count == 50

def test_render_collapsed():
    console = Console(file=StringIO(), force_terminal=True, width=80)
    s = CollapsibleSection(id="s1", title="ls output", content="file1\nfile2", tool_name="shell")
    render_collapsed(console, s)
    output = console.file.getvalue()
    assert "shell" in output
    assert "expand" in output.lower()

def test_render_expanded():
    console = Console(file=StringIO(), force_terminal=True, width=80)
    s = CollapsibleSection(id="s1", title="ls output", content="file1\nfile2", tool_name="shell", expanded=True)
    render_expanded(console, s)
    output = console.file.getvalue()
    assert "file1" in output
    assert "file2" in output

def test_create_section_shell():
    s = create_section_from_tool_call(
        "shell", {"command": "ls -la"}, {"output": "file1\nfile2", "exit_code": 0}, elapsed=0.5
    )
    assert "ls -la" in s.title
    assert s.tool_name == "shell"

def test_create_section_edit():
    s = create_section_from_tool_call(
        "edit_file", {"path": "/project/main.py"}, {"output": "ok"}, elapsed=0.1
    )
    assert "main.py" in s.title

def test_create_section_search():
    s = create_section_from_tool_call(
        "grep", {"pattern": "TODO"}, {"output": "main.py:10: TODO fix"}, elapsed=0.2
    )
    assert "TODO" in s.title

def test_clear():
    mgr = DisclosureManager()
    mgr.add_section("s1", "a", "b")
    mgr.clear()
    assert mgr.section_count == 0
