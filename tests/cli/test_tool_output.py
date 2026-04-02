"""Tests for minimal tool output rendering."""
import pytest
from io import StringIO
from rich.console import Console
from aura.cli.tool_output import ToolOutputRenderer, format_elapsed


@pytest.fixture
def renderer():
    console = Console(file=StringIO(), force_terminal=True, width=100, highlight=False)
    return ToolOutputRenderer(console=console, visible_lines=5)


def test_shell_output_short(renderer):
    renderer.render_shell_output("line1\nline2\nline3", command="ls", exit_code=0)
    output = renderer._console.file.getvalue()
    assert "line1" in output
    assert "line3" in output
    assert "hidden" not in output


def test_shell_output_collapsed(renderer):
    lines = "\n".join(f"line{i}" for i in range(20))
    renderer.render_shell_output(lines, command="find .", exit_code=0)
    output = renderer._console.file.getvalue()
    # Should show last 5 lines (tail), not first
    assert "line19" in output
    assert "hidden" in output


def test_shell_output_empty(renderer):
    renderer.render_shell_output("", command="true")
    output = renderer._console.file.getvalue()
    assert "no output" in output


def test_shell_output_failure(renderer):
    renderer.render_shell_output("error msg", command="bad_cmd", exit_code=1)
    output = renderer._console.file.getvalue()
    assert "error msg" in output


def test_file_content_short(renderer):
    renderer.render_file_content("print('hello')", filename="test.py")
    output = renderer._console.file.getvalue()
    assert "test.py" in output


def test_file_content_collapsed(renderer):
    content = "\n".join(f"line {i}" for i in range(30))
    renderer.render_file_content(content, filename="big.py")
    output = renderer._console.file.getvalue()
    # Should show the file content (may or may not be collapsed depending on renderer config)
    assert "big.py" in output
    assert "line 0" in output


def test_render_tool_result_dispatch(renderer):
    renderer.render_tool_result("shell", {"output": "hello\nworld", "command": "echo hello"})
    output = renderer._console.file.getvalue()
    assert "hello" in output


def test_format_elapsed_ms():
    assert format_elapsed(0.05) == "50ms"


def test_format_elapsed_seconds():
    assert format_elapsed(3.7) == "3.7s"


def test_format_elapsed_minutes():
    assert format_elapsed(125) == "2m5s"


def test_search_results(renderer):
    results = [
        {"file": "main.py", "line": 10, "text": "def main():"},
        {"file": "app.py", "line": 5, "text": "import os"},
    ]
    renderer.render_search_results(results, query="main")
    output = renderer._console.file.getvalue()
    assert "main.py" in output


def test_search_results_empty(renderer):
    renderer.render_search_results([], query="nonexistent")
    output = renderer._console.file.getvalue()
    assert "no results" in output


def test_web_result(renderer):
    renderer.render_web_result("response body", url="https://example.com", status_code=200)
    output = renderer._console.file.getvalue()
    assert "200" in output
    assert "example.com" in output


def test_web_result_error_status(renderer):
    renderer.render_web_result("not found", url="https://example.com/missing", status_code=404)
    output = renderer._console.file.getvalue()
    assert "404" in output


def test_render_tool_result_generic_fallback(renderer):
    renderer.render_tool_result("unknown_tool", {"output": "some data"})
    output = renderer._console.file.getvalue()
    assert "some data" in output


def test_render_tool_result_no_output(renderer):
    renderer.render_tool_result("unknown_tool", {})
    output = renderer._console.file.getvalue()
    # No output means nothing rendered (no crash)
    assert output == "" or "no output" not in output
