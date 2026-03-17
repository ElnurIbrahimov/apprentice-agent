"""Tests for Unix pipe mode."""
import pytest
import json
from io import StringIO
from unittest.mock import patch
from aura.cli.pipe_mode import PipeOutput, is_pipe_mode, read_piped_input, EXIT_SUCCESS, EXIT_ERROR

def test_pipe_output_text():
    out = StringIO()
    with patch("sys.stdout", out):
        po = PipeOutput(format="text")
        po.content("hello world")
    assert out.getvalue() == "hello world"

def test_pipe_output_json():
    out = StringIO()
    with patch("sys.stdout", out):
        po = PipeOutput(format="json")
        po.result({"response": "hello", "model": "test"})
    parsed = json.loads(out.getvalue())
    assert parsed["response"] == "hello"

def test_pipe_output_markdown():
    out = StringIO()
    with patch("sys.stdout", out):
        po = PipeOutput(format="markdown")
        po.result({"response": "# Hello\n\nWorld"})
    assert out.getvalue().strip() == "# Hello\n\nWorld"

def test_pipe_status_to_stderr():
    err = StringIO()
    with patch("sys.stderr", err):
        po = PipeOutput()
        po.status("processing...")
    assert "processing" in err.getvalue()

def test_pipe_error_to_stderr():
    err = StringIO()
    with patch("sys.stderr", err):
        po = PipeOutput()
        po.error("something failed")
    assert "error: something failed" in err.getvalue()

def test_is_pipe_mode_tty():
    with patch("sys.stdin") as mock_in, patch("sys.stdout") as mock_out:
        mock_in.isatty.return_value = True
        mock_out.isatty.return_value = True
        assert not is_pipe_mode()

def test_is_pipe_mode_piped():
    # Ensure TERM is unset so the MSYS2 heuristic doesn't activate
    with patch("sys.stdin") as mock_in, patch("sys.stdout") as mock_out, \
         patch.dict("os.environ", {}, clear=True):
        mock_in.isatty.return_value = False
        mock_out.isatty.return_value = True
        assert is_pipe_mode()

def test_read_piped_input():
    mock_stdin = StringIO("hello from pipe")
    mock_stdin.isatty = lambda: False
    with patch("sys.stdin", mock_stdin):
        result = read_piped_input()
    assert result == "hello from pipe"

def test_read_piped_input_empty():
    mock_stdin = StringIO("")
    mock_stdin.isatty = lambda: False
    with patch("sys.stdin", mock_stdin):
        result = read_piped_input()
    assert result is None

def test_exit_codes():
    assert EXIT_SUCCESS == 0
    assert EXIT_ERROR == 1
