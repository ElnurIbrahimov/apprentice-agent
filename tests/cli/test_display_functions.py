"""Tests for display.py — show_error, show_info, show_tool_call, show_response, helpers."""
import pytest
from unittest.mock import patch, MagicMock
from rich.text import Text


# ── show_error ────────────────────────────────────────────────────────────

@patch("aura.cli.display.console")
def test_show_error_basic(mock_console):
    from aura.cli.display import show_error
    show_error("something went wrong")
    mock_console.print.assert_called_once()
    arg = mock_console.print.call_args[0][0]
    assert "something went wrong" in str(arg)


@patch("aura.cli.display.console")
def test_show_error_empty_string(mock_console):
    from aura.cli.display import show_error
    show_error("")
    mock_console.print.assert_called_once()


@patch("aura.cli.display.console")
def test_show_error_special_chars(mock_console):
    from aura.cli.display import show_error
    show_error("Error: <tag> & 'quotes' \"double\" \nnewline")
    mock_console.print.assert_called_once()


@patch("aura.cli.display.console")
def test_show_error_very_long_string(mock_console):
    from aura.cli.display import show_error
    show_error("x" * 10000)
    mock_console.print.assert_called_once()


@patch("aura.cli.display.console")
def test_show_error_unicode(mock_console):
    from aura.cli.display import show_error
    show_error("Ошибка: что-то пошло не так 🔥")
    mock_console.print.assert_called_once()


# ── show_info ─────────────────────────────────────────────────────────────

@patch("aura.cli.display.console")
def test_show_info_basic(mock_console):
    from aura.cli.display import show_info
    show_info("indexing complete")
    mock_console.print.assert_called_once()
    arg = mock_console.print.call_args[0][0]
    assert "indexing complete" in str(arg)


@patch("aura.cli.display.console")
def test_show_info_empty_string(mock_console):
    from aura.cli.display import show_info
    show_info("")
    mock_console.print.assert_called_once()


@patch("aura.cli.display.console")
def test_show_info_special_chars(mock_console):
    from aura.cli.display import show_info
    show_info("File: C:\\Users\\test\\path with spaces & symbols <ok>")
    mock_console.print.assert_called_once()


# ── show_tool_call ────────────────────────────────────────────────────────

@patch("aura.cli.display.console")
def test_show_tool_call_basic(mock_console):
    from aura.cli.display import show_tool_call
    show_tool_call("read_file")
    mock_console.print.assert_called_once()
    arg = mock_console.print.call_args[0][0]
    # Tool name is mapped to a human label (e.g. "read_file" -> "Read")
    arg_text = arg.plain if hasattr(arg, "plain") else str(arg)
    assert "Read" in arg_text


@patch("aura.cli.display.console")
def test_show_tool_call_with_description(mock_console):
    from aura.cli.display import show_tool_call
    show_tool_call("edit_file", description="src/main.py")
    mock_console.print.assert_called_once()
    arg = mock_console.print.call_args[0][0]
    arg_text = arg.plain if hasattr(arg, "plain") else str(arg)
    assert "Edit" in arg_text
    assert "src/main.py" in arg_text


@patch("aura.cli.display.console")
def test_show_tool_call_with_elapsed(mock_console):
    from aura.cli.display import show_tool_call
    show_tool_call("web_search", description="query", elapsed=1.5)
    mock_console.print.assert_called_once()
    arg = mock_console.print.call_args[0][0]
    arg_text = arg.plain if hasattr(arg, "plain") else str(arg)
    assert "Web" in arg_text


@patch("aura.cli.display.console")
def test_show_tool_call_zero_elapsed(mock_console):
    from aura.cli.display import show_tool_call
    show_tool_call("run_command", elapsed=0.0)
    mock_console.print.assert_called_once()


@patch("aura.cli.display.console")
def test_show_tool_call_empty_name(mock_console):
    from aura.cli.display import show_tool_call
    show_tool_call("")
    mock_console.print.assert_called_once()


@patch("aura.cli.display.console")
def test_show_tool_call_with_dict_result_no_crash(mock_console):
    """Passing a result dict should not crash even if it has no output."""
    from aura.cli.display import show_tool_call
    show_tool_call("read_file", result={"path": "/tmp/x.py", "error": "not found"})
    # Should not raise — the error key prevents ToolOutputRenderer from being called
    assert mock_console.print.called


@patch("aura.cli.display.console")
def test_show_tool_call_with_string_result_invalid_json(mock_console):
    """Non-JSON string result should be silently ignored."""
    from aura.cli.display import show_tool_call
    show_tool_call("shell", result="some plain text output")
    assert mock_console.print.called


# ── show_response ─────────────────────────────────────────────────────────

@patch("aura.cli.display.console")
def test_show_response_non_streaming(mock_console):
    mock_console.width = 120
    from aura.cli.display import show_response
    show_response("Hello world", model="test-model", stream=False)
    assert mock_console.print.call_count >= 1


@patch("aura.cli.display.console")
def test_show_response_empty_text(mock_console):
    mock_console.width = 120
    from aura.cli.display import show_response
    show_response("", stream=False)
    # Should still call print (renders empty panel)
    assert mock_console.print.called


@patch("aura.cli.display.console")
def test_show_response_short_text_no_stream(mock_console):
    """Short text (<=20 chars) disables streaming even if stream=True."""
    mock_console.width = 120
    from aura.cli.display import show_response
    show_response("short", model="m", stream=True)
    # With stream=True but text <= 20 chars, it falls through to non-streaming
    assert mock_console.print.called


@patch("aura.cli.display.console")
def test_show_response_no_model(mock_console):
    mock_console.width = 120
    from aura.cli.display import show_response
    show_response("Some markdown **bold** text", stream=False)
    assert mock_console.print.called


@patch("aura.cli.display.console")
def test_show_response_markdown_content(mock_console):
    """Markdown with code blocks should render without crashing."""
    mock_console.width = 120
    from aura.cli.display import show_response
    text = "# Heading\n\n```python\nprint('hello')\n```\n\n- item 1\n- item 2"
    show_response(text, stream=False)
    assert mock_console.print.called


# ── get_thinking_label ────────────────────────────────────────────────────

def test_get_thinking_label_known_tool():
    from aura.cli.display import get_thinking_label
    assert get_thinking_label("web_search") == "Searching the web"
    assert get_thinking_label("read_file") == "Reading files"
    assert get_thinking_label("edit_file") == "Editing code"
    assert get_thinking_label("execute") == "Running code"


def test_get_thinking_label_unknown_tool():
    from aura.cli.display import get_thinking_label
    assert get_thinking_label("unknown_tool") == "Thinking..."


def test_get_thinking_label_none():
    from aura.cli.display import get_thinking_label
    assert get_thinking_label(None) == "Thinking..."


def test_get_thinking_label_empty_string():
    from aura.cli.display import get_thinking_label
    assert get_thinking_label("") == "Thinking..."


# ── _split_for_streaming ─────────────────────────────────────────────────

def test_split_for_streaming_short_text():
    from aura.cli.display import _split_for_streaming
    chunks = _split_for_streaming("hello world")
    assert len(chunks) >= 1
    assert "".join(chunks) == "hello world"


def test_split_for_streaming_empty():
    from aura.cli.display import _split_for_streaming
    chunks = _split_for_streaming("")
    assert chunks == [""]


def test_split_for_streaming_single_word():
    from aura.cli.display import _split_for_streaming
    chunks = _split_for_streaming("hello")
    assert "".join(chunks) == "hello"


def test_split_for_streaming_longer_text():
    from aura.cli.display import _split_for_streaming
    text = " ".join(["word"] * 300)
    chunks = _split_for_streaming(text)
    assert "".join(chunks) == text


# ── _split_into_blocks ───────────────────────────────────────────────────

def test_split_into_blocks_single_paragraph():
    from aura.cli.display import _split_into_blocks
    blocks = _split_into_blocks("Hello world")
    assert len(blocks) == 1
    assert blocks[0] == "Hello world"


def test_split_into_blocks_multiple_paragraphs():
    from aura.cli.display import _split_into_blocks
    text = "First paragraph.\n\nSecond paragraph."
    blocks = _split_into_blocks(text)
    assert len(blocks) == 2


def test_split_into_blocks_code_fence_stays_together():
    from aura.cli.display import _split_into_blocks
    text = "Before\n\n```python\nprint('a')\n\nprint('b')\n```\n\nAfter"
    blocks = _split_into_blocks(text)
    # The code fence block should remain as one block, not split on blank line inside
    code_blocks = [b for b in blocks if "```" in b]
    assert len(code_blocks) == 1
    assert "print('a')" in code_blocks[0]
    assert "print('b')" in code_blocks[0]


def test_split_into_blocks_empty_text():
    from aura.cli.display import _split_into_blocks
    blocks = _split_into_blocks("")
    assert blocks == []


def test_split_into_blocks_whitespace_only():
    from aura.cli.display import _split_into_blocks
    blocks = _split_into_blocks("   \n\n   ")
    assert blocks == []


# ── show_context_summary ─────────────────────────────────────────────────

@patch("aura.cli.display.console")
def test_show_context_summary_with_memories(mock_console):
    from aura.cli.display import show_context_summary
    show_context_summary(memory_count=3, mood="curious", model="devstral-2")
    assert mock_console.print.called


@patch("aura.cli.display.console")
def test_show_context_summary_empty(mock_console):
    from aura.cli.display import show_context_summary
    show_context_summary()
    # No parts, should not print
    mock_console.print.assert_not_called()


@patch("aura.cli.display.console")
def test_show_context_summary_with_snippets(mock_console):
    from aura.cli.display import show_context_summary
    show_context_summary(
        memory_count=2,
        memory_snippets=["prefers Python", "working on BroadMind", "uses RTX 4060", "extra"],
    )
    # Context summary prints once (snippets are accepted but not rendered separately)
    assert mock_console.print.call_count >= 1


@patch("aura.cli.display.console")
def test_show_context_summary_long_model_truncated(mock_console):
    from aura.cli.display import show_context_summary
    show_context_summary(model="a-very-long-model-name-that-exceeds-25-characters")
    assert mock_console.print.called


# ── show_rewind_result ───────────────────────────────────────────────────

@patch("aura.cli.display.console")
def test_show_rewind_result_success(mock_console):
    from aura.cli.display import show_rewind_result
    show_rewind_result(True, "abc123")
    arg = mock_console.print.call_args[0][0]
    arg_text = arg.plain if hasattr(arg, "plain") else str(arg)
    assert "abc123" in arg_text


@patch("aura.cli.display.console")
def test_show_rewind_result_failure(mock_console):
    from aura.cli.display import show_rewind_result
    show_rewind_result(False, "abc123")
    # show_rewind_result(False) calls show_error() which prints a string
    arg = mock_console.print.call_args[0][0]
    arg_text = arg.plain if hasattr(arg, "plain") else str(arg)
    assert "Failed" in arg_text or "fail" in arg_text.lower()


# ── StreamingResponse ────────────────────────────────────────────────────

def test_streaming_response_text_property():
    from aura.cli.display import StreamingResponse
    sr = StreamingResponse(model="test")
    assert sr.text == ""


def test_streaming_response_chunk_accumulates():
    from aura.cli.display import StreamingResponse
    sr = StreamingResponse()
    sr.chunk("hello ")
    sr.chunk("world")
    assert sr.text == "hello world"


def test_streaming_response_pause_resume_no_crash():
    from aura.cli.display import StreamingResponse
    sr = StreamingResponse()
    sr.pause()  # No live context — should not crash
    assert sr._live is None


def test_streaming_response_finish_no_start():
    """Finishing without starting should not crash."""
    from aura.cli.display import StreamingResponse
    sr = StreamingResponse()
    sr.finish()  # No live, no accumulated text — safe


# ── Fence state across pause/resume ──────────────────────────────────────

def test_fence_state_initial_is_false():
    from aura.cli.display import StreamingResponse
    sr = StreamingResponse()
    assert sr._in_fence is False


def test_wrap_for_render_appends_close_when_ending_mid_fence():
    from aura.cli.display import StreamingResponse
    sr = StreamingResponse()
    # Entering not in fence, opens but doesn't close → append close only
    rendered = sr._wrap_for_render("```python\nx = 1")
    assert rendered.endswith("\n```")
    assert rendered.startswith("```python")


def test_wrap_for_render_prepends_open_when_starting_mid_fence():
    from aura.cli.display import StreamingResponse
    sr = StreamingResponse()
    sr._in_fence = True  # Simulate: we're mid-block, prior chunks opened a fence
    # Nothing in new_content closes — wrap must open AND close
    rendered = sr._wrap_for_render("y = 2")
    assert rendered.startswith("```\n")
    assert rendered.endswith("\n```")


def test_wrap_for_render_inside_fence_that_closes():
    from aura.cli.display import StreamingResponse
    sr = StreamingResponse()
    sr._in_fence = True
    # new_content contains the closing fence — prepend open, no synthetic close
    rendered = sr._wrap_for_render("more code\n```")
    assert rendered.startswith("```\n")
    assert not rendered.endswith("\n```\n```")


def test_advance_fence_state_toggles_on_fence_line():
    from aura.cli.display import StreamingResponse
    sr = StreamingResponse()
    assert sr._in_fence is False
    sr._advance_fence_state("```python\ncode")
    assert sr._in_fence is True
    sr._advance_fence_state("more\n```")
    assert sr._in_fence is False


def test_pause_advances_fence_state_so_resume_knows_it_is_mid_block():
    """The scenario the fix targets: chunk opens a fence, pause fires for a
    tool call, resume+chunk continues the block. Without state tracking,
    the post-resume chunk would render as prose, not code."""
    from aura.cli.display import StreamingResponse

    sr = StreamingResponse()
    sr._accumulated = "```python\nx = 1"
    sr._permanent_len = 0
    # Pause without a live context skips the render path but still needs to
    # advance fence state so resume starts correctly. Call the helper
    # directly to mirror what pause() does.
    sr._advance_fence_state(sr._accumulated[sr._permanent_len:])
    assert sr._in_fence is True, "pause should leave us flagged mid-fence"

    # Next render (after resume) gets only the continuation — _wrap_for_render
    # must prepend an opener so Rich sees the code block.
    continuation = "\ny = 2\n```"
    rendered = sr._wrap_for_render(continuation)
    assert rendered.startswith("```\n"), "continuation needs a synthetic opener"
