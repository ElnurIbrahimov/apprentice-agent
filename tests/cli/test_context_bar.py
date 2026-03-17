"""Tests for context bar — token estimation and gauge rendering."""
import pytest
from aura.cli.context_bar import (
    estimate_tokens, estimate_messages_tokens, format_token_count,
    build_context_gauge, build_context_breakdown, get_context_limit,
)

def test_estimate_tokens_empty():
    assert estimate_tokens("") == 0

def test_estimate_tokens_simple():
    result = estimate_tokens("hello world this is a test")
    assert 5 <= result <= 10

def test_estimate_tokens_code():
    code = "def hello():\n    return 'world'\n"
    result = estimate_tokens(code)
    assert result > 0

def test_estimate_messages_tokens():
    msgs = [
        {"role": "user", "content": "hello world"},
        {"role": "assistant", "content": "hi there, how can I help?"},
    ]
    result = estimate_messages_tokens(msgs)
    assert result > 0

def test_estimate_messages_tokens_with_tool_calls():
    msgs = [{"role": "assistant", "content": "ok", "tool_calls": [{"name": "shell", "args": {"command": "ls"}}]}]
    result = estimate_messages_tokens(msgs)
    assert result > estimate_tokens("ok") + 4  # More than just content

def test_format_token_count_small():
    assert format_token_count(500) == "500"

def test_format_token_count_thousands():
    assert format_token_count(12400) == "12.4K"

def test_format_token_count_millions():
    assert format_token_count(1200000) == "1.2M"

def test_get_context_limit_default():
    assert get_context_limit("unknown-model") == 128_000

def test_get_context_limit_minimax():
    assert get_context_limit("minimax-m2.5") == 1_000_000

def test_context_gauge_low_usage():
    gauge = build_context_gauge(used=5000, limit=128000)
    assert "green" in gauge
    assert "5.0K" in gauge

def test_context_gauge_medium_usage():
    gauge = build_context_gauge(used=70000, limit=128000)
    assert "yellow" in gauge

def test_context_gauge_high_usage():
    gauge = build_context_gauge(used=110000, limit=128000)
    assert "red" in gauge

def test_context_gauge_zero_limit():
    gauge = build_context_gauge(used=0, limit=0)
    assert "0" in gauge

def test_context_breakdown():
    breakdown = build_context_breakdown(
        system_tokens=2000, history_tokens=8000, tools_tokens=1500, limit=128000
    )
    assert "System" in breakdown
    assert "History" in breakdown or "Conversation" in breakdown
    assert "Tool" in breakdown
