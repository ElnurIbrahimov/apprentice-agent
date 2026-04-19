"""Tests for memory fence-tag injection in SystemPromptBuilder.

Verifies that recalled episodic memory is wrapped in <memory-context> tags
with a system-note preamble so the LLM doesn't treat recalled facts as
current user input.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from aura.prompt_builder import SystemPromptBuilder


class _FakeRecord:
    def __init__(self, content: str, created_at: str = "2026-04-19"):
        self.content = content
        self.metadata = {"created_at": created_at}


def _make_builder() -> SystemPromptBuilder:
    b = SystemPromptBuilder()
    # Bypass the 25-char guard by using a long enough prompt
    return b


@patch("aura.memory.unified_memory.get_unified_memory")
def test_fence_tags_when_recall_has_results(mock_um):
    mock_mem = MagicMock()
    mock_mem.query.return_value = [
        _FakeRecord("user prefers Vercel", "2026-04-10"),
        _FakeRecord("Aura uses Kokoro TTS", "2026-04-11"),
    ]
    mock_um.return_value = mock_mem

    b = _make_builder()
    base = "BASE_PROMPT"
    prompt = "what do you remember about my preferences for this project?"
    out = b._inject_episodic_memory(base, prompt)

    assert "<memory-context" in out
    assert "</memory-context>" in out
    assert "System note" in out
    assert "user prefers Vercel" in out


@patch("aura.memory.unified_memory.get_unified_memory")
def test_no_tags_when_recall_empty(mock_um):
    mock_mem = MagicMock()
    mock_mem.query.return_value = []
    mock_um.return_value = mock_mem

    b = _make_builder()
    base = "BASE_PROMPT"
    prompt = "what do you remember about my preferences for this project?"
    out = b._inject_episodic_memory(base, prompt)

    assert "<memory-context" not in out
    assert out == base  # unchanged


@patch("aura.memory.unified_memory.get_unified_memory")
def test_paired_tags_always(mock_um):
    """Every opening tag must have a closing tag."""
    mock_mem = MagicMock()
    mock_mem.query.return_value = [_FakeRecord("x", "2026-04-10")]
    mock_um.return_value = mock_mem

    b = _make_builder()
    out = b._inject_episodic_memory("BASE", "can you tell me about my prior work?")
    assert out.count("<memory-context") == out.count("</memory-context>")


@patch("aura.memory.unified_memory.get_unified_memory")
def test_short_prompt_skips_recall(mock_um):
    """_inject_episodic_memory bails out for very short prompts (guard)."""
    mock_mem = MagicMock()
    mock_mem.query.return_value = [_FakeRecord("x")]
    mock_um.return_value = mock_mem

    b = _make_builder()
    # Prompt under 25 chars → guard activates, no recall attempted
    out = b._inject_episodic_memory("BASE", "hi")
    assert out == "BASE"
    assert not mock_mem.query.called
