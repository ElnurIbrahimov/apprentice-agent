"""Tests for research mode."""
import pytest
from aura.cli.research_mode import (
    Source, ResearchContext, render_sources,
    create_research_indicator, RESEARCH_SYSTEM_PROMPT,
)
from rich.console import Console
from io import StringIO

def test_source_citation():
    s = Source(id=1, title="Test Paper", url="https://example.com")
    assert s.to_citation() == "[1]"

def test_source_reference():
    s = Source(id=1, title="Test Paper", url="https://example.com")
    ref = s.to_reference()
    assert "[1]" in ref
    assert "Test Paper" in ref

def test_context_start_stop():
    ctx = ResearchContext()
    assert not ctx.is_active
    ctx.start("AI safety")
    assert ctx.is_active
    assert ctx.topic == "AI safety"
    ctx.stop()
    assert not ctx.is_active

def test_add_source():
    ctx = ResearchContext()
    ctx.start("test")
    s = ctx.add_source("Paper A", url="https://a.com", source_type="arxiv")
    assert s.id == 1
    assert ctx.source_count == 1

def test_cite():
    ctx = ResearchContext()
    ctx.start("test")
    ctx.add_source("Paper A")
    citation = ctx.cite(1)
    assert citation == "[1]"
    assert ctx.get_source(1).cited_count == 1

def test_cite_nonexistent():
    ctx = ResearchContext()
    assert ctx.cite(99) == "[99?]"

def test_add_finding():
    ctx = ResearchContext()
    ctx.start("test")
    ctx.add_finding("Key insight here")
    assert len(ctx.findings) == 1

def test_extract_citations():
    ctx = ResearchContext()
    refs = ctx.extract_citations_from_text("According to [1] and [3], this is true [2].")
    assert refs == [1, 3, 2]

def test_build_context_prompt():
    ctx = ResearchContext()
    ctx.start("AI safety")
    ctx.add_source("DeepMind Paper", snippet="Important finding about safety")
    prompt = ctx.build_context_prompt()
    assert "AI safety" in prompt
    assert "[1]" in prompt
    assert "DeepMind" in prompt

def test_export_markdown():
    ctx = ResearchContext()
    ctx.start("Test Topic")
    ctx.add_source("Source A", url="https://a.com", source_type="arxiv")
    ctx.add_finding("Finding one")
    md = ctx.export_markdown()
    assert "# Research: Test Topic" in md
    assert "Source A" in md
    assert "Finding one" in md

def test_render_sources():
    ctx = ResearchContext()
    ctx.start("test")
    ctx.add_source("Paper A", url="https://example.com")
    console = Console(file=StringIO(), force_terminal=True, width=100)
    render_sources(console, ctx)
    assert "Paper A" in console.file.getvalue()

def test_research_indicator_inactive():
    ctx = ResearchContext()
    assert create_research_indicator(ctx) == ""

def test_research_indicator_active():
    ctx = ResearchContext()
    ctx.start("test")
    ctx.add_source("A")
    indicator = create_research_indicator(ctx)
    assert "Research" in indicator
    assert "1" in indicator

def test_research_prompt():
    prompt = RESEARCH_SYSTEM_PROMPT.format(topic="AI", context="some context")
    assert "AI" in prompt
