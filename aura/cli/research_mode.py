# aura/cli/research_mode.py
"""Research mode — citation tracking, source management, and structured research."""
from __future__ import annotations
import re
import time
from dataclasses import dataclass, field
from typing import List, Optional, Dict
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


@dataclass
class Source:
    """A research source with metadata."""
    id: int
    title: str
    url: str = ""
    source_type: str = "web"  # web, arxiv, memory, manual
    snippet: str = ""
    added_at: float = field(default_factory=time.time)
    cited_count: int = 0

    def to_citation(self) -> str:
        """Format as inline citation."""
        return f"[{self.id}]"

    def to_reference(self) -> str:
        """Format as bibliography entry."""
        parts = [f"[{self.id}] {self.title}"]
        if self.url:
            parts.append(f"    URL: {self.url}")
        if self.source_type != "web":
            parts.append(f"    Source: {self.source_type}")
        return "\n".join(parts)


class ResearchContext:
    """Manages sources and citations for a research session."""

    def __init__(self):
        self._sources: List[Source] = []
        self._next_id: int = 1
        self._findings: List[str] = []
        self._topic: str = ""
        self._active: bool = False

    @property
    def is_active(self) -> bool:
        return self._active

    @property
    def topic(self) -> str:
        return self._topic

    def start(self, topic: str) -> None:
        """Start a research session."""
        self._topic = topic
        self._active = True
        self._sources.clear()
        self._findings.clear()
        self._next_id = 1

    def stop(self) -> None:
        """End research mode."""
        self._active = False

    def add_source(self, title: str, url: str = "", source_type: str = "web", snippet: str = "") -> Source:
        """Add a source and return it with an assigned citation number."""
        source = Source(
            id=self._next_id,
            title=title,
            url=url,
            source_type=source_type,
            snippet=snippet[:500],
        )
        self._sources.append(source)
        self._next_id += 1
        return source

    def add_finding(self, finding: str) -> None:
        """Record a research finding."""
        self._findings.append(finding)

    def get_source(self, source_id: int) -> Optional[Source]:
        """Get a source by citation number."""
        return next((s for s in self._sources if s.id == source_id), None)

    def cite(self, source_id: int) -> str:
        """Increment cite count and return citation string."""
        source = self.get_source(source_id)
        if source:
            source.cited_count += 1
            return source.to_citation()
        return f"[{source_id}?]"

    @property
    def sources(self) -> List[Source]:
        return list(self._sources)

    @property
    def source_count(self) -> int:
        return len(self._sources)

    @property
    def findings(self) -> List[str]:
        return list(self._findings)

    def extract_citations_from_text(self, text: str) -> List[int]:
        """Find all [N] citation references in text."""
        return [int(m) for m in re.findall(r'\[(\d+)\]', text)]

    def build_context_prompt(self) -> str:
        """Build a context string for the LLM with all sources."""
        if not self._sources:
            return ""
        lines = [f"Research topic: {self._topic}", "", "Sources gathered so far:"]
        for s in self._sources:
            lines.append(f"  [{s.id}] {s.title}")
            if s.snippet:
                lines.append(f"      {s.snippet[:200]}")
        if self._findings:
            lines.append("\nKey findings:")
            for f in self._findings:
                lines.append(f"  - {f}")
        lines.append("\nUse [N] citations when referencing sources.")
        return "\n".join(lines)

    def export_markdown(self) -> str:
        """Export research as structured Markdown."""
        lines = [
            f"# Research: {self._topic}",
            f"*Generated: {time.strftime('%Y-%m-%d %H:%M')}*",
            "",
        ]
        if self._findings:
            lines.append("## Key Findings")
            lines.append("")
            for f in self._findings:
                lines.append(f"- {f}")
            lines.append("")

        if self._sources:
            lines.append("## Sources")
            lines.append("")
            for s in self._sources:
                entry = f"{s.id}. **{s.title}**"
                if s.url:
                    entry += f" — [{s.url}]({s.url})"
                if s.source_type != "web":
                    entry += f" *({s.source_type})*"
                lines.append(entry)
                if s.snippet:
                    lines.append(f"   > {s.snippet[:300]}")
                lines.append("")

        return "\n".join(lines)


def render_sources(console: Console, context: ResearchContext) -> None:
    """Render all sources as a Rich table."""
    if not context.sources:
        console.print("[dim]No sources collected yet.[/dim]")
        return

    table = Table(title=f"Research Sources: {context.topic}", border_style="cyan")
    table.add_column("#", style="bold", width=4)
    table.add_column("Title", min_width=30)
    table.add_column("Type", width=8)
    table.add_column("Cited", width=6)
    table.add_column("URL", max_width=40, style="dim")

    for s in context.sources:
        table.add_row(
            str(s.id),
            s.title[:50],
            s.source_type,
            str(s.cited_count),
            (s.url[:40] + "..." if len(s.url) > 40 else s.url) if s.url else "—",
        )

    console.print(table)


def create_research_indicator(context: ResearchContext) -> str:
    """Status bar indicator for active research mode."""
    if not context.is_active:
        return ""
    return f"[magenta]🔬 Research ({context.source_count} sources)[/magenta]"


RESEARCH_SYSTEM_PROMPT = """You are in research mode investigating: {topic}

{context}

Guidelines:
- Search for information using available tools (web_search, arxiv, memory)
- When you find useful information, note the source clearly
- Use [N] citation format when referencing numbered sources
- Focus on gathering diverse, high-quality sources
- Summarize findings clearly and cite sources"""
