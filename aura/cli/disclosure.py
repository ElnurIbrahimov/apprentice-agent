# aura/cli/disclosure.py
"""Progressive disclosure — collapsible tool output with expand/collapse."""
from __future__ import annotations
import threading
from dataclasses import dataclass, field
from typing import List, Optional
from rich.console import Console
from rich.text import Text


@dataclass
class CollapsibleSection:
    """A section of output that can be expanded/collapsed."""
    id: str
    title: str  # One-line summary shown when collapsed
    content: str  # Full content shown when expanded
    expanded: bool = False
    tool_name: str = ""
    elapsed: float = 0.0

    @property
    def display(self) -> str:
        if self.expanded:
            return self.content
        return self.title


class DisclosureManager:
    """Manages collapsible sections in the conversation."""

    def __init__(self, default_expanded: bool = False):
        self._sections: List[CollapsibleSection] = []
        self._default_expanded = default_expanded
        self._verbose = False
        self._lock = threading.Lock()

    def set_verbose(self, verbose: bool) -> None:
        """Set verbose mode — all sections expanded by default."""
        self._verbose = verbose
        self._default_expanded = verbose

    def add_section(
        self,
        section_id: str,
        title: str,
        content: str,
        tool_name: str = "",
        elapsed: float = 0.0,
    ) -> CollapsibleSection:
        """Add a new collapsible section."""
        section = CollapsibleSection(
            id=section_id,
            title=title,
            content=content,
            expanded=self._default_expanded,
            tool_name=tool_name,
            elapsed=elapsed,
        )
        with self._lock:
            self._sections.append(section)
            # Keep only last 50 sections to prevent memory growth
            if len(self._sections) > 50:
                self._sections = self._sections[-50:]
        return section

    def toggle(self, section_id: str) -> bool:
        """Toggle a section's expanded state. Returns new state."""
        with self._lock:
            for s in self._sections:
                if s.id == section_id:
                    s.expanded = not s.expanded
                    return s.expanded
        return False

    def expand_all(self) -> None:
        """Expand all sections."""
        with self._lock:
            for s in self._sections:
                s.expanded = True

    def collapse_all(self) -> None:
        """Collapse all sections."""
        with self._lock:
            for s in self._sections:
                s.expanded = False

    def get_section(self, section_id: str) -> Optional[CollapsibleSection]:
        """Get a section by ID."""
        with self._lock:
            return next((s for s in self._sections if s.id == section_id), None)

    def get_recent(self, n: int = 10) -> List[CollapsibleSection]:
        """Get the N most recent sections."""
        with self._lock:
            return list(self._sections[-n:])

    @property
    def section_count(self) -> int:
        with self._lock:
            return len(self._sections)

    def clear(self) -> None:
        """Clear all sections."""
        with self._lock:
            self._sections.clear()


def render_collapsed(console: Console, section: CollapsibleSection) -> None:
    """Render a collapsed section — one-line summary with expand hint."""
    elapsed_str = f" ({section.elapsed:.1f}s)" if section.elapsed > 0.5 else ""
    console.print(f"  [dim]\u25b8[/dim] [yellow]{section.tool_name}[/yellow] {section.title}{elapsed_str}")


def render_expanded(console: Console, section: CollapsibleSection) -> None:
    """Render an expanded section — full content with collapse hint."""
    elapsed_str = f" ({section.elapsed:.1f}s)" if section.elapsed > 0.5 else ""
    console.print(f"  [dim]\u25be[/dim] [yellow]{section.tool_name}[/yellow] {section.title}{elapsed_str}")
    for line in section.content.splitlines():
        console.print(f"  [dim]\u2502[/dim] {line}")
    console.print(f"  [dim]\u2514[/dim]")


def render_section(console: Console, section: CollapsibleSection) -> None:
    """Render a section in its current state."""
    if section.expanded:
        render_expanded(console, section)
    else:
        render_collapsed(console, section)


def create_section_from_tool_call(
    tool_name: str,
    args: dict,
    result: dict,
    elapsed: float = 0.0,
) -> CollapsibleSection:
    """Create a CollapsibleSection from a tool call result."""
    import uuid

    # Build title (one-line summary)
    if tool_name in ("shell", "shell_executor", "bash", "run"):
        cmd = args.get("command", "")[:60]
        exit_code = result.get("exit_code", result.get("returncode", 0)) or 0
        icon = "\u2713" if exit_code == 0 else "\u2717"
        title = f"{icon} `{cmd}`"
    elif tool_name in ("edit_file", "write_file"):
        path = args.get("path", args.get("file_path", ""))
        path_short = path.split("/")[-1].split("\\")[-1] if path else "file"
        title = f"edited {path_short}"
    elif tool_name in ("read_file", "cat"):
        path = args.get("path", "")
        path_short = path.split("/")[-1].split("\\")[-1] if path else "file"
        title = f"read {path_short}"
    elif tool_name in ("grep", "search", "code_search"):
        query = args.get("pattern", args.get("query", ""))[:40]
        title = f"search '{query}'"
    elif tool_name in ("web_search", "browse"):
        query = args.get("query", args.get("url", ""))[:40]
        title = f"web: {query}"
    else:
        title = f"{tool_name}"

    # Build content (full output)
    output = result.get("output", result.get("content", result.get("result", "")))
    if isinstance(output, dict):
        import json
        output = json.dumps(output, indent=2)
    elif not isinstance(output, str):
        output = str(output) if output else ""

    return CollapsibleSection(
        id=f"sec_{uuid.uuid4().hex[:8]}",
        title=title,
        content=output[:5000],  # cap content at 5KB
        tool_name=tool_name,
        elapsed=elapsed,
    )
