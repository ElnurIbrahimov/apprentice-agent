"""Syntax-highlighted inline diff viewer for the terminal."""
from __future__ import annotations
import difflib
import logging
from typing import Optional

from rich.panel import Panel
from rich.text import Text

_logger = logging.getLogger(__name__)


# Extension to language mapping for syntax highlighting
_EXT_TO_LANG = {
    ".py": "python", ".js": "javascript", ".ts": "typescript",
    ".jsx": "jsx", ".tsx": "tsx", ".json": "json", ".yaml": "yaml",
    ".yml": "yaml", ".md": "markdown", ".html": "html", ".css": "css",
    ".sh": "bash", ".rs": "rust", ".go": "go", ".java": "java",
    ".cpp": "cpp", ".c": "c", ".rb": "ruby", ".sql": "sql",
    ".toml": "toml", ".xml": "xml", ".scss": "scss", ".less": "less",
    ".vue": "vue", ".svelte": "svelte", ".php": "php", ".swift": "swift",
    ".kt": "kotlin", ".lua": "lua", ".r": "r", ".jl": "julia",
}


def _detect_language(filename: str) -> str:
    """Detect programming language from filename extension."""
    if "." in filename:
        ext = "." + filename.rsplit(".", 1)[-1].lower()
        return _EXT_TO_LANG.get(ext, "")
    return ""


def _highlight_line(line_text: str, language: str, base_style: str) -> Text:
    """Apply syntax highlighting to a single line with a base diff style.

    Returns a Rich Text object with syntax-colored tokens on top of the
    base_style background (e.g. "on dark_green" or "on dark_red" or "dim").
    """
    from rich.syntax import Syntax

    syntax = Syntax(line_text, language, theme="monokai", background_color="default")
    highlighted = syntax.highlight(line_text)

    # Create result with the base diff style, then overlay syntax color spans
    result = Text(line_text, style=base_style)
    for span in highlighted._spans:
        result.stylize(span.style, span.start, span.end)
    return result


def generate_diff(old: str, new: str, filename: str = "file", context_lines: int = 3) -> str:
    """Generate unified diff string between old and new content."""
    if '\x00' in old or '\x00' in new:
        return f"(binary file: {filename})"
    old_lines = old.splitlines(keepends=True)
    new_lines = new.splitlines(keepends=True)
    diff_lines = list(difflib.unified_diff(
        old_lines, new_lines,
        fromfile=f"a/{filename}", tofile=f"b/{filename}",
        lineterm="",
        n=context_lines,
    ))
    if not diff_lines:
        return ""
    return "\n".join(diff_lines)


def diff_summary(old: str, new: str, filename: str = "file") -> str:
    """One-line summary: 'test.py (+3/-1)'."""
    old_lines = old.splitlines()
    new_lines = new.splitlines()
    added = removed = 0
    for line in difflib.unified_diff(old_lines, new_lines, lineterm=""):
        if line.startswith("+") and not line.startswith("+++"):
            added += 1
        elif line.startswith("-") and not line.startswith("---"):
            removed += 1
    if added == 0 and removed == 0:
        return f"{filename} (no changes)"
    return f"{filename} ([green]+{added}[/green]/[red]-{removed}[/red])"


def render_diff(old: str, new: str, filename: str = "file", context_lines: int = 3, language: str = "") -> Optional[Panel]:
    """Render a syntax-highlighted diff as a Rich Panel.

    If language is provided, uses it for syntax highlighting within diff lines.
    Otherwise auto-detects from the filename extension.
    Added lines get green background + syntax colors, removed lines get red
    background + syntax colors, context lines get dim syntax highlighting.
    Falls back to plain green/red coloring if highlighting fails.
    """
    diff_text = generate_diff(old, new, filename, context_lines=context_lines)
    if not diff_text:
        return None

    # Resolve language for syntax highlighting
    lang = language or _detect_language(filename)

    text = Text()
    for line in diff_text.split("\n"):
        if line.startswith("+++") or line.startswith("---"):
            text.append(line + "\n", style="bold")
        elif line.startswith("@@"):
            text.append(line + "\n", style="cyan")
        elif line.startswith("+"):
            if lang:
                try:
                    hl = _highlight_line(line[1:], lang, "on dark_green")
                    text.append("+", style="on dark_green")
                    text.append_text(hl)
                    text.append("\n")
                    continue
                except Exception:
                    _logger.debug("diff_highlight_add_failed", exc_info=True)
            text.append(line + "\n", style="green")
        elif line.startswith("-"):
            if lang:
                try:
                    hl = _highlight_line(line[1:], lang, "on dark_red")
                    text.append("-", style="on dark_red")
                    text.append_text(hl)
                    text.append("\n")
                    continue
                except Exception:
                    _logger.debug("diff_highlight_remove_failed", exc_info=True)
            text.append(line + "\n", style="red")
        else:
            if lang:
                try:
                    hl = _highlight_line(line, lang, "dim")
                    text.append_text(hl)
                    text.append("\n")
                    continue
                except Exception:
                    _logger.debug("diff_highlight_context_failed", exc_info=True)
            text.append(line + "\n", style="dim")

    summary = diff_summary(old, new, filename)
    return Panel(text, title=f"[bold]{summary}[/bold]", border_style="dim", padding=(0, 1))


def render_diff_compact(old: str, new: str, filename: str = "file", elapsed: float = 0.0) -> str:
    """Compact one-line diff summary for tool call display with elapsed time."""
    from aura.cli.tool_output import format_elapsed
    time_str = f" {format_elapsed(elapsed)}" if elapsed > 0 else ""
    return f"▸ [yellow]edit[/yellow] {diff_summary(old, new, filename)}{time_str}"
