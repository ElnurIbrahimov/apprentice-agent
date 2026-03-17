"""Streaming and collapsible tool output rendering."""
from __future__ import annotations
import time
from typing import Optional, List, Callable
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.syntax import Syntax


# Maximum lines shown by default before collapsing
DEFAULT_VISIBLE_LINES = 8
# Maximum output stored (prevent memory issues with huge outputs)
MAX_OUTPUT_LINES = 500


class ToolOutputRenderer:
    """Renders tool execution output with collapsing and streaming support."""

    def __init__(self, console: Optional[Console] = None, visible_lines: int = DEFAULT_VISIBLE_LINES):
        self._console = console or Console()
        self._visible_lines = visible_lines

    def render_shell_output(self, output: str, command: str = "", elapsed: float = 0.0, exit_code: int = 0) -> None:
        """Render shell command output with collapsing for long output."""
        lines = output.splitlines() if output else []

        if not lines:
            self._console.print(f"  [dim](no output)[/dim]")
            return

        # Header with command and timing
        status_icon = "[green]\u2713[/green]" if exit_code == 0 else "[red]\u2717[/red]"
        elapsed_str = f" ({elapsed:.1f}s)" if elapsed > 0.5 else ""
        header = f"{status_icon} [dim]{command}{elapsed_str}[/dim]" if command else ""
        if header:
            self._console.print(f"  {header}")

        # Show lines with collapsing
        if len(lines) <= self._visible_lines:
            for line in lines:
                self._console.print(f"  [dim]\u2502[/dim] {line}")
        else:
            for line in lines[:self._visible_lines]:
                self._console.print(f"  [dim]\u2502[/dim] {line}")
            hidden = len(lines) - self._visible_lines
            self._console.print(f"  [dim]\u2502 ... +{hidden} more lines (use -v to show all)[/dim]")

    def render_file_content(self, content: str, filename: str = "", language: str = "") -> None:
        """Render file content with syntax highlighting, collapsed if long."""
        lines = content.splitlines()

        if not language and filename:
            # Detect language from extension
            ext_map = {
                ".py": "python", ".js": "javascript", ".ts": "typescript",
                ".jsx": "jsx", ".tsx": "tsx", ".json": "json", ".yaml": "yaml",
                ".yml": "yaml", ".md": "markdown", ".html": "html", ".css": "css",
                ".sh": "bash", ".rs": "rust", ".go": "go", ".java": "java",
                ".cpp": "cpp", ".c": "c", ".rb": "ruby", ".sql": "sql",
            }
            ext = "." + filename.rsplit(".", 1)[-1] if "." in filename else ""
            language = ext_map.get(ext, "text")

        if len(lines) <= self._visible_lines + 5:
            # Small enough to show fully
            syntax = Syntax(content, language or "text", theme="monokai", line_numbers=True)
            self._console.print(Panel(syntax, title=f"[dim]{filename}[/dim]", border_style="dim", padding=(0, 1)))
        else:
            # Collapse — show first N lines
            preview = "\n".join(lines[:self._visible_lines])
            syntax = Syntax(preview, language or "text", theme="monokai", line_numbers=True)
            hidden = len(lines) - self._visible_lines
            self._console.print(Panel(
                syntax,
                title=f"[dim]{filename} ({len(lines)} lines)[/dim]",
                subtitle=f"[dim]+{hidden} more lines[/dim]",
                border_style="dim",
                padding=(0, 1),
            ))

    def render_search_results(self, results: List[dict], query: str = "") -> None:
        """Render search/grep results with highlighting."""
        if not results:
            self._console.print(f"  [dim](no results for '{query}')[/dim]")
            return

        shown = min(len(results), self._visible_lines)
        for r in results[:shown]:
            file_path = r.get("file", r.get("path", ""))
            line_num = r.get("line", "")
            text = r.get("text", r.get("content", ""))
            loc = f"[cyan]{file_path}[/cyan]"
            if line_num:
                loc += f"[dim]:{line_num}[/dim]"
            self._console.print(f"  {loc}  {text.strip()}")

        if len(results) > shown:
            self._console.print(f"  [dim]... +{len(results) - shown} more results[/dim]")

    def render_web_result(self, content: str, url: str = "", status_code: int = 200) -> None:
        """Render web request result."""
        status_color = "green" if 200 <= status_code < 300 else "yellow" if 300 <= status_code < 400 else "red"
        header = f"[{status_color}]{status_code}[/{status_color}]"
        if url:
            header += f" [dim]{url}[/dim]"
        self._console.print(f"  {header}")

        lines = content.splitlines() if content else []
        if len(lines) <= self._visible_lines:
            for line in lines:
                self._console.print(f"  [dim]\u2502[/dim] {line}")
        else:
            for line in lines[:self._visible_lines]:
                self._console.print(f"  [dim]\u2502[/dim] {line}")
            hidden = len(lines) - self._visible_lines
            self._console.print(f"  [dim]\u2502 ... +{hidden} more lines[/dim]")

    def render_tool_result(self, tool_name: str, result: dict, elapsed: float = 0.0) -> None:
        """Smart dispatcher — routes to the right renderer based on tool type."""
        output = result.get("output", result.get("content", result.get("result", "")))
        if isinstance(output, dict):
            import json
            output = json.dumps(output, indent=2)
        elif not isinstance(output, str):
            output = str(output) if output else ""

        if tool_name in ("shell", "shell_executor", "bash", "run"):
            self.render_shell_output(
                output=output,
                command=result.get("command", ""),
                elapsed=elapsed,
                exit_code=result.get("exit_code", result.get("returncode", 0)) or 0,
            )
        elif tool_name in ("read_file", "cat", "edit"):
            self.render_file_content(
                content=output,
                filename=result.get("path", result.get("file", "")),
            )
        elif tool_name in ("grep", "search", "find", "glob", "code_search"):
            results = result.get("results", result.get("matches", []))
            if isinstance(results, list):
                self.render_search_results(results, query=result.get("query", result.get("pattern", "")))
            else:
                self.render_shell_output(output=output, elapsed=elapsed)
        elif tool_name in ("web_search", "browse", "fetch"):
            self.render_web_result(
                content=output,
                url=result.get("url", ""),
                status_code=result.get("status_code", 200),
            )
        else:
            # Generic fallback
            if output:
                self.render_shell_output(output=output, elapsed=elapsed)


def format_elapsed(seconds: float) -> str:
    """Format elapsed time for display."""
    if seconds < 1:
        return f"{seconds*1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    else:
        mins = int(seconds // 60)
        secs = seconds % 60
        return f"{mins}m{secs:.0f}s"
