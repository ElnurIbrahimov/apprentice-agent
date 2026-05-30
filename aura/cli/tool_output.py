"""Minimal tool output rendering — no panels, compact output."""
from __future__ import annotations

import json
from typing import List

# File reads: show first/last N lines
_FILE_HEAD = 20
_FILE_TAIL = 20
# Shell output: show last N lines
_SHELL_TAIL = 40
# Max stored lines to prevent memory issues
MAX_OUTPUT_LINES = 500


def _get_disclosure_manager():
    """Return the process-wide DisclosureManager if one was installed.

    chat_session.py sets display._disclosure = DisclosureManager(...) when
    entering chat mode. If no manager is installed (non-interactive runs,
    early startup), returns None and callers fall back to direct rendering.
    """
    try:
        from . import display as _display_mod
        return getattr(_display_mod, "_disclosure", None)
    except Exception:
        return None


class ToolOutputRenderer:
    """Renders tool output — compact, no panels, dimmed."""

    def __init__(self, console=None, visible_lines: int = _SHELL_TAIL):
        if console is None:
            from rich.console import Console
            console = Console()
        self._console = console
        self._visible_lines = visible_lines

    def render_shell_output(self, output: str, command: str = "",
                            elapsed: float = 0.0, exit_code: int = 0) -> None:
        """Show last N lines of shell output, dimmed with themed status icon."""
        lines = output.splitlines() if output else []

        if not lines:
            self._console.print("    [dim](no output)[/dim]")
            return

        # Status icon with theme color (was hardcoded #4EBA65 / #FF6B80,
        # which bypassed theme switching and always rendered pink/green).
        from aura.cli.display import _get_theme_colors as _tc
        _c = _tc()
        success_c = _c.get("success", AuraTheme(name="fallback").success)
        error_c = _c.get("error", AuraTheme(name="fallback").error)
        if exit_code == 0:
            icon = f"[{success_c}]\u2713[/{success_c}]"
        else:
            icon = f"[{error_c}]\u2717 exit {exit_code}[/{error_c}]"
        elapsed_str = f" {format_elapsed(elapsed)}" if elapsed > 0.5 else ""
        if command:
            self._console.print(f"    {icon} [dim]{command}{elapsed_str}[/dim]")

        tail = self._visible_lines
        if len(lines) <= tail:
            for line in lines:
                self._console.print(f"    [dim]{line}[/dim]")
        else:
            hidden = len(lines) - tail
            self._console.print(f"    [dim]\u2026 {hidden} lines hidden[/dim]")
            for line in lines[-tail:]:
                self._console.print(f"    [dim]{line}[/dim]")

    def render_file_content(self, content: str, filename: str = "",
                            language: str = "") -> None:
        """Show first/last N lines of a file read with line count."""
        lines = content.splitlines()
        total = len(lines)

        if filename:
            self._console.print(f"    [dim]{filename} ({total} lines)[/dim]")

        if total <= _FILE_HEAD + _FILE_TAIL + 2:
            for line in lines:
                self._console.print(f"    [dim]{line}[/dim]")
        else:
            for line in lines[:_FILE_HEAD]:
                self._console.print(f"    [dim]{line}[/dim]")
            hidden = total - _FILE_HEAD - _FILE_TAIL
            self._console.print(f"    [dim]\u2026 {hidden} lines hidden[/dim]")
            for line in lines[-_FILE_TAIL:]:
                self._console.print(f"    [dim]{line}[/dim]")

    def render_search_results(self, results: List[dict], query: str = "") -> None:
        """Show search results, compact."""
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
            self._console.print(f"  [dim]... +{len(results) - shown} more[/dim]")

    def render_web_result(self, content: str, url: str = "",
                          status_code: int = 200) -> None:
        """Show web result, compact."""
        from aura.cli.display import _get_theme_colors as _tc_web
        _wc = _tc_web()
        if 200 <= status_code < 300:
            sc = _wc.get("success", "green")
        elif 300 <= status_code < 400:
            sc = _wc.get("warning", "yellow")
        else:
            sc = _wc.get("error", "red")
        header = f"[{sc}]{status_code}[/{sc}]"
        if url:
            header += f" [dim]{url}[/dim]"
        self._console.print(f"  {header}")

        lines = content.splitlines() if content else []
        tail = self._visible_lines
        if len(lines) <= tail:
            for line in lines:
                self._console.print(f"  [dim]{line}[/dim]")
        else:
            hidden = len(lines) - tail
            self._console.print(f"  [dim]... {hidden} lines hidden ...[/dim]")
            for line in lines[-tail:]:
                self._console.print(f"  [dim]{line}[/dim]")

    def render_tool_result(self, tool_name: str, result: dict,
                           elapsed: float = 0.0) -> None:
        """Route to the right renderer based on tool type."""
        output = result.get("output", result.get("content", result.get("result", "")))
        if isinstance(output, dict):
            output = json.dumps(output, indent=2)
        elif not isinstance(output, str):
            output = str(output) if output else ""

        # Auto-render the full colorized diff panel whenever an edit/write
        # tool returns a pre-generated unified diff. The renderer already
        # exists in diff_viewer.py but wasn't wired — tool output was just
        # dumping the raw diff into the shell formatter, which is illegible.
        if tool_name in ("edit_file", "edit", "write_file", "multi_edit") and result.get("success"):
            diff_text = result.get("diff") or ""
            if diff_text and not result.get("preview"):
                try:
                    from .diff_viewer import render_diff_from_text
                    path = result.get("path", result.get("file", "file"))
                    filename = path.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
                    panel = render_diff_from_text(diff_text, filename=filename)
                    if panel is not None:
                        from .display import console as _console
                        _console.print(panel)
                        return
                except Exception:
                    pass

        # Route non-diff tool output through the DisclosureManager when one
        # is installed (chat mode always installs one). --verbose / -v flips
        # default_expanded=True so full content shows; otherwise we render a
        # one-line collapsed summary and keep the panel scroll clean.
        disclosure = _get_disclosure_manager()
        if disclosure is not None and tool_name not in ("edit_file", "edit", "write_file", "multi_edit"):
            try:
                from .disclosure import create_section_from_tool_call, render_section
                from .display import console as _console
                section = create_section_from_tool_call(
                    tool_name=tool_name,
                    args=result.get("_args", {}) or {},
                    result=result,
                    elapsed=elapsed,
                )
                # Honor the manager's default_expanded / _verbose setting
                section.expanded = getattr(disclosure, "_default_expanded", False)
                # Register so toggle / get_recent work later
                try:
                    disclosure.add_section(
                        section_id=section.id,
                        title=section.title,
                        content=section.content,
                        tool_name=section.tool_name,
                        elapsed=section.elapsed,
                    )
                except Exception:
                    pass
                render_section(_console, section)
                return
            except Exception:
                # Fall through to the legacy renderers on any failure so the
                # user never loses tool output to a disclosure bug.
                pass

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
            if output:
                self.render_shell_output(output=output, elapsed=elapsed)


def format_elapsed(seconds: float) -> str:
    """Format elapsed time for display."""
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    else:
        mins = int(seconds // 60)
        secs = seconds % 60
        return f"{mins}m{secs:.0f}s"
