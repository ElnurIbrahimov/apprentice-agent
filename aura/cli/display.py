"""Rich-based display for AURA CLI — clean, polished, Claude Code / OpenCode aesthetic."""
from __future__ import annotations

import logging
import os
from typing import Any, Optional

logger = logging.getLogger(__name__)

from rich.console import Console
from rich.markdown import Markdown
from rich.padding import Padding
from rich.panel import Panel
from rich.live import Live
from rich.text import Text

from aura.cli.tool_output import ToolOutputRenderer, format_elapsed
from aura.cli.tool_icons import get_tool_icon, STATUS_ICONS

_no_color = os.environ.get("NO_COLOR") is not None
console: Console = Console(highlight=True, soft_wrap=True, no_color=_no_color)

import functools


@functools.lru_cache(maxsize=1)
def _get_tool_renderer() -> ToolOutputRenderer:
    return ToolOutputRenderer(console=console)


def _get_code_theme() -> str:
    """Get the current code theme from the theme system."""
    try:
        from aura.cli.themes import get_theme
        return get_theme().code_theme
    except (ImportError, AttributeError):
        return "monokai"


def _get_theme_colors() -> dict:
    """Get semantic colors from current theme."""
    try:
        from aura.cli.themes import get_theme
        theme = get_theme()
        return {
            "accent": theme.accent,
            "accent_dim": theme.accent_dim,
            "success": theme.success,
            "error": theme.error,
            "warning": theme.warning,
            "info": theme.info,
            "tool": theme.tool_color,
            "tool_success": theme.tool_success,
            "tool_error": theme.tool_error,
            "tool_pending": theme.tool_pending,
            "permission_border": theme.permission_border,
            "permission_accent": theme.permission_accent,
            "text_secondary": theme.text_secondary,
            "text_muted": theme.text_muted,
        }
    except (ImportError, AttributeError):
        return {
            "accent": "#D777AF",
            "accent_dim": "#B0578F",
            "success": "#4EBA65",
            "error": "#FF6B80",
            "warning": "#FFC107",
            "info": "#87AFFF",
            "tool": "#E6DB74",
            "tool_success": "#4EBA65",
            "tool_error": "#FF6B80",
            "tool_pending": "#87AFFF",
            "permission_border": "#FFC107",
            "permission_accent": "#B1B9F9",
            "text_secondary": "#999999",
            "text_muted": "#555555",
        }


# ─────────────────────────────────────────────────────────
# Banner & Welcome
# ─────────────────────────────────────────────────────────

def show_banner() -> None:
    """Display startup screen — cohesive bordered panel with logo, info, shortcuts."""
    from .banner import _apply_gradient, _LOGO_LINES
    from aura import __version__

    colors = _get_theme_colors()
    try:
        from aura.cli.themes import get_theme
        gradient = get_theme().gradient
    except (ImportError, AttributeError):
        gradient = ["#D777AF", "#B1B9F9", "#87D7D7"]

    # Build the panel content
    content = Text()
    for line in _LOGO_LINES:
        content.append(" ")
        content.append_text(_apply_gradient(line, gradient))
        content.append("\n")
    content.append(f" \u2728 v{__version__}", style="dim")

    panel = Panel(
        content,
        border_style=colors["accent"],
        padding=(0, 1),
        width=min(console.width - 2, 50),
    )
    console.print()
    console.print(panel)


def show_welcome_info(agent: Any) -> None:
    """Show model, tools, and keyboard shortcuts below the banner."""
    colors = _get_theme_colors()
    model = "auto"
    try:
        model = agent.brain._model_override or "auto"
    except AttributeError:
        logger.debug("welcome_model_read_failed", exc_info=True)

    tool_count = 0
    try:
        tool_count = len(agent.tools)
    except (TypeError, AttributeError):
        logger.debug("welcome_tool_count_failed", exc_info=True)

    # Model + tools line
    t = Text("  ")
    t.append(model, style=f"bold {colors['accent']}")
    if tool_count:
        t.append(f" \u00b7 {tool_count} tools", style="dim")
    console.print(t)

    # Compact shortcut hints
    h = Text("  ")
    shortcuts = [
        ("/", "commands"),
        ("Alt+M", "model"),
        ("?", "help"),
        ("Shift+Tab", "perms"),
        ("Ctrl+K", "palette"),
    ]
    for i, (key, desc) in enumerate(shortcuts):
        if i > 0:
            h.append("  ", style="dim")
        h.append(key, style=f"bold {colors['accent']}")
        h.append(f" {desc}", style="dim")
    console.print(h)
    console.print()


# ─────────────────────────────────────────────────────────
# Status Bar
# ─────────────────────────────────────────────────────────

def show_status_bar(
    model: str = "auto",
    cost_usd: float = 0.0,
    token_used: int = 0,
    token_limit: int = 128000,
    permission_mode: str = "careful",
    bg_indicator: str = "",
    research_indicator: str = "",
    mood_indicator: str = "",
    watch_indicator: str = "",
    steering_queue: object = None,
    session_title: str = "",
    message_count: int = 0,
    project_type: str = "",
) -> None:
    """Update the persistent bottom toolbar."""
    from .status_bar import build_status_bar
    from .input import set_bottom_toolbar

    toolbar_parts = build_status_bar(
        model=model,
        cost_usd=cost_usd,
        token_used=token_used,
        token_limit=token_limit,
        permission_mode=permission_mode,
        bg_indicator=bg_indicator,
        research_indicator=research_indicator,
        mood_indicator=mood_indicator,
        watch_indicator=watch_indicator,
        steering_queue=steering_queue,
        session_title=session_title,
        message_count=message_count,
        project_type=project_type,
        as_ansi=True,
    )
    set_bottom_toolbar(toolbar_parts)


# ─────────────────────────────────────────────────────────
# Thinking / Spinner
# ─────────────────────────────────────────────────────────

def show_thinking(label: str = None, step: int = None) -> Live:
    """Context manager — shows animated shimmer spinner while agent runs."""
    from .spinner import AuraSpinner
    spinner = AuraSpinner(label=label, step=step)
    live = Live(spinner, console=console, refresh_per_second=12, transient=True)
    live._aura_spinner = spinner  # expose for external verb/step/token updates
    return live


# Phase-aware verb mapping for contextual spinner labels
_TOOL_PHASE_VERBS: dict[str, str] = {
    "web_search": "Searching the web",
    "search_web": "Searching the web",
    "browse": "Browsing",
    "browse_url": "Browsing",
    "read_file": "Reading files",
    "edit_file": "Editing code",
    "write_file": "Writing code",
    "execute": "Running code",
    "run_command": "Running command",
    "shell": "Running shell",
    "analyze": "Analyzing",
    "summarize": "Summarizing",
    "translate": "Translating",
    "calculate": "Calculating",
    "research": "Researching",
    "deep_research": "Deep researching",
    "memory_recall": "Remembering",
    "memory_store": "Storing memory",
    "git": "Running git",
    "grep": "Searching code",
    "glob": "Finding files",
    "code_search": "Searching codebase",
}


def get_thinking_label(tool_name: str | None = None) -> str:
    """Return a phase-aware thinking label based on the current tool."""
    if tool_name and tool_name in _TOOL_PHASE_VERBS:
        return _TOOL_PHASE_VERBS[tool_name]
    return "Thinking..."


# ─────────────────────────────────────────────────────────
# Tool Call Display
# ─────────────────────────────────────────────────────────

def show_tool_call(
    tool_name: str,
    description: str = "",
    result: Any = None,
    elapsed: float = 0.0,
    status: str = "success",
    step: int = 0,
    max_steps: int = 0,
) -> None:
    """Print a tool call with step counter, status dot, icon, name, and description.

    Visual style: ┃ Step N · ● → tool_name description
    Shows the workflow progress so user can see what's happening.
    """
    colors = _get_theme_colors()
    time_str = f" {format_elapsed(elapsed)}" if elapsed > 0 else ""

    # For edit/write with diff info, show compact diff summary
    if tool_name in ("edit_file", "write_file") and result and isinstance(result, dict) and result.get("diff"):
        try:
            from aura.cli.diff_viewer import render_diff_compact
            filename = result.get("path", "file")
            filename = filename.split("/")[-1].split("\\")[-1]
            summary = render_diff_compact(
                result.get("old_content", ""),
                result.get("new_content", ""),
                filename=filename,
                elapsed=elapsed,
            )
            dot_char, dot_style = _get_status_dot(status, colors)
            step_prefix = _step_prefix(step, max_steps, colors)
            console.print(f"  {step_prefix}[{dot_style}]{dot_char}[/{dot_style}] {summary}")
            return
        except (ImportError, ValueError, KeyError, TypeError):
            pass

    # Status dot
    dot_char, dot_style = _get_status_dot(status, colors)

    # Tool icon
    icon = get_tool_icon(tool_name)

    # Build the line with left border + step counter
    line = Text()

    # Left border for visual grouping (like OpenCode's ┃)
    line.append("  \u2503 ", style=f"{colors['accent']}")

    # Step prefix
    if step > 0:
        line.append(f"[{step}] ", style=f"bold {colors['accent']}")

    line.append(dot_char, style=dot_style)
    line.append(f" {icon} ", style=f"{colors['tool']}")
    line.append(tool_name, style=f"bold {colors['tool']}")
    if description:
        line.append(f" {description}", style="dim")
    if time_str:
        line.append(f" {time_str}", style="dim")
    console.print(line)


def show_tool_result_inline(tool_name: str, result: Any) -> None:
    """Show a compact inline tool result — just enough to see what happened.

    This is the KEY to workflow visibility. User sees:
      Step 2 · ● → read_file src/main.py
        245 lines | python
      Step 2 · ● ← edit_file src/app.py
        +12/-3 lines changed
      Step 3 · ● $ shell npm test
        ✓ (4 lines)
    """
    if not result:
        return

    colors = _get_theme_colors()
    # Left border prefix for visual grouping
    B = f"  [{colors['accent']}]\u2503[/{colors['accent']}]   "

    try:
        import json
        if isinstance(result, str):
            parsed = json.loads(result)
        else:
            parsed = result
        if not isinstance(parsed, dict):
            return

        # Error display
        if parsed.get("error"):
            err_msg = str(parsed["error"])
            if len(err_msg) > 120:
                err_msg = err_msg[:120] + "\u2026"
            console.print(f"{B}[{colors['error']}]\u2717 {err_msg}[/{colors['error']}]")
            return

        # ── Search/match tools ──
        if tool_name in ("grep", "glob", "code_search", "search", "find"):
            results_list = parsed.get("results", parsed.get("matches", []))
            if isinstance(results_list, list):
                count = len(results_list)
                console.print(f"{B}[dim]{count} {'match' if count == 1 else 'matches'}[/dim]")
                return

        # ── Web tools ──
        if tool_name in ("web_search", "browse", "web_fetch", "browse_url"):
            status_code = parsed.get("status_code", 200)
            url = parsed.get("url", "")
            if url:
                short_url = url[:60] + "\u2026" if len(url) > 60 else url
                console.print(f"{B}[dim]{status_code} {short_url}[/dim]")
                return

        # ── Get output text ──
        output = parsed.get("output", parsed.get("content", parsed.get("result", "")))
        if not output or not isinstance(output, str):
            return

        lines = output.splitlines()
        line_count = len(lines)

        # ── File reads ──
        if tool_name in ("read_file", "cat"):
            lang = parsed.get("language", "")
            extras = []
            if line_count > 0:
                extras.append(f"{line_count} lines")
            if lang:
                extras.append(lang)
            if extras:
                console.print(f"{B}[dim]{' \u2502 '.join(extras)}[/dim]")

        # ── Shell commands ──
        elif tool_name in ("shell", "shell_executor", "bash", "run", "run_command", "execute"):
            exit_code = parsed.get("exit_code", parsed.get("returncode", 0)) or 0
            if exit_code == 0:
                ic = f"[{colors['success']}]\u2713[/{colors['success']}]"
            else:
                ic = f"[{colors['error']}]\u2717 exit {exit_code}[/{colors['error']}]"
            meaningful = [l.strip() for l in lines if l.strip()][-3:]
            if meaningful:
                console.print(f"{B}{ic} [dim]({line_count} lines)[/dim]")
                for ml in meaningful:
                    if len(ml) > 100:
                        ml = ml[:100] + "\u2026"
                    console.print(f"{B}[dim]{ml}[/dim]")
            else:
                console.print(f"{B}{ic} [dim](no output)[/dim]")

        # ── File edits ──
        elif tool_name in ("edit_file", "write_file"):
            added = sum(1 for l in lines if l.startswith("+") and not l.startswith("+++"))
            removed = sum(1 for l in lines if l.startswith("-") and not l.startswith("---"))
            if added or removed:
                console.print(f"{B}[{colors['success']}]+{added}[/{colors['success']}]/[{colors['error']}]-{removed}[/{colors['error']}] [dim]lines changed[/dim]")
            elif line_count > 0:
                console.print(f"{B}[dim]{line_count} lines written[/dim]")

        # ── Directory listing ──
        elif tool_name in ("list_dir", "ls"):
            items = [l.strip() for l in lines if l.strip()]
            count = len(items)
            preview = ", ".join(items[:6])
            if count > 6:
                preview += f" +{count - 6} more"
            if preview:
                console.print(f"{B}[dim]{count} items: {preview}[/dim]")

        # ── Generic fallback ──
        else:
            if line_count > 3:
                console.print(f"{B}[dim]{line_count} lines[/dim]")
            elif line_count > 0:
                for l in lines[:3]:
                    if len(l) > 100:
                        l = l[:100] + "\u2026"
                    console.print(f"{B}[dim]{l}[/dim]")

    except (json.JSONDecodeError, TypeError, ValueError, AttributeError):
        pass


def _step_prefix(step: int, max_steps: int, colors: dict) -> str:
    """Build a step counter prefix string."""
    if step <= 0:
        return ""
    return f"[bold {colors['accent']}]Step {step}[/bold {colors['accent']}] [dim]\u00b7[/dim] "


def _get_status_dot(status: str, colors: dict) -> tuple[str, str]:
    """Return (character, style) for a tool status indicator."""
    if status == "success":
        return "\u25cf", colors["tool_success"]      # ●
    elif status == "error":
        return "\u25cf", colors["tool_error"]         # ●
    elif status == "running":
        return "\u25cb", colors["tool_pending"]       # ○
    elif status == "pending":
        return "\u25cb", colors["text_muted"]         # ○
    else:
        return "\u25cf", colors["tool_success"]       # ●


# ─────────────────────────────────────────────────────────
# Permission Prompt (styled, not raw input)
# ─────────────────────────────────────────────────────────

def show_permission_prompt(
    action: str,
    detail: str = "",
    options: list[tuple[str, str]] | None = None,
) -> str:
    """Show a styled permission prompt with numbered options.

    Returns the selected option key, or "deny" if cancelled.
    """
    colors = _get_theme_colors()

    if options is None:
        options = [
            ("allow", "Allow this action"),
            ("session", "Allow for this session"),
            ("deny", "No, skip this"),
        ]

    console.print()

    # Header
    header = Text("  ")
    header.append("\u25b3 ", style=f"bold {colors['warning']}")
    header.append("Permission required", style=f"bold {colors['permission_accent']}")
    console.print(header)

    # Action detail
    if action:
        console.print(f"    {action}", style=f"{colors['tool']}")
    if detail:
        # Truncate very long details
        if len(detail) > 200:
            detail = detail[:200] + "\u2026"
        console.print(f"    {detail}", style="dim")

    console.print()

    # Numbered options
    for i, (key, label) in enumerate(options, 1):
        if i == 1:
            # First option highlighted
            console.print(f"    [{colors['permission_accent']}]> {i}. {label}[/{colors['permission_accent']}]")
        else:
            console.print(f"      {i}. {label}", style="dim")

    console.print()

    # Get input
    try:
        choice = console.input(f"  [{colors['text_muted']}]Choose (1-{len(options)}), Enter for 1, Esc to deny: [/{colors['text_muted']}]")
        choice = choice.strip()

        if not choice or choice == "1":
            return options[0][0]  # Default: first option
        elif choice == "y" or choice == "yes":
            return options[0][0]
        elif choice == "n" or choice == "no":
            return "deny"

        try:
            idx = int(choice) - 1
            if 0 <= idx < len(options):
                return options[idx][0]
        except ValueError:
            pass

        return "deny"
    except (EOFError, KeyboardInterrupt):
        return "deny"


# ─────────────────────────────────────────────────────────
# Response Rendering
# ─────────────────────────────────────────────────────────

def show_response(text: str, model: str = "", stream: bool = True) -> None:
    """Render agent response as clean markdown with block-level streaming."""
    code_theme = _get_code_theme()

    console.print()  # breathing room

    max_width = min(console.width - 4, 100)

    if stream and len(text) > 20:
        import time

        chunks = _split_for_streaming(text)
        accumulated = ""
        finalized_count = 0

        with Live(console=console, refresh_per_second=15, transient=True) as live:
            for chunk in chunks:
                accumulated += chunk
                blocks = _split_into_blocks(accumulated)

                # Print any newly finalized blocks permanently
                while finalized_count < len(blocks) - 1:
                    block_text = blocks[finalized_count]
                    try:
                        block_md = Markdown(block_text, code_theme=code_theme)
                    except (ValueError, TypeError):
                        block_md = Text(block_text)
                    live.update(Text(""))
                    console.print(Padding(block_md, (0, 2)), width=max_width)
                    finalized_count += 1

                # Live-update only the active (last) block
                if blocks:
                    active_block = blocks[-1]
                    try:
                        active_md = Markdown(active_block, code_theme=code_theme)
                    except (ValueError, TypeError):
                        active_md = Text(active_block)
                    live.update(Padding(active_md, (0, 2)))

                time.sleep(0.008)

        # Print the final active block permanently
        blocks = _split_into_blocks(accumulated)
        if blocks and finalized_count < len(blocks):
            for i in range(finalized_count, len(blocks)):
                try:
                    block_md = Markdown(blocks[i], code_theme=code_theme)
                except (ValueError, TypeError):
                    block_md = Text(blocks[i])
                console.print(Padding(block_md, (0, 2)), width=max_width)
    else:
        # Non-streaming: render full markdown directly
        try:
            md = Markdown(text, code_theme=code_theme)
        except (ValueError, TypeError):
            md = Text(text)
        console.print(Padding(md, (0, 2)), width=max_width)

    # Model attribution in dim text
    if model:
        console.print(f"  [dim]{model}[/dim]")

    console.print()  # breathing room


def show_response_attribution(model: str, elapsed: float, tokens: int = 0) -> None:
    """Show model + time + tokens after each response."""
    from .spinner import _format_elapsed
    colors = _get_theme_colors()

    t = Text("  ")
    # Model name — trim common suffixes
    model_short = model.replace(":cloud", "").replace(":latest", "")
    if len(model_short) > 30:
        model_short = model_short[:27] + "\u2026"
    t.append(model_short, style="dim")
    t.append(f" \u00b7 {_format_elapsed(elapsed)}", style="dim")
    if tokens > 0:
        if tokens >= 1000:
            t.append(f" \u00b7 {tokens / 1000:.1f}K tokens", style="dim")
        else:
            t.append(f" \u00b7 {tokens} tokens", style="dim")
    console.print(t)


def show_context_summary(
    memory_count: int = 0,
    kg_topic: str = "",
    mood: str = "",
    model: str = "",
    tool_count: int = 0,
    memory_snippets: Optional[list[str]] = None,
) -> None:
    """Show an ultra-minimal context line before responses."""
    if memory_snippets is None:
        memory_snippets = []
    colors = _get_theme_colors()

    parts = []
    if memory_count > 0:
        parts.append(f"{memory_count} memories")
    if kg_topic:
        parts.append(f'KG: "{kg_topic}"')
    if mood:
        parts.append(mood)
    if model:
        short = model.replace(":cloud", "").replace(":latest", "")
        if len(short) > 25:
            short = short[:22] + "\u2026"
        parts.append(short)

    if not parts:
        return

    t = Text("  ")
    t.append("context: ", style=f"{colors['text_muted']}")
    t.append("  \u2502  ".join(parts), style="dim")
    console.print(t)


# ─────────────────────────────────────────────────────────
# Errors, Info, Help
# ─────────────────────────────────────────────────────────

def show_error(message: str) -> None:
    """Display error — clean single line with themed color."""
    colors = _get_theme_colors()
    console.print(f"  [{colors['error']}]\u2717[/{colors['error']}] {message}")


def show_info(message: str) -> None:
    """Display info message — minimal dim text."""
    console.print(f"  [dim]{message}[/dim]")


def show_help() -> None:
    """Display help with clean aligned text, no heavy borders."""
    colors = _get_theme_colors()

    console.print()
    console.print(f"  [bold]Commands & Shortcuts[/bold]")
    console.print()

    sections = [
        ("Keyboard", [
            ("Alt+M", "Model picker"),
            ("Ctrl+L", "Clear screen"),
            ("Ctrl+N", "New session"),
            ("Ctrl+K", "Command palette"),
            ("Ctrl+G", "Open editor for long prompt"),
            ("Shift+Tab", "Cycle permission mode"),
            ("Ctrl+Z", "Rewind to checkpoint"),
            ("Alt+Enter", "Insert newline"),
            ("Ctrl+C / Ctrl+D", "Exit"),
            ("?", "Show this help"),
        ]),
        ("Model & Session", [
            ("/model [name]", "Pick or set model"),
            ("/sessions", "Manage sessions"),
            ("/compact", "Compress conversation"),
            ("/clear", "Clear conversation"),
        ]),
        ("Code & Files", [
            ("/grep <pattern>", "Search code content"),
            ("/search <query>", "Search files and definitions"),
            ("/edit <file>", "Read file with line numbers"),
            ("/project [cmd]", "Project context, indexing, search"),
        ]),
        ("Execution", [
            ("/shell <cmd>", "Execute shell command"),
            ("/plan <task>", "Create and execute a plan"),
            ("/agent <name> <task>", "Run specialist agent"),
            ("/goal <objective>", "Run a goal"),
        ]),
        ("Parallel & Background", [
            ("/fleet <task>", "Run parallel sub-agents"),
            ("/chain step1 -> step2", "Run prompt pipeline"),
            ("& <prompt>", "Run as background task"),
            ("/tasks", "Show background tasks"),
        ]),
        ("Research", [
            ("/research <topic>", "Start research mode"),
            ("/sources", "Show collected sources"),
            ("/export research", "Export to Markdown"),
        ]),
        ("Git", [
            ("/pr", "Create pull request"),
            ("/branch <name>", "Create and switch branch"),
            ("/stash [desc]", "Smart stash"),
            ("/blame file:N", "Explain line history"),
            ("/diff [args]", "Show git diff"),
            ("/git <command>", "Run read-only git commands"),
        ]),
        ("Testing & Watch", [
            ("/test [cmd]", "Run tests"),
            ("/watch", "Monitor files for AI comments"),
        ]),
        ("Utilities", [
            ("/browse <url>", "Browse web pages"),
            ("/speak <text>", "Text-to-speech"),
            ("/recall <query>", "Search memories"),
            ("/context", "Show context usage"),
            ("/rewind", "Rewind to checkpoint"),
            ("/theme [name]", "Switch color theme"),
            ("/mood", "Show emotional state"),
            ("/hook [cmd]", "Manage automation hooks"),
        ]),
        ("Multi-agent", [
            ("/debate <topic>", "Multi-agent debate"),
            ("/fork [name]", "Fork conversation"),
            ("/branches", "List branches"),
            ("/checkout <branch>", "Switch branch"),
            ("/merge <branch>", "Merge branch"),
            ("/undo", "Undo last file edit"),
        ]),
        ("MCP & Audit", [
            ("/mcp [cmd]", "Manage MCP servers"),
            ("/audit [cmd]", "Inspect audit chain"),
        ]),
        ("Autonomous", [
            ("/hand [cmd]", "Manage autonomous Hands"),
            ("/evolve [...]", "Evolve skills with GEPA"),
        ]),
        ("Other", [
            ("/voice", "Start voice mode"),
            ("/retry", "Re-run last prompt"),
            ("/cost", "Show token usage and cost"),
            ("/trust", "Toggle trust mode"),
            ("/quit", "Exit AURA"),
        ]),
    ]

    accent = colors["accent"]
    for section_name, commands in sections:
        console.print(f"  [dim]{section_name}[/dim]")
        for key, action in commands:
            padded_key = key.ljust(28)
            console.print(f"    [{accent}]{padded_key}[/{accent}] [dim]{action}[/dim]")
        console.print()


# ─────────────────────────────────────────────────────────
# Streaming Response Manager
# ─────────────────────────────────────────────────────────

class StreamingResponse:
    """Manages live token streaming to terminal via Rich.

    Clean output — no panels, no borders. Just markdown flowing in.
    When pause() is called (for tool-call display), accumulated text is
    printed permanently. When resume() is called, a fresh Live starts.
    """

    def __init__(self, model: str = "") -> None:
        self._accumulated: str = ""
        self._live: Optional[Live] = None
        self._model: str = model
        self._displayed: bool = False
        self._permanent_len: int = 0
        self._spinner_active: bool = False

    def start(self) -> None:
        """Begin live rendering context with a themed thinking spinner."""
        console.print()  # breathing room
        self._spinner_active = True
        colors = _get_theme_colors()

        # Use shimmer spinner instead of plain "thinking..."
        from .spinner import AuraSpinner
        spinner = AuraSpinner()
        self._live = Live(
            spinner,
            console=console, refresh_per_second=12, transient=True,
        )
        self._live._aura_spinner = spinner
        self._live.start()

    def chunk(self, text: str) -> None:
        """Append a text chunk and re-render NEW content since last pause."""
        if self._spinner_active:
            self._spinner_active = False
        self._accumulated += text
        if self._live:
            new_content = self._accumulated[self._permanent_len:]
            try:
                md = Markdown(new_content, code_theme=_get_code_theme())
                self._live.update(Padding(md, (0, 2)))
            except (ValueError, TypeError):
                self._live.update(Padding(Text(new_content), (0, 2)))

    def pause(self) -> None:
        """Pause live rendering for tool call display."""
        if self._live:
            new_content = self._accumulated[self._permanent_len:]
            if new_content.strip():
                try:
                    md = Markdown(new_content, code_theme=_get_code_theme())
                    self._live.update(Padding(md, (0, 2)))
                except (ValueError, TypeError):
                    self._live.update(Padding(Text(new_content), (0, 2)))
                self._live.transient = False
            self._live.stop()
            self._live = None
            self._permanent_len = len(self._accumulated)

    def resume(self) -> None:
        """Resume live rendering after tool call with a fresh spinner."""
        from .spinner import AuraSpinner
        spinner = AuraSpinner()
        self._live = Live(spinner, console=console, refresh_per_second=12, transient=True)
        self._live._aura_spinner = spinner
        self._live.start()

    def finish(self) -> None:
        """Finalize display — print remaining content, show attribution."""
        if self._live:
            new_content = self._accumulated[self._permanent_len:]
            if new_content.strip():
                try:
                    md = Markdown(new_content, code_theme=_get_code_theme())
                    final = Padding(md, (0, 2))
                except (ValueError, TypeError):
                    final = Padding(Text(new_content), (0, 2))
                self._live.update(final)
                self._live.transient = False
                self._live.stop()
                self._live = None
                self._permanent_len = len(self._accumulated)
                self._displayed = True
            else:
                self._live.stop()
                self._live = None
                self._displayed = bool(self._accumulated.strip())
        else:
            self._displayed = bool(self._accumulated.strip())

        # Model attribution below the response
        if self._model and self._displayed:
            console.print(f"  [dim]{self._model}[/dim]")

        console.print()  # breathing room

    @property
    def displayed(self) -> bool:
        return self._displayed

    @property
    def text(self) -> str:
        return self._accumulated


# ─────────────────────────────────────────────────────────
# Utility: Block splitting for streaming
# ─────────────────────────────────────────────────────────

def _split_for_streaming(text: str) -> list[str]:
    """Split text into word-based chunks for streaming display."""
    words = text.split(" ")
    chunks = []
    chunk_size = 1 if len(text) < 200 else (3 if len(text) < 1000 else 5)
    for i in range(0, len(words), chunk_size):
        chunk_words = words[i : i + chunk_size]
        chunk = " ".join(chunk_words)
        if i + chunk_size < len(words):
            chunk += " "
        chunks.append(chunk)
    return chunks


def _split_into_blocks(text: str) -> list[str]:
    """Split markdown text into top-level blocks by double-newlines.

    Respects code fences (``` blocks stay together as a single block).
    """
    lines = text.split("\n")
    blocks: list[str] = []
    current_lines: list[str] = []
    in_code_fence = False

    for line in lines:
        stripped = line.strip()

        if stripped.startswith("```"):
            in_code_fence = not in_code_fence
            current_lines.append(line)
            continue

        if in_code_fence:
            current_lines.append(line)
            continue

        if stripped == "":
            if current_lines and any(l.strip() for l in current_lines):
                while current_lines and current_lines[-1].strip() == "":
                    current_lines.pop()
                if current_lines:
                    blocks.append("\n".join(current_lines))
                current_lines = []
            else:
                current_lines.append(line)
        else:
            current_lines.append(line)

    if current_lines:
        remaining = "\n".join(current_lines).strip()
        if remaining:
            blocks.append(remaining)

    if not blocks and text.strip():
        blocks.append(text.strip())

    return blocks


# ─────────────────────────────────────────────────────────
# Checkpoint Picker
# ─────────────────────────────────────────────────────────

def show_checkpoint_picker(checkpoints: list[dict[str, Any]]) -> Optional[str]:
    """Display checkpoints in a styled numbered list."""
    import time as _time
    colors = _get_theme_colors()

    if not checkpoints:
        show_info("No checkpoints available.")
        return None

    display = checkpoints[:10]

    console.print()
    console.print(f"  [bold]Checkpoints[/bold]")
    console.print()

    now = _time.time()
    for i, cp in enumerate(display, 1):
        delta = now - cp.get("timestamp", now)
        if delta < 60:
            rel = f"{int(delta)}s ago"
        elif delta < 3600:
            rel = f"{int(delta / 60)}m ago"
        elif delta < 86400:
            rel = f"{int(delta / 3600)}h ago"
        else:
            rel = f"{int(delta / 86400)}d ago"

        label = cp.get("label", "") or "-"
        files = cp.get("files", [])
        file_names = ", ".join(f.get("backup_name", "?") for f in files[:3])
        if len(files) > 3:
            file_names += f" +{len(files) - 3}"

        num = str(i).rjust(2)
        console.print(
            f"  [{colors['accent']}]{num}[/{colors['accent']}]"
            f"  [dim]{rel.ljust(10)}[/dim]  {label.ljust(24)}  [dim]{file_names}[/dim]"
        )

    console.print()

    try:
        raw = console.input(f"  [dim]Pick checkpoint # (or Enter to cancel): [/dim]")
        raw = raw.strip()
        if not raw:
            return None
        idx = int(raw) - 1
        if 0 <= idx < len(display):
            return display[idx]["id"]
        else:
            show_error(f"Invalid selection: {raw}")
            return None
    except (ValueError, EOFError, KeyboardInterrupt):
        return None


def show_rewind_result(success: bool, checkpoint_id: str) -> None:
    """Display the result of a rewind operation."""
    colors = _get_theme_colors()
    if success:
        console.print(f"  [{colors['success']}]\u2713 Rewound to checkpoint {checkpoint_id}[/{colors['success']}]")
    else:
        show_error("Failed to rewind")
