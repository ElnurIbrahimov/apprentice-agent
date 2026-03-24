"""Rich-based display for AURA CLI — clean, minimal, Claude Code aesthetic."""
from __future__ import annotations

import logging
import os
from typing import Any, Optional

logger = logging.getLogger(__name__)

from rich.console import Console
from rich.markdown import Markdown
from rich.padding import Padding
from rich.spinner import Spinner
from rich.live import Live
from rich.text import Text

from aura.cli.tool_output import ToolOutputRenderer, format_elapsed

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


def show_banner() -> None:
    """Display clean 1-line banner."""
    from .banner import get_welcome_line
    from aura import __version__
    console.print()
    console.print(get_welcome_line(__version__))
    console.print()


def show_welcome_info(agent: Any) -> None:
    """Show a brief info line after the banner: model, tools."""
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

    if tool_count:
        console.print(f"  [dim]{model} \u2022 {tool_count} tools[/dim]")
    else:
        console.print(f"  [dim]{model}[/dim]")
    console.print()


def show_status_bar(
    model: str = "auto",
    cost_usd: float = 0.0,
    token_used: int = 0,
    token_limit: int = 128000,
    permission_mode: str = "careful",
    # Accept and ignore legacy kwargs so callers don't break
    **_ignored,
) -> None:
    """Print the minimal status bar line."""
    from .status_bar import build_status_bar
    bar = build_status_bar(
        model=model,
        cost_usd=cost_usd,
        token_used=token_used,
        token_limit=token_limit,
        permission_mode=permission_mode,
    )
    console.print(bar, style="on grey11", end="\n")


def show_thinking(label: str = "Working...") -> Live:
    """Context manager -- shows spinner while agent runs. Disappears when done."""
    return Live(
        Spinner("dots", text=f"  [dim]{label}[/dim]"),
        console=console,
        refresh_per_second=10,
        transient=True,
    )


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
}


def get_thinking_label(tool_name: str | None = None) -> str:
    """Return a phase-aware thinking label based on the current tool being used."""
    if tool_name and tool_name in _TOOL_PHASE_VERBS:
        return _TOOL_PHASE_VERBS[tool_name]
    return "Thinking..."


def show_tool_call(tool_name: str, description: str = "", result: Any = None, elapsed: float = 0.0) -> None:
    """Print a tool call as a single-line compact badge.

    Style: "  > tool_name description (elapsed)"
    No panels, no multi-line, no emoji.
    """
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
            console.print(f"  {summary}")
            return
        except (ImportError, ValueError, KeyError, TypeError):
            pass

    # Get theme-aware tool color
    try:
        from aura.cli.themes import get_theme
        tool_color = get_theme().tool_color
    except (ImportError, AttributeError):
        tool_color = "cyan"

    # Single-line badge
    line = Text()
    line.append("  > ", style=f"dim {tool_color}")
    line.append(tool_name, style=f"bold {tool_color}")
    if description:
        line.append(f" {description}", style="dim")
    if time_str:
        line.append(f" {time_str}", style="dim")
    console.print(line)

    # Render substantial tool output via ToolOutputRenderer
    if result and isinstance(result, (dict, str)):
        try:
            import json
            if isinstance(result, str):
                parsed = json.loads(result)
            else:
                parsed = result
            if isinstance(parsed, dict) and not parsed.get("error"):
                output = parsed.get("output", parsed.get("content", parsed.get("result", "")))
                if isinstance(output, str) and len(output) > 50:
                    _get_tool_renderer().render_tool_result(tool_name, parsed)
        except (json.JSONDecodeError, TypeError, ValueError):
            pass


def show_response(text: str, model: str = "", stream: bool = True) -> None:
    """Render agent response as clean markdown. No panels. No borders.

    The model's text IS the experience -- just clean markdown with breathing room.
    """
    code_theme = _get_code_theme()

    console.print()  # breathing room

    max_width = min(console.width - 4, 100)

    if stream and len(text) > 20:
        # Block-level streaming: freeze finalized blocks, only re-render the active one
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


def _split_for_streaming(text: str) -> list[str]:
    """Split text into word-based chunks for streaming display."""
    words = text.split(' ')
    chunks = []
    chunk_size = 1 if len(text) < 200 else (3 if len(text) < 1000 else 5)
    for i in range(0, len(words), chunk_size):
        chunk_words = words[i:i+chunk_size]
        chunk = ' '.join(chunk_words)
        if i + chunk_size < len(words):
            chunk += ' '
        chunks.append(chunk)
    return chunks


def _split_into_blocks(text: str) -> list[str]:
    """Split markdown text into top-level blocks by double-newlines.

    Respects code fences (``` blocks stay together as a single block).
    """
    lines = text.split('\n')
    blocks: list[str] = []
    current_lines: list[str] = []
    in_code_fence = False

    for line in lines:
        stripped = line.strip()

        if stripped.startswith('```'):
            in_code_fence = not in_code_fence
            current_lines.append(line)
            continue

        if in_code_fence:
            current_lines.append(line)
            continue

        if stripped == '':
            if current_lines and any(l.strip() for l in current_lines):
                while current_lines and current_lines[-1].strip() == '':
                    current_lines.pop()
                if current_lines:
                    blocks.append('\n'.join(current_lines))
                current_lines = []
            else:
                current_lines.append(line)
        else:
            current_lines.append(line)

    if current_lines:
        remaining = '\n'.join(current_lines).strip()
        if remaining:
            blocks.append(remaining)

    if not blocks and text.strip():
        blocks.append(text.strip())

    return blocks


def show_context_summary(
    memory_count: int = 0,
    kg_topic: str = "",
    mood: str = "",
    model: str = "",
    tool_count: int = 0,
    memory_snippets: Optional[list[str]] = None,
) -> None:
    """Show an ultra-minimal context line. Only if there's something noteworthy."""
    if memory_snippets is None:
        memory_snippets = []

    parts = []
    if memory_count > 0:
        parts.append(f"{memory_count} memories")
    if mood:
        parts.append(mood)
    if model:
        short = model.replace(":cloud", "").replace(":latest", "")
        if len(short) > 25:
            short = short[:22] + "..."
        parts.append(short)

    if not parts:
        return

    console.print(f"  [dim]{'  //  '.join(parts)}[/dim]")


def show_error(message: str) -> None:
    """Display error -- clean single line, no panel."""
    console.print(f"  [red]x[/red] {message}")


def show_info(message: str) -> None:
    """Display info message -- minimal dim text."""
    console.print(f"  [dim]{message}[/dim]")


def show_help() -> None:
    """Display help with clean aligned text, no heavy borders."""
    console.print()
    console.print("  [bold]Commands & Shortcuts[/bold]")
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

    for section_name, commands in sections:
        console.print(f"  [dim]{section_name}[/dim]")
        for key, action in commands:
            # Right-pad the key column for alignment
            padded_key = key.ljust(28)
            console.print(f"    [cyan]{padded_key}[/cyan] [dim]{action}[/dim]")
        console.print()


class StreamingResponse:
    """Manages live token streaming to terminal via Rich.

    Clean output -- no panels, no borders. Just markdown flowing in.

    When pause() is called (for tool-call display), the accumulated
    text so far is printed permanently, then Live is stopped.
    When resume() is called, a fresh Live starts with empty content.
    """

    def __init__(self, model: str = "") -> None:
        self._accumulated: str = ""
        self._live: Optional[Live] = None
        self._model: str = model
        self._displayed: bool = False
        self._permanent_len: int = 0

    def start(self) -> None:
        """Begin live rendering context."""
        console.print()  # breathing room before response
        self._live = Live("", console=console, refresh_per_second=15, transient=True)
        self._live.start()

    def chunk(self, text: str) -> None:
        """Append a text chunk and re-render only NEW content since last pause."""
        self._accumulated += text
        if self._live:
            new_content = self._accumulated[self._permanent_len:]
            try:
                md = Markdown(new_content, code_theme=_get_code_theme())
                self._live.update(Padding(md, (0, 2)))
            except (ValueError, TypeError):
                self._live.update(Padding(Text(new_content), (0, 2)))

    def pause(self) -> None:
        """Pause live rendering for tool call display.

        Prints accumulated-since-last-resume text permanently, then stops Live.
        """
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
        """Resume live rendering after tool call."""
        self._live = Live("", console=console, refresh_per_second=15, transient=True)
        self._live.start()

    def finish(self) -> None:
        """Finalize display -- print remaining content cleanly, no panel wrapping."""
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

        console.print()  # breathing room after response

    @property
    def displayed(self) -> bool:
        """Whether the response was already rendered to the terminal."""
        return self._displayed

    @property
    def text(self) -> str:
        return self._accumulated


def show_checkpoint_picker(checkpoints: list[dict[str, Any]]) -> Optional[str]:
    """Display checkpoints in a clean numbered list.

    Returns selected checkpoint ID, or None if cancelled.
    """
    import time as _time

    if not checkpoints:
        show_info("No checkpoints available.")
        return None

    display = checkpoints[:10]

    console.print()
    console.print("  [bold]Checkpoints[/bold]")
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
        console.print(f"  [bold]{num}[/bold]  [dim]{rel.ljust(10)}[/dim]  {label.ljust(24)}  [dim]{file_names}[/dim]")

    console.print()

    try:
        raw = console.input("[dim]  Pick checkpoint # (or Enter to cancel): [/dim]")
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
    if success:
        console.print(f"  [green]Rewound to checkpoint {checkpoint_id}[/green]")
    else:
        console.print(f"  [red]x[/red] Failed to rewind")
