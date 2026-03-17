"""Rich-based display for AURA CLI — panels, syntax highlighting, streaming."""

from rich.console import Console
from rich.markdown import Markdown
from rich.padding import Padding
from rich.panel import Panel
from rich.spinner import Spinner
from rich.live import Live
from rich.text import Text

from aura.cli.tool_output import ToolOutputRenderer, format_elapsed

console = Console(highlight=True, soft_wrap=True)

# Module-level tool output renderer (lazy init)
_tool_renderer = None

def _get_tool_renderer():
    global _tool_renderer
    if _tool_renderer is None:
        _tool_renderer = ToolOutputRenderer(console=console)
    return _tool_renderer


def show_banner():
    """Display ASCII art banner with gradient colors."""
    from .banner import get_banner, get_welcome_line
    from aura import __version__
    width = console.size.width
    console.print(get_banner(width))
    console.print(get_welcome_line(__version__))
    console.print()


def show_welcome_info(agent):
    """Show a brief info line after the banner: model, session, tool count."""
    model = "auto"
    try:
        model = agent.brain._model_override or "auto"
    except Exception:
        pass

    tool_count = 0
    try:
        tool_count = len(agent.tools)
    except Exception:
        pass

    session = "new"
    try:
        if hasattr(agent, 'memory') and hasattr(agent.memory, 'session_id'):
            sid = agent.memory.session_id
            if sid:
                session = str(sid)[:8]
    except Exception:
        pass

    info = Text()
    info.append("  Model: ", style="dim")
    info.append(model, style="dim bold")
    info.append("  |  ", style="dim")
    info.append("Session: ", style="dim")
    info.append(session, style="dim bold")
    info.append("  |  ", style="dim")
    info.append("Tools: ", style="dim")
    info.append(f"{tool_count} loaded", style="dim bold")
    console.print(info)
    console.print()


def show_status_bar(
    model: str = "auto",
    project_type: str = "",
    session_title: str = "",
    message_count: int = 0,
    thinking: bool = False,
    elapsed: float = 0.0,
    cost_usd: float = 0.0,
    tier: str = "",
    token_used: int = 0,
    token_limit: int = 128000,
    permission_mode: str = "careful",
    bg_indicator: str = "",
    research_indicator: str = "",
    mood_indicator: str = "",
):
    """Print the status bar line."""
    from .status_bar import build_status_bar
    width = console.size.width
    bar = build_status_bar(
        model=model, project_type=project_type,
        session_title=session_title, message_count=message_count,
        width=width, thinking=thinking, elapsed=elapsed,
        cost_usd=cost_usd, tier=tier,
        token_used=token_used, token_limit=token_limit,
        permission_mode=permission_mode,
        bg_indicator=bg_indicator,
        research_indicator=research_indicator,
        mood_indicator=mood_indicator,
    )
    console.print(bar, style="on grey11", end="\n")


def show_thinking(label: str = "Working..."):
    """Context manager — shows spinner while agent runs. Disappears when done."""
    return Live(
        Spinner("dots", text=f"  [dim]{label}[/dim]"),
        console=console,
        refresh_per_second=10,
        transient=True,
    )


def show_tool_call(tool_name: str, description: str = "", result=None):
    """Print a tool call in a compact styled format.

    If result is provided and contains diff info for edit/write, show a compact diff summary.
    If result is provided with substantial output, render via ToolOutputRenderer.
    """
    if tool_name in ("edit_file", "write_file") and result and isinstance(result, dict) and result.get("diff"):
        try:
            from aura.cli.diff_viewer import render_diff_compact
            filename = result.get("path", "file")
            filename = filename.split("/")[-1].split("\\")[-1]
            summary = render_diff_compact(
                result.get("old_content", ""),
                result.get("new_content", ""),
                filename=filename,
            )
            console.print(f"  {summary}")
            return
        except Exception:
            pass  # Fall through to default display

    # Get theme-aware tool color
    try:
        from aura.cli.themes import get_theme
        tool_color = get_theme().tool_color
    except Exception:
        tool_color = "yellow"

    line = Text()
    line.append("  ▸ ", style=f"dim {tool_color}")
    line.append(tool_name, style=f"bold {tool_color}")
    if description:
        line.append(f"  {description}", style=f"dim {tool_color}")
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


def show_response(text: str, model: str = "", stream: bool = True):
    """Render agent response as markdown with a left-border panel.

    Args:
        text: Response text (markdown)
        model: Model name to display
        stream: If True, simulate streaming with Live display
    """
    # Get theme colors
    try:
        from aura.cli.themes import get_theme
        theme = get_theme()
        border_style = f"dim {theme.response_border}"
        header_style = theme.response_header
        code_theme = theme.code_theme
    except Exception:
        border_style = "dim cyan"
        header_style = "bold cyan"
        code_theme = "monokai"

    console.print()

    # Header: label + model name
    label = Text()
    label.append(" ◆ ", style=header_style)
    label.append("AURA", style=header_style)
    if model:
        label.append(f"  ({model})", style="dim")

    if stream and len(text) > 20:
        # Streaming effect: render progressively longer text
        import time
        chunks = _split_for_streaming(text)
        accumulated = ""
        with Live(console=console, refresh_per_second=15, transient=True) as live:
            for chunk in chunks:
                accumulated += chunk
                try:
                    md = Markdown(accumulated, code_theme=code_theme)
                except Exception:
                    md = Text(accumulated)
                panel = Panel(
                    md, title=label, title_align="left",
                    border_style=border_style, padding=(0, 2), expand=True,
                )
                live.update(Padding(panel, (0, 2)))
                time.sleep(0.015)

    # Final render (clean, complete)
    try:
        md = Markdown(text, code_theme=code_theme)
    except Exception:
        md = Text(text)

    panel = Panel(
        md,
        title=label,
        title_align="left",
        border_style=border_style,
        padding=(0, 2),
        expand=True,
    )
    console.print(Padding(panel, (0, 2)))
    console.print()


def _split_for_streaming(text: str) -> list[str]:
    """Split text into chunks for streaming display — by words, not chars."""
    words = text.split(' ')
    chunks = []
    # Start with bigger chunks, slow down for first few words
    for i, word in enumerate(words):
        if i < 3:
            chunks.append(word + ' ')
        else:
            # Group 2-4 words per chunk for speed
            if i % 3 == 0:
                chunks.append(word + ' ')
            else:
                if chunks:
                    chunks[-1] += word + ' '
                else:
                    chunks.append(word + ' ')
    return chunks


def show_error(message: str):
    """Display error in a styled format."""
    try:
        from aura.cli.themes import get_theme
        error_color = get_theme().error_color
    except Exception:
        error_color = "red"
    err = Text()
    err.append("  ✗ ", style=f"bold {error_color}")
    err.append(message, style=error_color)
    console.print(err)


def show_info(message: str):
    """Display info message."""
    info = Text()
    info.append("  ● ", style="dim cyan")
    info.append(message, style="dim")
    console.print(info)


def show_help():
    """Display quick help with available commands and shortcuts."""
    from rich.table import Table

    table = Table(
        show_header=True, header_style="bold cyan",
        border_style="dim", padding=(0, 2),
        title="[bold]Commands & Shortcuts[/bold]",
    )
    table.add_column("Key / Command", style="cyan", width=24)
    table.add_column("Action", style="white")

    # --- Keyboard shortcuts ---
    table.add_row("Alt+M", "Model picker (interactive)")
    table.add_row("Ctrl+L", "Clear screen")
    table.add_row("Ctrl+N", "New session")
    table.add_row("Ctrl+K", "Command palette")
    table.add_row("Ctrl+G", "Open editor for long prompt")
    table.add_row("Shift+Tab", "Cycle permission mode (Plan / Careful / Auto-Edit / Full Auto)")
    table.add_row("Esc Esc", "Rewind to checkpoint")
    table.add_row("Ctrl+C / Ctrl+D", "Exit")
    table.add_row("?", "Show this help")

    table.add_row("", "")

    # --- Model & session ---
    table.add_row("/model [name]", "Pick model interactively or set by name")
    table.add_row("/sessions", "Manage sessions (list, switch, delete)")
    table.add_row("/compact", "Compress conversation history")
    table.add_row("/clear", "Clear conversation history")

    table.add_row("", "")

    # --- Code & files ---
    table.add_row("/grep <pattern>", "Search code content")
    table.add_row("/search, /find <query>", "Search files and definitions")
    table.add_row("/edit <file>", "Read file with line numbers")
    table.add_row("/project [info|index|search]", "Project context, indexing, semantic search")

    table.add_row("", "")

    # --- Execution ---
    table.add_row("/shell, /bash, /run <cmd>", "Execute shell command")
    table.add_row("/plan <task>", "Create and execute a plan")
    table.add_row("/agent <name> <task>", "Run specialist agent")
    table.add_row("/goal <objective>", "Run a goal")

    table.add_row("", "")

    # --- Parallel & background ---
    table.add_row("/fleet <task>", "Run parallel sub-agents on decomposed task")
    table.add_row("& <prompt>", "Run prompt as background task")
    table.add_row("/tasks", "Show background tasks")

    table.add_row("", "")

    # --- Research ---
    table.add_row("/research <topic>", "Start research mode with citation tracking")
    table.add_row("/sources", "Show collected research sources")
    table.add_row("/export research", "Export research session to Markdown")

    table.add_row("", "")

    # --- Emotional & hooks ---
    table.add_row("/mood", "Show current emotional state")
    table.add_row("/hook [list|add|remove]", "Manage automation hooks")

    table.add_row("", "")

    # --- Utilities ---
    table.add_row("/browse <url>", "Browse web pages")
    table.add_row("/speak <text>", "Text-to-speech")
    table.add_row("/recall <query>", "Search memories")
    table.add_row("/context", "Show context window usage")
    table.add_row("/rewind", "Rewind file changes to a checkpoint")
    table.add_row("/theme [name]", "Switch color theme (dark, light, monokai, dracula, solarized, nord)")

    table.add_row("", "")

    # --- Exit ---
    table.add_row("/quit, /exit", "Exit AURA")

    console.print()
    console.print(table)
    console.print()
