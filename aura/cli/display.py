"""Rich-based display for AURA CLI — panels, syntax highlighting, streaming."""

from rich.console import Console
from rich.markdown import Markdown
from rich.padding import Padding
from rich.panel import Panel
from rich.spinner import Spinner
from rich.live import Live
from rich.text import Text

console = Console(highlight=True, soft_wrap=True)


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

    line = Text()
    line.append("  ▸ ", style="dim yellow")
    line.append(tool_name, style="bold yellow")
    if description:
        line.append(f"  {description}", style="dim yellow")
    console.print(line)


def show_response(text: str, model: str = "", stream: bool = True):
    """Render agent response as markdown with a left-border panel.

    Args:
        text: Response text (markdown)
        model: Model name to display
        stream: If True, simulate streaming with Live display
    """
    console.print()

    # Header: label + model name
    label = Text()
    label.append(" ◆ ", style="bold cyan")
    label.append("AURA", style="bold cyan")
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
                    md = Markdown(accumulated, code_theme="monokai")
                except Exception:
                    md = Text(accumulated)
                panel = Panel(
                    md, title=label, title_align="left",
                    border_style="dim cyan", padding=(0, 2), expand=True,
                )
                live.update(Padding(panel, (0, 2)))
                time.sleep(0.015)

    # Final render (clean, complete)
    try:
        md = Markdown(text, code_theme="monokai")
    except Exception:
        md = Text(text)

    panel = Panel(
        md,
        title=label,
        title_align="left",
        border_style="dim cyan",
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
    err = Text()
    err.append("  ✗ ", style="bold red")
    err.append(message, style="red")
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

    # --- Utilities ---
    table.add_row("/browse <url>", "Browse web pages")
    table.add_row("/hook [list|add|remove]", "Manage event hooks")
    table.add_row("/speak <text>", "Text-to-speech")
    table.add_row("/recall <query>", "Search memories")
    table.add_row("/context", "Show context window usage")
    table.add_row("/rewind", "Rewind file changes to a checkpoint")

    table.add_row("", "")

    # --- Exit ---
    table.add_row("/quit, /exit", "Exit AURA")

    console.print()
    console.print(table)
    console.print()
