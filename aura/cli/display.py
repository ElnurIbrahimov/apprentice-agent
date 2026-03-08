"""Rich-based display for AURA CLI — panels, syntax highlighting, streaming."""

from rich.console import Console
from rich.markdown import Markdown
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


def show_status_bar(
    model: str = "auto",
    project_type: str = "",
    session_title: str = "",
    message_count: int = 0,
    thinking: bool = False,
    elapsed: float = 0.0,
):
    """Print the status bar line."""
    from .status_bar import build_status_bar
    width = console.size.width
    bar = build_status_bar(
        model=model, project_type=project_type,
        session_title=session_title, message_count=message_count,
        width=width, thinking=thinking, elapsed=elapsed,
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


def show_tool_call(tool_name: str, description: str = ""):
    """Print a tool call in a compact styled format."""
    line = Text()
    line.append("  ▸ ", style="cyan")
    line.append(tool_name, style="bold cyan")
    if description:
        line.append(f"  {description}", style="dim")
    console.print(line)


def show_response(text: str):
    """Render agent response as markdown with syntax highlighting."""
    console.print()
    label = Text()
    label.append("  ◆ ", style="bold cyan")
    label.append("AURA", style="bold cyan")
    console.print(label)
    try:
        md = Markdown(text, code_theme="monokai")
        console.print(md, padding=(0, 4))
    except Exception:
        console.print(text, padding=(0, 4))
    console.print()


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
    table.add_column("Key / Command", style="cyan", width=22)
    table.add_column("Action", style="white")

    table.add_row("Ctrl+M", "Switch model mid-session")
    table.add_row("Ctrl+C / Ctrl+D", "Exit")
    table.add_row("", "")
    table.add_row("/model [name]", "Show or set model")
    table.add_row("/sessions", "List / switch sessions")
    table.add_row("/compact", "Compress conversation history")
    table.add_row("/clear", "Clear history")
    table.add_row("", "")
    table.add_row("/grep <pattern>", "Search code content")
    table.add_row("/find def <name>", "Find definition")
    table.add_row("/edit <file>", "View file with line numbers")
    table.add_row("/project index", "Build semantic code index")
    table.add_row("/project search <q>", "Semantic code search")
    table.add_row("/shell <cmd>", "Run shell command")
    table.add_row("/agent <name> <task>", "Route to specialist agent")
    table.add_row("", "")
    table.add_row("/plan <task>", "Generate execution plan")
    table.add_row("/browse <url>", "Open URL in browser")

    console.print()
    console.print(table)
    console.print()
