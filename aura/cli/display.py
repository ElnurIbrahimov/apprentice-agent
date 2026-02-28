# D:/Aura/aura/cli/display.py
"""Rich-based display for AURA CLI."""

from rich.console import Console
from rich.markdown import Markdown
from rich.spinner import Spinner
from rich.live import Live

console = Console(highlight=False)


def show_banner():
    console.print("\n[bold cyan]AURA[/bold cyan]  Autonomous Universal Reasoning Agent\n")


def show_thinking(label: str = "Working..."):
    """Context manager — shows spinner while agent runs. Disappears when done."""
    return Live(
        Spinner("dots", text=f"  [dim]{label}[/dim]"),
        console=console,
        refresh_per_second=10,
        transient=True,
    )


def show_tool_call(tool_name: str, description: str = ""):
    """Print a tool call line inline."""
    console.print(f"  [dim cyan]>[/dim cyan] [cyan]{tool_name}[/cyan]  [dim]{description}[/dim]")


def show_response(text: str):
    """Render agent response as markdown."""
    console.print()
    console.print("[bold cyan]AURA[/bold cyan]")
    try:
        console.print(Markdown(text))
    except Exception:
        console.print(text)
    console.print()


def show_error(message: str):
    console.print(f"\n[bold red]Error:[/bold red] {message}\n")


def show_info(message: str):
    console.print(f"[dim]{message}[/dim]")
