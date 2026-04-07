from __future__ import annotations

import logging
import os
from typing import Any, Optional

logger = logging.getLogger(__name__)


def _rewind_picker(cp_mgr: Any, console: Any) -> bool:
    import time as _time
    cps = cp_mgr.list_checkpoints()
    if not cps:
        console.print("[dim]No checkpoints available[/dim]")
        return False
    console.print("\n[bold]Rewind to checkpoint:[/bold]")
    for i, cp in enumerate(cps[:10]):
        ts = _time.strftime("%H:%M:%S", _time.localtime(cp["timestamp"]))
        files = ", ".join(
            os.path.basename(f["original_path"])
            for f in cp["files"]
        )
        console.print(f"  {i+1}. [{ts}] {cp['label']} ({files})")
    console.print("  0. Cancel")
    try:
        choice = input("\nSelect checkpoint: ").strip()
    except (EOFError, KeyboardInterrupt):
        return False
    if choice.isdigit() and 0 < int(choice) <= min(10, len(cps)):
        selected = cps[int(choice) - 1]
        if cp_mgr.restore(selected["id"]):
            files = ", ".join(
                os.path.basename(f["original_path"])
                for f in selected["files"]
            )
            console.print(f"[green]Restored: {files}[/green]")
            return True
        else:
            console.print("[red]Restore failed.[/red]")
            return False
    return False


def _display_channel_message(console: Any, msg: Any) -> None:
    """Display an incoming channel message with a styled box."""
    from rich.text import Text
    source_name = msg.source.value.capitalize()
    width = min(console.size.width - 4, 60)
    header = f" {source_name} "
    line_rest = "\u2500" * max(width - len(header) - 2, 4)
    top = f"\u250c\u2500{header}{line_rest}"

    body = Text()
    body.append(f"\u2502 ", style="dim cyan")
    body.append(f"{msg.user_name}: ", style="bold")
    display_text = msg.text[:500]
    if len(msg.text) > 500:
        display_text += "..."
    body.append(display_text)

    bottom = "\u2514" + "\u2500" * (width)

    console.print()
    console.print(f"  [cyan]{top}[/cyan]")
    console.print(f"  ", end="")
    console.print(body)
    console.print(f"  [cyan]{bottom}[/cyan]")


def _display_channel_response(console: Any, msg: Any, response_text: str) -> None:
    """Display an outgoing response to a channel with a styled box."""
    from rich.text import Text
    source_name = msg.source.value.capitalize()
    width = min(console.size.width - 4, 60)
    header = f" \u2192 {source_name} "
    line_rest = "\u2500" * max(width - len(header) - 2, 4)
    top = f"\u250c\u2500{header}{line_rest}"

    # Truncate long responses for display
    display_text = response_text[:300]
    if len(response_text) > 300:
        display_text += "..."

    body = Text()
    body.append(f"\u2502 ", style="dim green")
    body.append("AURA: ", style="bold green")
    body.append(display_text)

    bottom = "\u2514" + "\u2500" * (width)

    console.print(f"  [green]{top}[/green]")
    console.print(f"  ", end="")
    console.print(body)
    console.print(f"  [green]{bottom}[/green]")


def run_chat_mode(agent: Any, speak: bool = False, trust: bool = False, model: Optional[str] = None, verbose: bool = False, tier: Optional[str] = None, bridge: Any = None, preference: Optional[str] = None) -> None:
    from .chat_session import ChatSession
    session = ChatSession(agent, speak=speak, trust=trust, model=model,
                          verbose=verbose, tier=tier, bridge=bridge,
                          preference=preference)
    session.run()
