"""Handle /snippet command: save, use, list, delete prompt snippets."""
from __future__ import annotations

from typing import Optional
from .common import command, TIER_BETA, TIER_EXPERIMENTAL, TIER_STABLE


@command("/snippet",  "Manage prompt templates/snippets",                 tier=TIER_BETA)
def handle_snippet(agent, arg: str, context: dict) -> Optional[str]:
    """Manage prompt snippets. /snippet save <name> <text>, /snippet <name>, /snippet list, /snippet delete <name>."""
    from ..display import console, show_error, show_info
    from ..snippets import SnippetManager

    mgr = SnippetManager()
    parts = arg.split(maxsplit=2) if arg else []
    subcmd = parts[0].lower() if parts else ""

    if subcmd == "save" and len(parts) >= 3:
        name, text = parts[1], parts[2]
        mgr.save_snippet(name, text)
        show_info(f"Saved snippet: {name}")
    elif subcmd == "list" or not subcmd:
        snippets = mgr.list_all()
        if not snippets:
            console.print("  [dim]No snippets saved. Use: /snippet save <name> <text>[/dim]")
        else:
            console.print()
            for name, text in snippets.items():
                preview = text[:60].replace("\n", " ")
                if len(text) > 60:
                    preview += "..."
                console.print(f"  [cyan]{name:15s}[/cyan] {preview}")
            console.print()
    elif subcmd == "delete" and len(parts) >= 2:
        if mgr.delete(parts[1]):
            show_info(f"Deleted snippet: {parts[1]}")
        else:
            show_error(f"Snippet not found: {parts[1]}")
    else:
        # Try to use as snippet name
        text = mgr.get(subcmd)
        if text:
            show_info(f"Using snippet: {subcmd}")
            return text  # Return text to be processed as input
        else:
            show_error(f"Unknown snippet: {subcmd}. Use /snippet list to see available snippets.")

    return None
