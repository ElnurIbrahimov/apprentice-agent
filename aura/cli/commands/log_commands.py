"""Top-level `aura log ...` subcommand handlers.

Previously this dispatcher was inlined directly in main.py (~25 lines) which
broke the pattern every other subcommand follows — handler lives in
aura/cli/commands/*_commands.py, registered in aura/core/commands.py's
handle_subcommand() dispatch table. This module restores that symmetry.
"""
from __future__ import annotations

import argparse

from aura.cli.display import console, show_info


def cmd_log(args: argparse.Namespace) -> int:
    """Dispatch `aura log <action>`."""
    from aura.cli.activity_log import ActivityLog

    log = ActivityLog()
    action = getattr(args, "action", None) or "recent"

    if action == "search":
        query = " ".join(args.query) if isinstance(args.query, list) else str(args.query or "")
        if not query.strip():
            console.print("[yellow]Usage: aura log search <query>[/yellow]")
            return 1
        results = log.search(query, limit=args.limit)
        for r in results:
            console.print(f"[cyan][{r['model']}][/cyan] {r['prompt'][:80]}")
            show_info(f"-> {r['response'][:120]}")
        return 0

    if action == "stats":
        stats = log.get_stats()
        for k, v in stats.items():
            console.print(f"  [bold]{k}[/bold]: {v}")
        return 0

    if action == "export":
        if not args.session:
            console.print("[yellow]Usage: aura log export --session <session_id>[/yellow]")
            return 1
        md = log.export_session(args.session, format=args.log_format)
        # Raw output — this is designed to be piped.
        print(md)
        return 0

    # Default: recent
    for r in log.get_recent(args.limit):
        show_info(f"{r['prompt'][:80]}")
    return 0
