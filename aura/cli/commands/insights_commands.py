"""/insights — usage analytics.

Token usage, cost breakdown, model usage, and interaction counts
over a time window. Mirrors Hermes Agent's `hermes insights` pattern.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Optional

from ..display import console
from .common import command, TIER_STABLE

logger = logging.getLogger(__name__)


@command("/insights", "Usage analytics — tokens, cost, models", tier=TIER_STABLE)
def handle_insights(agent: Any, arg: str, context: dict) -> Optional[str]:
    """Show usage analytics.

    Usage:
        /insights              Last 7 days
        /insights 30           Last 30 days
    """
    days = 7
    if arg:
        try:
            days = max(1, min(365, int(arg.strip())))
        except ValueError:
            pass

    _show_insights(days)
    return None


def _show_insights(days: int) -> None:
    """Show usage analytics for the given time window."""
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    try:
        from ..activity_log import ActivityLog
        log = ActivityLog()
    except Exception:
        console.print("[red]Activity log not available.[/red]")
        return

    cutoff = time.time() - (days * 86400)

    # Get all interactions in the time window
    all_recent = log.get_recent(limit=10000)
    recent = [r for r in all_recent if r.get("timestamp", 0) >= cutoff]

    if not recent:
        console.print(f"[dim]No interactions in the last {days} days.[/dim]")
        return

    # Aggregate stats
    total_interactions = len(recent)
    total_tokens_in = sum(r.get("tokens_in", 0) for r in recent)
    total_tokens_out = sum(r.get("tokens_out", 0) for r in recent)
    total_cost = sum(r.get("cost", 0.0) for r in recent)
    total_tool_calls = sum(r.get("tool_calls", 0) for r in recent)

    # Per-model breakdown
    model_stats: dict[str, dict] = {}
    for r in recent:
        model = r.get("model", "unknown")
        if model not in model_stats:
            model_stats[model] = {"count": 0, "tokens_in": 0, "tokens_out": 0, "cost": 0.0}
        model_stats[model]["count"] += 1
        model_stats[model]["tokens_in"] += r.get("tokens_in", 0)
        model_stats[model]["tokens_out"] += r.get("tokens_out", 0)
        model_stats[model]["cost"] += r.get("cost", 0.0)

    # Per-session breakdown
    session_stats: dict[str, int] = {}
    for r in recent:
        sid = r.get("session_id", "")
        if sid:
            session_stats[sid] = session_stats.get(sid, 0) + 1

    # Build summary text
    summary = Text()
    summary.append(f"Period: last {days} days\n", style="bold")
    summary.append(f"Interactions: {total_interactions}\n", style="cyan")
    summary.append(f"Sessions: {len(session_stats)}\n")
    summary.append(f"Tokens in:  {total_tokens_in:,}\n")
    summary.append(f"Tokens out: {total_tokens_out:,}\n")
    summary.append(f"Total tokens: {total_tokens_in + total_tokens_out:,}\n", style="bold")
    summary.append(f"Cost: ${total_cost:.4f}\n")
    summary.append(f"Tool calls: {total_tool_calls}\n")

    # Model table
    model_table = Table(box=None, padding=(0, 1), show_header=True, header_style="bold")
    model_table.add_column("Model", style="cyan", width=25)
    model_table.add_column("Calls", width=6, justify="right")
    model_table.add_column("Tokens In", width=10, justify="right")
    model_table.add_column("Tokens Out", width=10, justify="right")
    model_table.add_column("Cost", width=10, justify="right")

    for model in sorted(model_stats.keys(), key=lambda m: -model_stats[m]["count"]):
        s = model_stats[model]
        model_table.add_row(
            model[:25],
            str(s["count"]),
            f"{s['tokens_in']:,}",
            f"{s['tokens_out']:,}",
            f"${s['cost']:.4f}",
        )

    from rich.console import Group
    console.print()
    console.print(Panel(
        Group(summary, Text(), model_table),
        title=f"[bold cyan]Insights  ({days}d)[/bold cyan]",
        border_style="cyan",
        padding=(1, 2),
    ))
    console.print()


def _show_insights_shell(days: int = 7) -> None:
    """Shell-compatible version (no Rich dependencies required)."""
    try:
        from aura.cli.activity_log import ActivityLog
        log = ActivityLog()
    except Exception:
        print("Activity log not available.")
        return

    cutoff = time.time() - (days * 86400)
    all_recent = log.get_recent(limit=10000)
    recent = [r for r in all_recent if r.get("timestamp", 0) >= cutoff]

    if not recent:
        print(f"No interactions in the last {days} days.")
        return

    total_tokens_in = sum(r.get("tokens_in", 0) for r in recent)
    total_tokens_out = sum(r.get("tokens_out", 0) for r in recent)
    total_cost = sum(r.get("cost", 0.0) for r in recent)

    print(f"\n  Insights (last {days} days)")
    print(f"  {'─' * 40}")
    print(f"  Interactions: {len(recent)}")
    print(f"  Tokens in:    {total_tokens_in:,}")
    print(f"  Tokens out:   {total_tokens_out:,}")
    print(f"  Total tokens: {total_tokens_in + total_tokens_out:,}")
    print(f"  Cost:         ${total_cost:.4f}")
    print()
