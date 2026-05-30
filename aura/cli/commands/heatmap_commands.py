"""Cognitive heatmap — visualize tokens spent per tool and per file."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional
from .common import command, TIER_BETA, TIER_EXPERIMENTAL, TIER_STABLE

logger = logging.getLogger(__name__)

try:
    from aura.cli.display import console
except ImportError:
    from rich.console import Console
    console = Console()


_SPARKS = "▁▂▃▄▅▆▇█"


def _spark(value: int, maximum: int, width: int = 16) -> str:
    if maximum <= 0:
        return "." * width
    filled = max(1, int(round((value / maximum) * width)))
    block = _SPARKS[-1]
    return block * filled + " " * (width - filled)


def _render(title: str, rows: list[tuple[str, int]], *, top_n: int = 20) -> None:
    from rich.table import Table
    if not rows:
        console.print(f"  [dim]{title}: no data yet[/]")
        return
    rows_sorted = sorted(rows, key=lambda r: r[1], reverse=True)[:top_n]
    max_val = rows_sorted[0][1] if rows_sorted else 1
    total = sum(v for _, v in rows)
    t = Table(title=f"{title}  [dim](total {total:,} tok)[/]", show_header=True,
              header_style="bold", box=None, padding=(0, 1))
    t.add_column("key", style="cyan")
    t.add_column("tokens", justify="right")
    t.add_column("share", justify="right")
    t.add_column("")
    for key, val in rows_sorted:
        pct = 100.0 * val / total if total else 0.0
        t.add_row(
            (key[:48] + "…") if len(key) > 48 else key,
            f"{val:,}",
            f"{pct:4.1f}%",
            _spark(val, max_val),
        )
    console.print(t)


@command("/heatmap",  "Show cognitive heatmap (tokens by tool/file)",     tier=TIER_BETA)


def handle_heatmap(agent, arg, context) -> Optional[str]:
    """Slash command: /heatmap — show the current session's heatmap."""
    from aura.cli.context import get_ctx
    ctx = get_ctx()
    if not (ctx and ctx.agentic_loop):
        console.print("  No active agentic loop.")
        return None
    loop = ctx.agentic_loop
    tokens_by_tool = dict(getattr(loop, "_tokens_by_tool", {}) or {})
    tokens_by_file = dict(getattr(loop, "_tokens_by_file", {}) or {})
    console.print()
    _render("Tokens by tool", list(tokens_by_tool.items()))
    console.print()
    _render("Tokens by file (top 20)", list(tokens_by_file.items()))
    console.print()


def cmd_heatmap(args: argparse.Namespace) -> int:
    """`aura heatmap [--session <id>]` — show a saved heatmap or say none found."""
    session = getattr(args, "heatmap_session", None)
    try:
        from aura.paths import AURA_DATA_DIR
        heatmap_dir = AURA_DATA_DIR / "heatmaps"
        if not heatmap_dir.exists():
            console.print("  [dim]No heatmaps recorded yet (run a session first).[/]")
            return 0

        target: Optional[Path]
        if session:
            target = heatmap_dir / f"{session}.json"
            if not target.exists():
                console.print(f"  [yellow]No heatmap for session {session}[/]")
                return 1
        else:
            candidates = sorted(heatmap_dir.glob("*.json"),
                                key=lambda p: p.stat().st_mtime, reverse=True)
            if not candidates:
                console.print("  [dim]No heatmaps recorded yet.[/]")
                return 0
            target = candidates[0]

        import json as _json
        data = _json.loads(target.read_text(encoding="utf-8", errors="ignore"))
        console.print(f"\n  [bold]Heatmap[/] session [cyan]{target.stem}[/]\n")
        _render("Tokens by tool", list((data.get("tokens_by_tool") or {}).items()))
        console.print()
        _render("Tokens by file (top 20)", list((data.get("tokens_by_file") or {}).items()))
        console.print()
        return 0
    except Exception as e:
        console.print(f"[red]Heatmap read failed:[/] {e}")
        return 1


def persist_heatmap(loop) -> Optional[Path]:
    """Persist the current loop's heatmap to aura_data/heatmaps/<session>.json."""
    try:
        from aura.paths import AURA_DATA_DIR
        heatmap_dir = AURA_DATA_DIR / "heatmaps"
        heatmap_dir.mkdir(parents=True, exist_ok=True)
        session_id = "default"
        if getattr(loop, "session", None) is not None:
            session_id = getattr(loop.session, "session_id", "default") or "default"
        path = heatmap_dir / f"{session_id}.json"
        import json as _json
        payload = {
            "tokens_by_tool": dict(getattr(loop, "_tokens_by_tool", {}) or {}),
            "tokens_by_file": dict(getattr(loop, "_tokens_by_file", {}) or {}),
        }
        path.write_text(_json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return path
    except Exception:
        logger.debug("heatmap_persist_failed", exc_info=True)
        return None
