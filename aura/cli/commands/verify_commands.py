"""/verify — on-demand verification of the current session's edited files.

Re-runs the same VerificationStage the agentic loop uses automatically, but
scoped to whatever the user edited in this session (via AgenticLoop._hot_files).
Useful after a batch of /plan edits or when resuming a session to confirm
state is still green.

Usage:
  /verify          → run whatever `verification.on_edit` is configured (default: typecheck)
  /verify tests    → force tests only
  /verify typecheck → force typecheck only
  /verify both     → force typecheck + tests
"""
from __future__ import annotations

import logging
from typing import Optional

from ..context import get_ctx
from ..display import console

logger = logging.getLogger(__name__)


def handle_verify(agent, arg, context) -> Optional[str]:
    """Re-run verification on the current session's edited files."""
    ctx = get_ctx()
    if ctx is None or ctx.agentic_loop is None:
        console.print("  [yellow]No active session — /verify requires an agentic loop.[/yellow]")
        return None

    agentic = ctx.agentic_loop

    # Pull the files edited in this session. _hot_files is an LRU of up to
    # 10 entries (read/edit/write tools all touch it). For /verify we care
    # specifically about the mutations — but the loop doesn't separate reads
    # from writes in _hot_files, so we accept "recently touched" as the
    # approximation.
    hot = [p for p in getattr(agentic, "_hot_files", []) or [] if p]
    if not hot:
        console.print("  [dim]No files edited in this session yet.[/dim]")
        return None

    # Resolve the project root from the loop (it's set during build_session_bootstrap).
    project_root = getattr(agentic, "project_root", None) or "."

    # Determine mode: either arg override or inherit whatever VerificationStage was
    # built with (typecheck by default).
    arg_stripped = (arg or "").strip().lower()
    mode_override = None
    if arg_stripped in ("typecheck", "tests", "both", "none"):
        mode_override = arg_stripped

    from aura.core.verification_stage import VerificationStage

    # Reuse the loop's stage if present; otherwise build a fresh one from config.
    stage = getattr(agentic, "_verification_stage", None)
    if stage is None:
        aura_config = getattr(agentic, "aura_config", None) or {}
        shell_tool = getattr(agentic, "shell_tool", None)
        stage = VerificationStage(
            project_root=project_root,
            aura_config=aura_config,
            shell_tool=shell_tool,
        )

    if mode_override:
        # Respect the explicit ask without mutating the stage's configured default.
        original_mode = stage.mode
        stage.mode = mode_override
        try:
            outcome = stage.run(hot, emitter=None, session_id=_current_session_id(ctx))
        finally:
            stage.mode = original_mode
    else:
        outcome = stage.run(hot, emitter=None, session_id=_current_session_id(ctx))

    _render_result(outcome, hot)
    return None


def _current_session_id(ctx) -> str:
    try:
        return getattr(ctx.session, "session_id", "") or ""
    except AttributeError:
        return ""


def _render_result(outcome, changed_files: list) -> None:
    """Show a compact Rich panel summarizing the verification run."""
    from rich.panel import Panel
    from rich.text import Text

    body = Text()
    body.append(f"Verification · {len(changed_files)} file(s) · ", style="bold")
    body.append(f"{outcome.duration_s:.1f}s", style="dim")
    body.append("\n\n")

    if outcome.mode == "none":
        body.append("Mode is 'none' — nothing to run.\n", style="dim")
    elif outcome.skipped_reason and not outcome.stages:
        body.append(f"Skipped: {outcome.skipped_reason}\n", style="dim")
    else:
        for stage in outcome.stages:
            name = stage.get("name", "?")
            runner = stage.get("runner", "?")
            dur = stage.get("duration_s", 0.0)
            if stage.get("success"):
                body.append("  ✓ ", style="green bold")
                body.append(f"{name:10s} ", style="bold")
                body.append(f"{runner:10s} ", style="cyan")
                body.append(f"{dur:.1f}s  ok", style="dim")
                body.append("\n")
            else:
                body.append("  ✗ ", style="red bold")
                body.append(f"{name:10s} ", style="bold")
                body.append(f"{runner:10s} ", style="cyan")
                body.append(f"{dur:.1f}s  ", style="dim")
                failures = stage.get("failures", [])
                body.append(f"{len(failures)} failure(s)\n", style="red")
                for f in failures[:5]:
                    file = f.get("file", "?") or "?"
                    line = f.get("line", "?")
                    msg = f.get("message", "?")
                    if len(msg) > 90:
                        msg = msg[:90] + "…"
                    body.append(f"      {file}:{line}  ", style="yellow")
                    body.append(f"{msg}\n", style="dim")
                if len(failures) > 5:
                    body.append(f"      … and {len(failures) - 5} more\n", style="dim")

    border = "green" if outcome.success else "red"
    title = "[green]passed[/green]" if outcome.success else "[red]failed[/red]"
    console.print()
    console.print(Panel(body, title=title, border_style=border, padding=(0, 1)))
