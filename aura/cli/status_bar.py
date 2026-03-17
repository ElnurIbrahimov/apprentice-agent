"""Persistent status bar for AURA CLI — shows model, project, session info."""

import os
import subprocess
from pathlib import Path
from rich.text import Text

from aura.cli.context_bar import build_context_gauge
from aura.cli.permissions_ui import get_mode_short


def _get_git_branch() -> str:
    """Get current git branch name, or empty string if not in a git repo."""
    try:
        result = subprocess.run(
            ["git", "branch", "--show-current"],
            capture_output=True, text=True, timeout=2,
            cwd=os.getcwd(),
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return ""


def build_status_bar(
    model: str = "auto",
    project_type: str = "",
    session_title: str = "",
    message_count: int = 0,
    width: int = 80,
    thinking: bool = False,
    elapsed: float = 0.0,
    tool_count: int = 0,
    cost_usd: float = 0.0,
    tier: str = "",
    token_used: int = 0,
    token_limit: int = 128000,
    permission_mode: str = "careful",
    steering_queue=None,
    bg_indicator: str = "",
    research_indicator: str = "",
    mood_indicator: str = "",
    watch_indicator: str = "",
) -> Text:
    """Build a single-line status bar that spans the full terminal width."""

    # -- Left section: directory + project type + git branch --
    left = Text()
    cwd = os.getcwd()
    home = str(Path.home())
    if cwd.startswith(home):
        cwd = "~" + cwd[len(home):]
    cwd = cwd.replace("\\", "/")
    if len(cwd) > 35:
        cwd = "..." + cwd[-32:]

    left.append(f" {cwd}", style="dim white")

    if project_type and project_type != "unknown":
        left.append(f" ({project_type})", style="dim cyan")

    branch = _get_git_branch()
    if branch:
        left.append(" ", style="dim")
        left.append(branch, style="bold magenta")

    # -- Center section: model (or thinking indicator) + tool count --
    center = Text()
    if thinking:
        center.append(f"Thinking... ({elapsed:.1f}s)", style="bold yellow")
    else:
        model_short = model.replace(":cloud", "").replace(":latest", "")
        if len(model_short) > 25:
            model_short = model_short[:22] + "..."

        if model == "auto":
            center.append(model_short, style="bold cyan")
        else:
            # Actively overridden model — highlight in green
            center.append(model_short, style="bold green")

    if tool_count > 0:
        center.append(f" | {tool_count} tools", style="dim")

    if tier:
        center.append(f" | {tier}", style="dim yellow")

    if cost_usd > 0:
        center.append(f" | ${cost_usd:.3f}", style="dim green")

    # Context gauge
    if token_limit > 0:
        gauge_markup = build_context_gauge(token_used, token_limit)
        center.append(" | ", style="dim")
        center.append_text(Text.from_markup(gauge_markup))

    # Permission mode indicator
    mode_markup = get_mode_short(permission_mode)
    center.append(" | ", style="dim")
    center.append_text(Text.from_markup(mode_markup))

    # -- Right section: session info --
    right_info = Text()
    if session_title:
        title = session_title[:20]
        right_info.append(f'"{title}"', style="dim white")
        if message_count:
            right_info.append(f" ({message_count} msgs)", style="dim")
    elif message_count:
        right_info.append(f"{message_count} msgs", style="dim")

    # -- Phase 3 indicators: background, research, mood --
    if bg_indicator:
        right_info.append(" ")
        right_info.append_text(Text.from_markup(bg_indicator))
    if research_indicator:
        right_info.append(" ")
        right_info.append_text(Text.from_markup(research_indicator))
    if mood_indicator:
        right_info.append(f" {mood_indicator}")
    if watch_indicator:
        right_info.append(" ")
        right_info.append_text(Text.from_markup(watch_indicator))

    # -- Steering indicators --
    if steering_queue is not None:
        from aura.cli.steering import create_steering_indicator, create_follow_up_indicator
        steer_ind = create_steering_indicator(steering_queue)
        follow_ind = create_follow_up_indicator(steering_queue)
        if steer_ind:
            right_info.append(" ")
            right_info.append_text(Text.from_markup(steer_ind))
        if follow_ind:
            right_info.append(" ")
            right_info.append_text(Text.from_markup(follow_ind))

    # -- Hint (always rightmost) --
    hint = Text()
    hint.append("Alt+M", style="dim bold")
    hint.append(" model ", style="dim")

    # -- Assemble with separators, pad to full width --
    sep = Text(" | ", style="dim")

    bar = Text()
    bar.append_text(left)
    bar.append_text(sep)
    bar.append_text(center)

    if right_info.cell_len > 0:
        bar.append_text(sep)
        bar.append_text(right_info)

    # Calculate remaining space for right-alignment of hint
    used = bar.cell_len + hint.cell_len
    gap = width - used
    if gap < 0:
        # Terminal too narrow — drop hint to save space
        # No hint appended
        pass
    else:
        if gap > 0:
            bar.append(" " * gap)
        bar.append_text(hint)

    return bar
