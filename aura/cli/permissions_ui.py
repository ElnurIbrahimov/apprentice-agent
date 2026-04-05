"""Permission tier UI — mode cycling and status display."""
from __future__ import annotations
from enum import Enum
from typing import Optional


class PermissionMode(str, Enum):
    PLAN = "plan"
    PLAN_APPROVE = "plan_approve"
    CAREFUL = "careful"
    AUTO_EDIT = "auto_edit"
    FULL_AUTO = "full_auto"


# auto_edit: auto-approve file edits/writes, prompt for shell commands.
# Now wired up — confirm callback checks should_auto_approve_edit().
_MODE_ORDER = [
    PermissionMode.CAREFUL,
    PermissionMode.AUTO_EDIT,
    PermissionMode.PLAN_APPROVE,
    PermissionMode.FULL_AUTO,
]

_MODE_DESCRIPTIONS = {
    PermissionMode.PLAN: "Plan Mode — read-only, no file edits or commands",
    PermissionMode.PLAN_APPROVE: "Plan-Approve — agent shows plan first, then executes on approval",
    PermissionMode.CAREFUL: "Careful — approve every edit and shell command",
    PermissionMode.AUTO_EDIT: "Auto-Edit — file edits auto-apply, commands ask",
    PermissionMode.FULL_AUTO: "Full Auto — everything runs autonomously",
}

_MODE_INDICATORS = {
    PermissionMode.PLAN: "[blue]◎ PLAN[/blue]",
    PermissionMode.PLAN_APPROVE: "[magenta]◎ PLAN-APPROVE[/magenta]",
    PermissionMode.CAREFUL: "[yellow]◉ CAREFUL[/yellow]",
    PermissionMode.AUTO_EDIT: "[green]◉ AUTO-EDIT[/green]",
    PermissionMode.FULL_AUTO: "[red]● FULL-AUTO[/red]",
}

_MODE_SHORT = {
    PermissionMode.PLAN: "[blue]PLAN[/blue]",
    PermissionMode.PLAN_APPROVE: "[magenta]P-APR[/magenta]",
    PermissionMode.CAREFUL: "[yellow]CARE[/yellow]",
    PermissionMode.AUTO_EDIT: "[green]AUTO[/green]",
    PermissionMode.FULL_AUTO: "[red]FULL[/red]",
}


def cycle_permission_mode(current: str) -> str:
    """Cycle to the next permission mode.

    PLAN mode is entered via /plan, not cycling. If currently in PLAN or
    PLAN_APPROVE, cycling moves to CAREFUL (first in cycle order).
    """
    try:
        current_mode = PermissionMode(current)
    except ValueError:
        return PermissionMode.CAREFUL.value
    try:
        idx = _MODE_ORDER.index(current_mode)
    except ValueError:
        # Current mode not in cycle order (e.g. PLAN) — enter at CAREFUL
        return _MODE_ORDER[0].value
    next_idx = (idx + 1) % len(_MODE_ORDER)
    return _MODE_ORDER[next_idx].value


def get_mode_description(mode: str) -> str:
    """Get human-readable description of a permission mode."""
    try:
        return _MODE_DESCRIPTIONS[PermissionMode(mode)]
    except (ValueError, KeyError):
        return "Unknown mode"


def get_mode_indicator(mode: str) -> str:
    """Get full indicator string with icon."""
    try:
        return _MODE_INDICATORS[PermissionMode(mode)]
    except (ValueError, KeyError):
        return "[dim]???[/dim]"


def get_mode_short(mode: str) -> str:
    """Get short indicator for status bar."""
    try:
        return _MODE_SHORT[PermissionMode(mode)]
    except (ValueError, KeyError):
        return "[dim]???[/dim]"


def should_auto_approve_edit(mode: str) -> bool:
    return mode in (PermissionMode.AUTO_EDIT.value, PermissionMode.FULL_AUTO.value)

def should_auto_approve_command(mode: str) -> bool:
    return mode == PermissionMode.FULL_AUTO.value

def should_block_mutations(mode: str) -> bool:
    return mode == PermissionMode.PLAN.value

def is_plan_approve_mode(mode: str) -> bool:
    """Check if current mode is plan-approve-execute."""
    return mode == PermissionMode.PLAN_APPROVE.value
