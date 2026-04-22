"""Checkpoint picker UI + rewind-result display.

Extracted from the old monolithic ``display.py`` to keep the package tidy.
"""
from __future__ import annotations

from typing import Any, Optional

# Import the package as a module — attribute access at call time ensures
# tests that patch `aura.cli.display.console` see the mock, and keeps the
# submodule pluggable.
from aura.cli import display as _display


def show_checkpoint_picker(checkpoints: list[dict[str, Any]]) -> Optional[str]:
    """Display checkpoints in an arrow-key picker.

    Previously used ``console.input("Pick #")`` — slow, inconsistent with
    every other picker in the CLI (model, session, command palette all use
    prompt_toolkit's arrow-key picker). Now uses the shared ``run_picker``
    from ``aura.cli.picker``.
    """
    import time as _time

    if not checkpoints:
        _display.show_info("No checkpoints available.")
        return None

    display = checkpoints[:10]
    now = _time.time()

    def _rel(t: float) -> str:
        delta = now - t
        if delta < 60:
            return f"{int(delta)}s ago"
        if delta < 3600:
            return f"{int(delta / 60)}m ago"
        if delta < 86400:
            return f"{int(delta / 3600)}h ago"
        return f"{int(delta / 86400)}d ago"

    from aura.cli.picker import PickerItem, run_picker

    items: list[PickerItem] = []
    for cp in display:
        label = cp.get("label", "") or "-"
        files = cp.get("files", [])
        file_names = ", ".join(f.get("backup_name", "?") for f in files[:3])
        if len(files) > 3:
            file_names += f" +{len(files) - 3}"
        rel = _rel(cp.get("timestamp", now))
        items.append(PickerItem(
            id=str(cp["id"]),
            label=f"{rel.ljust(10)}  {label}",
            description=file_names,
        ))

    try:
        return run_picker(items, title="Rewind to checkpoint")
    except (EOFError, KeyboardInterrupt):
        return None


def show_rewind_result(success: bool, checkpoint_id: str) -> None:
    """Display the result of a rewind operation."""
    colors = _display._get_theme_colors()
    if success:
        _display.console.print(f"  [{colors['success']}]\u2713 Rewound to checkpoint {checkpoint_id}[/{colors['success']}]")
    else:
        _display.show_error("Failed to rewind")
