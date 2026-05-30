from __future__ import annotations

import difflib
from typing import Any, Callable

# ─────────────────────────────────────────────────────────────────────────────
# Trigger self-registration: importing each command module populates
# common._COMMAND_REGISTRY via the @command decorators.
# ─────────────────────────────────────────────────────────────────────────────
from . import (  # noqa: F401
    agent_commands,
    copy_command,
    git_commands,
    heatmap_commands,
    research_commands,
    session_commands,
    shadow_commands,
    snippet_command,
    system_commands,
    tool_commands,
    ui_commands,
    verify_commands,
    why_commands,
)
from .common import (
    TIER_BETA,
    TIER_EXPERIMENTAL,
    TIER_STABLE,
    _COMMAND_REGISTRY,
    command,
)

# Re-export for backward compatibility.
__all__: list[str] = [
    "TIER_STABLE",
    "TIER_BETA",
    "TIER_EXPERIMENTAL",
    "COMMANDS",
    "COMMAND_REGISTRY",
    "COMMAND_TIERS",
    "EXPERIMENTAL_COMMANDS",
    "SLASH_COMMANDS",
    "RUNTIME_ONLY_COMMANDS",
    "handle_command",
    "command",
]

# Local handler defined here rather than in a submodule.
@command("/voice", "Voice mode (speech input/output)", tier=TIER_BETA)
def _handle_voice(agent, args, context=None, **kwargs):
    from aura.cli.voice_mode import run_voice_mode
    run_voice_mode(agent)


# Pseudo-commands handled inline in chat_session_runtime.py — no registry entry,
# but they still appear in the completer so users can tab to them.
RUNTIME_ONLY_COMMANDS: list[tuple[str, str]] = [
    ("/retry",    "Re-run the last prompt"),
    ("/channels", "Show active channel bridges and status"),
]

# Derived command list (sorted by name for deterministic output).
COMMANDS: list[tuple[str, str, Callable[..., Any], list[str], str]] = sorted(
    _COMMAND_REGISTRY, key=lambda entry: entry[0]
)

# Derived dispatch map: canonical names + aliases → handler.
COMMAND_REGISTRY: dict[str, Callable[..., Any]] = {}
for _name, _desc, _handler, _aliases, _tier in COMMANDS:
    COMMAND_REGISTRY[_name] = _handler
    for _alias in _aliases:
        COMMAND_REGISTRY[_alias] = _handler

# Derived tier lookup: command_name → tier. Used by /help and the palette.
COMMAND_TIERS: dict[str, str] = {name: tier for name, _d, _h, _a, tier in COMMANDS}
EXPERIMENTAL_COMMANDS: set[str] = {
    name for name, _d, _h, _a, tier in COMMANDS if tier == TIER_EXPERIMENTAL
}

# Completer list: canonical commands + runtime-only. Aliases excluded to avoid
# showing /bash, /run, /say, /find, /memory, /exit as separate entries.
SLASH_COMMANDS: list[tuple[str, str]] = [
    (name, desc) for name, desc, _h, _a, _t in COMMANDS
] + RUNTIME_ONLY_COMMANDS


def handle_command(agent: Any, command: str, speak: bool = False) -> None:
    parts: list[str] = command.split(maxsplit=1)
    cmd: str = parts[0].lower()
    arg: str = parts[1] if len(parts) > 1 else ""

    # Special case: /export research needs to route to handle_export
    if cmd == "/export" and arg.strip().startswith("research"):
        handler = COMMAND_REGISTRY.get("/export")
    else:
        handler = COMMAND_REGISTRY.get(cmd)

    if handler is None:
        from aura.cli.display import console
        console.print(f"[red]Unknown command:[/red] {cmd}")
        matches = difflib.get_close_matches(cmd, COMMAND_REGISTRY.keys(), n=1, cutoff=0.6)
        if matches:
            console.print(f"  [dim]Did you mean[/dim] [cyan]{matches[0]}[/cyan]?")
        return

    context: dict[str, Any] = {"speak": speak}
    handler(agent, arg, context)
