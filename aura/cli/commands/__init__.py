from __future__ import annotations

import difflib
from typing import Any, Callable, Optional

from .agent_commands import (
    handle_agent,
    handle_chain,
    handle_debate,
    handle_fleet,
    handle_goal,
    handle_hand,
    handle_plan,
)
from .copy_command import handle_copy
from .git_commands import (
    handle_blame,
    handle_branch,
    handle_diff,
    handle_git,
    handle_pr,
    handle_stash,
)
from .research_commands import (
    handle_browse,
    handle_export,
    handle_recall,
    handle_research,
    handle_sources,
)
from .session_commands import (
    handle_branches,
    handle_changes,
    handle_checkout,
    handle_clear,
    handle_compact,
    handle_context,
    handle_cost,
    handle_fork,
    handle_merge,
    handle_rewind,
    handle_sessions,
)
from .snippet_command import handle_snippet
from .system_commands import (
    handle_audit,
    handle_evolve,
    handle_hook,
    handle_mcp,
)
from .tool_commands import (
    handle_edit,
    handle_grep,
    handle_project,
    handle_search,
    handle_shell,
    handle_test,
    handle_undo,
    handle_watch,
)
from .ui_commands import (
    handle_help,
    handle_model,
    handle_mood,
    handle_quit,
    handle_routing,
    handle_speak,
    handle_tasks,
    handle_theme,
    handle_trust,
)


def _handle_voice(agent, args, context=None, **kwargs):
    from aura.cli.voice_mode import run_voice_mode
    run_voice_mode(agent)


COMMAND_REGISTRY: dict[str, Callable[..., Any]] = {
    "/quit": handle_quit,
    "/exit": handle_quit,
    "/help": handle_help,
    "/goal": handle_goal,
    "/recall": handle_recall,
    "/clear": handle_clear,
    "/speak": handle_speak,
    "/say": handle_speak,
    "/model": handle_model,
    "/compact": handle_compact,
    "/plan": handle_plan,
    "/hand": handle_hand,
    "/audit": handle_audit,
    "/browse": handle_browse,
    "/grep": handle_grep,
    "/search": handle_search,
    "/find": handle_search,
    "/edit": handle_edit,
    "/project": handle_project,
    "/shell": handle_shell,
    "/bash": handle_shell,
    "/run": handle_shell,
    "/agent": handle_agent,
    "/evolve": handle_evolve,
    "/fleet": handle_fleet,
    "/tasks": handle_tasks,
    "/research": handle_research,
    "/sources": handle_sources,
    "/export": handle_export,
    "/mood": handle_mood,
    "/hook": handle_hook,
    "/sessions": handle_sessions,
    "/theme": handle_theme,
    "/trust": handle_trust,
    "/context": handle_context,
    "/rewind": handle_rewind,
    "/cost": handle_cost,
    "/undo": handle_undo,
    "/diff": handle_diff,
    "/git": handle_git,
    "/pr": handle_pr,
    "/branch": handle_branch,
    "/stash": handle_stash,
    "/blame": handle_blame,
    "/test": handle_test,
    "/watch": handle_watch,
    "/mcp": handle_mcp,
    "/debate": handle_debate,
    "/chain": handle_chain,
    "/fork": handle_fork,
    "/branches": handle_branches,
    "/checkout": handle_checkout,
    "/merge": handle_merge,
    "/voice": _handle_voice,
    "/changes": handle_changes,
    "/routing": handle_routing,
    "/copy": handle_copy,
    "/snippet": handle_snippet,
}


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
        print(f"Unknown command: {cmd}")
        matches = difflib.get_close_matches(cmd, COMMAND_REGISTRY.keys(), n=1, cutoff=0.6)
        if matches:
            print(f"  Did you mean {matches[0]}?")
        return

    context: dict[str, Any] = {"speak": speak}
    handler(agent, arg, context)
