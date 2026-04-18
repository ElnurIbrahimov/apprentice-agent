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
    handle_trace,
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


# ─────────────────────────────────────────────────────────────────────────────
# Single source of truth for slash commands.
# (name, description, handler, aliases) — aliases route to the same handler
# but are excluded from the completer list to keep autocomplete uncluttered.
# ─────────────────────────────────────────────────────────────────────────────
COMMANDS: list[tuple[str, str, Callable[..., Any], list[str]]] = [
    ("/quit",     "Exit AURA",                                       handle_quit,     ["/exit"]),
    ("/help",     "Show help",                                       handle_help,     []),
    ("/clear",    "Clear conversation history",                      handle_clear,    []),
    ("/model",    "View/set model (auto, <name>)",                   handle_model,    []),
    ("/compact",  "Compact conversation history",                    handle_compact,  []),
    ("/plan",     "Create and execute a plan",                       handle_plan,     []),
    ("/shell",    "Execute shell command",                           handle_shell,    ["/bash", "/run"]),
    ("/grep",     "Search code content",                             handle_grep,     []),
    ("/search",   "Search files by pattern",                         handle_search,   ["/find"]),
    ("/edit",     "View file contents with line numbers",            handle_edit,     []),
    ("/project",  "Project info/context/index",                      handle_project,  []),
    ("/agent",    "Run specialist agent",                            handle_agent,    []),
    ("/sessions", "Manage sessions",                                 handle_sessions, []),
    ("/browse",   "Browse web pages",                                handle_browse,   []),
    ("/hook",     "Manage hooks",                                    handle_hook,     []),
    ("/speak",    "Text-to-speech",                                  handle_speak,    ["/say"]),
    ("/recall",   "Search memories",                                 handle_recall,   ["/memory"]),
    ("/goal",     "Run a goal",                                      handle_goal,     []),
    ("/trust",    "Enable trust mode (auto-approve all tools)",      handle_trust,    []),
    ("/cost",     "Show session cost breakdown",                     handle_cost,     []),
    ("/context",  "Show context window usage",                       handle_context,  []),
    ("/trace",    "Show structured session trace and run summaries", handle_trace,    []),
    ("/rewind",   "Rewind file changes to a checkpoint",             handle_rewind,   []),
    ("/theme",    "Switch color theme",                              handle_theme,    []),
    ("/fleet",    "Run parallel sub-agents",                         handle_fleet,    []),
    ("/tasks",    "Show background tasks",                           handle_tasks,    []),
    ("/research", "Start research mode",                             handle_research, []),
    ("/sources",  "Show research sources",                           handle_sources,  []),
    ("/export",   "Export research to Markdown",                     handle_export,   []),
    ("/mood",     "Show emotional state",                            handle_mood,     []),
    ("/pr",       "Create pull request",                             handle_pr,       []),
    ("/branch",   "Create git branch",                               handle_branch,   []),
    ("/stash",    "Smart git stash",                                 handle_stash,    []),
    ("/blame",    "Git blame with context",                          handle_blame,    []),
    ("/test",     "Run tests",                                       handle_test,     []),
    ("/watch",    "Watch files for AI comments",                     handle_watch,    []),
    ("/evolve",   "Evolve skills with GEPA",                         handle_evolve,   []),
    ("/diff",     "Show git diff with syntax highlighting",          handle_diff,     []),
    ("/git",      "Run read-only git commands",                      handle_git,      []),
    ("/mcp",      "Manage MCP server connections",                   handle_mcp,      []),
    ("/audit",    "Inspect Merkle audit chain",                      handle_audit,    []),
    ("/hand",     "Manage autonomous Hands",                         handle_hand,     []),
    ("/undo",     "Undo last file edit",                             handle_undo,     []),
    ("/debate",   "Multi-model debate on a question",                handle_debate,   []),
    ("/fork",     "Fork conversation into a new branch",             handle_fork,     []),
    ("/branches", "List conversation branches",                      handle_branches, []),
    ("/checkout", "Switch to a conversation branch",                 handle_checkout, []),
    ("/merge",    "Merge branch back to parent",                     handle_merge,    []),
    ("/chain",    "Run prompt pipelines (step1 -> step2 -> ...)",    handle_chain,    []),
    ("/changes",  "Show files modified in this session",             handle_changes,  []),
    ("/routing",  "Show/set routing preference",                     handle_routing,  []),
    ("/copy",     "Copy last response or code block to clipboard",   handle_copy,     []),
    ("/voice",    "Voice mode (speech input/output)",                _handle_voice,   []),
    ("/snippet",  "Manage prompt templates/snippets",                handle_snippet,  []),
]

# Pseudo-commands handled inline in chat_session_runtime.py — no registry entry,
# but they still appear in the completer so users can tab to them.
RUNTIME_ONLY_COMMANDS: list[tuple[str, str]] = [
    ("/retry",    "Re-run the last prompt"),
    ("/channels", "Show active channel bridges and status"),
]

# Derived dispatch map: canonical names + aliases → handler.
COMMAND_REGISTRY: dict[str, Callable[..., Any]] = {}
for _name, _desc, _handler, _aliases in COMMANDS:
    COMMAND_REGISTRY[_name] = _handler
    for _alias in _aliases:
        COMMAND_REGISTRY[_alias] = _handler

# Completer list: canonical commands + runtime-only. Aliases excluded to avoid
# showing /bash, /run, /say, /find, /memory, /exit as separate entries.
SLASH_COMMANDS: list[tuple[str, str]] = [
    (name, desc) for name, desc, _h, _a in COMMANDS
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
