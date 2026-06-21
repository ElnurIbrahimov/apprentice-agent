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
    provider_commands,
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
from .agent_commands import (
    handle_agent,
    handle_chain,
    handle_debate,
    handle_fleet,
    handle_goal,
    handle_hand,
    handle_interrupt,
    handle_plan,
)
from .bench_commands import handle_bench
from .copy_command import handle_copy
from .doctor_commands import handle_doctor
from .fix_commands import handle_fix
from .git_commands import (
    handle_blame,
    handle_branch,
    handle_diff,
    handle_git,
    handle_pr,
    handle_stash,
)
from .heatmap_commands import handle_heatmap
from .history_commands import handle_history
from .provider_commands import (
    handle_provider,
    handle_providers,
)
from .research_commands import (
    handle_browse,
    handle_export,
    handle_recall,
    handle_research,
    handle_sources,
)
from .session_commands import (
    handle_blocks,
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
from .shadow_commands import handle_shadow
from .skill_commands import handle_skill
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
    handle_redo,
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
from .common import (
    TIER_BETA,
    TIER_EXPERIMENTAL,
    TIER_STABLE,
    _COMMAND_REGISTRY,
    command,
)
from .verify_commands import handle_verify
from .why_commands import handle_why

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
    # Handlers
    "handle_agent", "handle_chain", "handle_debate", "handle_fleet",
    "handle_goal", "handle_hand", "handle_interrupt", "handle_plan",
    "handle_copy", "handle_blame", "handle_branch", "handle_diff",
    "handle_git", "handle_pr", "handle_stash", "handle_browse",
    "handle_export", "handle_recall", "handle_research", "handle_sources",
    "handle_branches", "handle_changes", "handle_checkout", "handle_clear",
    "handle_compact", "handle_context", "handle_cost", "handle_fork",
    "handle_merge", "handle_rewind", "handle_sessions", "handle_trace",
    "handle_snippet", "handle_audit", "handle_evolve", "handle_hook",
    "handle_mcp", "handle_verify", "handle_why", "handle_edit",
    "handle_grep", "handle_project", "handle_search", "handle_shell",
    "handle_test", "handle_redo", "handle_undo", "handle_watch",
    "handle_heatmap", "handle_shadow", "handle_help", "handle_model",
    "handle_mood", "handle_quit", "handle_routing", "handle_speak",
    "handle_tasks", "handle_theme", "handle_trust",
    "handle_provider", "handle_providers",
]

# Local handler defined here rather than in a submodule.
@command("/voice", "Voice mode (speech input/output)", tier=TIER_BETA)
def _handle_voice(agent, args, context=None, **kwargs):
    from aura.cli.voice_mode import run_voice_mode
    run_voice_mode(agent)



COMMANDS: list[tuple[str, str, Callable[..., Any], list[str], str]] = [
    ("/quit",     "Exit AURA",                                       handle_quit,     ["/exit"],         TIER_STABLE),
    ("/help",     "Show help",                                       handle_help,     [],                TIER_STABLE),
    ("/clear",    "Clear conversation history",                      handle_clear,    [],                TIER_STABLE),
    ("/model",    "View/set model (auto, <name>)",                   handle_model,    [],                TIER_STABLE),
    ("/compact",  "Compact conversation history",                    handle_compact,  [],                TIER_STABLE),
    ("/plan",     "Create and execute a plan",                       handle_plan,     [],                TIER_STABLE),
    ("/shell",    "Execute shell command",                           handle_shell,    ["/bash", "/run"], TIER_STABLE),
    ("/grep",     "Search code content",                             handle_grep,     [],                TIER_STABLE),
    ("/search",   "Search files by pattern",                         handle_search,   ["/find"],         TIER_STABLE),
    ("/edit",     "View file contents with line numbers",            handle_edit,     [],                TIER_STABLE),
    ("/project",  "Project info/context/index",                      handle_project,  [],                TIER_STABLE),
    ("/agent",    "Run specialist agent",                            handle_agent,    [],                TIER_BETA),
    ("/sessions", "Manage sessions",                                 handle_sessions, [],                TIER_STABLE),
    ("/browse",   "Browse web pages",                                handle_browse,   [],                TIER_BETA),
    ("/hook",     "Manage hooks",                                    handle_hook,     [],                TIER_BETA),
    ("/speak",    "Text-to-speech",                                  handle_speak,    ["/say"],          TIER_BETA),
    ("/recall",   "Search memories",                                 handle_recall,   ["/memory"],       TIER_BETA),
    ("/goal",     "Run a goal",                                      handle_goal,     [],                TIER_BETA),
    ("/trust",    "Enable trust mode (auto-approve all tools)",      handle_trust,    [],                TIER_STABLE),
    ("/cost",     "Show session cost breakdown",                     handle_cost,     [],                TIER_STABLE),
    ("/history",  "Scannable conversation timeline",                 handle_history,  [],                TIER_STABLE),
    ("/context",  "Show context window usage",                       handle_context,  [],                TIER_STABLE),
    ("/trace",    "Show structured session trace and run summaries", handle_trace,    [],                TIER_BETA),
    ("/rewind",   "Rewind file changes to a checkpoint",             handle_rewind,   [],                TIER_STABLE),
    ("/theme",    "Switch color theme",                              handle_theme,    [],                TIER_STABLE),
    ("/fleet",    "Run parallel sub-agents",                         handle_fleet,    [],                TIER_BETA),
    ("/tasks",    "Show background tasks",                           handle_tasks,    [],                TIER_BETA),
    ("/research", "Start research mode",                             handle_research, [],                TIER_BETA),
    ("/sources",  "Show research sources",                           handle_sources,  [],                TIER_BETA),
    ("/export",   "Export research to Markdown",                     handle_export,   [],                TIER_BETA),
    ("/mood",     "Show emotional state",                            handle_mood,     [],                TIER_BETA),
    ("/pr",       "Create pull request",                             handle_pr,       [],                TIER_BETA),
    ("/branch",   "Create git branch",                               handle_branch,   [],                TIER_BETA),
    ("/stash",    "Smart git stash",                                 handle_stash,    [],                TIER_BETA),
    ("/blame",    "Git blame with context",                          handle_blame,    [],                TIER_BETA),
    ("/test",     "Run tests",                                       handle_test,     [],                TIER_STABLE),
    ("/verify",   "Verify this session's edits (typecheck/tests)",   handle_verify,   [],                TIER_STABLE),
    ("/why",      "Show edit history + triggering prompts for a file", handle_why,     [],                TIER_EXPERIMENTAL),
    ("/watch",    "Watch files for AI comments",                     handle_watch,    [],                TIER_BETA),
    ("/evolve",   "Evolve skills with GEPA",                         handle_evolve,   [],                TIER_EXPERIMENTAL),
    ("/diff",     "Show git diff with syntax highlighting",          handle_diff,     [],                TIER_STABLE),
    ("/git",      "Run read-only git commands",                      handle_git,      [],                TIER_STABLE),
    ("/mcp",      "Manage MCP server connections",                   handle_mcp,      [],                TIER_BETA),
    ("/audit",    "Inspect Merkle audit chain",                      handle_audit,    [],                TIER_BETA),
    ("/doctor",   "Run full system diagnostic",                       handle_doctor,   [],                TIER_STABLE),
    ("/fix",      "Auto-fix failing tests (run → fix → verify loop)",  handle_fix,      [],                TIER_BETA),
    ("/hand",     "Manage autonomous Hands",                         handle_hand,     [],                TIER_EXPERIMENTAL),
    ("/undo",     "Undo last file edit",                             handle_undo,     [],                TIER_STABLE),
    ("/redo",     "Redo the last /undo",                             handle_redo,     [],                TIER_STABLE),
    ("/heatmap",  "Show cognitive heatmap (tokens by tool/file)",    handle_heatmap,  [],                TIER_BETA),
    ("/shadow",   "Run prompt against 2 models in parallel (diff)",  handle_shadow,   [],                TIER_BETA),
    ("/debate",   "Multi-model debate on a question",                handle_debate,   [],                TIER_EXPERIMENTAL),
    ("/fork",     "Fork conversation into a new branch",             handle_fork,     [],                TIER_EXPERIMENTAL),
    ("/branches", "List conversation branches",                      handle_branches, [],                TIER_EXPERIMENTAL),
    ("/checkout", "Switch to a conversation branch",                 handle_checkout, [],                TIER_EXPERIMENTAL),
    ("/merge",    "Merge branch back to parent",                     handle_merge,    [],                TIER_EXPERIMENTAL),
    ("/chain",    "Run prompt pipelines (step1 -> step2 -> ...)",    handle_chain,    [],                TIER_BETA),
    ("/changes",  "Show files modified in this session",             handle_changes,  [],                TIER_BETA),
    ("/routing",  "Show/set routing preference",                     handle_routing,  [],                TIER_BETA),
    ("/blocks",    "List recent output blocks (/blocks N to expand)",   handle_blocks,   [],                TIER_BETA),
    ("/copy",     "Copy last response or code block to clipboard",   handle_copy,     [],                TIER_STABLE),
    ("/bench",    "Benchmark prompt across multiple models",          handle_bench,    [],                TIER_BETA),
    ("/voice",    "Voice mode (speech input/output)",                _handle_voice,   [],                TIER_BETA),
    ("/snippet",  "Manage prompt templates/snippets",                handle_snippet,  [],                TIER_BETA),
    ("/skill",    "Browse and load skills",                            handle_skill,    [],                TIER_BETA),
    ("/providers","List all configured providers with status and models",handle_providers,[],               TIER_STABLE),
    ("/provider", "Switch active provider interactively",              handle_provider, [],                TIER_STABLE),
    ("/interrupt","Abort running iteration with optional correction",handle_interrupt,["/stop"],         TIER_STABLE),
]

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


def handle_command(agent: Any, cmd: str, speak: bool = False) -> None:
    parts: list[str] = cmd.split(maxsplit=1)
    cmd_str: str = parts[0].lower()
    arg: str = parts[1] if len(parts) > 1 else ""

    # Special case: /export research needs to route to handle_export
    if cmd_str == "/export" and arg.strip().startswith("research"):
        handler = COMMAND_REGISTRY.get("/export")
    else:
        handler = COMMAND_REGISTRY.get(cmd_str)

    if handler is None:
        from aura.cli.display import console
        console.print(f"[red]Unknown command:[/red] {cmd_str}")
        matches = difflib.get_close_matches(cmd_str, COMMAND_REGISTRY.keys(), n=1, cutoff=0.6)
        if matches:
            console.print(f"  [dim]Did you mean[/dim] [cyan]{matches[0]}[/cyan]?")
        return

    context: dict[str, Any] = {"speak": speak}
    handler(agent, arg, context)
