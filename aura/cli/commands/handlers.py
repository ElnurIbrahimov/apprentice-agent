"""Re-export shim — all handlers moved to domain-specific modules.

Imports here exist solely for backward compatibility so that
``from aura.cli.commands.handlers import handle_*`` keeps working.
"""

from .agent_commands import (
    handle_agent,
    handle_fleet,
    handle_goal,
    handle_hand,
    handle_plan,
    print_result,
)
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
    handle_clear,
    handle_compact,
    handle_context,
    handle_cost,
    handle_rewind,
    handle_sessions,
)
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

__all__ = [
    "handle_agent",
    "handle_audit",
    "handle_blame",
    "handle_branch",
    "handle_browse",
    "handle_clear",
    "handle_compact",
    "handle_context",
    "handle_cost",
    "handle_diff",
    "handle_edit",
    "handle_evolve",
    "handle_export",
    "handle_fleet",
    "handle_git",
    "handle_goal",
    "handle_grep",
    "handle_hand",
    "handle_help",
    "handle_hook",
    "handle_mcp",
    "handle_model",
    "handle_mood",
    "handle_plan",
    "handle_pr",
    "handle_project",
    "handle_quit",
    "handle_recall",
    "handle_research",
    "handle_rewind",
    "handle_routing",
    "handle_search",
    "handle_sessions",
    "handle_shell",
    "handle_sources",
    "handle_speak",
    "handle_stash",
    "handle_tasks",
    "handle_test",
    "handle_theme",
    "handle_trust",
    "handle_undo",
    "handle_watch",
    "print_result",
]
