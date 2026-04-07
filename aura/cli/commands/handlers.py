"""Re-export shim — all handlers moved to domain-specific modules.

Imports here exist solely for backward compatibility so that
``from aura.cli.commands.handlers import handle_*`` keeps working.
"""

from .git_commands import (
    handle_git, handle_diff, handle_blame, handle_branch,
    handle_stash, handle_pr,
)
from .session_commands import (
    handle_sessions, handle_clear, handle_compact,
    handle_context, handle_cost, handle_rewind,
)
from .agent_commands import (
    handle_goal, handle_plan, handle_fleet, handle_agent,
    handle_hand, print_result,
)
from .research_commands import (
    handle_research, handle_sources, handle_export, handle_browse,
    handle_recall,
)
from .tool_commands import (
    handle_shell, handle_grep, handle_search, handle_edit,
    handle_test, handle_project, handle_watch, handle_undo,
)
from .ui_commands import (
    handle_model, handle_theme, handle_mood, handle_speak,
    handle_trust, handle_help, handle_quit, handle_tasks,
    handle_routing,
)
from .system_commands import (
    handle_hook, handle_mcp, handle_audit, handle_evolve,
)

__all__ = [
    "handle_quit", "handle_help", "handle_goal", "handle_recall",
    "handle_clear", "handle_speak", "handle_model", "handle_compact",
    "handle_plan", "handle_hand", "handle_audit", "handle_browse",
    "handle_grep", "handle_search", "handle_edit", "handle_project",
    "handle_shell", "handle_agent", "handle_evolve", "handle_fleet",
    "handle_tasks", "handle_research", "handle_sources", "handle_export",
    "handle_mood", "handle_hook", "handle_sessions", "handle_theme",
    "handle_trust", "handle_context", "handle_rewind",
    "handle_cost", "handle_undo", "handle_diff", "handle_git",
    "handle_pr", "handle_branch", "handle_stash", "handle_blame",
    "handle_test", "handle_watch", "handle_mcp", "print_result",
    "handle_routing",
]
