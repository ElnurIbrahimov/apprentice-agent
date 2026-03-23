"""CLIContext — typed container for CLI session state.

Replaces the pattern of bolting attributes onto the agent object
(agent._agentic_loop, agent._bg_manager, etc.) with a proper dataclass.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class CLIContext:
    """Holds all CLI session state that was previously bolted onto agent."""

    agent: Any
    agentic_loop: Any = None
    permissions: Any = None
    session: Any = None
    bg_manager: Any = None
    research_ctx: Any = None
    hook_manager: Any = None
    file_watcher: Any = None
    speak: bool = False
    verbose: bool = False
    resume_session_id: Optional[str] = None


# Module-level reference so handlers can access the context without
# needing it threaded through every call signature.  Set by run_chat_mode().
_current: Optional[CLIContext] = None


def get_ctx() -> Optional[CLIContext]:
    """Return the active CLIContext, or None if chat mode hasn't started."""
    return _current


def set_ctx(ctx: Optional[CLIContext]) -> None:
    """Set the active CLIContext (called once from run_chat_mode)."""
    global _current
    _current = ctx
