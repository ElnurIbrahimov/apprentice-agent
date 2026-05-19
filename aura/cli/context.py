"""CLIContext — typed container for CLI session state.

Replaces the pattern of bolting attributes onto the agent object
(agent._agentic_loop, agent._bg_manager, etc.) with a proper dataclass.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from aura.agent import ApprenticeAgent
    from aura.cli.background import BackgroundManager
    from aura.cli.blocks import BlockManager
    from aura.cli.chat_session import ChatSession
    from aura.cli.hooks import HookManager
    from aura.cli.research_mode import ResearchContext
    from aura.cli.steering import SteeringQueue
    from aura.cli.watch_mode import FileWatcher
    from aura.core.agentic_loop import AgenticLoop
    from aura.core.permissions import PermissionManager
    from aura.core.session import AgenticSession


@dataclass
class CLIContext:
    """Holds all CLI session state that was previously bolted onto agent."""

    agent: ApprenticeAgent
    agentic_loop: Optional[AgenticLoop] = None
    permissions: Optional[PermissionManager] = None
    session: Optional[AgenticSession] = None
    bg_manager: Optional[BackgroundManager] = None
    research_ctx: Optional[ResearchContext] = None
    hook_manager: Optional[HookManager] = None
    file_watcher: Optional[FileWatcher] = None
    steering: Optional[SteeringQueue] = None
    speak: bool = False
    verbose: bool = False
    resume_session_id: Optional[str] = None
    # Set by ChatSession.__init__ after set_ctx. Optional because non-interactive
    # entry points (ACP, MCP, oneshot) build a CLIContext without a ChatSession.
    chat_session: Optional[ChatSession] = None
    # Block-based output registry — created by ChatSession, accessible from
    # command handlers via get_ctx().blocks for /blocks, /copy, etc.
    blocks: Optional[BlockManager] = None


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
