# aura/cli/hooks.py
"""Programmable hooks — event-driven automation for the CLI."""
from __future__ import annotations
import shlex
import subprocess
import logging
import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable
from pathlib import Path

logger = logging.getLogger(__name__)


class HookEvent:
    """Hook event types."""
    PRE_TOOL_CALL = "pre_tool_call"
    POST_TOOL_CALL = "post_tool_call"
    PRE_RESPONSE = "pre_response"
    POST_RESPONSE = "post_response"
    SESSION_START = "session_start"
    SESSION_END = "session_end"
    POST_EDIT = "post_edit"
    POST_TEST = "post_test"

    ALL = [PRE_TOOL_CALL, POST_TOOL_CALL, PRE_RESPONSE, POST_RESPONSE,
           SESSION_START, SESSION_END, POST_EDIT, POST_TEST]


@dataclass
class Hook:
    """A single hook definition."""
    event: str
    command: str  # Shell command to run
    name: str = ""
    enabled: bool = True
    timeout: int = 30  # seconds
    run_count: int = 0
    last_run: float = 0.0
    last_exit_code: int = 0


class HookManager:
    """Manages and executes event hooks."""

    def __init__(self):
        self._hooks: List[Hook] = []
        self._builtins_loaded: bool = False

    def add(self, event: str, command: str, name: str = "") -> Hook:
        """Register a hook. Returns the hook object."""
        if event not in HookEvent.ALL:
            raise ValueError(f"Unknown event: {event}. Valid: {', '.join(HookEvent.ALL)}")
        hook = Hook(event=event, command=command, name=name or f"hook_{len(self._hooks)}")
        self._hooks.append(hook)
        return hook

    def remove(self, name: str) -> bool:
        """Remove a hook by name."""
        before = len(self._hooks)
        self._hooks = [h for h in self._hooks if h.name != name]
        return len(self._hooks) < before

    def list_hooks(self) -> List[Hook]:
        """List all hooks."""
        return list(self._hooks)

    def get_hooks_for_event(self, event: str) -> List[Hook]:
        """Get all enabled hooks for an event type."""
        return [h for h in self._hooks if h.event == event and h.enabled]

    def fire(self, event: str, context: Optional[Dict] = None) -> List[Dict]:
        """Fire all hooks for an event. Returns list of results."""
        hooks = self.get_hooks_for_event(event)
        results = []
        for hook in hooks:
            result = self._execute_hook(hook, context or {})
            results.append(result)
        return results

    def _execute_hook(self, hook: Hook, context: Dict) -> Dict:
        """Execute a single hook command."""
        import os

        # Clamp timeout to [1, 300] seconds
        timeout = max(1, min(300, hook.timeout))

        # Set context as environment variables
        env = os.environ.copy()
        env["AURA_HOOK_EVENT"] = hook.event
        for key, value in context.items():
            if isinstance(value, str):
                env[f"AURA_{key.upper()}"] = value[:1000]

        try:
            cmd_args = shlex.split(hook.command)
            result = subprocess.run(
                cmd_args,
                shell=False,  # SECURITY: no shell injection
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env,
                cwd=os.getcwd(),
            )
            hook.run_count += 1
            hook.last_run = time.time()
            hook.last_exit_code = result.returncode
            return {
                "hook": hook.name,
                "success": result.returncode == 0,
                "exit_code": result.returncode,
                "stdout": result.stdout[:2000],
                "stderr": result.stderr[:500],
            }
        except subprocess.TimeoutExpired:
            hook.last_exit_code = -1
            return {"hook": hook.name, "success": False, "error": f"Timeout ({hook.timeout}s)"}
        except Exception as e:
            return {"hook": hook.name, "success": False, "error": str(e)[:200]}

    def load_from_config(self, config: Dict) -> int:
        """Load hooks from AURA.md or config dict. Returns count loaded."""
        hooks_config = config.get("hooks", [])
        count = 0
        for h in hooks_config:
            if isinstance(h, dict) and "event" in h and "command" in h:
                self.add(
                    event=h["event"],
                    command=h["command"],
                    name=h.get("name", ""),
                )
                count += 1
        return count

    def load_builtin_hooks(self, project_config: Dict) -> None:
        """Load built-in convenience hooks based on project config."""
        if self._builtins_loaded:
            return
        self._builtins_loaded = True

        # Auto-lint after edits (if lint_cmd is configured)
        lint_cmd = project_config.get("lint_cmd")
        if lint_cmd:
            self.add(HookEvent.POST_EDIT, lint_cmd, name="auto_lint")

        # Auto-test after edits (if auto_test is enabled)
        test_cmd = project_config.get("test_cmd")
        auto_test = project_config.get("auto_test", False)
        if test_cmd and auto_test:
            self.add(HookEvent.POST_EDIT, test_cmd, name="auto_test")

    def clear(self) -> None:
        """Remove all hooks."""
        self._hooks.clear()
        self._builtins_loaded = False


def render_hooks_table(console, hooks: List[Hook]) -> None:
    """Render hooks as a Rich table."""
    from rich.table import Table

    if not hooks:
        console.print("[dim]No hooks registered.[/dim]")
        return

    table = Table(title="Registered Hooks", border_style="cyan")
    table.add_column("Name", style="bold", min_width=15)
    table.add_column("Event", width=15)
    table.add_column("Command", min_width=25)
    table.add_column("Runs", width=5)
    table.add_column("Status", width=10)

    for h in hooks:
        status = "[green]on[/green]" if h.enabled else "[red]off[/red]"
        table.add_row(h.name, h.event, h.command[:40], str(h.run_count), status)

    console.print(table)
