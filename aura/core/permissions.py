"""Permission system for Aura's agentic dev CLI.

Three tiers: AUTO (always allowed), PROMPT (ask user), BLOCKED (never).
Trust mode overrides all to AUTO.
"""

import logging
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class PermissionTier(Enum):
    AUTO = "auto"
    PROMPT = "prompt"
    BLOCKED = "blocked"


AUTO = PermissionTier.AUTO
PROMPT = PermissionTier.PROMPT
BLOCKED = PermissionTier.BLOCKED

# Default permission map for each tool (or tool.subaction)
DEFAULT_PERMISSIONS = {
    # Read-only: always allowed
    "read_file": AUTO,
    "grep": AUTO,
    "glob": AUTO,
    "list_dir": AUTO,
    "search_web": AUTO,
    "project_structure": AUTO,
    # Git read ops: always allowed
    "git.status": AUTO,
    "git.log": AUTO,
    "git.diff": AUTO,
    "git.branch": AUTO,
    # Mutating: ask user
    "edit_file": PROMPT,
    "write_file": PROMPT,
    "shell": PROMPT,
    # Git write ops: ask user
    "git.add": PROMPT,
    "git.commit": PROMPT,
    "git.push": PROMPT,
    "git.pull": PROMPT,
    # Sub-agent spawning
    "spawn_agent": PROMPT,
}

# Git actions that are read-only
GIT_READ_ACTIONS = frozenset({"status", "log", "diff", "branch"})


class PermissionManager:
    """Manages tool execution permissions."""

    def __init__(self, confirm_callback=None):
        self._permissions = dict(DEFAULT_PERMISSIONS)
        self._trust_mode = False
        self._confirm_callback = confirm_callback
        self._always_approved: set[str] = set()

    def set_trust_mode(self, enabled: bool) -> None:
        self._trust_mode = enabled
        logger.info(f"[Permissions] Trust mode: {'ON' if enabled else 'OFF'}")

    @property
    def trust_mode(self) -> bool:
        return self._trust_mode

    def set_confirm_callback(self, callback) -> None:
        self._confirm_callback = callback

    def override(self, tool_key: str, tier: PermissionTier) -> None:
        self._permissions[tool_key] = tier

    def apply_aura_md_overrides(self, config: dict) -> None:
        """Apply permission overrides from AURA.md frontmatter."""
        perms = config.get("permissions", {})
        for key, value in perms.items():
            try:
                self._permissions[key] = PermissionTier(value)
            except ValueError:
                logger.warning(f"[Permissions] Unknown tier '{value}' for '{key}'")

    def _resolve_key(self, tool_name: str, args: dict) -> str:
        """Resolve the permission key, handling git sub-actions."""
        if tool_name == "git":
            action = args.get("action", "status")
            return f"git.{action}"
        return tool_name

    def check(self, tool_name: str, args: dict) -> bool:
        """Check if a tool call is allowed. Returns True if approved.

        For PROMPT tier, calls the confirm_callback. If no callback is set,
        defaults to denied (safe default).
        """
        if self._trust_mode:
            return True

        key = self._resolve_key(tool_name, args)

        # Already permanently approved this session
        if key in self._always_approved:
            return True

        tier = self._permissions.get(key, PROMPT)

        if tier == AUTO:
            return True
        if tier == BLOCKED:
            logger.warning(f"[Permissions] BLOCKED: {key}")
            return False

        # PROMPT tier — ask user
        if self._confirm_callback:
            description = self._format_action_description(tool_name, args)
            result = self._confirm_callback(tool_name, description)
            if result == "always":
                self._always_approved.add(key)
                return True
            return bool(result)

        # No callback = deny by default
        return False

    def _format_action_description(self, tool_name: str, args: dict) -> str:
        """Format a human-readable description for the approval prompt."""
        if tool_name == "edit_file":
            path = args.get("path", "?")
            old = args.get("old_string", "")[:80]
            new = args.get("new_string", "")[:80]
            return f"Edit {path}\n  - {old}\n  + {new}"
        elif tool_name == "write_file":
            path = args.get("path", "?")
            size = len(args.get("content", ""))
            return f"Write {path} ({size} chars)"
        elif tool_name == "shell":
            cmd = args.get("command", "?")
            cwd = args.get("cwd", ".")
            return f"Run: {cmd}\n  in: {cwd}"
        elif tool_name == "git":
            action = args.get("action", "?")
            msg = args.get("message", "")
            if msg:
                return f"git {action}: {msg}"
            return f"git {action}"
        return f"{tool_name}({args})"
