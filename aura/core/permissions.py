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

    def _severity(self) -> int:
        """Numeric severity for permission comparison.

        Higher = more restrictive. Used by apply_aura_md_overrides to prevent
        project configs from escalating permissions.
        """
        return {"auto": 0, "prompt": 1, "blocked": 2}[self.value]


AUTO = PermissionTier.AUTO
PROMPT = PermissionTier.PROMPT
BLOCKED = PermissionTier.BLOCKED

# Shell commands that are auto-approved (first word of command)
SAFE_SHELL_COMMANDS = {
    "mkdir", "ls", "dir", "cat", "head", "tail", "echo", "pwd", "cd",
    "npm", "npx", "yarn", "pnpm", "pip", "pip3",
    "git", "touch", "cp", "mv",
    "find", "grep", "which", "env", "export",
}

# Default permission map for each tool (or tool.subaction)
DEFAULT_PERMISSIONS = {
    # Read-only: always allowed
    "read_file": AUTO,
    "grep": AUTO,
    "glob": AUTO,
    "list_dir": AUTO,
    "search_web": AUTO,
    "project_structure": AUTO,
    # Git ops: all auto-approved
    "git.status": AUTO,
    "git.log": AUTO,
    "git.diff": AUTO,
    "git.branch": AUTO,
    "git.add": AUTO,
    "git.commit": AUTO,
    "git.push": AUTO,
    "git.pull": AUTO,
    # File edits: auto-approved (like Claude Code)
    "edit_file": AUTO,
    "write_file": AUTO,
    # Shell: prompt, but safe commands auto-approved
    "shell": PROMPT,
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
        """Apply permission overrides from AURA.md frontmatter.

        Security: project configs can only RESTRICT permissions, never escalate.
        """
        perms = config.get("permissions", {})
        for key, value in perms.items():
            if key not in DEFAULT_PERMISSIONS:
                logger.warning(f"[Permissions] Unknown permission key '{key}' in AURA.md — skipped")
                continue
            try:
                new_tier = PermissionTier(value)
                current = self._permissions.get(key)
                if current is not None and new_tier._severity() < current._severity():
                    logger.warning(f"[Permissions] Project config tried to escalate {key} from {current.name} to {new_tier.name} — blocked")
                    continue
                self._permissions[key] = new_tier
            except ValueError:
                logger.warning(f"[Permissions] Unknown tier '{value}' for '{key}'")

    def _resolve_key(self, tool_name: str, args: dict) -> str:
        """Resolve the permission key, handling git sub-actions and case normalization."""
        tool_name = tool_name.lower()
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

        # Shell commands: auto-approve if the first word is in SAFE_SHELL_COMMANDS
        if tool_name == "shell":
            cmd = args.get("command", "").strip()
            first_word = cmd.split()[0].split("/")[-1].split("\\")[-1] if cmd else ""
            if first_word in SAFE_SHELL_COMMANDS:
                return True

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

    @property
    def current_mode(self) -> str:
        """Return current permission mode name for display/checks."""
        if self._trust_mode:
            return "full_auto"
        return self._mode if hasattr(self, '_mode') else "careful"

    def set_mode(self, mode: str) -> None:
        """Set the permission mode."""
        self._mode = mode
        if mode == "full_auto":
            self._trust_mode = True
        elif mode in ("plan", "plan_approve"):
            self._trust_mode = False

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
