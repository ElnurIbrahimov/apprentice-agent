"""Permission system for Aura's agentic dev CLI.

Three tiers: AUTO (always allowed), PROMPT (ask user), BLOCKED (never).
Trust mode overrides all to AUTO.
"""

import logging
from enum import Enum

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


class SandboxTier(Enum):
    """Global sandbox tier that clamps per-tool permissions at runtime.

    Pattern adopted from OpenAI's Codex CLI (Apache-2.0).  READ_ONLY is safest,
    UNRESTRICTED is the default (preserves existing behavior).
    """

    READ_ONLY = "read_only"
    WORKSPACE_WRITE = "workspace_write"
    UNRESTRICTED = "unrestricted"


# Tools that remain callable in READ_ONLY mode.  Everything else → BLOCKED.
_READ_ONLY_ALLOWED: set[str] = {
    "read_file",
    "grep",
    "glob",
    "list_dir",
    "search_web",
    "project_structure",
    "git.status",
    "git.log",
    "git.diff",
    "git.branch",
}

# Tools that must escalate from AUTO → PROMPT in WORKSPACE_WRITE mode
# (writes to cwd are allowed, but shell/spawn/push still require approval).
_WORKSPACE_WRITE_ESCALATE: set[str] = {
    "shell",
    "spawn_agent",
    "git.push",
    "git.pull",
}


_current_sandbox_tier: SandboxTier = SandboxTier.UNRESTRICTED


def set_sandbox_tier(tier: SandboxTier) -> None:
    """Set the process-wide sandbox tier.  Called once from CLI entrypoint."""
    global _current_sandbox_tier
    _current_sandbox_tier = tier
    logger.info(f"[Permissions] Sandbox tier: {tier.value}")


def get_sandbox_tier() -> SandboxTier:
    return _current_sandbox_tier


def _clamp_tier_for_sandbox(key: str, tier: PermissionTier) -> PermissionTier:
    """Apply the active sandbox tier as a clamp on the per-tool tier."""
    sandbox = _current_sandbox_tier
    if sandbox == SandboxTier.UNRESTRICTED:
        return tier
    if sandbox == SandboxTier.READ_ONLY:
        if key in _READ_ONLY_ALLOWED:
            return tier
        return BLOCKED
    if sandbox == SandboxTier.WORKSPACE_WRITE:
        if key in _WORKSPACE_WRITE_ESCALATE and tier == AUTO:
            return PROMPT
        return tier
    return tier

# Shell commands that are auto-approved (first word of command)
SAFE_SHELL_COMMANDS = {
    "ls",
    "dir",
    "pwd",
    "find",
    "grep",
    "which",
    "where",
    "rg",
    "type",
    "get-childitem",
    "get-content",
    "select-string",
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
    # Git ops: read-only auto-approved, mutations require explicit approval
    "git.status": AUTO,
    "git.log": AUTO,
    "git.diff": AUTO,
    "git.branch": AUTO,
    "git.add": PROMPT,
    "git.commit": PROMPT,
    "git.push": PROMPT,
    "git.pull": PROMPT,
    # File edits: auto-approved (like Claude Code)
    "edit_file": AUTO,
    "write_file": AUTO,
    # Shell: prompt, but safe commands auto-approved
    "shell": PROMPT,
    # Sub-agent spawning
    "spawn_agent": PROMPT,
    # Command-layer destructive or agentic follow-up flows
    "clear_history": PROMPT,
    "auto_fix_tests": PROMPT,
    "retry_tier_escalation": PROMPT,
}

# Git actions that are read-only
GIT_READ_ACTIONS = frozenset({"status", "log", "diff", "branch"})


class PermissionManager:
    """Manages tool execution permissions."""

    def __init__(self, confirm_callback=None):
        self._permissions = dict(DEFAULT_PERMISSIONS)
        self._mode = "careful"
        self._trust_mode = False
        self._confirm_callback = confirm_callback
        self._always_approved: set[str] = set()
        # Session-scope trust: tool keys approved for the current session only.
        # Cleared when the process exits, unlike _always_approved (which is
        # intended to represent persistent allow_always decisions).
        self._session_approved: set[str] = set()

    def set_trust_mode(self, enabled: bool) -> None:
        self._trust_mode = enabled
        logger.info(f"[Permissions] Trust mode: {'ON' if enabled else 'OFF'}")

    @property
    def trust_mode(self) -> bool:
        return self._trust_mode

    @property
    def has_confirm_callback(self) -> bool:
        return self._confirm_callback is not None

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
        key = self._resolve_key(tool_name, args)

        # Sandbox tier is a hard clamp — READ_ONLY beats trust_mode/session approvals
        # so `aura --sandboxed` stays safe even if the user previously granted trust.
        if _current_sandbox_tier == SandboxTier.READ_ONLY and key not in _READ_ONLY_ALLOWED:
            logger.warning(f"[Permissions] BLOCKED by READ_ONLY sandbox: {key}")
            return False

        if self._trust_mode and _current_sandbox_tier != SandboxTier.WORKSPACE_WRITE:
            return True

        # Already permanently approved this session
        if key in self._always_approved:
            # Still enforce workspace-write escalation for dangerous tools
            if _current_sandbox_tier == SandboxTier.WORKSPACE_WRITE and key in _WORKSPACE_WRITE_ESCALATE:
                pass  # fall through to confirm_callback
            else:
                return True

        # Approved for the current session only
        if key in self._session_approved:
            if _current_sandbox_tier == SandboxTier.WORKSPACE_WRITE and key in _WORKSPACE_WRITE_ESCALATE:
                pass
            else:
                return True

        tier = self._permissions.get(key, PROMPT)
        tier = _clamp_tier_for_sandbox(key, tier)

        if tier == AUTO:
            return True
        if tier == BLOCKED:
            logger.warning(f"[Permissions] BLOCKED: {key}")
            return False

        # Shell commands: auto-approve if the first word is in SAFE_SHELL_COMMANDS
        if tool_name == "shell":
            cmd = args.get("command", "").strip()
            first_word = cmd.split()[0].split("/")[-1].split("\\")[-1].lower() if cmd else ""
            if first_word in SAFE_SHELL_COMMANDS:
                return True

        # PROMPT tier — ask user
        if self._confirm_callback:
            description = self._format_action_description(tool_name, args)
            result = self._confirm_callback(tool_name, description)
            # New vocabulary from permissions_dialog.request_permission():
            if result == "allow_always" or result == "always":
                self._always_approved.add(key)
                return True
            if result == "allow_session":
                self._session_approved.add(key)
                return True
            if result == "allow_once":
                return True
            if result == "deny":
                return False
            # Back-compat: legacy callbacks returning bool
            return bool(result)

        # No callback = deny by default
        return False

    @property
    def current_mode(self) -> str:
        """Return current permission mode name for display/checks."""
        if self._trust_mode:
            return "full_auto"
        return self._mode

    @property
    def mode(self) -> str:
        """Compatibility alias for code paths that read permissions.mode."""
        return self.current_mode

    def set_mode(self, mode: str) -> None:
        """Set the base permission mode.

        Full-auto is modeled as a trust override so disabling trust can
        return to the previously selected base mode.
        """
        if mode == "full_auto":
            self._trust_mode = True
            return
        self._mode = mode
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
        elif tool_name == "spawn_agent":
            task = args.get("task", "").strip()
            specialist = args.get("specialist", "").strip()
            if specialist and task:
                return f"Spawn agent '{specialist}'\n  task: {task[:120]}"
            if task:
                return f"Spawn agent\n  task: {task[:120]}"
            return "Spawn agent"
        elif tool_name == "clear_history":
            return "Clear conversation history"
        elif tool_name == "auto_fix_tests":
            command = args.get("command", "").strip()
            failure_count = args.get("failure_count")
            if command and failure_count is not None:
                return f"Auto-fix failing tests\n  command: {command}\n  failures: {failure_count}"
            if command:
                return f"Auto-fix failing tests\n  command: {command}"
            return "Auto-fix failing tests"
        elif tool_name == "retry_tier_escalation":
            from_tier = args.get("from_tier", "").strip()
            to_tier = args.get("to_tier", "").strip()
            prompt = args.get("prompt", "").strip()
            if from_tier and to_tier:
                desc = f"Escalate retry tier\n  {from_tier} -> {to_tier}"
                if prompt:
                    desc += f"\n  prompt: {prompt[:120]}"
                return desc
            return "Escalate retry tier"
        return f"{tool_name}({args})"
