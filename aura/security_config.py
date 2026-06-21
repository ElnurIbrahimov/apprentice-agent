"""Security configuration — config-driven security toggles.

Inspired by Hermes Agent's config-driven security settings:
  security:
    redact_secrets: true     # auto-mask API keys in tool output
    redact_pii: false        # hash user IDs, strip phone numbers
    approvals_mode: manual   # manual | smart | off
    website_blocklist:
      enabled: false
      domains: [example.com]
  tool_loop_guardrails:
    warn_after: {exact_failure: 2, same_tool_failure: 3}
    hard_stop_after: {exact_failure: 5, same_tool_failure: 8}

These settings are read from config.yaml and accessed via this module.
The existing security subsystems (taint_tracker, ssrf_guard, audit_chain,
tool_signing) continue to work — this module adds config-driven control.
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def get_security_config() -> dict:
    """Get the security config section."""
    try:
        from aura.config_loader import get_config_value
        return get_config_value("security", {}) or {}
    except ImportError:
        return {}


def is_secret_redaction_enabled() -> bool:
    """Check if secret redaction is enabled."""
    cfg = get_security_config()
    return cfg.get("redact_secrets", False)


def is_pii_redaction_enabled() -> bool:
    """Check if PII redaction is enabled."""
    cfg = get_security_config()
    return cfg.get("redact_pii", False)


def get_approvals_mode() -> str:
    """Get the approval mode.

    Returns: 'manual' (default), 'smart', or 'off'.
    """
    cfg = get_security_config()
    return cfg.get("approvals_mode", "manual")


def get_website_blocklist() -> dict:
    """Get the website blocklist config."""
    cfg = get_security_config()
    return cfg.get("website_blocklist", {})


def get_tool_loop_guardrails() -> dict:
    """Get tool loop guardrail thresholds."""
    try:
        from aura.config_loader import get_config_value
        return get_config_value("tool_loop_guardrails", {}) or {}
    except ImportError:
        return {}


def get_guardrail_thresholds() -> dict:
    """Get warn/hard-stop thresholds for tool loop guardrails.

    Returns dict with 'warn' and 'hard_stop' keys, each containing
    {exact_failure, same_tool_failure, idempotent_no_progress} thresholds.
    """
    cfg = get_tool_loop_guardrails()
    return {
        "warn": cfg.get("warn_after", {
            "exact_failure": 2,
            "same_tool_failure": 3,
            "idempotent_no_progress": 2,
        }),
        "hard_stop": cfg.get("hard_stop_after", {
            "exact_failure": 5,
            "same_tool_failure": 8,
            "idempotent_no_progress": 5,
        }),
    }


def should_redact_text(text: str) -> str:
    """Redact secrets and PII from text if enabled.

    Uses the existing taint_tracker patterns for detection.
    Returns the text unchanged if redaction is disabled.
    """
    if not is_secret_redaction_enabled() and not is_pii_redaction_enabled():
        return text

    try:
        from aura.security.taint_tracker import TaintTracker, TaintLabel
        tracker = TaintTracker()
        # Check and redact
        if is_secret_redaction_enabled():
            # Redact SECRET-level patterns
            for pattern, label, name in tracker._SECRET_PATTERNS:
                import re
                text = re.sub(pattern, f"[REDACTED:{name}]", text)
        if is_pii_redaction_enabled():
            for pattern, label, name in tracker._PII_PATTERNS:
                import re
                text = re.sub(pattern, f"[REDACTED:{name}]", text)
    except ImportError:
        pass

    return text


def classify_command_risk(command: str, brain: Any = None) -> str:
    """Classify a shell command's risk level.

    Used by smart approval mode. Returns 'low', 'medium', or 'high'.

    If brain is provided and smart mode is enabled, uses an LLM call.
    Otherwise, uses rule-based classification.
    """
    mode = get_approvals_mode()
    if mode != "smart":
        return "unknown"  # smart mode not enabled

    # Try LLM-based classification
    if brain is not None:
        try:
            prompt = (
                f"Classify the risk of this shell command as 'low', 'medium', or 'high'.\n"
                f"Low: read-only commands (ls, cat, grep, git status, echo).\n"
                f"Medium: file writes, git commit, pip install.\n"
                f"High: rm, git push --force, git reset --hard, sudo, chmod.\n\n"
                f"Command: {command}\n"
                f"Risk level (respond with one word):"
            )
            from aura.config import Config
            client, model = brain._get_client_for_model(Config.MODEL_FAST)
            resp = client.chat(model, [{"role": "user", "content": prompt}])
            result = resp.get("message", {}).get("content", "").strip().lower()
            if "low" in result:
                return "low"
            elif "high" in result:
                return "high"
            elif "medium" in result:
                return "medium"
        except Exception:
            pass

    # Fallback: rule-based classification
    cmd_lower = command.lower().strip()
    high_patterns = ["rm -rf", "rm -r", "git push --force", "git reset --hard",
                     "sudo ", "chmod 777", "mkfs", "dd if=", ":(){ :|:& };:",
                     "git push -f", "> /dev/sd", "shutdown", "reboot"]
    medium_patterns = ["rm ", "git commit", "git push", "pip install", "npm install",
                       "mv ", "cp ", ">", ">>", "sed -i", "echo ", "curl", "wget"]

    for pattern in high_patterns:
        if pattern in cmd_lower:
            return "high"
    for pattern in medium_patterns:
        if pattern in cmd_lower:
            return "medium"
    return "low"


def should_auto_approve(command: str, brain: Any = None) -> bool:
    """Check if a command should be auto-approved based on approval mode.

    Returns True if the command is safe to run without prompting.
    """
    mode = get_approvals_mode()
    if mode == "off":
        return True
    if mode == "manual":
        return False
    if mode == "smart":
        risk = classify_command_risk(command, brain)
        return risk == "low"
    return False
