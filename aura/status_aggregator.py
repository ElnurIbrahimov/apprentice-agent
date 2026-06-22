"""Status aggregator — unified component status check.

Mirrors Hermes Agent's `hermes status --all` pattern.

Reports the status of all major Aura components in one view:
  - Models / providers
  - Memory systems
  - Active toolsets
  - Scheduled jobs
  - Session count
  - Activity log stats
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def get_full_status() -> dict:
    """Gather status from all major Aura subsystems.

    Returns a dict with sections:
      - providers: configured/total counts
      - models: active default model
      - profile: active profile name
      - toolsets: enabled/total
      - cron: active jobs count
      - sessions: total count
      - activity: total interactions, tokens, cost
      - security: approval mode, redaction enabled
    """
    status = {}

    # Providers
    try:
        from aura.providers import list_configured_providers
        providers = list_configured_providers()
        status["providers"] = {
            "total": len(providers),
            "configured": sum(1 for p in providers if p["configured"]),
            "names": [p["name"] for p in providers if p["configured"]],
        }
    except Exception as e:
        status["providers"] = {"error": str(e)}

    # Models
    try:
        from aura.config import Config
        status["models"] = {
            "default": Config.MODEL_NAME,
            "fast": Config.MODEL_FAST,
            "reason": Config.MODEL_REASON,
            "code": Config.MODEL_CODE,
        }
    except Exception as e:
        status["models"] = {"error": str(e)}

    # Profile
    try:
        from aura.profiles import get_active_profile_name
        status["profile"] = {"active": get_active_profile_name()}
    except Exception as e:
        status["profile"] = {"error": str(e)}

    # Toolsets
    try:
        from aura.toolsets import list_toolsets
        toolsets = list_toolsets()
        status["toolsets"] = {
            "total": len(toolsets),
            "enabled": sum(1 for ts in toolsets if ts["enabled"]),
        }
    except Exception as e:
        status["toolsets"] = {"error": str(e)}

    # Cron jobs
    try:
        import json
        from pathlib import Path
        cron_file = Path.home() / ".aura" / "cron_jobs.json"
        if cron_file.exists():
            jobs = json.loads(cron_file.read_text(encoding="utf-8"))
            status["cron"] = {
                "total": len(jobs),
                "active": sum(1 for j in jobs if j.get("enabled", True)),
            }
        else:
            status["cron"] = {"total": 0, "active": 0}
    except Exception as e:
        status["cron"] = {"error": str(e)}

    # Sessions
    try:
        from aura.cli.commands.sessions_cli_commands import _load_session_summaries
        sessions = _load_session_summaries()
        status["sessions"] = {"total": len(sessions)}
    except Exception as e:
        status["sessions"] = {"error": str(e)}

    # Activity log
    try:
        from aura.cli.activity_log import ActivityLog
        log = ActivityLog()
        stats = log.get_stats()
        status["activity"] = {
            "interactions": stats.get("total_interactions", 0),
            "tokens_in": stats.get("total_tokens_in", 0),
            "tokens_out": stats.get("total_tokens_out", 0),
            "cost": stats.get("total_cost", 0.0),
            "tool_calls": stats.get("total_tool_calls", 0),
        }
    except Exception as e:
        status["activity"] = {"error": str(e)}

    # Security
    try:
        from aura.security_config import (
            get_approvals_mode, is_secret_redaction_enabled, is_pii_redaction_enabled
        )
        status["security"] = {
            "approvals_mode": get_approvals_mode(),
            "redact_secrets": is_secret_redaction_enabled(),
            "redact_pii": is_pii_redaction_enabled(),
        }
    except Exception as e:
        status["security"] = {"error": str(e)}

    # Compression
    try:
        from aura.compression_config import is_compression_enabled, get_compression_threshold
        status["compression"] = {
            "enabled": is_compression_enabled(),
            "threshold": get_compression_threshold(),
        }
    except Exception as e:
        status["compression"] = {"error": str(e)}

    return status
