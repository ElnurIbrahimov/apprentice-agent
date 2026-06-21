"""Session auto-prune — automatic cleanup of old sessions.

Mirrors Hermes Agent's sessions.auto_prune config:
  sessions:
    auto_prune: true
    retention_days: 90
    vacuum_after_prune: true
    min_interval_hours: 24

When run, prunes sessions older than retention_days. The min_interval
prevents accidental double-runs.
"""
from __future__ import annotations

import json
import logging
import shutil
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


_LAST_PRUNE_FILE = Path.home() / ".aura" / ".last_session_prune"


def get_sessions_config() -> dict:
    """Get the sessions config."""
    try:
        from aura.config_loader import get_config_value
        return get_config_value("sessions", {}) or {}
    except ImportError:
        return {}


def is_auto_prune_enabled() -> bool:
    """Check if auto-prune is enabled."""
    return get_sessions_config().get("auto_prune", False)


def get_retention_days() -> int:
    """Get the retention period in days."""
    return int(get_sessions_config().get("retention_days", 90))


def get_min_interval_hours() -> int:
    """Get the minimum interval between prunes in hours."""
    return int(get_sessions_config().get("min_interval_hours", 24))


def _get_sessions_dir() -> Path:
    """Get the sessions directory (profile-aware)."""
    from aura.cli.commands.sessions_cli_commands import _get_sessions_dir
    return _get_sessions_dir()


def should_auto_prune() -> bool:
    """Check if auto-prune should run now (based on min interval)."""
    if not is_auto_prune_enabled():
        return False

    if not _LAST_PRUNE_FILE.exists():
        return True

    try:
        last = float(_LAST_PRUNE_FILE.read_text(encoding="utf-8").strip())
        return (time.time() - last) > (get_min_interval_hours() * 3600)
    except (OSError, ValueError):
        return True


def run_auto_prune(dry_run: bool = False) -> dict:
    """Run session auto-prune.

    Returns dict with:
      - pruned: int (sessions deleted)
      - errors: int
      - dry_run: bool
    """
    result = {"pruned": 0, "errors": 0, "dry_run": dry_run}

    if not is_auto_prune_enabled() and not dry_run:
        return result

    sessions_dir = _get_sessions_dir()
    if not sessions_dir.exists():
        return result

    cutoff = time.time() - (get_retention_days() * 86400)

    for d in sessions_dir.iterdir():
        if not d.is_dir():
            continue
        session_file = d / "session.json"
        if not session_file.exists():
            continue

        try:
            data = json.loads(session_file.read_text(encoding="utf-8"))
            updated = data.get("updated_at", 0)
            if updated > 0 and updated < cutoff:
                if not dry_run:
                    shutil.rmtree(d)
                result["pruned"] += 1
        except (json.JSONDecodeError, OSError) as e:
            result["errors"] += 1
            logger.debug(f"Prune error for {d}: {e}")

    if not dry_run:
        _LAST_PRUNE_FILE.parent.mkdir(parents=True, exist_ok=True)
        _LAST_PRUNE_FILE.write_text(str(time.time()), encoding="utf-8")

        # Optional: vacuum SQLite activity log
        if get_sessions_config().get("vacuum_after_prune", False):
            try:
                from aura.cli.activity_log import ActivityLog
                log = ActivityLog()
                # VACUUM isn't directly exposed; just log the count
                stats = log.get_stats()
                logger.info(f"Activity log: {stats.get('total_interactions', 0)} interactions")
            except Exception:
                pass

    return result
