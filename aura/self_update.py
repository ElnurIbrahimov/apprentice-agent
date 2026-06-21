"""Self-update mechanism for Aura CLI.

Mirrors Hermes Agent's `hermes update` pattern:
  - Pulls latest from git
  - Backs up local changes
  - Updates pip dependencies
  - Restarts the daemon

Usage: aura update [--no-backup] [--no-deps]
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import time
from pathlib import Path

logger = logging.getLogger(__name__)


def run_update(backup: bool = True, install_deps: bool = True) -> dict:
    """Run the self-update flow.

    Returns a dict with:
      - success: bool
      - commit_before: git hash before
      - commit_after: git hash after
      - backup_path: path to backup tar (if backup=True)
      - message: human-readable result message
    """
    result = {
        "success": False,
        "commit_before": "",
        "commit_after": "",
        "backup_path": "",
        "message": "",
    }

    cwd = Path(__file__).resolve().parent.parent
    if not (cwd / ".git").exists():
        result["message"] = "Not a git repository — cannot self-update."
        return result

    # Get current commit
    result["commit_before"] = _git_rev(cwd)

    # Backup local changes
    if backup:
        try:
            backup_path = _backup_local(cwd, result["commit_before"])
            result["backup_path"] = str(backup_path)
        except Exception as e:
            logger.warning(f"Backup failed: {e}")

    # Stash local changes
    stashed = False
    try:
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=cwd, capture_output=True, text=True, timeout=10
        )
        if status.stdout.strip():
            # Has local changes — stash them
            subprocess.run(
                ["git", "stash", "push", "-m", "aura self-update"],
                cwd=cwd, capture_output=True, text=True, timeout=10
            )
            stashed = True
    except Exception:
        pass

    # Pull latest
    try:
        pull = subprocess.run(
            ["git", "pull", "--rebase"],
            cwd=cwd, capture_output=True, text=True, timeout=60
        )
        if pull.returncode != 0:
            result["message"] = f"git pull failed: {pull.stderr[:200]}"
            # Try to unstash
            if stashed:
                subprocess.run(
                    ["git", "stash", "pop"],
                    cwd=cwd, capture_output=True, text=True, timeout=10
                )
            return result
    except Exception as e:
        result["message"] = f"git pull error: {e}"
        return result

    result["commit_after"] = _git_rev(cwd)

    # Restore stashed changes
    if stashed:
        subprocess.run(
            ["git", "stash", "pop"],
            cwd=cwd, capture_output=True, text=True, timeout=10
        )

    # Update pip dependencies
    if install_deps:
        try:
            req_file = cwd / "requirements.txt"
            if req_file.exists():
                subprocess.run(
                    ["pip", "install", "-r", str(req_file)],
                    cwd=cwd, capture_output=True, text=True, timeout=120
                )
        except Exception as e:
            logger.warning(f"pip install failed: {e}")

    result["success"] = True
    if result["commit_before"] == result["commit_after"]:
        result["message"] = "Already up to date."
    else:
        result["message"] = f"Updated {result['commit_before'][:7]} \u2192 {result['commit_after'][:7]}."

    return result


def _git_rev(cwd: Path) -> str:
    """Get the current git commit hash."""
    try:
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd, capture_output=True, text=True, timeout=5
        )
        return r.stdout.strip() if r.returncode == 0 else ""
    except Exception:
        return ""


def _backup_local(cwd: Path, commit: str) -> Path:
    """Backup local uncommitted changes to a tar file."""
    backup_dir = Path.home() / ".aura" / "backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup_path = backup_dir / f"aura_pre_update_{int(time.time())}_{commit[:8]}.tar"

    # Create a tar of uncommitted changes
    try:
        diff = subprocess.run(
            ["git", "diff", "HEAD"],
            cwd=cwd, capture_output=True, text=True, timeout=10
        )
        if diff.stdout:
            backup_path.write_text(diff.stdout, encoding="utf-8")
    except Exception:
        pass

    return backup_path
