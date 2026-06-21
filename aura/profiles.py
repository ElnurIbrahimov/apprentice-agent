"""Profile system for Aura — isolated instances with own config/sessions/skills.

Inspired by Hermes Agent's profile system. Each profile has its own
directory under ~/.aura/profiles/<name>/ with:
  - config.yaml    (provider config, auxiliary roles, toolsets)
  - .env           (API keys, secrets)
  - sessions/      (conversation history)
  - skills/        (custom skills)
  - memory/        (memory databases)

Usage:
  aura profile list              List all profiles
  aura profile create <name>     Create a new profile
  aura profile use <name>        Set active profile (sticky via AURA_PROFILE env)
  aura profile show [name]       Show profile details
  aura profile delete <name>     Delete a profile
  aura --profile <name>          Use profile for this session only

The active profile is stored in ~/.aura/active_profile file.
Env var AURA_PROFILE overrides the file for the current process.
"""
from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


def get_aura_home() -> Path:
    """Return ~/.aura or AURA_HOME."""
    env = os.environ.get("AURA_HOME")
    if env:
        return Path(env)
    return Path.home() / ".aura"


def get_profiles_dir() -> Path:
    """Return the profiles directory."""
    return get_aura_home() / "profiles"


def get_active_profile_name() -> str:
    """Get the active profile name.

    Priority: AURA_PROFILE env var > ~/.aura/active_profile file > 'default'.
    """
    env_profile = os.environ.get("AURA_PROFILE")
    if env_profile:
        return env_profile

    active_file = get_aura_home() / "active_profile"
    if active_file.exists():
        try:
            return active_file.read_text(encoding="utf-8").strip()
        except OSError:
            pass

    return "default"


def set_active_profile(name: str) -> bool:
    """Set the active profile (persists to ~/.aura/active_profile).

    Also sets AURA_PROFILE env var for the current process.
    """
    if not profile_exists(name):
        return False

    active_file = get_aura_home() / "active_profile"
    try:
        active_file.parent.mkdir(parents=True, exist_ok=True)
        active_file.write_text(name, encoding="utf-8")
        os.environ["AURA_PROFILE"] = name
        return True
    except OSError as e:
        logger.error(f"[Profiles] Failed to set active profile: {e}")
        return False


def get_profile_dir(name: str) -> Path:
    """Get the directory path for a profile."""
    if name == "default":
        return get_aura_home()
    return get_profiles_dir() / name


def profile_exists(name: str) -> bool:
    """Check if a profile exists."""
    if name == "default":
        return True
    return get_profile_dir(name).exists()


def list_profiles() -> List[dict]:
    """List all profiles with metadata."""
    profiles = []

    # Always include 'default'
    active = get_active_profile_name()
    profiles.append(_profile_info("default", active))

    # Scan profiles directory
    profiles_dir = get_profiles_dir()
    if profiles_dir.exists():
        for entry in sorted(profiles_dir.iterdir()):
            if entry.is_dir() and (entry / "config.yaml").exists():
                profiles.append(_profile_info(entry.name, active))

    return profiles


def _profile_info(name: str, active: str) -> dict:
    """Get info dict for a profile."""
    pdir = get_profile_dir(name)
    info = {
        "name": name,
        "active": name == active,
        "path": str(pdir),
        "has_config": (pdir / "config.yaml").exists(),
        "has_env": (pdir / ".env").exists(),
    }
    # Count sessions if directory exists
    sessions_dir = pdir / "sessions"
    if sessions_dir.exists():
        info["session_count"] = len(list(sessions_dir.glob("*.json")))
    else:
        info["session_count"] = 0
    return info


def create_profile(name: str, clone_from: Optional[str] = None) -> bool:
    """Create a new profile.

    Args:
        name: Profile name (must not be 'default' or already exist).
        clone_from: Optional profile name to clone config/env from.
    """
    if name == "default":
        logger.error("[Profiles] 'default' is reserved")
        return False

    pdir = get_profile_dir(name)
    if pdir.exists():
        logger.error(f"[Profiles] Profile '{name}' already exists")
        return False

    try:
        pdir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        (pdir / "sessions").mkdir(exist_ok=True)
        (pdir / "skills").mkdir(exist_ok=True)

        # Clone or create config.yaml
        if clone_from:
            src_dir = get_profile_dir(clone_from)
            if (src_dir / "config.yaml").exists():
                shutil.copy2(src_dir / "config.yaml", pdir / "config.yaml")
            else:
                (pdir / "config.yaml").write_text(
                    "# Aura configuration — cloned from default\n", encoding="utf-8"
                )
            if (src_dir / ".env").exists():
                shutil.copy2(src_dir / ".env", pdir / ".env")
        else:
            (pdir / "config.yaml").write_text(
                "# Aura configuration\n", encoding="utf-8"
            )

        logger.info(f"[Profiles] Created profile '{name}' at {pdir}")
        return True
    except OSError as e:
        logger.error(f"[Profiles] Failed to create profile: {e}")
        return False


def delete_profile(name: str) -> bool:
    """Delete a profile."""
    if name == "default":
        logger.error("[Profiles] Cannot delete 'default' profile")
        return False

    pdir = get_profile_dir(name)
    if not pdir.exists():
        logger.error(f"[Profiles] Profile '{name}' does not exist")
        return False

    try:
        shutil.rmtree(pdir)

        # If this was the active profile, switch back to default
        if get_active_profile_name() == name:
            set_active_profile("default")

        logger.info(f"[Profiles] Deleted profile '{name}'")
        return True
    except OSError as e:
        logger.error(f"[Profiles] Failed to delete profile: {e}")
        return False


def show_profile(name: Optional[str] = None) -> dict:
    """Show details for a profile (defaults to active)."""
    if name is None:
        name = get_active_profile_name()
    return _profile_info(name, name)
