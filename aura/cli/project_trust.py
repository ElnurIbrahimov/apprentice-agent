"""Per-project AURA.md trust store.

The CLI loads hooks from a project's AURA.md on session start; those hooks
execute arbitrary shell commands. A malicious or unreviewed AURA.md in a
cloned repo would fire hooks as the current user without consent.

This module gates hook loading behind a first-run trust prompt (see
permissions_dialog.request_project_trust). Approvals are persisted by
(project_root, sha256(AURA.md)) so re-opening the same repo with an
unchanged AURA.md doesn't re-prompt, but any modification re-prompts.

Trust store: ``~/.aura/trusted_projects.json`` — list of
``{"root": str, "sha256": str, "trusted_at": float}``.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_TRUST_FILE = Path.home() / ".aura" / "trusted_projects.json"


def _normalize_root(root: str) -> str:
    """Canonicalize a project root path for reliable equality."""
    try:
        return os.path.normcase(os.path.abspath(root))
    except (OSError, ValueError):
        return root


def _hash_file(path: str) -> Optional[str]:
    """SHA-256 the given file's raw bytes. Returns None if unreadable."""
    try:
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()
    except (OSError, PermissionError):
        return None


def _load_store() -> list[dict]:
    """Load the trust store, returning [] on missing/corrupt file."""
    try:
        if _TRUST_FILE.is_file():
            with open(_TRUST_FILE, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return data
    except (OSError, json.JSONDecodeError):
        logger.debug("trust_store_load_failed", exc_info=True)
    return []


def _save_store(entries: list[dict]) -> None:
    """Persist the trust store atomically with 0o600 perms."""
    try:
        _TRUST_FILE.parent.mkdir(parents=True, exist_ok=True)
        tmp = _TRUST_FILE.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(entries, f, indent=2)
        try:
            os.chmod(tmp, 0o600)
        except OSError:
            pass  # Windows doesn't honor chmod; ACL inherited from ~/.aura/
        os.replace(tmp, _TRUST_FILE)
    except OSError:
        logger.debug("trust_store_save_failed", exc_info=True)


def is_trusted(root: str, aura_md_path: str) -> bool:
    """Return True iff (root, sha256(AURA.md)) is in the trust store.

    Any mismatch — missing entry, changed file hash, unreadable file —
    yields False so the caller re-prompts.
    """
    sha = _hash_file(aura_md_path)
    if not sha:
        return False
    norm_root = _normalize_root(root)
    for entry in _load_store():
        if (entry.get("root") == norm_root
                and entry.get("sha256") == sha):
            return True
    return False


def mark_trusted(root: str, aura_md_path: str) -> bool:
    """Record (root, sha256) as trusted. Returns True on success."""
    sha = _hash_file(aura_md_path)
    if not sha:
        return False
    norm_root = _normalize_root(root)
    entries = _load_store()
    # Replace any prior entry for this root so the list doesn't grow
    # unbounded when AURA.md changes repeatedly.
    entries = [e for e in entries if e.get("root") != norm_root]
    entries.append({
        "root": norm_root,
        "sha256": sha,
        "trusted_at": time.time(),
    })
    _save_store(entries)
    return True
