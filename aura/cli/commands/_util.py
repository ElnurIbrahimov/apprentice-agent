"""Shared helpers for slash-command handlers.

Keeps per-command modules lean and prevents drift (each file redoing
tilde expansion / path normalization with slightly different bugs).
"""
from __future__ import annotations

from pathlib import Path


def resolve_user_path(raw: str) -> Path:
    """Normalize a user-supplied path.

    Expands ``~`` to the user's home directory and resolves any leading
    ``$VAR`` style env refs. Returns a ``Path`` object; the caller decides
    whether to call ``.resolve()`` (often unwise for non-existent files).

    On Windows ``Path("~/foo.py")`` returns a literal ``~`` folder, which
    confuses every downstream tool. Routing everything through this helper
    stops that bug class.
    """
    import os
    if not raw:
        return Path("")
    # expandvars first so `$HOME/foo` works even when the shell already expanded it
    expanded = os.path.expandvars(raw)
    return Path(expanded).expanduser()
