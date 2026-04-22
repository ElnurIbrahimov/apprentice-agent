"""Shared path constants for Aura.

Anchored on the package location — not the current working directory — so data
writes land in the same place regardless of where Aura is invoked from.
"""

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Union

AURA_ROOT: Path = Path(__file__).resolve().parent.parent
AURA_DATA_DIR: Path = AURA_ROOT / "aura_data"
SKILL_LIBRARY_DIR: Path = AURA_DATA_DIR / "skill_library"
EVOLUTION_RUNS_DIR: Path = AURA_DATA_DIR / "evolution_runs"
KNOWLEDGE_GRAPH_DIR: Path = AURA_DATA_DIR / "knowledge_graph"


def ensure_data_dirs() -> None:
    """Create core data directories if they don't exist. Safe to call repeatedly."""
    for d in (AURA_DATA_DIR, SKILL_LIBRARY_DIR, EVOLUTION_RUNS_DIR, KNOWLEDGE_GRAPH_DIR):
        d.mkdir(parents=True, exist_ok=True)


def atomic_write_json(
    path: Union[str, Path],
    payload: Any,
    *,
    indent: int = 2,
    ensure_ascii: bool = False,
) -> None:
    """Write *payload* as JSON to *path* with crash-safe durability.

    Writes to a sibling tempfile, fsyncs the data, renames over the target,
    then fsyncs the parent directory. On POSIX, after this returns the file
    survives power loss. On Windows, file-level fsync happens; parent-dir
    fsync is skipped (no-op — NTFS handles rename durability differently).

    Raises OSError on I/O failure. The tempfile is cleaned up on error.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(payload, indent=indent, ensure_ascii=ensure_ascii)
    fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, str(path))
    except Exception:
        try:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
        except OSError:
            pass
        raise
    # Parent-dir fsync (POSIX only) — rename itself is journalled on ext4,
    # but the parent's directory entry isn't durable until its metadata is flushed.
    if hasattr(os, "O_DIRECTORY"):
        try:
            dirfd = os.open(str(path.parent), os.O_DIRECTORY)
            try:
                os.fsync(dirfd)
            finally:
                os.close(dirfd)
        except OSError:
            pass
