"""JSONL log rotation utility for AURA.

Prevents unbounded growth of append-only JSONL files by rotating
them when they exceed a size threshold.
"""

import glob
import logging
import os
import time

logger = logging.getLogger(__name__)


def rotate_jsonl_if_needed(
    filepath,
    max_bytes: int = 5 * 1024 * 1024,
    max_backups: int = 10,
) -> None:
    """Rotate a JSONL file if it exceeds *max_bytes*.

    Rotation renames the current file to ``<stem>.<timestamp>.jsonl``
    and prunes old backups so at most *max_backups* rotated copies exist.

    The function is designed to be called **before** every append.
    All filesystem errors are caught so a failed rotation never
    crashes the caller.

    Args:
        filepath: Path (str or Path-like) to the JSONL file.
        max_bytes: Rotate when the file reaches this size (default 5 MB).
        max_backups: Maximum number of rotated files to keep.
    """
    try:
        filepath = str(filepath)

        # Nothing to rotate if file doesn't exist or is small enough
        try:
            size = os.path.getsize(filepath)
        except OSError:
            return
        if size < max_bytes:
            return

        # Build rotated filename with millisecond timestamp
        base, ext = os.path.splitext(filepath)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        rotated = f"{base}.{timestamp}{ext}"

        # Avoid collision (unlikely but possible within the same second)
        if os.path.exists(rotated):
            rotated = f"{base}.{timestamp}_{int(time.time() * 1000) % 1000}{ext}"

        os.rename(filepath, rotated)
        logger.info("Rotated %s -> %s (was %d bytes)", filepath, rotated, size)

        # Prune old backups
        pattern = f"{base}.*{ext}"
        backups = sorted(glob.glob(pattern))
        if len(backups) > max_backups:
            for old in backups[: len(backups) - max_backups]:
                try:
                    os.remove(old)
                    logger.info("Pruned old backup %s", old)
                except OSError:
                    pass

    except OSError as exc:
        logger.warning("JSONL rotation failed for %s: %s", filepath, exc)
