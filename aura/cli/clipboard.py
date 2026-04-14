"""Clipboard integration for chat-mode input: Ctrl+V text and image paste.

Both dependencies (pyperclip, Pillow) are optional; functions degrade to
returning None if unavailable so the rest of the CLI keeps working on
systems without them.
"""
from __future__ import annotations

import logging
import os
import tempfile
import time
from typing import Optional

logger = logging.getLogger(__name__)


def read_clipboard_text() -> Optional[str]:
    """Return clipboard text or None on failure / missing dep."""
    try:
        import pyperclip
    except ImportError:
        logger.debug("[clipboard] pyperclip not installed")
        return None
    try:
        text = pyperclip.paste()
        return text if text else None
    except Exception as exc:
        logger.debug(f"[clipboard] text paste failed: {exc}")
        return None


def read_clipboard_image() -> Optional[str]:
    """If the clipboard holds an image, save it to a temp PNG and return path.

    Returns None if no image is present, the dependency is missing, or
    the save fails.
    """
    try:
        from PIL import ImageGrab
    except ImportError:
        logger.debug("[clipboard] Pillow not installed")
        return None
    try:
        img = ImageGrab.grabclipboard()
    except Exception as exc:
        logger.debug(f"[clipboard] image grab failed: {exc}")
        return None
    if img is None:
        return None
    # On Linux ImageGrab may return a list of file paths for a copied file.
    if isinstance(img, list):
        for candidate in img:
            try:
                from PIL import Image
                Image.open(candidate).verify()
                return candidate
            except Exception:
                continue
        return None
    if not hasattr(img, "save"):
        return None
    try:
        path = os.path.join(tempfile.gettempdir(), f"aura_paste_{int(time.time())}.png")
        img.save(path, "PNG")
        return path
    except Exception as exc:
        logger.debug(f"[clipboard] image save failed: {exc}")
        return None
