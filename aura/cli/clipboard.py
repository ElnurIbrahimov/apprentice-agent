"""Clipboard integration for chat-mode input: Ctrl+V text and image paste.

Both dependencies (pyperclip, Pillow) are optional; functions degrade to
returning None if unavailable so the rest of the CLI keeps working on
systems without them.

The first time a dependency is missing we surface a one-line hint through the
shared console so users know WHY their paste didn't land — previously this
failed completely silently and users assumed Ctrl+V was broken.
"""
from __future__ import annotations

import logging
import os
import tempfile
import time
from typing import Optional

logger = logging.getLogger(__name__)

# One-shot warning flags so a missing dep emits exactly one user-visible hint
# per process, not a flood on every Ctrl+V.
_warned_pyperclip = False
_warned_pillow = False


def _warn_once(flag_attr: str, message: str) -> None:
    """Print a dim one-liner to the shared console exactly once per flag."""
    if globals().get(flag_attr, False):
        return
    globals()[flag_attr] = True
    try:
        from .display import console as _console
        _console.print(f"[dim yellow]{message}[/dim yellow]")
    except Exception:
        # Display may not be initialized yet in some code paths — fall back
        # to debug logging so we still leave a trace.
        logger.debug(message)


def read_clipboard_text() -> Optional[str]:
    """Return clipboard text or None on failure / missing dep."""
    try:
        import pyperclip
    except ImportError:
        _warn_once(
            "_warned_pyperclip",
            "clipboard: pyperclip not installed — text paste disabled. "
            "pip install pyperclip",
        )
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
        _warn_once(
            "_warned_pillow",
            "clipboard: Pillow not installed — image paste disabled. "
            "pip install Pillow",
        )
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
        import uuid as _uuid
        # uuid suffix prevents collisions when the user pastes twice inside
        # the same second (int(time.time()) was the only discriminator).
        path = os.path.join(
            tempfile.gettempdir(),
            f"aura_paste_{int(time.time())}_{_uuid.uuid4().hex[:6]}.png",
        )
        img.save(path, "PNG")
        return path
    except Exception as exc:
        logger.debug(f"[clipboard] image save failed: {exc}")
        return None
