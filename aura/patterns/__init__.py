"""Pattern recognition for AURA - cross-conversation pattern detection."""
import threading
from typing import Optional

from .pattern_prophet import PatternProphet, Pattern

__all__ = ["PatternProphet", "Pattern", "get_pattern_prophet"]

_prophet: Optional[PatternProphet] = None
_prophet_lock = threading.Lock()


def get_pattern_prophet() -> PatternProphet:
    """Get or create the singleton PatternProphet (double-checked locking)."""
    global _prophet
    if _prophet is None:
        with _prophet_lock:
            if _prophet is None:
                _prophet = PatternProphet()  # defaults to aura/data
    return _prophet
