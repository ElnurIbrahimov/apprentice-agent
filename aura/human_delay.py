"""Human delay — natural typing delay for responses.

Inspired by Hermes Agent's human_delay config:
  human_delay:
    mode: off   # off | natural | fixed
    min_ms: 800
    max_ms: 2500

Makes the agent feel more human in messaging contexts by adding
small delays between characters or words when streaming responses.
"""
from __future__ import annotations

import asyncio
import logging
import random
import time

logger = logging.getLogger(__name__)


def get_human_delay_config() -> dict:
    """Get the human delay config."""
    try:
        from aura.config_loader import get_config_value
        return get_config_value("human_delay", {}) or {}
    except ImportError:
        return {}


def is_human_delay_enabled() -> bool:
    """Check if human delay is enabled."""
    cfg = get_human_delay_config()
    return cfg.get("mode", "off") != "off"


def get_delay_range() -> tuple[int, int]:
    """Get the min/max delay in milliseconds."""
    cfg = get_human_delay_config()
    return (
        int(cfg.get("min_ms", 800)),
        int(cfg.get("max_ms", 2500)),
    )


def humanize_delay(estimated_tokens: int = 0) -> float:
    """Calculate a natural human-typing delay.

    Args:
        estimated_tokens: Number of tokens in the response (for proportion).

    Returns:
        Delay in seconds.
    """
    cfg = get_human_delay_config()
    mode = cfg.get("mode", "off")
    if mode == "off":
        return 0.0

    min_ms, max_ms = get_delay_range()

    if mode == "fixed":
        return min_ms / 1000.0

    # Natural mode: scale with response size, add jitter
    base_ms = min_ms + (max_ms - min_ms) * 0.3
    if estimated_tokens > 100:
        # Don't wait forever for long responses — cap at a fraction
        base_ms = min(base_ms, max_ms * 0.7)

    jitter = random.uniform(-0.2, 0.2) * (max_ms - min_ms)
    return (base_ms + jitter) / 1000.0


async def delay_async(seconds: float) -> None:
    """Asynchronous sleep for human delay."""
    if seconds > 0:
        await asyncio.sleep(seconds)


def delay_sync(seconds: float) -> None:
    """Synchronous sleep for human delay."""
    if seconds > 0:
        time.sleep(seconds)
