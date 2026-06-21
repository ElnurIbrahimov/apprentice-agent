"""Streaming configuration — fine-grained streaming behavior.

Mirrors Hermes Agent's streaming config:
  streaming:
    enabled: true
    transport: auto       # auto | sse | websocket
    edit_interval: 0.8   # seconds between stream updates
    buffer_threshold: 24  # min chars before flushing
    cursor: " ▉"
    fresh_final_after_seconds: 60

Controls how the CLI streams LLM responses to the terminal.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def get_streaming_config() -> dict:
    """Get the streaming config."""
    try:
        from aura.config_loader import get_config_value
        return get_config_value("streaming", {}) or {}
    except ImportError:
        return {}


def is_streaming_enabled() -> bool:
    """Check if streaming is enabled."""
    return get_streaming_config().get("enabled", False)


def get_streaming_transport() -> str:
    """Get the streaming transport: auto | sse | websocket."""
    return get_streaming_config().get("transport", "auto")


def get_edit_interval() -> float:
    """Get the interval between stream updates in seconds."""
    return float(get_streaming_config().get("edit_interval", 0.8))


def get_buffer_threshold() -> int:
    """Get the minimum chars before flushing a stream update."""
    return int(get_streaming_config().get("buffer_threshold", 24))


def get_cursor() -> str:
    """Get the cursor character shown while streaming."""
    return get_streaming_config().get("cursor", " \u2589")


def get_fresh_final_seconds() -> int:
    """Get seconds after which a final response is considered stale."""
    return int(get_streaming_config().get("fresh_final_after_seconds", 60))
