"""Context compression configuration — config-driven compression settings.

Inspired by Hermes Agent's compression config:
  compression:
    enabled: true
    threshold: 0.5        # trigger when context hits 50% of window
    target_ratio: 0.2     # compress to 20% of window
    protect_last_n: 20    # always keep last 20 messages
    protect_first_n: 3     # always keep first 3 messages (system prompt)
    auxiliary_model: ""    # use this model for compression (empty = main model)

This module reads the config and provides accessors. The actual
compression logic lives in aura.memory.context_compressor — this
module makes it configurable.
"""
from __future__ import annotations

import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


# Default values (matching the existing context_compressor.py behavior)
_DEFAULTS = {
    "enabled": True,
    "threshold": 0.5,
    "target_ratio": 0.2,
    "protect_last_n": 20,
    "protect_first_n": 3,
    "auxiliary_model": "",
    "auxiliary_provider": "",
    "hard_message_limit": 400,
    "abort_on_summary_failure": False,
}


def get_compression_config() -> dict:
    """Get the compression config section."""
    try:
        from aura.config_loader import get_config_value
        cfg = get_config_value("compression", {}) or {}
        # Merge with defaults
        return {**_DEFAULTS, **cfg}
    except ImportError:
        return dict(_DEFAULTS)


def is_compression_enabled() -> bool:
    """Check if context compression is enabled."""
    return get_compression_config().get("enabled", True)


def get_compression_threshold() -> float:
    """Get the compression trigger threshold (0-1).

    Compression triggers when context usage exceeds this fraction of
    the model's context window.
    """
    return float(get_compression_config().get("threshold", 0.5))


def get_compression_target_ratio() -> float:
    """Get the target compression ratio (0-1).

    After compression, the context should be at most this fraction
    of the model's context window.
    """
    return float(get_compression_config().get("target_ratio", 0.2))


def get_protect_last_n() -> int:
    """Get the number of recent messages to always keep (never compress)."""
    return int(get_compression_config().get("protect_last_n", 20))


def get_protect_first_n() -> int:
    """Get the number of initial messages to always keep."""
    return int(get_compression_config().get("protect_first_n", 3))


def get_hard_message_limit() -> int:
    """Get the hard message limit — compression is forced at this count."""
    return int(get_compression_config().get("hard_message_limit", 400))


def get_compression_model() -> Tuple[Optional[str], str]:
    """Get the model to use for compression.

    If an auxiliary model is configured, returns (provider, model).
    Otherwise returns (None, "") — caller should use the main model.

    Returns:
        (provider, model) tuple. Provider is None if using main model.
    """
    cfg = get_compression_config()
    aux_model = cfg.get("auxiliary_model", "")
    aux_provider = cfg.get("auxiliary_provider", "")

    if aux_model:
        return aux_provider, aux_model

    # Check the auxiliary.compression role
    try:
        from aura.auxiliary import get_auxiliary_config
        aux = get_auxiliary_config("compression")
        if aux:
            return aux.provider, aux.model
    except ImportError:
        pass

    return None, ""


def should_abort_on_failure() -> bool:
    """Check if compression should abort the conversation on summary failure."""
    return bool(get_compression_config().get("abort_on_summary_failure", False))
