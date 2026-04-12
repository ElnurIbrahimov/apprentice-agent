"""Shared pytest configuration for warning hygiene."""

from __future__ import annotations

import warnings

import pytest

warnings.filterwarnings(
    "ignore",
    message=r"urllib3 \(.*\) or chardet \(.*\)/charset_normalizer \(.*\) doesn't match a supported version",
    category=Warning,
)
warnings.filterwarnings(
    "ignore",
    message=r"builtin type SwigPyPacked has no __module__ attribute",
    category=DeprecationWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"builtin type SwigPyObject has no __module__ attribute",
    category=DeprecationWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"builtin type swigvarlink has no __module__ attribute",
    category=DeprecationWarning,
)


@pytest.fixture(autouse=True, scope="session")
def _shutdown_pools_before_atexit():
    """Shut down shared thread pools before atexit handlers fire.

    This prevents background tasks (brain.think, _quick_generate,
    update_profile_from_memories, DreamConsolidator) from attempting
    LLM calls after pytest teardown, which would produce noisy
    "cannot schedule new futures after shutdown" log lines.
    """
    yield
    try:
        from aura.pools import _shutdown_all
        _shutdown_all()
    except Exception:
        pass
