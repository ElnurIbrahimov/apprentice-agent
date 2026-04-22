"""Tests for rate-limit indicator in aura.cli.status_bar."""
from __future__ import annotations

import time
from unittest.mock import patch

from aura.cli.status_bar import _tightest_rate_limit, build_status_bar
from aura.reliability.provider_shim import record_rate_limit


def _clear_snapshots():
    from aura.reliability.provider_shim import _rate_limit_cache, _rate_limit_lock
    with _rate_limit_lock:
        _rate_limit_cache.clear()


def test_tightest_returns_none_when_no_data():
    _clear_snapshots()
    assert _tightest_rate_limit() is None


def test_tightest_returns_none_when_all_loose():
    _clear_snapshots()
    # All buckets at <50% usage — not worth showing
    record_rate_limit("loose", {
        "x-ratelimit-limit-requests": "100",
        "x-ratelimit-remaining-requests": "80",  # 20% used
    })
    assert _tightest_rate_limit() is None


def test_tightest_picks_highest_usage_bucket():
    _clear_snapshots()
    record_rate_limit("cool", {
        "x-ratelimit-limit-requests": "100",
        "x-ratelimit-remaining-requests": "95",  # 5% used
    })
    record_rate_limit("hot", {
        "x-ratelimit-limit-requests": "60",
        "x-ratelimit-remaining-requests": "10",  # 83% used → triggers
        "x-ratelimit-limit-tokens": "100000",
        "x-ratelimit-remaining-tokens": "20000",  # 80% used
    })
    result = _tightest_rate_limit()
    assert result is not None
    provider, remaining, limit, label = result
    assert provider == "hot"
    # RPM (83%) should beat TPM (80%)
    assert label == "RPM"
    assert remaining == 10


def test_status_bar_contains_rl_segment_at_120_cols():
    _clear_snapshots()
    record_rate_limit("hot", {
        "x-ratelimit-limit-requests": "100",
        "x-ratelimit-remaining-requests": "20",  # 80% used
    })
    rendered = build_status_bar(
        model="kimi-k2.6:cloud",
        token_used=1000,
        token_limit=200000,
        permission_mode="careful",
        cost_usd=0.05,
        _term_width=140,
    )
    text = rendered.plain
    assert "RPM" in text
    assert "20/100" in text


def test_status_bar_omits_rl_below_120_cols():
    _clear_snapshots()
    record_rate_limit("hot", {
        "x-ratelimit-limit-requests": "100",
        "x-ratelimit-remaining-requests": "20",
    })
    rendered = build_status_bar(
        model="kimi-k2.6:cloud",
        token_used=1000,
        token_limit=200000,
        permission_mode="careful",
        _term_width=100,  # P2 gated behind 120
    )
    text = rendered.plain
    assert "RPM" not in text


def test_status_bar_no_rl_when_snapshots_empty():
    _clear_snapshots()
    rendered = build_status_bar(
        model="kimi-k2.6:cloud",
        token_used=1000,
        token_limit=200000,
        permission_mode="careful",
        _term_width=140,
    )
    text = rendered.plain
    assert "RPM" not in text
    assert "TPM" not in text
