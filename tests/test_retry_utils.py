"""Tests for aura.reliability.retry_utils.jittered_backoff."""
from __future__ import annotations

import pytest

from aura.reliability.retry_utils import jittered_backoff


def test_backoff_monotonic_base():
    """Base delay (without jitter factor) doubles each attempt up to cap."""
    # With jitter_ratio=0, jittered_backoff returns pure exponential
    d1 = jittered_backoff(1, base_delay=5.0, max_delay=120.0, jitter_ratio=0.0)
    d2 = jittered_backoff(2, base_delay=5.0, max_delay=120.0, jitter_ratio=0.0)
    d3 = jittered_backoff(3, base_delay=5.0, max_delay=120.0, jitter_ratio=0.0)
    assert d1 == pytest.approx(5.0)
    assert d2 == pytest.approx(10.0)
    assert d3 == pytest.approx(20.0)


def test_backoff_respects_max_delay():
    """Very large attempt numbers should be clamped by max_delay."""
    d = jittered_backoff(30, base_delay=5.0, max_delay=60.0, jitter_ratio=0.0)
    assert d == pytest.approx(60.0)


def test_backoff_jitter_within_range():
    """With jitter_ratio=0.5, jitter is in [0, 0.5 * delay]."""
    for _ in range(100):
        d = jittered_backoff(1, base_delay=5.0, max_delay=120.0, jitter_ratio=0.5)
        assert 5.0 <= d <= 7.5 + 0.001  # 5 + [0, 2.5]


def test_backoff_decorrelated():
    """Successive calls with same attempt should produce different delays (jitter)."""
    samples = [jittered_backoff(2, base_delay=5.0) for _ in range(50)]
    assert len(set(samples)) > 10  # high variance


def test_backoff_extreme_attempt_no_overflow():
    """Attempt >= 63 should not overflow 2**exponent."""
    d = jittered_backoff(100, base_delay=5.0, max_delay=120.0, jitter_ratio=0.0)
    assert d == pytest.approx(120.0)


def test_backoff_zero_base_falls_to_max():
    """base_delay=0 should return max_delay (edge case)."""
    d = jittered_backoff(1, base_delay=0.0, max_delay=60.0, jitter_ratio=0.0)
    assert d == pytest.approx(60.0)
