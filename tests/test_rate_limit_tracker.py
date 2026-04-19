"""Tests for aura.reliability.rate_limit_tracker."""
from __future__ import annotations

import time

from aura.reliability.rate_limit_tracker import (
    RateLimitBucket,
    RateLimitState,
    format_rate_limit_compact,
    format_rate_limit_display,
    parse_rate_limit_headers,
)


def test_parse_returns_none_without_headers():
    assert parse_rate_limit_headers({}) is None
    assert parse_rate_limit_headers({"content-type": "application/json"}) is None


def test_parse_requests_min_bucket():
    headers = {
        "x-ratelimit-limit-requests": "100",
        "x-ratelimit-remaining-requests": "42",
        "x-ratelimit-reset-requests": "37",
    }
    state = parse_rate_limit_headers(headers, provider="openrouter")
    assert state is not None
    assert state.requests_min.limit == 100
    assert state.requests_min.remaining == 42
    assert state.requests_min.reset_seconds == 37.0
    assert state.provider == "openrouter"


def test_parse_all_four_buckets():
    headers = {
        "x-ratelimit-limit-requests": "60",
        "x-ratelimit-limit-requests-1h": "3000",
        "x-ratelimit-limit-tokens": "100000",
        "x-ratelimit-limit-tokens-1h": "5000000",
        "x-ratelimit-remaining-requests": "50",
        "x-ratelimit-remaining-requests-1h": "2500",
        "x-ratelimit-remaining-tokens": "80000",
        "x-ratelimit-remaining-tokens-1h": "4500000",
    }
    state = parse_rate_limit_headers(headers)
    assert state.requests_min.limit == 60
    assert state.requests_hour.limit == 3000
    assert state.tokens_min.limit == 100000
    assert state.tokens_hour.limit == 5000000


def test_case_insensitive_header_lookup():
    headers = {
        "X-RateLimit-Limit-Requests": "100",
        "X-RateLimit-Remaining-Requests": "10",
    }
    state = parse_rate_limit_headers(headers)
    assert state is not None
    assert state.requests_min.limit == 100


def test_bucket_usage_pct():
    b = RateLimitBucket(limit=100, remaining=25, captured_at=time.time())
    assert b.used == 75
    assert b.usage_pct == 75.0


def test_bucket_zero_limit_safe():
    b = RateLimitBucket(limit=0, remaining=0, captured_at=time.time())
    assert b.usage_pct == 0.0


def test_remaining_seconds_now_decays():
    b = RateLimitBucket(
        limit=100, remaining=50,
        reset_seconds=10.0,
        captured_at=time.time() - 3.0,
    )
    assert b.remaining_seconds_now <= 7.1
    assert b.remaining_seconds_now >= 6.9


def test_remaining_seconds_now_never_negative():
    b = RateLimitBucket(
        limit=100, remaining=50,
        reset_seconds=5.0,
        captured_at=time.time() - 60.0,
    )
    assert b.remaining_seconds_now == 0.0


def test_format_compact_empty():
    state = RateLimitState()
    assert format_rate_limit_compact(state) == "No rate limit data."


def test_format_compact_with_data():
    headers = {
        "x-ratelimit-limit-requests": "100",
        "x-ratelimit-remaining-requests": "42",
    }
    state = parse_rate_limit_headers(headers)
    out = format_rate_limit_compact(state)
    assert "RPM" in out
    assert "42" in out and "100" in out


def test_format_display_empty():
    state = RateLimitState()
    assert "No rate limit data" in format_rate_limit_display(state)


def test_format_display_warning_at_80pct():
    headers = {
        "x-ratelimit-limit-requests": "100",
        "x-ratelimit-remaining-requests": "15",
        "x-ratelimit-reset-requests": "30",
    }
    state = parse_rate_limit_headers(headers)
    out = format_rate_limit_display(state)
    assert "⚠" in out
