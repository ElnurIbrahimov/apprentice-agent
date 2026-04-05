"""Tests for the _RateLimiter class in aura.messaging.telegram.bot."""
import time
import pytest
from unittest.mock import patch

from aura.messaging.telegram.bot import _RateLimiter


class TestRateLimiter:
    def test_allows_under_limit(self):
        limiter = _RateLimiter()
        for _ in range(19):
            assert limiter.check("user1", max_per_min=20) is True

    def test_allows_exactly_at_limit(self):
        limiter = _RateLimiter()
        for _ in range(20):
            assert limiter.check("user1", max_per_min=20) is True

    def test_blocks_over_limit(self):
        limiter = _RateLimiter()
        for _ in range(20):
            limiter.check("user1", max_per_min=20)
        # 21st request should be blocked
        assert limiter.check("user1", max_per_min=20) is False

    def test_per_user_isolation(self):
        limiter = _RateLimiter()
        # Exhaust user_a's quota
        for _ in range(20):
            limiter.check("user_a", max_per_min=20)
        assert limiter.check("user_a", max_per_min=20) is False
        # user_b is unaffected
        assert limiter.check("user_b", max_per_min=20) is True

    def test_resets_after_window(self):
        limiter = _RateLimiter()
        # Exhaust quota
        for _ in range(20):
            limiter.check("user1", max_per_min=20)
        assert limiter.check("user1", max_per_min=20) is False

        # Advance time by 61 seconds — old timestamps expire
        future = time.time() + 61
        with patch("aura.messaging.telegram.bot._time.time", return_value=future):
            assert limiter.check("user1", max_per_min=20) is True

    def test_cleanup_removes_inactive_users(self):
        limiter = _RateLimiter()
        # Register activity for a user
        limiter.check("user_inactive", max_per_min=20)
        assert "user_inactive" in limiter._timestamps

        # Advance time past the cleanup threshold (3601s) and trigger a new check
        future = time.time() + 3601
        with patch("aura.messaging.telegram.bot._time.time", return_value=future):
            limiter.check("user_trigger", max_per_min=20)

        # Inactive user's entry should have been pruned
        assert "user_inactive" not in limiter._timestamps

    def test_custom_max_per_min(self):
        limiter = _RateLimiter()
        # Allow only 5 per minute
        for _ in range(5):
            assert limiter.check("user1", max_per_min=5) is True
        assert limiter.check("user1", max_per_min=5) is False

    def test_new_user_always_allowed(self):
        limiter = _RateLimiter()
        assert limiter.check("brand_new_user", max_per_min=20) is True
