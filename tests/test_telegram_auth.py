"""Tests for TelegramBot._is_user_allowed() logic.

Rather than instantiating the full TelegramBot (which requires a Telegram
token and heavy dependencies), we replicate the method on a minimal stub
that carries only the attributes the method reads:
  - self.allowed_users
  - self._allowed_users_cache
  - self._allowed_cache_time

The method is imported as an unbound function and bound to the stub.
"""
import os
import time
import types
import pytest
from unittest.mock import patch

# Pull the method source from the real class so we test its real logic.
from aura.messaging.telegram.bot import TelegramBot

_is_user_allowed_fn = TelegramBot._is_user_allowed


def _make_stub(allowed_users=None):
    """Return an object that satisfies _is_user_allowed's attribute reads."""
    stub = types.SimpleNamespace(
        allowed_users=allowed_users or [],
        _allowed_users_cache=set(),
        _allowed_cache_time=0.0,
    )
    # Bind the real method to the stub so self.* references work
    stub._is_user_allowed = types.MethodType(_is_user_allowed_fn, stub)
    return stub


class TestIsUserAllowed:
    def test_allowed_user_from_config(self):
        stub = _make_stub(allowed_users=["123"])
        assert stub._is_user_allowed(123) is True

    def test_rejected_user_not_in_list(self):
        stub = _make_stub(allowed_users=["123"])
        assert stub._is_user_allowed(456) is False

    def test_empty_allowed_list_rejects_all(self):
        stub = _make_stub(allowed_users=[])
        assert stub._is_user_allowed(123) is False

    def test_allowed_user_from_env(self):
        stub = _make_stub(allowed_users=[])
        with patch.dict(os.environ, {"TELEGRAM_ALLOWED_USERS": "789"}):
            assert stub._is_user_allowed(789) is True

    def test_env_user_rejected_if_not_listed(self):
        stub = _make_stub(allowed_users=[])
        with patch.dict(os.environ, {"TELEGRAM_ALLOWED_USERS": "789"}):
            assert stub._is_user_allowed(999) is False

    def test_config_and_env_combined(self):
        stub = _make_stub(allowed_users=["100"])
        with patch.dict(os.environ, {"TELEGRAM_ALLOWED_USERS": "200"}):
            assert stub._is_user_allowed(100) is True
            assert stub._is_user_allowed(200) is True
            assert stub._is_user_allowed(300) is False

    def test_cache_is_populated_after_first_call(self):
        stub = _make_stub(allowed_users=["42"])
        stub._is_user_allowed(42)
        assert "42" in stub._allowed_users_cache
        assert stub._allowed_cache_time > 0

    def test_cache_refreshes_after_300s(self):
        stub = _make_stub(allowed_users=["100"])
        # Prime the cache with user 100 only
        stub._is_user_allowed(100)

        # Now simulate that 301 seconds have passed and user 200 was added to env
        future_time = time.time() + 301
        with patch("aura.messaging.telegram.bot._time.time", return_value=future_time):
            with patch.dict(os.environ, {"TELEGRAM_ALLOWED_USERS": "200"}):
                assert stub._is_user_allowed(200) is True

    def test_user_id_as_int_compared_as_string(self):
        # user_id is passed as int, cache stores strings
        stub = _make_stub(allowed_users=["555"])
        assert stub._is_user_allowed(555) is True

    def test_whitespace_in_env_ignored(self):
        stub = _make_stub(allowed_users=[])
        with patch.dict(os.environ, {"TELEGRAM_ALLOWED_USERS": " 321 , 654 "}):
            assert stub._is_user_allowed(321) is True
            assert stub._is_user_allowed(654) is True
