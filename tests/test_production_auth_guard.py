"""Tests for the production auth hardening guard in api/bootstrap/http.py."""

from unittest.mock import MagicMock, patch

import pytest


def _call_configure(monkeypatch, api_key: str, auth_enabled: bool, aura_env: str):
    """Invoke configure_http_middleware with a fake Config."""
    from api.bootstrap import http as http_bootstrap

    monkeypatch.setenv("AURA_ENV", aura_env)
    fake_config = MagicMock()
    fake_config.API_KEY = api_key
    fake_config.API_AUTH_ENABLED = auth_enabled
    fake_config.API_RATE_LIMIT = 300
    fake_config.API_CORS_ORIGINS = "*"

    fake_app = MagicMock()
    fake_logger = MagicMock()

    # Patch config import inside the function body
    with patch.dict("sys.modules", {"aura.config": MagicMock(Config=fake_config)}):
        http_bootstrap.configure_http_middleware(fake_app, fake_logger)


def test_production_refuses_when_auth_disabled(monkeypatch):
    with pytest.raises(RuntimeError, match="AURA_API_AUTH_ENABLED"):
        _call_configure(monkeypatch, api_key="a-very-strong-secret-1234", auth_enabled=False, aura_env="production")


def test_production_refuses_with_empty_key(monkeypatch):
    with pytest.raises(RuntimeError, match="AURA_API_KEY"):
        _call_configure(monkeypatch, api_key="", auth_enabled=True, aura_env="production")


def test_production_refuses_with_weak_key(monkeypatch):
    with pytest.raises(RuntimeError, match="AURA_API_KEY"):
        _call_configure(monkeypatch, api_key="changeme", auth_enabled=True, aura_env="production")


def test_production_refuses_with_short_key(monkeypatch):
    with pytest.raises(RuntimeError, match="AURA_API_KEY"):
        _call_configure(monkeypatch, api_key="short", auth_enabled=True, aura_env="production")


def test_production_accepts_strong_config(monkeypatch):
    # Should NOT raise RuntimeError — middleware setup may still fail for other reasons,
    # but we're verifying the guard itself passes.
    try:
        _call_configure(
            monkeypatch,
            api_key="aB3dEfGhIjKlMnOpQrStUvWx",  # 24 chars, not on weak list
            auth_enabled=True,
            aura_env="production",
        )
    except RuntimeError as e:
        # A RuntimeError about the GUARD specifically is a failure.
        assert "AURA_API_AUTH_ENABLED" not in str(e)
        assert "AURA_API_KEY" not in str(e)


def test_dev_mode_skips_guard(monkeypatch):
    # Dev mode with auth off should not raise the production guard.
    try:
        _call_configure(monkeypatch, api_key="", auth_enabled=False, aura_env="development")
    except RuntimeError as e:
        assert "Production startup aborted" not in str(e)
