from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from api.bootstrap import http, routes


def test_load_route_modules_raises_for_critical_import_failure_in_production(monkeypatch):
    monkeypatch.setenv("AURA_ENV", "production")
    logger = MagicMock()
    original_import = __import__

    def fake_import(name, *args, **kwargs):
        if name == "api.routes.chat":
            raise ImportError("boom")
        return original_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=fake_import):
        with pytest.raises(RuntimeError, match="Critical route 'chat' failed to import"):
            routes.load_route_modules(logger)


def test_include_loaded_routers_raises_when_critical_router_missing_in_production(monkeypatch):
    monkeypatch.setenv("AURA_ENV", "production")
    app = MagicMock()
    loaded_modules = {"chat": SimpleNamespace()}

    with pytest.raises(RuntimeError, match="Critical route 'chat' loaded without any of the expected routers"):
        routes.include_loaded_routers(app, loaded_modules)


def test_configure_http_middleware_raises_when_auth_setup_fails_in_production(monkeypatch):
    """The fail-closed wrapper still fires when the middleware itself blows up
    (simulates any non-auth runtime failure during middleware construction).

    Note: the production auth guard runs BEFORE middleware setup, so we give
    it a valid key/flag first — this test is specifically for the inner
    add_middleware failure path, not the guard.
    """
    monkeypatch.setenv("AURA_ENV", "production")
    logger = MagicMock()
    app = MagicMock()

    fake_config = SimpleNamespace(
        API_KEY="a-strong-test-key-that-passes-the-guard",
        API_AUTH_ENABLED=True,
        API_RATE_LIMIT=300,
        API_CORS_ORIGINS="*",
    )

    def fake_add_middleware(middleware, *args, **kwargs):
        if middleware is http.APIKeyAuthMiddleware:
            raise RuntimeError("middleware boom")

    app.add_middleware.side_effect = fake_add_middleware

    with patch.dict("sys.modules", {"aura.config": SimpleNamespace(Config=fake_config)}):
        with pytest.raises(RuntimeError, match="auth/rate-limit middleware failed to initialize"):
            http.configure_http_middleware(app, logger)
