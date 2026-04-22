"""SSRF guard on FirecrawlTool.scrape."""
from __future__ import annotations

from unittest.mock import patch

from aura.tools.firecrawl_tool import FirecrawlTool


def _tool() -> FirecrawlTool:
    return FirecrawlTool(api_key="test-key")


def test_scrape_rejects_loopback():
    result = _tool().scrape("http://127.0.0.1/admin")
    assert "error" in result
    assert "SSRF" in result["error"] or "rejected" in result["error"].lower()


def test_scrape_rejects_private_rfc1918():
    result = _tool().scrape("http://10.0.0.5/")
    assert "error" in result


def test_scrape_rejects_link_local_metadata():
    """Cloud metadata endpoint — classic SSRF target."""
    result = _tool().scrape("http://169.254.169.254/latest/meta-data/")
    assert "error" in result


def test_scrape_rejects_file_scheme():
    result = _tool().scrape("file:///etc/passwd")
    assert "error" in result


def test_scrape_rejects_dns_rebinding(monkeypatch):
    """A hostname that resolves to a private IP must be blocked."""
    import aura.security.ssrf_guard as guard
    monkeypatch.setattr(guard, "_resolve_hostname", lambda h, timeout=5.0: ["10.1.2.3"])
    result = _tool().scrape("http://evil.example.com/")
    assert "error" in result


def test_scrape_allows_legit_url(monkeypatch):
    """A safe external URL gets past the guard and hits requests.post."""
    import aura.tools.firecrawl_tool as ft

    captured = {}

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"data": {"markdown": "hello", "metadata": {"title": "t"}}}

    def fake_post(url, headers=None, json=None, timeout=None):
        captured["url"] = url
        captured["json"] = json
        return FakeResponse()

    # Bypass real DNS — pretend evil.example.com resolves to a public IP.
    import aura.security.ssrf_guard as guard
    monkeypatch.setattr(guard, "_resolve_hostname", lambda h, timeout=5.0: ["93.184.216.34"])
    monkeypatch.setattr(ft.requests, "post", fake_post)

    result = _tool().scrape("https://evil.example.com/article")
    assert "error" not in result
    assert result["markdown"] == "hello"
    assert captured["json"]["url"] == "https://evil.example.com/article"
