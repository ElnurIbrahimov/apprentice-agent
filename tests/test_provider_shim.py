"""Tests for aura.reliability.provider_shim."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import requests

from aura.reliability.error_classifier import FailoverReason
from aura.reliability.provider_shim import (
    ProviderGiveUp,
    all_rate_limit_snapshots,
    get_last_rate_limit,
    record_rate_limit,
    request_with_retry,
)


def _fake_response(status_code: int, body: dict | None = None, headers: dict | None = None, text: str = ""):
    resp = MagicMock(spec=requests.Response)
    resp.status_code = status_code
    resp.headers = headers or {}
    resp.text = text or ""
    resp.json.return_value = body or {}
    resp.close = MagicMock()
    return resp


def test_200_returns_response_immediately():
    session = MagicMock()
    session.request.return_value = _fake_response(200, body={"ok": True})
    resp = request_with_retry(
        session=session, method="POST", url="http://x/test",
        json_body={"foo": "bar"}, provider="test", max_retries=3,
    )
    assert resp.status_code == 200
    assert session.request.call_count == 1


def test_429_retries_then_raises(monkeypatch):
    """429 should retry up to max_retries then raise ProviderGiveUp."""
    monkeypatch.setattr("time.sleep", lambda *_: None)  # no real sleeps
    session = MagicMock()
    session.request.return_value = _fake_response(429, text="rate limit exceeded")
    with pytest.raises(ProviderGiveUp) as exc_info:
        request_with_retry(
            session=session, method="POST", url="http://x/test",
            json_body={}, provider="test", max_retries=2,
        )
    assert exc_info.value.classified.reason == FailoverReason.rate_limit
    # 1 initial + 2 retries = 3 calls
    assert session.request.call_count == 3


def test_401_does_not_retry(monkeypatch):
    """Auth failures are non-retryable — give up immediately."""
    monkeypatch.setattr("time.sleep", lambda *_: None)
    session = MagicMock()
    session.request.return_value = _fake_response(401, text="unauthorized")
    with pytest.raises(ProviderGiveUp) as exc_info:
        request_with_retry(
            session=session, method="POST", url="http://x/test",
            json_body={}, provider="test", max_retries=5,
        )
    assert exc_info.value.classified.reason == FailoverReason.auth
    assert session.request.call_count == 1  # no retries


def test_500_then_200_succeeds(monkeypatch):
    """5xx is retryable; recover on next attempt."""
    monkeypatch.setattr("time.sleep", lambda *_: None)
    session = MagicMock()
    session.request.side_effect = [
        _fake_response(500, text="internal error"),
        _fake_response(200, body={"ok": True}),
    ]
    resp = request_with_retry(
        session=session, method="POST", url="http://x/test",
        json_body={}, provider="test", max_retries=3,
    )
    assert resp.status_code == 200
    assert session.request.call_count == 2


def test_timeout_then_200_succeeds(monkeypatch):
    """Transport timeouts retry."""
    monkeypatch.setattr("time.sleep", lambda *_: None)
    session = MagicMock()
    session.request.side_effect = [
        requests.exceptions.Timeout("read timeout"),
        _fake_response(200, body={"ok": True}),
    ]
    resp = request_with_retry(
        session=session, method="POST", url="http://x/test",
        json_body={}, provider="test", max_retries=3,
    )
    assert resp.status_code == 200


def test_rate_limit_headers_captured_on_success():
    session = MagicMock()
    session.request.return_value = _fake_response(
        200, body={},
        headers={
            "x-ratelimit-limit-requests": "60",
            "x-ratelimit-remaining-requests": "42",
            "x-ratelimit-reset-requests": "30",
        },
    )
    request_with_retry(
        session=session, method="POST", url="http://x/test",
        json_body={}, provider="test_capture_success",
    )
    state = get_last_rate_limit("test_capture_success")
    assert state is not None
    assert state.requests_min.remaining == 42
    assert state.requests_min.limit == 60


def test_rate_limit_headers_captured_on_429(monkeypatch):
    """Even failed responses carry headers worth capturing."""
    monkeypatch.setattr("time.sleep", lambda *_: None)
    session = MagicMock()
    session.request.return_value = _fake_response(
        429, text="rate limit",
        headers={
            "x-ratelimit-limit-requests": "100",
            "x-ratelimit-remaining-requests": "0",
            "x-ratelimit-reset-requests": "45",
        },
    )
    with pytest.raises(ProviderGiveUp):
        request_with_retry(
            session=session, method="POST", url="http://x/test",
            json_body={}, provider="test_capture_429", max_retries=1,
        )
    state = get_last_rate_limit("test_capture_429")
    assert state is not None
    assert state.requests_min.remaining == 0


def test_record_rate_limit_returns_state():
    state = record_rate_limit("direct_test", {"x-ratelimit-limit-requests": "50"})
    assert state is not None
    assert state.requests_min.limit == 50


def test_snapshot_contains_all_providers():
    record_rate_limit("prov_a", {"x-ratelimit-limit-requests": "10"})
    record_rate_limit("prov_b", {"x-ratelimit-limit-requests": "20"})
    snaps = all_rate_limit_snapshots()
    assert "prov_a" in snaps and "prov_b" in snaps
    assert snaps["prov_a"].requests_min.limit == 10
