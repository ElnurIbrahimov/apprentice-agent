"""Tests for aura.reliability.error_classifier."""
from __future__ import annotations

from aura.reliability.error_classifier import (
    ClassifiedError,
    FailoverReason,
    classify_api_error,
)


class _StatusError(Exception):
    """Fake SDK error with status_code attribute."""
    def __init__(self, msg: str, status_code: int = 0, body: dict | None = None):
        super().__init__(msg)
        self.status_code = status_code
        self.body = body or {}


def test_429_classifies_as_rate_limit():
    err = _StatusError("too many requests", status_code=429)
    result = classify_api_error(err)
    assert result.reason == FailoverReason.rate_limit
    assert result.retryable is True
    assert result.should_rotate_credential is True


def test_402_with_try_again_is_rate_limit_not_billing():
    """Key insight from OpenClaw: 402 with 'try again' is transient."""
    err = _StatusError("usage limit, try again in 5 minutes", status_code=402)
    result = classify_api_error(err)
    assert result.reason == FailoverReason.rate_limit


def test_402_without_transient_signal_is_billing():
    err = _StatusError("insufficient credits", status_code=402)
    result = classify_api_error(err)
    assert result.reason == FailoverReason.billing
    assert result.retryable is False


def test_401_is_auth():
    err = _StatusError("unauthorized", status_code=401)
    result = classify_api_error(err)
    assert result.reason == FailoverReason.auth
    assert result.should_rotate_credential is True


def test_503_is_overloaded():
    err = _StatusError("service unavailable", status_code=503)
    result = classify_api_error(err)
    assert result.reason == FailoverReason.overloaded


def test_400_context_overflow_message():
    err = _StatusError("context length exceeded", status_code=400)
    result = classify_api_error(err)
    assert result.reason == FailoverReason.context_overflow
    assert result.should_compress is True


def test_400_large_session_generic_error_is_context_overflow():
    """Anthropic sometimes returns bare 'Error' for context overflow on large sessions."""
    err = _StatusError("Error", status_code=400, body={"error": {"message": "Error"}})
    result = classify_api_error(
        err,
        approx_tokens=100000,
        context_length=200000,
        num_messages=100,
    )
    assert result.reason == FailoverReason.context_overflow


def test_thinking_signature_special_case():
    err = _StatusError(
        "thinking block signature invalid", status_code=400,
    )
    result = classify_api_error(err)
    assert result.reason == FailoverReason.thinking_signature


def test_long_context_tier_special_case():
    err = _StatusError(
        "extra usage required for long context", status_code=429,
    )
    result = classify_api_error(err)
    assert result.reason == FailoverReason.long_context_tier
    assert result.should_compress is True


def test_rate_limit_from_message_no_status_code():
    err = Exception("rate limit exceeded, try again in 5s")
    result = classify_api_error(err)
    assert result.reason == FailoverReason.rate_limit


def test_context_overflow_message_no_status():
    err = Exception("context length exceeded the limit")
    result = classify_api_error(err)
    assert result.reason == FailoverReason.context_overflow


def test_auth_message_no_status():
    err = Exception("invalid api key")
    result = classify_api_error(err)
    assert result.reason == FailoverReason.auth
    assert result.retryable is False


def test_unknown_error_is_retryable():
    err = Exception("some weird error we don't recognize")
    result = classify_api_error(err)
    assert result.reason == FailoverReason.unknown
    assert result.retryable is True


def test_classified_error_is_dataclass():
    err = _StatusError("rate limit", status_code=429)
    result = classify_api_error(err, provider="openrouter", model="kimi-k2.6:cloud")
    assert isinstance(result, ClassifiedError)
    assert result.provider == "openrouter"
    assert result.model == "kimi-k2.6:cloud"
    assert result.status_code == 429


def test_413_payload_too_large_compresses():
    err = _StatusError("payload too large", status_code=413)
    result = classify_api_error(err)
    assert result.reason == FailoverReason.payload_too_large
    assert result.should_compress is True
