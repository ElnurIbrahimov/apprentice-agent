"""Tests for the cookie-based web login module."""

import time

import pytest

from api.auth_session import (
    create_session_token,
    credentials_configured,
    extract_session_username,
    generate_salt,
    generate_secret,
    hash_password,
    verify_credentials,
    verify_password,
    verify_session_token,
)


@pytest.fixture
def configured_env(monkeypatch):
    """Set up valid username + hashed password + session secret."""
    salt = generate_salt()
    pw_hash = hash_password("super-secret-password", salt)
    secret = generate_secret()
    monkeypatch.setenv("AURA_WEB_USERNAME", "alice")
    monkeypatch.setenv("AURA_WEB_PASSWORD_SALT", salt)
    monkeypatch.setenv("AURA_WEB_PASSWORD_HASH", pw_hash)
    monkeypatch.setenv("AURA_SESSION_SECRET", secret)
    return salt, pw_hash, secret


def test_hash_password_is_deterministic_per_salt():
    salt = generate_salt()
    a = hash_password("pw", salt)
    b = hash_password("pw", salt)
    assert a == b


def test_hash_password_changes_with_salt():
    s1, s2 = generate_salt(), generate_salt()
    assert hash_password("pw", s1) != hash_password("pw", s2)


def test_verify_password_accepts_correct():
    salt = generate_salt()
    h = hash_password("correct-horse", salt)
    assert verify_password("correct-horse", salt, h) is True


def test_verify_password_rejects_wrong():
    salt = generate_salt()
    h = hash_password("correct-horse", salt)
    assert verify_password("battery-staple", salt, h) is False


def test_verify_password_handles_empty_fields():
    assert verify_password("pw", "", "") is False
    assert verify_password("", generate_salt(), "") is False


def test_credentials_configured_requires_all_fields(monkeypatch, configured_env):
    assert credentials_configured() is True
    monkeypatch.delenv("AURA_WEB_PASSWORD_HASH")
    assert credentials_configured() is False


def test_verify_credentials_accepts_right_pair(configured_env):
    assert verify_credentials("alice", "super-secret-password") is True


def test_verify_credentials_rejects_wrong_password(configured_env):
    assert verify_credentials("alice", "wrong") is False


def test_verify_credentials_rejects_wrong_username(configured_env):
    assert verify_credentials("bob", "super-secret-password") is False


def test_session_token_roundtrip(configured_env):
    token = create_session_token("alice")
    assert token is not None
    assert verify_session_token(token) == "alice"


def test_session_token_without_secret_returns_none(monkeypatch):
    monkeypatch.delenv("AURA_SESSION_SECRET", raising=False)
    assert create_session_token("alice") is None


def test_session_token_expired_is_rejected(configured_env):
    token = create_session_token("alice", ttl_seconds=-10)
    assert token is not None
    assert verify_session_token(token) is None


def test_session_token_tampered_signature_is_rejected(configured_env):
    token = create_session_token("alice")
    assert token is not None
    # Replace the entire signature with a known-different valid base64url string.
    # Avoid flipping single chars — last-char tampering can land in base64 padding
    # bits and decode to the same bytes, passing the HMAC check by accident.
    import base64
    u, e, _ = token.split(".")
    bogus_sig = base64.urlsafe_b64encode(b"\x00" * 32).rstrip(b"=").decode()
    tampered = f"{u}.{e}.{bogus_sig}"
    assert verify_session_token(tampered) is None


def test_session_token_tampered_username_is_rejected(configured_env):
    token = create_session_token("alice")
    assert token is not None
    # Replace the username segment with a different base64-encoded string
    import base64
    parts = token.split(".")
    parts[0] = base64.urlsafe_b64encode(b"mallory").rstrip(b"=").decode()
    tampered = ".".join(parts)
    assert verify_session_token(tampered) is None


def test_session_token_different_secret_is_rejected(configured_env, monkeypatch):
    token = create_session_token("alice")
    assert token is not None
    monkeypatch.setenv("AURA_SESSION_SECRET", generate_secret())
    assert verify_session_token(token) is None


def test_extract_session_username_reads_cookie_header(configured_env):
    token = create_session_token("alice")

    class FakeHeaders:
        def __init__(self, val):
            self.val = val
        def get(self, key, default=""):
            return self.val if key.lower() == "cookie" else default

    headers = FakeHeaders(f"foo=bar; aura_session={token}; path=/")
    assert extract_session_username(headers) == "alice"


def test_extract_session_username_none_when_cookie_absent():
    class FakeHeaders:
        def get(self, key, default=""):
            return default
    assert extract_session_username(FakeHeaders()) is None


def test_extract_session_username_none_for_invalid_token(configured_env):
    class FakeHeaders:
        def get(self, key, default=""):
            return "aura_session=not-a-real-token" if key.lower() == "cookie" else default
    assert extract_session_username(FakeHeaders()) is None
