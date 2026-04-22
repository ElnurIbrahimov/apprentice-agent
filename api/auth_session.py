"""Cookie-based session auth for the Aura web UI.

Single-user login: validates username/password against env-configured
credentials, issues an HMAC-signed session cookie. The cookie is accepted
everywhere the X-API-Key header is accepted, so the extension and raw API
clients keep working unchanged.

Env vars (all must be set to enable cookie auth — absence means disabled):
  AURA_WEB_USERNAME           e.g. "Elikos"
  AURA_WEB_PASSWORD_SALT      16+ byte random, hex-encoded
  AURA_WEB_PASSWORD_HASH      PBKDF2-SHA256(password + salt), hex-encoded, 200k iters
  AURA_SESSION_SECRET         32+ byte random, hex-encoded (HMAC key for cookies)

Cookie format:
  base64url(username) "." base64url(expiry_ts) "." base64url(hmac_sha256)
HttpOnly, Secure, SameSite=Lax. 7-day TTL, refreshed on use.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import logging
import os
import secrets
import threading
import time
from typing import Optional

logger = logging.getLogger(__name__)

SESSION_COOKIE_NAME = "aura_session"
SESSION_TTL_SECONDS = 7 * 24 * 3600  # 7 days
_PBKDF2_ITERS = 200_000

# Revocation table: signature hex -> expiry unix ts. In-memory only (single-process
# web server). On multi-process deploys this needs a shared store.
_REVOKED_SIGS: dict[str, int] = {}
_REVOKED_LOCK = threading.Lock()


# ---- Password hashing ------------------------------------------------------

def hash_password(password: str, salt_hex: str) -> str:
    """Return PBKDF2-SHA256 hex digest of password + salt."""
    salt = bytes.fromhex(salt_hex)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, _PBKDF2_ITERS)
    return dk.hex()


def verify_password(password: str, salt_hex: str, stored_hash_hex: str) -> bool:
    """Constant-time check of a submitted password against the stored hash."""
    if not salt_hex or not stored_hash_hex:
        return False
    try:
        computed = hash_password(password, salt_hex)
    except ValueError:
        return False
    return hmac.compare_digest(computed, stored_hash_hex)


# ---- Session token (signed cookie) -----------------------------------------

def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64url_decode(s: str) -> bytes:
    pad = "=" * (-len(s) % 4)
    return base64.urlsafe_b64decode(s + pad)


def _get_session_secret() -> Optional[bytes]:
    raw = os.environ.get("AURA_SESSION_SECRET", "")
    if not raw:
        return None
    try:
        return bytes.fromhex(raw) if all(c in "0123456789abcdefABCDEF" for c in raw) else raw.encode("utf-8")
    except Exception:
        return raw.encode("utf-8")


def create_session_token(username: str, ttl_seconds: int = SESSION_TTL_SECONDS) -> Optional[str]:
    """Build an HMAC-signed session token for `username`. Returns None if
    AURA_SESSION_SECRET is not configured."""
    secret = _get_session_secret()
    if not secret:
        logger.error("[auth_session] AURA_SESSION_SECRET not set — cannot mint tokens")
        return None
    expiry = int(time.time()) + int(ttl_seconds)
    payload = f"{username}|{expiry}".encode("utf-8")
    sig = hmac.new(secret, payload, hashlib.sha256).digest()
    return f"{_b64url(username.encode('utf-8'))}.{_b64url(str(expiry).encode('utf-8'))}.{_b64url(sig)}"


def verify_session_token(token: str) -> Optional[str]:
    """Return the username if the token is valid, unexpired, and not revoked."""
    if not token:
        return None
    secret = _get_session_secret()
    if not secret:
        return None
    try:
        u_part, e_part, s_part = token.split(".")
        username = _b64url_decode(u_part).decode("utf-8")
        expiry = int(_b64url_decode(e_part))
        sig = _b64url_decode(s_part)
    except (ValueError, UnicodeDecodeError):
        return None
    if expiry < int(time.time()):
        return None
    payload = f"{username}|{expiry}".encode("utf-8")
    expected = hmac.new(secret, payload, hashlib.sha256).digest()
    if not hmac.compare_digest(sig, expected):
        return None
    if _is_revoked(s_part):
        return None
    return username


def _prune_revoked(now_ts: int) -> None:
    """Drop revocation entries whose embedded expiry already passed. Called
    under _REVOKED_LOCK."""
    stale = [sig for sig, exp in _REVOKED_SIGS.items() if exp <= now_ts]
    for sig in stale:
        _REVOKED_SIGS.pop(sig, None)


def _is_revoked(sig_b64url: str) -> bool:
    with _REVOKED_LOCK:
        return sig_b64url in _REVOKED_SIGS


def revoke_session_token(token: str) -> bool:
    """Mark a session token as revoked. Idempotent. Returns True if the token
    parsed cleanly and was added to the revocation table (whether or not
    the HMAC is valid — we don't want to leak validity here). Returns False
    only on structurally malformed input."""
    if not token:
        return False
    try:
        _u, e_part, s_part = token.split(".")
        expiry = int(_b64url_decode(e_part))
    except (ValueError, UnicodeDecodeError):
        return False
    now_ts = int(time.time())
    if expiry <= now_ts:
        # Already expired — no need to revoke, but claim success.
        return True
    with _REVOKED_LOCK:
        _prune_revoked(now_ts)
        _REVOKED_SIGS[s_part] = expiry
    return True


# ---- Configured-credentials lookup -----------------------------------------

def get_configured_username() -> str:
    return os.environ.get("AURA_WEB_USERNAME", "")


def credentials_configured() -> bool:
    """True iff every env var needed for cookie auth is present."""
    return bool(
        os.environ.get("AURA_WEB_USERNAME")
        and os.environ.get("AURA_WEB_PASSWORD_SALT")
        and os.environ.get("AURA_WEB_PASSWORD_HASH")
        and os.environ.get("AURA_SESSION_SECRET")
    )


def verify_credentials(username: str, password: str) -> bool:
    """Constant-time username + password check against env config."""
    configured_user = os.environ.get("AURA_WEB_USERNAME", "")
    salt = os.environ.get("AURA_WEB_PASSWORD_SALT", "")
    stored = os.environ.get("AURA_WEB_PASSWORD_HASH", "")
    if not (configured_user and salt and stored):
        return False
    # Compare username constant-time too — don't leak which field was wrong.
    user_ok = hmac.compare_digest(username.encode("utf-8"), configured_user.encode("utf-8"))
    pass_ok = verify_password(password or "", salt, stored)
    return user_ok and pass_ok


# ---- Cookie extraction (shared by HTTP middleware + WS handler) ------------

def extract_session_username(headers) -> Optional[str]:
    """Given a mapping-like `headers` object (Starlette Headers or dict),
    pull the session cookie out of Cookie header and return the verified username."""
    cookie_header = headers.get("cookie") if hasattr(headers, "get") else ""
    if not cookie_header:
        return None
    for part in cookie_header.split(";"):
        name, _, value = part.strip().partition("=")
        if name == SESSION_COOKIE_NAME:
            return verify_session_token(value)
    return None


def generate_salt(nbytes: int = 16) -> str:
    """Helper for CLI / provisioning: return hex salt suitable for env var."""
    return secrets.token_hex(nbytes)


def generate_secret(nbytes: int = 32) -> str:
    """Helper for CLI / provisioning: return hex secret suitable for env var."""
    return secrets.token_hex(nbytes)
