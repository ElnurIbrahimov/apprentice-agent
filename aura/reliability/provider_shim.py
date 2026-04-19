"""Provider retry shim — wraps HTTP calls with structured failover.

Uses the classifier + jittered backoff + rate-limit tracker from Phase 1 so
every provider gets the same retry / compress / rotate semantics without
each adapter having to re-implement it.

Usage:
    from aura.reliability.provider_shim import request_with_retry

    resp = request_with_retry(
        session=self._session,
        method="POST",
        url=url,
        headers=headers,
        json_body=body,
        provider=self._provider_name,
        model=model,
        timeout=(10, 120),
    )

Returns a `requests.Response` on success. Raises `ProviderGiveUp` if all
retries are exhausted, with the final `ClassifiedError` attached for
callers that want to react (e.g. credential rotation at a higher level).
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

import requests

from .error_classifier import ClassifiedError, FailoverReason, classify_api_error
from .rate_limit_tracker import RateLimitState, parse_rate_limit_headers
from .retry_utils import jittered_backoff

logger = logging.getLogger(__name__)

# Max retries per call. Non-retryable errors short-circuit earlier.
_DEFAULT_MAX_RETRIES = 4


# ── Rate-limit state cache ──────────────────────────────────────────────

_rate_limit_cache: Dict[str, RateLimitState] = {}
_rate_limit_lock = threading.Lock()


def record_rate_limit(provider: str, headers: Mapping[str, str]) -> Optional[RateLimitState]:
    """Parse + store rate-limit headers for a provider; returns the parsed state."""
    state = parse_rate_limit_headers(headers, provider=provider)
    if state is None:
        return None
    with _rate_limit_lock:
        _rate_limit_cache[provider.lower()] = state
    return state


def get_last_rate_limit(provider: str) -> Optional[RateLimitState]:
    with _rate_limit_lock:
        return _rate_limit_cache.get(provider.lower())


def all_rate_limit_snapshots() -> Dict[str, RateLimitState]:
    """Snapshot of every known provider's most recent rate-limit state."""
    with _rate_limit_lock:
        return dict(_rate_limit_cache)


# ── Exception ───────────────────────────────────────────────────────────

class ProviderGiveUp(ConnectionError):
    """Raised when all retries are exhausted for a provider call."""

    def __init__(self, classified: ClassifiedError, attempts: int):
        super().__init__(
            f"{classified.provider or 'provider'} failed after {attempts} attempt(s): "
            f"{classified.reason.value} — {classified.message[:200]}"
        )
        self.classified = classified
        self.attempts = attempts


# ── Shim internals ──────────────────────────────────────────────────────

@dataclass
class _FakeStatusError(Exception):
    """Wrap a non-200 response so the classifier can read status_code + body."""
    status_code: int
    body: dict
    message_text: str

    def __str__(self) -> str:
        return self.message_text

    def __init__(self, status_code: int, body: dict, message_text: str):
        super().__init__(message_text)
        self.status_code = status_code
        self.body = body if isinstance(body, dict) else {}
        self.message_text = message_text


def _body_for_classifier(resp: requests.Response) -> dict:
    try:
        data = resp.json()
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


# ── Public API ──────────────────────────────────────────────────────────

def request_with_retry(
    *,
    session: requests.Session,
    method: str,
    url: str,
    headers: Optional[dict] = None,
    json_body: Optional[dict] = None,
    provider: str = "",
    model: str = "",
    timeout: Any = (10, 120),
    stream: bool = False,
    max_retries: int = _DEFAULT_MAX_RETRIES,
    approx_tokens: int = 0,
    context_length: int = 200000,
    num_messages: int = 0,
) -> requests.Response:
    """Perform an HTTP request with classifier-driven retries.

    - Captures rate-limit headers on every response (success or failure)
    - Retries transient errors (timeout, 5xx, 429, 503) with jittered backoff
    - Short-circuits non-retryable errors (400/401/403/404 unless they pattern-match
      a transient variant) and raises ProviderGiveUp with the classification

    The caller still handles the 200-response body — this shim only handles
    transport + HTTP-level failure.
    """
    attempt = 0
    last_classified: Optional[ClassifiedError] = None

    while True:
        attempt += 1
        try:
            resp = session.request(
                method,
                url,
                headers=headers,
                json=json_body,
                timeout=timeout,
                stream=stream,
            )
        except requests.exceptions.RequestException as exc:
            last_classified = classify_api_error(
                exc, provider=provider, model=model,
                approx_tokens=approx_tokens, context_length=context_length,
                num_messages=num_messages,
            )
            if not last_classified.retryable or attempt > max_retries:
                raise ProviderGiveUp(last_classified, attempt) from exc
            _sleep_with_log(attempt, last_classified, provider, model)
            continue

        # Opportunistically capture rate-limit headers on every response.
        try:
            record_rate_limit(provider, resp.headers)
        except Exception:
            pass

        if resp.status_code == 200:
            return resp

        # Non-200 — build a fake error so the classifier can do its job.
        body = _body_for_classifier(resp)
        try:
            text_for_matcher = str(resp.text or "")[:500]
        except Exception:
            text_for_matcher = ""
        fake_exc = _FakeStatusError(
            status_code=resp.status_code,
            body=body,
            message_text=text_for_matcher,
        )
        last_classified = classify_api_error(
            fake_exc, provider=provider, model=model,
            approx_tokens=approx_tokens, context_length=context_length,
            num_messages=num_messages,
        )

        if not last_classified.retryable or attempt > max_retries:
            # Close stream bodies before abandoning.
            try:
                resp.close()
            except Exception:
                pass
            raise ProviderGiveUp(last_classified, attempt)

        try:
            resp.close()
        except Exception:
            pass

        _sleep_with_log(attempt, last_classified, provider, model)


def _sleep_with_log(
    attempt: int,
    classified: ClassifiedError,
    provider: str,
    model: str,
) -> None:
    """Sleep for a jittered backoff delay, bounded by the rate-limit hint if present."""
    base_delay = 5.0
    if classified.reason == FailoverReason.rate_limit:
        # If we know when the window resets, prefer that over the fixed base.
        state = get_last_rate_limit(provider)
        if state is not None and state.has_data:
            reset_hint = min(
                state.requests_min.remaining_seconds_now,
                state.tokens_min.remaining_seconds_now,
            )
            if reset_hint > 0:
                base_delay = min(max(reset_hint, 2.0), 30.0)

    delay = jittered_backoff(attempt, base_delay=base_delay)
    logger.warning(
        "[%s] %s on attempt %d (%s) — backing off %.1fs",
        provider or "provider",
        classified.reason.value,
        attempt,
        model or "?",
        delay,
    )
    time.sleep(delay)
