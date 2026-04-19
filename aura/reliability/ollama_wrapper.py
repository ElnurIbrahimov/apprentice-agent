"""Resilient wrapper around ollama.Client for the hot path.

Aura's brain talks to Ollama Cloud via `ollama.Client(host="https://api.ollama.com", ...)`
— that's where every :cloud model call goes. This wrapper adds the Phase 1
reliability pieces (classifier + jittered backoff + credential pool) to
`chat()` calls without touching any other ollama.Client method.

Design:
- Pass-through proxy: `__getattr__` forwards everything except `chat`
- `chat()` wraps in retry loop using `classify_api_error` + `jittered_backoff`
- Streaming support: retries only apply to the opening of the stream. Once
  we've yielded one chunk, a mid-stream failure can't be retried safely.
- Credential pool: if `provider_label` has multiple keys registered, rotates
  on billing/rate-limit failures and re-creates the underlying client with
  the next key.

Usage in brain.py:
    from aura.reliability.ollama_wrapper import ResilientOllamaClient
    raw = ollama.Client(host=..., headers={"Authorization": f"Bearer {key}"})
    self._cloud_client = ResilientOllamaClient(raw, provider_label="ollama_cloud",
                                               api_key=key, host=..., timeout=...)
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Callable, Iterator, Optional

import ollama

from .error_classifier import FailoverReason, classify_api_error
from .retry_utils import jittered_backoff

logger = logging.getLogger(__name__)


_DEFAULT_MAX_RETRIES = 4


class ResilientOllamaClient:
    """Wraps ollama.Client with retry + classifier-driven failover."""

    def __init__(
        self,
        wrapped: ollama.Client,
        *,
        provider_label: str = "ollama_cloud",
        api_key: Optional[str] = None,
        host: Optional[str] = None,
        timeout: Any = None,
        max_retries: int = _DEFAULT_MAX_RETRIES,
        rebuild_client: Optional[Callable[[str], ollama.Client]] = None,
    ):
        """
        Args:
            wrapped: the underlying ollama.Client
            provider_label: name used for classifier/credential-pool lookups
            api_key: current API key (for credential pool rotation)
            host: host URL (for rebuild on key rotation)
            timeout: httpx.Timeout (for rebuild on key rotation)
            max_retries: attempts per chat() call
            rebuild_client: optional factory `(new_api_key) -> ollama.Client`
                for credential rotation. If None and a key is exhausted,
                we continue with the same client (no rotation).
        """
        self._wrapped = wrapped
        self._provider = provider_label
        self._api_key = api_key
        self._host = host
        self._timeout = timeout
        self._max_retries = max_retries
        self._rebuild_client = rebuild_client
        self._lock = threading.Lock()

    # ── Pass-through for non-chat methods ───────────────────────────────

    def __getattr__(self, name: str) -> Any:
        return getattr(self._wrapped, name)

    # ── Chat with retry + classifier ────────────────────────────────────

    def chat(self, *args, **kwargs):
        """Forward to wrapped.chat with retry on transient errors."""
        stream = kwargs.get("stream", False)
        if stream:
            return self._chat_stream(*args, **kwargs)
        return self._chat_sync(*args, **kwargs)

    def _chat_sync(self, *args, **kwargs):
        attempt = 0
        model = kwargs.get("model", "")
        messages = kwargs.get("messages", []) or []
        num_messages = len(messages)

        while True:
            attempt += 1
            try:
                return self._wrapped.chat(*args, **kwargs)
            except Exception as exc:
                classified = classify_api_error(
                    exc,
                    provider=self._provider,
                    model=model,
                    num_messages=num_messages,
                )
                self._maybe_rotate_credential(classified)

                if not classified.retryable or attempt > self._max_retries:
                    logger.error(
                        "[%s] gave up on %s after %d attempt(s): %s",
                        self._provider, model, attempt, classified.reason.value,
                    )
                    raise
                self._sleep_for_retry(attempt, classified, model)

    def _chat_stream(self, *args, **kwargs) -> Iterator[dict]:
        """Stream with retry on OPENING ONLY. Mid-stream failures propagate."""
        attempt = 0
        model = kwargs.get("model", "")
        messages = kwargs.get("messages", []) or []
        num_messages = len(messages)

        # Retry until we successfully start the stream
        iterator = None
        while True:
            attempt += 1
            try:
                iterator = self._wrapped.chat(*args, **kwargs)
                break
            except Exception as exc:
                classified = classify_api_error(
                    exc,
                    provider=self._provider,
                    model=model,
                    num_messages=num_messages,
                )
                self._maybe_rotate_credential(classified)

                if not classified.retryable or attempt > self._max_retries:
                    logger.error(
                        "[%s] stream gave up on %s after %d attempt(s): %s",
                        self._provider, model, attempt, classified.reason.value,
                    )
                    raise
                self._sleep_for_retry(attempt, classified, model)

        # Stream has opened — forward chunks as-is. No retry possible here.
        try:
            for chunk in iterator:
                yield chunk
        except Exception as exc:
            classified = classify_api_error(
                exc, provider=self._provider, model=model,
            )
            logger.warning(
                "[%s] mid-stream failure on %s (cannot retry): %s",
                self._provider, model, classified.reason.value,
            )
            raise

    # ── Credential rotation ─────────────────────────────────────────────

    def _maybe_rotate_credential(self, classified) -> None:
        """On billing/rate failures, retire current key and rebuild client."""
        if classified.reason not in (
            FailoverReason.rate_limit,
            FailoverReason.billing,
            FailoverReason.auth,
        ):
            return
        if not self._api_key or not self._rebuild_client:
            return

        try:
            from aura.providers.credential_pool import get_pool
            pool = get_pool()
            reason_str = classified.reason.value
            if classified.reason == FailoverReason.auth:
                reason_str = "rate_limit"  # treat 401 as cooldown, not permanent ban
            pool.mark_exhausted(self._provider, self._api_key, reason=reason_str)

            next_key = pool.acquire(self._provider)
            if next_key and next_key != self._api_key:
                logger.info(
                    "[%s] rotating API key ...%s → ...%s",
                    self._provider,
                    self._api_key[-4:] if len(self._api_key) >= 4 else "??",
                    next_key[-4:] if len(next_key) >= 4 else "??",
                )
                with self._lock:
                    self._wrapped = self._rebuild_client(next_key)
                    self._api_key = next_key
        except Exception:
            logger.exception("[%s] credential rotation failed", self._provider)

    # ── Backoff ─────────────────────────────────────────────────────────

    def _sleep_for_retry(self, attempt: int, classified, model: str) -> None:
        base_delay = 5.0
        if classified.reason == FailoverReason.rate_limit:
            base_delay = 10.0  # rate limits usually need a real pause
        elif classified.reason in (FailoverReason.overloaded, FailoverReason.server_error):
            base_delay = 3.0
        elif classified.reason == FailoverReason.timeout:
            base_delay = 2.0

        delay = jittered_backoff(attempt, base_delay=base_delay)
        logger.warning(
            "[%s] %s on %s (attempt %d) — backing off %.1fs",
            self._provider, classified.reason.value, model or "?", attempt, delay,
        )
        time.sleep(delay)
