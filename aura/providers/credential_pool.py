"""Thin multi-key credential pool for API providers.

Inspired by Hermes Agent's credential_pool.py (MIT, Nous Research), but
scoped to the 80/20 pattern Aura needs:

- Comma-separated env vars: `OPENROUTER_API_KEY=key1,key2,key3`
- Round-robin rotation per provider
- Per-key cooldown on 429 (rate_limit) and 402 (billing) for
  `_RATE_COOLDOWN_SECONDS` and `_BILLING_COOLDOWN_SECONDS`
- Thread-safe — one shared pool per process
- No OAuth refresh, no file watchers, no credential source prioritization
  (those live in Phase 3+ if we need them)

Integration: `OpenAICompatProvider._get_api_key()` calls
`pool.acquire(provider)` which returns the best available key. On a
failure, the provider adapter calls `pool.mark_exhausted(provider, key,
reason)` so future acquires skip it until the cooldown elapses.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


_RATE_COOLDOWN_SECONDS = 60.0   # 429 / rate-limit cooldown
_BILLING_COOLDOWN_SECONDS = 3600.0  # 402 / billing exhaustion cooldown


@dataclass
class _KeyEntry:
    key: str
    env_var: str
    cooldown_until: float = 0.0
    failure_count: int = 0
    last_used: float = 0.0

    @property
    def is_available(self) -> bool:
        return time.time() >= self.cooldown_until


@dataclass
class _ProviderPool:
    provider: str
    env_var: str
    keys: List[_KeyEntry] = field(default_factory=list)
    next_index: int = 0

    def load_from_env(self) -> None:
        raw = os.getenv(self.env_var, "").strip()
        if not raw:
            self.keys = []
            return
        # Support comma or semicolon separators
        parts = [p.strip() for p in raw.replace(";", ",").split(",") if p.strip()]
        # Dedupe preserving order
        seen = set()
        new_keys: List[_KeyEntry] = []
        for key in parts:
            if key in seen:
                continue
            seen.add(key)
            # Preserve state for existing keys
            existing = next((k for k in self.keys if k.key == key), None)
            if existing is not None:
                new_keys.append(existing)
            else:
                new_keys.append(_KeyEntry(key=key, env_var=self.env_var))
        self.keys = new_keys
        if self.next_index >= len(self.keys):
            self.next_index = 0


class CredentialPool:
    """Process-wide credential pool."""

    def __init__(self) -> None:
        self._pools: Dict[str, _ProviderPool] = {}
        self._lock = threading.Lock()

    def register(self, provider: str, env_var: str) -> None:
        """Register a provider + its env var. Idempotent."""
        key = provider.lower()
        with self._lock:
            if key not in self._pools:
                self._pools[key] = _ProviderPool(provider=provider, env_var=env_var)
            self._pools[key].load_from_env()

    def acquire(self, provider: str) -> Optional[str]:
        """Return the best-available key for this provider, or None if all cooling.

        Strategy: round-robin, skipping exhausted keys. If all keys are
        cooling, returns the one with the earliest cooldown_until so the
        caller can still try (better to hit a near-expired cooldown than
        fail immediately).
        """
        key = provider.lower()
        with self._lock:
            pool = self._pools.get(key)
            if pool is None:
                return None
            # Re-read env in case the user rotated keys mid-session
            pool.load_from_env()
            if not pool.keys:
                return None

            n = len(pool.keys)
            now = time.time()
            for i in range(n):
                idx = (pool.next_index + i) % n
                entry = pool.keys[idx]
                if entry.is_available:
                    entry.last_used = now
                    pool.next_index = (idx + 1) % n
                    return entry.key

            # All cooling — pick the one with the earliest expiry.
            entry = min(pool.keys, key=lambda e: e.cooldown_until)
            entry.last_used = now
            pool.next_index = (pool.keys.index(entry) + 1) % n
            logger.warning(
                "[CredentialPool] All %s keys cooling — forcing try on key ending %s",
                provider, entry.key[-4:] if len(entry.key) >= 4 else "??",
            )
            return entry.key

    def mark_exhausted(
        self,
        provider: str,
        key: str,
        *,
        reason: str = "rate_limit",
        reset_seconds: Optional[float] = None,
    ) -> None:
        """Mark a key as unavailable for the cooldown period."""
        pkey = provider.lower()
        if reset_seconds is not None and reset_seconds > 0:
            cooldown = float(reset_seconds)
        elif reason == "billing":
            cooldown = _BILLING_COOLDOWN_SECONDS
        else:
            cooldown = _RATE_COOLDOWN_SECONDS

        with self._lock:
            pool = self._pools.get(pkey)
            if pool is None:
                return
            entry = next((e for e in pool.keys if e.key == key), None)
            if entry is None:
                return
            entry.cooldown_until = time.time() + cooldown
            entry.failure_count += 1
            logger.info(
                "[CredentialPool] %s key ...%s cooling %.0fs (%s, total failures: %d)",
                provider,
                key[-4:] if len(key) >= 4 else "??",
                cooldown,
                reason,
                entry.failure_count,
            )

    def status(self, provider: str) -> List[Dict[str, object]]:
        """Debug: list each key's availability for a provider."""
        pkey = provider.lower()
        with self._lock:
            pool = self._pools.get(pkey)
            if pool is None:
                return []
            now = time.time()
            out: List[Dict[str, object]] = []
            for e in pool.keys:
                out.append({
                    "key_suffix": e.key[-4:] if len(e.key) >= 4 else e.key,
                    "available": e.is_available,
                    "cooldown_remaining": max(0.0, e.cooldown_until - now),
                    "failure_count": e.failure_count,
                })
            return out

    def pool_size(self, provider: str) -> int:
        """Number of keys registered for a provider."""
        pkey = provider.lower()
        with self._lock:
            pool = self._pools.get(pkey)
            return len(pool.keys) if pool else 0


_GLOBAL_POOL = CredentialPool()


def get_pool() -> CredentialPool:
    return _GLOBAL_POOL
