"""Thread-safe model profile store for the neural router."""

from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

DIMENSIONS = ("code", "reason", "speed", "context", "quality", "vision")
NEUTRAL_PROFILE: dict[str, float] = {
    "code": 0.5,
    "reason": 0.5,
    "speed": 0.5,
    "context": 0.5,
    "quality": 0.5,
    "vision": 0.0,
}


class ProfileStore:
    """Load, query, and persist model benchmark profiles."""

    def __init__(self, path: str) -> None:
        self._path = Path(path)
        self._lock = threading.Lock()
        self._profiles: dict[str, dict[str, float]] = {}
        self._load()

    # ── public API ──────────────────────────────────────────────

    def get(self, model: str) -> dict[str, float]:
        """Return profile for *model*, or NEUTRAL_PROFILE for unknowns."""
        with self._lock:
            if model in self._profiles:
                return dict(self._profiles[model])
            return dict(NEUTRAL_PROFILE)

    def update(self, model: str, deltas: dict[str, float]) -> None:
        """Add *deltas* to current values, clamp to [0.0, 1.0], save."""
        with self._lock:
            profile = self._profiles.get(model, dict(NEUTRAL_PROFILE))
            for dim, delta in deltas.items():
                if dim in DIMENSIONS:
                    profile[dim] = max(0.0, min(1.0, profile[dim] + delta))
            self._profiles[model] = profile
            self._save()

    def set(self, model: str, profile: dict[str, float]) -> None:
        """Overwrite entire profile for *model*, save."""
        with self._lock:
            self._profiles[model] = dict(profile)
            self._save()

    def list_available(self, require: dict[str, float] | None = None) -> list[str]:
        """Return model names that meet every minimum in *require*."""
        require = require or {}
        with self._lock:
            results: list[str] = []
            for model, profile in self._profiles.items():
                if all(profile.get(dim, 0.0) >= min_val for dim, min_val in require.items()):
                    results.append(model)
            return results

    def all_profiles(self) -> dict[str, dict[str, float]]:
        """Return a deep copy of every stored profile."""
        with self._lock:
            return {m: dict(p) for m, p in self._profiles.items()}

    # ── internals ───────────────────────────────────────────────

    def _load(self) -> None:
        if self._path.exists():
            try:
                with open(self._path, "r", encoding="utf-8") as f:
                    self._profiles = json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("[ProfileStore] Corrupt profile file, starting fresh: %s", e)
                self._profiles = {}

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        # Atomic write: write to temp file then rename to prevent corruption on crash
        tmp_path = self._path.with_suffix(".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(self._profiles, f, indent=2)
        os.replace(tmp_path, self._path)
