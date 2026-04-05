"""CustomHandStore — JSON persistence for user-defined DynamicHand configs.

Thread-safe with atomic writes (write to .tmp then os.replace).
Store path: data/custom_hands.json (relative to the aura package root).
"""

import json
import logging
import os
import threading
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Resolve store path relative to this file's package root (aura/)
_PACKAGE_ROOT = Path(__file__).parent.parent  # aura/hands/../  →  aura/
_STORE_PATH = _PACKAGE_ROOT / "data" / "custom_hands.json"


class CustomHandStore:
    """Thread-safe JSON store for custom DynamicHand configurations."""

    def __init__(self, path: Path = _STORE_PATH):
        self._path = path
        self._lock = threading.Lock()
        self._path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_all(self) -> list[dict]:
        """Return all stored custom hand configs as a list."""
        with self._lock:
            return self._read()

    def save(self, config: dict) -> None:
        """Insert or update a hand config (keyed by config['name'])."""
        if not config.get("name"):
            raise ValueError("Hand config must have a 'name' field")
        with self._lock:
            configs = self._read()
            # Replace existing entry with the same name, else append
            updated = [c for c in configs if c.get("name") != config["name"]]
            updated.append(config)
            self._write(updated)
        logger.debug(f"[CustomHandStore] Saved hand: {config['name']}")

    def delete(self, name: str) -> bool:
        """Delete a hand config by name. Returns True if it existed."""
        with self._lock:
            configs = self._read()
            filtered = [c for c in configs if c.get("name") != name]
            existed = len(filtered) < len(configs)
            if existed:
                self._write(filtered)
                logger.debug(f"[CustomHandStore] Deleted hand: {name}")
        return existed

    def get(self, name: str) -> Optional[dict]:
        """Retrieve a single hand config by name, or None if not found."""
        with self._lock:
            configs = self._read()
        for c in configs:
            if c.get("name") == name:
                return c
        return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _read(self) -> list[dict]:
        """Read raw JSON from disk. Must be called with self._lock held."""
        if not self._path.exists():
            return []
        try:
            with open(self._path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, list):
                return data
            logger.warning("[CustomHandStore] Unexpected format in store — resetting")
            return []
        except (json.JSONDecodeError, OSError) as exc:
            logger.error(f"[CustomHandStore] Failed to read store: {exc}")
            return []

    def _write(self, configs: list[dict]) -> None:
        """Atomic write to disk. Must be called with self._lock held."""
        tmp_path = self._path.with_suffix(".json.tmp")
        try:
            with open(tmp_path, "w", encoding="utf-8") as fh:
                json.dump(configs, fh, indent=2, ensure_ascii=False)
            os.replace(tmp_path, self._path)
        except OSError as exc:
            logger.error(f"[CustomHandStore] Failed to write store: {exc}")
            # Clean up temp file if it exists
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass
            raise


# ------------------------------------------------------------------
# Singleton
# ------------------------------------------------------------------

_store_instance: Optional[CustomHandStore] = None
_store_lock = threading.Lock()


def get_custom_hand_store() -> CustomHandStore:
    """Return the process-wide CustomHandStore singleton."""
    global _store_instance
    if _store_instance is None:
        with _store_lock:
            if _store_instance is None:
                _store_instance = CustomHandStore()
    return _store_instance
