"""Response cache — LRU with content hash, JSON persistence, configurable TTL."""
from __future__ import annotations

import hashlib
import json
import logging
import time
from collections import OrderedDict
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_CACHE_DIR = Path.home() / ".aura" / "cache"
_CACHE_FILE = _CACHE_DIR / "response_cache.json"
_MAX_ENTRIES = 500
_DEFAULT_TTL = 3600


class ResponseCache:
    """LRU cache for non-tool LLM responses."""

    def __init__(self, ttl: int = _DEFAULT_TTL, max_entries: int = _MAX_ENTRIES):
        self._cache: OrderedDict[str, dict] = OrderedDict()
        self._ttl = ttl
        self._max = max_entries
        self._hits = 0
        self._misses = 0
        self._load()

    @staticmethod
    def _make_key(prompt: str, system_prompt: str, model: str) -> str:
        raw = f"{model}:{system_prompt[:2000]}:{prompt}"
        return hashlib.sha256(raw.encode()).hexdigest()

    def get(self, prompt: str, system_prompt: str, model: str) -> Optional[str]:
        key = self._make_key(prompt, system_prompt, model)
        entry = self._cache.get(key)
        if entry and (time.time() - entry["ts"]) < self._ttl:
            self._hits += 1
            self._cache.move_to_end(key)
            return entry["response"]
        if entry:
            del self._cache[key]
        self._misses += 1
        return None

    def put(self, prompt: str, system_prompt: str, model: str, response: str):
        key = self._make_key(prompt, system_prompt, model)
        self._cache[key] = {"response": response, "ts": time.time()}
        self._cache.move_to_end(key)
        while len(self._cache) > self._max:
            self._cache.popitem(last=False)

    def _load(self):
        try:
            _CACHE_DIR.mkdir(parents=True, exist_ok=True)
            if _CACHE_FILE.exists():
                data = json.loads(_CACHE_FILE.read_text(encoding="utf-8"))
                self._cache = OrderedDict(data.get("entries", {}))
        except Exception:
            self._cache = OrderedDict()

    def save(self):
        try:
            _CACHE_DIR.mkdir(parents=True, exist_ok=True)
            tmp = _CACHE_FILE.with_suffix(".tmp")
            tmp.write_text(json.dumps({"entries": dict(self._cache)}, indent=1), encoding="utf-8")
            tmp.replace(_CACHE_FILE)
        except Exception:
            logger.debug("response_cache_save_failed", exc_info=True)

    @property
    def stats(self) -> dict:
        total = self._hits + self._misses
        return {"hits": self._hits, "misses": self._misses,
                "hit_rate": self._hits / total if total else 0,
                "entries": len(self._cache)}
