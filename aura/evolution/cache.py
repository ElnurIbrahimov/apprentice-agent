"""
Evaluation Cache — SHA256-keyed memoization for (candidate, example) pairs.
"""

import json
import logging
import os
from collections import OrderedDict
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


class EvaluationCache:
    """Caches evaluation results to avoid redundant LLM calls (LRU eviction)."""

    def __init__(self, cache_dir: Optional[str] = None):
        self._memory: OrderedDict[str, float] = OrderedDict()
        self._cache_path = Path(cache_dir) / "eval_cache.json" if cache_dir else None
        self._hits = 0
        self._misses = 0

        if self._cache_path and self._cache_path.exists():
            try:
                with open(self._cache_path, 'r') as f:
                    self._memory = json.load(f)
                logger.info(f"Loaded {len(self._memory)} cached evaluations")
            except Exception:
                self._memory = {}

    def _make_key(self, candidate_hash: str, example_hash: str) -> str:
        return f"{candidate_hash}:{example_hash}"

    def get(self, candidate_hash: str, example_hash: str) -> Optional[float]:
        key = self._make_key(candidate_hash, example_hash)
        if key in self._memory:
            self._hits += 1
            # Move to end (most recently used)
            self._memory.move_to_end(key)
            return self._memory[key]
        self._misses += 1
        return None

    _MAX_ENTRIES = 5000

    def put(self, candidate_hash: str, example_hash: str, score: float):
        key = self._make_key(candidate_hash, example_hash)
        self._memory[key] = score
        self._memory.move_to_end(key)
        # Evict least-recently-used entries when over capacity
        while len(self._memory) > self._MAX_ENTRIES:
            self._memory.popitem(last=False)  # pop oldest

    def save(self):
        if self._cache_path:
            import tempfile
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp_path = tempfile.mkstemp(dir=str(self._cache_path.parent), suffix=".tmp")
            try:
                with os.fdopen(fd, 'w') as f:
                    json.dump(self._memory, f)
                os.replace(tmp_path, str(self._cache_path))
            except Exception:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise

    @property
    def stats(self) -> Dict[str, int]:
        return {
            "entries": len(self._memory),
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / max(self._hits + self._misses, 1)
        }
