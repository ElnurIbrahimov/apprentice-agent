"""
Evaluation Cache — SHA256-keyed memoization for (candidate, example) pairs.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class EvaluationCache:
    """Caches evaluation results to avoid redundant LLM calls."""

    def __init__(self, cache_dir: Optional[str] = None):
        self._memory: Dict[str, float] = {}
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
            return self._memory[key]
        self._misses += 1
        return None

    def put(self, candidate_hash: str, example_hash: str, score: float):
        key = self._make_key(candidate_hash, example_hash)
        self._memory[key] = score

    def save(self):
        if self._cache_path:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._cache_path, 'w') as f:
                json.dump(self._memory, f)

    @property
    def stats(self) -> Dict[str, int]:
        return {
            "entries": len(self._memory),
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / max(self._hits + self._misses, 1)
        }
