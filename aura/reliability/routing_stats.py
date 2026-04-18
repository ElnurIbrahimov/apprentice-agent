"""
Outcome-aware Model Routing Stats — Phase 3.

Records per-(microtask_category, model) outcomes and provides
select_model_for_task() which picks the empirically best performer.
Falls back to the existing Config chain when data is insufficient.

Thread-safe. Persists to a small JSONL file so stats survive restarts.

Author: Aura reliability upgrade (2026-03)
"""

from __future__ import annotations

import json
import logging
import math
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from aura.jsonl_utils import rotate_jsonl_if_needed

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Microtask categories — extend as needed
# ---------------------------------------------------------------------------

class MicrotaskCategory:
    MEMORY_SUMMARIZATION = "memory_summarization"
    BROWSER_PLANNING     = "browser_planning"
    BROWSER_RECOVERY     = "browser_recovery"
    OCR_CLEANUP          = "ocr_cleanup"
    WRITING_REWRITE      = "writing_rewrite"
    CODE_EDIT            = "code_edit"
    LONG_DOC_EXTRACTION  = "long_doc_extraction"
    EMOTIONAL_RESPONSE   = "emotional_response"
    TOOL_SELECTION       = "tool_selection"
    KG_MERGE             = "kg_merge"
    GENERAL              = "general"


# ---------------------------------------------------------------------------
# Stats record
# ---------------------------------------------------------------------------

@dataclass
class ModelStats:
    model: str
    category: str
    successes: int = 0
    failures: int = 0
    total_latency_ms: float = 0.0
    last_seen: float = field(default_factory=time.time)

    @property
    def calls(self) -> int:
        return self.successes + self.failures

    @property
    def success_rate(self) -> float:
        return self.successes / self.calls if self.calls else 0.0

    @property
    def avg_latency_ms(self) -> float:
        return self.total_latency_ms / self.calls if self.calls else 0.0

    def score(self) -> float:
        """
        Composite score for model selection.
        Balances success rate (primary) with latency proxy.
        Adds UCB-style exploration bonus for low-data models.
        """
        if self.calls == 0:
            return 0.5   # neutral for unseen models
        sr = self.success_rate
        # Latency penalty: normalise to 0-1 range assuming 30s max
        lat_penalty = min(self.avg_latency_ms / 30_000, 1.0) * 0.1
        # UCB exploration: sqrt(ln(N) / n_i)
        total_calls = max(1, self.calls)
        ucb_bonus = math.sqrt(math.log(max(2, total_calls)) / total_calls) * 0.15
        return min(1.0, sr - lat_penalty + ucb_bonus)

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "category": self.category,
            "successes": self.successes,
            "failures": self.failures,
            "calls": self.calls,
            "success_rate": round(self.success_rate, 3),
            "avg_latency_ms": round(self.avg_latency_ms, 1),
            "score": round(self.score(), 3),
            "last_seen": self.last_seen,
        }


# ---------------------------------------------------------------------------
# Routing stats store
# ---------------------------------------------------------------------------

class RoutingStatsStore:
    """
    In-memory stats store with JSONL persistence.

    Key: (category, model) → ModelStats
    """

    MIN_SAMPLES = 5   # Need this many samples before trusting stats

    def __init__(self, persist_path: Optional[str] = None) -> None:
        self._lock = threading.Lock()
        self._stats: Dict[Tuple[str, str], ModelStats] = {}

        path = persist_path or os.getenv(
            "AURA_ROUTING_STATS_PATH",
            "logs/routing_stats.jsonl"
        )
        self._persist_path = Path(path)
        try:
            self._persist_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError:
            self._persist_path = None  # type: ignore

        self._load()

    # ------------------------------------------------------------------
    # Record outcomes
    # ------------------------------------------------------------------

    def record(
        self,
        category: str,
        model: str,
        success: bool,
        latency_ms: float = 0.0,
    ) -> None:
        key = (category, model)
        with self._lock:
            if key not in self._stats:
                self._stats[key] = ModelStats(model=model, category=category)
            s = self._stats[key]
            if success:
                s.successes += 1
            else:
                s.failures += 1
            s.total_latency_ms += latency_ms
            s.last_seen = time.time()
        self._append(category, model, success, latency_ms)
        self._emit_telemetry(category, model, success, latency_ms)

    # ------------------------------------------------------------------
    # Model selection
    # ------------------------------------------------------------------

    def select_model_for_task(
        self,
        category: str,
        candidates: List[str],
    ) -> str:
        """
        Return the best candidate model for a given microtask category.
        Falls back to candidates[0] if insufficient data.
        """
        with self._lock:
            scored = []
            for model in candidates:
                key = (category, model)
                if key in self._stats and self._stats[key].calls >= self.MIN_SAMPLES:
                    scored.append((self._stats[key].score(), model))

        if not scored:
            logger.debug(
                "[RoutingStats] Insufficient data for %s, using default: %s",
                category, candidates[0] if candidates else "?",
            )
            return candidates[0] if candidates else ""

        best_score, best_model = max(scored)
        logger.debug(
            "[RoutingStats] Selected %s for %s (score=%.3f)",
            best_model, category, best_score,
        )
        return best_model

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> dict:
        with self._lock:
            rows = [s.to_dict() for s in self._stats.values()]
        rows.sort(key=lambda r: (r["category"], -r["score"]))
        return {"total_records": len(rows), "by_category_model": rows}

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if not self._persist_path or not self._persist_path.exists():
            return
        try:
            with open(self._persist_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                        cat   = entry["category"]
                        model = entry["model"]
                        key   = (cat, model)
                        if key not in self._stats:
                            self._stats[key] = ModelStats(model=model, category=cat)
                        s = self._stats[key]
                        if entry.get("success"):
                            s.successes += 1
                        else:
                            s.failures += 1
                        s.total_latency_ms += entry.get("latency_ms", 0.0)
                        s.last_seen = entry.get("ts", s.last_seen)
                    except Exception as e:
                        logger.debug(f"[RoutingStats] Failed to parse stat entry: {e}")
        except Exception as e:
            logger.warning("[RoutingStats] Failed to load persist file: %s", e)

    def _append(
        self, category: str, model: str, success: bool, latency_ms: float
    ) -> None:
        if not self._persist_path:
            return
        try:
            entry = {
                "ts": time.time(),
                "category": category,
                "model": model,
                "success": success,
                "latency_ms": round(latency_ms, 1),
            }
            rotate_jsonl_if_needed(self._persist_path)
            with open(self._persist_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except OSError:
            pass

    def _emit_telemetry(
        self, category: str, model: str, success: bool, latency_ms: float
    ) -> None:
        try:
            from aura.reliability.telemetry import TelemetryKind, emit
            emit(
                TelemetryKind.MODEL_ROUTING,
                success=success,
                latency_ms=latency_ms,
                model_used=model,
                extra={"category": category},
            )
        except Exception as e:
            logger.debug(f"[RoutingStats] Telemetry emission failed: {e}")


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_store_instance: Optional[RoutingStatsStore] = None
_store_lock = threading.Lock()


def get_routing_stats() -> RoutingStatsStore:
    global _store_instance
    if _store_instance is None:
        with _store_lock:
            if _store_instance is None:
                _store_instance = RoutingStatsStore()
    return _store_instance


def select_model_for_task(category: str, role: str = "fast") -> str:
    """
    Convenience: pick best model for a microtask category.

    Args:
        category: MicrotaskCategory constant
        role:     Config role chain to use as candidates ("fast","reason","code",…)
    """
    try:
        from aura.config import Config
        chain_map = {
            "fast":    Config.MODEL_FAST_CHAIN,
            "reason":  Config.MODEL_REASON_CHAIN,
            "code":    Config.MODEL_CODE_CHAIN,
            "vision":  Config.MODEL_VISION_CHAIN,
            "think":   Config.MODEL_THINK_CHAIN,
            "longctx": Config.MODEL_LONGCTX_CHAIN,
        }
        candidates = chain_map.get(role, [Config.get_model(role)])
    except Exception:
        return ""

    return get_routing_stats().select_model_for_task(category, candidates)


__all__ = [
    "MicrotaskCategory",
    "ModelStats",
    "RoutingStatsStore",
    "get_routing_stats",
    "select_model_for_task",
]
