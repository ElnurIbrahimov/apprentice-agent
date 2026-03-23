"""FadeMem — Biologically-inspired memory strength decay.

Implements spaced-repetition-aware exponential decay:
  strength(t) = s0 * exp(-decay_rate * hours_since_last_access)

Default half-life: 2 weeks → decay_rate ≈ 0.00206 / hour
Each access reinforces: +0.05 strength, -10% decay_rate (slower forgetting).
Memories with strength < 0.05 are marked 'forgotten'.

Author: Aura Development Team
Created: 2026-03-16
"""

import logging
import math
from datetime import datetime
from typing import Optional

from .store import MemoryStore, get_memory_store

logger = logging.getLogger(__name__)

# Read from Config if available, else use defaults
try:
    from aura.config import Config
    DEFAULT_HALF_LIFE_HOURS = getattr(Config, "FADEM_HALF_LIFE_HOURS", 14 * 24)
    PRUNE_THRESHOLD = getattr(Config, "FADEM_PRUNE_THRESHOLD", 0.05)
except (ImportError, AttributeError) as e:
    DEFAULT_HALF_LIFE_HOURS = 14 * 24  # 336 hours = 2 weeks
    PRUNE_THRESHOLD = 0.05

DEFAULT_DECAY_RATE = math.log(2) / DEFAULT_HALF_LIFE_HOURS  # ≈ 0.00206

REINFORCE_STRENGTH_DELTA = 0.05
REINFORCE_DECAY_REDUCTION = 0.10  # 10% reduction in decay_rate per access
MIN_DECAY_RATE = 0.0001  # Floor — memories can't become fully permanent


def compute_strength(
    s0: float,
    decay_rate: float,
    hours_since_access: float,
) -> float:
    """Compute current memory strength after time elapsed.

    Args:
        s0: Initial/stored strength (0-1)
        decay_rate: Per-hour decay rate
        hours_since_access: Hours since last access

    Returns:
        Current strength clamped to [0, 1]
    """
    strength = s0 * math.exp(-decay_rate * max(0, hours_since_access))
    return max(0.0, min(1.0, strength))


def reinforce(
    store: MemoryStore,
    memory_id: str,
) -> Optional[float]:
    """Touch a memory: increase strength, reduce decay rate (spaced repetition).

    Uses a single atomic UPDATE to avoid TOCTOU races.
    Returns approximate new strength, or None if memory not found.
    """
    now = datetime.now().isoformat()

    # Atomic single-statement update — no separate read step
    sql = """UPDATE memories
             SET strength = MIN(1.0, strength + ?),
                 decay_rate = MAX(?, decay_rate * ?),
                 last_accessed = ?,
                 access_count = access_count + 1,
                 updated_at = ?
             WHERE id = ?"""
    params = (
        REINFORCE_STRENGTH_DELTA,
        MIN_DECAY_RATE,
        1.0 - REINFORCE_DECAY_REDUCTION,
        now,
        now,
        memory_id,
    )
    with store._lock:
        conn = store._get_conn()
        cur = conn.execute(sql, params)
        conn.commit()
        affected = cur.rowcount
    if affected == 0:
        return None

    # Read back the new strength for the return value
    record = store.get(memory_id)
    return record.strength if record else None


def get_current_strength(
    store: MemoryStore,
    memory_id: str,
) -> Optional[float]:
    """Get the real-time strength of a memory (applying time decay)."""
    record = store.get(memory_id)
    if not record:
        return None

    try:
        la_ts = datetime.fromisoformat(record.last_accessed).timestamp()
    except (ValueError, TypeError):
        la_ts = datetime.now().timestamp()

    hours = max(0, (datetime.now().timestamp() - la_ts) / 3600)
    return compute_strength(record.strength, record.decay_rate, hours)


def batch_decay_and_prune(
    store: Optional[MemoryStore] = None,
    prune_threshold: float = PRUNE_THRESHOLD,
) -> dict:
    """Run batch decay on all memories, then prune forgotten ones.

    Returns dict with decay_count and prune_count.
    """
    if store is None:
        store = get_memory_store()

    decay_count = store.batch_decay()
    prune_count = store.prune_forgotten(threshold=prune_threshold)

    logger.info(
        "[FadeMem] Batch decay: %d memories decayed, %d pruned (threshold=%.3f)",
        decay_count, prune_count, prune_threshold,
    )
    return {"decay_count": decay_count, "prune_count": prune_count}


__all__ = [
    "compute_strength",
    "reinforce",
    "get_current_strength",
    "batch_decay_and_prune",
    "DEFAULT_DECAY_RATE",
    "DEFAULT_HALF_LIFE_HOURS",
    "PRUNE_THRESHOLD",
]
