"""Strategy Bandit state endpoint — powers the Mini App "Strategies" live chart.

Exposes the bandit's per-category arm stats (Beta(alpha, beta) posteriors,
total pulls, mean reward) via a simple JSON endpoint. The Mini App polls
this on mount and then subscribes to websocket_hub `bandit_pull` events
for live updates without refetching.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException

from api.auth import require_api_key

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/bandit", tags=["bandit"], dependencies=[Depends(require_api_key)])


@router.get("/state")
async def get_bandit_state() -> Dict[str, Any]:
    """Return current Strategy Bandit state grouped by category.

    Response shape:
    {
        "categories": {
            "<category>": [
                {"strategy": str, "alpha": float, "beta": float,
                 "mean_reward": float, "total_pulls": int,
                 "total_reward": float, "last_updated": str},
                ...
            ], ...
        },
        "summary": {
            "total_arms": int,
            "total_outcomes": int,
            "category_counts": {...},
        },
    }
    """
    try:
        from aura.consciousness.strategy_bandit import get_strategy_bandit
        bandit = get_strategy_bandit()
    except Exception as exc:
        logger.debug("[bandit] init failed: %s", exc)
        raise HTTPException(status_code=503, detail="Strategy bandit unavailable") from None

    try:
        categories = bandit.get_arm_stats()
    except Exception as exc:
        logger.error("[bandit] get_arm_stats failed: %s", exc)
        raise HTTPException(status_code=500, detail="bandit stats error") from None

    try:
        summary = bandit.get_stats_summary()
    except Exception as exc:
        logger.debug("[bandit] get_stats_summary failed: %s", exc)
        summary = {}

    return {"categories": categories or {}, "summary": summary}
