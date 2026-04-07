"""Layer 2 – dot-product profile matcher with preference weighting."""

from __future__ import annotations

from typing import Dict, Optional

from aura.routing.profiles import DIMENSIONS

PREFERENCE_WEIGHTS: Dict[str, Dict[str, float]] = {
    "prefer-fast":    {"code": 1.0, "reason": 0.8, "speed": 2.0, "context": 1.0, "quality": 0.7, "vision": 1.0},
    "balanced":       {"code": 1.0, "reason": 1.0, "speed": 1.0, "context": 1.0, "quality": 1.0, "vision": 1.0},
    "prefer-quality": {"code": 1.0, "reason": 1.2, "speed": 0.5, "context": 1.0, "quality": 2.0, "vision": 1.0},
}


def match(
    task_needs: Dict[str, float],
    preference: str,
    profile_store,
    stats_bonus: Optional[Dict[str, float]] = None,
) -> str:
    """Return the best model name for *task_needs* given user *preference*.

    Algorithm:
      1. Weight each dimension by PREFERENCE_WEIGHTS[preference].
      2. Dot-product score every model profile against task_needs.
      3. Add optional stats_bonus per model.
      4. Hard-filter: vision > 0.5 eliminates vision==0 models;
         context > 0.8 eliminates context < 0.5 models.
      5. Return top scorer (or first model as fallback if all filtered).
    """
    weights = PREFERENCE_WEIGHTS.get(preference, PREFERENCE_WEIGHTS["balanced"])
    profiles = profile_store.all_profiles()

    if not profiles:
        raise ValueError("profile_store is empty")

    # ── score every model ───────────────────────────────────────
    scores: Dict[str, float] = {}
    for model, profile in profiles.items():
        score = sum(
            task_needs.get(dim, 0.0) * profile.get(dim, 0.0) * weights.get(dim, 1.0)
            for dim in DIMENSIONS
        )
        if stats_bonus and model in stats_bonus:
            score += stats_bonus[model]
        scores[model] = score

    # ── hard filters ────────────────────────────────────────────
    candidates = dict(scores)

    if task_needs.get("vision", 0.0) > 0.5:
        candidates = {
            m: s for m, s in candidates.items()
            if profiles[m].get("vision", 0) != 0
        }

    if task_needs.get("context", 0.0) > 0.8:
        candidates = {
            m: s for m, s in candidates.items()
            if profiles[m].get("context", 0.0) >= 0.5
        }

    # ── pick winner ─────────────────────────────────────────────
    if not candidates:
        # all filtered out → fallback to first model in store
        return next(iter(profiles))

    return max(candidates, key=candidates.get)  # type: ignore[arg-type]
