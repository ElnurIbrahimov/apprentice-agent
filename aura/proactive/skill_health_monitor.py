"""
Skill Health Monitor — Detects underperforming skills and suggests evolution.

Queries the SkillStore index for skills with low success rates and emits
proactive suggestions to evolve them via /evolve --skill-ids.

Designed to be called as an idle task from the IdlePresenceEngine.
"""

import logging
import time
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Rate limiting: max once per 6 hours
_last_check_time: float = 0.0
_CHECK_COOLDOWN_SECONDS = 6 * 3600  # 6 hours


def find_weak_skills(
    min_uses: int = 5,
    max_success_rate: float = 0.6,
) -> List[Dict]:
    """Query SkillStore for underperforming skills.

    Args:
        min_uses: Minimum total_uses to consider (avoids flagging new skills).
        max_success_rate: Skills below this threshold are considered weak.

    Returns:
        List of dicts with id, name, success_rate, total_uses for weak skills.
    """
    try:
        from aura_skill_library.skill_store import SkillStore
        store = SkillStore(storage_path="./aura_data/skill_library")
    except Exception as e:
        logger.debug(f"[SkillHealth] Could not load SkillStore: {e}")
        return []

    if not store.index:
        return []

    weak = []
    for skill_id, info in store.index.items():
        total_uses = info.get("total_uses", 0)
        success_rate = info.get("success_rate", 0.0)

        if total_uses >= min_uses and success_rate < max_success_rate:
            weak.append({
                "id": skill_id,
                "name": info.get("name", skill_id),
                "success_rate": success_rate,
                "total_uses": total_uses,
            })

    # Sort by success rate ascending (worst first)
    weak.sort(key=lambda s: s["success_rate"])
    return weak


def check_and_suggest() -> Optional[str]:
    """Check for weak skills and return a suggestion string, or None.

    Respects the 6-hour cooldown. Returns a single formatted suggestion
    for the worst-performing skills found.
    """
    global _last_check_time

    now = time.time()
    if now - _last_check_time < _CHECK_COOLDOWN_SECONDS:
        return None

    _last_check_time = now

    weak = find_weak_skills()
    if not weak:
        return None

    # Build suggestion for up to 3 worst skills
    top = weak[:3]
    if len(top) == 1:
        s = top[0]
        s["name"]
        skill_ids = s["id"]
        detail = f"{s['name']} has been underperforming ({s['success_rate']:.0%} success rate over {s['total_uses']} uses)"
    else:
        ", ".join(s["name"] for s in top)
        skill_ids = ",".join(s["id"] for s in top)
        details = [f"{s['name']} ({s['success_rate']:.0%})" for s in top]
        detail = f"These skills have been underperforming: {', '.join(details)}"

    return (
        f"I've noticed {detail}. "
        f"Want me to try evolving {'it' if len(top) == 1 else 'them'}? "
        f"Use /evolve --skill-ids {skill_ids}"
    )
