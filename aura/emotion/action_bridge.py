"""Bridge between ALMA emotional state and autonomous behaviors.

Monitors emotional state and triggers appropriate proactive actions
when emotional signals cross thresholds. This closes the loop between
feeling and doing.

The bridge reads ALMA's actual state (PAD + neuromodulators + active
emotions) and intrinsic drive urgencies, then evaluates rule-based
conditions to produce EmotionActions. The gateway daemon calls
evaluate() each tick and converts triggered actions into proactive
messages via the existing proactive_messages system.
"""
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


@dataclass
class EmotionAction:
    """An action triggered by emotional state."""
    action_type: str   # "check_in", "offer_help", "suggest_break", "explore_topic", "consolidate"
    reason: str        # Why this action was triggered
    priority: float    # 0-1
    cooldown: float    # Seconds before this type can fire again
    context: Dict[str, Any] = field(default_factory=dict)  # Extra data for message generation


class EmotionActionBridge:
    """Monitors ALMA state and generates actions for the proactive system.

    Rules combine neuromodulator levels, PAD values, intrinsic drive
    urgencies, and idle time into concrete action triggers with cooldowns.
    """

    def __init__(self):
        self._last_action_times: Dict[str, float] = {}
        self._action_rules = self._build_rules()

    def _build_rules(self) -> list:
        """Define emotion->action rules.

        Each rule checks a combination of ALMA neuromodulators, PAD
        values, and intrinsic drive urgencies. The flat state dict
        passed to evaluate() is built by _gather_state() in the
        gateway daemon integration.

        Expected keys in state dict:
            dopamine, serotonin, norepinephrine, oxytocin, acetylcholine  (0-1)
            pleasure, arousal, dominance  (-1 to 1)
            curiosity_urgency, social_urgency, competence_urgency, coherence_urgency  (0-1)
            idle_minutes  (float)
            recent_memories  (int, count of recent memory entries)
        """
        return [
            # High curiosity neuromodulators + curiosity drive -> explore
            {
                "name": "curiosity_explore",
                "condition": lambda s: (
                    s.get("dopamine", 0.5) > 0.65
                    and s.get("curiosity_urgency", 0) > 0.4
                    and s.get("arousal", 0) > 0.1
                ),
                "action": EmotionAction(
                    action_type="explore_topic",
                    reason="High dopamine + curiosity drive",
                    priority=0.6,
                    cooldown=1800,  # 30 min
                    context={},
                ),
            },
            # Low serotonin + high norepinephrine + negative pleasure -> suggest break
            {
                "name": "stress_break",
                "condition": lambda s: (
                    s.get("serotonin", 0.5) < 0.35
                    and s.get("norepinephrine", 0.5) > 0.65
                    and s.get("pleasure", 0) < -0.15
                ),
                "action": EmotionAction(
                    action_type="suggest_break",
                    reason="Elevated stress pattern (low serotonin, high norepinephrine)",
                    priority=0.8,
                    cooldown=3600,  # 1 hr
                    context={},
                ),
            },
            # High oxytocin + social drive + idle -> check in
            {
                "name": "social_checkin",
                "condition": lambda s: (
                    s.get("oxytocin", 0.5) > 0.55
                    and s.get("social_urgency", 0) > 0.4
                    and s.get("idle_minutes", 0) > 20
                ),
                "action": EmotionAction(
                    action_type="check_in",
                    reason="Social drive with extended idle",
                    priority=0.4,
                    cooldown=7200,  # 2 hr
                    context={},
                ),
            },
            # Low dopamine + high norepinephrine + competence drive -> offer help
            {
                "name": "frustration_help",
                "condition": lambda s: (
                    s.get("dopamine", 0.5) < 0.35
                    and s.get("norepinephrine", 0.5) > 0.6
                    and s.get("competence_urgency", 0) > 0.35
                ),
                "action": EmotionAction(
                    action_type="offer_help",
                    reason="Frustration pattern (low reward, high alertness)",
                    priority=0.7,
                    cooldown=900,  # 15 min
                    context={},
                ),
            },
            # High acetylcholine + coherence drive -> consolidate
            {
                "name": "consolidation_trigger",
                "condition": lambda s: (
                    s.get("acetylcholine", 0.5) > 0.65
                    and s.get("coherence_urgency", 0) > 0.35
                    and s.get("recent_memories", 0) > 10
                ),
                "action": EmotionAction(
                    action_type="consolidate",
                    reason="High learning activity with coherence need",
                    priority=0.3,
                    cooldown=14400,  # 4 hr
                    context={},
                ),
            },
            # Very positive mood + high arousal -> enthusiastic engagement
            {
                "name": "positive_engagement",
                "condition": lambda s: (
                    s.get("pleasure", 0) > 0.4
                    and s.get("arousal", 0) > 0.3
                    and s.get("dopamine", 0.5) > 0.6
                    and s.get("idle_minutes", 0) > 5
                ),
                "action": EmotionAction(
                    action_type="explore_topic",
                    reason="Positive mood with high energy — good time to explore",
                    priority=0.5,
                    cooldown=2400,  # 40 min
                    context={},
                ),
            },
        ]

    def evaluate(self, state: Dict[str, Any]) -> List[EmotionAction]:
        """Check all rules against current state. Returns triggered actions.

        Args:
            state: Flat dict with neuromodulators, PAD values, drive
                   urgencies, and context metrics. Built by the daemon.

        Returns:
            List of EmotionActions sorted by priority (highest first).
        """
        now = time.monotonic()
        triggered: List[EmotionAction] = []

        for rule in self._action_rules:
            name = rule["name"]
            action = rule["action"]

            # Check cooldown
            last = self._last_action_times.get(name, 0)
            if now - last < action.cooldown:
                continue

            # Check condition
            try:
                if rule["condition"](state):
                    triggered.append(action)
                    self._last_action_times[name] = now
                    logger.info(
                        f"[EMOTION-ACTION] Triggered: {name} — {action.reason}"
                    )
            except Exception as e:
                logger.debug(f"[EMOTION-ACTION] Rule {name} eval error: {e}")

        return sorted(triggered, key=lambda a: a.priority, reverse=True)

    def get_status(self) -> Dict[str, Any]:
        """Return bridge status for API/debug."""
        now = time.monotonic()
        rules_status = {}
        for rule in self._action_rules:
            name = rule["name"]
            last = self._last_action_times.get(name, 0)
            cd = rule["action"].cooldown
            remaining = max(0, cd - (now - last)) if last > 0 else 0
            rules_status[name] = {
                "last_fired_ago": round(now - last, 1) if last > 0 else None,
                "cooldown": cd,
                "cooldown_remaining": round(remaining, 1),
                "action_type": rule["action"].action_type,
                "priority": rule["action"].priority,
            }
        return {
            "active": True,
            "rules": rules_status,
            "total_rules": len(self._action_rules),
        }
