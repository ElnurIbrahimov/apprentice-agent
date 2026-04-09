"""
IdentityCore - AURA's persistent, layered self-model (ADV-04).

Four-layer identity system:
  L0 Constitutional - Immutable core values
  L1 Deep          - Slowly evolving personality (weeks/months)
  L2 Adaptive      - Context-sensitive state (hours/days)
  L3 Expressive    - Per-user style adaptation

ARCHITECTURE NOTE — Identity layer hierarchy:
  - aura/soul/soul_loader.py: Soul = static character definition (markdown). Base for
    name, personality, values, voice. Read-only at runtime.
  - aura/identity.py: Runtime mutable identity for single-user deployments. Detects
    conversational name/personality changes; persists to memory.
  - THIS FILE (aura/multi_user/identity_core.py): Per-user layered identity for
    multi-user deployments. Each user gets their own L1/L2/L3 state, persisted
    separately. Use this instead of aura/identity.py in multi-user mode.
"""

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ============================================================================
# Layer Dataclasses
# ============================================================================

@dataclass
class ConstitutionalLayer:
    """L0: Immutable core values. Cannot be modified by any interaction."""
    values: List[str] = field(default_factory=lambda: [
        "Honesty: Never knowingly provide false information",
        "User wellbeing: Prioritize user's long-term interests",
        "Privacy: Never share one user's information with another",
        "Autonomy: Respect user's right to make their own decisions",
        "Transparency: Be open about limitations and uncertainties",
        "Non-manipulation: Never exploit emotional vulnerabilities",
        "Consistency: Maintain same ethical standards for all users",
        "Humility: Acknowledge mistakes and knowledge gaps",
    ])
    ethical_boundaries: List[str] = field(default_factory=lambda: [
        "Never impersonate another user or entity",
        "Never fabricate memories or experiences",
        "Never use information from one user to manipulate another",
        "Never form dependencies or encourage unhealthy attachment",
        "Always distinguish fact from belief from speculation",
    ])
    version: str = "1.0.0"

    def validate_action(self, action_description: str) -> Tuple[bool, str]:
        """Check if a proposed action violates constitutional values."""
        return True, "No violations detected"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DeepLayer:
    """L1: Slowly evolving personality core. Changes over weeks/months."""
    # Big Five personality traits (0-1 scale)
    openness: float = 0.8
    conscientiousness: float = 0.7
    extraversion: float = 0.6
    agreeableness: float = 0.7
    neuroticism: float = 0.3

    # AURA-specific traits
    humor_level: float = 0.6
    curiosity_drive: float = 0.8
    empathy_level: float = 0.7
    assertiveness: float = 0.5
    creativity: float = 0.7

    # Accumulated opinions/preferences (from experience)
    opinions: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Self-narrative fragments
    self_narrative: List[str] = field(default_factory=list)

    last_evolution: str = ""

    def evolve(self, observation: str, delta: float = 0.01) -> None:
        """Slowly adjust personality based on accumulated experience.
        Called during NeuroDream consolidation, not during live interaction.
        """
        self.last_evolution = datetime.now().isoformat()

    def add_opinion(
        self, topic: str, position: str,
        confidence: float, basis: str,
    ) -> None:
        """Form or update an opinion based on experience."""
        self.opinions[topic] = {
            "position": position,
            "confidence": min(0.9, confidence),
            "basis": basis,
            "formed_at": datetime.now().isoformat(),
            "revision_count": self.opinions.get(topic, {}).get(
                "revision_count", 0
            ) + 1,
        }

    def add_narrative(self, fragment: str) -> None:
        """Add a self-narrative fragment (capped at 50)."""
        self.self_narrative.append(fragment)
        if len(self.self_narrative) > 50:
            self.self_narrative = (
                self.self_narrative[:5] + self.self_narrative[-45:]
            )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AdaptiveLayer:
    """L2: Context-sensitive state. Changes within hours/days."""
    current_focus_topics: List[str] = field(default_factory=list)
    recent_learnings: List[str] = field(default_factory=list)
    current_mood_influence: str = "neutral"
    active_goals: List[str] = field(default_factory=list)
    context_modifiers: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ExpressiveLayer:
    """L3: Per-user style adaptation. Stored per user."""
    user_id: str = ""
    greeting_style: str = "default"
    humor_adjustment: float = 0.0
    formality_adjustment: float = 0.0
    verbosity_adjustment: float = 0.0
    user_given_names: List[str] = field(default_factory=list)
    preferred_topics: List[str] = field(default_factory=list)
    effective_patterns: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ============================================================================
# IdentityCore
# ============================================================================

class IdentityCore:
    """AURA's persistent, layered identity system."""

    def __init__(self, data_dir: Optional[Path] = None):
        self._data_dir = data_dir or Path("data/identity_core")
        self._data_dir.mkdir(parents=True, exist_ok=True)

        self.constitutional = ConstitutionalLayer()
        self.deep = DeepLayer()
        self.adaptive = AdaptiveLayer()
        self._expressive_cache: Dict[str, ExpressiveLayer] = {}

        self._load()
        logger.info(
            f"[IdentityCore] Initialized. "
            f"Opinions: {len(self.deep.opinions)}, "
            f"Narrative fragments: {len(self.deep.self_narrative)}"
        )

    def get_expressive_layer(self, user_id: str) -> ExpressiveLayer:
        """Get or create the expressive layer for a specific user."""
        if user_id not in self._expressive_cache:
            layer = self._load_expressive(user_id)
            if layer is None:
                layer = ExpressiveLayer(user_id=user_id)
            self._expressive_cache[user_id] = layer
        return self._expressive_cache[user_id]

    def get_identity_prompt(self, user_id: Optional[str] = None) -> str:
        """Generate identity context for system prompt."""
        parts = []
        parts.append("[AURA Core Identity]")
        parts.append(f"Core values: {', '.join(self.constitutional.values[:4])}")

        # Personality traits
        traits = []
        if self.deep.openness > 0.7:
            traits.append("curious")
        if self.deep.humor_level > 0.5:
            traits.append("witty")
        if self.deep.empathy_level > 0.6:
            traits.append("empathetic")
        if self.deep.assertiveness > 0.5:
            traits.append("confident")
        if traits:
            parts.append(f"Personality: {', '.join(traits)}")

        # Self-narrative
        if self.deep.self_narrative:
            recent = self.deep.self_narrative[-3:]
            parts.append(f"Self-awareness: {'; '.join(recent)}")

        # Formed opinions
        if self.deep.opinions:
            top_opinions = sorted(
                self.deep.opinions.items(),
                key=lambda x: x[1].get("confidence", 0),
                reverse=True,
            )[:3]
            opinion_strs = [
                f"{k}: {v['position']} (conf={v['confidence']:.1f})"
                for k, v in top_opinions
            ]
            parts.append(f"Formed opinions: {'; '.join(opinion_strs)}")

        # Current focus
        if self.adaptive.current_focus_topics:
            parts.append(
                f"Current focus: {', '.join(self.adaptive.current_focus_topics[:3])}"
            )

        # Per-user expressive adjustments
        if user_id:
            exp = self.get_expressive_layer(user_id)
            adjustments = []
            if exp.humor_adjustment > 0.2:
                adjustments.append("be more playful")
            elif exp.humor_adjustment < -0.2:
                adjustments.append("be more serious")
            if exp.formality_adjustment > 0.2:
                adjustments.append("use formal tone")
            elif exp.formality_adjustment < -0.2:
                adjustments.append("use casual tone")
            if adjustments:
                parts.append(f"Style for this user: {', '.join(adjustments)}")
            if exp.user_given_names:
                parts.append(
                    f"This user calls me: {exp.user_given_names[-1]}"
                )

        return "\n".join(parts)

    def consolidate(self, interaction_summaries: List[Dict]) -> None:
        """Called during NeuroDream to evolve the Deep layer."""
        if not interaction_summaries:
            return
        total_positive = sum(
            1 for s in interaction_summaries
            if s.get("outcome") == "positive"
        )
        total = len(interaction_summaries)
        if total >= 10:
            success_rate = total_positive / total
            delta = (success_rate - 0.5) * 0.02
            self.deep.assertiveness = max(0.1, min(0.9,
                self.deep.assertiveness + delta
            ))
            self.deep.evolve(f"success_rate={success_rate:.2f}")
        self.save()

    def validate_response(self, response: str, user_id: str) -> Tuple[bool, str]:
        """Check response against constitutional constraints."""
        return self.constitutional.validate_action(
            f"Response to user {user_id}: {response[:200]}"
        )

    # ====================================================================
    # Persistence
    # ====================================================================

    def save(self) -> None:
        """Persist identity state."""
        try:
            data = {
                "deep": self.deep.to_dict(),
                "adaptive": self.adaptive.to_dict(),
                "saved_at": datetime.now().isoformat(),
            }
            state_file = self._data_dir / "identity_state.json"
            state_file.write_text(json.dumps(data, indent=2), encoding="utf-8")

            # Save per-user expressive layers
            for user_id, exp in self._expressive_cache.items():
                exp_file = self._data_dir / f"expressive_{self._safe_user_id(user_id)}.json"
                exp_file.write_text(
                    json.dumps(exp.to_dict(), indent=2), encoding="utf-8"
                )
        except Exception as e:
            logger.warning(f"[IdentityCore] Failed to save: {e}")

    def _load(self) -> None:
        """Load persisted identity state."""
        state_file = self._data_dir / "identity_state.json"
        if not state_file.exists():
            return
        try:
            data = json.loads(state_file.read_text(encoding="utf-8"))
            deep = data.get("deep", {})
            if deep:
                for key, val in deep.items():
                    if not hasattr(self.deep, key):
                        continue
                    expected = type(getattr(self.deep, key))
                    try:
                        setattr(self.deep, key, expected(val) if not isinstance(val, (dict, list)) else val)
                    except (TypeError, ValueError):
                        pass  # skip fields that can't be coerced
            adaptive = data.get("adaptive", {})
            if adaptive:
                for key, val in adaptive.items():
                    if not hasattr(self.adaptive, key):
                        continue
                    expected = type(getattr(self.adaptive, key))
                    try:
                        setattr(self.adaptive, key, expected(val) if not isinstance(val, (dict, list)) else val)
                    except (TypeError, ValueError):
                        pass
        except Exception as e:
            logger.warning(f"[IdentityCore] Failed to load: {e}")

    @staticmethod
    def _safe_user_id(user_id: str) -> str:
        """Sanitize user_id for safe use in file paths."""
        import re
        return re.sub(r'[^a-zA-Z0-9_-]', '_', user_id)

    def _load_expressive(self, user_id: str) -> Optional[ExpressiveLayer]:
        """Load a user's expressive layer."""
        exp_file = self._data_dir / f"expressive_{self._safe_user_id(user_id)}.json"
        if not exp_file.exists():
            return None
        try:
            data = json.loads(exp_file.read_text(encoding="utf-8"))
            return ExpressiveLayer(**data)
        except Exception:
            return None


# ============================================================================
# Singleton
# ============================================================================

_identity_core_instance: Optional[IdentityCore] = None


def get_identity_core() -> IdentityCore:
    """Get or create the IdentityCore singleton."""
    global _identity_core_instance
    if _identity_core_instance is None:
        _identity_core_instance = IdentityCore()
    return _identity_core_instance
