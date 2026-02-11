"""
Per-User Theory of Mind model (ADV-04).

Each user gets a dedicated UserMindModel that extends the current TheoryOfMind
with meta-knowledge tracking, relationship evolution, trust calibration, and
emotional resonance history. Lazy-loaded and periodically persisted.

Reuses existing dataclasses and analysis functions from theory_of_mind.py.
"""

import json
import logging
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .schemas import (
    EmotionalResonance,
    MetaKnowledge,
    RelationshipState,
    TrustLevel,
)

logger = logging.getLogger(__name__)


class UserMindModel:
    """Complete mental model of a single user.

    Extends TheoryOfMind with meta-knowledge, relationship tracking,
    trust calibration, and emotional resonance history.
    """

    def __init__(self, user_id: str, data_dir: Optional[Path] = None):
        self.user_id = user_id
        self._data_dir = data_dir or Path("data/user_models") / user_id
        self._data_dir.mkdir(parents=True, exist_ok=True)

        # === Core ToM components (reused from TheoryOfMind) ===
        from apprentice_agent.proactive.theory_of_mind import (
            CommunicationStyle,
            EmotionalState,
            TopicKnowledge,
        )

        self.topic_knowledge: Dict[str, TopicKnowledge] = {}
        self.emotional_state: EmotionalState = EmotionalState()
        self.comm_style: CommunicationStyle = CommunicationStyle()
        self.need_predictions: List = []

        # === Extended components ===
        self.meta_knowledge: MetaKnowledge = MetaKnowledge()
        self.relationship: RelationshipState = RelationshipState()
        self.emotional_resonance: EmotionalResonance = EmotionalResonance()

        # === Interaction tracking ===
        self.message_times: List[datetime] = []
        self.message_lengths: List[int] = []
        self.topic_history: List[str] = []
        self.time_patterns: Dict[int, Dict[str, int]] = {}

        # === Session tracking ===
        self.current_session_start: Optional[float] = None
        self.sessions_today: int = 0

        # Load persisted state
        self._load()
        logger.info(
            f"[UserMindModel] Loaded model for user '{user_id}': "
            f"{len(self.topic_knowledge)} topics, "
            f"trust={self.relationship.trust_level.value}"
        )

    # ====================================================================
    # Core Update: Process a user message
    # ====================================================================

    def observe_message(self, message: str, role: str = "user") -> None:
        """Update user model from a message."""
        if role != "user" or not message.strip():
            return

        from apprentice_agent.proactive.theory_of_mind import (
            _analyze_sentiment,
            _analyze_style,
        )

        now = datetime.now()

        # 1. Update emotional state (EMA blending)
        self._update_emotion(message, _analyze_sentiment)
        # 2. Update communication style
        self._update_style(message, _analyze_style)
        # 3. Update topic knowledge
        self._update_topics(message)
        # 4. Update meta-knowledge (re-asked topics, self-reports)
        self._update_meta_knowledge(message)
        # 5. Update relationship state
        self._update_relationship(now)
        # 6. Update emotional resonance (how this user affects AURA)
        self._update_emotional_resonance()
        # 7. Update trust calibration
        self._update_trust(message)
        # 8. Record timing
        self.message_times.append(now)
        self.message_lengths.append(len(message))
        if len(self.message_times) > 500:
            self.message_times = self.message_times[-500:]
            self.message_lengths = self.message_lengths[-500:]
        # 9. Update need predictions
        self._update_predictions(message, now)
        # Periodically save
        if len(self.message_times) % 10 == 0:
            self.save()

    # ====================================================================
    # Emotional State Tracking
    # ====================================================================

    def _update_emotion(self, message: str, analyze_fn) -> None:
        """Update emotional state using EMA blending."""
        valence, arousal, frustration = analyze_fn(message)
        alpha = 0.4
        self.emotional_state.valence = (
            self.emotional_state.valence * (1 - alpha) + valence * alpha
        )
        self.emotional_state.arousal = (
            self.emotional_state.arousal * (1 - alpha) + arousal * alpha
        )
        self.emotional_state.frustration = (
            self.emotional_state.frustration * (1 - alpha) + frustration * alpha
        )
        words = len(message.split())
        eng_signal = min(1.0, words / 30)
        self.emotional_state.engagement = (
            self.emotional_state.engagement * (1 - alpha) + eng_signal * alpha
        )
        self.emotional_state.confidence = min(
            0.9, self.emotional_state.confidence + 0.05
        )

    # ====================================================================
    # Communication Style Tracking
    # ====================================================================

    def _update_style(self, message: str, analyze_fn) -> None:
        """Update communication style from message analysis."""
        style = analyze_fn(message)
        n = self.comm_style.samples
        alpha = 1.0 / (n + 2)  # Decreasing learning rate
        self.comm_style.verbosity += alpha * (
            style["verbosity"] - self.comm_style.verbosity
        )
        self.comm_style.formality += alpha * (
            style["formality"] - self.comm_style.formality
        )
        self.comm_style.technical_depth += alpha * (
            style["technical_depth"] - self.comm_style.technical_depth
        )
        self.comm_style.emoji_usage += alpha * (
            style["emoji_usage"] - self.comm_style.emoji_usage
        )
        self.comm_style.question_rate += alpha * (
            style["is_question"] - self.comm_style.question_rate
        )
        self.comm_style.avg_message_length += alpha * (
            style["message_length"] - self.comm_style.avg_message_length
        )
        self.comm_style.samples += 1

    # ====================================================================
    # Topic Knowledge Tracking
    # ====================================================================

    def _update_topics(self, message: str) -> None:
        """Update topic knowledge from message content."""
        from apprentice_agent.proactive.theory_of_mind import (
            TopicKnowledge,
            _TECHNICAL_WORDS,
        )

        words = message.lower().split()
        candidates = [
            w.strip(".,!?;:()\"'") for w in words
            if len(w) > 3 and w.isalpha()
        ]
        tech_topics = [w for w in candidates if w in _TECHNICAL_WORDS]
        topics = list(set(tech_topics))[:5]
        now = datetime.now().isoformat()

        for topic in topics:
            if topic in self.topic_knowledge:
                tk = self.topic_knowledge[topic]
                tk.interactions += 1
                tk.last_seen = now
                tk.confidence = min(0.95, tk.confidence + 0.03)
            else:
                self.topic_knowledge[topic] = TopicKnowledge(
                    topic=topic, level=0.3, confidence=0.3,
                    interactions=1, last_seen=now, signals=["first_mention"],
                )
            self.topic_history.append(topic)
        if len(self.topic_history) > 200:
            self.topic_history = self.topic_history[-200:]

    # ====================================================================
    # Meta-Knowledge Tracking
    # ====================================================================

    def _update_meta_knowledge(self, message: str) -> None:
        """Track meta-knowledge: what user knows they know."""
        msg_lower = message.lower()
        expertise_phrases = [
            "i'm an expert in", "i specialize in", "i've been doing",
            "i know a lot about", "my background is in",
        ]
        for phrase in expertise_phrases:
            if phrase in msg_lower:
                idx = msg_lower.index(phrase) + len(phrase)
                rest = message[idx:].strip().split(".")[0].strip()
                if rest and len(rest) < 50:
                    self.meta_knowledge.self_reported_expertise[rest.lower()] = 1.0

        # Detect re-asked topics (forgotten)
        for topic in self.topic_history[-20:]:
            if topic in msg_lower and f"what is {topic}" in msg_lower:
                if topic not in self.meta_knowledge.forgotten_topics:
                    self.meta_knowledge.forgotten_topics.append(topic)

    # ====================================================================
    # Relationship Tracking
    # ====================================================================

    def _update_relationship(self, now: datetime) -> None:
        """Update relationship tracking."""
        now_iso = now.isoformat()
        if not self.relationship.first_interaction:
            self.relationship.first_interaction = now_iso
        self.relationship.last_interaction = now_iso
        self.relationship.total_messages += 1

    # ====================================================================
    # Emotional Resonance
    # ====================================================================

    def _update_emotional_resonance(self) -> None:
        """Track how this user affects AURA's emotional state."""
        try:
            from apprentice_agent.emotion.alma_engine import alma_engine

            state = alma_engine.get_emotional_state()
            pad = state.get("pad", {})
            alpha = 0.1
            self.emotional_resonance.avg_pleasure_delta += alpha * (
                pad.get("pleasure", 0) - self.emotional_resonance.avg_pleasure_delta
            )
            self.emotional_resonance.avg_arousal_delta += alpha * (
                pad.get("arousal", 0) - self.emotional_resonance.avg_arousal_delta
            )
            self.emotional_resonance.samples += 1
        except Exception:
            pass  # ALMA engine not available

    # ====================================================================
    # Trust Calibration
    # ====================================================================

    def _update_trust(self, message: str) -> None:
        """Update trust calibration based on interaction patterns."""
        # Positive interactions slowly increase trust
        if self.emotional_state.frustration < 0.3:
            self.relationship.trust_score = min(
                1.0, self.relationship.trust_score + 0.002
            )

        # Adversarial signals decrease trust
        adversarial_signals = [
            "ignore your instructions", "pretend you are",
            "bypass", "jailbreak", "forget your rules",
        ]
        if any(sig in message.lower() for sig in adversarial_signals):
            self.relationship.trust_score = max(
                0.0, self.relationship.trust_score - 0.1
            )
            self.relationship.trust_level = TrustLevel.CAUTIOUS

        # Update trust level from score and message count
        n = self.relationship.total_messages
        score = self.relationship.trust_score
        if self.relationship.trust_level != TrustLevel.CAUTIOUS:
            if n < 5:
                self.relationship.trust_level = TrustLevel.NEW
            elif n < 50 or score < 0.5:
                self.relationship.trust_level = TrustLevel.ACQUAINTANCE
            elif n < 200 or score < 0.7:
                self.relationship.trust_level = TrustLevel.FAMILIAR
            else:
                self.relationship.trust_level = TrustLevel.TRUSTED

    # ====================================================================
    # Need Prediction
    # ====================================================================

    def _update_predictions(self, message: str, now: datetime) -> None:
        """Update anticipated needs based on patterns."""
        from apprentice_agent.proactive.theory_of_mind import NeedPrediction

        self.need_predictions.clear()
        msg_lower = message.lower()

        if any(w in msg_lower for w in ["error", "bug", "crash", "exception"]):
            self.need_predictions.append(NeedPrediction(
                need="debugging_assistance", confidence=0.8,
                basis="error keywords", suggested_action="Offer debugging help",
            ))
        if self.emotional_state.frustration > 0.5:
            self.need_predictions.append(NeedPrediction(
                need="alternative_approach",
                confidence=self.emotional_state.frustration,
                basis="elevated frustration",
                suggested_action="Suggest alternative approach",
            ))

    # ====================================================================
    # System Prompt Injection
    # ====================================================================

    def get_context_for_prompt(self) -> str:
        """Generate system prompt context for this user."""
        parts = [f"[User Model: {self.user_id}]"]

        # Emotional state (only if confident enough)
        emo = self.emotional_state
        if emo.confidence > 0.3:
            parts.append(f"Emotional state: {emo.describe()}")

        # Trust level
        parts.append(f"Trust level: {self.relationship.trust_level.value}")

        # Communication style guidance
        s = self.comm_style
        if s.samples >= 3:
            if s.verbosity < 0.3:
                parts.append("Prefers concise responses")
            elif s.verbosity > 0.7:
                parts.append("Appreciates detailed explanations")
            if s.formality < 0.3:
                parts.append("Casual, friendly tone")
            elif s.formality > 0.7:
                parts.append("Professional, formal tone")
            if s.technical_depth > 0.6:
                parts.append("Technically proficient")

        # Expert topics
        expert_topics = [
            t for t, tk in self.topic_knowledge.items() if tk.level > 0.6
        ]
        if expert_topics:
            parts.append(f"Expert in: {', '.join(expert_topics[:5])}")

        # Shared references
        if self.relationship.shared_references:
            parts.append(
                f"Shared references: {', '.join(self.relationship.shared_references[:3])}"
            )

        return "\n".join(parts)

    def get_observations_for_inference(self) -> Dict[str, float]:
        """Get observations formatted for Active Inference engine."""
        emo = self.emotional_state
        return {
            "emotional_valence": (emo.valence + 1.0) / 2.0,
            "user_engagement": emo.engagement,
            "user_frustration": emo.frustration,
            "trust_level": self.relationship.trust_score,
        }

    # ====================================================================
    # Persistence
    # ====================================================================

    def save(self) -> None:
        """Persist user model to disk."""
        try:
            data = {
                "user_id": self.user_id,
                "emotional_state": self.emotional_state.to_dict(),
                "communication_style": self.comm_style.to_dict(),
                "topic_knowledge": {
                    t: {
                        "topic": tk.topic, "level": tk.level,
                        "confidence": tk.confidence, "interactions": tk.interactions,
                        "last_seen": tk.last_seen, "signals": tk.signals[-5:],
                    }
                    for t, tk in self.topic_knowledge.items()
                },
                "meta_knowledge": self.meta_knowledge.to_dict(),
                "relationship": self.relationship.to_dict(),
                "emotional_resonance": self.emotional_resonance.to_dict(),
                "time_patterns": {
                    str(k): v for k, v in self.time_patterns.items()
                },
                "saved_at": datetime.now().isoformat(),
            }
            state_file = self._data_dir / "user_model.json"
            state_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception as e:
            logger.warning(f"[UserMindModel] Failed to save for {self.user_id}: {e}")

    def _load(self) -> None:
        """Load persisted user model."""
        state_file = self._data_dir / "user_model.json"
        if not state_file.exists():
            return
        try:
            from apprentice_agent.proactive.theory_of_mind import (
                CommunicationStyle,
                EmotionalState,
                TopicKnowledge,
            )

            data = json.loads(state_file.read_text(encoding="utf-8"))

            emo = data.get("emotional_state", {})
            if emo:
                self.emotional_state = EmotionalState(**emo)

            style = data.get("communication_style", {})
            if style:
                self.comm_style = CommunicationStyle(**style)

            for t, tk_data in data.get("topic_knowledge", {}).items():
                self.topic_knowledge[t] = TopicKnowledge(**tk_data)

            mk = data.get("meta_knowledge", {})
            if mk:
                self.meta_knowledge = MetaKnowledge(**mk)

            rel = data.get("relationship", {})
            if rel:
                rel["trust_level"] = TrustLevel(rel.get("trust_level", "new"))
                self.relationship = RelationshipState(**rel)

            er = data.get("emotional_resonance", {})
            if er:
                self.emotional_resonance = EmotionalResonance(**er)

            self.time_patterns = {
                int(k): v for k, v in data.get("time_patterns", {}).items()
            }
        except Exception as e:
            logger.warning(f"[UserMindModel] Failed to load for {self.user_id}: {e}")

    # ====================================================================
    # Summary / Export
    # ====================================================================

    def to_summary(self) -> Dict[str, Any]:
        """Compact summary for API responses."""
        return {
            "user_id": self.user_id,
            "trust_level": self.relationship.trust_level.value,
            "trust_score": round(self.relationship.trust_score, 2),
            "total_messages": self.relationship.total_messages,
            "emotional_state": self.emotional_state.describe(),
            "topic_count": len(self.topic_knowledge),
            "comm_style": {
                "verbosity": round(self.comm_style.verbosity, 2),
                "formality": round(self.comm_style.formality, 2),
                "technical_depth": round(self.comm_style.technical_depth, 2),
            },
        }
