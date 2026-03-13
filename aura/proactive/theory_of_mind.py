"""
Theory of Mind - Dynamic User Mental Model (Phase 6C).

Builds and maintains a model of the user's:
1. Knowledge level per topic (what they know/don't know)
2. Emotional state (predicted from message patterns)
3. Communication preferences (style adaptation)
4. Anticipated needs (proactive prediction)

Integrates with:
- Active Inference: Feeds user state observations
- Brain: Injects user model into system prompts
- Gateway Daemon: Updates from interaction context
- NeuroDream: Consolidates user patterns during sleep
"""

import json
import logging
import math
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from aura.config import Config

logger = logging.getLogger(__name__)

_EMOTIONAL_STATE_FIELDS = frozenset({"valence", "arousal", "engagement", "frustration", "confidence"})
_COMM_STYLE_FIELDS = frozenset({"verbosity", "formality", "technical_depth", "emoji_usage", "question_rate", "avg_message_length", "samples"})
_TOPIC_KNOWLEDGE_FIELDS = frozenset({"topic", "level", "confidence", "interactions", "last_seen", "signals"})


def _phrase_match(text: str, phrases: List[str]) -> bool:
    """Whole-word/phrase match — avoids substring false positives."""
    text_lower = text.lower()
    for phrase in phrases:
        pattern = r'(?<!\w)' + re.escape(phrase.lower()) + r'(?!\w)'
        if re.search(pattern, text_lower):
            return True
    return False


# ============================================================================
# Data Models
# ============================================================================

@dataclass
class TopicKnowledge:
    """User's knowledge level on a specific topic."""
    topic: str
    level: float           # 0=novice, 0.5=intermediate, 1=expert
    confidence: float      # How sure we are (0-1)
    interactions: int      # Number of interactions about this topic
    last_seen: str         # ISO timestamp
    signals: List[str] = field(default_factory=list)  # Evidence

    def decay(self, hours_elapsed: float) -> None:
        """Reduce confidence over time (we become less sure)."""
        half_life = 168.0  # 1 week
        decay = math.exp(-0.693 * hours_elapsed / half_life)
        self.confidence *= decay


@dataclass
class EmotionalState:
    """Predicted user emotional state."""
    valence: float = 0.0       # -1 (negative) to +1 (positive)
    arousal: float = 0.0       # 0 (calm) to 1 (excited/agitated)
    engagement: float = 0.5    # 0 (disengaged) to 1 (highly engaged)
    frustration: float = 0.0   # 0 to 1
    confidence: float = 0.3    # How sure we are

    def to_dict(self) -> Dict[str, float]:
        return {
            "valence": round(self.valence, 3),
            "arousal": round(self.arousal, 3),
            "engagement": round(self.engagement, 3),
            "frustration": round(self.frustration, 3),
            "confidence": round(self.confidence, 3),
        }

    def describe(self) -> str:
        """Human-readable description."""
        if self.frustration > 0.6:
            mood = "frustrated"
        elif self.valence > 0.3:
            mood = "positive" if self.arousal < 0.5 else "enthusiastic"
        elif self.valence < -0.3:
            mood = "down" if self.arousal < 0.5 else "stressed"
        else:
            mood = "neutral" if self.arousal < 0.5 else "focused"

        eng = "engaged" if self.engagement > 0.6 else "casual"
        return f"{mood}, {eng}"


@dataclass
class CommunicationStyle:
    """User's preferred communication style."""
    verbosity: float = 0.5         # 0=terse, 1=verbose
    formality: float = 0.5         # 0=casual, 1=formal
    technical_depth: float = 0.5   # 0=simple, 1=technical
    emoji_usage: float = 0.0       # 0=none, 1=frequent
    question_rate: float = 0.0     # How often user asks questions
    avg_message_length: float = 50.0  # Average chars per message
    samples: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "verbosity": round(self.verbosity, 2),
            "formality": round(self.formality, 2),
            "technical_depth": round(self.technical_depth, 2),
            "emoji_usage": round(self.emoji_usage, 2),
            "question_rate": round(self.question_rate, 2),
            "avg_message_length": round(self.avg_message_length, 1),
            "samples": self.samples,
        }


@dataclass
class NeedPrediction:
    """Predicted user need."""
    need: str
    confidence: float       # 0-1
    basis: str              # What evidence this is based on
    suggested_action: str   # What AURA should do


# ============================================================================
# Sentiment & Style Analysis (lightweight, no ML required)
# ============================================================================

# Emotion lexicon (word -> valence)
_POSITIVE_WORDS = {
    "thanks", "thank", "great", "awesome", "perfect", "love", "excellent",
    "amazing", "wonderful", "nice", "good", "helpful", "appreciate", "cool",
    "brilliant", "fantastic", "yes", "yeah", "yep", "exactly", "right",
    "happy", "glad", "pleased",
}
_NEGATIVE_WORDS = {
    "wrong", "bad", "terrible", "awful", "hate", "stupid", "useless",
    "broken", "error", "fail", "failed", "bug", "crash", "frustrated",
    "annoyed", "confused", "stuck", "lost", "help", "ugh", "damn",
    "no", "nope", "incorrect", "worse",
}
_FRUSTRATION_SIGNALS = {
    "why", "still", "again", "doesn't work", "not working", "broken",
    "wrong again", "already told you", "i said", "!!!",
}
_TECHNICAL_WORDS = {
    "api", "function", "class", "module", "database", "query", "async",
    "docker", "git", "deploy", "kubernetes", "sql", "json", "http",
    "algorithm", "binary", "cache", "regex", "lambda", "typescript",
    "python", "javascript", "rust", "interface", "endpoint",
}
_FORMAL_MARKERS = {
    "please", "kindly", "could you", "would you", "i would like",
    "appreciate", "regarding", "furthermore", "therefore",
}
_CASUAL_MARKERS = {
    "hey", "yo", "lol", "haha", "btw", "gonna", "wanna", "kinda",
    "nah", "yep", "nope", "cool", "dude", "bruh",
}


def _analyze_sentiment(text: str) -> Tuple[float, float, float]:
    """Lightweight sentiment analysis. Returns (valence, arousal, frustration)."""
    words = {w.strip(".,!?;:()\"'") for w in text.lower().split()}
    text_lower = text.lower()

    pos = len(words & _POSITIVE_WORDS)
    neg = len(words & _NEGATIVE_WORDS)
    total = pos + neg

    valence = 0.0
    if total > 0:
        valence = (pos - neg) / total

    # Arousal from punctuation and caps
    excl = text.count("!")
    quest = text.count("?")
    caps_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
    arousal = min(1.0, (excl * 0.15 + quest * 0.1 + caps_ratio * 2.0))

    # Frustration
    frust_count = sum(1 for sig in _FRUSTRATION_SIGNALS if sig in text_lower)
    frustration = min(1.0, frust_count * 0.25 + (neg * 0.15))

    return valence, arousal, frustration


def _analyze_style(text: str) -> Dict[str, float]:
    """Analyze communication style from a message."""
    words = text.lower().split()
    word_set = set(words)

    technical = len(word_set & _TECHNICAL_WORDS) / max(len(words), 1)
    formal_count = sum(1 for m in _FORMAL_MARKERS if m in text.lower())
    casual_count = sum(1 for m in _CASUAL_MARKERS if m in text.lower())

    formality = 0.5
    if formal_count + casual_count > 0:
        formality = formal_count / (formal_count + casual_count)

    emoji_count = len(re.findall(
        r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF]',
        text
    ))
    has_question = "?" in text

    return {
        "technical_depth": min(1.0, technical * 10),
        "formality": formality,
        "emoji_usage": min(1.0, emoji_count * 0.3),
        "is_question": 1.0 if has_question else 0.0,
        "message_length": len(text),
        "verbosity": min(1.0, len(words) / 100),
    }


# ============================================================================
# Theory of Mind Engine
# ============================================================================

class TheoryOfMind:
    """Dynamic mental model of the user.

    Maintains beliefs about the user's knowledge, emotional state,
    communication preferences, and anticipated needs. Updates from
    every interaction to build an increasingly accurate user model.
    """

    def __init__(self, data_dir: Optional[str] = None):
        if data_dir is None:
            base = Path(__file__).resolve().parent.parent.parent
            data_dir = str(base / "data" / "theory_of_mind")

        self._data_dir = Path(data_dir)
        self._data_dir.mkdir(parents=True, exist_ok=True)

        # User model components
        self._topic_knowledge: Dict[str, TopicKnowledge] = {}
        self._emotional_state = EmotionalState()
        self._comm_style = CommunicationStyle()
        self._need_predictions: List[NeedPrediction] = []

        # Interaction history (for pattern detection)
        self._message_times: List[datetime] = []
        self._message_lengths: List[int] = []
        self._topic_history: List[str] = []

        # Time-of-day patterns: hour -> typical topics/activity
        self._time_patterns: Dict[int, Dict[str, int]] = {}

        # Load persisted state
        self._load_state()

        logger.info(
            f"[ToM] Initialized with {len(self._topic_knowledge)} topics, "
            f"style samples={self._comm_style.samples}"
        )

    # ====================================================================
    # Core Update: Process a user message
    # ====================================================================

    def observe_message(self, message: str, role: str = "user") -> None:
        """Update user model from a message.

        Args:
            message: The message content
            role: "user" or "assistant"
        """
        if role != "user" or not message.strip():
            return

        now = datetime.now()

        # 1. Update emotional state
        self._update_emotion(message)

        # 2. Update communication style
        self._update_style(message)

        # 3. Extract and update topic knowledge
        self._update_topics(message)

        # 4. Record timing
        self._message_times.append(now)
        self._message_lengths.append(len(message))
        if len(self._message_times) > 200:
            self._message_times = self._message_times[-200:]
            self._message_lengths = self._message_lengths[-200:]

        # 5. Update time-of-day patterns
        hour = now.hour
        if hour not in self._time_patterns:
            self._time_patterns[hour] = {}
        topics = self._extract_topics(message)
        for t in topics:
            self._time_patterns[hour][t] = self._time_patterns[hour].get(t, 0) + 1

        # 6. Update need predictions
        self._update_predictions(message, now)

        # Periodically save
        if len(self._message_times) % 10 == 0:
            self._save_state()

    # ====================================================================
    # Emotional State Tracking
    # ====================================================================

    def _update_emotion(self, message: str) -> None:
        """Update emotional state estimate from message."""
        valence, arousal, frustration = _analyze_sentiment(message)

        # Blend with existing state (exponential moving average)
        alpha = Config.TOM_EMA_ALPHA
        self._emotional_state.valence = (
            self._emotional_state.valence * (1 - alpha) + valence * alpha
        )
        self._emotional_state.arousal = (
            self._emotional_state.arousal * (1 - alpha) + arousal * alpha
        )
        self._emotional_state.frustration = (
            self._emotional_state.frustration * (1 - alpha) + frustration * alpha
        )

        # Engagement from message length and frequency
        words = len(message.split())
        eng_signal = min(1.0, words / 30)  # Long messages = engaged
        self._emotional_state.engagement = (
            self._emotional_state.engagement * (1 - alpha) + eng_signal * alpha
        )

        # Confidence increases with more data
        self._emotional_state.confidence = min(
            0.9, self._emotional_state.confidence + 0.05
        )

    def get_emotional_state(self) -> EmotionalState:
        """Get current predicted emotional state."""
        return self._emotional_state

    # ====================================================================
    # Communication Style Tracking
    # ====================================================================

    def _update_style(self, message: str) -> None:
        """Update communication style model from message."""
        style = _analyze_style(message)
        n = self._comm_style.samples
        alpha = 1.0 / (n + 2)  # Decreasing learning rate

        self._comm_style.verbosity += alpha * (style["verbosity"] - self._comm_style.verbosity)
        self._comm_style.formality += alpha * (style["formality"] - self._comm_style.formality)
        self._comm_style.technical_depth += alpha * (
            style["technical_depth"] - self._comm_style.technical_depth
        )
        self._comm_style.emoji_usage += alpha * (
            style["emoji_usage"] - self._comm_style.emoji_usage
        )
        self._comm_style.question_rate += alpha * (
            style["is_question"] - self._comm_style.question_rate
        )
        self._comm_style.avg_message_length += alpha * (
            style["message_length"] - self._comm_style.avg_message_length
        )
        self._comm_style.samples += 1

    def get_communication_style(self) -> CommunicationStyle:
        """Get user's communication style profile."""
        return self._comm_style

    def get_style_guidance(self) -> str:
        """Get style adaptation guidance for the system prompt."""
        s = self._comm_style
        if s.samples < 3:
            return ""

        parts = []

        # Verbosity
        if s.verbosity < 0.3:
            parts.append("Keep responses concise — user prefers brief answers")
        elif s.verbosity > 0.7:
            parts.append("User appreciates detailed explanations")

        # Formality
        if s.formality < 0.3:
            parts.append("Use casual, friendly tone")
        elif s.formality > 0.7:
            parts.append("Use professional, formal tone")

        # Technical depth
        if s.technical_depth > 0.6:
            parts.append("User is technically proficient — use precise terminology")
        elif s.technical_depth < 0.2:
            parts.append("Explain technical concepts in simple terms")

        if not parts:
            return ""

        return "[User Communication Style]\n" + "\n".join(f"- {p}" for p in parts)

    # ====================================================================
    # Topic Knowledge Tracking
    # ====================================================================

    def _extract_topics(self, message: str) -> List[str]:
        """Extract topic keywords from a message."""
        words = message.lower().split()
        # Filter for meaningful words (> 3 chars, alphabetic)
        candidates = [w.strip(".,!?;:()\"'") for w in words if len(w) > 3 and w.isalpha()]

        # Use technical words as topics
        tech_topics = [w for w in candidates if w in _TECHNICAL_WORDS]

        # Also extract noun-like words (capitalized in original, or long)
        orig_words = message.split()
        proper = [
            w.strip(".,!?;:()\"'").lower()
            for w in orig_words
            if len(w) > 4 and w[0].isupper() and not w.isupper()
        ]

        topics = list(set(tech_topics + proper))
        return topics[:5]  # Limit

    def _update_topics(self, message: str) -> None:
        """Update topic knowledge from message content."""
        topics = self._extract_topics(message)
        now = datetime.now().isoformat()
        message_lower = message.lower()

        for topic in topics:
            if topic in self._topic_knowledge:
                tk = self._topic_knowledge[topic]
                tk.interactions += 1
                tk.last_seen = now

                # Adjust knowledge level based on message signals
                if self._indicates_expertise(message_lower, topic):
                    tk.level = min(1.0, tk.level + 0.05)
                    tk.signals.append("expertise_signal")
                elif self._indicates_learning(message_lower, topic):
                    tk.level = max(0.0, tk.level - 0.02)
                    tk.signals.append("learning_signal")

                tk.confidence = min(0.95, tk.confidence + 0.03)
                tk.signals = tk.signals[-10:]
            else:
                # New topic
                level = 0.3  # Default: beginner-intermediate
                if self._indicates_expertise(message_lower, topic):
                    level = 0.6
                elif self._indicates_learning(message_lower, topic):
                    level = 0.2

                self._topic_knowledge[topic] = TopicKnowledge(
                    topic=topic,
                    level=level,
                    confidence=0.3,
                    interactions=1,
                    last_seen=now,
                    signals=["first_mention"],
                )

            self._topic_history.append(topic)

        if len(self._topic_history) > 100:
            self._topic_history = self._topic_history[-100:]

    def _indicates_expertise(self, message: str, topic: str) -> bool:
        """Check if message indicates user expertise on topic."""
        expertise_patterns = [
            "i know", "i've used", "i built", "i wrote", "my experience",
            "in my project", "i implemented", "i've been working with",
            "i prefer", "i always use", "i think the best",
        ]
        return _phrase_match(message, expertise_patterns)

    def _indicates_learning(self, message: str, topic: str) -> bool:
        """Check if message indicates user is learning about topic."""
        learning_patterns = [
            "what is", "how do", "how does", "can you explain",
            "i don't understand", "what's the difference",
            "is it possible", "how to", "why does", "i'm new to",
            "i'm learning", "help me understand",
        ]
        return _phrase_match(message, learning_patterns)

    def get_topic_knowledge(self, topic: str) -> Optional[TopicKnowledge]:
        """Get user's knowledge level for a specific topic."""
        return self._topic_knowledge.get(topic.lower())

    def get_knowledge_summary(self, top_n: int = 10) -> List[Dict[str, Any]]:
        """Get summary of user's topic knowledge."""
        # Sort by interaction count
        sorted_topics = sorted(
            self._topic_knowledge.values(),
            key=lambda t: t.interactions,
            reverse=True,
        )

        return [
            {
                "topic": t.topic,
                "level": t.level,
                "label": (
                    "expert" if t.level > 0.7 else
                    "intermediate" if t.level > 0.4 else "beginner"
                ),
                "confidence": t.confidence,
                "interactions": t.interactions,
            }
            for t in sorted_topics[:top_n]
        ]

    # ====================================================================
    # Need Prediction
    # ====================================================================

    def _update_predictions(self, message: str, now: datetime) -> None:
        """Update anticipated needs based on patterns."""
        self._need_predictions.clear()
        message_lower = message.lower()

        # Pattern: User asking about errors → likely needs debugging help
        if any(w in message_lower for w in ["error", "bug", "crash", "exception", "traceback"]):
            self._need_predictions.append(NeedPrediction(
                need="debugging_assistance",
                confidence=0.8,
                basis="error-related keywords in message",
                suggested_action="Offer systematic debugging approach",
            ))

        # Pattern: User seems stuck → offer alternative approaches
        if self._emotional_state.frustration > 0.5:
            self._need_predictions.append(NeedPrediction(
                need="alternative_approach",
                confidence=self._emotional_state.frustration,
                basis="elevated frustration level",
                suggested_action="Suggest alternative approach or take a step back",
            ))

        # Pattern: Time-of-day based prediction
        hour = now.hour
        if hour in self._time_patterns:
            common_topics = sorted(
                self._time_patterns[hour].items(),
                key=lambda x: -x[1],
            )
            if common_topics:
                top_topic = common_topics[0][0]
                count = common_topics[0][1]
                if count >= 3:
                    self._need_predictions.append(NeedPrediction(
                        need=f"topic_{top_topic}",
                        confidence=min(0.7, count * 0.1),
                        basis=f"User often works on '{top_topic}' at this hour",
                        suggested_action=f"Be ready for {top_topic}-related questions",
                    ))

        # Pattern: Long message → user needs help formulating
        if len(message) > 500:
            self._need_predictions.append(NeedPrediction(
                need="problem_decomposition",
                confidence=0.5,
                basis="long message suggests complex problem",
                suggested_action="Help break down the problem into steps",
            ))

        # Pattern: Short repeated messages → user is iterating
        if len(self._message_lengths) >= 3:
            recent = self._message_lengths[-3:]
            if all(l < 50 for l in recent):
                self._need_predictions.append(NeedPrediction(
                    need="quick_iteration",
                    confidence=0.6,
                    basis="series of short messages",
                    suggested_action="Stay responsive, keep answers concise",
                ))

    def get_need_predictions(self) -> List[Dict[str, Any]]:
        """Get current need predictions."""
        return [
            {
                "need": p.need,
                "confidence": p.confidence,
                "basis": p.basis,
                "suggested_action": p.suggested_action,
            }
            for p in sorted(self._need_predictions, key=lambda x: -x.confidence)
        ]

    # ====================================================================
    # System Prompt Injection
    # ====================================================================

    def get_context_for_prompt(self) -> str:
        """Get Theory of Mind context for system prompt injection."""
        parts = []

        # Emotional state (only if confident enough)
        emo = self._emotional_state
        if emo.confidence > 0.3:
            parts.append(f"User appears: {emo.describe()}")

        # Communication style guidance
        style_guide = self.get_style_guidance()
        if style_guide:
            parts.append(style_guide)

        # Key topic knowledge
        knowledge = self.get_knowledge_summary(top_n=5)
        expert_topics = [k["topic"] for k in knowledge if k["level"] > 0.6]
        beginner_topics = [k["topic"] for k in knowledge if k["level"] < 0.3]

        if expert_topics:
            parts.append(f"User is experienced with: {', '.join(expert_topics)}")
        if beginner_topics:
            parts.append(f"User is learning: {', '.join(beginner_topics)}")

        # Active needs
        needs = self.get_need_predictions()
        high_conf_needs = [n for n in needs if n["confidence"] > 0.6]
        if high_conf_needs:
            action = high_conf_needs[0]["suggested_action"]
            parts.append(f"Anticipated need: {action}")

        if not parts:
            return ""

        return "[User Model]\n" + "\n".join(parts)

    def get_observations_for_inference(self) -> Dict[str, float]:
        """Get observations formatted for Active Inference engine."""
        emo = self._emotional_state
        return {
            "emotional_valence": (emo.valence + 1.0) / 2.0,  # Map -1..1 to 0..1
            "user_engagement": emo.engagement,
            "user_frustration": emo.frustration,
        }

    # ====================================================================
    # Status & Export
    # ====================================================================

    def get_status(self) -> Dict[str, Any]:
        """Get Theory of Mind status for API."""
        return {
            "emotional_state": self._emotional_state.to_dict(),
            "emotional_description": self._emotional_state.describe(),
            "communication_style": self._comm_style.to_dict(),
            "topic_count": len(self._topic_knowledge),
            "top_topics": self.get_knowledge_summary(top_n=5),
            "need_predictions": self.get_need_predictions(),
            "message_count": len(self._message_times),
            "style_samples": self._comm_style.samples,
        }

    def get_full_model(self) -> Dict[str, Any]:
        """Get complete user mental model."""
        return {
            "emotional_state": self._emotional_state.to_dict(),
            "communication_style": self._comm_style.to_dict(),
            "topic_knowledge": {
                t: {
                    "level": tk.level, "confidence": tk.confidence,
                    "interactions": tk.interactions,
                }
                for t, tk in self._topic_knowledge.items()
            },
            "need_predictions": self.get_need_predictions(),
            "time_patterns": {
                str(h): dict(sorted(topics.items(), key=lambda x: -x[1])[:3])
                for h, topics in self._time_patterns.items()
                if topics
            },
            "system_prompt_preview": self.get_context_for_prompt(),
        }

    # ====================================================================
    # Persistence
    # ====================================================================

    def _state_file(self) -> Path:
        return self._data_dir / "tom_state.json"

    def _load_state(self) -> None:
        """Load persisted state."""
        sf = self._state_file()
        if not sf.exists():
            return

        try:
            data = json.loads(sf.read_text(encoding="utf-8"))

            # Emotional state
            emo = data.get("emotional_state", {})
            if emo:
                safe_emo = {k: float(v) for k, v in emo.items() if k in _EMOTIONAL_STATE_FIELDS}
                self._emotional_state = EmotionalState(**safe_emo)

            # Communication style
            style = data.get("communication_style", {})
            if style:
                safe_style = {k: v for k, v in style.items() if k in _COMM_STYLE_FIELDS}
                self._comm_style = CommunicationStyle(**safe_style)

            # Topic knowledge
            for t, tk_data in data.get("topic_knowledge", {}).items():
                safe_tk = {k: v for k, v in tk_data.items() if k in _TOPIC_KNOWLEDGE_FIELDS}
                self._topic_knowledge[t] = TopicKnowledge(**safe_tk)

            # Time patterns
            self._time_patterns = {
                int(k): v for k, v in data.get("time_patterns", {}).items()
            }

        except Exception as e:
            logger.warning(f"[ToM] Failed to load state: {e}")

    def _save_state(self) -> None:
        """Save state to disk."""
        try:
            data = {
                "emotional_state": self._emotional_state.to_dict(),
                "communication_style": self._comm_style.to_dict(),
                "topic_knowledge": {
                    t: {
                        "topic": tk.topic, "level": tk.level,
                        "confidence": tk.confidence,
                        "interactions": tk.interactions,
                        "last_seen": tk.last_seen,
                        "signals": tk.signals[-5:],
                    }
                    for t, tk in self._topic_knowledge.items()
                },
                "time_patterns": {
                    str(k): v for k, v in self._time_patterns.items()
                },
                "saved_at": datetime.now().isoformat(),
            }
            self._state_file().write_text(
                json.dumps(data, indent=2), encoding="utf-8"
            )
        except Exception as e:
            logger.warning(f"[ToM] Failed to save state: {e}")


# ============================================================================
# Singleton
# ============================================================================

_tom_instance: Optional[TheoryOfMind] = None


def get_theory_of_mind() -> TheoryOfMind:
    """Get or create the Theory of Mind singleton."""
    global _tom_instance
    if _tom_instance is None:
        _tom_instance = TheoryOfMind()
    return _tom_instance
