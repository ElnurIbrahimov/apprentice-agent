"""
PatternProphet - Cross-Conversation Pattern Recognition for AURA v3.0

Recognizes and learns patterns across conversations:
- Topic sequences (A often followed by B)
- Time-based patterns (always asks X in morning)
- Behavioral patterns (frustration -> needs break)
- Interest clusters (related topics)

Makes AURA anticipate needs and provide contextual help.
"""

import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Set
from collections import defaultdict

from aura.jsonl_utils import rotate_jsonl_if_needed

logger = logging.getLogger(__name__)


@dataclass
class Pattern:
    """A recognized behavioral pattern."""
    name: str
    pattern_type: str  # "sequence", "temporal", "behavioral", "cluster"
    description: str
    confidence: float = 0.5  # 0.0-1.0
    occurrences: int = 1
    triggers: List[str] = field(default_factory=list)
    predictions: List[str] = field(default_factory=list)
    last_seen: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Pattern":
        return cls(**data)


@dataclass
class Interaction:
    """Record of a single interaction for pattern analysis."""
    timestamp: str
    topic: str
    keywords: List[str]
    hour: int
    day_of_week: int
    sentiment: str  # "positive", "neutral", "negative"
    duration_seconds: int = 0


class PatternProphet:
    """
    Pattern recognition engine that learns user behavior.

    Features:
    - Topic sequence detection
    - Time-of-day patterns
    - Sentiment-based predictions
    - Interest clustering
    """

    # Minimum occurrences to consider a pattern valid
    MIN_PATTERN_OCCURRENCES = 3

    # Confidence decay per day without seeing pattern
    CONFIDENCE_DECAY = 0.05

    def __init__(self, data_dir: Optional[str] = None):
        """
        Initialize the pattern recognizer.

        Args:
            data_dir: Directory for storing patterns
        """
        if data_dir is None:
            data_dir = Path(__file__).parent.parent / "data"

        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.patterns_file = self.data_dir / "patterns.json"
        self.interactions_file = self.data_dir / "interactions.jsonl"

        # Load existing patterns
        self.patterns: Dict[str, Pattern] = self._load_patterns()
        self.interactions: List[Interaction] = []

        # Working memory for pattern detection
        self._topic_sequences: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self._temporal_patterns: Dict[int, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self._keyword_cooccurrence: Dict[str, Set[str]] = defaultdict(set)

        # Load recent interactions
        self._load_recent_interactions(days=30)

        logger.info(f"PatternProphet initialized with {len(self.patterns)} patterns")

    def _load_patterns(self) -> Dict[str, Pattern]:
        """Load patterns from file."""
        patterns = {}
        if self.patterns_file.exists():
            try:
                data = json.loads(self.patterns_file.read_text(encoding="utf-8"))
                for name, pdata in data.items():
                    patterns[name] = Pattern.from_dict(pdata)
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                logger.warning(f"Error loading patterns: {e}")
        return patterns

    def _save_patterns(self) -> None:
        """Save patterns to file."""
        try:
            import tempfile
            data = {name: p.to_dict() for name, p in self.patterns.items()}
            fd, tmp_path = tempfile.mkstemp(dir=str(self.patterns_file.parent), suffix=".tmp")
            try:
                with os.fdopen(fd, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2)
                os.replace(tmp_path, str(self.patterns_file))
            except Exception:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise
        except IOError as e:
            logger.error(f"Error saving patterns: {e}")

    def _load_recent_interactions(self, days: int = 30) -> None:
        """Load recent interactions for analysis."""
        if not self.interactions_file.exists():
            return

        cutoff = datetime.now() - timedelta(days=days)

        try:
            with open(self.interactions_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        ts = datetime.fromisoformat(data["timestamp"])
                        if ts > cutoff:
                            self.interactions.append(Interaction(**data))
                    except (json.JSONDecodeError, KeyError, TypeError):
                        continue
        except IOError as e:
            logger.error(f"Error loading interactions: {e}")

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract significant keywords from text."""
        # Simple keyword extraction - could be enhanced with NLP
        words = text.lower().split()

        # Filter out common words
        stop_words = {
            "a", "an", "the", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "must", "shall",
            "can", "need", "dare", "ought", "used", "to", "of", "in",
            "for", "on", "with", "at", "by", "from", "as", "into",
            "through", "during", "before", "after", "above", "below",
            "between", "under", "again", "further", "then", "once",
            "i", "me", "my", "you", "your", "he", "she", "it", "we",
            "they", "what", "which", "who", "whom", "this", "that",
            "these", "those", "am", "and", "but", "if", "or", "because",
            "until", "while", "how", "all", "each", "few", "more",
            "most", "other", "some", "such", "no", "not", "only",
            "own", "same", "so", "than", "too", "very", "just"
        }

        keywords = [w for w in words if len(w) > 3 and w not in stop_words]

        # Return unique keywords, limited
        seen = set()
        unique = []
        for kw in keywords:
            if kw not in seen:
                seen.add(kw)
                unique.append(kw)
        return unique[:10]

    def _classify_topic(self, text: str, keywords: List[str]) -> str:
        """Classify text into a broad topic category."""
        text_lower = text.lower()

        topic_indicators = {
            "coding": ["code", "python", "javascript", "function", "bug", "error", "debug", "programming"],
            "learning": ["learn", "understand", "explain", "teach", "how does", "what is"],
            "planning": ["plan", "todo", "task", "schedule", "organize", "project"],
            "creative": ["write", "create", "design", "idea", "story", "art"],
            "troubleshooting": ["problem", "issue", "fix", "broken", "help", "wrong"],
            "research": ["search", "find", "look up", "information", "about"],
            "casual": ["hello", "hi", "hey", "thanks", "bye", "chat"]
        }

        scores = {topic: 0 for topic in topic_indicators}

        for topic, indicators in topic_indicators.items():
            for indicator in indicators:
                if indicator in text_lower or indicator in keywords:
                    scores[topic] += 1

        best_topic = max(scores, key=scores.get)
        return best_topic if scores[best_topic] > 0 else "general"

    def _detect_sentiment(self, text: str) -> str:
        """Simple sentiment detection."""
        text_lower = text.lower()

        positive = ["thanks", "great", "awesome", "perfect", "love", "excellent", "good"]
        negative = ["wrong", "error", "bad", "hate", "terrible", "frustrated", "annoyed"]

        pos_count = sum(1 for w in positive if w in text_lower)
        neg_count = sum(1 for w in negative if w in text_lower)

        if pos_count > neg_count:
            return "positive"
        elif neg_count > pos_count:
            return "negative"
        return "neutral"

    def record_interaction(
        self,
        user_input: str,
        previous_topic: Optional[str] = None
    ) -> Interaction:
        """
        Record an interaction for pattern analysis.

        Args:
            user_input: The user's message
            previous_topic: Topic of previous interaction (if any)

        Returns:
            The recorded interaction
        """
        now = datetime.now()
        keywords = self._extract_keywords(user_input)
        topic = self._classify_topic(user_input, keywords)
        sentiment = self._detect_sentiment(user_input)

        interaction = Interaction(
            timestamp=now.isoformat(),
            topic=topic,
            keywords=keywords,
            hour=now.hour,
            day_of_week=now.weekday(),
            sentiment=sentiment
        )

        self.interactions.append(interaction)

        # Record to file
        try:
            rotate_jsonl_if_needed(self.interactions_file)
            with open(self.interactions_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(interaction)) + "\n")
        except IOError as e:
            logger.error(f"Error recording interaction: {e}")

        # Update working memory
        if previous_topic:
            self._topic_sequences[previous_topic][topic] += 1

        self._temporal_patterns[now.hour][topic] += 1

        for kw in keywords:
            self._keyword_cooccurrence[kw].update(keywords)

        # Trigger pattern detection
        self._detect_patterns()

        return interaction

    def _detect_patterns(self) -> None:
        """Analyze data and detect patterns."""
        self._detect_sequence_patterns()
        self._detect_temporal_patterns()
        self._detect_cluster_patterns()
        self._save_patterns()

    def _detect_sequence_patterns(self) -> None:
        """Detect topic sequence patterns (A -> B)."""
        for topic_a, followers in self._topic_sequences.items():
            for topic_b, count in followers.items():
                if count >= self.MIN_PATTERN_OCCURRENCES:
                    name = f"seq_{topic_a}_{topic_b}"
                    confidence = min(0.9, count / 10)

                    if name in self.patterns:
                        self.patterns[name].occurrences = count
                        self.patterns[name].confidence = confidence
                        self.patterns[name].last_seen = datetime.now().isoformat()
                    else:
                        self.patterns[name] = Pattern(
                            name=name,
                            pattern_type="sequence",
                            description=f"After {topic_a}, user often moves to {topic_b}",
                            confidence=confidence,
                            occurrences=count,
                            triggers=[topic_a],
                            predictions=[topic_b]
                        )

    def _detect_temporal_patterns(self) -> None:
        """Detect time-of-day patterns."""
        for hour, topics in self._temporal_patterns.items():
            for topic, count in topics.items():
                if count >= self.MIN_PATTERN_OCCURRENCES:
                    name = f"time_{hour}_{topic}"
                    confidence = min(0.85, count / 15)

                    time_desc = self._hour_to_period(hour)

                    if name in self.patterns:
                        self.patterns[name].occurrences = count
                        self.patterns[name].confidence = confidence
                        self.patterns[name].last_seen = datetime.now().isoformat()
                    else:
                        self.patterns[name] = Pattern(
                            name=name,
                            pattern_type="temporal",
                            description=f"User often does {topic} activities {time_desc}",
                            confidence=confidence,
                            occurrences=count,
                            triggers=[f"time:{hour}"],
                            predictions=[topic]
                        )

    def _detect_cluster_patterns(self) -> None:
        """Detect keyword clusters (related interests)."""
        # Find keywords that frequently appear together
        for keyword, cowords in self._keyword_cooccurrence.items():
            if len(cowords) >= 5:  # Enough co-occurrences
                # Find the strongest connections (exclude the keyword itself)
                related_words = [w for w in cowords if w != keyword]
                top_related = sorted(related_words, key=lambda w: len(self._keyword_cooccurrence.get(w, set()) & cowords))[-5:]

                if len(top_related) >= 3:
                    name = f"cluster_{keyword}"
                    # Create clean description without redundancy
                    unique_related = [w for w in top_related[:3] if w != keyword]
                    if unique_related:
                        description = f"Interest in '{keyword}' often comes with: {', '.join(unique_related)}"
                    else:
                        description = f"Recurring interest in '{keyword}'"

                    if name not in self.patterns:
                        self.patterns[name] = Pattern(
                            name=name,
                            pattern_type="cluster",
                            description=description,
                            confidence=0.6,
                            occurrences=len(cowords),
                            triggers=[keyword],
                            predictions=top_related
                        )

    def _hour_to_period(self, hour: int) -> str:
        """Convert hour to human-readable period."""
        if 5 <= hour < 9:
            return "in early morning"
        elif 9 <= hour < 12:
            return "mid-morning"
        elif 12 <= hour < 14:
            return "around lunch"
        elif 14 <= hour < 17:
            return "in the afternoon"
        elif 17 <= hour < 21:
            return "in the evening"
        elif 21 <= hour or hour < 5:
            return "late at night"
        return "at this hour"

    def predict(self, context: str, current_hour: Optional[int] = None) -> List[Tuple[str, float]]:
        """
        Predict what the user might want based on context.

        Args:
            context: Current topic or context
            current_hour: Current hour (default: now)

        Returns:
            List of (prediction, confidence) tuples
        """
        if current_hour is None:
            current_hour = datetime.now().hour

        predictions = []

        for pattern in self.patterns.values():
            # Check sequence patterns
            if pattern.pattern_type == "sequence" and context in pattern.triggers:
                for pred in pattern.predictions:
                    predictions.append((pred, pattern.confidence))

            # Check temporal patterns
            if pattern.pattern_type == "temporal" and f"time:{current_hour}" in pattern.triggers:
                for pred in pattern.predictions:
                    predictions.append((pred, pattern.confidence * 0.8))  # Slightly lower

            # Check cluster patterns
            if pattern.pattern_type == "cluster":
                context_words = set(context.lower().split())
                if any(t in context_words for t in pattern.triggers):
                    for pred in pattern.predictions[:3]:
                        predictions.append((f"related: {pred}", pattern.confidence * 0.7))

        # Deduplicate and sort by confidence
        seen = set()
        unique_predictions = []
        for pred, conf in sorted(predictions, key=lambda x: x[1], reverse=True):
            if pred not in seen:
                seen.add(pred)
                unique_predictions.append((pred, conf))

        return unique_predictions[:5]

    def get_insights(self) -> List[str]:
        """Get human-readable insights about user patterns."""
        insights = []

        # Most confident patterns
        confident = sorted(
            self.patterns.values(),
            key=lambda p: p.confidence * p.occurrences,
            reverse=True
        )[:5]

        for pattern in confident:
            if pattern.confidence > 0.5:
                insights.append(f"{pattern.description} ({pattern.occurrences} times)")

        if not insights:
            insights.append("Still learning your patterns...")

        return insights

    def get_status(self) -> Dict:
        """Get pattern recognizer status."""
        return {
            "total_patterns": len(self.patterns),
            "total_interactions": len(self.interactions),
            "pattern_types": {
                ptype: sum(1 for p in self.patterns.values() if p.pattern_type == ptype)
                for ptype in ["sequence", "temporal", "behavioral", "cluster"]
            },
            "avg_confidence": round(
                sum(p.confidence for p in self.patterns.values()) / max(1, len(self.patterns)),
                2
            )
        }


if __name__ == "__main__":
    print("=" * 60)
    print("PatternProphet - Pattern Recognition Test")
    print("=" * 60)

    prophet = PatternProphet()

    # Simulate some interactions
    print("\n--- Recording interactions ---")

    interactions = [
        ("How do I write a Python function?", None),
        ("Can you explain decorators?", "coding"),
        ("What about async functions?", "learning"),
        ("Help me debug this error", "coding"),
        ("Thanks, that fixed it!", "troubleshooting"),
    ]

    prev_topic = None
    for text, _ in interactions:
        interaction = prophet.record_interaction(text, prev_topic)
        print(f"  [{interaction.topic}] {text[:40]}...")
        prev_topic = interaction.topic

    # Get predictions
    print("\n--- Predictions ---")
    predictions = prophet.predict("coding")
    for pred, conf in predictions:
        print(f"  {pred}: {conf:.2f}")

    # Get insights
    print("\n--- Insights ---")
    for insight in prophet.get_insights():
        print(f"  - {insight}")

    # Status
    print("\n--- Status ---")
    status = prophet.get_status()
    for k, v in status.items():
        print(f"  {k}: {v}")

    print("\n" + "=" * 60)
    print("Test complete!")
