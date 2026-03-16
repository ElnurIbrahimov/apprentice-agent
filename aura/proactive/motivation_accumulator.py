"""
Motivation-Threshold Proactive Message System (Roadmap 4.2).

Accumulates motivation per potential proactive message using a 5-factor
weighted formula. Messages are only delivered when motivation exceeds a
learned threshold AND the user is not busy.

The threshold adapts from user feedback: engaged responses lower it
(more messages), dismissals/ignores raise it (fewer messages).

Factors:
    relevance_to_user  * 0.30  — How relevant is this to user's current focus
    time_since_similar * 0.20  — How long since a similar message was sent
    emotional_urgency  * 0.20  — ALMA emotional state urgency signal
    curiosity_drive    * 0.15  — Intrinsic curiosity drive intensity
    user_receptivity   * 0.15  — Active Inference belief: user is receptive

Integrates with:
    - GatewayDaemon: replaces raw rate-limit gating with scored motivation
    - ALMA Engine: emotional_urgency signal
    - IntrinsicMotivation: curiosity_drive signal
    - ActiveInference: user_receptivity belief
    - Theory of Mind: user engagement for receptivity
    - Persistence: stores engagement history for threshold learning
"""

import json
import logging
import os
import tempfile
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ============================================================================
# Data Models
# ============================================================================

@dataclass
class PotentialMessage:
    """A candidate proactive message with its motivation factors."""
    message_id: str
    content: str
    source: str                       # e.g. "curiosity", "social", "coherence", "insight"
    relevance_to_user: float = 0.5    # 0-1
    time_since_similar: float = 0.5   # 0-1 (normalized: 0=just sent, 1=long ago)
    emotional_urgency: float = 0.5    # 0-1
    curiosity_drive: float = 0.5      # 0-1
    user_receptivity: float = 0.5     # 0-1
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)


@dataclass
class EngagementRecord:
    """Record of user engagement with a delivered proactive message."""
    message_id: str
    source: str
    motivation_score: float
    delivered_at: float
    response_type: str = "pending"    # "engaged", "dismissed", "ignored"
    response_time: Optional[float] = None   # seconds until response
    responded_at: Optional[float] = None


# ============================================================================
# Motivation Accumulator
# ============================================================================

class MotivationAccumulator:
    """Scores potential proactive messages and gates delivery via a learned threshold.

    The threshold starts at 0.55 and adapts based on user engagement:
    - Engaged responses: threshold -= 0.01 (min 0.30)
    - Dismissed: threshold += 0.02 (max 0.85)
    - Ignored: threshold += 0.015 (max 0.85)

    Per-source thresholds are tracked separately so that message types
    the user likes get delivered more often.
    """

    # Factor weights (must sum to 1.0)
    W_RELEVANCE = 0.30
    W_TIME_SINCE = 0.20
    W_EMOTIONAL = 0.20
    W_CURIOSITY = 0.15
    W_RECEPTIVITY = 0.15

    # Threshold bounds
    DEFAULT_THRESHOLD = 0.55
    MIN_THRESHOLD = 0.30
    MAX_THRESHOLD = 0.85

    # Engagement history window
    MAX_HISTORY = 50

    def __init__(self, data_dir: Optional[str] = None):
        if data_dir is None:
            base = Path(__file__).resolve().parent.parent.parent
            data_dir = str(base / "data" / "proactive")
        self._data_dir = Path(data_dir)
        self._data_dir.mkdir(parents=True, exist_ok=True)

        self._lock = threading.RLock()

        # Global threshold
        self._threshold: float = self.DEFAULT_THRESHOLD

        # Per-source thresholds (override global when enough data)
        self._source_thresholds: Dict[str, float] = {}

        # Engagement history: list of EngagementRecord dicts
        self._engagement_history: List[Dict[str, Any]] = []

        # Track last delivery time per source for time_since_similar
        self._last_delivery: Dict[str, float] = {}

        # Stats
        self._stats = {
            "messages_scored": 0,
            "messages_passed": 0,
            "messages_blocked": 0,
            "threshold_adjustments": 0,
        }

        self._load_state()
        logger.info(
            f"[MotivationAccumulator] Initialized, threshold={self._threshold:.3f}, "
            f"source_thresholds={self._source_thresholds}"
        )

    # ====================================================================
    # Scoring
    # ====================================================================

    def score(self, msg: PotentialMessage) -> float:
        """Compute motivation score for a potential message using the 5-factor formula.

        Returns a float 0-1.
        """
        motivation = (
            msg.relevance_to_user * self.W_RELEVANCE
            + msg.time_since_similar * self.W_TIME_SINCE
            + msg.emotional_urgency * self.W_EMOTIONAL
            + msg.curiosity_drive * self.W_CURIOSITY
            + msg.user_receptivity * self.W_RECEPTIVITY
        )
        self._stats["messages_scored"] += 1
        return min(1.0, max(0.0, motivation))

    def should_deliver(self, msg: PotentialMessage, user_busy: bool = False) -> bool:
        """Check if a message should be delivered.

        Args:
            msg: The potential message with its factor scores.
            user_busy: Whether the user appears busy (DND, focused, etc).

        Returns:
            True if motivation > threshold AND user is not busy.
        """
        if user_busy:
            return False

        motivation = self.score(msg)
        threshold = self._get_threshold(msg.source)

        passed = motivation > threshold
        if passed:
            self._stats["messages_passed"] += 1
        else:
            self._stats["messages_blocked"] += 1

        logger.debug(
            f"[MotivationAccumulator] {msg.source}/{msg.message_id}: "
            f"motivation={motivation:.3f}, threshold={threshold:.3f}, "
            f"passed={passed}, busy={user_busy}"
        )
        return passed

    def enrich_factors(self, msg: PotentialMessage) -> PotentialMessage:
        """Auto-fill motivation factors from live system state.

        Fills in any factors that are still at the default 0.5 by querying
        ALMA, IntrinsicMotivation, ActiveInference, and ToM.
        """
        # -- emotional_urgency from ALMA --
        try:
            from aura.emotion.alma_engine import get_emotional_state
            state = get_emotional_state()
            pad = state.get("pad", {})
            arousal = pad.get("arousal", 0.0)
            pleasure = pad.get("pleasure", 0.0)
            # High arousal + negative pleasure = urgency; high arousal + positive = mild urgency
            if arousal > 0.3:
                msg.emotional_urgency = min(1.0, arousal * (0.5 + abs(pleasure) * 0.5))
            else:
                msg.emotional_urgency = arousal * 0.5
        except Exception:
            pass

        # -- curiosity_drive from IntrinsicMotivation --
        try:
            from aura.consciousness.intrinsic_motivation import get_intrinsic_motivation
            im = get_intrinsic_motivation()
            drives = im.get_drives_summary()
            msg.curiosity_drive = drives.get("curiosity", 0.5)
        except Exception:
            pass

        # -- user_receptivity from ActiveInference beliefs + ToM --
        try:
            from aura.proactive.gateway_daemon import get_gateway_daemon
            daemon = get_gateway_daemon()
            beliefs = daemon.inference_engine.get_beliefs()
            msg.user_receptivity = beliefs.user_receptive
        except Exception:
            pass

        # Blend in ToM engagement
        try:
            from aura.proactive.theory_of_mind import get_theory_of_mind
            tom = get_theory_of_mind()
            emo = tom.get_emotional_state()
            # High engagement = more receptive, high frustration = less
            tom_receptivity = emo.engagement * 0.6 + (1.0 - emo.frustration) * 0.4
            # Blend with belief-based receptivity
            msg.user_receptivity = msg.user_receptivity * 0.6 + tom_receptivity * 0.4
        except Exception:
            pass

        # -- time_since_similar from delivery history --
        last = self._last_delivery.get(msg.source, 0.0)
        if last > 0:
            hours_since = (time.time() - last) / 3600.0
            # Normalize: 0 at 0h, ~0.5 at 1h, ~0.8 at 4h, ~1.0 at 8h+
            msg.time_since_similar = min(1.0, hours_since / 8.0)
        else:
            msg.time_since_similar = 1.0  # Never sent before

        return msg

    # ====================================================================
    # Delivery Recording
    # ====================================================================

    def record_delivery(self, msg: PotentialMessage, motivation_score: float) -> str:
        """Record that a message was delivered.

        Returns the message_id for later feedback tracking.
        """
        with self._lock:
            self._last_delivery[msg.source] = time.time()

            record = EngagementRecord(
                message_id=msg.message_id,
                source=msg.source,
                motivation_score=motivation_score,
                delivered_at=time.time(),
            )
            self._engagement_history.append({
                "message_id": record.message_id,
                "source": record.source,
                "motivation_score": record.motivation_score,
                "delivered_at": record.delivered_at,
                "response_type": "pending",
                "response_time": None,
            })

            # Cap history
            if len(self._engagement_history) > self.MAX_HISTORY:
                self._engagement_history = self._engagement_history[-self.MAX_HISTORY:]

            self._save_state()
        return msg.message_id

    def record_engagement(
        self,
        message_id: str,
        response_type: str,
        response_time: Optional[float] = None,
    ) -> None:
        """Record user engagement with a delivered message.

        Args:
            message_id: ID of the delivered message.
            response_type: "engaged", "dismissed", or "ignored".
            response_time: Seconds between delivery and response (optional).
        """
        with self._lock:
            # Find and update the record
            for record in reversed(self._engagement_history):
                if record["message_id"] == message_id:
                    record["response_type"] = response_type
                    record["response_time"] = response_time
                    break

            # Adapt threshold
            self._adapt_threshold(response_type, message_id)
            self._save_state()

        logger.info(
            f"[MotivationAccumulator] Engagement recorded: {message_id} -> {response_type}"
        )

    # ====================================================================
    # Threshold Learning
    # ====================================================================

    def _get_threshold(self, source: str) -> float:
        """Get effective threshold for a message source.

        Uses per-source threshold if enough data (>= 5 records), else global.
        """
        source_th = self._source_thresholds.get(source)
        if source_th is not None:
            # Count records for this source
            source_count = sum(
                1 for r in self._engagement_history
                if r.get("source") == source and r.get("response_type") != "pending"
            )
            if source_count >= 5:
                return source_th
        return self._threshold

    def _adapt_threshold(self, response_type: str, message_id: str) -> None:
        """Adapt threshold based on user response.

        Engaged -> lower threshold (deliver more)
        Dismissed -> raise threshold (deliver less)
        Ignored -> raise threshold slightly
        """
        # Find the source for this message
        source = None
        for record in reversed(self._engagement_history):
            if record["message_id"] == message_id:
                source = record.get("source")
                break

        # Global threshold adjustment
        if response_type == "engaged":
            self._threshold = max(self.MIN_THRESHOLD, self._threshold - 0.01)
        elif response_type == "dismissed":
            self._threshold = min(self.MAX_THRESHOLD, self._threshold + 0.02)
        elif response_type == "ignored":
            self._threshold = min(self.MAX_THRESHOLD, self._threshold + 0.015)

        # Per-source threshold adjustment
        if source:
            current = self._source_thresholds.get(source, self._threshold)
            if response_type == "engaged":
                self._source_thresholds[source] = max(self.MIN_THRESHOLD, current - 0.015)
            elif response_type == "dismissed":
                self._source_thresholds[source] = min(self.MAX_THRESHOLD, current + 0.025)
            elif response_type == "ignored":
                self._source_thresholds[source] = min(self.MAX_THRESHOLD, current + 0.02)

        # Converge toward default over time (prevents runaway)
        self._threshold += (self.DEFAULT_THRESHOLD - self._threshold) * 0.02
        for src in self._source_thresholds:
            self._source_thresholds[src] += (
                self.DEFAULT_THRESHOLD - self._source_thresholds[src]
            ) * 0.02

        self._stats["threshold_adjustments"] += 1

        logger.debug(
            f"[MotivationAccumulator] Threshold adapted: global={self._threshold:.3f}, "
            f"source={source}={self._source_thresholds.get(source, 'N/A')}"
        )

    # ====================================================================
    # Relevance Scoring
    # ====================================================================

    def compute_relevance(self, content: str, topics: Optional[List[str]] = None) -> float:
        """Estimate relevance of message content to user's current focus.

        Uses focus context tracker + keyword overlap.
        """
        relevance = 0.3  # Base relevance

        # Get user's current focus topics
        user_topics: List[str] = []
        try:
            from api.routes.context import get_tracker
            ctx = get_tracker()
            focus = ctx.get_focus_state(limit=5)
            user_topics = [item["name"].lower() for item in focus.get("items", [])]
        except Exception:
            pass

        # Get current app/task from daemon context
        try:
            from aura.proactive.gateway_daemon import get_gateway_daemon
            daemon = get_gateway_daemon()
            if daemon.user_context.current_task:
                user_topics.append(daemon.user_context.current_task.lower())
            if daemon.user_context.focus_keywords:
                user_topics.extend(k.lower() for k in daemon.user_context.focus_keywords)
        except Exception:
            pass

        if not user_topics:
            return relevance

        # Check content overlap with user topics
        content_lower = content.lower()
        matches = sum(1 for t in user_topics if t in content_lower)
        if matches > 0:
            relevance += min(0.5, matches * 0.15)

        # Check explicitly provided topics
        if topics:
            topic_matches = sum(1 for t in topics if t.lower() in content_lower)
            relevance += min(0.2, topic_matches * 0.1)

        return min(1.0, relevance)

    # ====================================================================
    # Engagement Analytics
    # ====================================================================

    def get_engagement_rate(self, source: Optional[str] = None, last_n: int = 20) -> float:
        """Get engagement rate for recent messages.

        Args:
            source: Filter by source type. None = all sources.
            last_n: Look at last N records.

        Returns:
            Engagement rate 0-1.
        """
        records = [
            r for r in self._engagement_history
            if r.get("response_type") != "pending"
            and (source is None or r.get("source") == source)
        ]
        recent = records[-last_n:] if records else []
        if not recent:
            return 0.5  # No data, assume neutral

        engaged = sum(1 for r in recent if r["response_type"] == "engaged")
        return engaged / len(recent)

    def get_avg_response_time(self, source: Optional[str] = None) -> Optional[float]:
        """Get average response time for engaged messages."""
        records = [
            r for r in self._engagement_history
            if r.get("response_type") == "engaged"
            and r.get("response_time") is not None
            and (source is None or r.get("source") == source)
        ]
        if not records:
            return None
        return sum(r["response_time"] for r in records) / len(records)

    # ====================================================================
    # Status & API
    # ====================================================================

    def get_status(self) -> Dict[str, Any]:
        """Get accumulator status for API/debugging."""
        return {
            "global_threshold": round(self._threshold, 3),
            "source_thresholds": {
                k: round(v, 3) for k, v in self._source_thresholds.items()
            },
            "engagement_history_size": len(self._engagement_history),
            "engagement_rate": round(self.get_engagement_rate(), 3),
            "avg_response_time": self.get_avg_response_time(),
            "last_delivery_times": {
                k: round(time.time() - v, 0) for k, v in self._last_delivery.items()
            },
            "stats": dict(self._stats),
        }

    # ====================================================================
    # Persistence
    # ====================================================================

    def _state_file(self) -> Path:
        return self._data_dir / "motivation_state.json"

    def _load_state(self) -> None:
        sf = self._state_file()
        if not sf.exists():
            return
        try:
            data = json.loads(sf.read_text(encoding="utf-8"))
            self._threshold = data.get("threshold", self.DEFAULT_THRESHOLD)
            self._source_thresholds = data.get("source_thresholds", {})
            self._engagement_history = data.get("engagement_history", [])
            self._last_delivery = data.get("last_delivery", {})
            self._stats.update(data.get("stats", {}))
        except Exception as e:
            logger.warning(f"[MotivationAccumulator] Load state error: {e}")

    def _save_state(self) -> None:
        try:
            data = {
                "threshold": self._threshold,
                "source_thresholds": self._source_thresholds,
                "engagement_history": self._engagement_history[-self.MAX_HISTORY:],
                "last_delivery": self._last_delivery,
                "stats": self._stats,
                "saved_at": datetime.now().isoformat(),
            }
            target = self._state_file()
            fd, tmp_path = tempfile.mkstemp(
                dir=str(target.parent), suffix=".tmp"
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, default=str)
                os.replace(tmp_path, str(target))
            except BaseException:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise
        except Exception as e:
            logger.warning(f"[MotivationAccumulator] Save state error: {e}")


# ============================================================================
# Singleton
# ============================================================================

_accumulator: Optional[MotivationAccumulator] = None
_accumulator_lock = threading.Lock()


def get_motivation_accumulator() -> MotivationAccumulator:
    """Get or create the global MotivationAccumulator."""
    global _accumulator
    if _accumulator is None:
        with _accumulator_lock:
            if _accumulator is None:
                _accumulator = MotivationAccumulator()
    return _accumulator
