"""
Global Workspace Theory (GWT) Engine for AURA.

Implements a central broadcast architecture where specialist modules
(emotion, motivation, theory of mind, metacognition, etc.) compete
for access to a shared "global workspace". The winner is broadcast
to all modules via the EventBus, creating a unified conscious experience.

Inspired by Baars' GWT and Franklin's LIDA cognitive architecture:
1. GATHER: Codelets poll specialist modules for salient content
2. COMPETE: Activation scoring + noise → winner selection
3. BROADCAST: Winner published to EventBus "workspace" channel
4. DECAY: Habituation prevents monopoly by any single module

Cognitive cycle runs at ~300ms (adaptive 200-500ms).
"""

import json
import logging
import os
import random
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# ============================================================================
# Data Models
# ============================================================================


@dataclass
class WorkspaceContent:
    """What a codelet submits for competition in the global workspace."""

    source_module: str       # e.g. "alma_emotion", "theory_of_mind"
    content_type: str        # e.g. "emotion_shift", "user_state", "drive_urge"
    summary: str             # Human-readable description
    activation: float        # 0-1, how strongly the module wants broadcast
    salience: float          # 0-1, intrinsic importance of this content
    pad_signature: Optional[Dict[str, float]] = None  # Emotional coloring (P/A/D)
    payload: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    @property
    def effective_activation(self) -> float:
        """Combined activation score: 60% activation, 40% salience."""
        return self.activation * 0.6 + self.salience * 0.4

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_module": self.source_module,
            "content_type": self.content_type,
            "summary": self.summary,
            "activation": round(self.activation, 3),
            "salience": round(self.salience, 3),
            "effective_activation": round(self.effective_activation, 3),
            "pad_signature": self.pad_signature,
            "payload": self.payload,
            "timestamp": self.timestamp,
        }


@dataclass
class ConsciousState:
    """Current workspace contents — the sparse 'c' vector of consciousness."""

    broadcast_content: Optional[WorkspaceContent] = None  # Winner
    secondary_content: List[WorkspaceContent] = field(default_factory=list)  # Runner-ups (max 2)
    attention_focus: str = "idle"  # What AURA is currently attending to
    attention_intensity: float = 0.0
    attention_schema: Dict[str, Any] = field(default_factory=dict)  # AST self-model
    cycle_number: int = 0

    def to_prompt_context(self) -> str:
        """Generate [Conscious Focus] block for system prompt injection."""
        if not self.broadcast_content:
            return ""

        bc = self.broadcast_content
        lines = [
            "[Conscious Focus]",
            f"Attending to: {self.attention_focus}",
            f"Current awareness: {bc.summary}",
            f"Source: {bc.source_module} ({bc.content_type})",
            f"Intensity: {self.attention_intensity:.1f}",
        ]

        if self.secondary_content:
            peripherals = ", ".join(s.summary for s in self.secondary_content[:2])
            lines.append(f"Peripheral awareness: {peripherals}")

        if bc.pad_signature:
            p, a, d = bc.pad_signature.get("pleasure", 0), bc.pad_signature.get("arousal", 0), bc.pad_signature.get("dominance", 0)
            lines.append(f"Emotional coloring: P={p:+.1f} A={a:.1f} D={d:.1f}")

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "broadcast_content": self.broadcast_content.to_dict() if self.broadcast_content else None,
            "secondary_content": [s.to_dict() for s in self.secondary_content],
            "attention_focus": self.attention_focus,
            "attention_intensity": round(self.attention_intensity, 3),
            "attention_schema": self.attention_schema,
            "cycle_number": self.cycle_number,
            "prompt_context": self.to_prompt_context(),
        }


@dataclass
class Codelet:
    """Specialist module wrapper — gathers content and tracks competition history."""

    name: str
    gather_fn: Callable[[], Optional[WorkspaceContent]]
    priority_weight: float = 1.0
    cooldown_seconds: float = 1.0

    # Internal tracking
    _consecutive_wins: int = 0
    _last_win_time: float = 0.0
    _last_gather_time: float = 0.0
    _total_wins: int = 0
    _total_submissions: int = 0
    _total_errors: int = 0

    def compete(self) -> Optional[WorkspaceContent]:
        """Call gather function, apply priority weight and habituation."""
        now = time.time()

        # Cooldown check
        if now - self._last_gather_time < self.cooldown_seconds:
            return None

        self._last_gather_time = now

        try:
            content = self.gather_fn()
        except Exception as e:
            self._total_errors += 1
            logger.debug(f"[GWT] Codelet '{self.name}' gather error: {e}")
            return None

        if content is None:
            return None

        self._total_submissions += 1

        # Apply priority weight
        content.activation = min(1.0, content.activation * self.priority_weight)

        # Habituation: consecutive wins > 2 reduce activation
        if self._consecutive_wins > 2:
            habituation = max(0.5, 1.0 - self._consecutive_wins * 0.1)
            content.activation *= habituation

        return content

    def record_win(self):
        self._consecutive_wins += 1
        self._last_win_time = time.time()
        self._total_wins += 1

    def record_loss(self):
        self._consecutive_wins = 0

    def get_stats(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "priority_weight": self.priority_weight,
            "cooldown_seconds": self.cooldown_seconds,
            "consecutive_wins": self._consecutive_wins,
            "total_wins": self._total_wins,
            "total_submissions": self._total_submissions,
            "total_errors": self._total_errors,
        }


@dataclass
class BroadcastEvent:
    """What gets published to the EventBus after a competition round."""

    conscious_state: ConsciousState
    cycle_number: int
    cycle_duration_ms: float
    competing_count: int
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "conscious_state": self.conscious_state.to_dict(),
            "cycle_number": self.cycle_number,
            "cycle_duration_ms": round(self.cycle_duration_ms, 1),
            "competing_count": self.competing_count,
            "timestamp": self.timestamp,
        }


# ============================================================================
# Global Workspace Engine
# ============================================================================

class GlobalWorkspaceEngine:
    """Core GWT engine — runs cognitive cycles in a background daemon thread.

    Each cycle:
    1. GATHER: All registered codelets poll their specialist modules
    2. COMPETE: Submissions scored (activation + noise + continuity boost)
    3. BROADCAST: Winner published to EventBus "workspace" channel
    4. DECAY: Habituation tracking updated
    """

    def __init__(self, data_dir: str = "data/global_workspace"):
        self._lock = threading.RLock()
        self._data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)

        # State
        self._conscious_state = ConsciousState()
        self._broadcast_history: deque = deque(maxlen=200)
        self._codelets: Dict[str, Codelet] = {}

        # Attention Schema Theory self-model
        self._attention_schema: Dict[str, Any] = {
            "sustained_focus_seconds": 0.0,
            "current_source": None,
            "switch_count": 0,
            "dominant_sources": {},       # source → win count
            "attention_trajectory": [],   # recent source sequence (max 20)
            "last_switch_time": 0.0,
        }

        # Cycle timing
        self._cycle_interval = 0.3       # 300ms base
        self._min_cycle_interval = 0.2   # 200ms floor
        self._max_cycle_interval = 0.5   # 500ms ceiling

        # Competition params
        self._noise_temperature = 0.1
        self._noise_clip = 0.2
        self._continuity_boost = 0.15    # 15% boost for sustaining attention
        self._habituation_decay = 0.85

        # Runtime
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._cycle_number = 0

        # Stats
        self._stats = {
            "total_cycles": 0,
            "total_broadcasts": 0,
            "empty_cycles": 0,
            "avg_cycle_ms": 0.0,
            "avg_competing": 0.0,
            "start_time": 0.0,
        }

        # Register all codelets
        self._register_codelets()

        # Load persisted state
        self._load_state()

        logger.info(f"[GWT] GlobalWorkspaceEngine initialized with {len(self._codelets)} codelets")

    # ====================================================================
    # Codelet Registration
    # ====================================================================

    def _register_codelets(self):
        """Register all specialist module codelets."""
        codelets = [
            Codelet("alma_emotion", self._gather_alma, priority_weight=1.1, cooldown_seconds=0.5),
            Codelet("intrinsic_motivation", self._gather_motivation, priority_weight=1.0, cooldown_seconds=1.0),
            Codelet("theory_of_mind", self._gather_tom, priority_weight=1.2, cooldown_seconds=1.5),
            Codelet("metacognition", self._gather_metacognition, priority_weight=0.9, cooldown_seconds=2.0),
            Codelet("gateway_daemon", self._gather_gateway, priority_weight=1.0, cooldown_seconds=1.0),
            Codelet("inner_thoughts", self._gather_inner_thoughts, priority_weight=0.8, cooldown_seconds=1.0),
            Codelet("neurodream", self._gather_neurodream, priority_weight=1.1, cooldown_seconds=3.0),
            Codelet("active_inference", self._gather_active_inference, priority_weight=0.9, cooldown_seconds=1.5),
            Codelet("strategy_bandit", self._gather_strategy_bandit, priority_weight=0.8, cooldown_seconds=5.0),
            Codelet("world_model", self._gather_world_model, priority_weight=0.9, cooldown_seconds=2.0),
        ]
        for c in codelets:
            self._codelets[c.name] = c

    # ====================================================================
    # Gather Functions (lazy imports, graceful degradation)
    # ====================================================================

    def _gather_alma(self) -> Optional[WorkspaceContent]:
        """Gather emotional state when intensity is notable."""
        from apprentice_agent.emotion.alma_engine import alma_engine
        state = alma_engine.get_emotional_state()
        intensity = state.get("intensity", 0)
        if intensity < 0.3:
            return None

        dominant = state.get("dominant_emotion", "neutral")
        pad = state.get("pad", {})
        return WorkspaceContent(
            source_module="alma_emotion",
            content_type="emotion_shift",
            summary=f"Feeling {dominant} (intensity {intensity:.1f})",
            activation=min(1.0, intensity * 1.2),
            salience=intensity,
            pad_signature={
                "pleasure": pad.get("pleasure", 0),
                "arousal": pad.get("arousal", 0),
                "dominance": pad.get("dominance", 0),
            },
            payload={"dominant_emotion": dominant, "mood": state.get("mood")},
        )

    def _gather_motivation(self) -> Optional[WorkspaceContent]:
        """Gather dominant drive when urgency is notable."""
        from apprentice_agent.consciousness.intrinsic_motivation import get_intrinsic_motivation
        im = get_intrinsic_motivation()
        status = im.get_status()

        urgency = status.get("dominant_urgency", 0)
        if urgency < 0.35:
            return None

        drive = status.get("dominant_drive", "unknown")
        return WorkspaceContent(
            source_module="intrinsic_motivation",
            content_type="drive_urge",
            summary=f"Drive: {drive} (urgency {urgency:.1f})",
            activation=urgency,
            salience=min(1.0, urgency * 1.1),
            payload={"drive": drive, "drives": status.get("drives", {})},
        )

    def _gather_tom(self) -> Optional[WorkspaceContent]:
        """Gather user state when frustration high or engagement low."""
        from apprentice_agent.config import Config
        if Config.MULTI_USER_ENABLED:
            from apprentice_agent.multi_user import get_multi_user_manager
            manager = get_multi_user_manager()
            user_model = manager.get_active_user_model()
            if not user_model:
                return None
            es = user_model.emotional_state
        else:
            from apprentice_agent.proactive.theory_of_mind import get_theory_of_mind
            tom = get_theory_of_mind()
            es = tom.get_emotional_state()

        frustration = getattr(es, "frustration", 0)
        engagement = getattr(es, "engagement", 0.5)
        valence = getattr(es, "valence", 0)

        # Trigger on high frustration or low engagement
        if frustration < 0.4 and engagement > 0.3:
            return None

        salience = max(frustration, 1.0 - engagement)
        summary_parts = []
        if frustration >= 0.4:
            summary_parts.append(f"user frustrated ({frustration:.1f})")
        if engagement <= 0.3:
            summary_parts.append(f"user disengaged ({engagement:.1f})")

        return WorkspaceContent(
            source_module="theory_of_mind",
            content_type="user_state",
            summary="User state: " + ", ".join(summary_parts),
            activation=salience,
            salience=salience,
            payload={
                "frustration": frustration,
                "engagement": engagement,
                "valence": valence,
            },
        )

    def _gather_metacognition(self) -> Optional[WorkspaceContent]:
        """Gather self-assessment when active goals exist."""
        from apprentice_agent.consciousness.metacognition import get_metacognitive_engine
        mc = get_metacognitive_engine()
        model = mc.get_self_model()

        active_goals = [g for g in model.learning_goals if g.status in ("pending", "active")]
        if not active_goals:
            return None

        # Find weakest capability
        weaknesses = model.weaknesses
        summary = f"Self-monitoring: {len(active_goals)} active goals"
        if weaknesses:
            summary += f", weak in {weaknesses[0]}"

        salience = min(1.0, len(active_goals) / 5.0)
        return WorkspaceContent(
            source_module="metacognition",
            content_type="self_assessment",
            summary=summary,
            activation=salience * 0.8,
            salience=salience,
            payload={
                "active_goals": len(active_goals),
                "weaknesses": weaknesses[:3],
                "strengths": model.strengths[:3],
            },
        )

    def _gather_gateway(self) -> Optional[WorkspaceContent]:
        """Gather proactive signals when task urgency or uncertainty is high."""
        from apprentice_agent.proactive.gateway_daemon import get_gateway_daemon
        daemon = get_gateway_daemon()
        stats = daemon.get_stats()

        if stats.get("state") != "running":
            return None

        beliefs = stats.get("beliefs", {})
        task_urgent = beliefs.get("task_urgent", 0)
        uncertainty = beliefs.get("uncertainty", 0)

        # Trigger on high urgency or high uncertainty
        max_signal = max(task_urgent, uncertainty)
        if max_signal < 0.4:
            return None

        parts = []
        if task_urgent >= 0.4:
            parts.append(f"task urgent ({task_urgent:.1f})")
        if uncertainty >= 0.4:
            parts.append(f"high uncertainty ({uncertainty:.1f})")

        return WorkspaceContent(
            source_module="gateway_daemon",
            content_type="proactive_signal",
            summary="Proactive: " + ", ".join(parts),
            activation=max_signal,
            salience=max_signal * 0.9,
            payload={"beliefs": beliefs, "pending_messages": stats.get("pending_messages", 0)},
        )

    def _gather_inner_thoughts(self) -> Optional[WorkspaceContent]:
        """Gather latest inner thought if within 30 seconds."""
        from api.services.inner_thoughts_engine import get_inner_thoughts_engine
        engine = get_inner_thoughts_engine()
        recent = engine.get_recent(limit=1)

        if not recent:
            return None

        thought = recent[0]
        age = time.time() - thought.get("timestamp", 0)
        if age > 30:
            return None

        # More recent thoughts are more salient
        recency_factor = max(0.3, 1.0 - age / 30.0)
        return WorkspaceContent(
            source_module="inner_thoughts",
            content_type="inner_thought",
            summary=f"Thinking: {thought.get('content', '')[:80]}",
            activation=recency_factor * 0.7,
            salience=recency_factor,
            payload={"thought_type": thought.get("type"), "content": thought.get("content", "")},
        )

    def _gather_neurodream(self) -> Optional[WorkspaceContent]:
        """Gather dream phase when sleeping."""
        from apprentice_agent.tools.neurodream import get_neurodream
        nd = get_neurodream()
        status = nd.get_status()

        if not status.get("is_sleeping"):
            return None

        phase = status.get("phase", "light")
        phase_salience = {"light": 0.4, "deep": 0.6, "rem": 0.8, "waking": 0.3}
        salience = phase_salience.get(phase, 0.3)

        osc = status.get("oscillation", {})
        osc_value = osc.get("current_value", 0) if osc else 0

        return WorkspaceContent(
            source_module="neurodream",
            content_type="dream_phase",
            summary=f"Dreaming: {phase} phase (oscillation {osc_value:.2f})",
            activation=salience,
            salience=salience,
            payload={
                "phase": phase,
                "oscillation": osc_value,
                "session": status.get("current_session"),
            },
        )

    def _gather_active_inference(self) -> Optional[WorkspaceContent]:
        """Gather belief state when strongly polarized."""
        from apprentice_agent.proactive.gateway_daemon import get_gateway_daemon
        daemon = get_gateway_daemon()

        if not hasattr(daemon, "inference_engine") or daemon.inference_engine is None:
            return None

        beliefs = daemon.inference_engine.get_beliefs()
        if beliefs is None:
            return None

        # Check for strongly polarized beliefs (far from 0.5)
        belief_dict = {
            "user_busy": getattr(beliefs, "user_busy", 0.5),
            "user_receptive": getattr(beliefs, "user_receptive", 0.5),
            "task_urgent": getattr(beliefs, "task_urgent", 0.5),
            "context_stable": getattr(beliefs, "context_stable", 0.5),
        }

        max_polarization = max(abs(v - 0.5) for v in belief_dict.values())
        if max_polarization < 0.25:
            return None

        # Find most polarized belief
        most_polarized = max(belief_dict, key=lambda k: abs(belief_dict[k] - 0.5))
        value = belief_dict[most_polarized]
        direction = "high" if value > 0.5 else "low"

        return WorkspaceContent(
            source_module="active_inference",
            content_type="belief_state",
            summary=f"Belief: {most_polarized} is {direction} ({value:.2f})",
            activation=max_polarization * 2,  # Scale 0-0.5 → 0-1
            salience=max_polarization * 1.5,
            payload={"beliefs": belief_dict, "most_polarized": most_polarized},
        )

    def _gather_strategy_bandit(self) -> Optional[WorkspaceContent]:
        """Gather strategy performance awareness when meaningful spread exists."""
        from apprentice_agent.consciousness.strategy_bandit import get_strategy_bandit, ProblemCategory

        try:
            bandit = get_strategy_bandit()
            if not bandit.enabled:
                return None

            stats = bandit.get_arm_stats()
            if not stats:
                return None

            # Find category with most meaningful spread between best/worst
            best_spread = 0.0
            best_category = None
            best_info = None

            for cat_name, arms in stats.items():
                if not arms:
                    continue
                active_arms = [a for a in arms if a["total_pulls"] > 0]
                if len(active_arms) < 2:
                    continue

                rewards = [a["mean_reward"] for a in active_arms]
                spread = max(rewards) - min(rewards)

                if spread > best_spread:
                    best_spread = spread
                    best_category = cat_name
                    best_arm = max(active_arms, key=lambda a: a["mean_reward"])
                    worst_arm = min(active_arms, key=lambda a: a["mean_reward"])
                    best_info = {
                        "category": cat_name,
                        "best": best_arm["strategy"],
                        "best_reward": best_arm["mean_reward"],
                        "worst": worst_arm["strategy"],
                        "worst_reward": worst_arm["mean_reward"],
                        "spread": round(spread, 3),
                    }

            # Only broadcast when there's a meaningful spread (>0.1)
            if best_spread < 0.1 or best_info is None:
                return None

            return WorkspaceContent(
                source_module="strategy_bandit",
                content_type="performance_insight",
                summary=(
                    f"Strategy insight: {best_info['best']} outperforms "
                    f"{best_info['worst']} for {best_category} "
                    f"(spread={best_info['spread']:.2f})"
                ),
                activation=min(1.0, best_spread * 2),
                salience=min(1.0, best_spread * 1.5),
                payload=best_info,
            )
        except Exception as e:
            logger.debug(f"[GWT] Strategy bandit gather error: {e}")
            return None

    def _gather_world_model(self) -> Optional[WorkspaceContent]:
        """Gather world model state when notable insights exist."""
        from apprentice_agent.consciousness.proactive_awareness import get_proactive_awareness_engine
        engine = get_proactive_awareness_engine()
        pending = engine.get_pending_insights(max_count=1)
        if not pending:
            return None

        top = pending[0]
        urgency = top.get("urgency", 0.5)
        return WorkspaceContent(
            source_module="world_model",
            content_type="proactive_insight",
            summary=top.get("title", "World model insight"),
            activation=urgency,
            salience=min(1.0, urgency * 1.1),
            payload={"insight_type": top.get("insight_type"), "entity": top.get("related_entity_id")},
        )

    # ====================================================================
    # Core Cognitive Cycle
    # ====================================================================

    def _cognitive_cycle(self):
        """Execute one gather → compete → broadcast → decay cycle."""
        cycle_start = time.time()
        self._cycle_number += 1

        # 1. GATHER: collect submissions from all codelets
        submissions: List[tuple] = []  # (codelet_name, content)
        for name, codelet in self._codelets.items():
            content = codelet.compete()
            if content is not None:
                submissions.append((name, content))

        # 2. COMPETE: select winner
        if not submissions:
            self._stats["empty_cycles"] += 1
            # Keep current state but let it age
            with self._lock:
                if self._conscious_state.broadcast_content:
                    age = time.time() - self._conscious_state.broadcast_content.timestamp
                    if age > 5.0:  # Content expires after 5 seconds
                        self._conscious_state.broadcast_content = None
                        self._conscious_state.attention_focus = "idle"
                        self._conscious_state.attention_intensity = 0.0
            self._stats["total_cycles"] += 1
            return

        winner_name, winner_content, runner_ups = self._run_competition(submissions)

        # 3. BROADCAST: update state and emit
        old_source = self._conscious_state.attention_focus
        with self._lock:
            self._conscious_state = ConsciousState(
                broadcast_content=winner_content,
                secondary_content=runner_ups,
                attention_focus=winner_content.source_module,
                attention_intensity=winner_content.effective_activation,
                attention_schema=self._attention_schema.copy(),
                cycle_number=self._cycle_number,
            )

        # Update attention schema
        self._update_attention_schema(winner_name, old_source)

        # Record win/loss
        for name, codelet in self._codelets.items():
            if name == winner_name:
                codelet.record_win()
            elif any(name == sn for sn, _ in submissions):
                codelet.record_loss()

        # Create broadcast event
        cycle_duration_ms = (time.time() - cycle_start) * 1000
        broadcast = BroadcastEvent(
            conscious_state=self._conscious_state,
            cycle_number=self._cycle_number,
            cycle_duration_ms=cycle_duration_ms,
            competing_count=len(submissions),
        )

        with self._lock:
            self._broadcast_history.append(broadcast)

        # Emit to EventBus
        self._emit_broadcast(broadcast)

        # Update stats
        self._stats["total_cycles"] += 1
        self._stats["total_broadcasts"] += 1
        alpha = 0.05  # EMA smoothing
        self._stats["avg_cycle_ms"] = (
            self._stats["avg_cycle_ms"] * (1 - alpha) + cycle_duration_ms * alpha
        )
        self._stats["avg_competing"] = (
            self._stats["avg_competing"] * (1 - alpha) + len(submissions) * alpha
        )

    def _run_competition(self, submissions: List[tuple]) -> tuple:
        """Score submissions and select winner.

        Returns (winner_name, winner_content, runner_ups).
        """
        current_source = self._conscious_state.attention_focus

        scored = []
        for name, content in submissions:
            score = content.effective_activation

            # Attention continuity boost — sustaining attention is easier
            if name == current_source:
                score += self._continuity_boost

            # Gaussian noise for stochasticity
            noise = random.gauss(0, self._noise_temperature)
            noise = max(-self._noise_clip, min(self._noise_clip, noise))
            score += noise

            scored.append((score, name, content))

        # Sort descending
        scored.sort(key=lambda x: x[0], reverse=True)

        winner_score, winner_name, winner_content = scored[0]
        runner_ups = [content for _, _, content in scored[1:3]]  # Max 2

        return winner_name, winner_content, runner_ups

    def _update_attention_schema(self, winner_name: str, old_source: str):
        """Update Attention Schema Theory self-model."""
        now = time.time()
        schema = self._attention_schema

        if winner_name == old_source and old_source != "idle":
            # Sustained attention
            schema["sustained_focus_seconds"] += self._cycle_interval
        else:
            # Attention switch
            schema["sustained_focus_seconds"] = 0.0
            schema["switch_count"] += 1
            schema["last_switch_time"] = now

        schema["current_source"] = winner_name

        # Track dominant sources
        schema["dominant_sources"][winner_name] = (
            schema["dominant_sources"].get(winner_name, 0) + 1
        )

        # Attention trajectory (ring buffer of 20)
        trajectory = schema["attention_trajectory"]
        trajectory.append(winner_name)
        if len(trajectory) > 20:
            schema["attention_trajectory"] = trajectory[-20:]

    def _emit_broadcast(self, broadcast: BroadcastEvent):
        """Publish broadcast to EventBus 'workspace' channel."""
        try:
            import asyncio
            from apprentice_agent.proactive.gateway_daemon import get_gateway_daemon
            daemon = get_gateway_daemon()
            if hasattr(daemon, "event_bus") and daemon.event_bus is not None:
                payload = {
                    "type": "conscious_broadcast",
                    "cycle": broadcast.cycle_number,
                    "source": broadcast.conscious_state.attention_focus,
                    "summary": (
                        broadcast.conscious_state.broadcast_content.summary
                        if broadcast.conscious_state.broadcast_content else ""
                    ),
                    "intensity": broadcast.conscious_state.attention_intensity,
                    "competing_count": broadcast.competing_count,
                    "timestamp": broadcast.timestamp,
                }
                # event_bus.publish() is async — schedule it on the running loop
                try:
                    loop = asyncio.get_running_loop()
                    loop.create_task(daemon.event_bus.publish("workspace", payload))
                except RuntimeError:
                    # No running event loop (sync context) — skip silently
                    pass
        except Exception as e:
            logger.debug(f"[GWT] EventBus emit failed: {e}")

    # ====================================================================
    # Lifecycle
    # ====================================================================

    def start(self):
        """Start the cognitive cycle background daemon."""
        if self._running:
            return
        self._running = True
        self._stats["start_time"] = time.time()
        self._thread = threading.Thread(
            target=self._cycle_loop, daemon=True, name="GWT-CogCycle"
        )
        self._thread.start()
        logger.info("[GWT] Global Workspace Engine started")

    def stop(self):
        """Stop the cognitive cycle daemon."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None
        self._save_state()
        logger.info("[GWT] Global Workspace Engine stopped")

    def _cycle_loop(self):
        """Main loop — runs cognitive cycles with adaptive timing."""
        while self._running:
            try:
                self._cognitive_cycle()
            except Exception as e:
                logger.error(f"[GWT] Cognitive cycle error: {e}")

            # Adaptive timing: speed up when many competitors, slow down when idle
            with self._lock:
                has_content = self._conscious_state.broadcast_content is not None
            if has_content:
                interval = self._cycle_interval
            else:
                interval = self._max_cycle_interval

            time.sleep(max(self._min_cycle_interval, min(self._max_cycle_interval, interval)))

    # ====================================================================
    # Public API
    # ====================================================================

    def get_conscious_state(self) -> ConsciousState:
        """Get current conscious state (thread-safe)."""
        with self._lock:
            return self._conscious_state

    def get_broadcast_history(self, limit: int = 20) -> List[BroadcastEvent]:
        """Get recent broadcast history."""
        with self._lock:
            items = list(self._broadcast_history)
        return items[-limit:]

    def get_attention_schema(self) -> Dict[str, Any]:
        """Get Attention Schema Theory self-model."""
        with self._lock:
            return self._attention_schema.copy()

    def get_codelet_stats(self) -> Dict[str, Any]:
        """Get per-codelet competition statistics."""
        return {name: c.get_stats() for name, c in self._codelets.items()}

    def get_stats(self) -> Dict[str, Any]:
        """Get engine-level statistics."""
        with self._lock:
            stats = self._stats.copy()
        stats["running"] = self._running
        stats["cycle_number"] = self._cycle_number
        stats["codelet_count"] = len(self._codelets)
        stats["history_size"] = len(self._broadcast_history)
        if stats["start_time"]:
            stats["uptime_seconds"] = round(time.time() - stats["start_time"], 1)
        return stats

    def get_cognitive_load_contribution(self) -> float:
        """Return workspace contribution to overall cognitive load (0-1)."""
        with self._lock:
            state = self._conscious_state
        if not state.broadcast_content:
            return 0.05  # Minimal idle load
        return min(1.0, state.attention_intensity * 0.8 + 0.1)

    def get_broadcasts_for_consolidation(self, since: float = 0) -> List[Dict]:
        """Get broadcast summaries for NeuroDream memory consolidation."""
        with self._lock:
            items = list(self._broadcast_history)
        results = []
        for b in items:
            if b.timestamp >= since and b.conscious_state.broadcast_content:
                bc = b.conscious_state.broadcast_content
                results.append({
                    "source": bc.source_module,
                    "type": bc.content_type,
                    "summary": bc.summary,
                    "activation": bc.effective_activation,
                    "timestamp": b.timestamp,
                })
        return results

    def inject_content(self, content: WorkspaceContent):
        """Inject content directly for testing — bypasses competition."""
        with self._lock:
            self._conscious_state = ConsciousState(
                broadcast_content=content,
                attention_focus=content.source_module,
                attention_intensity=content.effective_activation,
                attention_schema=self._attention_schema.copy(),
                cycle_number=self._cycle_number,
            )
            self._broadcast_history.append(BroadcastEvent(
                conscious_state=self._conscious_state,
                cycle_number=self._cycle_number,
                cycle_duration_ms=0.0,
                competing_count=0,
            ))
        logger.info(f"[GWT] Injected content from {content.source_module}: {content.summary}")

    # ====================================================================
    # Persistence
    # ====================================================================

    def _save_state(self):
        """Persist engine state to JSON."""
        state_path = os.path.join(self._data_dir, "workspace_state.json")
        try:
            data = {
                "stats": self._stats,
                "attention_schema": self._attention_schema,
                "codelet_stats": self.get_codelet_stats(),
                "saved_at": time.time(),
            }
            with open(state_path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"[GWT] Failed to save state: {e}")

    def _load_state(self):
        """Load persisted engine state."""
        state_path = os.path.join(self._data_dir, "workspace_state.json")
        if not os.path.exists(state_path):
            return
        try:
            with open(state_path) as f:
                data = json.load(f)

            # Restore attention schema
            saved_schema = data.get("attention_schema", {})
            self._attention_schema["switch_count"] = saved_schema.get("switch_count", 0)
            self._attention_schema["dominant_sources"] = saved_schema.get("dominant_sources", {})

            # Restore codelet win counts
            saved_codelets = data.get("codelet_stats", {})
            for name, stats in saved_codelets.items():
                if name in self._codelets:
                    self._codelets[name]._total_wins = stats.get("total_wins", 0)
                    self._codelets[name]._total_submissions = stats.get("total_submissions", 0)

            logger.info("[GWT] Restored persisted workspace state")
        except Exception as e:
            logger.warning(f"[GWT] Failed to load state: {e}")


# ============================================================================
# Singleton
# ============================================================================

_instance: Optional[GlobalWorkspaceEngine] = None
_instance_lock = threading.Lock()


def get_global_workspace() -> GlobalWorkspaceEngine:
    """Get or create the GlobalWorkspaceEngine singleton (double-checked locking)."""
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = GlobalWorkspaceEngine()
    return _instance
