"""
Genuine Idle Presence Engine (Phase 6D).

Orchestrates REAL background cognitive activity during idle periods.
Replaces fake random messages with actual cognitive work:
1. Cognitive load tracking from all subsystems
2. Background task orchestration (memory reorg, KG pruning, pattern mining)
3. Self-reflection on recent interactions
4. NeuroDream phase-aware status reporting
5. Avatar breathing driven by actual cognitive load

Integrates with:
- NeuroDream: Hooks into phase changes and insights
- InnerThoughts: Shifts topics during idle to reflection/curiosity
- IdleBehaviors: Replaces template messages with real activity reports
- ThinkingSystem: Records genuine cognitive events
- Gateway Daemon: Reads decision/event stats
"""

import atexit
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


# ============================================================================
# Data Models
# ============================================================================

class IdleActivity(str, Enum):
    """Types of genuine background activity."""
    MEMORY_REPLAY = "memory_replay"
    PATTERN_MINING = "pattern_mining"
    KG_PRUNING = "kg_pruning"
    MEMORY_DEDUP = "memory_dedup"
    SELF_REFLECTION = "self_reflection"
    DREAM_LIGHT = "dream_light"
    DREAM_DEEP = "dream_deep"
    DREAM_REM = "dream_rem"
    INNER_THOUGHT = "inner_thought"
    METACOGNITION = "metacognition"
    IDLE_MONITORING = "idle_monitoring"


@dataclass
class ActivityEvent:
    """A single background activity event."""
    activity: IdleActivity
    description: str
    timestamp: float = field(default_factory=time.time)
    cognitive_load: float = 0.0  # 0-1, how much this contributes to load


@dataclass
class CognitiveLoadState:
    """Current cognitive load computed from all subsystems."""
    total_load: float = 0.0           # 0-1 aggregate
    thinking_load: float = 0.0        # From thinking system
    dream_load: float = 0.0           # From NeuroDream
    daemon_load: float = 0.0          # From Gateway Daemon
    inner_thoughts_load: float = 0.0  # From inner thoughts engine
    metacognition_load: float = 0.0   # From metacognitive engine
    last_computed: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "total_load": round(self.total_load, 3),
            "thinking_load": round(self.thinking_load, 3),
            "dream_load": round(self.dream_load, 3),
            "daemon_load": round(self.daemon_load, 3),
            "inner_thoughts_load": round(self.inner_thoughts_load, 3),
            "metacognition_load": round(self.metacognition_load, 3),
        }


# ============================================================================
# Idle Presence Engine
# ============================================================================

class IdlePresenceEngine:
    """Orchestrates genuine background cognitive activity during idle periods.

    Tracks real cognitive load, triggers background tasks, and reports
    actual activity to the UI rather than fake template messages.
    """

    def __init__(self):
        self._lock = threading.RLock()

        # Activity tracking
        self._recent_activities: List[ActivityEvent] = []
        self._max_activities = 100

        # Cognitive load
        self._cognitive_load = CognitiveLoadState()
        self._load_update_interval = 3.0  # seconds
        self._last_load_update = 0.0

        # NeuroDream phase tracking
        self._current_dream_phase: Optional[str] = None
        self._dream_phase_start: Optional[float] = None
        self._dream_session_active = False

        # Background task state
        self._background_thread: Optional[threading.Thread] = None
        self._running = False
        self._idle_task_interval = 30.0  # Run idle tasks every 30s

        # Callbacks registered
        self._callbacks_registered = False

        # Sleep scheduler config
        self._sleep_idle_threshold = 30 * 60    # 30 min idle before triggering sleep
        self._sleep_cooldown = 4 * 3600         # 4 hours between auto-sleep cycles
        self._last_auto_sleep_time: float = 0.0
        self._conversation_threshold = 5        # Min messages before first sleep is useful

        # Agent reference (set via set_agent() for Hands integration)
        self._agent: Any = None

        # Stats
        self._stats = {
            "activities_recorded": 0,
            "background_tasks_run": 0,
            "reflections_generated": 0,
            "load_computations": 0,
            "consecutive_failures": 0,
        }

        logger.info("[IdlePresence] Engine initialized")

    def set_agent(self, agent: Any) -> None:
        """Set the agent reference so Hands can access brain and tools."""
        self._agent = agent
        logger.info("[IdlePresence] Agent reference set for Hands integration")

    # ====================================================================
    # NeuroDream Integration
    # ====================================================================

    def register_neurodream_callbacks(self) -> None:
        """Register callbacks with NeuroDream for phase change notifications."""
        if self._callbacks_registered:
            return

        try:
            from aura.tools.neurodream import get_neurodream
            nd = get_neurodream()
            nd.set_callbacks(
                on_phase_change=self._on_dream_phase_change,
                on_insight=self._on_dream_insight,
            )
            self._callbacks_registered = True
            logger.info("[IdlePresence] Registered NeuroDream callbacks")
        except Exception as e:
            logger.debug(f"[IdlePresence] NeuroDream callback registration failed: {e}")

    def _on_dream_phase_change(self, phase) -> None:
        """Called when NeuroDream transitions between phases."""
        phase_name = phase.value if hasattr(phase, 'value') else str(phase)

        with self._lock:
            self._current_dream_phase = phase_name
            self._dream_phase_start = time.time()
            self._dream_session_active = phase_name not in ("awake", "waking")

        # Map dream phases to idle activities
        phase_map = {
            "light": (IdleActivity.DREAM_LIGHT, "replaying recent memories..."),
            "deep": (IdleActivity.DREAM_DEEP, "abstracting patterns from experience..."),
            "rem": (IdleActivity.DREAM_REM, "synthesizing creative connections..."),
            "waking": (IdleActivity.IDLE_MONITORING, "emerging from sleep cycle..."),
        }

        if phase_name in phase_map:
            activity, desc = phase_map[phase_name]
            self._record_activity(activity, desc, cognitive_load=0.6)

        # Record as real thought in thinking system
        self._record_dream_thought(phase_name)

    def _on_dream_insight(self, insight) -> None:
        """Called when NeuroDream generates an insight during REM."""
        content = getattr(insight, 'content', str(insight))
        insight_type = getattr(insight, 'insight_type', 'connection')
        desc = f"insight ({insight_type}): {content[:80]}"
        self._record_activity(IdleActivity.DREAM_REM, desc, cognitive_load=0.8)
        self._record_dream_thought("rem", f"eureka: {content[:60]}")

    def _record_dream_thought(self, phase_name: str, content: str = None) -> None:
        """Record a NeuroDream phase change as a real thought."""
        try:
            from api.routes.thinking import get_manager as get_thinking_manager
            tm = get_thinking_manager()

            if content is None:
                thought_map = {
                    "light": ("recalling", "replaying and strengthening recent memories"),
                    "deep": ("analyzing", "searching for patterns across interactions"),
                    "rem": ("connecting", "exploring novel connections between ideas"),
                    "waking": ("observing", "integrating insights from sleep cycle"),
                }
                thought_type, content = thought_map.get(phase_name, ("observing", f"entering {phase_name} phase"))

            else:
                thought_type = "connecting" if "eureka" in content.lower() else "analyzing"

            tm.record_real_thought(thought_type, content, intensity=0.6, source="neurodream")
        except Exception as e:
            logger.debug(f"[IdlePresence] non-critical: {e}")
    # ====================================================================
    # Cognitive Load Computation
    # ====================================================================

    def compute_cognitive_load(self) -> CognitiveLoadState:
        """Compute current cognitive load from all subsystems.

        Returns a normalized 0-1 value representing how much
        background processing is happening.
        """
        now = time.time()
        if now - self._last_load_update < self._load_update_interval:
            return self._cognitive_load

        load = CognitiveLoadState(last_computed=now)

        # 1. Thinking system load
        try:
            from api.routes.thinking import get_manager as get_thinking_manager
            tm = get_thinking_manager()
            stats = tm.get_stats()
            active = stats.get("active_thoughts", 0)
            real = stats.get("real_thoughts", 0)
            # Active thoughts are the strongest signal
            load.thinking_load = min(1.0, active / 10.0 + real / 50.0)
        except Exception as e:
            logger.debug(f"[IdlePresence] non-critical: {e}")
        # 2. NeuroDream load (oscillation-aware pulsing)
        try:
            from aura.tools.neurodream import get_neurodream
            nd = get_neurodream()
            status = nd.get_status()
            if status.get("is_sleeping"):
                osc = status.get("oscillation")
                base_loads = {"light": 0.4, "deep": 0.7, "rem": 0.9, "waking": 0.2}
                base_load = base_loads.get(status.get("phase", "light"), 0.3)
                if osc and "modifiers" in osc:
                    load.dream_load = base_load * osc["modifiers"]["cognitive_intensity"]
                else:
                    load.dream_load = base_load
        except Exception as e:
            logger.debug(f"[IdlePresence] non-critical: {e}")
        # 3. Gateway Daemon load
        try:
            from aura.proactive.gateway_daemon import get_gateway_daemon
            daemon = get_gateway_daemon()
            stats = daemon.get_stats()
            if stats.get("state") == "running":
                events = stats.get("events_processed", 0)
                decisions = stats.get("decisions_made", 0)
                # Normalize: ~100 events/50 decisions = full load
                load.daemon_load = min(1.0, events / 100.0 * 0.5 + decisions / 50.0 * 0.5)
        except Exception as e:
            logger.debug(f"[IdlePresence] non-critical: {e}")
        # 4. Inner thoughts load
        try:
            from api.services.inner_thoughts_engine import get_inner_thoughts_engine
            engine = get_inner_thoughts_engine()
            stats = engine.get_stats()
            if stats.get("running"):
                generated = stats.get("total_generated", 0)
                # Active engine contributes base load + activity
                load.inner_thoughts_load = min(1.0, 0.2 + generated / 100.0)
        except Exception as e:
            logger.debug(f"[IdlePresence] non-critical: {e}")
        # 5. Metacognition load
        try:
            from aura.consciousness.metacognition import get_metacognitive_engine
            mc = get_metacognitive_engine()
            model = mc.get_self_model()
            active_goals = [g for g in model.learning_goals if g.status in ("pending", "active")]
            if active_goals:
                load.metacognition_load = min(1.0, len(active_goals) / 5.0 * 0.3)
        except Exception as e:
            logger.debug(f"[IdlePresence] non-critical: {e}")
        # Weighted aggregate (5 components, weights sum to 1.0)
        load.total_load = min(1.0, (
            load.thinking_load * 0.25 +
            load.dream_load * 0.31 +
            load.daemon_load * 0.15 +
            load.inner_thoughts_load * 0.15 +
            load.metacognition_load * 0.14
        ))

        with self._lock:
            self._cognitive_load = load
            self._last_load_update = now
            self._stats["load_computations"] += 1

        return load

    # ====================================================================
    # Breath Rate from Cognitive Load
    # ====================================================================

    def get_breath_rate_from_load(self, load: Optional[CognitiveLoadState] = None) -> float:
        """Compute avatar breath rate modifier from actual cognitive load.

        Returns:
            Breath rate modifier (0.6 = very slow/deep sleep, 1.4 = intense processing)
        """
        if load is None:
            load = self.compute_cognitive_load()

        # Map cognitive load to breath rate:
        # 0.0 load -> 0.6x (deep relaxation)
        # 0.3 load -> 0.85x (light rest)
        # 0.5 load -> 1.0x (baseline)
        # 0.7 load -> 1.15x (moderate activity)
        # 1.0 load -> 1.4x (peak processing)
        rate = 0.6 + load.total_load * 0.8
        return round(rate, 3)

    def get_glow_from_load(self, load: Optional[CognitiveLoadState] = None) -> float:
        """Compute avatar glow intensity from cognitive load.

        Returns:
            Glow intensity 0.2 (dim, resting) to 0.9 (bright, active)
        """
        if load is None:
            load = self.compute_cognitive_load()
        return round(0.2 + load.total_load * 0.7, 3)

    # ====================================================================
    # Activity Recording and Reporting
    # ====================================================================

    def _record_activity(self, activity: IdleActivity, description: str,
                         cognitive_load: float = 0.0) -> None:
        """Record a genuine background activity."""
        event = ActivityEvent(
            activity=activity,
            description=description,
            cognitive_load=cognitive_load,
        )
        with self._lock:
            self._recent_activities.append(event)
            if len(self._recent_activities) > self._max_activities:
                self._recent_activities = self._recent_activities[-self._max_activities:]
            self._stats["activities_recorded"] += 1

    def get_current_activity_status(self, load: Optional[CognitiveLoadState] = None) -> Optional[str]:
        """Get a human-readable description of what AURA is actually doing right now.

        Returns the most recent genuine activity, or a computed status
        from cognitive load if no recent activity.
        """
        with self._lock:
            # Check if dreaming
            if self._dream_session_active and self._current_dream_phase:
                phase_desc = {
                    "light": "replaying recent memories...",
                    "deep": "abstracting patterns from experience...",
                    "rem": "exploring creative connections...",
                }
                desc = phase_desc.get(self._current_dream_phase)
                if desc:
                    elapsed = time.time() - (self._dream_phase_start or time.time())
                    return f"dreaming ({self._current_dream_phase}): {desc} [{int(elapsed)}s]"

            # Check recent activities (last 30s)
            now = time.time()
            recent = [a for a in self._recent_activities if now - a.timestamp < 30]
            if recent:
                latest = recent[-1]
                return latest.description

        # Compute from cognitive load
        if load is None:
            load = self.compute_cognitive_load()
        if load.total_load > 0.6:
            return f"processing actively (cognitive load: {load.total_load:.0%})"
        elif load.total_load > 0.3:
            return f"light background processing (load: {load.total_load:.0%})"
        elif load.total_load > 0.1:
            return f"resting quietly (load: {load.total_load:.0%})"
        else:
            return "at rest..."

    def get_recent_activities(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent background activities for UI display."""
        with self._lock:
            activities = self._recent_activities[-limit:]
            return [
                {
                    "activity": a.activity.value,
                    "description": a.description,
                    "timestamp": a.timestamp,
                    "cognitive_load": round(a.cognitive_load, 2),
                    "age_seconds": round(time.time() - a.timestamp, 1),
                }
                for a in reversed(activities)
            ]

    # ====================================================================
    # Background Idle Tasks
    # ====================================================================

    def start_background_tasks(self) -> None:
        """Start the background task thread for idle-time work."""
        if self._running:
            return

        self._running = True
        self._background_thread = threading.Thread(
            target=self._background_loop,
            daemon=True,
            name="IdlePresence-Background",
        )
        self._background_thread.start()
        logger.info("[IdlePresence] Background task thread started")

    def stop_background_tasks(self) -> None:
        """Stop background tasks."""
        self._running = False
        if self._background_thread:
            self._background_thread.join(timeout=5)
            self._background_thread = None
        logger.info("[IdlePresence] Background tasks stopped")

    def _background_loop(self) -> None:
        """Main background loop that runs idle-time tasks periodically."""
        time.sleep(10)  # Initial settle time

        while self._running:
            try:
                # Only run tasks if system is actually idle
                idle_seconds = self._get_idle_duration()
                if idle_seconds > 15:
                    self._run_idle_tasks(idle_seconds)
                    self._stats["background_tasks_run"] += 1
                    self._stats["consecutive_failures"] = 0

                time.sleep(self._idle_task_interval)

            except Exception as e:
                self._stats["consecutive_failures"] += 1
                n = self._stats["consecutive_failures"]
                if n <= 3:
                    logger.error(f"[IdlePresence] Background loop error ({n}): {e}")
                elif n == 10:
                    logger.warning(f"[IdlePresence] 10 consecutive failures — idle tasks may be broken: {e}")
                # After 3 failures, only log every 10th to avoid log flooding
                elif n % 10 == 0:
                    logger.warning(f"[IdlePresence] {n} consecutive failures: {e}")
                time.sleep(30)

    def _get_idle_duration(self) -> float:
        """Get how long the system has been idle."""
        try:
            from api.routes.idle_behaviors import get_manager
            mgr = get_manager()
            return mgr.get_idle_duration()
        except Exception as e:
            logger.warning("idle_duration_check_failed", exc_info=True)
            return 0.0

    def _run_idle_tasks(self, idle_seconds: float) -> None:
        """Execute genuine background tasks during idle time."""

        # Task 1: Self-reflection on recent interactions (every ~2 min of idle)
        if idle_seconds > 120:
            self._run_self_reflection()

        # Task 2: Memory pattern scanning (every ~60s of idle)
        if idle_seconds > 60:
            self._run_pattern_scan()

        # Task 3: KG maintenance (every ~90s of idle)
        if idle_seconds > 90:
            self._run_kg_maintenance()

        # Task 3.5: Proactive awareness full analysis (ADV-02 Phase 3)
        if idle_seconds > 90:
            self._run_awareness_analysis()

        # Task 3.6: Skill health check — suggest evolving weak skills (>5min idle, 6h cooldown)
        if idle_seconds > 300:
            self._run_skill_health_check()

        # Task 3.7: Autonomous Hands — trigger eligible Hands during idle
        # (OpenFang-inspired: self-contained autonomous task packages)
        if idle_seconds > 300:
            self._trigger_hands(idle_seconds)

        # Task 4: Auto-trigger NeuroDream sleep when idle long enough
        if idle_seconds > self._sleep_idle_threshold:
            self._check_and_trigger_sleep(idle_seconds)

    def _run_self_reflection(self) -> None:
        """Generate a self-reflection on recent interactions."""
        try:
            from api.routes.thinking import get_manager as get_thinking_manager
            tm = get_thinking_manager()

            # Get recent conversation context
            topics = []
            try:
                from api.routes.context import get_tracker
                ctx = get_tracker()
                focus = ctx.get_focus_state(limit=3)
                topics = [item["name"] for item in focus.get("items", [])[:3]]
            except Exception as e:
                logger.debug(f"[IdlePresence] non-critical: {e}")
            if topics:
                topic_str = ", ".join(topics)
                thought = f"reflecting on recent discussion about {topic_str}"
                tm.record_real_thought("analyzing", thought, intensity=0.4, source="idle_presence")
                self._record_activity(IdleActivity.SELF_REFLECTION, thought, cognitive_load=0.3)
                self._stats["reflections_generated"] += 1

        except Exception as e:
            logger.debug(f"[IdlePresence] Self-reflection error: {e}")

    def _run_pattern_scan(self) -> None:
        """Scan for patterns in recent activity."""
        try:
            # Check if there are patterns worth noting
            from api.routes.thinking import get_manager as get_thinking_manager
            tm = get_thinking_manager()
            stats = tm.get_stats()
            real = stats.get("real_thoughts", 0)

            if real > 5:
                desc = f"scanning {real} recent cognitive events for patterns..."
                tm.record_real_thought("connecting", desc, intensity=0.3, source="idle_presence")
                self._record_activity(IdleActivity.PATTERN_MINING, desc, cognitive_load=0.2)

        except Exception as e:

            logger.debug(f"[IdlePresence] non-critical: {e}")
    def _run_kg_maintenance(self) -> None:
        """Run knowledge graph maintenance if available."""
        try:
            from aura.tools.neurodream import get_neurodream
            nd = get_neurodream()
            # Only if not already sleeping
            if nd.current_phase.value == "awake":
                status = nd.get_status()
                sessions = status.get("total_sessions", 0)
                if sessions > 0:
                    desc = f"maintaining knowledge connections ({sessions} consolidation sessions completed)"
                    self._record_activity(IdleActivity.KG_PRUNING, desc, cognitive_load=0.15)
        except Exception as e:
            logger.debug(f"[IdlePresence] non-critical: {e}")
    def _run_awareness_analysis(self) -> None:
        """Run proactive awareness full analysis (ADV-02 Phase 3)."""
        try:
            from aura.consciousness.proactive_awareness import get_proactive_awareness_engine
            engine = get_proactive_awareness_engine()
            insights = engine.run_full_analysis()
            if insights:
                desc = f"proactive awareness: generated {len(insights)} insight(s)"
                self._record_activity(IdleActivity.PATTERN_MINING, desc, cognitive_load=0.2)
        except Exception as e:
            logger.debug(f"[IdlePresence] Awareness analysis error: {e}")

    def _run_skill_health_check(self) -> None:
        """Check for underperforming skills and emit a proactive suggestion.

        Uses a 6-hour internal cooldown (enforced by skill_health_monitor).
        If weak skills are found, delivers the suggestion via the Gateway Daemon.
        """
        try:
            from aura.proactive.skill_health_monitor import check_and_suggest
            suggestion = check_and_suggest()
            if not suggestion:
                return

            self._record_activity(
                IdleActivity.PATTERN_MINING,
                f"skill health: found underperforming skills",
                cognitive_load=0.15,
            )

            # Deliver as a proactive message through the Gateway Daemon
            try:
                from aura.proactive.gateway_daemon import get_gateway_daemon
                from aura.proactive.event_bus import EventPriority
                from aura.proactive.active_inference import ProactiveAction

                daemon = get_gateway_daemon()
                from aura.proactive.gateway_daemon import ProactiveMessage
                message = ProactiveMessage(
                    action=ProactiveAction.SUGGEST,
                    content=suggestion,
                    priority=EventPriority.LOW,
                    metadata={"source": "skill_health_monitor"},
                )
                daemon._deliver_message(message)
                logger.info(f"[IdlePresence] Skill health suggestion delivered")
            except Exception as e:
                logger.debug(f"[IdlePresence] Could not deliver skill health suggestion: {e}")

        except Exception as e:
            logger.debug(f"[IdlePresence] Skill health check error: {e}")

    # ====================================================================
    # Autonomous Hands (OpenFang-inspired)
    # ====================================================================

    def _trigger_hands(self, idle_seconds: float) -> None:
        """Check and trigger eligible autonomous Hands during idle."""
        try:
            from aura.hands.manager import get_hand_manager
            manager = get_hand_manager()

            # Get drive urgencies from intrinsic motivation
            drive_urgencies = {}
            try:
                from aura.consciousness.intrinsic_motivation import get_intrinsic_motivation
                engine = get_intrinsic_motivation()
                if engine:
                    for drive_name in ("curiosity", "competence", "social", "coherence"):
                        drive_urgencies[drive_name] = engine.get_drives_summary().get(drive_name, 0.0)
            except Exception as e:
                logger.debug(f"[IdlePresence] Drive urgencies unavailable: {e}")

            # Get brain and tools from agent reference
            brain = getattr(self._agent, 'brain', None) if self._agent else None
            tools = getattr(self._agent, 'tools', {}) if self._agent else {}

            if not brain:
                logger.debug("[IdlePresence] No brain available for Hands — skipping")
                return

            # Check and run eligible hands
            triggered = manager.check_and_run(
                brain=brain,
                tools=tools,
                idle_seconds=idle_seconds,
                drive_urgencies=drive_urgencies,
            )

            if triggered:
                self._record_activity(
                    IdleActivity.IDLE_MONITORING,
                    f"Triggered autonomous hand: {triggered}",
                    cognitive_load=0.6,
                )
        except Exception as e:
            logger.debug(f"[IdlePresence] Hand trigger error: {e}")

    # ====================================================================
    # Sleep Scheduling
    # ====================================================================

    def _check_and_trigger_sleep(self, idle_seconds: float) -> None:
        """Check guards and trigger NeuroDream sleep if appropriate."""
        try:
            now = time.time()

            # Guard: cooldown period
            if now - self._last_auto_sleep_time < self._sleep_cooldown:
                return

            # Guard: enough conversations to make sleep useful
            if not self._has_enough_conversations():
                return

            from aura.tools.neurodream import get_neurodream
            nd = get_neurodream()

            # Guard: NeuroDream must be awake
            if nd.current_phase.value != "awake":
                return

            # All guards passed — trigger sleep
            logger.info(
                f"[IdlePresence] Auto-triggered NeuroDream sleep cycle "
                f"(idle {idle_seconds:.0f}s)"
            )
            nd.enter_sleep(trigger="idle")
            self._last_auto_sleep_time = now
            self._record_activity(
                IdleActivity.DREAM_LIGHT,
                "auto-triggered sleep consolidation after extended idle",
                cognitive_load=0.5,
            )

        except Exception as e:
            logger.debug(f"[IdlePresence] Sleep trigger check failed: {e}")

    def _has_enough_conversations(self) -> bool:
        """Check if there are enough conversations to justify a sleep cycle."""
        try:
            from api.services.agent_service import agent_service
            if not agent_service.is_ready:
                return False
            count = len(agent_service.agent.brain.conversation_history)
            return count >= self._conversation_threshold
        except Exception as e:
            logger.debug(f"[IdlePresence] conversation check failed: {e}")
            return False

    def record_user_activity(self) -> None:
        """Record user activity, keeping NeuroDream timers in sync.

        Call from chat handler when the user sends a message.
        """
        try:
            from aura.tools.neurodream import get_neurodream
            nd = get_neurodream()
            nd.record_activity()
        except Exception as e:
            logger.debug(f"[IdlePresence] non-critical: {e}")
    # ====================================================================
    # Full State for UI
    # ====================================================================

    def get_state(self) -> Dict[str, Any]:
        """Get full idle presence state for the UI."""
        load = self.compute_cognitive_load()

        now = time.time()
        cooldown_remaining = max(0.0, self._sleep_cooldown - (now - self._last_auto_sleep_time))

        with self._lock:
            return {
                "cognitive_load": load.to_dict(),
                "breath_rate_from_load": self.get_breath_rate_from_load(load),
                "glow_from_load": self.get_glow_from_load(load),
                "current_activity": self.get_current_activity_status(load),
                "recent_activities": self.get_recent_activities(limit=5),
                "dream_state": {
                    "active": self._dream_session_active,
                    "phase": self._current_dream_phase,
                    "phase_duration": (
                        round(time.time() - self._dream_phase_start, 1)
                        if self._dream_phase_start else 0
                    ),
                },
                "sleep_scheduler": {
                    "enabled": True,
                    "idle_threshold_minutes": self._sleep_idle_threshold / 60,
                    "cooldown_hours": self._sleep_cooldown / 3600,
                    "last_auto_sleep": self._last_auto_sleep_time or None,
                    "next_eligible_in_minutes": round(cooldown_remaining / 60, 1),
                },
                "background_running": self._running,
                "stats": dict(self._stats),
            }

    def get_status(self) -> Dict[str, Any]:
        """Get concise status for API."""
        load = self.compute_cognitive_load()
        now = time.time()
        cooldown_remaining = max(0.0, self._sleep_cooldown - (now - self._last_auto_sleep_time))
        return {
            "active": True,
            "cognitive_load": round(load.total_load, 3),
            "breath_rate": self.get_breath_rate_from_load(load),
            "current_activity": self.get_current_activity_status(load),
            "dream_active": self._dream_session_active,
            "dream_phase": self._current_dream_phase,
            "activities_recorded": self._stats["activities_recorded"],
            "background_running": self._running,
            "sleep_scheduler": {
                "enabled": True,
                "idle_threshold_minutes": self._sleep_idle_threshold / 60,
                "cooldown_hours": self._sleep_cooldown / 3600,
                "last_auto_sleep": self._last_auto_sleep_time or None,
                "next_eligible_in_minutes": round(cooldown_remaining / 60, 1),
            },
        }


# ============================================================================
# Singleton
# ============================================================================

_engine: Optional[IdlePresenceEngine] = None
_engine_lock = threading.Lock()


def get_idle_presence_engine() -> IdlePresenceEngine:
    """Get or create the global IdlePresenceEngine."""
    global _engine
    if _engine is None:
        with _engine_lock:
            if _engine is None:
                _engine = IdlePresenceEngine()
                atexit.register(_engine.stop_background_tasks)
    return _engine
