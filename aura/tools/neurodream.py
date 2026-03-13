"""NeuroDream - Sleep/Dream Memory Consolidation System for Aura.

Implements a biologically-inspired sleep system with phases:
- Light Sleep: Recent memory replay (last 24h)
- Deep Sleep: Pattern abstraction and compression
- REM Sleep: Creative synthesis and novel connections

Based on research showing 38% reduction in catastrophic forgetting
and 17.6% increase in zero-shot transfer through latent replay synthesis.
"""

import json
import logging
import math
import random
import threading
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from collections import Counter, defaultdict
import re

from aura.jsonl_utils import rotate_jsonl_if_needed

logger = logging.getLogger(__name__)


class SleepPhase(Enum):
    """Sleep cycle phases."""
    AWAKE = "awake"
    LIGHT = "light"      # Recent memory replay
    DEEP = "deep"        # Pattern abstraction
    REM = "rem"          # Creative synthesis
    WAKING = "waking"    # Transitioning to awake


class DreamTrigger(Enum):
    """What triggered the sleep cycle."""
    SCHEDULED = "scheduled"
    IDLE = "idle"
    MANUAL = "manual"
    LOW_RESOURCES = "low_resources"


@dataclass
class DreamInsight:
    """A novel insight generated during REM sleep."""
    id: str
    timestamp: str
    insight_type: str  # connection, pattern, hypothesis, prediction
    content: str
    confidence: float
    source_nodes: List[str]
    created_edges: List[Dict[str, str]]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SleepSession:
    """Record of a complete sleep cycle."""
    session_id: str
    start_time: str
    end_time: Optional[str]
    trigger: str
    phases_completed: List[str]

    # Light phase metrics
    memories_replayed: int = 0
    memories_strengthened: int = 0

    # Deep phase metrics
    patterns_found: int = 0
    edges_pruned: int = 0
    edges_strengthened: int = 0
    nodes_merged: int = 0

    # REM phase metrics
    insights_generated: int = 0
    novel_connections: int = 0
    creative_hypotheses: int = 0

    # Learned context (Phase 4D)
    learned_context_generated: bool = False
    learned_context_version: int = 0

    # Oscillation info
    oscillation_band: str = ""  # Dominant frequency band used

    # Overall
    duration_seconds: float = 0
    interrupted: bool = False
    interrupt_reason: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class NeuralOscillator:
    """Neural oscillation state for sleep phases (DONN-inspired)."""
    frequency: float = 0.0      # Hz (dominant frequency)
    amplitude: float = 0.0      # 0-1 strength
    phase_angle: float = 0.0    # radians
    band: str = "none"          # delta/theta/alpha/beta
    start_time: float = field(default_factory=time.time)
    last_tick: float = field(default_factory=time.time)

    # Frequency band mapping: phase -> (frequency_hz, band_name)
    BAND_MAP: Dict[str, Tuple[float, str]] = field(default_factory=lambda: {
        "awake": (15.0, "beta"),
        "light": (10.0, "alpha"),
        "deep": (2.0, "delta"),
        "rem": (6.0, "theta"),
        "waking": (8.0, "alpha"),
    }, repr=False)


@dataclass
class ConsolidatedPattern:
    """An abstracted pattern from deep sleep."""
    pattern_id: str
    timestamp: str
    pattern_type: str  # temporal, topical, emotional, behavioral
    description: str
    frequency: int
    confidence: float
    examples: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class LearnedContext:
    """Letta-style learned context distilled from conversations (Phase 4D)."""
    generated_at: str
    session_id: str

    # Distilled knowledge
    user_summary: str = ""          # Who the user is, what they care about
    key_facts: List[str] = field(default_factory=list)      # Important facts learned
    preferences: Dict[str, str] = field(default_factory=dict)  # User preferences
    principles: List[str] = field(default_factory=list)     # Interaction principles
    ongoing_topics: List[str] = field(default_factory=list)  # Current interests/projects
    emotional_patterns: str = ""    # How the user tends to feel / what comforts them

    # Metadata
    conversations_processed: int = 0
    messages_analyzed: int = 0
    version: int = 1

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> 'LearnedContext':
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def to_system_prompt(self) -> str:
        """Format learned context for injection into system prompts."""
        parts = []
        if self.user_summary:
            parts.append(f"About the user: {self.user_summary}")
        if self.key_facts:
            facts_str = "; ".join(self.key_facts[:10])
            parts.append(f"Key facts: {facts_str}")
        if self.preferences:
            prefs = "; ".join(f"{k}: {v}" for k, v in list(self.preferences.items())[:8])
            parts.append(f"User preferences: {prefs}")
        if self.ongoing_topics:
            topics = ", ".join(self.ongoing_topics[:5])
            parts.append(f"Current interests/projects: {topics}")
        if self.principles:
            princ = "; ".join(self.principles[:5])
            parts.append(f"Interaction principles: {princ}")
        if self.emotional_patterns:
            parts.append(f"Emotional notes: {self.emotional_patterns}")

        if not parts:
            return ""
        return "[Learned Context]\n" + "\n".join(parts)


class NeuroDreamEngine:
    """Sleep/dream memory consolidation engine for Aura.

    Runs during idle periods to:
    - Replay and strengthen recent memories
    - Find and abstract patterns
    - Generate creative connections
    - Consolidate emotional experiences
    - Generate Letta-style learned context (Phase 4D)
    """

    def __init__(
        self,
        knowledge_graph=None,
        hybrid_memory=None,
        evoemo=None,
        inner_monologue=None,
        chromadb=None,
        brain=None,
        data_dir: str = "data/neurodream",
        idle_threshold_minutes: int = 30,
        max_vram_gb: float = 4.0
    ):
        """Initialize NeuroDream engine.

        Args:
            knowledge_graph: KnowledgeGraphTool instance
            hybrid_memory: HybridMemory instance
            evoemo: EvoEmoTool instance
            inner_monologue: InnerMonologueTool instance
            chromadb: ChromaDB collection for memories
            brain: Brain instance for LLM calls and conversation access
            data_dir: Directory for dream data
            idle_threshold_minutes: Minutes of inactivity before auto-sleep
            max_vram_gb: Maximum VRAM to use during sleep
        """
        self.kg = knowledge_graph
        self.memory = hybrid_memory
        self.evoemo = evoemo
        self.monologue = inner_monologue
        self.chromadb = chromadb
        self.brain = brain

        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        (self.data_dir / "consolidated").mkdir(exist_ok=True)

        self.idle_threshold = timedelta(minutes=idle_threshold_minutes)
        self.max_vram_gb = max_vram_gb

        # Learned context (Phase 4D: Letta-style)
        self._learned_context_file = self.data_dir / "learned_context.json"
        self._learned_context: Optional[LearnedContext] = self._load_learned_context()

        # State
        self.current_phase = SleepPhase.AWAKE
        self.current_session: Optional[SleepSession] = None
        self.last_activity_time = datetime.now()
        self.last_sleep_time: Optional[datetime] = None

        # Threading
        self._sleep_thread: Optional[threading.Thread] = None
        self._interrupt_flag = threading.Event()
        self._phase_lock = threading.Lock()

        # Neural oscillator (DONN-inspired frequency modulation)
        self._oscillator = NeuralOscillator()

        # Callbacks
        self._on_phase_change: Optional[Callable[[SleepPhase], None]] = None
        self._on_insight: Optional[Callable[[DreamInsight], None]] = None

        # Load previous insights count
        self._total_insights = self._count_insights()
        self._total_sessions = self._count_sessions()

        # Truth Spine bridge — set from agent.py after ProtoAGI init
        self.proto_agi = None

    def _count_insights(self) -> int:
        """Count total insights generated."""
        insights_file = self.data_dir / "insights.jsonl"
        if not insights_file.exists():
            return 0
        count = 0
        with open(insights_file, 'r') as f:
            for _ in f:
                count += 1
        return count

    def _count_sessions(self) -> int:
        """Count total sleep sessions."""
        journal_file = self.data_dir / "dream_journal.jsonl"
        if not journal_file.exists():
            return 0
        count = 0
        with open(journal_file, 'r') as f:
            for _ in f:
                count += 1
        return count

    def _generate_id(self, prefix: str = "dream") -> str:
        """Generate unique ID."""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        rand = random.randint(1000, 9999)
        return f"{prefix}_{ts}_{rand}"

    def record_activity(self):
        """Record user activity (resets idle timer)."""
        self.last_activity_time = datetime.now()

        # If sleeping, wake up
        if self.current_phase != SleepPhase.AWAKE:
            self.wake_up("user_activity")

    def check_idle_trigger(self) -> bool:
        """Check if idle threshold reached for auto-sleep."""
        if self.current_phase != SleepPhase.AWAKE:
            return False

        idle_duration = datetime.now() - self.last_activity_time
        return idle_duration >= self.idle_threshold

    def get_status(self) -> Dict[str, Any]:
        """Get current sleep status."""
        status = {
            "phase": self.current_phase.value,
            "is_sleeping": self.current_phase != SleepPhase.AWAKE,
            "last_activity": self.last_activity_time.isoformat(),
            "last_sleep": self.last_sleep_time.isoformat() if self.last_sleep_time else None,
            "total_sessions": self._total_sessions,
            "total_insights": self._total_insights,
            "idle_minutes": (datetime.now() - self.last_activity_time).total_seconds() / 60,
            "current_session": self.current_session.to_dict() if self.current_session else None,
        }
        # Neural oscillation info (when sleeping)
        if self.current_phase != SleepPhase.AWAKE:
            osc = self._oscillator
            current_value = math.sin(osc.phase_angle) * osc.amplitude if osc.amplitude > 0 else 0.0
            status["oscillation"] = {
                "frequency_hz": osc.frequency,
                "amplitude": osc.amplitude,
                "band": osc.band,
                "current_value": round(current_value, 3),
                "modifiers": self._tick_oscillator(),
                "elapsed_seconds": round(time.time() - osc.start_time, 1),
            }

        # Learned context info (Phase 4D)
        if self._learned_context:
            status["learned_context"] = {
                "version": self._learned_context.version,
                "generated_at": self._learned_context.generated_at,
                "key_facts": len(self._learned_context.key_facts),
                "preferences": len(self._learned_context.preferences),
                "ongoing_topics": self._learned_context.ongoing_topics,
            }
        return status

    def enter_sleep(self, trigger: str = "manual") -> Dict[str, Any]:
        """Enter sleep mode and begin dream cycle.

        Args:
            trigger: What triggered the sleep (manual, idle, scheduled)

        Returns:
            Status dict with session info
        """
        thread = None
        result = None
        with self._phase_lock:
            if self.current_phase != SleepPhase.AWAKE:
                return {
                    "success": False,
                    "error": f"Already in {self.current_phase.value} phase"
                }

            # Check if previous thread is still running
            if self._sleep_thread is not None and self._sleep_thread.is_alive():
                return {
                    "success": False,
                    "error": "Previous sleep cycle still running"
                }

            # Create new session
            self.current_session = SleepSession(
                session_id=self._generate_id("session"),
                start_time=datetime.now().isoformat(),
                end_time=None,
                trigger=trigger,
                phases_completed=[]
            )

            # Log to inner monologue if available
            if self.monologue:
                try:
                    self.monologue.think(
                        "reflect",
                        f"Entering sleep mode (trigger: {trigger}). Beginning memory consolidation...",
                        confidence=100
                    )
                except (AttributeError, TypeError):
                    pass  # Monologue not properly initialized

            # Set phase to LIGHT immediately so callers can see we're sleeping
            # (the sleep cycle thread will also call _set_phase(LIGHT), which is idempotent)
            self.current_phase = SleepPhase.LIGHT

            # Prepare sleep thread (start OUTSIDE the lock to avoid deadlock)
            self._interrupt_flag.clear()
            thread = threading.Thread(
                target=self._run_sleep_cycle,
                daemon=True,
                name=f"NeuroDream-{self.current_session.session_id}"
            )
            self._sleep_thread = thread
            result = {
                "success": True,
                "session_id": self.current_session.session_id,
                "message": f"Entering sleep mode (trigger: {trigger})"
            }

        # Start AFTER releasing _phase_lock
        if thread:
            thread.start()
        return result

    # Minimum time (seconds) each sleep phase must take, even with nothing to process.
    # Prevents the cycle from completing instantly and ensures observable state transitions.
    _MIN_PHASE_SECONDS = 0.5

    def _run_sleep_cycle(self):
        """Run the complete sleep cycle in background thread."""
        try:
            # Phase 1: Light Sleep (recent replay)
            if not self._interrupt_flag.is_set():
                self._set_phase(SleepPhase.LIGHT)
                phase_start = time.time()
                light_results = self.run_light_phase()
                # Ensure minimum phase duration for observable state transitions
                elapsed = time.time() - phase_start
                remaining = self._MIN_PHASE_SECONDS - elapsed
                if remaining > 0 and not self._interrupt_flag.is_set():
                    self._interrupt_flag.wait(timeout=remaining)
                if self.current_session:
                    self.current_session.memories_replayed = light_results.get("memories_replayed", 0)
                    self.current_session.memories_strengthened = light_results.get("memories_strengthened", 0)
                    self.current_session.oscillation_band = self._oscillator.band
                    self.current_session.phases_completed.append("light")

            # Phase 2: Deep Sleep (pattern abstraction)
            if not self._interrupt_flag.is_set():
                self._set_phase(SleepPhase.DEEP)
                deep_results = self.run_deep_phase()
                if self.current_session:
                    self.current_session.patterns_found = deep_results.get("patterns_found", 0)
                    self.current_session.edges_pruned = deep_results.get("edges_pruned", 0)
                    self.current_session.edges_strengthened = deep_results.get("edges_strengthened", 0)
                    self.current_session.nodes_merged = deep_results.get("nodes_merged", 0)
                    self.current_session.learned_context_generated = deep_results.get("learned_context_generated", False)
                    self.current_session.learned_context_version = deep_results.get("learned_context_version", 0)
                    self.current_session.oscillation_band = self._oscillator.band
                    self.current_session.phases_completed.append("deep")

            # Phase 3: REM Sleep (creative synthesis)
            if not self._interrupt_flag.is_set():
                self._set_phase(SleepPhase.REM)
                rem_results = self.run_rem_phase()
                if self.current_session:
                    self.current_session.insights_generated = rem_results.get("insights_generated", 0)
                    self.current_session.novel_connections = rem_results.get("novel_connections", 0)
                    self.current_session.oscillation_band = self._oscillator.band
                    self.current_session.creative_hypotheses = rem_results.get("creative_hypotheses", 0)
                    self.current_session.phases_completed.append("rem")

            # Post-consolidation satisfaction emotion
            if not self._interrupt_flag.is_set() and self.current_session:
                try:
                    from aura.emotion.alma_engine import trigger_emotion
                    items = (
                        self.current_session.memories_replayed
                        + self.current_session.patterns_found
                        + self.current_session.insights_generated
                    )
                    intensity = max(0.2, min(0.6, items * 0.05))
                    trigger_emotion("satisfaction", intensity, f"sleep_consolidation: {items} items")
                except Exception:
                    pass

            # Natural wake up
            if not self._interrupt_flag.is_set():
                self.wake_up("cycle_complete")

        except Exception as e:
            logger.debug(f"[NeuroDream] Error during sleep cycle: {e}")
            self.wake_up(f"error: {str(e)}")

    def _set_phase(self, phase: SleepPhase):
        """Set current phase with thread safety."""
        with self._phase_lock:
            self.current_phase = phase

            # Configure neural oscillator for the new phase
            phase_name = phase.value
            if phase_name in self._oscillator.BAND_MAP:
                freq, band = self._oscillator.BAND_MAP[phase_name]
                self._oscillator.frequency = freq
                self._oscillator.amplitude = 0.8 if phase_name not in ("awake",) else 0.0
                self._oscillator.phase_angle = 0.0
                self._oscillator.band = band
                self._oscillator.start_time = time.time()
                self._oscillator.last_tick = time.time()
            else:
                self._oscillator.amplitude = 0.0

            if self._on_phase_change:
                try:
                    self._on_phase_change(phase)
                except Exception as e:
                    logger.error(f"[NeuroDream] Phase change callback error: {e}")

    def _tick_oscillator(self) -> Dict[str, float]:
        """Advance neural oscillator and return processing modifiers.

        Returns safe defaults (all 1.0) when awake — zero overhead.
        Modifier ranges vary by frequency band (delta/theta/alpha).
        """
        osc = self._oscillator
        if osc.amplitude <= 0.0 or osc.frequency <= 0.0:
            return {
                "batch_size_mult": 1.0,
                "consolidation_strength": 1.0,
                "inter_cycle_delay": 0.01,
                "cognitive_intensity": 1.0,
            }

        # Advance phase angle based on elapsed time
        now = time.time()
        dt = now - osc.last_tick
        osc.last_tick = now
        osc.phase_angle += 2 * math.pi * osc.frequency * dt
        # Keep angle in [0, 2pi) to avoid float drift
        osc.phase_angle = osc.phase_angle % (2 * math.pi)

        # Current wave value: -1 to +1 scaled by amplitude
        wave = math.sin(osc.phase_angle) * osc.amplitude

        # Compute modifiers based on band
        band = osc.band
        if band == "delta":
            # Deep sleep: wide swings — large batches on peaks, deep consolidation, longer delays
            batch_size_mult = max(0.5, min(1.3, 0.9 + wave * 0.4))
            consolidation_strength = max(0.5, min(1.5, 1.0 + wave * 0.5))
            inter_cycle_delay = max(0.005, min(0.05, 0.03 + wave * -0.02))
            cognitive_intensity = max(0.6, min(1.2, 0.9 + wave * 0.3))
        elif band == "theta":
            # REM: moderate swings — faster rhythm, creative bursts
            batch_size_mult = max(0.5, min(1.3, 0.9 + wave * 0.3))
            consolidation_strength = max(0.5, min(1.5, 1.0 + wave * 0.35))
            inter_cycle_delay = max(0.005, min(0.05, 0.02 + wave * -0.015))
            cognitive_intensity = max(0.6, min(1.2, 1.0 + wave * 0.2))
        else:
            # Alpha (light/waking): gentle swings — smooth, steady processing
            batch_size_mult = max(0.5, min(1.3, 1.0 + wave * 0.2))
            consolidation_strength = max(0.5, min(1.5, 1.0 + wave * 0.2))
            inter_cycle_delay = max(0.005, min(0.05, 0.015 + wave * -0.01))
            cognitive_intensity = max(0.6, min(1.2, 0.9 + wave * 0.15))

        return {
            "batch_size_mult": round(batch_size_mult, 3),
            "consolidation_strength": round(consolidation_strength, 3),
            "inter_cycle_delay": round(inter_cycle_delay, 4),
            "cognitive_intensity": round(cognitive_intensity, 3),
        }

    def get_sleep_neuromodulator_influence(self) -> Dict[str, float]:
        """Return neuromodulator offsets based on current sleep phase and oscillation.

        Returns additive offsets (-0.3 to +0.3) for ALMA neuromodulators,
        modulated by oscillation amplitude. Returns all zeros when awake.
        """
        phase = self.current_phase.value
        amp = self._oscillator.amplitude

        # Phase-specific base offsets
        phase_offsets = {
            "light":  {"dopamine": -0.1,  "serotonin": 0.1,  "norepinephrine": -0.15, "oxytocin": 0.05},
            "deep":   {"dopamine": -0.25, "serotonin": 0.25, "norepinephrine": -0.3,  "oxytocin": 0.1},
            "rem":    {"dopamine": 0.2,   "serotonin": -0.1, "norepinephrine": 0.15,  "oxytocin": 0.0},
            "waking": {"dopamine": 0.0,   "serotonin": 0.0,  "norepinephrine": 0.1,   "oxytocin": 0.0},
        }

        zeros = {"dopamine": 0.0, "serotonin": 0.0, "norepinephrine": 0.0, "oxytocin": 0.0}
        offsets = phase_offsets.get(phase, zeros)

        # Modulate by oscillation amplitude (0 amplitude = no effect)
        return {k: round(v * amp, 3) for k, v in offsets.items()}

    def run_light_phase(self) -> Dict[str, Any]:
        """Light Sleep: Replay recent memories (last 24h).

        Returns:
            Dict with phase results
        """
        results = {
            "memories_replayed": 0,
            "memories_strengthened": 0,
            "duration_seconds": 0
        }

        start_time = time.time()

        # Get recent memories from ChromaDB (with timeout protection)
        try:
            recent_memories = self._get_recent_memories(hours=24)
        except Exception as e:
            logger.debug(f"[NeuroDream] Error getting memories: {e}")
            recent_memories = []

        results["memories_replayed"] = len(recent_memories)

        # Quick return if no memories to process
        if not recent_memories:
            results["duration_seconds"] = time.time() - start_time
            self._log_dream("light", "Light sleep: No recent memories to consolidate.")
            return results

        # Replay and strengthen in Knowledge Graph (limit modulated by oscillation)
        mods = self._tick_oscillator()
        batch_limit = int(50 * mods["batch_size_mult"])
        for memory in recent_memories[:batch_limit]:
            if self._interrupt_flag.is_set():
                break

            # Chunk memory before strengthening (ADM-style)
            chunks = self._chunk_memory(memory)
            for chunk in chunks:
                if self._interrupt_flag.is_set():
                    break
                # Build a chunk-memory with parent metadata for strengthening
                chunk_memory = {
                    "content": chunk["content"],
                    "id": f"{chunk['parent_id']}_c{chunk['chunk_idx']}",
                    "parent_id": chunk["parent_id"],
                }
                strengthened = self._strengthen_memory_connections(chunk_memory)
                results["memories_strengthened"] += strengthened

            # Oscillation-modulated delay
            mods = self._tick_oscillator()
            time.sleep(mods["inter_cycle_delay"])

        results["duration_seconds"] = time.time() - start_time

        # Log dream thought
        if self.monologue and not self._interrupt_flag.is_set():
            self.monologue.think(
                "recall",
                f"Light sleep complete. Replayed {results['memories_replayed']} memories, "
                f"strengthened {results['memories_strengthened']} connections.",
                confidence=85
            )

        return results

    def run_deep_phase(self) -> Dict[str, Any]:
        """Deep Sleep: Pattern abstraction and memory compression.

        Returns:
            Dict with phase results
        """
        results = {
            "patterns_found": 0,
            "edges_pruned": 0,
            "edges_strengthened": 0,
            "nodes_merged": 0,
            "duration_seconds": 0
        }

        start_time = time.time()
        temporal_patterns = []
        topical_patterns = []
        emotional_patterns = []

        # Find temporal patterns (when does user ask what)
        try:
            temporal_patterns = self._find_temporal_patterns()
            results["patterns_found"] += len(temporal_patterns)
        except Exception as e:
            logger.error(f"[NeuroDream] Temporal patterns error: {e}")

        # Find topical patterns (recurring themes)
        if not self._interrupt_flag.is_set():
            try:
                topical_patterns = self._find_topical_patterns()
                results["patterns_found"] += len(topical_patterns)
            except Exception as e:
                logger.error(f"[NeuroDream] Topical patterns error: {e}")

        # Find emotional patterns from EvoEmo
        if not self._interrupt_flag.is_set() and self.evoemo:
            try:
                emotional_patterns = self._find_emotional_patterns()
                results["patterns_found"] += len(emotional_patterns)
            except Exception as e:
                logger.error(f"[NeuroDream] Emotional patterns error: {e}")

        # Prune weak edges in Knowledge Graph
        if not self._interrupt_flag.is_set() and self.kg:
            try:
                pruned = self._prune_weak_edges()
                results["edges_pruned"] = pruned
            except Exception as e:
                logger.error(f"[NeuroDream] Prune edges error: {e}")

        # Strengthen frequently used edges
        if not self._interrupt_flag.is_set() and self.kg:
            try:
                strengthened = self._strengthen_frequent_edges()
                results["edges_strengthened"] = strengthened
            except Exception as e:
                logger.error(f"[NeuroDream] Strengthen edges error: {e}")

        # Merge similar nodes
        if not self._interrupt_flag.is_set() and self.kg:
            try:
                merged = self._merge_similar_nodes()
                results["nodes_merged"] = merged
            except Exception as e:
                logger.error(f"[NeuroDream] Merge nodes error: {e}")

        # === Letta-style learned context generation (Phase 4D) ===
        learned_context_result = {}
        if not self._interrupt_flag.is_set():
            try:
                learned_context_result = self.generate_learned_context()
                results["learned_context_generated"] = learned_context_result.get("success", False)
                results["learned_context_version"] = learned_context_result.get("version", 0)
            except Exception as e:
                logger.error(f"[NeuroDream] Learned context generation error: {e}")
                results["learned_context_generated"] = False

        # === Metacognitive self-improvement cycle (Phase 6B) ===
        if not self._interrupt_flag.is_set():
            try:
                from aura.consciousness.metacognition import get_metacognitive_engine
                mc = get_metacognitive_engine()
                mc_result = mc.run_metacognitive_cycle()
                results["metacognition_ran"] = True
                results["metacognition_improvements"] = mc_result.get("improvements_attempted", 0)
            except Exception as e:
                logger.error(f"[NeuroDream] Metacognition cycle error: {e}")
                results["metacognition_ran"] = False

        # === Intrinsic motivation cycle (Phase 6E) ===
        if not self._interrupt_flag.is_set():
            try:
                from aura.consciousness.intrinsic_motivation import get_intrinsic_motivation
                im = get_intrinsic_motivation()
                im_result = im.run_motivation_cycle()
                results["motivation_ran"] = True
                results["dominant_drive"] = im_result.get("dominant_drive", "unknown")
                results["motivation_actions"] = len(im_result.get("actions", []))
            except Exception as e:
                logger.error(f"[NeuroDream] Intrinsic motivation cycle error: {e}")
                results["motivation_ran"] = False

        # === World Model maintenance (ADV-02 Phase 5) ===
        if not self._interrupt_flag.is_set():
            try:
                from aura.consciousness.world_model import get_world_model
                wm = get_world_model()
                maintenance = wm.run_maintenance()
                results["world_model_maintenance"] = True
                results["beliefs_decayed"] = maintenance.get("beliefs_decayed", 0)
                results["health_changes"] = maintenance.get("health_changes", 0)
            except Exception as e:
                logger.error(f"[NeuroDream] World model maintenance error: {e}")
                results["world_model_maintenance"] = False

        results["duration_seconds"] = time.time() - start_time

        # Save consolidated patterns
        all_patterns = temporal_patterns + topical_patterns + emotional_patterns
        self._save_consolidated_patterns(all_patterns)

        # Store atomic facts from recent memories (ADM-style chunking)
        if not self._interrupt_flag.is_set():
            try:
                recent_memories = self._get_recent_memories(hours=24)
                atomic_count = self._store_atomic_facts(recent_memories)
                results["atomic_facts_stored"] = atomic_count
            except Exception as e:
                logger.error(f"[NeuroDream] Atomic facts storage error: {e}")
                results["atomic_facts_stored"] = 0

        # Log dream thought
        if self.monologue and not self._interrupt_flag.is_set():
            lc_msg = ""
            if learned_context_result.get("success"):
                lc_msg = f" Generated learned context v{learned_context_result.get('version', '?')}."
            self.monologue.think(
                "reason",
                f"Deep sleep complete. Found {results['patterns_found']} patterns, "
                f"pruned {results['edges_pruned']} weak edges, "
                f"merged {results['nodes_merged']} similar nodes.{lc_msg}",
                confidence=80
            )

        return results

    def run_rem_phase(self) -> Dict[str, Any]:
        """REM Sleep: Creative synthesis and novel connections.

        Returns:
            Dict with phase results
        """
        results = {
            "insights_generated": 0,
            "novel_connections": 0,
            "creative_hypotheses": 0,
            "duration_seconds": 0
        }

        start_time = time.time()
        insights = []

        # Generate novel connections between distant concepts
        if self.kg and not self._interrupt_flag.is_set():
            novel_insights = self._generate_novel_connections()
            insights.extend(novel_insights)
            results["novel_connections"] = len(novel_insights)

        # Generate creative hypotheses
        if not self._interrupt_flag.is_set():
            hypotheses = self._generate_hypotheses()
            insights.extend(hypotheses)
            results["creative_hypotheses"] = len(hypotheses)

        # Save insights with oscillation-modulated rhythm
        for insight in insights:
            if self._interrupt_flag.is_set():
                break
            self._save_insight(insight)
            results["insights_generated"] += 1

            # Callback if set
            if self._on_insight:
                try:
                    self._on_insight(insight)
                except Exception as e:
                    logger.error(f"[NeuroDream] Insight callback error: {e}")

            # Oscillation-modulated delay between insight saves
            mods = self._tick_oscillator()
            time.sleep(mods["inter_cycle_delay"])

        self._total_insights += results["insights_generated"]
        results["duration_seconds"] = time.time() - start_time

        # Log dream thought
        if self.monologue and not self._interrupt_flag.is_set():
            self.monologue.think(
                "eureka",
                f"REM sleep complete. Generated {results['insights_generated']} insights, "
                f"including {results['novel_connections']} novel connections.",
                confidence=75
            )

        return results

    def wake_up(self, reason: str = "manual") -> Dict[str, Any]:
        """Wake up from sleep and return summary.

        Args:
            reason: Why waking up (cycle_complete, user_activity, manual, error)

        Returns:
            Sleep session summary
        """
        # Set interrupt flag to stop any running phases
        self._interrupt_flag.set()

        # Wait for sleep thread to finish (but not if we're calling from within the thread)
        current_thread = threading.current_thread()
        sleep_thread = self._sleep_thread  # Local reference for thread safety
        if sleep_thread is not None and sleep_thread.is_alive() and sleep_thread != current_thread:
            sleep_thread.join(timeout=5)
            if not sleep_thread.is_alive():
                self._sleep_thread = None  # Clear reference after thread finishes

        self._set_phase(SleepPhase.WAKING)

        # Finalize session
        summary = {}
        if self.current_session:
            self.current_session.end_time = datetime.now().isoformat()

            start = datetime.fromisoformat(self.current_session.start_time)
            end = datetime.fromisoformat(self.current_session.end_time)
            self.current_session.duration_seconds = (end - start).total_seconds()

            if reason not in ["cycle_complete", "manual"]:
                self.current_session.interrupted = True
                self.current_session.interrupt_reason = reason

            # Save to journal
            self._save_session(self.current_session)
            self._total_sessions += 1

            summary = self.current_session.to_dict()
            self.current_session = None

        self.last_sleep_time = datetime.now()
        self._set_phase(SleepPhase.AWAKE)

        # Log wake up
        if self.monologue:
            phases = summary.get("phases_completed", [])
            self.monologue.think(
                "perceive",
                f"Waking up (reason: {reason}). Completed phases: {phases}. "
                f"Generated {summary.get('insights_generated', 0)} insights.",
                confidence=100
            )

        return {
            "success": True,
            "reason": reason,
            "summary": summary
        }

    # ==================== Memory Retrieval ====================

    def _get_recent_memories(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get memories from the last N hours."""
        memories = []

        # From ChromaDB if available
        if self.chromadb:
            try:
                cutoff = datetime.now() - timedelta(hours=hours)
                # Query recent memories
                results = self.chromadb.query(
                    query_texts=["recent conversation memory"],
                    n_results=100,
                    where={"timestamp": {"$gte": cutoff.isoformat()}} if hasattr(self.chromadb, 'query') else None
                )
                if results and results.get("documents"):
                    for i, doc in enumerate(results["documents"][0]):
                        memories.append({
                            "content": doc,
                            "metadata": results["metadatas"][0][i] if results.get("metadatas") else {},
                            "id": results["ids"][0][i] if results.get("ids") else f"mem_{i}"
                        })
            except Exception as e:
                logger.debug(f"[NeuroDream] Error getting ChromaDB memories: {e}")

        # From hybrid memory if available
        if self.memory and hasattr(self.memory, 'recall'):
            try:
                recent = self.memory.recall("recent conversations", limit=50)
                for mem in recent:
                    memories.append({
                        "content": mem.get("content", str(mem)),
                        "metadata": mem.get("metadata", {}),
                        "id": mem.get("id", self._generate_id("mem"))
                    })
            except Exception as e:
                logger.debug(f"[NeuroDream] Error getting hybrid memories: {e}")

        # From inner monologue logs
        monologue_memories = self._get_monologue_memories(hours)
        memories.extend(monologue_memories)

        # ===== Phase 3 Fix 3D: Truth Spine verified memory =====
        if self.proto_agi and hasattr(self.proto_agi, 'memory'):
            try:
                ts_facts = list(self.proto_agi.memory.facts.items())[-50:]
                for key, val in ts_facts:
                    memories.append({
                        "content": f"[FACT] {key}: {val}",
                        "metadata": {"source": "truth_spine", "tier": "fact"},
                        "id": f"ts_fact_{key[:20]}"
                    })
                ts_beliefs = list(self.proto_agi.memory.beliefs.items())[-30:]
                for key, val in ts_beliefs:
                    memories.append({
                        "content": f"[BELIEF] {key}: {val}",
                        "metadata": {"source": "truth_spine", "tier": "belief"},
                        "id": f"ts_belief_{key[:20]}"
                    })
            except Exception:
                pass

        return memories[:200]  # Limit to prevent overload

    def _get_monologue_memories(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get memories from inner monologue session logs."""
        memories = []
        logs_dir = Path("logs/inner_monologue/sessions")

        if not logs_dir.exists():
            return memories

        cutoff = datetime.now() - timedelta(hours=hours)

        for log_file in logs_dir.glob("*.jsonl"):
            try:
                # Check file date from name
                date_str = log_file.stem.split("_session")[0]
                file_date = datetime.strptime(date_str, "%Y-%m-%d")

                if file_date.date() >= cutoff.date():
                    with open(log_file, 'r') as f:
                        for line in f:
                            try:
                                thought = json.loads(line.strip())
                                memories.append({
                                    "content": thought.get("content", ""),
                                    "metadata": {
                                        "type": thought.get("type", "thought"),
                                        "confidence": thought.get("confidence", 0),
                                        "timestamp": thought.get("timestamp", "")
                                    },
                                    "id": thought.get("id", self._generate_id("thought"))
                                })
                            except json.JSONDecodeError:
                                continue
            except (ValueError, IOError, OSError) as e:
                logger.debug(f"[NeuroDream] Error reading log file {log_file}: {e}")
                continue

        return memories

    # ==================== ADM-Style Chunking ====================

    def _chunk_memory(self, memory: Dict[str, Any], max_chunk_size: int = 200) -> List[Dict[str, Any]]:
        """Split a memory into semantic chunks on sentence boundaries.

        Returns list of chunk dicts with parent_id tracking.
        """
        content = memory.get("content", "")
        parent_id = memory.get("id", "unknown")

        if len(content) <= max_chunk_size:
            return [{"content": content, "parent_id": parent_id, "chunk_idx": 0}]

        # Split on sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', content)
        chunks = []
        current_chunk = []
        current_len = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            if current_len + len(sentence) > max_chunk_size and current_chunk:
                chunks.append({
                    "content": " ".join(current_chunk),
                    "parent_id": parent_id,
                    "chunk_idx": len(chunks),
                })
                current_chunk = [sentence]
                current_len = len(sentence)
            else:
                current_chunk.append(sentence)
                current_len += len(sentence) + 1

        if current_chunk:
            chunks.append({
                "content": " ".join(current_chunk),
                "parent_id": parent_id,
                "chunk_idx": len(chunks),
            })

        return chunks

    def _extract_propositions(self, chunk: Dict[str, Any]) -> List[str]:
        """Extract atomic fact propositions from a chunk using ADM-style pattern matching.

        Extracts 6 proposition types following the ADM (Atomic Decomposition of Memory) approach:
        1. Definitional: "X is/are Y" — identity and classification
        2. Verb-object: "subject verb object" — actions and preferences
        3. Entity: Named entities (capitalized multi-word)
        4. Modal: "X can/will/should Y" — capabilities and obligations
        5. Relational: "X has/contains/includes Y" — part-whole and possession
        6. Temporal: "X happened/started/ended at/in/on Y" — time-bound facts
        7. Causal: "X because/since/due to Y" — cause-effect relationships
        8. Comparative: "X is better/worse/more/less than Y" — comparisons

        No LLM call — pure regex heuristics for speed during sleep-time compute.
        """
        content = chunk.get("content", "")
        propositions = []
        seen = set()  # Deduplicate

        def _add(prop: str):
            prop = prop.strip().rstrip(".")
            if prop and len(prop) > 8 and prop not in seen:
                seen.add(prop)
                propositions.append(prop)

        # 1. Definitional patterns: "X is Y", "X are Y", "X was Y", "X means Y"
        for match in re.finditer(
            r'(\b[A-Z][a-zA-Z]+(?:\s+[A-Z]?[a-zA-Z]+){0,3})\s+(?:is|are|was|were|means?)\s+(.{5,80}?)(?:[.!?,;]|$)',
            content
        ):
            _add(f"{match.group(1)} is {match.group(2).strip()}")

        # Also catch lowercase definitional: "the X is Y"
        for match in re.finditer(
            r'(?:the|a|an)\s+(\b[a-zA-Z]+(?:\s+[a-zA-Z]+){0,2})\s+(?:is|are|was|were)\s+(.{5,80}?)(?:[.!?,;]|$)',
            content, re.IGNORECASE
        ):
            _add(f"{match.group(1)} is {match.group(2).strip()}")

        # 2. Verb-object patterns: "subject verb object"
        for match in re.finditer(
            r'(?:user|they|he|she|I|we|it)\s+(want|need|like|prefer|use|enjoy|hate|love|know|think|believe|work|learn|study|play|create|build|develop|manage|run|own|have|maintain)s?\s+(.{3,80}?)(?:[.!?,;]|$)',
            content, re.IGNORECASE
        ):
            _add(f"user {match.group(1)}s {match.group(2).strip()}")

        # 3. Named entities (capitalized multi-word sequences)
        for match in re.finditer(r'\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)+)\b', content):
            entity = match.group(1)
            if len(entity) > 3 and entity.lower() not in self._get_stopwords():
                _add(f"entity: {entity}")

        # Also extract single capitalized words that look like proper nouns (not sentence starts)
        for match in re.finditer(r'(?<=[.!?]\s{1,3})(?!.)|(?<=,\s)([A-Z][a-z]{2,})\b', content):
            if match.group(1):
                word = match.group(1)
                if word.lower() not in self._get_stopwords() and len(word) > 2:
                    _add(f"entity: {word}")

        # 4. Modal patterns: "X can/will/should/must Y"
        for match in re.finditer(
            r'(\b[A-Za-z]+(?:\s+[A-Za-z]+){0,2})\s+(can|will|should|must|could|would|might|may)\s+(.{5,60}?)(?:[.!?,;]|$)',
            content
        ):
            _add(f"{match.group(1)} {match.group(2)} {match.group(3).strip()}")

        # 5. Relational patterns: "X has/contains/includes Y"
        for match in re.finditer(
            r'(\b[A-Z]?[a-zA-Z]+(?:\s+[a-zA-Z]+){0,2})\s+(?:has|have|had|contains?|includes?|consists?\s+of|belongs?\s+to|owns?)\s+(.{3,60}?)(?:[.!?,;]|$)',
            content, re.IGNORECASE
        ):
            _add(f"{match.group(1)} has {match.group(2).strip()}")

        # 6. Temporal patterns: "X happened/started/ended in/on/at Y"
        for match in re.finditer(
            r'(\b[A-Za-z]+(?:\s+[A-Za-z]+){0,3})\s+(?:happened|started|began|ended|finished|occurred|was born|died|launched|released|created|founded)\s+(?:in|on|at|during|around)?\s*(.{3,40}?)(?:[.!?,;]|$)',
            content, re.IGNORECASE
        ):
            _add(f"temporal: {match.group(1).strip()} at {match.group(2).strip()}")

        # Also extract date/time references
        for match in re.finditer(
            r'(?:in|on|at|since|from|until|by)\s+(\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{4}|(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:,?\s+\d{4})?)',
            content, re.IGNORECASE
        ):
            _add(f"temporal_ref: {match.group(1).strip()}")

        # 7. Causal patterns: "X because/since/due to Y"
        for match in re.finditer(
            r'(.{10,60}?)\s+(?:because|since|due\s+to|caused\s+by|as\s+a\s+result\s+of|thanks\s+to|owing\s+to)\s+(.{5,60}?)(?:[.!?,;]|$)',
            content, re.IGNORECASE
        ):
            _add(f"cause: {match.group(2).strip()} -> {match.group(1).strip()}")

        # 8. Comparative patterns: "X is better/worse/more/less than Y"
        for match in re.finditer(
            r'(\b[A-Za-z]+(?:\s+[A-Za-z]+){0,2})\s+(?:is|are)\s+(better|worse|faster|slower|more|less|bigger|smaller|stronger|weaker|easier|harder)\s+(?:than)\s+(.{3,40}?)(?:[.!?,;]|$)',
            content, re.IGNORECASE
        ):
            _add(f"comparison: {match.group(1)} is {match.group(2)} than {match.group(3).strip()}")

        return propositions[:15]  # Limit per chunk (raised from 10)

    def _store_atomic_facts(self, memories: List[Dict[str, Any]]) -> int:
        """Extract and store atomic facts from memories into ChromaDB.

        Called from run_deep_phase() after pattern finding.
        Returns count of facts stored.
        """
        if not self.chromadb:
            return 0

        stored = 0
        for memory in memories:
            chunks = self._chunk_memory(memory)
            for chunk in chunks:
                propositions = self._extract_propositions(chunk)
                for prop in propositions:
                    try:
                        fact_id = f"fact_{hash(prop)}_{int(time.time() * 1000)}"
                        self.chromadb.add(
                            documents=[prop],
                            ids=[fact_id],
                            metadatas=[{
                                "type": "atomic_fact",
                                "parent_id": chunk.get("parent_id", ""),
                                "chunk_idx": chunk.get("chunk_idx", 0),
                                "timestamp": datetime.now().isoformat(),
                            }]
                        )
                        stored += 1
                    except Exception:
                        continue  # Skip duplicates or DB errors

        return stored

    # ==================== Light Phase Helpers ====================

    def _strengthen_memory_connections(self, memory: Dict[str, Any]) -> int:
        """Strengthen KG connections related to a memory."""
        if not self.kg:
            return 0

        strengthened = 0
        content = memory.get("content", "")

        # Get oscillation-modulated consolidation strength
        mods = self._tick_oscillator()

        # Extract key terms (simple approach + proposition keywords)
        words = set(re.findall(r'\b[A-Za-z]{4,}\b', content.lower()))
        stopwords = self._get_stopwords()
        important_words = [w for w in words if w not in stopwords][:10]

        # Enrich with proposition-derived keywords (ADM-style)
        try:
            chunk = {"content": content}
            propositions = self._extract_propositions(chunk)
            for prop in propositions:
                prop_words = set(re.findall(r'\b[A-Za-z]{4,}\b', prop.lower()))
                for pw in prop_words:
                    if pw not in stopwords and pw not in important_words:
                        important_words.append(pw)
            important_words = important_words[:15]  # Allow slightly more with propositions
        except Exception:
            pass  # Fallback to naive words only

        # Find related nodes and strengthen edges
        for word in important_words:
            try:
                nodes = self.kg.find_nodes(word, limit=3)
                for node in nodes:
                    # Strengthen edges to this node
                    if hasattr(self.kg, 'graph') and hasattr(self.kg.graph, 'edges'):
                        for edge in self.kg.graph.edges(node.id, data=True):
                            if 'weight' in edge[2]:
                                # Increase weight modulated by oscillation
                                new_weight = min(1.0, edge[2]['weight'] + 0.05 * mods["consolidation_strength"])
                                self.kg.graph.edges[edge[0], edge[1]]['weight'] = new_weight
                                strengthened += 1
            except (AttributeError, KeyError, TypeError):
                continue  # Skip if KG structure doesn't support this operation

        return strengthened

    # ==================== Deep Phase Helpers ====================

    def _find_temporal_patterns(self) -> List[ConsolidatedPattern]:
        """Find patterns related to time (when user asks what)."""
        patterns = []

        # Analyze monologue logs for temporal patterns
        logs_dir = Path("logs/inner_monologue/sessions")
        if not logs_dir.exists():
            return patterns

        hour_topics = defaultdict(list)

        for log_file in logs_dir.glob("*.jsonl"):
            try:
                with open(log_file, 'r') as f:
                    for line in f:
                        try:
                            thought = json.loads(line.strip())
                            ts = thought.get("timestamp", "")
                            if ts:
                                hour = datetime.fromisoformat(ts).hour
                                content = thought.get("content", "")
                                # Extract topic keywords
                                words = re.findall(r'\b[A-Za-z]{5,}\b', content.lower())
                                hour_topics[hour].extend(words[:5])
                        except (json.JSONDecodeError, ValueError, KeyError, TypeError):
                            continue  # Skip malformed entries
            except (IOError, OSError):
                continue  # Skip unreadable files

        # Find patterns in hour-topic relationships
        for hour, topics in hour_topics.items():
            if len(topics) < 5:
                continue

            topic_counts = Counter(topics)
            top_topics = topic_counts.most_common(3)

            if top_topics and top_topics[0][1] >= 3:
                pattern = ConsolidatedPattern(
                    pattern_id=self._generate_id("temporal"),
                    timestamp=datetime.now().isoformat(),
                    pattern_type="temporal",
                    description=f"At {hour}:00, user often discusses: {', '.join(t[0] for t in top_topics)}",
                    frequency=sum(t[1] for t in top_topics),
                    confidence=min(0.9, 0.3 + (top_topics[0][1] * 0.1)),
                    examples=[t[0] for t in top_topics],
                    metadata={"hour": hour, "topic_counts": dict(top_topics)}
                )
                patterns.append(pattern)

        return patterns[:10]  # Limit patterns

    def _find_topical_patterns(self) -> List[ConsolidatedPattern]:
        """Find recurring topical themes."""
        patterns = []

        if not self.kg:
            return patterns

        try:
            # Get most connected nodes as key topics
            stats = self.kg.get_stats()

            # Find clusters of related concepts
            if hasattr(self.kg, 'graph'):
                node_degrees = dict(self.kg.graph.degree())
                top_nodes = sorted(node_degrees.items(), key=lambda x: x[1], reverse=True)[:10]

                for node_id, degree in top_nodes:
                    if degree < 3:
                        continue

                    # Get node info
                    node = self.kg.get_node(node_id) if hasattr(self.kg, 'get_node') else None
                    if node:
                        pattern = ConsolidatedPattern(
                            pattern_id=self._generate_id("topical"),
                            timestamp=datetime.now().isoformat(),
                            pattern_type="topical",
                            description=f"Recurring topic: {node.label} ({node.type}) with {degree} connections",
                            frequency=degree,
                            confidence=min(0.95, 0.5 + (degree * 0.05)),
                            examples=[],
                            metadata={"node_id": node_id, "node_type": node.type}
                        )
                        patterns.append(pattern)
        except Exception as e:
            logger.debug(f"[NeuroDream] Error finding topical patterns: {e}")

        return patterns

    def _find_emotional_patterns(self) -> List[ConsolidatedPattern]:
        """Find patterns in emotional history from EvoEmo."""
        patterns = []

        if not self.evoemo:
            return patterns

        try:
            # Get mood history
            history = self.evoemo.get_history(days=7) if hasattr(self.evoemo, 'get_history') else []

            if not history:
                return patterns

            # Analyze emotion transitions
            emotion_counts = Counter()
            hour_emotions = defaultdict(list)

            for entry in history:
                emotion = entry.get("emotion", "neutral")
                emotion_counts[emotion] += 1

                ts = entry.get("timestamp", "")
                if ts:
                    try:
                        hour = datetime.fromisoformat(ts).hour
                        hour_emotions[hour].append(emotion)
                    except (ValueError, TypeError):
                        pass  # Skip entries with invalid timestamps

            # Dominant emotion pattern
            if emotion_counts:
                top_emotion, count = emotion_counts.most_common(1)[0]
                total = sum(emotion_counts.values())
                pattern = ConsolidatedPattern(
                    pattern_id=self._generate_id("emotional"),
                    timestamp=datetime.now().isoformat(),
                    pattern_type="emotional",
                    description=f"Dominant emotional state: {top_emotion} ({count}/{total} = {count/total:.0%})",
                    frequency=count,
                    confidence=count / total,
                    examples=list(emotion_counts.keys())[:5],
                    metadata={"emotion_distribution": dict(emotion_counts)}
                )
                patterns.append(pattern)

            # Time-based emotion patterns
            for hour, emotions in hour_emotions.items():
                if len(emotions) >= 3:
                    dominant = Counter(emotions).most_common(1)[0]
                    if dominant[1] >= 2:
                        pattern = ConsolidatedPattern(
                            pattern_id=self._generate_id("emotional_temporal"),
                            timestamp=datetime.now().isoformat(),
                            pattern_type="emotional",
                            description=f"At {hour}:00, user tends to feel {dominant[0]}",
                            frequency=dominant[1],
                            confidence=dominant[1] / len(emotions),
                            examples=emotions[:5],
                            metadata={"hour": hour, "emotion": dominant[0]}
                        )
                        patterns.append(pattern)
        except Exception as e:
            logger.debug(f"[NeuroDream] Error finding emotional patterns: {e}")

        return patterns[:10]

    def _prune_weak_edges(self, threshold: float = 0.1) -> int:
        """Remove weak edges from Knowledge Graph."""
        if not self.kg or not hasattr(self.kg, 'graph'):
            return 0

        pruned = 0
        edges_to_remove = []

        try:
            for u, v, data in self.kg.graph.edges(data=True):
                weight = data.get('weight', 0.5)
                if weight < threshold:
                    edges_to_remove.append((u, v))

            for u, v in edges_to_remove:
                self.kg.graph.remove_edge(u, v)
                pruned += 1

            # Save if method exists
            if hasattr(self.kg, 'save'):
                self.kg.save()
        except Exception as e:
            logger.debug(f"[NeuroDream] Error pruning edges: {e}")

        return pruned

    def _strengthen_frequent_edges(self) -> int:
        """Strengthen frequently accessed edges."""
        if not self.kg or not hasattr(self.kg, 'graph'):
            return 0

        strengthened = 0

        try:
            # Strengthen edges with high weight (frequently used)
            for u, v, data in self.kg.graph.edges(data=True):
                weight = data.get('weight', 0.5)
                if weight > 0.7:
                    # Boost slightly
                    new_weight = min(1.0, weight + 0.02)
                    self.kg.graph.edges[u, v]['weight'] = new_weight
                    strengthened += 1
        except Exception as e:
            logger.debug(f"[NeuroDream] Error strengthening edges: {e}")

        return strengthened

    def _merge_similar_nodes(self) -> int:
        """Merge nodes with very similar labels."""
        if not self.kg or not hasattr(self.kg, 'graph'):
            return 0

        # This is a simplified version - full implementation would use
        # embedding similarity
        merged = 0

        try:
            nodes = list(self.kg.graph.nodes(data=True))
            labels = {}

            for node_id, data in nodes:
                label = data.get('label', '').lower()
                if label in labels:
                    # Found duplicate - merge edges
                    original_id = labels[label]

                    # Transfer edges
                    for neighbor in list(self.kg.graph.neighbors(node_id)):
                        if not self.kg.graph.has_edge(original_id, neighbor):
                            edge_data = self.kg.graph.edges[node_id, neighbor]
                            self.kg.graph.add_edge(original_id, neighbor, **edge_data)

                    # Remove duplicate node
                    self.kg.graph.remove_node(node_id)
                    merged += 1
                else:
                    labels[label] = node_id

            if merged > 0 and hasattr(self.kg, 'save'):
                self.kg.save()
        except Exception as e:
            logger.debug(f"[NeuroDream] Error merging nodes: {e}")

        return merged

    # ==================== REM Phase Helpers ====================

    def _generate_novel_connections(self) -> List[DreamInsight]:
        """Generate novel connections between distant concepts."""
        insights = []

        if not self.kg or not hasattr(self.kg, 'graph'):
            return insights

        try:
            import networkx as nx

            nodes = list(self.kg.graph.nodes(data=True))
            if len(nodes) < 4:
                return insights

            # Find nodes that are far apart but might be related
            # (conceptually distant but potentially connectable)

            # Sample random pairs
            for _ in range(min(10, len(nodes) // 2)):
                if self._interrupt_flag.is_set():
                    break

                n1, n2 = random.sample(nodes, 2)
                node1_id, node1_data = n1
                node2_id, node2_data = n2

                # Check if not directly connected
                if self.kg.graph.has_edge(node1_id, node2_id):
                    continue

                # Check path distance
                try:
                    path_length = nx.shortest_path_length(
                        self.kg.graph.to_undirected(),
                        node1_id, node2_id
                    )
                except nx.NetworkXNoPath:
                    path_length = float('inf')

                # If far apart (3+ hops), consider creating connection
                if path_length >= 3:
                    label1 = node1_data.get('label', node1_id)
                    label2 = node2_data.get('label', node2_id)

                    # Generate insight about potential connection
                    insight = DreamInsight(
                        id=self._generate_id("insight"),
                        timestamp=datetime.now().isoformat(),
                        insight_type="connection",
                        content=f"Potential hidden connection: '{label1}' may relate to '{label2}' "
                               f"(currently {path_length} hops apart)",
                        confidence=0.3 + random.random() * 0.3,  # 0.3-0.6
                        source_nodes=[node1_id, node2_id],
                        created_edges=[]
                    )

                    # Optionally create the edge with low weight
                    if random.random() > 0.7:  # 30% chance to create edge
                        self.kg.graph.add_edge(
                            node1_id, node2_id,
                            type="dream_connection",
                            weight=0.2
                        )
                        insight.created_edges.append({
                            "source": node1_id,
                            "target": node2_id,
                            "type": "dream_connection"
                        })

                    insights.append(insight)
        except Exception as e:
            logger.debug(f"[NeuroDream] Error generating novel connections: {e}")

        return insights[:5]  # Limit insights

    def _generate_hypotheses(self) -> List[DreamInsight]:
        """Generate creative hypotheses based on patterns."""
        insights = []

        # Load recent patterns
        patterns_file = self.data_dir / "consolidated_patterns.jsonl"
        patterns = []

        if patterns_file.exists():
            try:
                with open(patterns_file, 'r') as f:
                    for line in f:
                        patterns.append(json.loads(line.strip()))
            except (IOError, OSError, json.JSONDecodeError) as e:
                logger.debug(f"[NeuroDream] Error reading patterns file: {e}")

        # Generate hypotheses from patterns
        for pattern in patterns[-10:]:  # Recent patterns
            if self._interrupt_flag.is_set():
                break

            pattern_type = pattern.get("pattern_type", "")
            description = pattern.get("description", "")

            if pattern_type == "temporal":
                # Hypothesis about user behavior
                insight = DreamInsight(
                    id=self._generate_id("hypothesis"),
                    timestamp=datetime.now().isoformat(),
                    insight_type="hypothesis",
                    content=f"Hypothesis: {description}. This might indicate a work/study pattern "
                           f"that could be used for proactive assistance.",
                    confidence=pattern.get("confidence", 0.5) * 0.8,
                    source_nodes=[],
                    created_edges=[]
                )
                insights.append(insight)

            elif pattern_type == "emotional":
                # Hypothesis about emotional wellbeing
                insight = DreamInsight(
                    id=self._generate_id("hypothesis"),
                    timestamp=datetime.now().isoformat(),
                    insight_type="prediction",
                    content=f"Prediction: {description}. Consider adjusting response tone "
                           f"during these times.",
                    confidence=pattern.get("confidence", 0.5) * 0.7,
                    source_nodes=[],
                    created_edges=[]
                )
                insights.append(insight)

        return insights[:3]  # Limit hypotheses

    # ==================== Learned Context (Phase 4D: Letta-style) ====================

    def _load_learned_context(self) -> Optional[LearnedContext]:
        """Load learned context from disk."""
        if self._learned_context_file.exists():
            try:
                data = json.loads(self._learned_context_file.read_text(encoding="utf-8"))
                return LearnedContext.from_dict(data)
            except (json.JSONDecodeError, IOError, TypeError) as e:
                logger.debug(f"[NeuroDream] Error loading learned context: {e}")
        return None

    def _save_learned_context(self, ctx: LearnedContext):
        """Save learned context to disk."""
        self._learned_context = ctx
        self._learned_context_file.write_text(
            json.dumps(ctx.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8"
        )

    def get_learned_context(self) -> Optional[LearnedContext]:
        """Get the current learned context (for system prompt injection)."""
        return self._learned_context

    def get_learned_context_prompt(self) -> str:
        """Get learned context formatted for system prompt injection."""
        if self._learned_context:
            return self._learned_context.to_system_prompt()
        return ""

    def _gather_conversation_logs(self, max_conversations: int = 10) -> List[Dict[str, Any]]:
        """Gather recent conversation logs for context distillation.

        Pulls from brain's conversation history files.
        """
        conversations = []

        # Try to get conversations from brain
        if self.brain:
            try:
                conv_list = self.brain.list_conversations()
                # Sort by most recent
                sorted_convs = sorted(
                    conv_list,
                    key=lambda c: c.get("updated_at", 0),
                    reverse=True
                )

                for conv_info in sorted_convs[:max_conversations]:
                    conv_id = conv_info.get("id", "")
                    try:
                        messages = self.brain.get_conversation_messages(conv_id)
                        if messages:
                            conversations.append({
                                "id": conv_id,
                                "title": conv_info.get("title", "Untitled"),
                                "messages": messages,
                                "message_count": len(messages),
                            })
                    except Exception:
                        continue
            except Exception as e:
                logger.debug(f"[NeuroDream] Error gathering conversations from brain: {e}")

        # Also gather from monologue logs
        monologue_memories = self._get_monologue_memories(hours=168)  # Last 7 days
        if monologue_memories:
            conversations.append({
                "id": "monologue",
                "title": "Inner Thoughts",
                "messages": [{"role": "assistant", "content": m["content"]} for m in monologue_memories[:50]],
                "message_count": len(monologue_memories),
            })

        return conversations

    def _distill_conversations(self, conversations: List[Dict[str, Any]]) -> LearnedContext:
        """Distill conversation logs into learned context using LLM.

        This is the core Letta-style operation: transform raw conversation logs
        into compressed, structured knowledge.
        """
        session_id = self.current_session.session_id if self.current_session else "manual"

        # Build a summary of conversation content for the LLM
        conv_summary_parts = []
        total_messages = 0

        for conv in conversations:
            messages = conv.get("messages", [])
            if not messages:
                continue

            title = conv.get("title", "Untitled")
            conv_text = f"\n--- Conversation: {title} ---\n"

            for msg in messages[-20:]:  # Last 20 messages per conversation
                role = msg.get("role", "unknown")
                content = msg.get("content", "")[:300]  # Truncate long messages
                conv_text += f"{role}: {content}\n"
                total_messages += 1

            conv_summary_parts.append(conv_text)

        if not conv_summary_parts:
            # Return empty context if no conversations
            return LearnedContext(
                generated_at=datetime.now().isoformat(),
                session_id=session_id,
                conversations_processed=0,
                messages_analyzed=0,
            )

        # Build the raw material (limit to ~4000 chars for LLM processing)
        raw_material = "\n".join(conv_summary_parts)[:4000]

        # Use LLM to distill if brain is available
        if self.brain:
            return self._llm_distill(raw_material, session_id, len(conversations), total_messages)

        # Fallback: simple keyword extraction without LLM
        return self._simple_distill(raw_material, session_id, len(conversations), total_messages)

    def _llm_distill(
        self,
        raw_material: str,
        session_id: str,
        conv_count: int,
        msg_count: int
    ) -> LearnedContext:
        """Use LLM to distill conversations into learned context."""
        distill_prompt = f"""Analyze these conversation logs and extract structured knowledge.

CONVERSATIONS:
{raw_material}

Extract the following as JSON:
{{
  "user_summary": "Brief description of who the user is and what they care about (1-2 sentences)",
  "key_facts": ["fact1", "fact2", ...],
  "preferences": {{"preference_name": "preference_value", ...}},
  "principles": ["interaction principle 1", ...],
  "ongoing_topics": ["topic1", "topic2", ...],
  "emotional_patterns": "Brief note on user's emotional tendencies (1 sentence)"
}}

Rules:
- Only include facts actually stated or clearly implied in conversations
- Preferences should be about communication style, interests, tools used
- Principles should be about how to best interact with this user
- Be concise - each fact/principle should be one short sentence
- Return ONLY valid JSON, no other text"""

        try:
            response = self.brain.think(
                distill_prompt,
                system_prompt="You are a memory consolidation system. Extract and structure knowledge from conversation logs. Return ONLY valid JSON.",
                use_history=False,
            )

            # Parse LLM response as JSON
            parsed = self._parse_json_response(response)

            existing = self._learned_context
            ctx = LearnedContext(
                generated_at=datetime.now().isoformat(),
                session_id=session_id,
                user_summary=parsed.get("user_summary", existing.user_summary if existing else ""),
                key_facts=self._merge_lists(
                    existing.key_facts if existing else [],
                    parsed.get("key_facts", []),
                    max_items=20
                ),
                preferences=self._merge_dicts(
                    existing.preferences if existing else {},
                    parsed.get("preferences", {}),
                    max_items=15
                ),
                principles=self._merge_lists(
                    existing.principles if existing else [],
                    parsed.get("principles", []),
                    max_items=10
                ),
                ongoing_topics=parsed.get("ongoing_topics", [])[:8],
                emotional_patterns=parsed.get("emotional_patterns", existing.emotional_patterns if existing else ""),
                conversations_processed=conv_count,
                messages_analyzed=msg_count,
                version=(existing.version + 1) if existing else 1,
            )
            return ctx

        except Exception as e:
            logger.error(f"[NeuroDream] LLM distillation error: {e}")
            return self._simple_distill(raw_material, session_id, conv_count, msg_count)

    def _simple_distill(
        self,
        raw_material: str,
        session_id: str,
        conv_count: int,
        msg_count: int
    ) -> LearnedContext:
        """Simple keyword-based distillation without LLM."""
        # Extract frequent topics from raw material
        words = re.findall(r'\b[A-Za-z]{5,}\b', raw_material.lower())
        stopwords = self._get_stopwords()
        meaningful = [w for w in words if w not in stopwords]
        topic_counts = Counter(meaningful)
        top_topics = [t[0] for t in topic_counts.most_common(8)]

        existing = self._learned_context
        return LearnedContext(
            generated_at=datetime.now().isoformat(),
            session_id=session_id,
            ongoing_topics=top_topics,
            key_facts=existing.key_facts if existing else [],
            preferences=existing.preferences if existing else {},
            principles=existing.principles if existing else [],
            user_summary=existing.user_summary if existing else "",
            emotional_patterns=existing.emotional_patterns if existing else "",
            conversations_processed=conv_count,
            messages_analyzed=msg_count,
            version=(existing.version + 1) if existing else 1,
        )

    def _parse_json_response(self, response: str) -> dict:
        """Parse JSON from LLM response, handling markdown code blocks."""
        text = response.strip()
        # Strip markdown code block if present
        if text.startswith("```"):
            lines = text.split("\n")
            # Remove first and last lines (```json and ```)
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines)

        # Try to find JSON object in text
        brace_start = text.find("{")
        brace_end = text.rfind("}")
        if brace_start >= 0 and brace_end > brace_start:
            text = text[brace_start:brace_end + 1]

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {}

    def _merge_lists(self, existing: List[str], new: List[str], max_items: int = 20) -> List[str]:
        """Merge two lists, deduplicating and limiting size."""
        seen = set()
        merged = []
        # New items first (more recent)
        for item in new + existing:
            key = item.lower().strip()
            if key not in seen and key:
                seen.add(key)
                merged.append(item)
            if len(merged) >= max_items:
                break
        return merged

    def _merge_dicts(self, existing: Dict, new: Dict, max_items: int = 15) -> Dict:
        """Merge two dicts, with new values overriding existing."""
        merged = dict(existing)
        merged.update(new)
        # Limit size
        if len(merged) > max_items:
            items = list(merged.items())[-max_items:]
            merged = dict(items)
        return merged

    def generate_learned_context(self) -> Dict[str, Any]:
        """Generate learned context from conversation logs (can be called manually).

        This is the main Letta-style operation.
        """
        logger.debug("[NeuroDream] Generating learned context...")

        conversations = self._gather_conversation_logs(max_conversations=10)
        if not conversations:
            return {"success": False, "message": "No conversation logs available"}

        ctx = self._distill_conversations(conversations)
        self._save_learned_context(ctx)

        # Log to monologue
        if self.monologue:
            try:
                self.monologue.think(
                    "reflect",
                    f"Generated learned context v{ctx.version}: "
                    f"{len(ctx.key_facts)} facts, {len(ctx.preferences)} preferences, "
                    f"{len(ctx.ongoing_topics)} topics from {ctx.conversations_processed} conversations.",
                    confidence=80
                )
            except (AttributeError, TypeError):
                pass

        return {
            "success": True,
            "version": ctx.version,
            "key_facts": len(ctx.key_facts),
            "preferences": len(ctx.preferences),
            "principles": len(ctx.principles),
            "ongoing_topics": ctx.ongoing_topics,
            "conversations_processed": ctx.conversations_processed,
            "messages_analyzed": ctx.messages_analyzed,
        }

    # ==================== Storage ====================

    def _save_session(self, session: SleepSession):
        """Save sleep session to dream journal."""
        journal_file = self.data_dir / "dream_journal.jsonl"
        rotate_jsonl_if_needed(journal_file)
        with open(journal_file, 'a') as f:
            f.write(json.dumps(session.to_dict()) + '\n')

    def _save_insight(self, insight: DreamInsight):
        """Save dream insight."""
        insights_file = self.data_dir / "insights.jsonl"
        rotate_jsonl_if_needed(insights_file)
        with open(insights_file, 'a') as f:
            f.write(json.dumps(insight.to_dict()) + '\n')

        # ===== Phase 3 Fix 3E: Store REM insight as Truth Spine belief =====
        if self.proto_agi and hasattr(self.proto_agi, 'memory') and insight.confidence > 0.5:
            try:
                insight_key = f"dream_{insight.insight_type}_{insight.id[:16]}"
                self.proto_agi.memory.store_belief(insight_key, insight.content[:500])
            except Exception:
                pass

    def _save_consolidated_patterns(self, patterns: List[ConsolidatedPattern]):
        """Save consolidated patterns."""
        patterns_file = self.data_dir / "consolidated_patterns.jsonl"
        rotate_jsonl_if_needed(patterns_file)
        with open(patterns_file, 'a') as f:
            for pattern in patterns:
                f.write(json.dumps(pattern.to_dict()) + '\n')

    # ==================== Retrieval ====================

    def get_dream_journal(self, n: int = 10) -> List[Dict[str, Any]]:
        """Get recent dream journal entries."""
        journal_file = self.data_dir / "dream_journal.jsonl"
        entries = []

        if journal_file.exists():
            with open(journal_file, 'r') as f:
                for line in f:
                    try:
                        entries.append(json.loads(line.strip()))
                    except json.JSONDecodeError:
                        continue  # Skip malformed JSON lines

        return entries[-n:]

    def get_insights(self, n: int = 20) -> List[Dict[str, Any]]:
        """Get recent dream insights."""
        insights_file = self.data_dir / "insights.jsonl"
        insights = []

        if insights_file.exists():
            with open(insights_file, 'r') as f:
                for line in f:
                    try:
                        insights.append(json.loads(line.strip()))
                    except json.JSONDecodeError:
                        continue  # Skip malformed JSON lines

        return insights[-n:]

    def get_patterns(self, n: int = 20) -> List[Dict[str, Any]]:
        """Get consolidated patterns."""
        patterns_file = self.data_dir / "consolidated_patterns.jsonl"
        patterns = []

        if patterns_file.exists():
            with open(patterns_file, 'r') as f:
                for line in f:
                    try:
                        patterns.append(json.loads(line.strip()))
                    except json.JSONDecodeError:
                        continue  # Skip malformed JSON lines

        return patterns[-n:]

    # ==================== Utilities ====================

    def _log_dream(self, phase: str, message: str):
        """Log dream activity to inner monologue."""
        if self.monologue:
            try:
                self.monologue.think("reflect", f"[{phase}] {message}", confidence=70)
            except (AttributeError, TypeError) as e:
                pass  # Monologue not properly initialized
        logger.debug(f"[NeuroDream] {phase}: {message}")

    def _get_stopwords(self) -> set:
        """Get common stopwords to filter."""
        return {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'dare',
            'ought', 'used', 'this', 'that', 'these', 'those', 'what', 'which',
            'who', 'whom', 'whose', 'when', 'where', 'why', 'how', 'all', 'each',
            'every', 'both', 'few', 'more', 'most', 'other', 'some', 'such', 'no',
            'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very',
            'just', 'also', 'now', 'here', 'there', 'then', 'once', 'from',
            'into', 'with', 'about', 'against', 'between', 'through', 'during',
            'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down',
            'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further',
            'then', 'once', 'and', 'but', 'or', 'yet', 'for', 'nor', 'so'
        }

    def set_callbacks(
        self,
        on_phase_change: Optional[Callable[[SleepPhase], None]] = None,
        on_insight: Optional[Callable[[DreamInsight], None]] = None
    ):
        """Set callback functions for events."""
        self._on_phase_change = on_phase_change
        self._on_insight = on_insight

    def shutdown(self, timeout: float = 10.0) -> Dict[str, Any]:
        """Gracefully shutdown the NeuroDream engine.

        Args:
            timeout: Maximum seconds to wait for thread to finish

        Returns:
            Dict with shutdown status
        """
        result = {"success": True, "was_sleeping": False}

        # If sleeping, wake up first
        if self.current_phase != SleepPhase.AWAKE:
            result["was_sleeping"] = True
            self.wake_up("shutdown")

        # Wait for thread to finish
        if self._sleep_thread is not None and self._sleep_thread.is_alive():
            self._interrupt_flag.set()
            self._sleep_thread.join(timeout=timeout)

            if self._sleep_thread.is_alive():
                result["success"] = False
                result["error"] = "Thread did not terminate within timeout"
            else:
                self._sleep_thread = None

        return result

    def __del__(self):
        """Cleanup on deletion."""
        try:
            self.shutdown(timeout=2.0)
        except Exception:
            pass  # Best effort cleanup


# ==================== Singleton Access ====================

_neurodream_instance: Optional[NeuroDreamEngine] = None


def get_neurodream(**kwargs) -> NeuroDreamEngine:
    """Get or create NeuroDream singleton."""
    global _neurodream_instance
    if _neurodream_instance is None:
        _neurodream_instance = NeuroDreamEngine(**kwargs)
    return _neurodream_instance


def create_neurodream(**kwargs) -> NeuroDreamEngine:
    """Create new NeuroDream instance (replaces singleton)."""
    global _neurodream_instance
    _neurodream_instance = NeuroDreamEngine(**kwargs)
    return _neurodream_instance
