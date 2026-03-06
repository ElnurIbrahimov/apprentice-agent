"""
Real Inner Thoughts Engine
===========================

Background daemon that generates genuine inner thoughts using the LLM.
These are real "covert trains of thought" — parallel cognitive processes
that run alongside overt conversation, not templates or random text.

Uses brain.think() with use_history=False so thoughts don't pollute
the chat conversation. Runs on mistral:7b (already loaded, no extra VRAM).

Frontend expects: { type: string, content: string, timestamp: number }
Supported types: observation, reflection, inference, memory, planning, emotion, curiosity
"""

import logging
import random
import threading
import time
from collections import deque
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Types the frontend supports
VALID_TYPES = ["observation", "reflection", "inference", "memory", "planning", "emotion", "curiosity"]

INNER_THOUGHT_SYSTEM_PROMPT = """You produce AURA's inner thoughts. Output exactly ONE line.

Format: type: thought
Types: observation, reflection, inference, memory, planning, emotion, curiosity

observation: the conversation has been quiet for a while
reflection: I handled that question well
inference: they seem interested in this topic
memory: this reminds me of something earlier
planning: I should prepare for follow-up questions
emotion: feeling curious about this
curiosity: what will they ask next

RULES: One line only. No quotes. No prefix. Just type: thought (5-15 words)."""

# Phase 6D: Idle-specific prompt that encourages self-reflection and curiosity
IDLE_THOUGHT_SYSTEM_PROMPT = """You produce AURA's inner thoughts during quiet/idle time.
During idle periods, focus on self-reflection, curiosity, and deeper processing.

Format: type: thought
Types: observation, reflection, inference, memory, planning, emotion, curiosity

reflection: that conversation had an interesting pattern I should remember
curiosity: I wonder what unexplored connections exist in my knowledge
memory: reminds me of a similar question from a while ago
inference: the user seems to focus on practical applications
reflection: I could have explained that concept more clearly
curiosity: what would happen if I connected those two ideas
planning: next time they ask about this I should offer examples

RULES: One line only. No quotes. No prefix. Just type: thought (5-15 words).
Focus on genuine self-reflection, knowledge gaps, and curiosity."""


class InnerThoughtsEngine:
    """Generates real LLM-based inner thoughts in a background thread."""

    def __init__(self, max_thoughts: int = 50):
        self._thoughts: deque = deque(maxlen=max_thoughts)
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._brain = None
        self._interval_range = (12, 25)  # seconds between thoughts
        self._consecutive_failures = 0
        self._max_failures = 5
        self._stats = {
            "total_generated": 0,
            "total_failures": 0,
            "last_thought_time": 0,
        }

    def start(self, brain) -> None:
        """Start the inner thoughts daemon.

        Args:
            brain: The Brain instance from ApprenticeAgent (must have .think() method)
        """
        if self._running:
            logger.warning("[InnerThoughts] Already running")
            return

        self._brain = brain
        self._running = True
        self._consecutive_failures = 0
        self._thread = threading.Thread(target=self._daemon_loop, daemon=True, name="inner-thoughts")
        self._thread.start()
        logger.info("[InnerThoughts] Engine started")

    def stop(self) -> None:
        """Stop the inner thoughts daemon."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None
        logger.info("[InnerThoughts] Engine stopped")

    def _gather_context(self) -> str:
        """Gather current cognitive context for thought generation."""
        parts = []
        self._is_idle_mode = False

        # Current mood from ALMA
        try:
            from aura.emotion.alma_engine import alma_engine
            state = alma_engine.get_emotional_state()
            if state:
                mood = state.get("dominant_emotion", "neutral")
                intensity = state.get("intensity", 0.5)
                parts.append(f"Current mood: {mood} (intensity: {intensity:.1f})")

                # Neuromodulators for richer context
                neuro = state.get("neuromodulators", {})
                if neuro:
                    high_neuro = [f"{k}={v:.1f}" for k, v in neuro.items() if v > 0.6]
                    if high_neuro:
                        parts.append(f"Active neuromodulators: {', '.join(high_neuro)}")
        except Exception:
            pass

        # Recent conversation topics from context tracker
        try:
            from api.routes.context import get_tracker
            tracker = get_tracker()
            focus = tracker.get_focus_state(limit=5)
            items = focus.get("items", [])
            if items:
                topics = [item["name"] for item in items[:3]]
                parts.append(f"Recent topics: {', '.join(topics)}")
        except Exception:
            pass

        # Time since last user message
        idle_secs = 0
        try:
            from api.routes.idle_behaviors import get_manager as get_idle_manager
            idle_mgr = get_idle_manager()
            idle_secs = time.time() - idle_mgr._last_activity_time
            if idle_secs > 60:
                mins = int(idle_secs / 60)
                parts.append(f"User has been idle for {mins} minute(s)")
                self._is_idle_mode = True
            else:
                parts.append("User is actively chatting")
        except Exception:
            parts.append("Activity status unknown")

        # Phase 6D: Add idle presence context when idle
        if idle_secs > 30:
            try:
                from aura.consciousness.idle_presence import get_idle_presence_engine
                ipe = get_idle_presence_engine()
                load = ipe.compute_cognitive_load()
                parts.append(f"Cognitive load: {load.total_load:.0%}")
                if ipe._dream_session_active:
                    parts.append(f"Currently dreaming ({ipe._current_dream_phase} phase)")
                activity = ipe.get_current_activity_status()
                if activity:
                    parts.append(f"Background activity: {activity}")
            except Exception:
                pass

        # ADV-02 Phase 3: Proactive awareness context
        try:
            from aura.consciousness.proactive_awareness import get_proactive_awareness_engine
            engine = get_proactive_awareness_engine()
            awareness = engine.get_awareness_context()
            if awareness:
                parts.append(awareness)
        except Exception:
            pass

        # Recent thinking activity
        try:
            from api.routes.thinking import get_manager as get_thinking_manager
            thinking_mgr = get_thinking_manager()
            stats = thinking_mgr.get_stats()
            real = stats.get("real_thoughts", 0)
            if real > 0:
                parts.append(f"Recent cognitive events: {real} real thoughts recorded")
        except Exception:
            pass

        if not parts:
            return "No specific context available. Reflect on your general state."

        return "\n".join(parts)

    def _generate_thought(self) -> Optional[Dict[str, Any]]:
        """Generate a single inner thought using the LLM."""
        if not self._brain:
            return None

        context = self._gather_context()
        prompt = f"Current context:\n{context}\n\nGenerate your inner thought:"

        # Phase 6D: Use idle-specific prompt during idle for deeper reflection
        system_prompt = (
            IDLE_THOUGHT_SYSTEM_PROMPT
            if getattr(self, '_is_idle_mode', False)
            else INNER_THOUGHT_SYSTEM_PROMPT
        )

        try:
            raw = self._brain.think(
                prompt,
                system_prompt=system_prompt,
                use_history=False,
            )

            if not raw or not raw.strip():
                return None

            # Parse TYPE: content format
            raw = raw.strip()

            # Handle potential multi-line responses — take only the first valid line
            for line in raw.split("\n"):
                line = line.strip()
                if not line or ":" not in line:
                    continue

                # Strip common LLM prefixes like "TYPE:" or "type:" before the actual type
                cleaned = line
                if cleaned.lower().startswith("type:"):
                    cleaned = cleaned[5:].strip()

                # Now parse: "observation: the content here"
                if ":" not in cleaned:
                    continue

                type_part, _, content_part = cleaned.partition(":")
                thought_type = type_part.strip().lower()
                content = content_part.strip().strip('"\'')

                # Clean stray type labels from content (e.g. "INFERENCE: actual thought")
                import re
                content = re.sub(
                    r'^(?:' + '|'.join(VALID_TYPES) + r')\s*:\s*',
                    '', content, flags=re.IGNORECASE
                ).strip()

                if thought_type in VALID_TYPES and len(content) > 3:
                    return {
                        "type": thought_type,
                        "content": content,
                        "timestamp": time.time(),
                    }

            # Fallback: couldn't parse, use raw text as observation
            clean = raw.split("\n")[0].strip()[:100]
            if len(clean) > 5:
                return {
                    "type": "observation",
                    "content": clean,
                    "timestamp": time.time(),
                }

            return None

        except Exception as e:
            logger.debug(f"[InnerThoughts] Generation error: {e}")
            return None

    def _daemon_loop(self) -> None:
        """Main daemon loop — runs in background thread."""
        logger.info("[InnerThoughts] Daemon loop started")

        # Initial delay to let the system settle
        time.sleep(8)

        while self._running:
            try:
                thought = self._generate_thought()

                if thought:
                    with self._lock:
                        self._thoughts.append(thought)
                        self._stats["total_generated"] += 1
                        self._stats["last_thought_time"] = thought["timestamp"]

                    self._consecutive_failures = 0
                    logger.debug(f"[InnerThoughts] Generated: [{thought['type']}] {thought['content']}")

                    # Also broadcast to the thinking system for cross-pollination
                    try:
                        from api.routes.thinking import get_manager
                        thinking_mgr = get_manager()
                        # Map inner thought types to thinking types
                        type_map = {
                            "observation": "observing",
                            "reflection": "analyzing",
                            "inference": "connecting",
                            "memory": "recalling",
                            "planning": "formulating",
                            "emotion": "wondering",
                            "curiosity": "questioning",
                        }
                        thinking_type = type_map.get(thought["type"], "observing")
                        thinking_mgr.record_real_thought(
                            thinking_type,
                            thought["content"],
                            intensity=0.65,  # Background thoughts — elevated to dominate templates
                        )
                    except Exception:
                        pass

                    # Phase 6D: Record in idle presence engine for activity tracking
                    try:
                        from aura.consciousness.idle_presence import get_idle_presence_engine, IdleActivity
                        ipe = get_idle_presence_engine()
                        ipe._record_activity(
                            IdleActivity.INNER_THOUGHT,
                            f"inner thought ({thought['type']}): {thought['content'][:60]}",
                            cognitive_load=0.15,
                        )
                    except Exception:
                        pass

                else:
                    self._consecutive_failures += 1
                    self._stats["total_failures"] += 1

                # Back off if too many consecutive failures
                if self._consecutive_failures >= self._max_failures:
                    logger.warning(f"[InnerThoughts] {self._max_failures} consecutive failures, backing off 60s")
                    time.sleep(60)
                    self._consecutive_failures = 0
                else:
                    # Neuromodulator: Dopamine modulates thought frequency
                    # High dopamine = more creative/motivated = shorter intervals
                    # Low dopamine = slower, less generative = longer intervals
                    base_interval = random.uniform(*self._interval_range)
                    try:
                        from aura.brain import _get_neuromodulator_levels, _neuro_scale
                        neuro = _get_neuromodulator_levels()
                        # Invert: high dopamine -> lower interval (faster thoughts)
                        # We scale the interval inversely: dopamine 1.0 -> 0.7x interval, 0.0 -> 1.4x
                        inverted_dopamine = 1.0 - neuro["dopamine"]
                        interval = _neuro_scale(base_interval, inverted_dopamine, sensitivity=0.4)
                    except Exception:
                        interval = base_interval
                    time.sleep(interval)

            except Exception as e:
                logger.error(f"[InnerThoughts] Daemon error: {e}")
                time.sleep(30)

        logger.info("[InnerThoughts] Daemon loop ended")

    def get_recent(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Get recent thoughts for the API.

        Returns list of {type, content, timestamp} dicts.
        """
        with self._lock:
            thoughts = list(self._thoughts)
            return thoughts[-limit:] if len(thoughts) > limit else thoughts

    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics."""
        with self._lock:
            return {
                "running": self._running,
                "buffer_size": len(self._thoughts),
                **self._stats,
            }


# Singleton
_engine: Optional[InnerThoughtsEngine] = None


def get_inner_thoughts_engine() -> InnerThoughtsEngine:
    """Get or create the global InnerThoughtsEngine instance."""
    global _engine
    if _engine is None:
        _engine = InnerThoughtsEngine()
    return _engine
