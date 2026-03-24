"""Narrative self, emotion, AURA context, and coherent loop mixin.

Extracted from agent.py (2026-03-23) to reduce class size.
All methods assume self has: tools, brain, identity, _soul, _visible_thinking,
_temporal_lock, _prev_message, _prev_response, aura_enabled, monologue.
"""

import logging
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class NarrativeMixin:
    """Mixin providing narrative-self, emotion, AURA context, and coherent-loop methods."""

    # ------------------------------------------------------------------
    # Emotion helpers
    # ------------------------------------------------------------------

    def _analyze_emotion(self, message: str):
        """Analyze emotional state from user message."""
        try:
            if "evoemo" in self.tools and self.tools["evoemo"].is_enabled():
                return self.tools["evoemo"].analyze_text(message)
        except (AttributeError, KeyError, TypeError, ValueError) as e:
            logger.debug(f"[EvoEmo] Analysis error: {e}")
        return None

    def get_current_mood(self):
        """Get current emotional state (for external use)."""
        try:
            if "evoemo" in self.tools:
                return self.tools["evoemo"].get_current_mood()
        except (AttributeError, KeyError, TypeError):
            pass  # EvoEmo tool not properly initialized
        return None

    def get_mood_emoji(self) -> str:
        """Get emoji for current mood (for GUI)."""
        try:
            if "evoemo" in self.tools:
                return self.tools["evoemo"].get_mood_emoji()
        except (AttributeError, KeyError, TypeError):
            pass  # EvoEmo tool not properly initialized
        return "\U0001f610"  # neutral face

    # ------------------------------------------------------------------
    # Soul / personality
    # ------------------------------------------------------------------

    def _get_soul_prompt(self) -> str:
        """Get soul's system prompt addition for personality injection."""
        if self._soul is None:
            return ""
        try:
            return self._soul.get_system_prompt_addition()
        except (AttributeError, KeyError, TypeError) as e:
            logger.debug(f"[Soul] Prompt generation failed: {e}")
            return ""

    # ------------------------------------------------------------------
    # Temporal grounding
    # ------------------------------------------------------------------

    def _temporal_grounding(self) -> Optional[str]:
        """Build temporal grounding context if this is a new session.

        Detects session start (>5 min since last interaction), loads narrative
        self-model, calculates time elapsed, returns grounding context.
        """
        with self._temporal_lock:
            now = time.time()
            last = getattr(self, '_last_interaction_ts', 0)
            self._last_interaction_ts = now

        if last == 0:
            # First call ever — skip grounding
            return None

        elapsed_minutes = (now - last) / 60
        if elapsed_minutes < 5:
            # Same session — no grounding needed
            return None

        # This is a session start
        elapsed_hours = elapsed_minutes / 60
        parts = []

        # Time awareness
        if elapsed_hours < 1:
            parts.append(f"It's been about {int(elapsed_minutes)} minutes since we last talked.")
        elif elapsed_hours < 24:
            parts.append(f"It's been about {elapsed_hours:.0f} hours since we last talked.")
        else:
            days = elapsed_hours / 24
            parts.append(f"It's been about {days:.0f} days since we last talked.")

        # Load narrative relationship state
        try:
            from aura.narrative_self import get_narrative_self
            narrative = get_narrative_self()
            if narrative.relationship_state:
                parts.append(narrative.relationship_state)
        except (ImportError, AttributeError, TypeError) as e:
            logger.debug(f"[Temporal] Narrative self load failed: {e}")

        # Load dream insights from last sleep cycle (Phase 4)
        try:
            _project_root = Path(__file__).resolve().parent.parent.parent
            dream_queue = _project_root / "data" / "neurodream" / "dream_proactive_queue.json"
            if dream_queue.exists():
                import json as _json
                queue_data = _json.loads(dream_queue.read_text(encoding='utf-8'))
                dream_msgs = queue_data.get("messages", [])
                if dream_msgs:
                    dream_text = "\n".join(
                        f"- [{m['type']}] {m['content']}" for m in dream_msgs[:3]
                    )
                    parts.append(
                        "Thoughts from my last sleep cycle:\n" + dream_text
                    )
                # Clear the queue so it doesn't repeat
                dream_queue.unlink(missing_ok=True)
        except (OSError, IOError, ValueError, KeyError) as e:
            logger.debug(f"[Temporal] Dream queue load failed: {e}")

        if not parts:
            return None

        return "SESSION CONTEXT:\n" + " ".join(parts)

    # ------------------------------------------------------------------
    # AURA context
    # ------------------------------------------------------------------

    def _build_aura_context(self, message: str) -> dict:
        """Build AURA context using ALMA and unified memory."""
        context = {"mood": "neutral", "tone": None, "memory_context": "", "thinking_prefix": ""}
        try:
            from aura.emotion.alma_engine import get_alma_engine
            alma = get_alma_engine()
            if alma:
                state = alma.get_emotional_state()
                context["mood"] = state.get("dominant_emotion", "neutral")
        except (ImportError, AttributeError, KeyError, TypeError) as e:
            logger.debug(f"[Agent] non-critical: {e}")
        # Generate thinking prefix via VisibleThinking
        if self._visible_thinking:
            try:
                prefix = self._visible_thinking.generate_thinking_prefix(message)
                if prefix:
                    context["thinking_prefix"] = prefix
            except (AttributeError, TypeError, ValueError) as e:
                logger.debug(f"[Agent] non-critical: {e}")
        return context

    # ------------------------------------------------------------------
    # Coherent Loop — Pre-response appraisal & post-response feedback
    # ------------------------------------------------------------------

    def _pre_response_appraisal(self, message: str) -> None:
        """Run chain-of-emotion appraisal BEFORE generating a response.

        Calls the fast model to ask "how would I naturally feel about this
        message?" and feeds the result into ALMA so the mood is updated
        before the response style prompt is generated.

        Must be synchronous (with a short timeout) so the mood is ready
        by the time the response generation starts.
        """
        try:
            from aura.emotion.integration import appraise_message
            from aura.core.thought_recorder import record_thought as _record_thought
            result = appraise_message(message, self.brain)
            if result:
                _record_thought(
                    "observing",
                    f"emotional appraisal: {result.get('emotion', '?')} "
                    f"(intensity={result.get('intensity', 0):.1f})",
                    0.4, "emotion",
                )
        except (ImportError, AttributeError, KeyError, TypeError, ValueError, ConnectionError, TimeoutError) as e:
            logger.debug("[ALMA] Pre-response appraisal error: %s", e)

    def _post_response_feedback(self, current_message: str) -> None:
        """Analyze the user's new message as a reaction to our previous response.

        Closes the coherent loop: response outcome feeds back into ALMA so
        the mood drifts based on how the user actually reacted.  Runs only
        when there is a previous exchange to compare against.
        """
        if not self._prev_message or not self._prev_response:
            return
        try:
            from aura.emotion.integration import analyze_user_reaction
            from aura.core.thought_recorder import record_thought as _record_thought
            result = analyze_user_reaction(
                current_message, self._prev_response, self.brain,
            )
            if result:
                _record_thought(
                    "reflecting",
                    f"user reaction: {result.get('emotion', '?')} "
                    f"(sat={result.get('satisfaction', 0):.1f} "
                    f"eng={result.get('engagement', 0):.1f})",
                    0.4, "emotion",
                )
        except (ImportError, AttributeError, KeyError, TypeError, ValueError, ConnectionError, TimeoutError) as e:
            logger.debug("[ALMA] Post-response feedback error: %s", e)

    # ------------------------------------------------------------------
    # AURA command handler
    # ------------------------------------------------------------------

    def _handle_aura_command(self, message: str) -> Optional[str]:
        """Handle AURA system commands (legacy AURAEngine removed; uses ALMA/unified memory)."""
        message_lower = message.lower()

        aura_commands = [
            "aura status", "aura mood", "aura soul", "aura memory",
            "aura patterns", "aura insights", "remember this",
            "aura remember", "what do you remember"
        ]

        if not any(cmd in message_lower for cmd in aura_commands):
            return None

        try:
            if "status" in message_lower:
                tool_count = len(self.tools)
                name = self.identity.get('name', 'AURA')
                return f"AURA Status:\n- Name: {name}\n- Tools: {tool_count} loaded\n- Status: Online"

            elif "mood" in message_lower:
                evoemo = self.tools.get("evoemo")
                if evoemo and hasattr(evoemo, 'get_state'):
                    state = evoemo.get_state()
                    return f"Current mood: {state.get('dominant_emotion', 'neutral')}"
                return "Mood system active. Feeling ready!"

            elif "soul" in message_lower:
                name = self.identity.get('name', 'AURA')
                personality = self.identity.get('personality', 'friendly and helpful')
                return f"My Identity:\n- Name: {name}\n- Personality: {personality}"

            elif "remember this" in message_lower or "aura remember" in message_lower:
                fact = message.replace("remember this:", "").replace("aura remember:", "").strip()
                fact = fact.replace("remember this", "").replace("aura remember", "").strip()
                if fact:
                    try:
                        self.memory.store(content=fact, source="user_fact", importance=0.7)
                        return f"Got it, I'll remember: '{fact[:50]}...'"
                    except (AttributeError, TypeError, OSError) as e:
                        logger.debug(f"[AURA] Memory store failed: {e}")
                        return "I couldn't store that memory."
                return "What would you like me to remember?"

            elif "memory" in message_lower or "what do you remember" in message_lower:
                return "Memory system active. I store and recall conversations automatically."

            elif "patterns" in message_lower or "insights" in message_lower:
                return "Pattern detection is handled by ALMA and EvoEmo subsystems."

        except (AttributeError, KeyError, TypeError, ValueError) as e:
            logger.debug(f"[AURA] Command error: {e}")
            return f"AURA command error: {e}"

        return None

    # ------------------------------------------------------------------
    # EvoEmo command handler
    # ------------------------------------------------------------------

    def _handle_evoemo_command(self, message: str) -> Optional[str]:
        """Handle EvoEmo-specific commands."""
        message_lower = message.lower()

        evoemo_commands = [
            "my mood", "how am i feeling", "current mood", "mood status",
            "mood history", "emotion history", "clear mood", "disable mood",
            "enable mood", "mood patterns"
        ]

        if not any(cmd in message_lower for cmd in evoemo_commands):
            return None

        try:
            evoemo = self.tools.get("evoemo")
            if not evoemo:
                return None

            # SECURITY: Destructive mood commands require confirmation in non-auto mode
            # Default restrictive: block if permissions manager is not available
            if any(kw in message_lower for kw in ("clear", "disable")):
                try:
                    from aura.core.permissions import PermissionTier
                    pm = getattr(self, "permissions", None)
                    if pm is None or not pm.trust_mode:
                        logger.info(f"[EvoEmo] Destructive command blocked (non-auto mode): {message_lower[:40]}")
                        return "Destructive mood commands require full-auto mode or explicit confirmation via the API."
                except (ImportError, AttributeError) as e:
                    logger.debug(f"[EvoEmo] Permission check failed: {e}")

            if "clear" in message_lower:
                result = evoemo.clear_history()
                return "Mood history cleared." if result.get("success") else "Failed to clear history."

            elif "disable" in message_lower:
                evoemo.set_enabled(False)
                return "Mood tracking disabled."

            elif "enable" in message_lower:
                evoemo.set_enabled(True)
                return "Mood tracking enabled."

            elif "history" in message_lower:
                history = evoemo.get_history(days=7)
                if not history:
                    return "No mood history yet."
                # Summarize
                from collections import Counter
                emotions = [h["emotion"] for h in history]
                dist = Counter(emotions)
                summary = ", ".join(f"{e}: {c}" for e, c in dist.most_common())
                return f"Mood history (7 days, {len(history)} readings): {summary}"

            elif "pattern" in message_lower:
                patterns = evoemo.get_patterns()
                if patterns.get("status") == "insufficient_data":
                    return f"Not enough data for patterns yet ({patterns.get('readings', 0)} readings)."
                dominant = patterns.get("dominant_emotion", "calm")
                stress_hours = patterns.get("stress_hours", [])
                stress_info = f" Stress tends to peak around: {stress_hours}" if stress_hours else ""
                return f"Your dominant mood: {dominant}.{stress_info}"

            else:
                # Current mood
                mood = evoemo.get_current_mood()
                if mood:
                    emoji = evoemo.get_mood_emoji()
                    return f"Current mood: {emoji} {mood.emotion} ({mood.confidence}% confidence)"
                return "No mood data yet. Keep chatting and I'll pick up on how you're feeling."

        except (AttributeError, KeyError, TypeError, ValueError) as e:
            logger.debug(f"[EvoEmo] Command error: {e}")
            return None
