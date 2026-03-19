"""
AURA Fast Path - MINIMAL Interception

Only handles EXPLICIT commands. Everything else goes to LLM.

This is NOT the place for pattern matching on emotional responses!
Let the LLM handle actual conversation.
"""

import random
import logging
from datetime import datetime
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .emotion import EmotionalEngine

logger = logging.getLogger(__name__)


class FastPathHandler:
    """
    Handles ONLY explicit commands instantly.

    IMPORTANT: Do NOT add pattern matching for emotional responses here!
    Let the LLM handle actual conversation.
    """

    def __init__(
        self,
        memory_store=None,
        emotional_engine: Optional['EmotionalEngine'] = None
    ):
        """
        Initialize the fast-path handler.

        Args:
            memory_store: AURA memory system for storing facts
            emotional_engine: AURA emotion system for mood awareness
        """
        self.memory = memory_store
        self.emotions = emotional_engine
        self._agent = None

    def try_fast_path(self, user_input: str) -> Optional[str]:
        """
        Only handle EXPLICIT commands.
        Returns None for everything else (goes to LLM).

        Args:
            user_input: The user's message

        Returns:
            Response string if handled, None if needs LLM.
        """
        if not user_input:
            return None

        input_lower = user_input.lower().strip()

        # Known command words (only these trigger command handling)
        known_commands = {
            "status", "state",
            "memory", "memories", "remember", "what do you remember",
            "help", "commands", "?",
            "dream", "consolidate", "dream-memory",
        }

        # 1. Slash commands (explicit /commands)
        if input_lower.startswith("/"):
            cmd_word = input_lower.lstrip("/").split()[0] if input_lower.lstrip("/") else ""
            if cmd_word in known_commands or input_lower.lstrip("/").startswith("remember "):
                return self._handle_command(input_lower)
            # Unknown slash command - let it go to LLM instead of error
            return None

        # 2. AURA commands (only "aura <known_command>")
        if input_lower.startswith("aura "):
            after_aura = input_lower[5:].strip()
            first_word = after_aura.split()[0] if after_aura else ""
            # Only handle if it's a known command, otherwise let LLM respond
            if first_word in known_commands or after_aura.startswith("remember "):
                return self._handle_command(input_lower)

        # 2. Memory commands (explicit "remember this:" etc.)
        memory_triggers = [
            "remember this:", "remember:", "note:",
            "save this:", "store this:", "don't forget:",
            "keep in mind:", "fyi:", "important:", "reminder:"
        ]
        if any(input_lower.startswith(t) for t in memory_triggers):
            return self._handle_memory(user_input)

        # 3. THAT'S IT. Everything else goes to LLM.
        # No emotional detection here - let LLM handle conversation!
        return None

    def _handle_command(self, cmd: str) -> str:
        """Handle /commands and aura commands."""

        # Clean the command
        cmd = cmd.lstrip("/").replace("aura ", "").strip()

        if cmd in ["status", "state"]:
            return self._get_status()
        elif cmd in ["mood", "feeling", "how are you"]:
            return self._get_mood()
        elif cmd in ["memory", "memories", "remember", "what do you remember"]:
            return self._get_memory_summary()
        elif cmd in ["help", "commands", "?"]:
            return self._get_help()
        elif cmd.startswith("remember "):
            # Redirect to memory handler
            fact = cmd[9:].strip()
            return self._handle_memory(f"remember: {fact}")
        elif cmd in ('dream', 'consolidate', 'dream-memory'):
            if hasattr(self, '_agent') and hasattr(self._agent, 'run_dream_consolidation'):
                result = self._agent.run_dream_consolidation()
                n = result.get('insights_generated', 0)
                return f"Memory consolidation complete. {n} new insight{'s' if n != 1 else ''} generated."
            return "Dream consolidation is unavailable right now."
        else:
            return f"Unknown command: /{cmd}. Try /help"

    def _handle_memory(self, user_input: str) -> str:
        """Store a memory."""

        # Extract the fact (everything after the colon)
        fact = user_input.split(":", 1)[-1].strip()

        if not fact:
            return "What would you like me to remember?"

        # Store in AURA memory if available
        stored = False
        if self.memory:
            try:
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')
                if hasattr(self.memory, 'add_entry'):
                    self.memory.add_entry("learned_facts", "User-Specific", f"[{timestamp}] {fact}", importance=0.8)
                elif hasattr(self.memory, 'store'):
                    self.memory.store(f"[{timestamp}] {fact}", {"type": "user_fact", "importance": 0.8})
                stored = True
            except Exception as e:
                logger.warning(f"Memory storage error: {e}")

        # Simple confirmation - no elaborate responses
        confirmations = [
            "Got it!",
            "Noted!",
            "I'll remember that.",
            "Saved!",
        ]
        return random.choice(confirmations)

    def _get_status(self) -> str:
        """Get AURA system status."""
        mood = "warm"
        energy = "good"

        if self.emotions:
            try:
                state = self.emotions.state
                mood = state.mood.value.capitalize()
                energy_val = getattr(state, 'energy', 0.5)
                energy = f"{int(energy_val * 100)}%"
            except (AttributeError, TypeError, KeyError) as e:
                logger.debug(f"[FastPath] Status mood error: {e}")

        return f"""**AURA Status**
Mood: {mood}
Energy: {energy}
Memory: Online

Ready to chat!"""

    def _get_mood(self) -> str:
        """Get current mood."""
        if self.emotions:
            try:
                state = self.emotions.state
                mood = state.mood.value
                reason = getattr(state, 'mood_reason', None) or "Just being me"
                return f"Feeling **{mood}**. {reason}"
            except (AttributeError, TypeError, KeyError) as e:
                logger.debug(f"[FastPath] Mood error: {e}")
        return "Feeling good and ready to help!"

    def _get_memory_summary(self) -> str:
        """Get memory system status."""
        if self.memory:
            try:
                stats = self.memory.get_stats()
                total = sum(s.get('entries', 0) for s in stats.values())
                return f"""**Memory Status**
Total memories: {total}
System: Active

I remember our conversations!"""
            except (AttributeError, TypeError, KeyError) as e:
                logger.debug(f"[FastPath] Memory summary error: {e}")
        return "Memory system is active!"

    def _get_help(self) -> str:
        """Get AURA commands help."""
        return """**Commands**
/status - My current state
/mood - How I'm feeling
/memory - What I remember
/help - This message
remember this: <fact> - Store a fact

For everything else, just talk to me!"""


# Convenience function
def create_fast_path(
    memory_store=None,
    emotional_engine=None
) -> FastPathHandler:
    """Create a configured FastPathHandler."""
    return FastPathHandler(
        memory_store=memory_store,
        emotional_engine=emotional_engine
    )


if __name__ == "__main__":
    print("=" * 60)
    print("AURA Fast Path - Minimal Test")
    print("=" * 60)

    handler = FastPathHandler()

    test_messages = [
        # Should be handled by fast-path
        ("remember this: meeting tomorrow at 3pm", True),
        ("aura status", True),
        ("/help", True),
        ("/mood", True),

        # Should NOT be handled (goes to LLM)
        ("hi", False),
        ("how are you?", False),
        ("I got the job!", False),
        ("I'm feeling stressed", False),
        ("guess what happened today", False),
        ("can you help me write code", False),
        ("thanks!", False),
    ]

    for msg, should_handle in test_messages:
        result = handler.try_fast_path(msg)
        was_handled = result is not None
        status = "OK" if was_handled == should_handle else "WRONG"
        path = "FAST" if was_handled else "LLM"
        print(f"\n[{status}] [{path}] '{msg}'")
        if result:
            print(f"  -> {result[:60]}...")

    print("\n" + "=" * 60)
    print("Test complete!")
