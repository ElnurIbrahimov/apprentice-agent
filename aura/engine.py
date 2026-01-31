"""
AURAEngine - Main Orchestrator for AURA v3.0 ALIVE System

Brings all ALIVE components together:
- Memory (MarkdownStore)
- Emotion (EmotionalEngine)
- Proactive (HeartbeatMonitor)
- Patterns (PatternProphet)
- Thinking (VisibleThinking)
- Humanization (ResponseHumanizer)
- Soul (SoulLoader)

This is the entry point for AURA's "aliveness".
"""

import logging
from pathlib import Path
from typing import Dict, Optional, List, Any
from dataclasses import dataclass, field

# Import all ALIVE components
from .memory import MarkdownStore
from .emotion import EmotionalEngine, Mood
from .proactive import HeartbeatMonitor, Notification
from .patterns import PatternProphet
from .thinking import VisibleThinking, ThoughtType
from .humanize import ResponseHumanizer, ResponseTone
from .soul import SoulLoader, SoulConfig

logger = logging.getLogger(__name__)


@dataclass
class AURAResponse:
    """Complete response from AURA including all context."""
    content: str                      # Main response text
    thinking: str = ""                # Visible thinking (if enabled)
    notifications: List[str] = field(default_factory=list)  # Any proactive messages
    mood: str = "neutral"             # Current mood
    confidence: float = 0.7           # Response confidence
    humanized: bool = False           # Was response humanized
    patterns_used: List[str] = field(default_factory=list)  # Patterns that informed response


class AURAEngine:
    """
    The ALIVE orchestrator - makes AURA feel present and human.

    Coordinates all subsystems to create coherent, emotionally
    appropriate, contextually aware responses.
    """

    def __init__(
        self,
        data_dir: Optional[str] = None,
        soul_name: str = "SOUL_PERSONAL",
        enable_proactive: bool = True,
        enable_thinking: bool = True,
        enable_humanization: bool = True
    ):
        """
        Initialize the AURA engine.

        Args:
            data_dir: Base directory for all data storage
            soul_name: Which soul configuration to load
            enable_proactive: Enable proactive notifications
            enable_thinking: Show visible thinking
            enable_humanization: Humanize responses
        """
        if data_dir is None:
            data_dir = Path(__file__).parent / "data"
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Feature flags
        self.enable_proactive = enable_proactive
        self.enable_thinking = enable_thinking
        self.enable_humanization = enable_humanization

        # Initialize all components
        logger.info("Initializing AURA ALIVE System...")

        # Memory system
        self.memory = MarkdownStore(data_dir=self.data_dir / "memory")
        logger.info("  ✓ Memory system ready")

        # Emotional engine
        self.emotion = EmotionalEngine(state_file=self.data_dir / "emotional_state.json")
        logger.info(f"  ✓ Emotion engine ready (mood: {self.emotion.state.mood.value})")

        # Proactive system
        self.proactive = HeartbeatMonitor(data_dir=self.data_dir)
        if enable_proactive:
            self.proactive.start()
            logger.info("  ✓ Proactive system started")
        else:
            logger.info("  - Proactive system disabled")

        # Pattern recognition
        self.patterns = PatternProphet(data_dir=self.data_dir)
        logger.info(f"  ✓ Pattern recognition ready ({len(self.patterns.patterns)} patterns)")

        # Visible thinking
        self.thinking = VisibleThinking(show_thoughts=enable_thinking)
        logger.info("  ✓ Thinking system ready")

        # Response humanizer
        self.humanizer = ResponseHumanizer()
        logger.info("  ✓ Humanizer ready")

        # Soul configuration
        self.soul_loader = SoulLoader()
        self.soul = self.soul_loader.load(soul_name)
        logger.info(f"  ✓ Soul loaded: {self.soul.name}")

        # Track conversation state
        self._last_topic: Optional[str] = None
        self._turn_count: int = 0

        logger.info("AURA ALIVE System initialized!")

    def _determine_tone(self) -> ResponseTone:
        """Determine appropriate response tone from emotional state."""
        mood = self.emotion.state.mood

        tone_map = {
            Mood.EXCITED: ResponseTone.ENTHUSIASTIC,
            Mood.HAPPY: ResponseTone.WARM,
            Mood.CONTENT: ResponseTone.WARM,
            Mood.NEUTRAL: ResponseTone.CASUAL,
            Mood.THOUGHTFUL: ResponseTone.THOUGHTFUL,
            Mood.TIRED: ResponseTone.DIRECT,
            Mood.CONCERNED: ResponseTone.EMPATHETIC,
            Mood.FRUSTRATED: ResponseTone.DIRECT
        }

        return tone_map.get(mood, ResponseTone.WARM)

    def process_input(self, user_input: str) -> Dict[str, Any]:
        """
        Process user input through all ALIVE systems.

        This doesn't generate the response (that's the LLM's job),
        but prepares all the context and state for response generation.

        Args:
            user_input: The user's message

        Returns:
            Context dict with all relevant information for response
        """
        self._turn_count += 1

        # Record user activity
        self.proactive.record_activity()

        # Process through emotion engine
        self.emotion.process_interaction(user_input)

        # Record interaction for pattern learning
        interaction = self.patterns.record_interaction(user_input, self._last_topic)
        self._last_topic = interaction.topic

        # Start thinking process
        if self.enable_thinking:
            self.thinking.start_thinking(f"Processing: {user_input[:50]}...")

        # Get relevant memory
        memory_context = self.memory.get_context_for_llm()

        # Get pattern predictions
        predictions = self.patterns.predict(interaction.topic)

        # Get any pending notifications
        notifications = []
        if self.enable_proactive:
            pending = self.proactive.get_pending_notifications()
            notifications = [n.message for n in pending]

        # Build context
        context = {
            "user_input": user_input,
            "topic": interaction.topic,
            "sentiment": interaction.sentiment,
            "mood": self.emotion.state.mood.value,
            "mood_reason": self.emotion.state.mood_reason,
            "energy": self.emotion.state.energy,
            "tone": self._determine_tone().value,
            "memory_context": memory_context,
            "predictions": predictions,
            "notifications": notifications,
            "thinking_prefix": self.thinking.generate_thinking_prefix(user_input) if self.enable_thinking else "",
            "soul_guidance": self.soul.get_system_prompt_addition(),
            "greeting_style": self.emotion.get_greeting_style() if self._turn_count == 1 else "",
            "turn_count": self._turn_count
        }

        # Add thinking
        if self.enable_thinking:
            self.thinking.think(f"Topic detected: {interaction.topic}", ThoughtType.ANALYZING)
            self.thinking.think(f"User sentiment: {interaction.sentiment}", ThoughtType.ANALYZING)
            if predictions:
                self.thinking.think(f"User might want: {predictions[0][0]}", ThoughtType.CONNECTING)

        return context

    def process_response(self, response: str, context: Dict[str, Any]) -> AURAResponse:
        """
        Process the LLM response through ALIVE systems.

        Args:
            response: Raw LLM response
            context: Context from process_input

        Returns:
            Complete AURAResponse with all enhancements
        """
        # Humanize if enabled
        humanized = False
        final_response = response

        if self.enable_humanization:
            tone = ResponseTone(context.get("tone", "warm"))
            result = self.humanizer.humanize(
                response,
                tone=tone,
                query=context.get("user_input", "")
            )
            final_response = result.humanized
            humanized = bool(result.modifications)

        # Complete thinking
        thinking_output = ""
        if self.enable_thinking:
            self.thinking.conclude("Response generated")
            thinking_output = self.thinking.get_formatted_thinking()

        # Store in conversation memory (key highlights only)
        if len(final_response) > 50:
            summary = final_response[:100].split(".")[0] + "..."
            self.memory.add_entry(
                "conversations",
                "Recent",
                f"Discussed: {summary}",
                tags=[context.get("topic", "general")],
                importance=0.5
            )

        return AURAResponse(
            content=final_response,
            thinking=thinking_output,
            notifications=context.get("notifications", []),
            mood=context.get("mood", "neutral"),
            confidence=0.7,  # Could be enhanced with actual confidence
            humanized=humanized,
            patterns_used=[p[0] for p in context.get("predictions", [])[:2]]
        )

    def get_system_prompt_enhancement(self) -> str:
        """
        Get ALIVE enhancements for the system prompt.

        Returns:
            Additional context to add to system prompts
        """
        parts = []

        # Soul guidance
        if self.soul:
            parts.append(self.soul.get_system_prompt_addition())

        # Emotional context
        mood = self.emotion.state.mood.value
        tone = self.emotion.state.get_tone_modifier()
        parts.append(f"\nCurrent mood: {mood}. Respond in a {tone} manner.")

        # Memory context
        memory = self.memory.get_context_for_llm(max_tokens=200)
        if memory:
            parts.append(f"\nContext from memory:\n{memory}")

        return "\n".join(parts)

    def remember(self, fact: str, category: str = "General", importance: float = 0.6) -> bool:
        """
        Store a fact in memory.

        Args:
            fact: The fact to remember
            category: Category (User-Specific, Technical, General)
            importance: How important (0.0-1.0)

        Returns:
            True if stored successfully
        """
        return self.memory.add_entry(
            "learned_facts",
            category,
            fact,
            importance=importance
        )

    def get_greeting(self) -> str:
        """Get an appropriate greeting based on current state."""
        base_greeting = self.emotion.get_greeting_style()

        # Check for notifications
        if self.enable_proactive:
            pending = self.proactive.get_pending_notifications()
            for n in pending:
                if n.category == "greeting":
                    return n.message

        return base_greeting

    def shutdown(self) -> None:
        """Gracefully shutdown the ALIVE system."""
        logger.info("Shutting down AURA ALIVE System...")

        if self.enable_proactive:
            self.proactive.stop()

        # Could persist additional state here

        logger.info("AURA ALIVE System shutdown complete")

    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        return {
            "version": "3.0 ALIVE",
            "soul": self.soul.name if self.soul else "None",
            "mood": self.emotion.get_status(),
            "memory": self.memory.get_stats(),
            "patterns": self.patterns.get_status(),
            "proactive": self.proactive.get_status(),
            "features": {
                "proactive": self.enable_proactive,
                "thinking": self.enable_thinking,
                "humanization": self.enable_humanization
            },
            "turns": self._turn_count
        }


# Convenience function for quick setup
def create_aura(
    data_dir: Optional[str] = None,
    soul: str = "SOUL_PERSONAL"
) -> AURAEngine:
    """
    Create a configured AURA engine.

    Args:
        data_dir: Data directory (default: aura/data)
        soul: Soul configuration to use

    Returns:
        Configured AURAEngine
    """
    return AURAEngine(data_dir=data_dir, soul_name=soul)


if __name__ == "__main__":
    print("=" * 60)
    print("AURA ALIVE System - Engine Test")
    print("=" * 60)

    # Initialize engine
    engine = create_aura()

    # Get greeting
    print(f"\n{engine.get_greeting()}")

    # Test conversation
    test_inputs = [
        "Hi! How are you today?",
        "Can you help me understand Python decorators?",
        "Thanks, that was really helpful!",
        "I'm a bit frustrated with this bug I've been debugging for hours.",
    ]

    for user_input in test_inputs:
        print(f"\n--- User: {user_input} ---")

        # Process input
        context = engine.process_input(user_input)

        # Simulate LLM response (in real use, this comes from the LLM)
        mock_response = "I would be happy to help you with that. Let me explain..."

        # Process response
        response = engine.process_response(mock_response, context)

        if response.thinking:
            print(f"Thinking: {response.thinking[:100]}...")
        print(f"Mood: {response.mood}")
        print(f"Response: {response.content[:100]}...")

        if response.notifications:
            print(f"Notifications: {response.notifications}")

    # Show status
    print("\n--- System Status ---")
    status = engine.get_status()
    print(f"Version: {status['version']}")
    print(f"Soul: {status['soul']}")
    print(f"Mood: {status['mood']['mood']}")
    print(f"Patterns: {status['patterns']['total_patterns']}")
    print(f"Turns: {status['turns']}")

    # Shutdown
    engine.shutdown()

    print("\n" + "=" * 60)
    print("Test complete!")
