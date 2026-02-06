"""
AURAEngine - Main Orchestrator for AURA v3.0 ALIVE System

Brings all ALIVE components together:
- LLM (OllamaClient) - The brain that generates responses
- Memory (MarkdownStore + MemoryRetriever) - What AURA remembers
- Emotion (EmotionalEngine) - How AURA feels
- Proactive (HeartbeatMonitor) - AURA messaging first
- Patterns (PatternProphet) - Behavioral patterns
- Thinking (VisibleThinking) - Show AURA's thought process
- Humanization (ResponseHumanizer) - Natural speech
- Soul (SoulLoader) - Personality configuration
- FastPath - Quick command handling

This is the entry point for AURA's "aliveness".
"""

import logging
from pathlib import Path
from typing import Dict, Optional, List, Any, Callable
from dataclasses import dataclass, field

# Import all ALIVE components
from .memory import MarkdownStore
from .memory.retriever import MemoryRetriever
from .emotion import EmotionalEngine, Mood
from .proactive import HeartbeatMonitor, Notification
from .patterns import PatternProphet
from .thinking import VisibleThinking, ThoughtType
from .humanize import ResponseHumanizer, ResponseTone
from .soul import SoulLoader, SoulConfig
from .fast_path import FastPathHandler
from .llm import OllamaClient

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
        enable_humanization: bool = True,
        model: str = "llama3:8b"
    ):
        """
        Initialize the AURA engine.

        Args:
            data_dir: Base directory for all data storage
            soul_name: Which soul configuration to load
            enable_proactive: Enable proactive notifications
            enable_thinking: Show visible thinking
            enable_humanization: Humanize responses
            model: LLM model to use (default: llama3:8b)
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

        # LLM Client - The brain
        self.llm = OllamaClient(model=model)
        logger.info(f"  [OK] LLM client ready (model: {model})")

        # Memory system
        self.memory = MarkdownStore(data_dir=self.data_dir / "memory")
        logger.info("  [OK] Memory system ready")

        # Memory retriever - Makes AURA actually remember
        self.memory_retriever = MemoryRetriever(
            memory_store=self.memory,
            data_dir=self.data_dir / "memory"
        )
        logger.info("  [OK] Memory retriever ready")

        # Emotional engine
        self.emotion = EmotionalEngine(state_file=self.data_dir / "emotional_state.json")
        logger.info(f"  [OK] Emotion engine ready (mood: {self.emotion.state.mood.value})")

        # Fast path - Only explicit commands
        self.fast_path = FastPathHandler(
            memory_store=self.memory,
            emotional_engine=self.emotion
        )
        logger.info("  [OK] Fast path ready (minimal - only commands)")

        # Proactive system
        self.proactive = HeartbeatMonitor(data_dir=self.data_dir)
        if enable_proactive:
            self.proactive.start()
            logger.info("  [OK] Proactive system started")
        else:
            logger.info("  [--] Proactive system disabled")

        # Pattern recognition
        self.patterns = PatternProphet(data_dir=self.data_dir)
        logger.info(f"  [OK] Pattern recognition ready ({len(self.patterns.patterns)} patterns)")

        # Visible thinking
        self.thinking = VisibleThinking(show_thoughts=enable_thinking)
        logger.info("  [OK] Thinking system ready")

        # Response humanizer
        self.humanizer = ResponseHumanizer()
        logger.info("  [OK] Humanizer ready")

        # Soul configuration
        self.soul_loader = SoulLoader()
        self.soul = self.soul_loader.load(soul_name)
        logger.info(f"  [OK] Soul loaded: {self.soul.name}")

        # Track conversation state
        self._last_topic: Optional[str] = None
        self._turn_count: int = 0
        self._conversation_history: List[Dict] = []

        # Telegram callback for proactive messaging
        self._send_message_callback: Optional[Callable] = None

        # Initial sync to populate markdown files
        self.sync_to_markdown()

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

        # Extract profile information from message
        self.memory.extract_and_store_profile(user_input)

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

        # Sync to markdown periodically (every 5 turns)
        if self._turn_count % 5 == 0:
            self.sync_to_markdown()

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

    def generate_response(
        self,
        user_message: str,
        chat_id: Optional[str] = None
    ) -> str:
        """
        Generate response with FULL context injection.

        This is the MAIN method - it:
        1. Tries fast-path (only for explicit commands)
        2. Retrieves relevant memories
        3. Gets emotional context
        4. Calls LLM with all context
        5. Stores interaction for future memory
        6. Checks for follow-up triggers

        Args:
            user_message: What the user said
            chat_id: Optional chat identifier for multi-user support

        Returns:
            AURA's response
        """
        self._turn_count += 1

        # Record user activity for proactive system
        self.proactive.record_activity()

        # 1. Try fast-path (only explicit commands)
        fast_response = self.fast_path.try_fast_path(user_message)
        if fast_response:
            logger.debug(f"Fast-path handled: {user_message[:50]}...")
            return fast_response

        # 2. Retrieve relevant memories
        memories = self.memory_retriever.get_relevant_memories(user_message)
        logger.debug(f"Retrieved {len(memories)} relevant memories")

        # === PHASE 1: Wire real memory recall tracking ===
        try:
            from api.routes.memory import record_memory_recall
            if memories:
                memory_texts = [str(m)[:100] for m in memories[:5]]
                record_memory_recall("retriever", len(memories), user_message, memory_texts)
        except Exception:
            pass
        try:
            from api.routes.context import track_context_from_memory
            if memories:
                track_context_from_memory([str(m)[:100] for m in memories[:5]])
        except Exception:
            pass

        # 3. Get user profile
        user_profile = self.memory_retriever.get_user_profile()

        # 4. Get emotional context and react to message
        self.emotion.process_interaction(user_message)
        emotional_context = {
            "mood": self.emotion.state.mood.value,
            "energy": getattr(self.emotion.state, 'energy', 0.5),
            "warmth": getattr(self.emotion.state, 'warmth', 0.5)
        }

        # === PHASE 1: Wire emotion tracking to ContextHeatmap ===
        try:
            from api.routes.context import track_context_from_emotion
            track_context_from_emotion(emotional_context["mood"], getattr(self.emotion.state, 'energy', 0.5))
        except Exception:
            pass

        # 5. Extract profile information from message
        self.memory.extract_and_store_profile(user_message)

        # === PHASE 1: Record real thinking — what AURA is actually processing ===
        try:
            from api.routes.thinking import get_manager as get_thinking_manager
            tm = get_thinking_manager()
            # Record that we're recalling memories (real cognitive step)
            if memories:
                tm.record_real_thought("recalling", f"retrieved {len(memories)} memories about: {user_message[:50]}", intensity=0.6, source="engine")
            # Record emotional processing
            tm.record_real_thought("analyzing", f"emotional state: {emotional_context['mood']}", intensity=0.4, source="emotion")
        except Exception:
            pass

        # 6. Generate with LLM + full context
        response = self.llm.generate(
            user_message=user_message,
            conversation_history=self._conversation_history[-10:],
            memories=memories,
            emotional_context=emotional_context,
            user_profile=user_profile,
            additional_context=self.soul.get_system_prompt_addition() if self.soul else None
        )

        # 7. Optional humanization (light touch, LLM already has personality)
        if self.enable_humanization and len(response) > 50:
            tone = self._determine_tone()
            result = self.humanizer.humanize(response, tone=tone, query=user_message)
            # Only use humanized if it's not too different
            if len(result.modifications) <= 2:
                response = result.humanized

        # === PHASE 1: Record that LLM generation completed ===
        try:
            from api.routes.thinking import get_manager as get_thinking_manager
            tm = get_thinking_manager()
            tm.record_real_thought("formulating", f"generated response ({len(response)} chars)", intensity=0.5, source="engine")
        except Exception:
            pass

        # 8. Store interaction for future memory
        self.memory_retriever.store_interaction(user_message, response, chat_id)

        # 9. Update conversation history
        self._conversation_history.append({"role": "user", "content": user_message})
        self._conversation_history.append({"role": "assistant", "content": response})

        # Keep history bounded
        if len(self._conversation_history) > 20:
            self._conversation_history = self._conversation_history[-20:]

        # 10. Check for follow-up triggers (interview, meeting, etc.)
        self._check_follow_up_triggers(user_message, chat_id)

        # 11. Sync to markdown periodically
        if self._turn_count % 5 == 0:
            self.sync_to_markdown()

        return response

    def _check_follow_up_triggers(self, message: str, chat_id: Optional[str]) -> None:
        """Check if message mentions something to follow up on."""

        if not chat_id:
            return

        msg_lower = message.lower()

        # Interview detection
        if "interview" in msg_lower:
            if "tomorrow" in msg_lower:
                self._schedule_follow_up(chat_id, "the interview", days=2)
            elif "next week" in msg_lower:
                self._schedule_follow_up(chat_id, "the interview", days=8)
            elif any(day in msg_lower for day in ["monday", "tuesday", "wednesday", "thursday", "friday"]):
                self._schedule_follow_up(chat_id, "the interview", days=3)

        # Meeting detection
        if "meeting" in msg_lower:
            if "tomorrow" in msg_lower:
                self._schedule_follow_up(chat_id, "that meeting", days=2)

        # Important event detection
        important_events = ["exam", "presentation", "deadline", "surgery", "appointment"]
        for event in important_events:
            if event in msg_lower and "tomorrow" in msg_lower:
                self._schedule_follow_up(chat_id, f"the {event}", days=2)
                break

    def _schedule_follow_up(self, chat_id: str, topic: str, days: int) -> None:
        """Schedule a follow-up notification."""
        try:
            from datetime import datetime, timedelta
            follow_up_date = datetime.now() + timedelta(days=days)
            message = f"Hey! How did {topic} go?"

            # Add to proactive notifications
            self.proactive.add_notification(
                message=message,
                category="follow_up",
                action_hint=f"Ask about {topic}"
            )
            logger.info(f"Scheduled follow-up about '{topic}' in {days} days")
        except Exception as e:
            logger.warning(f"Failed to schedule follow-up: {e}")

    def set_send_callback(self, callback: Callable) -> None:
        """Set the callback for sending proactive messages (e.g., Telegram)."""
        self._send_message_callback = callback
        logger.info("Proactive messaging callback set")

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

    def sync_to_markdown(self) -> bool:
        """
        Sync JSON state to markdown files for human readability.

        Returns:
            True if sync was successful
        """
        success = True

        # Sync emotional state
        try:
            state_data = self.emotion.state.to_dict()
            success = success and self.memory.sync_emotional_state(state_data)
        except Exception as e:
            logger.error(f"Error syncing emotional state: {e}")
            success = False

        # Sync patterns
        try:
            patterns_data = {name: p.to_dict() for name, p in self.patterns.patterns.items()}
            success = success and self.memory.sync_patterns(patterns_data)
        except Exception as e:
            logger.error(f"Error syncing patterns: {e}")
            success = False

        return success

    def shutdown(self) -> None:
        """Gracefully shutdown the ALIVE system."""
        logger.info("Shutting down AURA ALIVE System...")

        # Sync state to markdown before shutdown
        self.sync_to_markdown()

        if self.enable_proactive:
            self.proactive.stop()

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
        mock_response = "Sure! Let me explain how this works..."

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
