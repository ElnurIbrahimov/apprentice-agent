"""Singleton wrapper for ApprenticeAgent."""

import sys
import os
import threading
import logging
from typing import Optional, Dict, Any, Generator

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from apprentice_agent import ApprenticeAgent
from api.models.schemas import MoodState

# Import ALMA directly for mood detection
try:
    from apprentice_agent.emotion.alma_engine import alma_engine
    from apprentice_agent.emotion.integration import get_mood_emoji
    ALMA_AVAILABLE = True
except ImportError:
    ALMA_AVAILABLE = False
    alma_engine = None

logger = logging.getLogger(__name__)


class AgentService:
    """Singleton service for managing ApprenticeAgent instance."""

    _instance: Optional['AgentService'] = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._agent: Optional[ApprenticeAgent] = None
        self._agent_lock = threading.RLock()
        self._initialized = True
        logger.info("[AgentService] Singleton initialized")

    @property
    def is_ready(self) -> bool:
        """Check if agent is ready."""
        return self._agent is not None

    def initialize(self, fast_init: bool = True) -> None:
        """Initialize the agent instance.

        Args:
            fast_init: If True, use fast initialization (skips heavy tools)
        """
        with self._agent_lock:
            if self._agent is None:
                logger.info(f"[AgentService] Initializing agent (fast_init={fast_init})...")
                self._agent = ApprenticeAgent(fast_init=fast_init)
                logger.info("[AgentService] Agent initialized successfully")

    @property
    def agent(self) -> ApprenticeAgent:
        """Get the agent instance, initializing if needed."""
        if self._agent is None:
            self.initialize()
        return self._agent

    def _needs_tools(self, message: str) -> bool:
        """Check if message requires tool execution (search, code, etc.)."""
        message_lower = message.lower()

        # Search indicators
        search_keywords = [
            'search', 'look up', 'find online', 'google', 'web search',
            'what is the price', 'current price', 'latest news',
            'search online', 'search for', 'look online'
        ]

        # Code/calculation indicators
        code_keywords = [
            'calculate', 'compute', 'run code', 'execute', 'factorial',
            'fibonacci', 'prime number', 'run python'
        ]

        # Check for search patterns
        if any(kw in message_lower for kw in search_keywords):
            return True

        # Check for code patterns
        if any(kw in message_lower for kw in code_keywords):
            return True

        return False

    def chat(self, message: str, speak: bool = False, model_override: Optional[str] = None) -> Dict[str, Any]:
        """Send a chat message to the agent.

        Args:
            message: User message
            speak: Enable TTS
            model_override: Optional model to use instead of auto-selection

        Returns:
            Dict with response, fast_path flag, and mood
        """
        with self._agent_lock:
            # Set model override if provided (uses brain's override mechanism)
            if model_override:
                self.agent.brain.set_model_override(model_override)
                logger.info(f"[AgentService] Using model override: {model_override}")

            try:
                # Use agent.chat() which has direct handlers for search/crypto
                # This bypasses the slow agent.run() loop and prevents hallucination
                response = self.agent.chat(message, speak=speak)

                return {
                    "response": response,
                    "fast_path": self._was_fast_path(message),
                    "mood": self._get_mood(),
                    "model_used": self.agent.brain.get_last_model_used()
                }
            finally:
                # Clear model override after request
                if model_override:
                    self.agent.brain.set_model_override(None)

    def chat_stream(self, message: str, model_override: Optional[str] = None):
        """Stream a chat response from the agent.

        Args:
            message: User message
            model_override: Optional model to use

        Yields:
            Response chunks as they're generated
        """
        with self._agent_lock:
            # Set model override if provided
            if model_override:
                self.agent.brain.set_model_override(model_override)
                logger.info(f"[AgentService] Streaming with model override: {model_override}")

            try:
                # Check if brain has streaming support
                if hasattr(self.agent.brain, 'think_stream'):
                    for chunk in self.agent.brain.think_stream(message):
                        yield {"type": "chunk", "content": chunk}

                    # After streaming, yield final result
                    yield {
                        "type": "done",
                        "mood": self._get_mood(),
                        "model_used": self.agent.brain.get_last_model_used()
                    }
                else:
                    # Fallback to non-streaming
                    response = self.agent.chat(message, speak=False)
                    yield {"type": "chunk", "content": response}
                    yield {
                        "type": "done",
                        "mood": self._get_mood(),
                        "model_used": self.agent.brain.get_last_model_used()
                    }
            finally:
                # Clear model override after request
                if model_override:
                    self.agent.brain.set_model_override(None)

    def run(self, goal: str, context: Optional[Dict] = None,
            use_fastpath: Optional[bool] = None, max_iterations: int = 10) -> Dict[str, Any]:
        """Run the agent with a goal.

        Args:
            goal: Goal for the agent
            context: Additional context
            use_fastpath: Force fast-path mode
            max_iterations: Max iterations

        Returns:
            Run result dict
        """
        with self._agent_lock:
            # Set max iterations temporarily
            original_max = getattr(self.agent, 'max_iterations', 10)
            self.agent.max_iterations = max_iterations

            try:
                result = self.agent.run(goal, context=context, use_fastpath=use_fastpath)
                result["mood"] = self._get_mood()
                return result
            finally:
                self.agent.max_iterations = original_max

    def get_status(self) -> Dict[str, Any]:
        """Get agent status information."""
        with self._agent_lock:
            agent = self.agent

            return {
                "online": True,
                "model": getattr(agent.brain, 'model', 'unknown'),
                "aura_enabled": getattr(agent, 'aura_enabled', False),
                "mood": self._get_mood(),
                "memory_count": len(agent.memory.memories) if hasattr(agent.memory, 'memories') else 0,
                "query_count": getattr(agent.brain, '_total_query_count', 0),
                "last_model_used": agent.brain.get_last_model_used()
            }

    def clear_history(self) -> bool:
        """Clear conversation history."""
        with self._agent_lock:
            try:
                self.agent.brain.clear_history()
                return True
            except Exception as e:
                logger.error(f"[AgentService] Failed to clear history: {e}")
                return False

    def _get_mood(self) -> Optional[MoodState]:
        """Extract AURA's current mood from ALMA emotional engine."""
        print(f"[DEBUG _get_mood] ALMA_AVAILABLE={ALMA_AVAILABLE}, alma_engine exists={alma_engine is not None}", flush=True)
        try:
            # Try ALMA directly first (most reliable)
            if ALMA_AVAILABLE and alma_engine:
                try:
                    alma_state = alma_engine.get_emotional_state()
                    print(f"[DEBUG] ALMA state: {alma_state}", flush=True)
                    if alma_state:
                        pad = alma_state.get('pad', {})
                        emoji = get_mood_emoji() if ALMA_AVAILABLE else '🤖'
                        mood = MoodState(
                            emotion=alma_state.get('dominant_emotion', 'neutral'),
                            confidence=int(alma_state.get('intensity', 0.5) * 100),
                            valence=pad.get('pleasure', 0.0),
                            arousal=pad.get('arousal', 0.0),
                            dominance=pad.get('dominance', 0.0),
                            emoji=emoji
                        )
                        print(f"[DEBUG] Returning ALMA mood: {mood.model_dump()}", flush=True)
                        return mood
                except Exception as e:
                    print(f"[DEBUG] ALMA direct state ERROR: {e}", flush=True)

            agent = self.agent

            # Fallback: Try ALMA via brain
            if hasattr(agent.brain, '_alma_enabled') and agent.brain._alma_enabled:
                try:
                    alma_state = agent.brain.get_emotional_state()
                    if alma_state:
                        pad = alma_state.get('pad', {})
                        return MoodState(
                            emotion=alma_state.get('dominant_emotion', 'neutral'),
                            confidence=int(alma_state.get('intensity', 0.5) * 100),
                            valence=pad.get('pleasure', 0.0),
                            arousal=pad.get('arousal', 0.0),
                            dominance=pad.get('dominance', 0.0),
                            emoji=agent.brain.get_mood_emoji()
                        )
                except Exception as e:
                    logger.debug(f"[AgentService] ALMA brain state error: {e}")

            # Fallback: Try legacy AURA
            if hasattr(agent, 'aura') and agent.aura:
                aura_state = agent.aura.get_state() if hasattr(agent.aura, 'get_state') else None
                if aura_state:
                    return MoodState(
                        emotion=aura_state.get('emotion', 'neutral'),
                        confidence=aura_state.get('confidence', 50),
                        valence=aura_state.get('valence', 0.0),
                        arousal=aura_state.get('arousal', 0.0),
                        dominance=0.0,
                        emoji='😐'
                    )

            # Fallback: Try EvoEmo tool (user emotion, not AURA)
            if 'evoemo' in agent.tools:
                evoemo = agent.tools['evoemo']
                if hasattr(evoemo, 'get_state'):
                    state = evoemo.get_state()
                    return MoodState(
                        emotion=state.get('emotion', 'neutral'),
                        confidence=state.get('confidence', 50),
                        valence=state.get('valence', 0.0),
                        arousal=state.get('arousal', 0.0),
                        dominance=0.0,
                        emoji='😐'
                    )

            # Default neutral mood with ALMA defaults
            logger.info("[AgentService] Using default mood (no ALMA/AURA)")
            return MoodState(
                emotion='neutral',
                confidence=50,
                valence=0.3,  # Slightly positive baseline
                arousal=0.1,
                dominance=0.3,
                emoji='🤖'
            )

        except Exception as e:
            logger.warning(f"[AgentService] _get_mood exception: {e}")
            return MoodState(
                emotion='neutral',
                confidence=50,
                valence=0.0,
                arousal=0.0,
                dominance=0.0,
                emoji='🤖'
            )

    def _was_fast_path(self, message: str) -> bool:
        """Check if message was handled via fast path."""
        try:
            return self.agent._is_simple_query(message)
        except Exception:
            return False

    def get_available_models(self) -> Dict[str, Any]:
        """Get list of available models (local and cloud)."""
        with self._agent_lock:
            try:
                import requests
                from apprentice_agent.config import VERIFIED_LOCAL_MODELS, VERIFIED_CLOUD_MODELS

                local_models = []
                cloud_models = list(VERIFIED_CLOUD_MODELS)

                # Get locally installed models from Ollama
                try:
                    ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
                    response = requests.get(f"{ollama_host}/api/tags", timeout=5)
                    if response.status_code == 200:
                        local_models = [m["name"] for m in response.json().get("models", [])]
                except Exception as e:
                    logger.warning(f"[AgentService] Could not fetch local models: {e}")
                    local_models = list(VERIFIED_LOCAL_MODELS)

                current_model = getattr(self.agent.brain, 'model', 'auto')

                return {
                    "local": sorted(local_models),
                    "cloud": sorted(cloud_models),
                    "current": current_model
                }
            except Exception as e:
                logger.error(f"[AgentService] Failed to get models: {e}")
                return {
                    "local": [],
                    "cloud": [],
                    "current": "unknown"
                }


# Global instance
agent_service = AgentService()
