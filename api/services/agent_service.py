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

    def chat_stream(self, message: str) -> Generator[str, None, Dict[str, Any]]:
        """Stream a chat response from the agent.

        Args:
            message: User message

        Yields:
            Response chunks

        Returns:
            Final response dict with mood
        """
        with self._agent_lock:
            # Check if brain has streaming support
            if hasattr(self.agent.brain, 'think_stream'):
                full_response = ""
                for chunk in self.agent.brain.think_stream(message):
                    full_response += chunk
                    yield chunk

                return {
                    "response": full_response,
                    "fast_path": self._was_fast_path(message),
                    "mood": self._get_mood(),
                    "model_used": self.agent.brain.get_last_model_used()
                }
            else:
                # Fallback to non-streaming
                response = self.agent.chat(message, speak=False)
                yield response
                return {
                    "response": response,
                    "fast_path": self._was_fast_path(message),
                    "mood": self._get_mood(),
                    "model_used": self.agent.brain.get_last_model_used()
                }

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
        """Extract current mood from AURA/EvoEmo."""
        try:
            agent = self.agent

            # Try AURA first
            if hasattr(agent, 'aura') and agent.aura:
                aura_state = agent.aura.get_state() if hasattr(agent.aura, 'get_state') else None
                if aura_state:
                    return MoodState(
                        emotion=aura_state.get('emotion', 'neutral'),
                        confidence=aura_state.get('confidence', 50),
                        valence=aura_state.get('valence', 0.0),
                        arousal=aura_state.get('arousal', 0.0)
                    )

            # Try EvoEmo tool
            if 'evoemo' in agent.tools:
                evoemo = agent.tools['evoemo']
                if hasattr(evoemo, 'get_state'):
                    state = evoemo.get_state()
                    return MoodState(
                        emotion=state.get('emotion', 'neutral'),
                        confidence=state.get('confidence', 50),
                        valence=state.get('valence', 0.0),
                        arousal=state.get('arousal', 0.0)
                    )

            # Default neutral mood
            return MoodState(emotion='neutral', confidence=50)

        except Exception as e:
            logger.debug(f"[AgentService] Could not get mood: {e}")
            return MoodState(emotion='neutral', confidence=50)

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
