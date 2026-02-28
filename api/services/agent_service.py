"""Singleton wrapper for ApprenticeAgent."""

import sys
import os
import threading
import logging
from typing import Optional, Dict, Any, Generator

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from aura import ApprenticeAgent
from api.models.schemas import MoodState

# Import ALMA directly for mood detection
try:
    from aura.emotion.alma_engine import alma_engine
    from aura.emotion.integration import get_mood_emoji
    ALMA_AVAILABLE = True
except ImportError:
    ALMA_AVAILABLE = False
    alma_engine = None

logger = logging.getLogger(__name__)

# Truth Spine singleton — shared across all chat() calls to avoid per-call disk I/O
_truth_spine_instance = None
_truth_spine_lock = threading.Lock()
_MemoryTier = None  # lazy import


def _get_truth_spine():
    """Get or create the VerifiedMemory singleton. Thread-safe."""
    global _truth_spine_instance, _MemoryTier
    if _truth_spine_instance is None:
        with _truth_spine_lock:
            if _truth_spine_instance is None:
                try:
                    from aura.truth_spine import VerifiedMemory, MemoryTier
                    _truth_spine_instance = VerifiedMemory()
                    _MemoryTier = MemoryTier
                except Exception as e:
                    logger.warning(f"[TruthSpine] Init failed: {e}")
                    return None
    return _truth_spine_instance


# Re-export MemoryTier lazily so callers can use the tier values
class _MemoryTierProxy:
    """Proxy for MemoryTier enum that resolves lazily."""
    @property
    def FACT(self): return _MemoryTier.FACT if _MemoryTier else None
    @property
    def BELIEF(self): return _MemoryTier.BELIEF if _MemoryTier else None
    @property
    def SPECULATION(self): return _MemoryTier.SPECULATION if _MemoryTier else None

MemoryTier = _MemoryTierProxy()

# Thinking system integration — safe import
def _record_thought(thought_type: str, content: str, intensity: float = 0.6, source: str = "service"):
    """Record a real thought event. Safe to call even if thinking system isn't ready."""
    try:
        from api.routes.thinking import record_thought
        record_thought(thought_type, content, intensity, source)
    except Exception:
        pass

# =============================================================================
#                    ACTION MODE TRIGGER SYSTEM
# =============================================================================

# Trigger words that activate different agent modes
# Format: trigger_word -> (action_mode, model_config)

ACTION_TRIGGERS = {
    # ===== SEARCH MODE =====
    # Quick web search - uses fast cloud model
    "search": "search",
    "google": "search",
    "lookup": "search",
    "find online": "search",
    "web search": "search",
    "search online": "search",
    "look up": "search",
    "search for": "search",
    "search the web": "search",

    # ===== RESEARCH MODE =====
    # Deep research with multiple sources - uses powerful reasoning model
    "research": "research",
    "deep dive": "research",
    "analyze": "research",
    "investigate": "research",
    "comprehensive": "research",
    "in-depth": "research",
    "detailed analysis": "research",
    "thorough research": "research",
    "deep research": "research",
    "full analysis": "research",

    # ===== AGENT MODE =====
    # Autonomous multi-step tasks - uses agentic model
    "agent": "agent",
    "autonomous": "agent",
    "execute": "agent",
    "automate": "agent",
    "do this for me": "agent",
    "handle this": "agent",
    "take care of": "agent",
    "multi-step": "agent",
    "workflow": "agent",
    "[agent mode]": "agent",

    # ===== CODE MODE =====
    # Code generation/analysis - uses code-specialized model
    "code": "code",
    "program": "code",
    "script": "code",
    "implement": "code",
    "debug": "code",
    "fix code": "code",
    "write code": "code",
    "coding": "code",
    "refactor": "code",
    "optimize code": "code",

    # ===== VISION MODE =====
    # Image analysis - uses vision model
    "describe image": "vision",
    "analyze image": "vision",
    "what's in this": "vision",
    "look at this": "vision",
    "explain this image": "vision",
    "screenshot": "vision",

    # ===== DEEP RESEARCH MODE =====
    # Multi-query, multi-source research with page reading
    "deep research": "deep_research",
    "thorough research": "deep_research",
    "extensive research": "deep_research",
    "full research": "deep_research",
    "research everything": "deep_research",
    "research in depth": "deep_research",

    # ===== SWARM MODE =====
    # Multiple agents working in parallel
    # Note: Compound triggers (longer) must come first to match before individual words
    "swarm research": "swarm",
    "swarm search": "swarm",
    "swarm analyze": "swarm",
    "swarm mode": "swarm",
    "swarm": "swarm",
    "multi-agent": "swarm",
    "multiple agents": "swarm",
    "team research": "swarm",
    "collaborative research": "swarm",
    "collaborative": "swarm",
    "all agents": "swarm",
    "agent team": "swarm",
}

# Best models for each action mode
ACTION_MODE_MODELS = {
    "search": {
        "preferred": "gemini-3-flash-preview:cloud",
        "fallbacks": ["nemotron-3-nano:30b-cloud", "kimi-k2.5:cloud"],
        "description": "Quick web search"
    },
    "research": {
        "preferred": "qwen3.5:397b-cloud",
        "fallbacks": ["cogito-2.1:671b-cloud", "deepseek-v3.2:cloud"],
        "description": "Comprehensive research"
    },
    "agent": {
        "preferred": "kimi-k2.5:cloud",
        "fallbacks": ["devstral-2:123b-cloud", "deepseek-v3.2:cloud"],
        "description": "Autonomous task execution"
    },
    "code": {
        "preferred": "qwen3-coder:480b-cloud",
        "fallbacks": ["devstral-2:123b-cloud", "qwen3-coder-next:cloud"],
        "description": "Code generation and analysis"
    },
    "vision": {
        "preferred": "qwen3-vl:235b-cloud",
        "fallbacks": ["kimi-k2.5:cloud", "gemini-3-flash-preview:cloud"],
        "description": "Image analysis"
    },
    "deep_research": {
        "preferred": "kimi-k2-thinking:cloud",
        "fallbacks": ["qwen3.5:397b-cloud", "cogito-2.1:671b-cloud"],
        "description": "Multi-source deep research with page reading"
    },
    "swarm": {
        "preferred": "qwen3.5:397b-cloud",
        "fallbacks": ["cogito-2.1:671b-cloud", "kimi-k2.5:cloud"],
        "description": "Multi-agent parallel collaboration"
    }
}


def detect_action_mode(message: str) -> Optional[str]:
    """Detect action mode from trigger words in message.

    Scans the message for trigger words and returns the corresponding action mode.
    Trigger words can appear anywhere in the message (not just at the start).

    Returns:
        'search', 'research', 'agent', 'code', 'vision', or None
    """
    msg_lower = message.lower().strip()

    # Check for trigger words (longer phrases first to avoid partial matches)
    # Sort by length descending so "search online" matches before "search"
    sorted_triggers = sorted(ACTION_TRIGGERS.keys(), key=len, reverse=True)

    for trigger in sorted_triggers:
        if trigger in msg_lower:
            mode = ACTION_TRIGGERS[trigger]
            logger.info(f"[ActionMode] Trigger '{trigger}' detected -> mode: {mode}")
            return mode

    return None


_available_models: set = set()
_models_loaded: bool = False
_models_loaded_at: float = 0.0
_MODELS_TTL_SECONDS: int = 300  # 5 minutes
_models_lock = threading.Lock()


def _load_available_models() -> None:
    """Populate _available_models cache from Ollama. Thread-safe, auto-refreshes every 5 minutes."""
    import time as _time
    global _available_models, _models_loaded, _models_loaded_at
    with _models_lock:
        now = _time.time()
        if _models_loaded and (now - _models_loaded_at) < _MODELS_TTL_SECONDS:
            return  # Cache still fresh
        try:
            import ollama as _ollama_client
            result = _ollama_client.list()
            _available_models = {m.model for m in result.models}
            _models_loaded_at = now
            _models_loaded = True
            logger.info(f"[AutoModel] Loaded {len(_available_models)} models (refreshed)")
        except Exception as e:
            logger.warning(f"[AutoModel] Could not list Ollama models: {e}")
            if not _models_loaded:
                _available_models = set()
            # On refresh failure: keep old data, don't reset _models_loaded


def _is_model_available(model: str) -> bool:
    """Check if a model is available locally or is a cloud model."""
    if not model:
        return False
    # Cloud models end with -cloud or :cloud; Ollama.com serves them on demand
    if model.endswith(("-cloud", ":cloud")):
        return True
    return model in _available_models


def get_model_for_action(action_mode: str) -> Optional[str]:
    """Get the best available model for an action mode.

    Checks Ollama availability before returning a model.
    Returns None to use the default model if nothing matches.
    """
    if action_mode not in ACTION_MODE_MODELS:
        return None

    _load_available_models()

    config = ACTION_MODE_MODELS[action_mode]
    candidates = [config.get("preferred")] + config.get("fallbacks", [])

    for model in candidates:
        if _is_model_available(model):
            logger.info(f"[AutoModel] Action '{action_mode}' -> {model}")
            return model

    logger.warning(f"[AutoModel] No available model for action '{action_mode}'")
    return None


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
        self._initializing = False
        self._initialized = True
        logger.info("[AgentService] Singleton initialized")

    @property
    def is_ready(self) -> bool:
        """Check if agent is ready."""
        return self._agent is not None

    def start_background_init(self) -> None:
        """Start agent initialization in a background thread.

        Non-blocking - returns immediately. The agent will be available
        once initialization completes.
        """
        with self._agent_lock:
            if self._agent is not None or self._initializing:
                return
            self._initializing = True
        # Start thread OUTSIDE the lock
        thread = threading.Thread(target=self._background_init, daemon=True)
        thread.start()
        logger.info("[AgentService] Background initialization started")

    def _background_init(self) -> None:
        """Initialize agent in background thread."""
        try:
            self.initialize(fast_init=False)
        except Exception as e:
            logger.error(f"[AgentService] Background init failed: {e}")
        finally:
            self._initializing = False

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

                # Start Real Inner Thoughts Engine
                try:
                    from api.services.inner_thoughts_engine import get_inner_thoughts_engine
                    engine = get_inner_thoughts_engine()
                    engine.start(self._agent.brain)
                    logger.info("[AgentService] Inner Thoughts Engine started")
                except Exception as e:
                    logger.warning(f"[AgentService] Inner Thoughts Engine failed to start: {e}")

    @property
    def agent(self) -> ApprenticeAgent:
        """Get the agent instance, initializing if needed."""
        if self._agent is None:
            if self._initializing:
                # Wait for background init to complete (up to 30s)
                import time
                deadline = time.time() + 30
                while self._agent is None and time.time() < deadline:
                    time.sleep(0.2)
            if self._agent is None:
                self.initialize(fast_init=False)
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
            # Handle /think command for System 1/2 switching
            if message.strip().startswith("/think"):
                return self._handle_thinking_mode_command(message.strip())

            # Detect action mode and auto-select model
            detected_action = detect_action_mode(message)
            effective_model = model_override

            # === Record real thought: processing begins ===
            _record_thought("analyzing", f"processing: {message[:60]}...", 0.7, "service")

            # Record interaction for ALMA emotional drift (Phase 2D)
            try:
                from aura.emotion.alma_engine import alma_engine
                alma_engine.record_interaction(success=True)
            except Exception:
                pass

            # Active inference outcome learning: if user replied within 60s
            # of a proactive message, record as engaged
            try:
                import time as _time
                from aura.proactive.gateway_daemon import get_gateway_daemon
                daemon = get_gateway_daemon()
                time_since_proactive = _time.time() - daemon._last_proactive_message_time
                if 0 < time_since_proactive < 60:
                    daemon.record_user_response(engaged=True, response_type="replied")
            except Exception:
                pass

            # Track context for heatmap (covers direct handlers that bypass agent.chat)
            try:
                from api.routes.context import track_context_from_message
                track_context_from_message(message, is_user=True)
            except Exception:
                pass

            # Record activity for idle panel
            try:
                from api.routes.idle_behaviors import record_user_activity
                record_user_activity()
            except Exception:
                pass

            if not effective_model and detected_action:
                effective_model = get_model_for_action(detected_action)
                if effective_model:
                    logger.info(f"[AgentService] Chat auto-selected model for {detected_action}: {effective_model}")
                    _record_thought("connecting", f"action mode: {detected_action} -> {effective_model}", 0.5, "service")

            # Set model override if we have one
            if effective_model:
                self.agent.brain.set_model_override(effective_model)
                logger.info(f"[AgentService] Using model: {effective_model}")

            try:
                # ===== SWARM MODE HANDLER =====
                if detected_action == "swarm":
                    import concurrent.futures

                    logger.info(f"[AgentService] Swarm mode (REST) for: {message[:50]}...")
                    _record_thought("connecting", "activating multi-agent swarm: Research, Analyst, Creative, Strategist", 0.8, "service")

                    # Check if query needs real-time data (news, latest, current, etc.)
                    needs_search_keywords = [
                        "news", "latest", "current", "recent", "today", "now",
                        "update", "happening", "trending", "2024", "2025", "2026",
                        "research", "developments", "breakthroughs", "announced"
                    ]
                    msg_lower = message.lower()
                    needs_search = any(kw in msg_lower for kw in needs_search_keywords)

                    # Gather real data first if needed
                    search_context = ""
                    if needs_search:
                        logger.info("[AgentService] Swarm: Gathering real-time data first...")
                        try:
                            # Extract topic from message (remove swarm trigger words)
                            topic = msg_lower
                            for trigger in ["swarm", "multi-agent", "multiple agents", "team research", "collaborative", "all agents", "agent team"]:
                                topic = topic.replace(trigger, "").strip()
                            topic = topic.strip(" :,.-")

                            # Use the search tool to get real data
                            from aura.tools.web_search import WebSearchTool
                            search_tool = WebSearchTool()
                            search_results = search_tool.search(topic, num_results=8)

                            if search_results.get("success") and search_results.get("results"):
                                results_text = []
                                for i, r in enumerate(search_results["results"][:8], 1):
                                    results_text.append(f"{i}. **{r.get('title', 'No title')}**\n   {r.get('snippet', '')}\n   Source: {r.get('url', '')}")
                                search_context = f"\n\n**Search Results for '{topic}':**\n\n" + "\n\n".join(results_text) + "\n\n---\n\n"
                                logger.info(f"[AgentService] Swarm: Got {len(search_results['results'])} search results")
                            else:
                                logger.warning(f"[AgentService] Swarm: Search returned no results")
                        except Exception as e:
                            logger.error(f"[AgentService] Swarm search error: {e}")

                    # Build the prompt with search context if available
                    agent_prompt = message
                    if search_context:
                        agent_prompt = f"Based on the following real-time search results, analyze and respond to: {message}\n{search_context}\nUse the search results above to provide informed, factual analysis."

                    agents = {
                        "Research": "You are a Research Agent. Analyze the provided data and extract key facts, findings, and evidence. Cite sources when available.",
                        "Analyst": "You are an Analyst Agent. Provide critical analysis of the data, identify patterns, trends, and assess implications.",
                        "Creative": "You are a Creative Agent. Think outside the box, find unexpected connections, and propose innovative perspectives on the data.",
                        "Strategist": "You are a Strategy Agent. Consider long-term implications, identify opportunities and risks, and provide actionable recommendations."
                    }

                    results = {}
                    def run_agent(name, system_prompt):
                        try:
                            return name, self.agent.brain.think(agent_prompt, system_prompt=system_prompt, use_history=False)
                        except Exception as e:
                            return name, f"Error: {e}"

                    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                        futures = {executor.submit(run_agent, name, prompt): name for name, prompt in agents.items()}
                        for future in concurrent.futures.as_completed(futures, timeout=120):
                            try:
                                name, response = future.result()
                                results[name] = response
                            except Exception as e:
                                results[futures[future]] = f"Error: {e}"

                    # Build response
                    mode_text = "parallel + search" if search_context else "parallel"
                    header = f"## Agent Swarm\n\n**Agents:** Research, Analyst, Creative, Strategist\n**Mode:** {mode_text}\n\n---\n"
                    response_parts = [header]

                    for name, resp in results.items():
                        response_parts.append(f"### {name} Agent\n\n{resp}\n\n---\n")

                    # Synthesis
                    if len(results) >= 2:
                        synthesis_prompt = f"""You are synthesizing insights from multiple AI agents who analyzed: "{message}"

Here are their perspectives:

{chr(10).join([f"**{name}:** {resp[:1500]}" for name, resp in results.items()])}

Write a well-formatted synthesis in readable prose (NOT JSON). Use this structure:

**Key Consensus Points**
- List the main points where agents agree

**Diverse Perspectives**
- Note any different angles or unique insights from specific agents

**Conclusion**
Write a brief integrative conclusion that captures the essential takeaways.

Keep it concise, readable, and well-formatted with markdown."""
                        synthesis = self.agent.brain.think(synthesis_prompt)
                        response_parts.append(f"### Synthesis\n\n{synthesis}")

                    return {
                        "response": "\n".join(response_parts),
                        "fast_path": False,
                        "mood": self._get_mood(),
                        "model_used": effective_model or "swarm"
                    }

                # ===== DEEP RESEARCH HANDLER =====
                if detected_action == "deep_research":
                    _record_thought("analyzing", "initiating deep research pipeline...", 0.8, "service")
                    from aura.tools.deep_research import DeepResearchTool
                    deep_tool = DeepResearchTool()

                    topic = message.lower()
                    for trigger in ["deep research", "thorough research", "extensive research"]:
                        topic = topic.replace(trigger, "").strip()
                    topic = topic.strip(" on about for")

                    result = deep_tool.research(topic, depth="deep")

                    if result.get("success"):
                        synthesis_prompt = f"""Summarize this research on '{topic}':
{result.get('content', '')[:8000]}

Provide key findings and cite sources."""
                        synthesized = self.agent.brain.think(synthesis_prompt)
                        response = f"## Deep Research: {topic}**\n\n{synthesized}\n\n---\n*{result.get('summary', '')}*"
                    else:
                        response = f"Research failed: {result.get('error', 'Unknown error')}"

                    return {
                        "response": response,
                        "fast_path": False,
                        "mood": self._get_mood(),
                        "model_used": effective_model or "deep_research"
                    }

                # Screen context injection (Phase 3D)
                screen_hint = ""
                try:
                    from aura.tools.screenpipe import get_screenpipe_client
                    sp = get_screenpipe_client()
                    if sp.is_available():
                        ctx = sp.get_screen_context(minutes=1, max_chars=500)
                        if ctx.get("available") and ctx.get("current_app"):
                            screen_hint = (
                                f"\n[Screen context: user is in {ctx['current_app']}"
                                + (f" — {ctx['current_window']}" if ctx.get("current_window") else "")
                                + (". Error visible on screen." if ctx.get("has_errors") else "")
                                + "]"
                            )
                except Exception:
                    pass

                enriched_msg = message + screen_hint if screen_hint else message

                # Use agent.chat() which has direct handlers for search/crypto
                response = self.agent.chat(enriched_msg, speak=speak)

                # Truth Spine: classify response and tag to VerifiedMemory (non-blocking)
                # Uses module-level singleton to avoid per-call disk reads.
                try:
                    _verified_mem = _get_truth_spine()
                    _resp_lower = response.lower()
                    import re as _re
                    # FACT: actual URL present (not just the word "verified")
                    _has_url = bool(_re.search(r'https?://\S+', response))
                    _has_artifact = any(m in response for m in ["sha256:", "✓"])
                    if _has_url or _has_artifact:
                        _tier = MemoryTier.BELIEF  # URL ≠ verified fact; downgrade to BELIEF
                        _verified_mem.store_belief(
                            content=response[:500],
                            source="chat",
                            reasoning="Response cites a URL or artifact (unverified by agent)"
                        )
                    elif any(m in _resp_lower for m in ["i think", "i believe", "likely", "probably", "it seems"]):
                        _tier = MemoryTier.BELIEF
                        _verified_mem.store_belief(
                            content=response[:500],
                            source="chat",
                            reasoning="Response contains inference or belief language"
                        )
                    else:
                        _tier = MemoryTier.SPECULATION
                        _verified_mem.store_speculation(
                            content=response[:500],
                            source="chat",
                            reason="Unverified LLM output"
                        )
                    logger.debug(f"[TruthSpine] Response classified as {_tier.value}")
                except Exception as _ts_err:
                    logger.debug(f"[TruthSpine] Classification error: {_ts_err}")

                return {
                    "response": response,
                    "fast_path": self._was_fast_path(message),
                    "mood": self._get_mood(),
                    "model_used": self.agent.brain.get_last_model_used()
                }
            finally:
                # Clear model override after request
                if effective_model:
                    self.agent.brain.set_model_override(None)

    def chat_stream(self, message: str, model_override: Optional[str] = None, action_mode: Optional[str] = None):
        """Stream a chat response from the agent.

        Args:
            message: User message
            model_override: Optional model to use (explicit selection takes priority)
            action_mode: Optional action mode ('search', 'research', 'agent')

        Yields:
            Response chunks as they're generated

        Note:
            CRITICAL FIX: Lock is only held briefly for setup/teardown, NOT during streaming.
            This prevents blocking concurrent requests (WebSocket, health checks, etc.).
        """
        brain = None
        effective_model = None
        # ===== SETUP PHASE - Brief lock =====
        with self._agent_lock:
            effective_model = model_override
            detected_action = action_mode or detect_action_mode(message)

            if not effective_model and detected_action:
                effective_model = get_model_for_action(detected_action)
                if effective_model:
                    logger.info(f"[AgentService] Auto-selected model for {detected_action}: {effective_model}")

            if effective_model:
                self.agent.brain.set_model_override(effective_model)
                logger.info(f"[AgentService] Streaming with model: {effective_model}")

            # Get references we need (agent is thread-safe for reads)
            agent = self.agent
            brain = self.agent.brain
        # ===== END SETUP - Lock released, streaming can proceed without blocking =====

        try:
            # === Record real thought: stream processing begins ===
            _record_thought("formulating", f"streaming: {message[:60]}...", 0.7, "service")

            # Track context for heatmap (covers direct handlers that bypass agent.chat)
            try:
                from api.routes.context import track_context_from_message
                track_context_from_message(message, is_user=True)
            except Exception:
                pass

            # Record activity for idle panel
            try:
                from api.routes.idle_behaviors import record_user_activity
                record_user_activity()
            except Exception:
                pass

            # ===== DIRECT SEARCH HANDLER =====
            # Check for direct search before streaming to prevent query hallucination
            if hasattr(agent, '_handle_direct_search'):
                search_response = agent._handle_direct_search(message)
                if search_response:
                    logger.info("[AgentService] Direct search handled, returning result")
                    yield {"type": "chunk", "content": search_response}
                    yield {"type": "done", "mood": self._get_mood(), "model_used": "direct_search"}
                    return

            # ===== DIRECT CRYPTO HANDLER =====
            if hasattr(agent, '_handle_direct_crypto'):
                crypto_response = agent._handle_direct_crypto(message)
                if crypto_response:
                    logger.info("[AgentService] Direct crypto handled, returning result")
                    yield {"type": "chunk", "content": crypto_response}
                    yield {"type": "done", "mood": self._get_mood(), "model_used": "direct_crypto"}
                    return

            # ===== DEEP RESEARCH HANDLER =====
            if detected_action == "deep_research":
                try:
                    from aura.tools.deep_research import DeepResearchTool
                    deep_tool = DeepResearchTool()

                    topic = message.lower()
                    for trigger in ["deep research", "thorough research", "extensive research", "full research", "research everything", "research in depth"]:
                        topic = topic.replace(trigger, "").strip()
                    topic = topic.strip(" on about for")

                    logger.info(f"[AgentService] Deep research on: {topic}")
                    yield {"type": "chunk", "content": f"## Deep Research: {topic}\n\n"}

                    result = deep_tool.research(topic, depth="deep")

                    if result.get("success"):
                        synthesis_prompt = f"""Based on this deep research, provide a comprehensive summary:

Topic: {topic}
Sources Found: {result.get('urls_found', 0)}
Pages Read: {result.get('pages_read', 0)}

Content:
{result.get('content', '')[:8000]}

Provide a well-structured, informative summary with key findings and cite sources."""

                        synthesized = brain.think(synthesis_prompt)
                        yield {"type": "chunk", "content": synthesized}
                        yield {"type": "chunk", "content": f"\n\n---\n*{result.get('summary', '')}*"}
                    else:
                        yield {"type": "chunk", "content": f"Research failed: {result.get('error', 'Unknown error')}"}

                    yield {"type": "done", "mood": self._get_mood(), "model_used": effective_model or "deep_research"}
                    return
                except Exception as e:
                    logger.error(f"[AgentService] Deep research error: {e}")
                    yield {"type": "chunk", "content": f"Deep research error: {e}"}
                    yield {"type": "done", "mood": self._get_mood(), "model_used": "error"}
                    return

            # ===== SWARM/MULTI-AGENT HANDLER =====
            if detected_action == "swarm":
                try:
                    import concurrent.futures

                    logger.info(f"[AgentService] Swarm mode activated for: {message[:50]}...")
                    yield {"type": "chunk", "content": "## Agent Swarm\n\n"}

                    needs_search_keywords = [
                        "news", "latest", "current", "recent", "today", "now",
                        "update", "happening", "trending", "2024", "2025", "2026",
                        "research", "developments", "breakthroughs", "announced"
                    ]
                    msg_lower = message.lower()
                    needs_search = any(kw in msg_lower for kw in needs_search_keywords)

                    search_context = ""
                    if needs_search:
                        yield {"type": "chunk", "content": "Gathering real-time data...\n\n"}
                        try:
                            topic = msg_lower
                            for trigger in ["swarm", "multi-agent", "multiple agents", "team research", "collaborative", "all agents", "agent team"]:
                                topic = topic.replace(trigger, "").strip()
                            topic = topic.strip(" :,.-")

                            from aura.tools.web_search import WebSearchTool
                            search_tool = WebSearchTool()
                            search_results = search_tool.search(topic, num_results=8)

                            if search_results.get("success") and search_results.get("results"):
                                results_text = []
                                for i, r in enumerate(search_results["results"][:8], 1):
                                    results_text.append(f"{i}. **{r.get('title', 'No title')}**\n   {r.get('snippet', '')}\n   Source: {r.get('url', '')}")
                                search_context = f"\n\n**Search Results for '{topic}':**\n\n" + "\n\n".join(results_text) + "\n\n---\n\n"
                                yield {"type": "chunk", "content": f"Found {len(search_results['results'])} sources.\n\n"}
                        except Exception as e:
                            logger.error(f"[AgentService] Swarm search error: {e}")

                    agent_prompt = message
                    if search_context:
                        agent_prompt = f"Based on the following real-time search results, analyze: {message}\n{search_context}\nUse the search results to provide informed analysis."

                    agents_config = {
                        "Research": "You are a Research Agent. Analyze the provided data and extract key facts, findings, and evidence. Cite sources when available.",
                        "Analyst": "You are an Analyst Agent. Provide critical analysis of the data, identify patterns, trends, and assess implications.",
                        "Creative": "You are a Creative Agent. Think outside the box, find unexpected connections, and propose innovative perspectives.",
                        "Strategist": "You are a Strategy Agent. Consider long-term implications, identify opportunities and risks, and provide actionable recommendations."
                    }

                    mode_text = "parallel + search" if search_context else "parallel"
                    yield {"type": "chunk", "content": f"**Agents:** {', '.join(agents_config.keys())}\n**Mode:** {mode_text}\n\n---\n\n"}

                    # Execute all agents in parallel (brain reference captured at setup)
                    results = {}
                    def run_agent(name, system_prompt):
                        try:
                            response = brain.think(
                                agent_prompt,
                                system_prompt=system_prompt,
                                use_history=False
                            )
                            return name, response
                        except Exception as e:
                            return name, f"Error: {e}"

                    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                        futures = {executor.submit(run_agent, name, prompt): name for name, prompt in agents_config.items()}

                        for future in concurrent.futures.as_completed(futures, timeout=120):
                            try:
                                name, response = future.result()
                                results[name] = response
                                yield {"type": "chunk", "content": f"### {name} Agent\n\n{response}\n\n---\n\n"}
                            except Exception as e:
                                name = futures[future]
                                yield {"type": "chunk", "content": f"### {name} Agent\n\nError: {e}\n\n---\n\n"}

                    if len(results) >= 2:
                        yield {"type": "chunk", "content": "### Synthesis\n\n"}

                        synthesis_prompt = f"""You are synthesizing insights from multiple AI agents who analyzed: "{message}"

Here are their perspectives:

{chr(10).join([f"**{name}:** {resp[:1500]}" for name, resp in results.items()])}

Write a well-formatted synthesis in readable prose (NOT JSON). Use this structure:

**Key Consensus Points**
- List the main points where agents agree

**Diverse Perspectives**
- Note any different angles or unique insights from specific agents

**Conclusion**
Write a brief integrative conclusion that captures the essential takeaways.

Keep it concise, readable, and well-formatted with markdown."""

                        # Stream the synthesis
                        if hasattr(brain, 'think_stream'):
                            for chunk in brain.think_stream(synthesis_prompt):
                                yield {"type": "chunk", "content": chunk}
                        else:
                            synthesis = brain.think(synthesis_prompt)
                            yield {"type": "chunk", "content": synthesis}

                    yield {"type": "done", "mood": self._get_mood(), "model_used": effective_model or "swarm"}
                    return
                except Exception as e:
                    logger.error(f"[AgentService] Swarm mode error: {e}")
                    yield {"type": "chunk", "content": f"Swarm mode error: {e}"}
                    yield {"type": "done", "mood": self._get_mood(), "model_used": "error"}
                    return

            # ===== DIRECT CODE HANDLER =====
            if hasattr(agent, '_handle_direct_code'):
                code_response = agent._handle_direct_code(message)
                if code_response:
                    logger.info("[AgentService] Direct code handled, returning result")
                    yield {"type": "chunk", "content": code_response}
                    yield {"type": "done", "mood": self._get_mood(), "model_used": "direct_code"}
                    return

            # ===== SCREEN CONTEXT INJECTION (Phase 3D) =====
            # Inject screen awareness so AURA knows what the user is looking at
            screen_hint = ""
            try:
                from aura.tools.screenpipe import get_screenpipe_client
                sp = get_screenpipe_client()
                if sp.is_available():
                    ctx = sp.get_screen_context(minutes=1, max_chars=500)
                    if ctx.get("available") and ctx.get("current_app"):
                        screen_hint = (
                            f"\n[Screen context: user is in {ctx['current_app']}"
                            + (f" — {ctx['current_window']}" if ctx.get("current_window") else "")
                            + (". Error visible on screen." if ctx.get("has_errors") else "")
                            + "]"
                        )
            except Exception:
                pass

            enriched_message = message + screen_hint if screen_hint else message

            # ===== STANDARD STREAMING =====
            if hasattr(brain, 'think_stream'):
                for chunk in brain.think_stream(enriched_message):
                    yield {"type": "chunk", "content": chunk}
                yield {"type": "done", "mood": self._get_mood(), "model_used": brain.get_last_model_used()}
            else:
                # Fallback to non-streaming
                response = agent.chat(message, speak=False)
                yield {"type": "chunk", "content": response}
                yield {"type": "done", "mood": self._get_mood(), "model_used": brain.get_last_model_used()}

        finally:
            # ===== TEARDOWN =====
            # NOTE: Model override NOT cleared here. The override persists for subsequent requests
            # until explicitly replaced by a new request's model_override or action detection.
            # Clearing it here would race with subsequent requests and lose user intent.
            # The override is naturally scoped to the session/context, not per-request.
            pass

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
        """Get agent status information.

        NOTE: No lock needed - only reads properties. Holding the lock here
        caused deadlocks when concurrent polling endpoints all waited for it.
        """
        if self._agent is None:
            return {
                "online": False,
                "model": "initializing...",
                "aura_enabled": False,
                "mood": None,
                "memory_count": 0,
                "query_count": 0,
                "last_model_used": None
            }

        agent = self._agent
        try:
            return {
                "online": True,
                "model": getattr(agent.brain, 'model', 'unknown'),
                "aura_enabled": getattr(agent, 'aura_enabled', False),
                "mood": self._get_mood(),
                "memory_count": len(agent.memory.memories) if hasattr(agent.memory, 'memories') else 0,
                "query_count": getattr(agent.brain, '_total_query_count', 0),
                "last_model_used": agent.brain.get_last_model_used()
            }
        except Exception as e:
            logger.error(f"[AgentService] get_status error: {e}")
            return {
                "online": True,
                "model": "error",
                "aura_enabled": False,
                "mood": None,
                "memory_count": 0,
                "query_count": 0,
                "last_model_used": None
            }

    def clear_history(self) -> bool:
        """Clear conversation history."""
        try:
            if self._agent is not None:
                self._agent.brain.clear_history()
                return True
            return False
        except Exception as e:
            logger.error(f"[AgentService] Failed to clear history: {e}")
            return False

    # =========================================================================
    # Multi-Conversation Management
    # =========================================================================

    def create_conversation(self, title: Optional[str] = None) -> Dict[str, Any]:
        """Create a new conversation."""
        if self._agent is None:
            return {"error": "Agent not initialized"}
        conv_id = self._agent.brain.create_conversation(title)
        return {"id": conv_id, "title": title or "New Chat", "messages": []}

    def list_conversations(self) -> list:
        """List all conversations."""
        if self._agent is None:
            return []
        return self._agent.brain.list_conversations()

    def switch_conversation(self, conversation_id: str) -> Dict[str, Any]:
        """Switch to a different conversation."""
        if self._agent is None:
            return {"error": "Agent not initialized"}

        success = self._agent.brain.switch_conversation(conversation_id)
        if not success:
            return {"error": f"Conversation not found: {conversation_id}"}

        # Reset transient context systems
        self._reset_conversation_context()

        messages = list(self._agent.brain.conversation_history)
        # Get title from index
        convs = self._agent.brain.list_conversations()
        title = "Unknown"
        for c in convs:
            if c["id"] == conversation_id:
                title = c["title"]
                break

        return {"id": conversation_id, "title": title, "messages": messages}

    def delete_conversation(self, conversation_id: str) -> Dict[str, Any]:
        """Delete a conversation."""
        if self._agent is None:
            return {"success": False, "error": "Agent not initialized"}
        success = self._agent.brain.delete_conversation(conversation_id)
        return {
            "success": success,
            "new_active_id": self._agent.brain.get_current_conversation_id(),
        }

    def rename_conversation(self, conversation_id: str, title: str) -> bool:
        """Rename a conversation."""
        if self._agent is None:
            return False
        return self._agent.brain.rename_conversation(conversation_id, title)

    def save_conversation_to_memory(self, conversation_id: Optional[str] = None) -> Dict[str, Any]:
        """Save a conversation to AURA's long-term memory."""
        if self._agent is None:
            return {"success": False, "error": "Agent not initialized"}
        return self._agent.brain.save_conversation_to_memory(conversation_id)

    def _reset_conversation_context(self):
        """Reset transient systems when switching conversations."""
        # Clear context heatmap topics
        try:
            from api.routes.context import get_tracker
            get_tracker().clear()
        except Exception:
            pass

        # Clear active thinking state
        try:
            from api.routes.thinking import get_manager
            get_manager().clear_active()
        except Exception:
            pass

    def _get_mood(self) -> Optional[MoodState]:
        """Extract AURA's current mood from ALMA emotional engine.

        NOTE: This method must NOT acquire _agent_lock since it's called from
        contexts that may or may not already hold the lock.
        """
        try:
            # Try ALMA directly first (most reliable - no agent lock needed)
            if ALMA_AVAILABLE and alma_engine:
                try:
                    alma_state = alma_engine.get_emotional_state()
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
                        return mood
                except Exception as e:
                    logger.debug(f"[AgentService] ALMA direct state error: {e}")

            agent = self._agent
            if agent is None:
                return MoodState(
                    emotion='neutral', confidence=50,
                    valence=0.3, arousal=0.1, dominance=0.3, emoji='🤖'
                )

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

    def _handle_thinking_mode_command(self, message: str) -> Dict[str, Any]:
        """Handle /think s1|s2|auto|status commands."""
        try:
            from aura.thinking_mode import get_thinking_mode_manager, ThinkingMode
            tmm = get_thinking_mode_manager()

            parts = message.split(maxsplit=1)
            arg = parts[1].strip().lower() if len(parts) > 1 else "status"

            if arg in ("s1", "system1"):
                tmm.mode = ThinkingMode.SYSTEM1
                response = "Thinking mode set to **System 1** (fast/intuitive). All queries will use the fast model."
            elif arg in ("s2", "system2"):
                tmm.mode = ThinkingMode.SYSTEM2
                response = "Thinking mode set to **System 2** (deliberative/reasoning). All queries will use the reasoning model."
            elif arg == "auto":
                tmm.mode = ThinkingMode.AUTO
                response = "Thinking mode set to **Auto**. The system will choose S1/S2 based on query complexity and cognitive load."
            elif arg == "status":
                state = tmm.get_state()
                load = state["cognitive_load"]
                response = (
                    f"**Thinking Mode:** {state['mode']}\n"
                    f"**Cognitive Load:** {load['load_score']:.2f} "
                    f"(window: {load['window_size']}, suggestion: {load['suggestion']})"
                )
            else:
                response = "Usage: `/think s1` | `/think s2` | `/think auto` | `/think status`"

            return {
                "response": response,
                "fast_path": True,
                "mood": self._get_mood(),
                "model_used": "command",
            }
        except Exception as e:
            return {
                "response": f"Thinking mode command failed: {e}",
                "fast_path": True,
                "mood": "neutral",
                "model_used": "command",
            }

    def _was_fast_path(self, message: str) -> bool:
        """Check if message was handled via fast path."""
        try:
            return self.agent._is_simple_query(message)
        except Exception:
            return False

    def get_available_models(self) -> Dict[str, Any]:
        """Get list of available models (local and cloud).

        NOTE: No lock needed - reads config and makes HTTP request to Ollama.
        Holding the lock here while waiting on Ollama HTTP caused deadlocks.
        """
        try:
            import requests
            from aura.config import VERIFIED_LOCAL_MODELS, VERIFIED_CLOUD_MODELS

            local_models = []
            cloud_models = list(VERIFIED_CLOUD_MODELS)

            # Non-chat models to exclude from the selector (embeddings, OCR, etc.)
            NON_CHAT_MODELS = {"nomic-embed-text:latest", "glm-ocr:latest"}

            # Get locally installed models from Ollama — keep only true local chat models
            try:
                ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
                response = requests.get(f"{ollama_host}/api/tags", timeout=3)
                if response.status_code == 200:
                    all_ollama = [m["name"] for m in response.json().get("models", [])]
                    # Cloud models (handled by cloud_models list) and non-chat models excluded
                    local_models = [
                        m for m in all_ollama
                        if not m.endswith(("-cloud", ":cloud"))
                        and m not in NON_CHAT_MODELS
                    ]
            except Exception as e:
                logger.warning(f"[AgentService] Could not fetch local models: {e}")
                local_models = []

            current_model = "auto"
            if self._agent is not None:
                current_model = getattr(self._agent.brain, 'model', 'auto')

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
