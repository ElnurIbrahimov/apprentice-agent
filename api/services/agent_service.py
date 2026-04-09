"""Singleton wrapper for ApprenticeAgent."""

import logging
import os
import sys
import threading
import time
from typing import Any, Dict, Optional

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from api.models.schemas import MoodState
from aura import ApprenticeAgent
from aura.core.conversation_manager import get_conversation_manager

# Import ALMA directly for mood detection
try:
    from aura.emotion.alma_engine import alma_engine
    from aura.emotion.integration import get_mood_emoji
    ALMA_AVAILABLE = True
except ImportError:
    ALMA_AVAILABLE = False
    alma_engine = None

logger = logging.getLogger(__name__)

import re


def _filter_skill_json(text: str) -> str:
    """Remove skill learning JSON artifacts from chat responses.

    The skill learner sometimes bleeds raw JSON into responses like:
    {"name": "...", "trigger_patterns": [...], "procedure": "..."}
    This should never be shown to the user.
    """
    # Remove "Analyze these successful interactions..." prompt blocks
    text = re.sub(
        r'Analyze these successful interactions and extract a reusable skill\..*?Respond ONLY with the JSON,? no other text\.?',
        '', text, flags=re.DOTALL
    )
    # Remove skill definition JSON blocks
    text = re.sub(
        r'\{[^{}]*"name"\s*:.*?"trigger_patterns"\s*:.*?"procedure"\s*:.*?\}',
        '', text, flags=re.DOTALL
    )
    # Remove "Create a skill definition with:" instruction blocks
    text = re.sub(
        r'Create a skill definition with:.*?Respond in this exact JSON format:',
        '', text, flags=re.DOTALL
    )
    # Clean up excessive whitespace left behind
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

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
                    from aura.truth_spine import MemoryTier, VerifiedMemory
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
        logger.debug("record_thought_failed", exc_info=True)

# =============================================================================
#                    ACTION MODE TRIGGER SYSTEM
# =============================================================================

# Trigger words that activate different agent modes
# Format: trigger_word -> (action_mode, model_config)

ACTION_TRIGGERS = {
    # ===== FRONTEND / UI / DESIGN MODE =====
    # Only specific UI terms — NO generic verbs like "build a", "create a"
    "landing page": "frontend",
    "dashboard design": "frontend",
    "web page": "frontend",
    "web app": "frontend",
    "webapp": "frontend",
    "website": "frontend",
    "frontend": "frontend",
    "user interface": "frontend",
    "design a page": "frontend",
    "design a site": "frontend",
    "design a ui": "frontend",
    "build a page": "frontend",
    "build a site": "frontend",
    "build a website": "frontend",
    "build a webapp": "frontend",
    "build a web app": "frontend",
    "build a dashboard": "frontend",
    "react component": "frontend",
    "tailwind": "frontend",
    "pricing page": "frontend",
    "signup page": "frontend",
    "login page": "frontend",
    "settings page": "frontend",

    # ===== RAPID / PROTOTYPE MODE =====
    "quick prototype": "rapid",
    "rapid prototype": "rapid",
    "scaffold": "rapid",
    "quick mock": "rapid",
    "sketch out": "rapid",

    # ===== ARTIFACT / COMPONENT MODE =====
    "artifact": "artifact",
    "ui component": "artifact",
    "react widget": "artifact",
    "build a component": "artifact",
    "create a component": "artifact",

    # ===== DEBUG / REVIEW MODE =====
    "not working": "debug",
    "fix this": "debug",
    "fix the": "debug",
    "debug": "debug",
    "find the bug": "debug",
    "why is this": "debug",
    "broken": "debug",
    "code review": "debug",

    # ===== SEARCH MODE =====
    # Quick web search - uses fastest cloud model
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

    # ===== CODE / BACKEND MODE =====
    # Code generation/analysis - uses top SWE-bench model
    "code": "code",
    "program": "code",
    "script": "code",
    "implement": "code",
    "write code": "code",
    "coding": "code",
    "refactor": "code",
    "optimize code": "code",
    "backend": "code",
    "api": "code",
    "database": "code",
    "server": "code",

    # ===== VISION MODE =====
    # Image analysis - uses best multimodal model
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
    "fleet": "swarm",
}

# Best models for each action mode — optimized for design quality (2026-03 research)
ACTION_MODE_MODELS = {
    "frontend": {
        "preferred": "kimi-k2.5:cloud",
        "fallbacks": ["chatgpt:gpt-5.3-codex"],
        "description": "Frontend/UI/design — best vision-to-code, polished UI"
    },
    "rapid": {
        "preferred": "chatgpt:gpt-5.3-codex-spark",
        "fallbacks": ["nemotron-3-super:cloud"],
        "description": "Rapid prototyping — 1000 tok/s instant iteration"
    },
    "code": {
        "preferred": "minimax-m2.5:cloud",
        "fallbacks": ["glm-5.1:cloud", "qwen3-coder:480b-cloud"],
        "description": "Backend/code — 80.2% SWE-bench top open model"
    },
    "search": {
        "preferred": "nemotron-3-super:cloud",
        "fallbacks": ["glm-5.1:cloud", "glm-5:cloud"],
        "description": "Quick web search — fastest model"
    },
    "research": {
        "preferred": "qwen3.5:397b-cloud",
        "fallbacks": ["kimi-k2.5:cloud"],
        "description": "Comprehensive research — best reasoning + 256K context"
    },
    "deep_research": {
        "preferred": "qwen3.5:397b-cloud",
        "fallbacks": ["kimi-k2.5:cloud"],
        "description": "Multi-source deep research — best reasoning + 256K context"
    },
    "debug": {
        "preferred": "chatgpt:gpt-5.4-thinking",
        "fallbacks": ["glm-5.1:cloud", "minimax-m2.7:cloud"],
        "description": "Debug/review — extended thinking for hard bugs"
    },
    "vision": {
        "preferred": "kimi-k2.5:cloud",
        "fallbacks": ["chatgpt:gpt-5.4"],
        "description": "Image analysis — best multimodal"
    },
    "swarm": {
        "preferred": "minimax-m2.7:cloud",
        "fallbacks": ["qwen3.5:397b-cloud"],
        "description": "Multi-agent swarm — 1M context, self-evolving"
    },
    "artifact": {
        "preferred": "kimi-k2.5:cloud",
        "fallbacks": ["minimax-m2.5:cloud"],
        "description": "UI components/artifacts — best for generating UI pieces"
    },
    "agent": {
        "preferred": "kimi-k2.5:cloud",
        "fallbacks": ["glm-5.1:cloud", "minimax-m2.7:cloud"],
        "description": "Autonomous task execution"
    },
}


# ---------------------------------------------------------------------------
#  LLM-based intent classifier (cached client singleton)
# ---------------------------------------------------------------------------
_classifier_client = None
_classifier_client_lock = threading.Lock()


def _get_classifier_client():
    """Get or create a cached Ollama cloud client for intent classification."""
    global _classifier_client
    if _classifier_client is None:
        with _classifier_client_lock:
            if _classifier_client is None:
                try:
                    import ollama
                    api_key = os.getenv("OLLAMA_API_KEY", "")
                    if api_key and not api_key.startswith("YOUR_"):
                        _classifier_client = ollama.Client(
                            host="https://api.ollama.com",
                            headers={"Authorization": f"Bearer {api_key}"}
                        )
                        logger.info("[ActionMode] LLM classifier client initialized")
                    else:
                        logger.debug("[ActionMode] No OLLAMA_API_KEY — LLM classifier unavailable")
                except Exception as e:
                    logger.warning(f"[ActionMode] Failed to create classifier client: {e}")
    return _classifier_client


_CLASSIFICATION_PROMPT = """Classify this user request into ONE category. Reply with ONLY the category name, nothing else.

Categories:
- frontend: Building websites, web pages, landing pages, dashboards, UI components, React/HTML/CSS
- code: Backend code, APIs, databases, scripts, algorithms, non-UI programming
- debug: Fixing bugs, errors, debugging, code review
- search: Looking up information online, web search
- deep_research: Extensive multi-source research, thorough investigation
- research: Deep analysis, comprehensive research, investigation
- vision: Analyzing images, screenshots, visual content
- rapid: Quick prototyping, scaffolding, fast iteration
- swarm: Multi-agent collaborative tasks, team research
- agent: Autonomous multi-step task execution, automation
- artifact: Generating standalone UI components or widgets
- general: Conversation, questions, explanations, greetings, anything else

User request: "{message}"

Category:"""

_VALID_MODES = {"frontend", "code", "debug", "search", "research", "deep_research",
                "vision", "rapid", "swarm", "agent", "artifact"}


def detect_action_mode(message: str) -> Optional[str]:
    """Classify user intent using a fast LLM call (~100 tokens, <1s).

    Uses nemotron-3-super (fastest cloud model, ~415 tok/s) for classification.
    Falls back to keyword matching if LLM classification fails.

    Returns:
        'frontend', 'rapid', 'artifact', 'debug', 'search', 'research',
        'agent', 'code', 'vision', 'deep_research', 'swarm', or None
    """
    words = message.split()

    # Quick skip for short messages — most are conversational
    if len(words) < 4:
        return None

    # Skip LLM classification for casual/conversational messages (≤12 words, no task indicators)
    # This avoids 1-5s overhead and frequent misclassification as "agent"
    if len(words) <= 12:
        msg_lower = message.lower()
        _TASK_INDICATORS = {
            'create', 'build', 'make', 'generate', 'write', 'code', 'fix', 'debug',
            'search', 'find', 'look up', 'research', 'analyze', 'deploy', 'implement',
            'design', 'draw', 'render', 'screenshot', 'review', 'compare', 'test',
            'refactor', 'optimize', 'automate', 'scrape', 'crawl', 'translate',
        }
        if not any(ind in msg_lower for ind in _TASK_INDICATORS):
            return None

    # Try LLM classification first
    client = _get_classifier_client()
    if client is None:
        return _keyword_fallback(message)

    try:
        import concurrent.futures

        prompt = _CLASSIFICATION_PROMPT.format(message=message[:200])

        def _classify():
            return client.chat(
                model="nemotron-3-super:cloud",
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0, "num_predict": 10},
            )

        # 5-second timeout — if the model is slow, fall back to keywords
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(_classify)
            resp = future.result(timeout=5)

        raw = resp.get("message", {}).get("content", "").strip().lower()
        # Take first word only (model might add explanation)
        category = raw.split()[0].rstrip(".,;:") if raw else ""

        if category in _VALID_MODES:
            logger.info(f"[ActionMode] LLM classified as: {category}")
            return category

        # "general" or unrecognised → no special routing
        logger.info(f"[ActionMode] LLM returned '{category}' -> no special mode")
        return None

    except Exception as e:
        logger.warning(f"[ActionMode] LLM classification failed: {e}, using keyword fallback")
        return _keyword_fallback(message)


def _keyword_fallback(message: str) -> Optional[str]:
    """Fallback keyword-based detection when LLM classifier is unavailable."""
    import re as _re
    msg_lower = message.lower().strip()

    # Check for trigger words (longer phrases first to avoid partial matches).
    # Use word boundaries to prevent "research" matching inside "re-searching".
    sorted_triggers = sorted(ACTION_TRIGGERS.keys(), key=len, reverse=True)

    for trigger in sorted_triggers:
        # Multi-word phrases: plain substring match is fine (low false-positive risk)
        # Single words: require word boundary to avoid substring false positives
        if " " in trigger:
            if trigger in msg_lower:
                mode = ACTION_TRIGGERS[trigger]
                logger.info(f"[ActionMode] Keyword fallback '{trigger}' -> mode: {mode}")
                return mode
        else:
            if _re.search(r'\b' + _re.escape(trigger) + r'\b', msg_lower):
                mode = ACTION_TRIGGERS[trigger]
                logger.info(f"[ActionMode] Keyword fallback '{trigger}' -> mode: {mode}")
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
    candidates = [config.get("preferred"), *config.get("fallbacks", [])]

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
        self._orchestrator = None
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

                # Pre-warm UnifiedMemory backends in background
                def _prewarm_memory():
                    try:
                        from aura.memory.unified_memory import get_unified_memory
                        mem = get_unified_memory()
                        if hasattr(mem, '_init_backends'):
                            mem._init_backends()
                    except Exception as e:
                        logger.warning(f"[AgentService] Memory pre-warm failed: {e}")

                threading.Thread(target=_prewarm_memory, daemon=True, name="memory-prewarm").start()

                # Start Real Inner Thoughts Engine
                try:
                    from api.services.inner_thoughts_engine import get_inner_thoughts_engine
                    engine = get_inner_thoughts_engine()
                    engine.start(self._agent.brain)
                    logger.info("[AgentService] Inner Thoughts Engine started")
                except Exception as e:
                    logger.warning(f"[AgentService] Inner Thoughts Engine failed to start: {e}")

                # Initialize ConversationManager for cross-surface sync (Phase 2)
                try:
                    manager = get_conversation_manager()
                    manager.initialize(self._agent.brain)
                    logger.info("[AgentService] ConversationManager initialized")
                except Exception as e:
                    logger.warning(f"[AgentService] ConversationManager init failed: {e}")

    @property
    def agent(self) -> ApprenticeAgent:
        """Get the agent instance, initializing if needed."""
        if self._agent is None:
            if self._initializing:
                # Wait for background init to complete (up to 30s)
                deadline = time.time() + 30
                while self._agent is None and time.time() < deadline:
                    time.sleep(0.2)
            if self._agent is None:
                self.initialize(fast_init=False)
        return self._agent

    def chat(self, message: str, speak: bool = False, model_override: Optional[str] = None) -> Dict[str, Any]:
        """Send a chat message to the agent.

        Args:
            message: User message
            speak: Enable TTS
            model_override: Optional model to use instead of auto-selection

        Returns:
            Dict with response, fast_path flag, and mood

        Note:
            CRITICAL FIX: Lock is only held briefly for setup/teardown, NOT during
            LLM inference. Mirrors chat_stream() pattern to prevent 30-60s contention.
        """
        # ===== SETUP PHASE — Brief lock =====
        with self._agent_lock:
            # Handle /think command for System 1/2 switching
            if message.strip().startswith("/think"):
                return self._handle_thinking_mode_command(message.strip())

            # Detect action mode (still used for prompt injection, NOT model selection)
            detected_action = detect_action_mode(message)
            effective_model = model_override

            # === Record real thought: processing begins ===
            _record_thought("analyzing", f"processing: {message[:60]}...", 0.7, "service")

            # Record interaction for ALMA emotional drift (Phase 2D)
            try:
                if alma_engine is not None:
                    alma_engine.record_interaction(success=True)
            except Exception:
                logger.debug("alma_interaction_record_failed", exc_info=True)

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
                logger.debug("proactive_response_record_failed", exc_info=True)

            # Track context for heatmap (covers direct handlers that bypass agent.chat)
            try:
                from api.routes.context import track_context_from_message
                track_context_from_message(message, is_user=True)
            except Exception:
                logger.debug("chat_context_heatmap_track_failed", exc_info=True)

            # Record activity for idle panel
            try:
                from api.routes.idle_behaviors import record_user_activity
                record_user_activity()
            except Exception:
                logger.debug("chat_idle_activity_record_failed", exc_info=True)

            # Set explicit model override only if user selected one (not auto-detected)
            if effective_model:
                self.agent.brain.set_model_override(effective_model)
                logger.info(f"[AgentService] Using explicit model override: {effective_model}")

            # Set routing context on brain (the neural router reads these)
            self.agent.brain._routing_preference = 'balanced'
            self.agent.brain._conversation_id = None

            # Set action mode for context-aware prompt injection (design system, etc.)
            self.agent.brain.set_action_mode(detected_action)

            # Get references we need (agent is thread-safe for reads)
            agent = self.agent
        # ===== END SETUP — Lock released, LLM calls proceed without blocking =====

        try:
            # ===== FRONTEND VISUAL FEEDBACK HANDLER =====
            if detected_action == "frontend":
                logger.info(f"[AgentService] Frontend visual feedback for: {message[:50]}...")
                _record_thought("creating", "visual feedback loop — generate, render, screenshot, iterate", 0.8, "service")

                try:
                    from aura.tools.visual_feedback import VisualFeedbackLoop
                    vfl = VisualFeedbackLoop(brain=agent.brain)
                    result = vfl.generate_with_feedback(message, max_iterations=2)

                    # Build response with code + iteration info
                    parts = []
                    if result.get("improvements"):
                        parts.append(f"**Visual Feedback Loop** completed {result['iterations']} iteration(s) using `{result.get('model_used', 'auto')}`:")
                        for imp in result["improvements"]:
                            parts.append(f"- {imp}")
                        parts.append("")

                    code = result.get("code", "")
                    parts.append(f"```html\n{code}\n```")

                    if result.get("screenshot_path"):
                        parts.append(f"\n*Screenshot: `{result['screenshot_path']}`*")

                    response = "\n".join(parts)

                    return {
                        "response": response,
                        "fast_path": False,
                        "mood": self._get_mood(),
                        "model_used": result.get("model_used", effective_model or "visual_feedback"),
                        "screenshot_base64": result.get("screenshot_base64"),
                        "screenshot_path": result.get("screenshot_path"),
                    }

                except Exception as e:
                    logger.error(f"[AgentService] Visual feedback error: {e}", exc_info=True)
                    _record_thought("observing", f"visual feedback failed, falling back: {e}", 0.5, "service")
                    # Fall through to normal chat if visual feedback fails

            # ===== SWARM MODE HANDLER (via MultiAgentOrchestrator) =====
            if detected_action == "swarm":
                logger.info(f"[AgentService] Multi-agent mode for: {message[:50]}...")
                _record_thought("connecting", "activating multi-agent orchestrator", 0.8, "service")

                try:
                    from aura.multi_agent.orchestrator import MultiAgentOrchestrator

                    # Get or create orchestrator for this session
                    if not hasattr(self, '_orchestrator') or self._orchestrator is None:
                        tool_registry = getattr(agent, 'tool_registry', {})
                        def llm_func(system_prompt, user_message):
                            # Read model dynamically from brain's current override
                            current_override = agent.brain._model_override if hasattr(agent.brain, '_model_override') else None
                            return agent.brain.think(user_message, system_prompt=system_prompt, use_history=False, model_override=current_override)
                        self._orchestrator = MultiAgentOrchestrator(
                            tool_registry=tool_registry,
                            llm_func=llm_func,
                        )

                    # Optionally gather search context for real-time queries
                    needs_search_keywords = [
                        "news", "latest", "current", "recent", "today", "now",
                        "update", "happening", "trending", "2024", "2025", "2026",
                        "research", "developments", "breakthroughs", "announced"
                    ]
                    msg_lower = message.lower()
                    query = message
                    if any(kw in msg_lower for kw in needs_search_keywords):
                        try:
                            from aura.tools.web_search import WebSearchTool
                            topic = msg_lower
                            for trigger in ["swarm", "multi-agent", "multiple agents", "team research", "collaborative", "all agents", "agent team"]:
                                topic = topic.replace(trigger, "").strip()
                            topic = topic.strip(" :,.-")
                            search_results = WebSearchTool().search(topic, num_results=8)
                            if search_results.get("success") and search_results.get("results"):
                                ctx = "\n".join([f"- {r.get('title','')}: {r.get('snippet','')}" for r in search_results["results"][:8]])
                                query = f"{message}\n\nSearch context:\n{ctx}"
                        except Exception as e:
                            logger.warning(f"[AgentService] Swarm search error: {e}")

                    # Route through the proper orchestrator
                    response = self._orchestrator.chat(query)

                    return {
                        "response": response,
                        "fast_path": False,
                        "mood": self._get_mood(),
                        "model_used": effective_model or "multi-agent"
                    }

                except Exception as e:
                    logger.error(f"[AgentService] Multi-agent error: {e}")
                    return {
                        "response": f"Multi-agent system error: {e}. Falling back to single agent.",
                        "fast_path": False,
                        "mood": self._get_mood(),
                        "model_used": effective_model
                    }

            # ===== DEEP RESEARCH HANDLER =====
            if detected_action == "deep_research":
                _record_thought("analyzing", "initiating deep research pipeline...", 0.8, "service")
                from aura.tools.deep_research import DeepResearchTool
                deep_tool = DeepResearchTool()

                topic = message.lower()
                for trigger in ["deep research", "thorough research", "extensive research"]:
                    topic = topic.replace(trigger, "").strip()
                topic = re.sub(r'\s+(?:on|about|for)\s*$', '', topic).strip()

                result = deep_tool.research(topic, depth="deep")

                if result.get("success"):
                    synthesis_prompt = f"""Summarize this research on '{topic}':
{result.get('content', '')[:8000]}

Provide key findings and cite sources."""
                    synthesized = agent.brain.think(synthesis_prompt, model_override=effective_model)
                    response = f"## Deep Research: {topic}\n\n{synthesized}\n\n---\n*{result.get('summary', '')}*"
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
                logger.debug("chat_screenpipe_context_failed", exc_info=True)

            enriched_msg = message + screen_hint if screen_hint else message

            # Use agent.chat() which has direct handlers for search/crypto
            response = agent.chat(enriched_msg, speak=speak)

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

            # Track messages in ConversationManager for cross-surface sync
            try:
                conv_manager = get_conversation_manager()
                conv_id = conv_manager.get_current_conversation_id()
                if conv_id:
                    conv_manager.on_message_added(conv_id, "user", message, surface="web", surface_user="web_default")
                    conv_manager.on_message_added(conv_id, "assistant", response, surface="web", surface_user="web_default")
            except Exception:
                logger.debug("chat_conv_manager_tracking_failed", exc_info=True)

            return {
                "response": response,
                "fast_path": self._was_fast_path(message),
                "mood": self._get_mood(),
                "model_used": agent.brain.get_last_model_used()
            }
        finally:
            # Clear model override and action mode after request
            if effective_model:
                agent.brain.set_model_override(None)
            agent.brain.set_action_mode(None)

    def chat_stream(self, message: str, model_override: Optional[str] = None, action_mode: Optional[str] = None,
                     routing_opts: Optional[Dict] = None):
        """Stream a chat response from the agent.

        Args:
            message: User message
            model_override: Optional model to use (explicit selection takes priority)
            action_mode: Optional action mode ('search', 'research', 'agent')
            routing_opts: Optional routing options from WebSocket (preference, feature, etc.)

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
            effective_model = model_override or (routing_opts.get("model") if routing_opts else None)
            detected_action = action_mode or detect_action_mode(message)

            # Set explicit model override only if user selected one (not auto-detected)
            if effective_model:
                self.agent.brain.set_model_override(effective_model)
                logger.info(f"[AgentService] Streaming with explicit model override: {effective_model}")
            else:
                # Clear any previous override so the neural router decides
                self.agent.brain.set_model_override(None)

            # Set routing context on brain (the neural router reads these)
            self.agent.brain._routing_preference = (routing_opts.get('preference', 'balanced') if routing_opts else 'balanced')
            self.agent.brain._conversation_id = (routing_opts.get('conversation_id') if routing_opts else None)
            self.agent.brain._has_attachment = bool(routing_opts.get('has_attachment')) if routing_opts else False

            # Set action mode for context-aware prompt injection (design system, etc.)
            self.agent.brain.set_action_mode(detected_action)

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
                logger.debug("stream_context_heatmap_track_failed", exc_info=True)

            # Record activity for idle panel
            try:
                from api.routes.idle_behaviors import record_user_activity
                record_user_activity()
            except Exception:
                logger.debug("stream_idle_activity_record_failed", exc_info=True)

            # ===== EXPLICIT SEARCH MODE (from UI button) =====
            # When user explicitly clicks "Search" button, ALWAYS do web search
            if detected_action == "search":
                logger.info(f"[AgentService] Explicit search mode for: {message[:50]}...")
                try:
                    from aura.tools.search_fallback import web_search_with_fallback
                    search_result = web_search_with_fallback(query=message, max_results=8)

                    if search_result.get("results"):
                        raw_results = ""
                        _search_citations = []
                        for i, r in enumerate(search_result["results"][:8], 1):
                            title = r.get("title", "No title")
                            snippet = r.get("snippet", r.get("content", ""))[:200]
                            url = r.get("url", "")
                            raw_results += f"{i}. {title}\n   {snippet}\n   URL: {url}\n\n"
                            if url:
                                _search_citations.append({"id": i, "title": title, "url": url, "snippet": snippet})

                        # Synthesize with LLM
                        synthesis_prompt = f"""You are summarizing REAL web search results for: '{message}'

SEARCH RESULTS:
{raw_results}

STRICT RULES:
- ONLY use information from the search results above
- Do NOT add information from your own knowledge
- Include actual URLs as references
- If results don't fully answer the query, say so
- Be concise but accurate, use markdown"""

                        if hasattr(brain, 'think_stream'):
                            for chunk in brain.think_stream(synthesis_prompt, model_override=effective_model):
                                yield {"type": "chunk", "content": chunk}
                        else:
                            yield {"type": "chunk", "content": brain.think(synthesis_prompt, model_override=effective_model)}

                        if _search_citations:
                            yield {"type": "citations", "citations": _search_citations}
                    else:
                        yield {"type": "chunk", "content": f"No search results found for: {message}"}

                    yield {"type": "done", "mood": self._get_mood(), "model_used": effective_model or "search"}
                    return
                except Exception as e:
                    logger.error(f"[AgentService] Search mode error: {e}")
                    yield {"type": "chunk", "content": f"Search failed: {e}. Falling back to standard response.\n\n"}

            # ===== EXPLICIT RESEARCH MODE (from UI button) =====
            if detected_action == "research":
                logger.info(f"[AgentService] Explicit research mode for: {message[:50]}...")
                try:
                    from aura.tools.deep_research import DeepResearchTool
                    research_tool = DeepResearchTool()

                    yield {"type": "chunk", "content": f"## Researching: {message}\n\n"}
                    yield {"type": "tool_trace", "event": "start", "tool": "research", "detail": f'Researching "{message[:50]}"', "timestamp": time.time()}

                    import queue as _rq
                    _rprogress: _rq.Queue = _rq.Queue()
                    research_tool.set_ws_callback(lambda evt: _rprogress.put(evt))

                    _rstart = time.time()
                    _rfuture = research_tool._executor.submit(research_tool.research, message, "standard")

                    while not _rfuture.done():
                        try:
                            evt = _rprogress.get(timeout=0.25)
                            yield evt
                        except _rq.Empty:
                            pass
                    while not _rprogress.empty():
                        try:
                            yield _rprogress.get_nowait()
                        except _rq.Empty:
                            break

                    result = _rfuture.result(timeout=5)
                    _relapsed = int((time.time() - _rstart) * 1000)
                    yield {"type": "tool_trace", "event": "done", "tool": "research", "detail": f'{result.get("urls_found", 0)} sources', "elapsed_ms": _relapsed, "timestamp": time.time()}

                    if result.get("success"):
                        synthesis_prompt = f"""Based on this research, provide a clear summary:

Topic: {message}
Sources: {result.get('urls_found', 0)} found, {result.get('pages_read', 0)} read

Content:
{result.get('content', '')[:6000]}

Summarize key findings with [1], [2] citations. Be factual and concise."""

                        if hasattr(brain, 'think_stream'):
                            for chunk in brain.think_stream(synthesis_prompt, model_override=effective_model):
                                yield {"type": "chunk", "content": chunk}
                        else:
                            yield {"type": "chunk", "content": brain.think(synthesis_prompt, model_override=effective_model)}

                        citations = result.get("citations", [])
                        if not citations:
                            for i, s in enumerate(result.get("sources", [])[:10], 1):
                                if isinstance(s, dict) and s.get("url"):
                                    citations.append({"id": i, "title": s.get("title", ""), "url": s["url"], "snippet": s.get("snippet", "")})
                        if citations:
                            yield {"type": "citations", "citations": citations}
                    else:
                        yield {"type": "chunk", "content": f"Research failed: {result.get('error', 'Unknown error')}"}

                    yield {"type": "done", "mood": self._get_mood(), "model_used": effective_model or "research"}
                    return
                except Exception as e:
                    logger.error(f"[AgentService] Research mode error: {e}")
                    yield {"type": "chunk", "content": f"Research error: {e}. Falling back.\n\n"}

            # ===== DIRECT SEARCH HANDLER (auto-detect from message text) =====
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

            # ===== FRONTEND VISUAL FEEDBACK HANDLER (streaming) =====
            if detected_action == "frontend":
                logger.info(f"[AgentService] Frontend visual feedback (stream) for: {message[:50]}...")
                try:
                    from aura.tools.visual_feedback import VisualFeedbackLoop
                    vfl = VisualFeedbackLoop(brain=brain)

                    for evt in vfl.generate_stream(message, max_iterations=2):
                        yield evt

                    yield {"type": "done", "mood": self._get_mood(), "model_used": effective_model or "visual_feedback"}
                    return
                except Exception as e:
                    logger.error(f"[AgentService] Visual feedback stream error: {e}", exc_info=True)
                    yield {"type": "chunk", "content": f"*Visual feedback loop failed ({e}), falling back to standard generation...*\n\n"}
                    # Fall through to standard streaming

            # ===== DEEP RESEARCH HANDLER =====
            if detected_action == "deep_research":
                try:
                    import queue as _queue

                    from aura.tools.deep_research import DeepResearchTool
                    deep_tool = DeepResearchTool()

                    topic = message.lower()
                    for trigger in ["deep research", "thorough research", "extensive research", "full research", "research everything", "research in depth"]:
                        topic = topic.replace(trigger, "").strip()
                    topic = re.sub(r'\s+(?:on|about|for)\s*$', '', topic).strip()

                    logger.info(f"[AgentService] Deep research on: {topic}")
                    yield {"type": "chunk", "content": f"## Deep Research: {topic}\n\n"}
                    yield {"type": "tool_trace", "event": "start", "tool": "deep_research", "detail": f'Researching "{topic[:50]}"', "timestamp": time.time()}

                    # Wire WebSocket progress emitter — research() runs in
                    # a thread pool, but chat_stream is consumed from another
                    # thread that feeds asyncio.Queue → WebSocket.  We use a
                    # plain queue.Queue so the emitter callback (called inside
                    # research()) can safely push events that we drain below.
                    _progress_q: _queue.Queue = _queue.Queue()
                    deep_tool.set_ws_callback(lambda evt: _progress_q.put(evt))

                    # Run research in a background thread so we can drain
                    # progress events while it's running.
                    _dr_start = time.time()
                    _research_future = deep_tool._executor.submit(
                        deep_tool.research, topic, "deep"
                    )

                    # Drain progress events until research finishes
                    while not _research_future.done():
                        try:
                            evt = _progress_q.get(timeout=0.25)
                            yield evt
                        except _queue.Empty:
                            pass
                    # Drain any remaining events
                    while not _progress_q.empty():
                        try:
                            yield _progress_q.get_nowait()
                        except _queue.Empty:
                            break

                    result = _research_future.result(timeout=5)
                    _dr_elapsed = int((time.time() - _dr_start) * 1000)
                    yield {"type": "tool_trace", "event": "done", "tool": "deep_research", "detail": f'{result.get("urls_found", 0)} sources, {result.get("pages_read", 0)} pages', "elapsed_ms": _dr_elapsed, "timestamp": time.time()}

                    if result.get("success"):
                        synthesis_prompt = f"""Based on this deep research, provide a comprehensive summary:

Topic: {topic}
Sources Found: {result.get('urls_found', 0)}
Pages Read: {result.get('pages_read', 0)}

Content:
{result.get('content', '')[:8000]}

Provide a well-structured, informative summary with key findings and cite sources using [1], [2] etc."""

                        if hasattr(brain, 'think_stream'):
                            for chunk in brain.think_stream(synthesis_prompt, model_override=effective_model):
                                yield {"type": "chunk", "content": chunk}
                        else:
                            yield {"type": "chunk", "content": brain.think(synthesis_prompt, model_override=effective_model)}
                        yield {"type": "chunk", "content": f"\n\n---\n*{result.get('summary', '')}*"}

                        # Yield structured citations from deep research
                        # v3: use pre-built citations with claim-source mapping
                        citations = result.get("citations", [])
                        if not citations:
                            # Fallback: build from raw sources (backward compat)
                            raw_sources = result.get("sources", [])
                            for i, s in enumerate(raw_sources[:15], 1):
                                if isinstance(s, dict):
                                    citations.append({"id": i, "title": s.get("title", s.get("url", "")), "url": s.get("url", ""), "snippet": s.get("snippet", "")})
                                elif isinstance(s, str):
                                    citations.append({"id": i, "title": s, "url": s, "snippet": ""})
                        if citations:
                            yield {"type": "citations", "citations": citations}
                    else:
                        yield {"type": "chunk", "content": f"Research failed: {result.get('error', 'Unknown error')}"}

                    yield {"type": "done", "mood": self._get_mood(), "model_used": effective_model or "deep_research"}
                    return
                except Exception as e:
                    logger.error(f"[AgentService] Deep research error: {e}")
                    yield {"type": "chunk", "content": f"Deep research error: {e}"}
                    yield {"type": "done", "mood": self._get_mood(), "model_used": "error"}
                    return

            # ===== SWARM/MULTI-AGENT HANDLER (via MultiAgentOrchestrator) =====
            if detected_action == "swarm":
                try:
                    from aura.multi_agent.orchestrator import MultiAgentOrchestrator

                    logger.info(f"[AgentService] Multi-agent mode (WS) for: {message[:50]}...")
                    yield {"type": "chunk", "content": "## Multi-Agent Orchestrator\n\n"}

                    # Get or create orchestrator
                    if not hasattr(self, '_orchestrator') or self._orchestrator is None:
                        tool_registry = getattr(self.agent, 'tool_registry', {})
                        def llm_func(system_prompt, user_message):
                            # Read model dynamically from brain's current override
                            current_override = brain._model_override if hasattr(brain, '_model_override') else None
                            return brain.think(user_message, system_prompt=system_prompt, use_history=False, model_override=current_override)
                        self._orchestrator = MultiAgentOrchestrator(
                            tool_registry=tool_registry,
                            llm_func=llm_func,
                        )

                    # Optionally gather search context
                    needs_search_keywords = [
                        "news", "latest", "current", "recent", "today", "now",
                        "update", "happening", "trending", "2024", "2025", "2026",
                        "research", "developments", "breakthroughs", "announced"
                    ]
                    msg_lower = message.lower()
                    query = message
                    _swarm_citations = []

                    if any(kw in msg_lower for kw in needs_search_keywords):
                        yield {"type": "chunk", "content": "Gathering real-time data...\n\n"}
                        try:
                            topic = msg_lower
                            for trigger in ["swarm", "multi-agent", "multiple agents", "team research", "collaborative", "all agents", "agent team"]:
                                topic = topic.replace(trigger, "").strip()
                            topic = topic.strip(" :,.-")

                            from aura.tools.search_fallback import web_search_with_fallback
                            search_results = web_search_with_fallback(query=topic, max_results=8)

                            if search_results.get("success") and search_results.get("results"):
                                ctx = "\n".join([f"- {r.get('title','')}: {r.get('snippet','')}" for r in search_results["results"][:8]])
                                query = f"{message}\n\nSearch context:\n{ctx}"
                                yield {"type": "chunk", "content": f"Found {len(search_results['results'])} sources.\n\n"}

                                for i, r in enumerate(search_results["results"][:15], 1):
                                    if r.get("url"):
                                        _swarm_citations.append({"id": i, "title": r.get("title", r.get("url", "")), "url": r["url"], "snippet": r.get("snippet", "")[:200]})
                        except Exception as e:
                            logger.warning(f"[AgentService] Swarm search error: {e}")

                    # Route through orchestrator (handles routing, collaboration mode, synthesis)
                    response = self._orchestrator.chat(query)
                    yield {"type": "chunk", "content": response}

                    if _swarm_citations:
                        yield {"type": "citations", "citations": _swarm_citations}
                    yield {"type": "done", "mood": self._get_mood(), "model_used": effective_model or "multi-agent"}
                    return
                except Exception as e:
                    logger.error(f"[AgentService] Multi-agent error: {e}")
                    yield {"type": "chunk", "content": f"Multi-agent error: {e}"}
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
                logger.debug("stream_screenpipe_context_failed", exc_info=True)

            enriched_message = message + screen_hint if screen_hint else message

            # ===== AGENT MODE (agentic loop with tools) =====
            if detected_action == "agent" and hasattr(agent, 'run'):
                logger.info("[AgentService] Using agent.run() for agent mode")
                try:
                    import queue as _queue
                    result_q = _queue.Queue()

                    def _run_agent():
                        try:
                            result = agent.run(enriched_message, timeout_seconds=120)
                            result_q.put(result)
                        except Exception as e:
                            result_q.put({"response": f"Agent failed: {e}", "timeout": True})

                    t = threading.Thread(target=_run_agent, daemon=True)
                    t.start()

                    yield {"type": "tool_status", "tool_name": "agent", "tool_action": "running agent"}

                    t.join(timeout=125)
                    if not result_q.empty():
                        result = result_q.get_nowait()
                        response_text = result.get("response", "") if isinstance(result, dict) else str(result)
                        if response_text:
                            response_text = _filter_skill_json(response_text)
                            yield {"type": "chunk", "content": response_text}
                    else:
                        yield {"type": "chunk", "content": "The agent task is taking longer than expected. Please try again with a more specific query."}

                    yield {"type": "tool_status", "tool_name": "", "tool_action": ""}
                    yield {"type": "done", "mood": self._get_mood(), "model_used": effective_model or "agent_run"}
                    return
                except Exception as e:
                    logger.error(f"[AgentService] Agent run failed: {e}", exc_info=True)
                    yield {"type": "chunk", "content": "Agent encountered an error. Falling back to standard response.\n\n"}

            # ===== STANDARD STREAMING =====
            if hasattr(brain, 'think_stream'):
                full_response = ""
                yield {"type": "tool_status", "tool_name": "brain", "tool_action": "thinking"}
                chunk_buffer = ""
                for chunk in brain.think_stream(enriched_message, model_override=effective_model):
                    full_response += chunk
                    chunk_buffer += chunk
                    # Check if buffer contains skill learning artifacts
                    if '"trigger_patterns"' in chunk_buffer or 'Analyze these successful interactions' in chunk_buffer:
                        # Accumulate and filter at the end instead of streaming garbage
                        continue
                    yield {"type": "chunk", "content": chunk}
                    chunk_buffer = ""
                # Flush any held-back content (filtered)
                if chunk_buffer:
                    cleaned = _filter_skill_json(chunk_buffer)
                    if cleaned.strip():
                        yield {"type": "chunk", "content": cleaned}
                yield {"type": "tool_status", "tool_name": "", "tool_action": ""}  # clear status

                # Single memory write via UnifiedMemory (consolidated from 3 systems, 2026-03-22)
                _clean_msg = message.split("\n[Screen context:")[0].strip()
                if len(full_response) > 30:
                    try:
                        def _store_to_unified():
                            try:
                                from aura.memory.unified_memory import get_unified_memory
                                get_unified_memory().store_gated(
                                    content=f"Q: {_clean_msg[:300]}\nA: {full_response[:500]}",
                                    source="conversation",
                                    importance=0.6,
                                )
                            except Exception:
                                logger.debug("unified_memory_store_failed", exc_info=True)
                        threading.Thread(target=_store_to_unified, daemon=True).start()
                    except Exception:
                        logger.debug("unified_memory_thread_start_failed", exc_info=True)

                # Track messages in ConversationManager for cross-surface sync
                try:
                    conv_manager = get_conversation_manager()
                    conv_id = conv_manager.get_current_conversation_id()
                    if conv_id:
                        _clean_user_msg = message.split("\n[Screen context:")[0].strip()
                        conv_manager.on_message_added(conv_id, "user", _clean_user_msg, surface="web", surface_user="web_default")
                        conv_manager.on_message_added(conv_id, "assistant", full_response, surface="web", surface_user="web_default")
                except Exception:
                    logger.debug("stream_conv_manager_tracking_failed", exc_info=True)

                yield {"type": "done", "mood": self._get_mood(), "model_used": brain.get_last_model_used()}
            else:
                # Fallback to non-streaming
                response = agent.chat(message, speak=False)

                # Track messages in ConversationManager for cross-surface sync
                try:
                    conv_manager = get_conversation_manager()
                    conv_id = conv_manager.get_current_conversation_id()
                    if conv_id:
                        conv_manager.on_message_added(conv_id, "user", message, surface="web", surface_user="web_default")
                        conv_manager.on_message_added(conv_id, "assistant", response, surface="web", surface_user="web_default")
                except Exception:
                    logger.debug("stream_fallback_conv_manager_tracking_failed", exc_info=True)

                yield {"type": "chunk", "content": response}
                yield {"type": "done", "mood": self._get_mood(), "model_used": brain.get_last_model_used()}

        finally:
            # ===== TEARDOWN =====
            # Clear model override and action mode after request (matches chat() behavior)
            if effective_model and brain:
                brain.set_model_override(None)
            if brain:
                brain.set_action_mode(None)

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

        Note:
            CRITICAL FIX: Lock is only held briefly for setup/teardown, NOT during
            agent.run(). Mirrors chat() pattern to prevent long lock contention.
        """
        # ===== SETUP PHASE — Brief lock =====
        with self._agent_lock:
            agent = self.agent
            original_max = getattr(agent, 'max_iterations', 10)
            agent.max_iterations = max_iterations
        # ===== END SETUP — Lock released =====

        try:
            result = agent.run(goal, context=context, use_fastpath=use_fastpath)
            result["mood"] = self._get_mood()
            return result
        finally:
            # ===== TEARDOWN — Brief lock =====
            with self._agent_lock:
                agent.max_iterations = original_max

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
        """Create a new conversation via ConversationManager."""
        if self._agent is None:
            return {"error": "Agent not initialized"}
        try:
            manager = get_conversation_manager()
            conv_id = manager.create_conversation(title, surface="web")
        except Exception:
            # Fallback to direct brain call if ConversationManager not ready
            conv_id = self._agent.brain.create_conversation(title)
        return {"id": conv_id, "title": title or "New Chat", "messages": []}

    def list_conversations(self) -> list:
        """List all conversations via ConversationManager (includes surface activity)."""
        if self._agent is None:
            return []
        try:
            manager = get_conversation_manager()
            return manager.list_conversations()
        except Exception:
            # Fallback to direct brain call if ConversationManager not ready
            return self._agent.brain.list_conversations()

    def switch_conversation(self, conversation_id: str) -> Dict[str, Any]:
        """Switch to a different conversation via ConversationManager."""
        if self._agent is None:
            return {"error": "Agent not initialized"}

        try:
            manager = get_conversation_manager()
            success = manager.switch_conversation(conversation_id, surface="web")
        except Exception:
            # Fallback to direct brain call if ConversationManager not ready
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
        """Delete a conversation via ConversationManager."""
        if self._agent is None:
            return {"success": False, "error": "Agent not initialized"}
        try:
            manager = get_conversation_manager()
            success = manager.delete_conversation(conversation_id)
        except Exception:
            # Fallback to direct brain call if ConversationManager not ready
            success = self._agent.brain.delete_conversation(conversation_id)
        return {
            "success": success,
            "new_active_id": self._agent.brain.get_current_conversation_id(),
        }

    def rename_conversation(self, conversation_id: str, title: str) -> bool:
        """Rename a conversation via ConversationManager."""
        if self._agent is None:
            return False
        try:
            manager = get_conversation_manager()
            return manager.rename_conversation(conversation_id, title)
        except Exception:
            # Fallback to direct brain call if ConversationManager not ready
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
            logger.debug("context_tracker_clear_failed", exc_info=True)

        # Clear active thinking state
        try:
            from api.routes.thinking import get_manager
            get_manager().clear_active()
        except Exception:
            logger.debug("thinking_manager_clear_failed", exc_info=True)

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
            from aura.thinking_mode import ThinkingMode, get_thinking_mode_manager
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
                "mood": self._get_mood(),
                "model_used": "command",
            }

    def _was_fast_path(self, message: str) -> bool:
        """Check if message was handled via fast path."""
        try:
            return self.agent._is_simple_query(message)
        except Exception:
            logger.debug("fast_path_check_failed", exc_info=True)
            return False

    def get_available_models(self) -> Dict[str, Any]:
        """Get list of available models from verified config — no Ollama API query.

        Cloud models come from VERIFIED_CLOUD_MODELS (hardcoded, trusted).
        Local models come from VERIFIED_LOCAL_MODELS (utility only).
        ChatGPT models come from chatgpt_client or hardcoded fallback.
        """
        try:
            from aura.config import VERIFIED_CLOUD_MODELS, VERIFIED_LOCAL_MODELS

            cloud_models = sorted(VERIFIED_CLOUD_MODELS)
            local_models = sorted(VERIFIED_LOCAL_MODELS)

            # Always include ChatGPT models — auth checked at request time
            try:
                from aura.auth.chatgpt_client import ALL_CHATGPT_MODELS
                chatgpt_models = sorted(ALL_CHATGPT_MODELS)
            except ImportError:
                chatgpt_models = [
                    "chatgpt:gpt-5.1", "chatgpt:gpt-5.1-codex", "chatgpt:gpt-5.1-codex-max", "chatgpt:gpt-5.1-codex-mini",
                    "chatgpt:gpt-5.2", "chatgpt:gpt-5.2-codex",
                    "chatgpt:gpt-5.3", "chatgpt:gpt-5.3-codex", "chatgpt:gpt-5.3-codex-spark",
                    "chatgpt:gpt-5.4", "chatgpt:gpt-5.4-pro", "chatgpt:gpt-5.4-thinking",
                ]

            current_model = "auto"
            if self._agent is not None:
                current_model = getattr(self._agent.brain, 'model', 'auto')

            return {
                "local": local_models,
                "cloud": cloud_models,
                "chatgpt": chatgpt_models,
                "current": current_model
            }
        except Exception as e:
            logger.error(f"[AgentService] Failed to get models: {e}")
            return {
                "local": [],
                "cloud": [],
                "chatgpt": [],
                "current": "unknown"
            }


# Global instance
agent_service = AgentService()
