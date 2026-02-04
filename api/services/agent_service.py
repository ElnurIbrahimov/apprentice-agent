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
        "preferred": "devstral-small-2:24b-cloud",
        "fallbacks": ["qwen2.5:7b", "llama3.2:3b"],
        "description": "Quick web search"
    },
    "research": {
        "preferred": "deepseek-v3.1:671b-cloud",
        "fallbacks": ["cogito-2.1:671b-cloud", "qwen3-next:80b-cloud"],
        "description": "Comprehensive research"
    },
    "agent": {
        "preferred": "kimi-k2.5-cloud",
        "fallbacks": ["devstral-2:123b-cloud", "deepseek-v3.1:671b-cloud"],
        "description": "Autonomous task execution"
    },
    "code": {
        "preferred": "glm-4.7-cloud",
        "fallbacks": ["devstral-2:123b-cloud", "qwen3-coder:480b-cloud"],
        "description": "Code generation and analysis"
    },
    "vision": {
        "preferred": "qwen3-vl:235b-cloud",
        "fallbacks": ["kimi-k2.5-cloud", "llava:13b"],
        "description": "Image analysis"
    },
    "deep_research": {
        "preferred": "deepseek-v3.1:671b-cloud",
        "fallbacks": ["cogito-2.1:671b-cloud", "kimi-k2.5-cloud"],
        "description": "Multi-source deep research with page reading"
    },
    "swarm": {
        "preferred": "deepseek-v3.1:671b-cloud",
        "fallbacks": ["cogito-2.1:671b-cloud", "kimi-k2.5-cloud"],
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


def get_model_for_action(action_mode: str) -> Optional[str]:
    """Get the best model for an action mode.

    Returns:
        Model name to use, or None to use default
    """
    if action_mode not in ACTION_MODE_MODELS:
        return None

    config = ACTION_MODE_MODELS[action_mode]
    preferred = config["preferred"]

    # For cloud models, just return them (Ollama.com handles availability)
    if preferred.endswith("-cloud"):
        logger.info(f"[AutoModel] Action '{action_mode}' -> using cloud model: {preferred}")
        return preferred

    # For local models, validate availability
    from apprentice_agent.config import validate_model
    if validate_model(preferred):
        logger.info(f"[AutoModel] Action '{action_mode}' -> using local model: {preferred}")
        return preferred

    # Try fallbacks
    for fallback in config["fallbacks"]:
        if fallback.endswith("-cloud") or validate_model(fallback):
            logger.info(f"[AutoModel] Action '{action_mode}' -> using fallback: {fallback}")
            return fallback

    logger.warning(f"[AutoModel] No model available for action '{action_mode}'")
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
            # Detect action mode and auto-select model
            detected_action = detect_action_mode(message)
            effective_model = model_override

            if not effective_model and detected_action:
                effective_model = get_model_for_action(detected_action)
                if effective_model:
                    logger.info(f"[AgentService] Chat auto-selected model for {detected_action}: {effective_model}")

            # Set model override if we have one
            if effective_model:
                self.agent.brain.set_model_override(effective_model)
                logger.info(f"[AgentService] Using model: {effective_model}")

            try:
                # ===== SWARM MODE HANDLER =====
                if detected_action == "swarm":
                    import concurrent.futures

                    logger.info(f"[AgentService] Swarm mode (REST) for: {message[:50]}...")

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
                            from apprentice_agent.tools.search import SearchTool
                            search_tool = SearchTool()
                            search_results = search_tool.search(topic, num_results=8)

                            if search_results.get("success") and search_results.get("results"):
                                results_text = []
                                for i, r in enumerate(search_results["results"][:8], 1):
                                    results_text.append(f"{i}. **{r.get('title', 'No title')}**\n   {r.get('snippet', '')}\n   Source: {r.get('link', '')}")
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
                        synthesis_prompt = f"""Synthesize these agent perspectives on: "{message}"

{chr(10).join([f"**{name}:** {resp[:1500]}" for name, resp in results.items()])}

Provide a unified synthesis with key consensus points and a final conclusion."""
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
                    from apprentice_agent.tools.deep_research import DeepResearchTool
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

                # Use agent.chat() which has direct handlers for search/crypto
                response = self.agent.chat(message, speak=speak)

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
        """
        with self._agent_lock:
            # Determine model to use:
            # 1. Explicit model_override takes priority
            # 2. Auto-detect action mode from message prefix
            # 3. Use action_mode parameter if provided
            # 4. Fall back to default model

            effective_model = model_override
            detected_action = action_mode or detect_action_mode(message)

            if not effective_model and detected_action:
                # Auto-select model based on action mode
                effective_model = get_model_for_action(detected_action)
                if effective_model:
                    logger.info(f"[AgentService] Auto-selected model for {detected_action}: {effective_model}")

            # Set model override if we have one
            if effective_model:
                self.agent.brain.set_model_override(effective_model)
                logger.info(f"[AgentService] Streaming with model: {effective_model}")

            try:
                # ===== DIRECT SEARCH HANDLER =====
                # Check for direct search before streaming to prevent query hallucination
                # This matches the logic in agent.chat() which was being bypassed
                if hasattr(self.agent, '_handle_direct_search'):
                    search_response = self.agent._handle_direct_search(message)
                    if search_response:
                        logger.info("[AgentService] Direct search handled, returning result")
                        yield {"type": "chunk", "content": search_response}
                        yield {
                            "type": "done",
                            "mood": self._get_mood(),
                            "model_used": "direct_search"
                        }
                        return

                # ===== DIRECT CRYPTO HANDLER =====
                # Check for crypto price requests
                if hasattr(self.agent, '_handle_direct_crypto'):
                    crypto_response = self.agent._handle_direct_crypto(message)
                    if crypto_response:
                        logger.info("[AgentService] Direct crypto handled, returning result")
                        yield {"type": "chunk", "content": crypto_response}
                        yield {
                            "type": "done",
                            "mood": self._get_mood(),
                            "model_used": "direct_crypto"
                        }
                        return

                # ===== DEEP RESEARCH HANDLER =====
                # Use DeepResearchTool for thorough multi-source research
                if detected_action == "deep_research":
                    try:
                        from apprentice_agent.tools.deep_research import DeepResearchTool
                        deep_tool = DeepResearchTool()

                        # Extract topic from message
                        topic = message.lower()
                        for trigger in ["deep research", "thorough research", "extensive research", "full research", "research everything", "research in depth"]:
                            topic = topic.replace(trigger, "").strip()
                        topic = topic.strip(" on about for")

                        logger.info(f"[AgentService] Deep research on: {topic}")
                        yield {"type": "chunk", "content": f"## Deep Research: {topic}\n\n"}

                        result = deep_tool.research(topic, depth="deep")

                        if result.get("success"):
                            # Synthesize with LLM
                            synthesis_prompt = f"""Based on this deep research, provide a comprehensive summary:

Topic: {topic}
Sources Found: {result.get('urls_found', 0)}
Pages Read: {result.get('pages_read', 0)}

Content:
{result.get('content', '')[:8000]}

Provide a well-structured, informative summary with key findings and cite sources."""

                            synthesized = self.agent.brain.think(synthesis_prompt)
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
                # Force parallel execution of ALL agents for true swarm behavior
                if detected_action == "swarm":
                    try:
                        import concurrent.futures

                        logger.info(f"[AgentService] Swarm mode activated for: {message[:50]}...")
                        yield {"type": "chunk", "content": "## Agent Swarm\n\n"}

                        # Check if query needs real-time data
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
                            yield {"type": "chunk", "content": "Gathering real-time data...\n\n"}
                            try:
                                topic = msg_lower
                                for trigger in ["swarm", "multi-agent", "multiple agents", "team research", "collaborative", "all agents", "agent team"]:
                                    topic = topic.replace(trigger, "").strip()
                                topic = topic.strip(" :,.-")

                                from apprentice_agent.tools.search import SearchTool
                                search_tool = SearchTool()
                                search_results = search_tool.search(topic, num_results=8)

                                if search_results.get("success") and search_results.get("results"):
                                    results_text = []
                                    for i, r in enumerate(search_results["results"][:8], 1):
                                        results_text.append(f"{i}. **{r.get('title', 'No title')}**\n   {r.get('snippet', '')}\n   Source: {r.get('link', '')}")
                                    search_context = f"\n\n**Search Results for '{topic}':**\n\n" + "\n\n".join(results_text) + "\n\n---\n\n"
                                    yield {"type": "chunk", "content": f"Found {len(search_results['results'])} sources.\n\n"}
                            except Exception as e:
                                logger.error(f"[AgentService] Swarm search error: {e}")

                        # Build prompt with search context
                        agent_prompt = message
                        if search_context:
                            agent_prompt = f"Based on the following real-time search results, analyze: {message}\n{search_context}\nUse the search results to provide informed analysis."

                        # Define agent perspectives
                        agents = {
                            "Research": "You are a Research Agent. Analyze the provided data and extract key facts, findings, and evidence. Cite sources when available.",
                            "Analyst": "You are an Analyst Agent. Provide critical analysis of the data, identify patterns, trends, and assess implications.",
                            "Creative": "You are a Creative Agent. Think outside the box, find unexpected connections, and propose innovative perspectives.",
                            "Strategist": "You are a Strategy Agent. Consider long-term implications, identify opportunities and risks, and provide actionable recommendations."
                        }

                        mode_text = "parallel + search" if search_context else "parallel"
                        yield {"type": "chunk", "content": f"**Agents:** {', '.join(agents.keys())}\n**Mode:** {mode_text}\n\n---\n\n"}

                        # Execute all agents in parallel
                        results = {}
                        def run_agent(name, system_prompt):
                            try:
                                response = self.agent.brain.think(
                                    agent_prompt,
                                    system_prompt=system_prompt,
                                    use_history=False
                                )
                                return name, response
                            except Exception as e:
                                return name, f"Error: {e}"

                        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                            futures = {executor.submit(run_agent, name, prompt): name for name, prompt in agents.items()}

                            for future in concurrent.futures.as_completed(futures, timeout=120):
                                try:
                                    name, response = future.result()
                                    results[name] = response
                                    yield {"type": "chunk", "content": f"### {name} Agent\n\n{response}\n\n---\n\n"}
                                except Exception as e:
                                    name = futures[future]
                                    yield {"type": "chunk", "content": f"### {name} Agent\n\nError: {e}\n\n---\n\n"}

                        # Synthesize final summary
                        if len(results) >= 2:
                            yield {"type": "chunk", "content": "### Synthesis\n\n"}

                            synthesis_prompt = f"""You are synthesizing insights from multiple AI agents who analyzed this query: "{message}"

Here are their perspectives:

{chr(10).join([f"**{name}:** {resp[:1500]}" for name, resp in results.items()])}

Provide a unified synthesis that:
1. Identifies key consensus points
2. Notes important disagreements or different angles
3. Gives a final integrated conclusion

Be concise but comprehensive."""

                            # Stream the synthesis
                            if hasattr(self.agent.brain, 'think_stream'):
                                for chunk in self.agent.brain.think_stream(synthesis_prompt):
                                    yield {"type": "chunk", "content": chunk}
                            else:
                                synthesis = self.agent.brain.think(synthesis_prompt)
                                yield {"type": "chunk", "content": synthesis}

                        yield {"type": "done", "mood": self._get_mood(), "model_used": effective_model or "swarm"}
                        return
                    except Exception as e:
                        logger.error(f"[AgentService] Swarm mode error: {e}")
                        yield {"type": "chunk", "content": f"Swarm mode error: {e}"}
                        yield {"type": "done", "mood": self._get_mood(), "model_used": "error"}
                        return

                # ===== DIRECT CODE HANDLER =====
                # Check for explicit code execution requests
                if hasattr(self.agent, '_handle_direct_code'):
                    code_response = self.agent._handle_direct_code(message)
                    if code_response:
                        logger.info("[AgentService] Direct code handled, returning result")
                        yield {"type": "chunk", "content": code_response}
                        yield {
                            "type": "done",
                            "mood": self._get_mood(),
                            "model_used": "direct_code"
                        }
                        return

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
                if effective_model:
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
