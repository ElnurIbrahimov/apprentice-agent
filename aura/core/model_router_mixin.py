"""Model routing/selection mixin extracted from brain.py.

Provides model selection (System 1/System 2 dual-process routing),
client resolution (local/cloud/ChatGPT/direct API), fallback chains,
outcome-aware routing stats, and model lifecycle (unload).

All methods reference ``self.*`` attributes initialised by
``OllamaBrain.__init__`` — this module is only useful as a mixin base
for that class.
"""

import logging
import time
from typing import Optional

from aura.config import Config

logger = logging.getLogger(__name__)


class ModelRouterMixin:
    """Model routing and selection methods for OllamaBrain."""

    def _get_client_for_model(self, model: str) -> tuple:
        """Get the appropriate client (local, cloud, ChatGPT, or direct API) based on model name.

        Routing:
        - chatgpt:* models -> ChatGPT OAuth client (Codex Responses API)
        - anthropic:*/openai:*/gemini:*/grok:*/perplexity:*/deepseek:*/minimax:*/qwen:*/kimi:*/glm:*
          -> Direct API provider client
        - *-cloud / *:cloud models -> Ollama cloud client (or local bridge)
        - everything else -> local Ollama client

        Returns:
            Tuple of (client, actual_model_name) - model name may be modified for fallback
        """
        # ChatGPT OAuth models (e.g., chatgpt:gpt-5.1-codex)
        if model.startswith("chatgpt:"):
            if self._chatgpt_client:
                logger.debug(f"[BRAIN] Using ChatGPT OAuth client for model: {model}")
                return self._chatgpt_client, model
            else:
                logger.warning(f"[BRAIN] ChatGPT not authenticated, cannot use {model}")
                # Fall back to cloud client if available, not local Ollama
                if self._cloud_client:
                    return self._cloud_client, Config.MODEL_FAST
                return self.client, Config.MODEL_FAST

        # Direct API providers (anthropic:, openai:, gemini:, grok:, etc.)
        if ":" in model and not model.endswith(("-cloud", ":cloud", ":latest")):
            prefix = model.split(":")[0]
            try:
                from aura.providers import get_provider
                provider = get_provider(prefix)
                if provider and provider.is_configured():
                    logger.debug(f"[BRAIN] Using {provider.display_name} API for model: {model}")
                    return provider, model
                elif provider:
                    logger.warning(f"[BRAIN] {provider.display_name} API key not set, cannot use {model}")
                    # Fall back to fast model instead of silently routing
                    # the provider-prefixed name to local Ollama (which would fail).
                    if self._cloud_client:
                        return self._cloud_client, Config.MODEL_FAST
                    return self.client, Config.MODEL_FAST
            except Exception as e:
                logger.debug(f"[BRAIN] Provider lookup failed for {prefix}: {e}")
                # Same fallback on provider error
                if self._cloud_client:
                    return self._cloud_client, Config.MODEL_FAST
                return self.client, Config.MODEL_FAST

        if model.endswith(("-cloud", ":cloud")):
            # Cloud models: prefer the dedicated cloud client (api.ollama.com) if available,
            # fall back to local Ollama bridge (for setups where Ollama Pro runs locally).
            if self._cloud_client:
                logger.debug(f"[BRAIN] Using Ollama cloud API for model: {model}")
                return self._cloud_client, model
            logger.debug(f"[BRAIN] Using local Ollama bridge for cloud model: {model}")
            return self.client, model

        # For local models / "auto": check if local Ollama is actually reachable.
        # If not, route to cloud with a cloud-capable model instead of failing.
        if self._cloud_client and not self._local_ollama_ok:
            logger.info(f"[BRAIN] Local Ollama not available, routing '{model}' to cloud")
            cloud_model = model if model.endswith(":cloud") else Config.MODEL_FAST
            return self._cloud_client, cloud_model

        return self.client, model

    def _get_fallback_chain(self, model: str) -> list:
        """Return the fallback chain for the given model (Phase 4 -- model fallback)."""
        chains = [
            Config.MODEL_FAST_CHAIN,
            Config.MODEL_REASON_CHAIN,
            Config.MODEL_CODE_CHAIN,
            Config.MODEL_VISION_CHAIN,
            Config.MODEL_THINK_CHAIN,
            Config.MODEL_LONGCTX_CHAIN,
        ]
        for chain in chains:
            if model in chain:
                return chain
        return []

    def _resolve_tool_model(
        self,
        model_override: str = None,
        options: dict = None,
    ) -> tuple:
        """Resolve model, client, and LLM options for tool-calling methods.

        Returns:
            (client, actual_model, llm_options) tuple
        """
        model = model_override or self._model_override or Config.MODEL_CODE
        client, actual_model = self._get_client_for_model(model)
        llm_options = options or {"temperature": 0.2, "num_predict": 4096}
        return client, actual_model, llm_options

    def _resolve_chat_client(self, model: str) -> tuple:
        """Resolve client and actual model, record thinking panel event.

        Returns:
            (client, actual_model) tuple
        """
        client, actual_model = self._get_client_for_model(model)
        if actual_model != model:
            logger.info(f"[BRAIN] Model fallback: {model} -> {actual_model}")
        self._last_model_used = actual_model

        # Record real thinking -- LLM inference starting
        try:
            from api.routes.thinking import get_manager as get_thinking_manager
            tm = get_thinking_manager()
            tm.record_real_thought("formulating", f"reasoning with {actual_model}...", intensity=0.7, source="brain")
        except Exception as e:
            logger.debug(f"[Brain] non-critical: {e}")

        return client, actual_model

    def _is_complex_query(self, prompt: str) -> bool:
        """Detect if a query is complex and needs cloud model.

        Complex queries include:
        - Research/analysis requests
        - Multi-step reasoning
        - Comparisons requiring deep knowledge
        - Long-form content generation
        """
        prompt_lower = prompt.lower()
        words = prompt.split()

        # Long prompts are likely complex
        if len(words) > 50:
            return True

        # Complex task indicators -- must be explicit task requests, not conversational references
        # Bad: 'research', 'review', 'tell me about' -- match casual questions like
        #      "what do you think about this research?" -> wrongly triggers 397B model
        complex_patterns = [
            'write an essay', 'write a report', 'write a detailed',
            'comprehensive analysis', 'in-depth analysis', 'thorough analysis',
            'deep dive into', 'deep search', 'investigate in detail',
            'pros and cons of', 'advantages and disadvantages',
            'step by step guide', 'detailed explanation of',
            'compare and contrast',
        ]

        if any(pattern in prompt_lower for pattern in complex_patterns):
            return True

        return False

    def set_model_override(self, model: Optional[str]) -> None:
        """Set a manual model override that bypasses auto-selection.

        Args:
            model: Model name to force, or None to return to auto-selection
        """
        self._model_override = model
        if model:
            logger.info(f"[BRAIN] Model override set: {model}")
        else:
            logger.info("[BRAIN] Model override cleared, returning to auto-selection")

    def set_action_mode(self, mode: Optional[str]) -> None:
        """Set the current action mode for context-aware prompt injection.

        Used by agent_service to pass the detected action mode so that
        _build_full_system_prompt can inject mode-specific prompts
        (e.g., design system for frontend/artifact modes).

        Args:
            mode: Action mode string or None to clear
        """
        self._action_mode = mode

    def _get_domain_confidence(self, prompt: str) -> tuple:
        """Get domain and confidence score from metacognition for a prompt.

        Returns:
            (domain_name: str, confidence: float) tuple.
            Falls back to (None, 0.5) if metacognition unavailable.
        """
        try:
            from aura.consciousness.metacognition import get_metacognitive_engine
            engine = get_metacognitive_engine()
            domain = engine.get_domain_for_query(prompt)
            if domain is None:
                return (None, 0.5)
            caps = engine.assess_capabilities()
            cap = caps.get(domain.value)
            if cap and cap.confidence > 0.1:
                return (domain.value, cap.score)
            return (domain.value, 0.5)
        except Exception as e:
            logger.debug(f"[BRAIN] Capability check failed: {e}")
            return (None, 0.5)

    def _should_escalate_to_system2(self, prompt: str, task_type=None) -> tuple:
        """Decide whether to use System 2 (deliberative) over System 1 (fast).

        Implements Kahneman-inspired dual-process routing:
        - Direct System 2 triggers for known complex patterns
        - Confidence-based escalation via metacognition
        - Neuromodulator tie-breaking for mid-range confidence

        Returns:
            (use_system2: bool, domain: str, confidence: float, reason: str)
        """
        from aura.brain import TaskType, _get_neuromodulator_levels

        # Direct System 2 triggers
        if self._is_complex_query(prompt):
            return (True, None, 0.0, "complex_query_heuristic")
        if task_type == TaskType.REASONING:
            return (True, None, 0.0, "explicit_reasoning_task")

        # Confidence-based escalation
        domain, confidence = self._get_domain_confidence(prompt)

        if confidence < Config.S2_CONFIDENCE_THRESHOLD:
            return (True, domain, confidence, "low_confidence")
        if confidence > Config.S1_CONFIDENCE_THRESHOLD:
            return (False, domain, confidence, "high_confidence")

        # Mid-range confidence: use neuromodulator state as tie-breaker
        neuro = _get_neuromodulator_levels()
        if neuro["norepinephrine"] > 0.6:
            return (True, domain, confidence, "high_norepinephrine")
        if neuro["dopamine"] > 0.7:
            return (False, domain, confidence, "high_dopamine")

        return (False, domain, confidence, "default_fast")

    # ------------------------------------------------------------------
    # Outcome-aware routing helpers
    # ------------------------------------------------------------------

    def _routing_stats_override(self, model: str, task_type=None, user_selected: bool = False) -> str:
        """Apply outcome-aware routing stats overlay to heuristic model selection.

        Only activates when ENABLE_OUTCOME_AWARE_ROUTING=True and RoutingStats
        has >=MIN_SAMPLES data for the selected chain + microtask category.
        Falls back to heuristic model unchanged when data is insufficient.

        Args:
            user_selected: If True, the model was explicitly chosen by the user
                           (via UI dropdown or CLI). NEVER override user selections.
        """
        from aura.brain import TaskType

        if not getattr(Config, "ENABLE_OUTCOME_AWARE_ROUTING", True):
            return model
        # Don't override user's explicit model choice (parameter OR instance-level)
        if user_selected or self._model_override:
            return model
        try:
            from aura.reliability.routing_stats import get_routing_stats, MicrotaskCategory
            _CAT_MAP = {
                TaskType.CODE:      MicrotaskCategory.CODE_EDIT,
                TaskType.VISION:    MicrotaskCategory.LONG_DOC_EXTRACTION,
                TaskType.REASONING: MicrotaskCategory.TOOL_SELECTION,
                TaskType.SIMPLE:    MicrotaskCategory.GENERAL,
            }
            category = _CAT_MAP.get(task_type, MicrotaskCategory.GENERAL)
            chain = self._get_fallback_chain(model) or [model]
            stats_model = get_routing_stats().select_model_for_task(category, chain)
            if stats_model and stats_model != model:
                logger.info(
                    "[BRAIN] RoutingStats override: %s -> %s (cat=%s)",
                    model, stats_model, category,
                )
                return stats_model
        except Exception as e:
            logger.debug(f"[Brain] non-critical: {e}")
        return model

    def _record_routing_outcome(
        self, model: str, task_type, success: bool, latency_ms: float
    ) -> None:
        """Record routing outcome to RoutingStatsStore (called in background)."""
        from aura.brain import TaskType

        try:
            from aura.reliability.routing_stats import get_routing_stats, MicrotaskCategory
            _CAT_MAP = {
                TaskType.CODE:      MicrotaskCategory.CODE_EDIT,
                TaskType.VISION:    MicrotaskCategory.LONG_DOC_EXTRACTION,
                TaskType.REASONING: MicrotaskCategory.TOOL_SELECTION,
                TaskType.SIMPLE:    MicrotaskCategory.GENERAL,
            }
            category = _CAT_MAP.get(task_type, MicrotaskCategory.GENERAL)
            get_routing_stats().record(category, model, success=success, latency_ms=latency_ms)
        except Exception as e:
            logger.debug(f"[Brain] non-critical: {e}")

    def _select_model(self, prompt: str, task_type=None) -> str:
        """Select the appropriate model based on task type and complexity.

        SYSTEM 1/SYSTEM 2 HYBRID ROUTING (Kahneman dual-process):
        - System 1 (fast): Simple queries, high confidence -> MODEL_FAST
        - System 2 (deliberative): Complex queries, low confidence -> MODEL_REASON
        - Specialized: Vision/Code tasks use dedicated model chains
        - Cloud: Complex queries that need cloud-scale models

        Args:
            prompt: The prompt to analyze
            task_type: Explicit task type, or None for auto-detection

        Returns:
            Model name to use
        """
        from aura.brain import TaskType

        # Check for manual override first
        if self._model_override:
            logger.info(f"[BRAIN] Using manual model override: {self._model_override}")
            return self._model_override

        # Short conversational queries always use fast model -- skip all escalation logic.
        # No 397B model needed for "what do you think?" or "how does this work?"
        words = prompt.split()
        # Code/dev keywords override the short-query fast-model forcing
        _CODE_KWS = {
            'code', 'bug', 'fix', 'debug', 'test', 'tests', 'function',
            'script', 'error', 'implement', 'refactor', 'compile', 'run',
            'deploy', 'build', 'import', 'class', 'method', 'api',
            'database', 'query', 'sql', 'python', 'javascript',
        }
        if len(words) <= 5 and any(kw in prompt.lower() for kw in _CODE_KWS):
            logger.info(f"[BRAIN] Short code query ({len(words)} words) -> code model")
            return Config.get_model("code")
        # Trivial queries (<=5 words): always fast model, no escalation possible
        if len(words) <= 5:
            logger.info(f"[BRAIN] Trivial query ({len(words)} words) -> fast model (forced)")
            return Config.MODEL_FAST
        # Short queries (6-15 words): fast model unless complex
        if len(words) <= 15 and not self._is_complex_query(prompt):
            logger.info(f"[BRAIN] Short query ({len(words)} words) -> fast model")
            return Config.MODEL_FAST

        use_cloud = self._is_complex_query(prompt)
        prompt_lower = prompt.lower()

        # Specialized task routing (Vision/Code have dedicated models)
        if task_type == TaskType.VISION or any(kw in prompt_lower for kw in ['image', 'picture', 'screenshot', 'photo', 'analyze image']):
            return Config.get_model("vision")

        if task_type == TaskType.CODE:
            return Config.get_model("code")

        # Code detection from prompt keywords
        code_patterns = [
            'calculate', 'compute', 'factorial', 'fibonacci', 'prime',
            'print(', 'import ', 'def ', 'python',
            'code', 'script', 'function', 'algorithm',
            'debug', 'fix this', 'fix the', 'write a script', 'implement',
            'refactor', 'class ', 'method', 'variable', 'loop',
            'error', 'exception', 'traceback', 'bug', 'syntax'
        ]
        if any(pattern in prompt_lower for pattern in code_patterns):
            return Config.get_model("code")

        # Identity questions always use reasoning model
        identity_patterns = [
            'what is your name', 'who are you', 'your name', 'are you called',
            'what should i call you', 'introduce yourself', 'tell me about yourself',
            'what are you', 'are you an ai', 'are you a bot', 'what model are you'
        ]
        if any(pattern in prompt_lower for pattern in identity_patterns):
            return Config.get_model("reason")

        # System 1/System 2 decision for all other queries
        use_s2, domain, confidence, reason = self._should_escalate_to_system2(prompt, task_type)

        # Apply explicit thinking-mode override + cognitive load
        try:
            from aura.thinking_mode import get_thinking_mode_manager
            tmm = get_thinking_mode_manager()
            use_s2, reason = tmm.get_effective_decision(use_s2)
            # Track this query in cognitive load window
            was_complex = use_cloud or (task_type == TaskType.CODE)
            tmm.cognitive_load.record_query(confidence, was_complex, use_s2)
        except Exception as e:
            logger.debug(f"[BRAIN] Thinking mode not available: {e}")

        if use_s2:
            model = Config.get_model("reason")
            logger.info(f"[BRAIN] System 2 (deliberative): domain={domain}, confidence={confidence:.2f}, reason={reason}, model={model}")
            return model
        else:
            model = Config.MODEL_FAST
            logger.info(f"[BRAIN] System 1 (fast): domain={domain}, confidence={confidence:.2f}, reason={reason}, model={model}")
            return model

    def get_last_model_used(self) -> str:
        """Get the model used in the last think() call."""
        return self._last_model_used

    # (observe/plan/decide_action/evaluate removed -- OPAE loop replaced by ReAct)

    def unload_model(self, model: str = None) -> bool:
        """Unload a model from Ollama to free VRAM.

        Args:
            model: Model name to unload. If None, unloads the last used model.

        Returns:
            True if successful, False otherwise.
        """
        from aura.brain import call_with_timeout

        model_to_unload = model or self._last_model_used
        try:
            # Send empty generate with keep_alive=0 to unload (with timeout)
            result = call_with_timeout(
                lambda: self.client.generate(
                    model=model_to_unload,
                    prompt="",
                    keep_alive="0s"
                ),
                timeout=10,
                default=None
            )
            return result is not None
        except Exception as e:
            logger.debug(f"[BRAIN] Model keep-alive failed: {e}")
            return False

    def unload_all_models(self) -> dict:
        """Unload all commonly used models to free VRAM.

        Returns:
            Dict with unload status for each model.
        """
        models = [Config.MODEL_FAST, Config.MODEL_REASON, Config.MODEL_CODE, Config.MODEL_VISION]
        results = {}
        for model in models:
            results[model] = self.unload_model(model)
        return results
