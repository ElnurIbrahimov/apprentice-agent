"""Action-mode detection and model selection helpers for AgentService."""

from __future__ import annotations

import logging
import os
import threading
from typing import Optional

logger = logging.getLogger(__name__)

ACTION_TRIGGERS = {
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
    "quick prototype": "rapid",
    "rapid prototype": "rapid",
    "scaffold": "rapid",
    "quick mock": "rapid",
    "sketch out": "rapid",
    "artifact": "artifact",
    "ui component": "artifact",
    "react widget": "artifact",
    "build a component": "artifact",
    "create a component": "artifact",
    "not working": "debug",
    "fix this": "debug",
    "fix the": "debug",
    "debug": "debug",
    "find the bug": "debug",
    "why is this": "debug",
    "broken": "debug",
    "code review": "debug",
    "search": "search",
    "google": "search",
    "lookup": "search",
    "find online": "search",
    "web search": "search",
    "search online": "search",
    "look up": "search",
    "search for": "search",
    "search the web": "search",
    "research": "research",
    "deep dive": "research",
    "analyze": "research",
    "investigate": "research",
    "comprehensive": "research",
    "in-depth": "research",
    "detailed analysis": "research",
    "full analysis": "research",
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
    "describe image": "vision",
    "analyze image": "vision",
    "what's in this": "vision",
    "look at this": "vision",
    "explain this image": "vision",
    "screenshot": "vision",
    "deep research": "deep_research",
    "thorough research": "deep_research",
    "extensive research": "deep_research",
    "full research": "deep_research",
    "research everything": "deep_research",
    "research in depth": "deep_research",
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

ACTION_MODE_MODELS = {
    "frontend": {
        "preferred": "kimi-k2.5:cloud",
        "fallbacks": ["chatgpt:gpt-5.3-codex"],
    },
    "rapid": {
        "preferred": "chatgpt:gpt-5.3-codex-spark",
        "fallbacks": ["nemotron-3-super:cloud"],
    },
    "code": {
        "preferred": "minimax-m2.5:cloud",
        "fallbacks": ["glm-5.1:cloud", "qwen3-coder:480b-cloud"],
    },
    "search": {
        "preferred": "nemotron-3-super:cloud",
        "fallbacks": ["glm-5.1:cloud", "glm-5:cloud"],
    },
    "research": {
        "preferred": "qwen3.5:397b-cloud",
        "fallbacks": ["kimi-k2.5:cloud"],
    },
    "deep_research": {
        "preferred": "qwen3.5:397b-cloud",
        "fallbacks": ["kimi-k2.5:cloud"],
    },
    "debug": {
        "preferred": "chatgpt:gpt-5.4-thinking",
        "fallbacks": ["glm-5.1:cloud", "minimax-m2.7:cloud"],
    },
    "vision": {
        "preferred": "kimi-k2.5:cloud",
        "fallbacks": ["chatgpt:gpt-5.4"],
    },
    "swarm": {
        "preferred": "minimax-m2.7:cloud",
        "fallbacks": ["qwen3.5:397b-cloud"],
    },
    "artifact": {
        "preferred": "kimi-k2.5:cloud",
        "fallbacks": ["minimax-m2.5:cloud"],
    },
    "agent": {
        "preferred": "kimi-k2.5:cloud",
        "fallbacks": ["glm-5.1:cloud", "minimax-m2.7:cloud"],
    },
}

_classifier_client = None
_classifier_client_lock = threading.Lock()
_available_models: set[str] = set()
_models_loaded = False
_models_loaded_at = 0.0
_MODELS_TTL_SECONDS = 300
_models_lock = threading.Lock()

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

_VALID_MODES = {
    "frontend",
    "code",
    "debug",
    "search",
    "research",
    "deep_research",
    "vision",
    "rapid",
    "swarm",
    "agent",
    "artifact",
}


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
                            headers={"Authorization": f"Bearer {api_key}"},
                        )
                        logger.info("[ActionMode] LLM classifier client initialized")
                    else:
                        logger.debug("[ActionMode] No OLLAMA_API_KEY, classifier unavailable")
                except Exception as exc:
                    logger.warning("[ActionMode] Failed to create classifier client: %s", exc)
    return _classifier_client


def detect_action_mode(message: str) -> Optional[str]:
    """Classify intent with a fast LLM call, then fall back to keywords."""
    words = message.split()
    if len(words) < 4:
        return None

    if len(words) <= 12:
        msg_lower = message.lower()
        task_indicators = {
            "create",
            "build",
            "make",
            "generate",
            "write",
            "code",
            "fix",
            "debug",
            "search",
            "find",
            "look up",
            "research",
            "analyze",
            "deploy",
            "implement",
            "design",
            "draw",
            "render",
            "screenshot",
            "review",
            "compare",
            "test",
            "refactor",
            "optimize",
            "automate",
            "scrape",
            "crawl",
            "translate",
        }
        if not any(indicator in msg_lower for indicator in task_indicators):
            return None

    client = _get_classifier_client()
    if client is None:
        return _keyword_fallback(message)

    try:
        from aura.pools import llm_pool

        prompt = _CLASSIFICATION_PROMPT.format(message=message[:200])

        def _classify():
            return client.chat(
                model="nemotron-3-super:cloud",
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0, "num_predict": 10},
            )

        response = llm_pool().submit(_classify).result(timeout=5)

        raw = response.get("message", {}).get("content", "").strip().lower()
        category = raw.split()[0].rstrip(".,;:") if raw else ""
        if category in _VALID_MODES:
            logger.info("[ActionMode] LLM classified as: %s", category)
            return category

        logger.info("[ActionMode] LLM returned '%s' -> no special mode", category)
        return None
    except Exception as exc:
        logger.warning("[ActionMode] LLM classification failed: %s; using keyword fallback", exc)
        return _keyword_fallback(message)


def _keyword_fallback(message: str) -> Optional[str]:
    """Fallback keyword-based detection when the classifier is unavailable."""
    import re

    msg_lower = message.lower().strip()
    sorted_triggers = sorted(ACTION_TRIGGERS.keys(), key=len, reverse=True)
    for trigger in sorted_triggers:
        if " " in trigger:
            if trigger in msg_lower:
                return ACTION_TRIGGERS[trigger]
        elif re.search(r"\b" + re.escape(trigger) + r"\b", msg_lower):
            return ACTION_TRIGGERS[trigger]
    return None


def _load_available_models() -> None:
    """Populate available-model cache from Ollama."""
    import time

    global _available_models, _models_loaded, _models_loaded_at
    with _models_lock:
        now = time.time()
        if _models_loaded and (now - _models_loaded_at) < _MODELS_TTL_SECONDS:
            return
        try:
            import ollama

            result = ollama.list()
            _available_models = {model.model for model in result.models}
            _models_loaded = True
            _models_loaded_at = now
            logger.info("[AutoModel] Loaded %d models", len(_available_models))
        except Exception as exc:
            logger.warning("[AutoModel] Could not list Ollama models: %s", exc)
            if not _models_loaded:
                _available_models = set()


def _is_model_available(model: str) -> bool:
    """Check if a routed model is available locally or via cloud."""
    if not model:
        return False
    if model.endswith(("-cloud", ":cloud")):
        return True
    return model in _available_models


def get_model_for_action(action_mode: str) -> Optional[str]:
    """Return the best available model for a detected action mode."""
    if action_mode not in ACTION_MODE_MODELS:
        return None

    _load_available_models()
    config = ACTION_MODE_MODELS[action_mode]
    candidates = [config.get("preferred"), *config.get("fallbacks", [])]
    for model in candidates:
        if model and _is_model_available(model):
            logger.info("[AutoModel] Action '%s' -> %s", action_mode, model)
            return model

    logger.warning("[AutoModel] No available model for action '%s'", action_mode)
    return None
