"""Auxiliary model roles — per-task model routing.

Inspired by Hermes Agent's auxiliary config section. Each auxiliary task
(vision, compression, world_model, memory, etc.) can use a different
model/provider, offloading from the main llm_pool.

Config section (in ~/.aura/config.yaml):
    auxiliary:
      vision:      { provider: ollama-cloud, model: kimi-k2.6:cloud }
      compression: { provider: ollama-cloud, model: nemotron-3-super:cloud }
      world_model: { provider: ollama-cloud, model: glm-5:cloud }
      memory:      { provider: ollama-cloud, model: nemotron-3-super:cloud }

If a role is not configured, falls back to the main model (Config.MODEL_FAST).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class AuxiliaryRole:
    """A configured auxiliary model role."""
    role: str
    provider: str
    model: str
    timeout: int = 120


# Known auxiliary roles with descriptions
AUXILIARY_ROLES = {
    "vision": "Image analysis / OCR / visual reasoning",
    "compression": "Context compression / conversation summarization",
    "world_model": "World model extraction from conversations",
    "memory": "Memory retrieval and indexing",
    "title_generation": "Session title generation",
    "mcp": "MCP tool selection / routing",
    "session_search": "Session history semantic search",
}


def get_auxiliary_config(role: str) -> Optional[AuxiliaryRole]:
    """Get the auxiliary config for a specific role.

    Returns None if the role is not configured (caller should fall back
    to the main model).
    """
    try:
        from aura.config_loader import get_auxiliary_config as _get_cfg
        aux_cfg = _get_cfg()
    except ImportError:
        return None

    cfg = aux_cfg.get(role)
    if not cfg or not isinstance(cfg, dict):
        return None

    provider = cfg.get("provider", "")
    model = cfg.get("model", "")
    timeout = cfg.get("timeout", 120)

    if not provider or not model:
        return None

    return AuxiliaryRole(
        role=role,
        provider=provider,
        model=model,
        timeout=timeout,
    )


def get_auxiliary_model(role: str) -> Tuple[Optional[object], str]:
    """Resolve the (client, model) pair for an auxiliary role.

    Uses the same routing logic as OllamaBrain._get_client_for_model():
    - provider-prefixed models route to direct API providers
    - *:cloud models route to Ollama cloud
    - everything else routes to local Ollama

    Falls back to Config.MODEL_FAST if the role is not configured.

    Returns:
        (client, model_name) tuple. Client is None on failure.
    """
    aux = get_auxiliary_config(role)

    if aux is None:
        # Fall back to fast model
        try:
            from aura.config import Config
            fallback_model = Config.MODEL_FAST
        except ImportError:
            return None, ""

        return _resolve_client(fallback_model)

    return _resolve_client(aux.model)


def _resolve_client(model: str) -> Tuple[Optional[object], str]:
    """Resolve a model name to a (client, model) pair.

    This mirrors OllamaBrain._get_client_for_model() but is standalone
    so it can be called from any context (brain, tools, memory, etc.)
    without needing a brain instance.
    """
    # ChatGPT OAuth
    if model.startswith("chatgpt:"):
        try:
            from aura.auth.chatgpt_oauth import is_authenticated
            if is_authenticated():
                from aura.auth.chatgpt_client import ChatGPTClient
                return ChatGPTClient(), model
        except ImportError:
            pass

    # Direct API providers (anthropic:, openai:, gemini:, etc.)
    if ":" in model and not model.endswith(("-cloud", ":cloud", ":latest")):
        prefix = model.split(":")[0]
        try:
            from aura.providers import get_provider
            provider = get_provider(prefix)
            if provider and provider.is_configured():
                return provider, model
        except Exception:
            pass

    # Ollama cloud
    if model.endswith(("-cloud", ":cloud")):
        try:
            import ollama
            from aura.config import Config
            api_key = Config.OLLAMA_API_KEY if hasattr(Config, "OLLAMA_API_KEY") else ""
            if api_key:
                import os
                key = os.getenv("OLLAMA_API_KEY", "")
                if key:
                    return ollama.Client(
                        host="https://api.ollama.com",
                        headers={"Authorization": f"Bearer {key}"},
                        timeout=120,
                    ), model
        except ImportError:
            pass

    # Local Ollama
    try:
        import ollama
        from aura.config import Config
        return ollama.Client(host=Config.OLLAMA_HOST, timeout=120), model
    except ImportError:
        return None, model


def list_auxiliary_roles() -> list[dict]:
    """List all known auxiliary roles with their current configuration."""
    result = []
    for role, description in AUXILIARY_ROLES.items():
        aux = get_auxiliary_config(role)
        if aux:
            result.append({
                "role": role,
                "description": description,
                "configured": True,
                "provider": aux.provider,
                "model": aux.model,
            })
        else:
            result.append({
                "role": role,
                "description": description,
                "configured": False,
                "provider": "",
                "model": "",
            })
    return result
