"""Provider registry — get_provider(), list_configured_providers(), list_all_provider_models().

Lazily instantiates providers on first access to avoid import overhead.
"""

import logging
from typing import Optional

from .base import BaseProvider
from .registry import OPENAI_COMPATIBLE_PROVIDERS, PROVIDER_CONFIGS

logger = logging.getLogger(__name__)

# Lazy-initialized provider cache
_providers: dict[str, BaseProvider] = {}
_initialized = False


def _init_providers():
    """Create provider instances (once).

    Reads custom providers from config.yaml via config_loader, then
    registers built-in providers from registry.py. Custom providers
    from config.yaml are merged on top of built-ins.
    """
    global _initialized
    if _initialized:
        return
    _initialized = True

    # Get merged provider configs (config.yaml + built-in defaults)
    try:
        from aura.config_loader import get_providers_config
        merged_configs = get_providers_config()
    except ImportError:
        merged_configs = dict(PROVIDER_CONFIGS)

    # Determine which providers are OpenAI-compatible vs custom
    from .openai_compat import OpenAICompatProvider

    # Build-in OpenAI-compatible providers (from registry.py list)
    for name in OPENAI_COMPATIBLE_PROVIDERS:
        cfg = merged_configs.get(name, PROVIDER_CONFIGS.get(name))
        if not cfg:
            continue
        _providers[name] = OpenAICompatProvider(
            provider_name=name,
            base_url=cfg["base_url"],
            env_var=cfg.get("env_var", cfg.get("api_key_env", "")),
            display_name=cfg.get("display_name", name),
            default_models=cfg.get("default_models", cfg.get("models", [])),
        )

    # Custom OpenAI-compatible providers from config.yaml (not in built-in list)
    builtin_names = set(OPENAI_COMPATIBLE_PROVIDERS) | {"anthropic", "gemini"}
    for name, cfg in merged_configs.items():
        if name in builtin_names:
            continue
        if not isinstance(cfg, dict) or "base_url" not in cfg:
            continue
        # Only register if it looks like an OpenAI-compatible endpoint
        # (custom Anthropic/Gemini would need their own provider classes)
        env_var = cfg.get("env_var", cfg.get("api_key_env", ""))
        models = cfg.get("default_models", cfg.get("models", []))
        # Normalize model names: if they don't have provider prefix, add it
        normalized_models = [
            m if ":" in m else f"{name}:{m}"
            for m in (models or [])
        ]
        _providers[name] = OpenAICompatProvider(
            provider_name=name,
            base_url=cfg["base_url"],
            env_var=env_var,
            display_name=cfg.get("display_name", name),
            default_models=normalized_models,
        )
        logger.info(f"[Providers] Custom provider '{name}' registered from config.yaml")

    # Built-in custom-API providers (non-OpenAI format)
    from .anthropic_provider import AnthropicProvider
    _providers["anthropic"] = AnthropicProvider()

    from .gemini_provider import GeminiProvider
    _providers["gemini"] = GeminiProvider()


def get_provider(prefix_or_name: str) -> Optional[BaseProvider]:
    """Get a provider by prefix ('anthropic:') or name ('anthropic').

    Returns None if no such provider exists.
    """
    _init_providers()
    name = prefix_or_name.rstrip(":")
    return _providers.get(name)


def list_configured_providers() -> list[dict]:
    """Return list of all providers with their configuration status."""
    _init_providers()
    result = []
    for name, provider in sorted(_providers.items()):
        result.append({
            "name": name,
            "display_name": provider.display_name,
            "prefix": provider.prefix,
            "configured": provider.is_configured(),
            "model_count": len(provider.list_models()),
        })
    return result


def list_all_provider_models() -> list[tuple[str, str]]:
    """Return all models from configured providers.

    Returns list of (prefixed_model_name, display_name) tuples.
    Only includes providers with API keys set.
    """
    _init_providers()
    result = []
    for _name, provider in sorted(_providers.items()):
        if provider.is_configured():
            for model in provider.list_models():
                result.append((model, provider.display_name))
    return result


def get_all_providers() -> dict[str, BaseProvider]:
    """Return all provider instances."""
    _init_providers()
    return dict(_providers)
