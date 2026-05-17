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
    """Create provider instances (once)."""
    global _initialized
    if _initialized:
        return
    _initialized = True

    # OpenAI-compatible providers (8 of 10)
    from .openai_compat import OpenAICompatProvider
    for name in OPENAI_COMPATIBLE_PROVIDERS:
        cfg = PROVIDER_CONFIGS[name]
        _providers[name] = OpenAICompatProvider(
            provider_name=name,
            base_url=cfg["base_url"],
            env_var=cfg["env_var"],
            display_name=cfg["display_name"],
            default_models=cfg["default_models"],
        )

    # Custom providers
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
