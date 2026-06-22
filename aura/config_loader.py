"""Aura configuration loader — YAML config file + env var merging.

This is the central config system, inspired by Hermes Agent's config.yaml.
It loads ~/.aura/config.yaml (or a profile-specific path), merges with
environment variables (env takes priority), and provides a typed accessor.

Config file location:
  - Default: ~/.aura/config.yaml
  - Profile-specific: ~/.aura/profiles/<name>/config.yaml
  - Override: AURA_CONFIG_FILE env var

The config file is optional — all values have defaults from the existing
Config class and providers/registry.py. If the file doesn't exist, the
system works exactly as before (pure env-var + hardcoded defaults).

Schema (all sections optional):

  model:
    default: kimi-k2.6:cloud       # main model
    provider: ollama-cloud          # provider name
    tier: balanced                  # fast | balanced | max

  providers:
    <name>:
      base_url: https://...
      api_key: ''                   # empty = read from .env
      api_key_env: MY_API_KEY       # env var to read
      models: [model-a, model-b]    # available models

  auxiliary:
    vision:      { provider: ollama-cloud, model: kimi-k2.6:cloud }
    compression: { provider: ollama-cloud, model: nemotron-3-super:cloud }
    world_model: { provider: ollama-cloud, model: glm-5:cloud }
    memory:      { provider: ollama-cloud, model: nemotron-3-super:cloud }

  fallback_providers: [deepseek, openrouter, ollama-cloud]

  toolsets:
    enabled:  [core, research, media]
    disabled: [cognitive, voice]
    platform:
      cli:      [core, research, media, system]
      telegram: [core, research, knowledge]
      api:      [core, research]

  profiles:
    active: default  # current active profile name
"""
from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Any, List, Optional

logger = logging.getLogger(__name__)

# ── Paths ───────────────────────────────────────────────────────────────

def get_aura_home() -> Path:
    """Return the Aura home directory (~/.aura or AURA_HOME env var)."""
    env = os.environ.get("AURA_HOME")
    if env:
        return Path(env)
    return Path.home() / ".aura"


def get_config_path() -> Path:
    """Return the path to config.yaml for the current profile."""
    profile = os.environ.get("AURA_PROFILE", "default")
    if profile == "default":
        return get_aura_home() / "config.yaml"
    return get_aura_home() / "profiles" / profile / "config.yaml"


def get_env_path() -> Path:
    """Return the path to .env for the current profile."""
    profile = os.environ.get("AURA_PROFILE", "default")
    if profile == "default":
        return get_aura_home() / ".env"
    return get_aura_home() / "profiles" / profile / ".env"


# ── Config loading ──────────────────────────────────────────────────────

_lock = threading.Lock()
_cached_config: Optional[dict] = None
_config_mtime: float = 0.0


def _load_yaml(path: Path) -> dict:
    """Load a YAML file, returning empty dict on missing/corrupt."""
    if not path.exists():
        return {}
    try:
        import yaml
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if isinstance(data, dict):
            return data
        logger.warning(f"[Config] {path} contains non-dict YAML, ignoring")
    except ImportError:
        logger.debug("[Config] PyYAML not installed, config.yaml disabled")
    except Exception as e:
        logger.warning(f"[Config] Failed to load {path}: {e}")
    return {}


def _deep_merge(base: dict, override: dict) -> dict:
    """Deep-merge override into base. Override values win. Returns new dict."""
    result = dict(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result


def load_config(force: bool = False) -> dict:
    """Load and cache the merged config (config.yaml + env overrides).

    Env vars take priority over config.yaml values for the keys they set.
    The result is cached and only reloaded if the file mtime changes.

    Args:
        force: Bypass cache and reload from disk.
    """
    global _cached_config, _config_mtime

    config_path = get_config_path()

    with _lock:
        # Check if we need to reload
        try:
            current_mtime = config_path.stat().st_mtime if config_path.exists() else 0.0
        except OSError:
            current_mtime = 0.0

        if not force and _cached_config is not None and current_mtime == _config_mtime:
            return _cached_config

        # Load YAML config
        yaml_config = _load_yaml(config_path)

        # Build env overrides — env vars take priority
        env_overrides = _build_env_overrides()

        config = _deep_merge(yaml_config, env_overrides)

        _cached_config = config
        _config_mtime = current_mtime
        return config


def _build_env_overrides() -> dict:
    """Build config overrides from environment variables.

    Only a few critical keys are overridden by env vars to maintain backward
    compatibility. Most config lives in config.yaml.
    """
    overrides: dict[str, Any] = {}

    # Model defaults
    model_overrides: dict[str, Any] = {}
    if os.getenv("AURA_MODEL"):
        model_overrides["default"] = os.getenv("AURA_MODEL")
    if os.getenv("AURA_MODEL_PROVIDER"):
        model_overrides["provider"] = os.getenv("AURA_MODEL_PROVIDER")
    if os.getenv("AURA_TIER"):
        model_overrides["tier"] = os.getenv("AURA_TIER")
    if model_overrides:
        overrides["model"] = model_overrides

    # Profile
    if os.getenv("AURA_PROFILE"):
        overrides.setdefault("profiles", {})["active"] = os.getenv("AURA_PROFILE")

    return overrides


def get_config_value(key_path: str, default: Any = None) -> Any:
    """Get a config value by dotted key path (e.g. 'model.default').

    Args:
        key_path: Dot-separated path into the config dict.
        default: Value to return if key doesn't exist.
    """
    config = load_config()
    parts = key_path.split(".")
    current: Any = config
    for part in parts:
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return default
    return current


def set_config_value(key_path: str, value: Any) -> bool:
    """Set a config value by dotted key path and write to config.yaml.

    Args:
        key_path: Dot-separated path (e.g. 'model.default').
        value: Value to set (will be YAML-serialized).

    Returns:
        True if written successfully.
    """
    config_path = get_config_path()

    # Load current file state (without cache, to get raw YAML)
    raw = _load_yaml(config_path)

    # Navigate to the target key, creating intermediate dicts
    parts = key_path.split(".")
    current = raw
    for part in parts[:-1]:
        if part not in current or not isinstance(current[part], dict):
            current[part] = {}
        current = current[part]
    current[parts[-1]] = value

    # Write back
    try:
        import yaml
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(raw, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    except ImportError:
        logger.error("[Config] PyYAML not installed — cannot write config.yaml")
        return False
    except Exception as e:
        logger.error(f"[Config] Failed to write {config_path}: {e}")
        return False

    # Invalidate cache
    global _cached_config, _config_mtime
    with _lock:
        _cached_config = None
        _config_mtime = 0.0

    return True


def get_providers_config() -> dict:
    """Get the providers section of config, merged with built-in defaults.

    Returns a dict mapping provider_name -> {base_url, api_key, api_key_env, models}.
    Custom providers from config.yaml are merged on top of the built-in
    PROVIDER_CONFIGS from registry.py.
    """
    # Start with built-in defaults
    try:
        from aura.providers.registry import PROVIDER_CONFIGS
        defaults = dict(PROVIDER_CONFIGS)
    except ImportError:
        defaults = {}

    # Override with config.yaml providers
    yaml_providers = get_config_value("providers", {}) or {}
    for name, cfg in yaml_providers.items():
        if name in defaults:
            defaults[name] = _deep_merge(defaults[name], cfg)
        else:
            # Custom provider — must have base_url
            if isinstance(cfg, dict) and "base_url" in cfg:
                defaults[name] = cfg

    return defaults


def get_auxiliary_config() -> dict:
    """Get the auxiliary model roles config section."""
    return get_config_value("auxiliary", {}) or {}


def get_fallback_providers() -> List[str]:
    """Get the fallback provider list."""
    return get_config_value("fallback_providers", []) or []


def get_toolsets_config() -> dict:
    """Get the toolsets config section."""
    return get_config_value("toolsets", {}) or {}


def get_active_profile() -> str:
    """Get the active profile name."""
    return get_config_value("profiles.active", "default") or "default"


def invalidate_cache() -> None:
    """Force next load_config() call to re-read from disk."""
    global _cached_config, _config_mtime
    with _lock:
        _cached_config = None
        _config_mtime = 0.0
