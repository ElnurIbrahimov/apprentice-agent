"""Interactive model picker for AURA CLI — arrow-key navigable, shows ALL models."""
from __future__ import annotations

import logging
import os
import time
from typing import Any, Optional

from rich.console import Console as _Console

_logger = logging.getLogger(__name__)

# Model roles with display info (updated from Config on startup)
MODEL_ROLES: list[tuple[str, str, str]] = [
    ("fast", "nemotron-3-super:cloud", "fast inference"),
    ("reason", "kimi-k2.6:cloud", "256K ctx, top agentic"),
    ("code", "minimax-m2.7:cloud", "1M ctx, SWE-Pro 56.2%"),
    ("think", "qwen3.5:397b-cloud", "256K ctx, hybrid thinking"),
    ("vision", "kimi-k2.6:cloud", "multimodal"),
    ("longctx", "minimax-m2.7:cloud", "1M ctx"),
]

# TTL-based cache for fetched Ollama models (avoids permanent stale list)
_MODELS_CACHE_TTL: float = 60.0
_models_cache_result: list[str] = []
_models_cache_ts: float = 0.0
# Last fetch status — surfaces a one-line error in the picker footer when the
# Ollama fetch fails, instead of silently returning an empty list.
_models_fetch_error: Optional[str] = None


def invalidate_models_cache() -> None:
    """Clear the Ollama model cache so the next fetch refreshes.

    Bound to F5 inside the picker; lets users force-refresh after starting a
    new model pull or restarting the Ollama daemon without waiting out the
    60s TTL.
    """
    global _models_cache_result, _models_cache_ts
    _models_cache_result = []
    _models_cache_ts = 0.0


def _fetch_all_models(force: bool = False) -> list[str]:
    """Fetch all available models from Ollama (cached for 60s).

    force=True bypasses the cache — used by the picker's F5 refresh binding.
    """
    global _models_cache_result, _models_cache_ts, _models_fetch_error
    now = time.monotonic()
    if not force and _models_cache_result and (now - _models_cache_ts) < _MODELS_CACHE_TTL:
        return _models_cache_result
    try:
        import requests
        host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        resp = requests.get(f"{host}/api/tags", timeout=5)
        if resp.status_code == 200:
            models = resp.json().get("models", [])
            _models_cache_result = [m["name"] for m in models]
            _models_cache_ts = now
            _models_fetch_error = None
        else:
            _models_fetch_error = f"Ollama returned HTTP {resp.status_code}"
    except Exception as exc:
        _models_fetch_error = f"Ollama fetch failed: {exc}"
        _logger.debug("ollama_model_fetch_failed", exc_info=True)
    return _models_cache_result


def last_fetch_error() -> Optional[str]:
    """Read-only accessor for the last Ollama fetch error (or None)."""
    return _models_fetch_error


def _fetch_chatgpt_models() -> list[str]:
    """Get available ChatGPT models if authenticated."""
    try:
        from aura.auth.chatgpt_oauth import is_authenticated
        if is_authenticated():
            from aura.auth.chatgpt_client import ALL_CHATGPT_MODELS
            return list(ALL_CHATGPT_MODELS)
    except ImportError:
        pass
    return []


def _build_model_list(current_model: str) -> list[tuple[str, str, str]]:
    """Build the full model list: auto first, then ChatGPT, then role models, then all others."""
    items: list[tuple[str, str, str]] = []

    # First item: auto
    role_tag = "auto-route"
    items.append(("auto", role_tag, ""))

    # ChatGPT OAuth models (if authenticated)
    chatgpt_models = _fetch_chatgpt_models()
    seen: set[str] = set()
    for m in chatgpt_models:
        if m not in seen:
            seen.add(m)
            items.append((m, "chatgpt", "$0"))

    # Role-mapped models (deduplicated)
    for role, model, ctx in MODEL_ROLES:
        if model not in seen:
            seen.add(model)
            items.append((model, role, ctx))

    # Cloud models from config (these don't appear in /api/tags)
    try:
        from aura.config import VERIFIED_CLOUD_MODELS
        for m in sorted(VERIFIED_CLOUD_MODELS):
            if m not in seen:
                seen.add(m)
                items.append((m, "cloud", ""))
    except ImportError:
        pass

    # Direct API provider models (if keys are configured)
    try:
        from aura.providers import list_all_provider_models
        for m, provider_name in list_all_provider_models():
            if m not in seen:
                seen.add(m)
                items.append((m, provider_name, "api"))
    except ImportError:
        pass

    # All Ollama local models not already listed
    all_models = _fetch_all_models()
    for m in all_models:
        if m not in seen:
            seen.add(m)
            if m.startswith("chatgpt:"):
                tag, ctx = "chatgpt", "$0"
            elif ":cloud" in m or "-cloud" in m:
                tag, ctx = "cloud", ""
            else:
                tag, ctx = "local", ""
            items.append((m, tag, ctx))

    return items


def pick_model(console: _Console, current_model: str = "auto") -> Optional[str]:
    """Show interactive model picker with arrow-key navigation.

    Returns model name, 'auto', or None (cancelled).
    """
    try:
        return _pick_model_interactive(current_model)
    except Exception:
        # Fallback to simple input if prompt_toolkit fails
        return _pick_model_fallback(console, current_model)


def _pick_model_interactive(current_model: str) -> Optional[str]:
    """Full interactive picker using shared picker component."""
    from aura.cli.picker import PickerItem, run_picker

    raw_items = _build_model_list(current_model)
    if not raw_items:
        return None

    # Build PickerItem list with pre-formatted descriptions and status indicators
    available_models = set(_fetch_all_models())
    picker_items: list[Any] = []
    for model, role, ctx in raw_items:
        model_display = model.replace(":cloud", "").replace(":latest", "")
        # Determine availability status
        is_cloud = ":cloud" in model or "-cloud" in model
        is_chatgpt = model.startswith("chatgpt:") or role == "chatgpt"
        is_auto = model == "auto"
        is_api = ctx == "api"
        if is_auto or is_cloud or is_chatgpt or is_api or model in available_models:
            status = "\u2713 "  # checkmark — ready
        else:
            status = "\u2717 "  # cross — offline/unavailable
        # Build description from status + role tag + context + current marker
        desc_parts = [status]
        if role:
            desc_parts.append(role)
        if ctx:
            desc_parts.append(ctx)
        if model == current_model:
            desc_parts.append("<-")
        description = "  ".join(desc_parts)
        picker_items.append(PickerItem(
            id=model,
            label=model_display,
            description=description,
        ))

    cur_display = current_model.replace(":cloud", "").replace(":latest", "")
    return run_picker(
        picker_items,
        title=f"Model Picker  (current: {cur_display})",
        max_visible=20,
    )


def _pick_model_fallback(console: _Console, current_model: str) -> Optional[str]:
    """Simple fallback picker when prompt_toolkit can't create an Application."""
    while True:
        items = _build_model_list(current_model)
        err = last_fetch_error()
        console.print("\n[bold cyan]  Model Picker[/bold cyan]")
        if err:
            console.print(f"  [yellow]{err} — showing cached results. Type 'r' to refresh.[/yellow]")
        for i, (model, role, ctx) in enumerate(items):
            marker = " [green]<-[/green]" if model == current_model else ""
            model_short = model.replace(":cloud", "").replace(":latest", "")
            console.print(f"  [bold cyan]{i + 1:>2}[/bold cyan]. {model_short:<32s} [dim yellow]{role:<10s}[/dim yellow] [dim]{ctx}[/dim]{marker}")

        console.print()
        try:
            pick = input("  Pick # or name (q to cancel, r to refresh) > ").strip()
        except (EOFError, KeyboardInterrupt):
            return None

        if not pick or pick.lower() in ("q", "esc"):
            return None

        if pick.lower() in ("r", "refresh"):
            invalidate_models_cache()
            console.print("  [dim]Refreshing model list...[/dim]")
            continue

        try:
            idx = int(pick) - 1
            if 0 <= idx < len(items):
                return items[idx][0]
        except ValueError:
            pass

        for model, _, _ in items:
            if pick.lower() in model.lower():
                return model

        return pick


def update_model_roles_from_config() -> None:
    """Refresh MODEL_ROLES from Config at runtime."""
    global MODEL_ROLES
    try:
        from aura.config import Config
        MODEL_ROLES = [
            ("fast", Config.MODEL_FAST, "1M ctx"),
            ("reason", Config.MODEL_REASON, "256K ctx"),
            ("code", Config.MODEL_CODE, "196K ctx"),
            ("think", Config.MODEL_THINK, "256K ctx"),
            ("vision", Config.MODEL_VISION, "256K ctx"),
            ("longctx", Config.MODEL_LONGCTX, "1M ctx"),
        ]
    except Exception:
        pass
