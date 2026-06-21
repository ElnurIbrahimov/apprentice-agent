"""/providers and /provider — provider management commands.

Lists all configured direct API providers, their status, model count,
and models. Also allows switching the active provider interactively.

Mirrors Hermes Agent's `hermes model` and `hermes auth list` pattern:
the provider list is the source of truth, not just Ollama tags.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from ..context import get_ctx
from ..display import console
from .common import command, TIER_STABLE

logger = logging.getLogger(__name__)


@command("/providers", "List all configured providers with status and models", tier=TIER_STABLE)
def handle_providers(agent: Any, arg: str, context: dict) -> Optional[str]:
    """Show a Rich table of all providers, their configuration status, and models."""
    from rich.table import Table
    from rich.panel import Panel

    try:
        from aura.providers import list_configured_providers, list_all_provider_models
    except ImportError:
        console.print("[red]Provider system not available.[/red]")
        return None

    # ── Ollama (local + cloud) ──
    import os
    ollama_host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    ollama_cloud_key = os.environ.get("OLLAMA_API_KEY", "")

    # ── ChatGPT OAuth ──
    chatgpt_available = False
    chatgpt_model_count = 0
    try:
        from aura.auth.chatgpt_oauth import is_authenticated
        if is_authenticated():
            from aura.auth.chatgpt_client import ALL_CHATGPT_MODELS
            chatgpt_available = True
            chatgpt_model_count = len(ALL_CHATGPT_MODELS)
    except ImportError:
        pass

    # ── Direct API providers ──
    providers = list_configured_providers()
    all_models = list_all_provider_models()

    # Group models by provider display name
    models_by_provider: dict[str, list[str]] = {}
    for model, display_name in all_models:
        models_by_provider.setdefault(display_name, []).append(model)

    table = Table(box=None, padding=(0, 1), show_header=True, header_style="bold")
    table.add_column("Provider", style="bold", width=18)
    table.add_column("Status", width=8, justify="center")
    table.add_column("Models", width=6, justify="right")
    table.add_column("Example Models", min_width=40)

    def _status_icon(configured: bool, available: bool) -> str:
        if configured and available:
            return "[green]\u2713[/green]"
        elif configured:
            return "[yellow]\u26a0[/yellow]"
        return "[dim]\u2717[/dim]"

    # Ollama local
    ollama_local_ok = _check_ollama_local(ollama_host)
    local_models = _get_local_ollama_models()
    table.add_row(
        "Ollama (local)",
        _status_icon(True, ollama_local_ok),
        str(len(local_models)),
        ", ".join(local_models[:4]) + (f" +{len(local_models) - 4} more" if len(local_models) > 4 else ""),
    )

    # Ollama cloud
    table.add_row(
        "Ollama Cloud",
        _status_icon(bool(ollama_cloud_key), bool(ollama_cloud_key)),
        "13" if ollama_cloud_key else "0",
        "kimi-k2.6, nemotron-3-super, qwen3.5..." if ollama_cloud_key else "Set OLLAMA_API_KEY",
    )

    # ChatGPT OAuth
    if chatgpt_available:
        table.add_row(
            "ChatGPT (OAuth)",
            "[green]\u2713[/green]",
            str(chatgpt_model_count),
            "gpt-5.4, gpt-5.4-codex, gpt-5.3...",
        )
    else:
        table.add_row(
            "ChatGPT (OAuth)",
            "[dim]\u2717[/dim]",
            "0",
            "Run: aura --login chatgpt",
        )

    # Direct API providers
    for p in providers:
        display = p["display_name"]
        configured = p["configured"]
        count = p["model_count"]
        models = models_by_provider.get(display, [])
        example = ", ".join(m.split(":", 1)[-1] for m in models[:4])
        if len(models) > 4:
            example += f" +{len(models) - 4} more"
        table.add_row(
            display,
            _status_icon(configured, configured),
            str(count),
            example or "No models configured",
        )

    # Summary counts
    configured_count = sum(1 for p in providers if p["configured"]) + (2 if ollama_cloud_key else 1) + (1 if chatgpt_available else 0)
    total_count = len(providers) + 3  # +3 for ollama local, ollama cloud, chatgpt

    console.print()
    console.print(Panel(
        table,
        title=f"[bold cyan]\U0001f310 Providers  ({configured_count}/{total_count} configured)[/bold cyan]",
        border_style="cyan",
        padding=(1, 2),
    ))
    console.print(
        "  [dim]Use /provider <name> to switch, or /model to pick a specific model.[/dim]"
    )
    console.print()

    return None


@command("/provider", "Switch active provider interactively", tier=TIER_STABLE)
def handle_provider(agent: Any, arg: str, context: dict) -> Optional[str]:
    """Switch the active provider by name, or show interactive picker if no arg.

    Usage:
        /provider              Interactive picker
        /provider anthropic    Switch to Anthropic
        /provider ollama       Switch to local Ollama
        /provider ollama-cloud Switch to Ollama Cloud
        /provider chatgpt      Switch to ChatGPT OAuth
    """
    if arg and arg.strip():
        _switch_provider(agent, arg.strip())
        return None

    # Interactive picker
    from aura.cli.picker import PickerItem, run_picker

    items = _build_provider_picker_items()
    selected = run_picker(items, title="Switch Provider", max_visible=18)
    if selected:
        _switch_provider(agent, selected)
    return None


def _switch_provider(agent: Any, name: str) -> None:
    """Switch to a named provider by setting a sensible default model."""
    name_lower = name.lower().strip()

    # Special cases: Ollama variants and ChatGPT
    if name_lower in ("ollama", "ollama-local", "local"):
        _set_model_on_agent(agent, "auto")
        console.print("[green]Switched to local Ollama (auto-routing).[/green]")
        return
    if name_lower in ("ollama-cloud", "cloud"):
        from aura.config import Config
        _set_model_on_agent(agent, Config.MODEL_FAST)
        console.print(f"[green]Switched to Ollama Cloud.[/green] [dim](model: {Config.MODEL_FAST})[/dim]")
        return
    if name_lower in ("chatgpt", "openai-oauth"):
        try:
            from aura.auth.chatgpt_oauth import is_authenticated
            if not is_authenticated():
                console.print("[red]ChatGPT not authenticated. Run: aura --login chatgpt[/red]")
                return
            from aura.auth.chatgpt_client import ALL_CHATGPT_MODELS
            first_model = sorted(ALL_CHATGPT_MODELS)[0]
            _set_model_on_agent(agent, first_model)
            console.print(f"[green]Switched to ChatGPT OAuth.[/green] [dim](model: {first_model})[/dim]")
            return
        except ImportError:
            console.print("[red]ChatGPT OAuth not available.[/red]")
            return

    # Direct API providers
    try:
        from aura.providers import get_provider
    except ImportError:
        console.print("[red]Provider system not available.[/red]")
        return

    provider = get_provider(name_lower)
    if provider is None:
        console.print(f"[red]Unknown provider: {name}[/red]")
        console.print("[dim]Available: ollama, ollama-cloud, chatgpt, anthropic, openai, gemini, "
                       "grok, perplexity, deepseek, minimax, qwen, kimi, glm, mistral, cohere, "
                       "groq, together, fireworks, openrouter, crof[/dim]")
        return

    if not provider.is_configured():
        env_var = ""
        try:
            from aura.providers.registry import PROVIDER_CONFIGS
            cfg = PROVIDER_CONFIGS.get(name_lower, {})
            env_var = cfg.get("env_var", "")
        except Exception:
            pass
        console.print(f"[red]{provider.display_name} not configured.[/red]")
        if env_var:
            console.print(f"[dim]Set {env_var} in your .env file.[/dim]")
        return

    models = provider.list_models()
    if not models:
        console.print(f"[yellow]{provider.display_name} has no models configured.[/yellow]")
        return

    first_model = models[0]
    _set_model_on_agent(agent, first_model)
    console.print(
        f"[green]Switched to {provider.display_name}.[/green] [dim](model: {first_model})[/dim]\n"
        f"  [dim]Use /model to pick a different model from this provider.[/dim]"
    )


def _set_model_on_agent(agent: Any, model_name: str) -> None:
    """Set model override on agent brain and agentic loop."""
    ctx = get_ctx()
    if ctx and getattr(ctx, "chat_session", None):
        ctx.chat_session.apply_model_override(model_name)
        return
    agent.brain.set_model_override(model_name)
    if ctx and ctx.agentic_loop:
        ctx.agentic_loop.model_override = model_name


def _build_provider_picker_items() -> list:
    """Build the provider picker list with status indicators."""
    from aura.cli.picker import PickerItem
    import os

    items: list[PickerItem] = []

    # Ollama local
    ollama_local_ok = _check_ollama_local(os.environ.get("OLLAMA_HOST", "http://localhost:11434"))
    local_icon = "\u2713" if ollama_local_ok else "\u2717"
    items.append(PickerItem(
        id="ollama",
        label="Ollama (local)",
        description=f"{local_icon} local models",
        category="runtime",
    ))

    # Ollama cloud
    cloud_key = os.environ.get("OLLAMA_API_KEY", "")
    if cloud_key:
        items.append(PickerItem(
            id="ollama-cloud",
            label="Ollama Cloud",
            description="\u2713 13 cloud models",
            category="runtime",
        ))
    else:
        items.append(PickerItem(
            id="ollama-cloud",
            label="Ollama Cloud",
            description="\u2717 Set OLLAMA_API_KEY",
            category="runtime",
        ))

    # ChatGPT OAuth
    chatgpt_ok = False
    chatgpt_count = 0
    try:
        from aura.auth.chatgpt_oauth import is_authenticated
        if is_authenticated():
            from aura.auth.chatgpt_client import ALL_CHATGPT_MODELS
            chatgpt_ok = True
            chatgpt_count = len(ALL_CHATGPT_MODELS)
    except ImportError:
        pass
    if chatgpt_ok:
        items.append(PickerItem(
            id="chatgpt",
            label="ChatGPT (OAuth)",
            description=f"\u2713 {chatgpt_count} models",
            category="oauth",
        ))
    else:
        items.append(PickerItem(
            id="chatgpt",
            label="ChatGPT (OAuth)",
            description="\u2717 Not authenticated",
            category="oauth",
        ))

    # Direct API providers
    try:
        from aura.providers import list_configured_providers
        for p in list_configured_providers():
            icon = "\u2713" if p["configured"] else "\u2717"
            items.append(PickerItem(
                id=p["name"],
                label=p["display_name"],
                description=f"{icon} {p['model_count']} models",
                category="api",
            ))
    except ImportError:
        pass

    return items


def _check_ollama_local(host: str) -> bool:
    """Quick reachability check for local Ollama."""
    try:
        import urllib.request
        req = urllib.request.Request(host, method="HEAD")
        urllib.request.urlopen(req, timeout=2)
        return True
    except Exception:
        return False


def _get_local_ollama_models() -> list[str]:
    """Fetch local Ollama model names (best effort, empty on failure)."""
    try:
        from aura.cli.model_picker import _fetch_all_models
        return _fetch_all_models()
    except Exception:
        return []
