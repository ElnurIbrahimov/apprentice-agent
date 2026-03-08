"""Interactive model picker for AURA CLI — select model mid-session."""

import os
from rich.console import Console
from rich.text import Text
from rich.panel import Panel

# Model roles with display info (updated from Config on startup)
MODEL_ROLES = [
    ("fast", "gemini-3-flash-preview:cloud", "1M ctx"),
    ("reason", "kimi-k2.5:cloud", "256K ctx"),
    ("code", "minimax-m2.5:cloud", "196K ctx"),
    ("think", "kimi-k2-thinking:cloud", "256K ctx"),
    ("vision", "qwen3-vl:235b-cloud", "256K ctx"),
    ("longctx", "gemini-3-flash-preview:cloud", "1M ctx"),
]

# Cache for all available models
_all_models_cache = []


def _fetch_all_models() -> list:
    """Fetch all available models from Ollama."""
    global _all_models_cache
    if _all_models_cache:
        return _all_models_cache
    try:
        import requests
        host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        resp = requests.get(f"{host}/api/tags", timeout=5)
        if resp.status_code == 200:
            models = resp.json().get("models", [])
            _all_models_cache = [m["name"] for m in models]
    except Exception:
        pass
    return _all_models_cache


def pick_model(console: Console, current_model: str = "auto") -> "str | None":
    """Show interactive model picker. Returns model name, 'auto', or None."""
    lines = Text()
    lines.append("\n")

    for i, (role, model, ctx) in enumerate(MODEL_ROLES, 1):
        model_short = model.replace(":cloud", "")
        is_current = model == current_model

        num_style = "bold cyan" if not is_current else "bold green"
        model_style = "white" if not is_current else "bold green"

        lines.append(f"    {i}", style=num_style)
        lines.append(f". {model_short:<35s}", style=model_style)
        lines.append(f" {role:<8s}", style="dim yellow")
        lines.append(f" {ctx}", style="dim")
        if is_current:
            lines.append(" <-", style="green")
        lines.append("\n")

    lines.append("\n")
    lines.append("  [1-6]", style="bold cyan")
    lines.append(" select  ", style="dim")
    lines.append("[a]", style="bold cyan")
    lines.append(" auto  ", style="dim")
    lines.append("[l]", style="bold cyan")
    lines.append(" list all  ", style="dim")
    lines.append("[Esc/q]", style="bold cyan")
    lines.append(" cancel", style="dim")
    lines.append("\n  ", style="dim")
    lines.append("Or type any model name directly", style="dim italic")

    header = Text()
    header.append("  Select model", style="bold white")
    cur = current_model.replace(":cloud", "").replace(":latest", "")
    header.append(f"  (current: {cur})", style="dim")

    panel = Panel(
        lines,
        title="[bold cyan]Model Picker[/bold cyan]",
        subtitle=header,
        border_style="cyan",
        padding=(0, 1),
    )
    console.print(panel)

    try:
        choice = input("  > ").strip()
    except (EOFError, KeyboardInterrupt):
        return None

    choice_lower = choice.lower()

    if not choice or choice_lower in ("q", "esc", "escape"):
        console.print("  [dim]Cancelled.[/dim]")
        return None
    elif choice_lower in ("a", "auto"):
        return "auto"
    elif choice_lower == "l":
        # Show all available models
        all_models = _fetch_all_models()
        if not all_models:
            console.print("  [dim]Could not fetch models from Ollama.[/dim]")
            return None

        console.print(f"\n  [bold]All available models ({len(all_models)}):[/bold]\n")
        for j, m in enumerate(all_models, 1):
            marker = " [green]<-[/green]" if m == current_model else ""
            is_cloud = ":cloud" in m or "-cloud" in m
            tag = " [dim cyan](cloud)[/dim cyan]" if is_cloud else " [dim](local)[/dim]"
            console.print(f"    [bold cyan]{j:>2}[/bold cyan]. {m}{tag}{marker}")

        console.print()
        try:
            pick = input("  Pick # or name > ").strip()
        except (EOFError, KeyboardInterrupt):
            return None

        if not pick:
            return None

        # Try number selection
        try:
            idx = int(pick) - 1
            if 0 <= idx < len(all_models):
                return all_models[idx]
        except ValueError:
            pass

        # Try name match (exact or partial)
        for m in all_models:
            if pick == m or pick.lower() == m.lower():
                return m
        for m in all_models:
            if pick.lower() in m.lower():
                return m

        # Accept as-is (user might know a model not in the list)
        console.print(f"  [dim]Using: {pick}[/dim]")
        return pick
    else:
        # Try role number (1-6)
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(MODEL_ROLES):
                _, model, _ = MODEL_ROLES[idx]
                return model
        except ValueError:
            pass

        # Try partial match against roles
        for _, model, _ in MODEL_ROLES:
            if choice_lower in model.lower():
                return model

        # Try match against all available models
        all_models = _fetch_all_models()
        for m in all_models:
            if choice == m or choice_lower == m.lower():
                return m
        for m in all_models:
            if choice_lower in m.lower():
                return m

        # Accept any typed model name directly
        console.print(f"  [dim]Using: {choice}[/dim]")
        return choice

    console.print(f"  [dim]Invalid choice: {choice}[/dim]")
    return None


def update_model_roles_from_config():
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
