"""Interactive model picker for AURA CLI — select model mid-session."""

from rich.console import Console
from rich.text import Text
from rich.panel import Panel

# Model roles with display info
MODEL_ROLES = [
    ("fast", "gemini-3-flash-preview:cloud", "1M ctx"),
    ("reason", "kimi-k2.5:cloud", "256K ctx"),
    ("code", "minimax-m2.5:cloud", "196K ctx"),
    ("think", "kimi-k2-thinking:cloud", "256K ctx"),
    ("vision", "qwen3-vl:235b-cloud", "256K ctx"),
    ("longctx", "gemini-3-flash-preview:cloud", "1M ctx"),
]


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
            lines.append(" ← current", style="green")
        lines.append("\n")

    lines.append("\n")
    lines.append("  [1-6]", style="bold cyan")
    lines.append(" select  ", style="dim")
    lines.append("[a]", style="bold cyan")
    lines.append(" auto  ", style="dim")
    lines.append("[Esc/q]", style="bold cyan")
    lines.append(" cancel", style="dim")

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
        choice = input("  > ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return None

    if not choice or choice in ("q", "esc", "escape"):
        console.print("  [dim]Cancelled.[/dim]")
        return None
    elif choice == "a" or choice == "auto":
        return "auto"
    else:
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(MODEL_ROLES):
                _, model, _ = MODEL_ROLES[idx]
                return model
        except ValueError:
            pass
        for _, model, _ in MODEL_ROLES:
            if choice in model:
                return model

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
