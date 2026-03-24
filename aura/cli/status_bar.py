"""Minimal status bar for AURA CLI — model, cost, context gauge, permission mode."""

from rich.text import Text


def build_status_bar(
    *,
    model: str = "auto",
    token_used: int = 0,
    token_limit: int = 128000,
    permission_mode: str = "careful",
    cost_usd: float = 0.0,
    # Accept and ignore legacy kwargs so callers don't break
    **_ignored,
) -> Text:
    """Build a clean 4-5 item status bar.

    Format:  model  |  $0.023  |  1.2K/128K ████░░░░ 1%  |  CARE
    """
    parts: list[Text] = []

    # -- Model name (cyan if auto, green if manual override) --
    model_short = model.replace(":cloud", "").replace(":latest", "")
    if len(model_short) > 25:
        model_short = model_short[:22] + "..."
    color = "cyan" if model == "auto" else "green"
    t = Text(model_short, style=f"bold {color}")
    parts.append(t)

    # -- Cost (only shown when > 0) --
    if cost_usd > 0:
        parts.append(Text(f"${cost_usd:.3f}", style="dim"))

    # -- Context gauge --
    pct = int(100 * token_used / max(token_limit, 1))
    if token_used >= 1000:
        used_k = f"{token_used / 1000:.1f}K"
    else:
        used_k = str(token_used)
    limit_k = f"{token_limit / 1000:.0f}K"

    if pct < 50:
        gauge_color = "green"
    elif pct < 80:
        gauge_color = "yellow"
    else:
        gauge_color = "red"

    filled = int(8 * pct / 100)
    bar_chars = "\u2588" * filled + "\u2591" * (8 - filled)

    gauge = Text()
    gauge.append(f"{used_k}/{limit_k} ", style="dim")
    gauge.append(bar_chars, style=gauge_color)
    gauge.append(f" {pct}%", style="dim")
    parts.append(gauge)

    # -- Permission mode --
    _mode_colors = {
        "careful": "yellow",
        "auto_edit": "green",
        "full_auto": "red",
        "plan": "blue",
        "plan_approve": "magenta",
    }
    _mode_short = {
        "careful": "CARE",
        "auto_edit": "AUTO",
        "full_auto": "FULL",
        "plan": "PLAN",
        "plan_approve": "P-APR",
    }
    mc = _mode_colors.get(permission_mode, "white")
    ms = _mode_short.get(permission_mode, permission_mode.upper())
    parts.append(Text(ms, style=f"bold {mc}"))

    # -- Assemble --
    sep = Text("  \u2502  ", style="dim")
    result = Text("  ")
    for i, part in enumerate(parts):
        if i > 0:
            result.append_text(sep)
        result.append_text(part)

    return result
