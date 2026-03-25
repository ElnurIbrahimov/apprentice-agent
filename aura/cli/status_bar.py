"""Status bar for AURA CLI — model, cost, context gauge, permissions, and all indicators."""

from rich.text import Text


def build_status_bar(
    *,
    model: str = "auto",
    token_used: int = 0,
    token_limit: int = 128000,
    permission_mode: str = "careful",
    cost_usd: float = 0.0,
    bg_indicator: str = "",
    research_indicator: str = "",
    mood_indicator: str = "",
    watch_indicator: str = "",
    steering_queue: object = None,
    session_title: str = "",
    message_count: int = 0,
    project_type: str = "",
    as_ansi: bool = False,
) -> "Text | list[tuple[str, str]]":
    """Build a clean status bar with all indicators wired up.

    When *as_ansi* is False (default), returns a Rich ``Text`` object.
    When *as_ansi* is True, returns a list of ``(style, text)`` tuples
    suitable for prompt_toolkit's ``FormattedText``.
    """

    # -- Model name --
    model_short = model.replace(":cloud", "").replace(":latest", "")
    if len(model_short) > 25:
        model_short = model_short[:22] + "\u2026"
    model_color = "ansibrightcyan" if model == "auto" else "ansibrightgreen"

    # -- Cost --
    cost_str = f"${cost_usd:.3f}" if cost_usd > 0.001 else ""

    # -- Context gauge --
    pct = int(100 * token_used / max(token_limit, 1))
    if token_used >= 1000:
        used_k = f"{token_used / 1000:.1f}K"
    else:
        used_k = str(token_used)
    limit_k = f"{token_limit / 1000:.0f}K"

    if pct < 50:
        gauge_color = "ansibrightgreen"
    elif pct < 80:
        gauge_color = "ansibrightyellow"
    else:
        gauge_color = "ansibrightred"

    filled = int(8 * pct / 100)
    bar_chars = "\u2588" * filled + "\u2591" * (8 - filled)

    # -- Permission mode --
    _mode_map = {
        "careful":      ("CARE",  "ansibrightyellow"),
        "auto_edit":    ("AUTO",  "ansibrightgreen"),
        "full_auto":    ("FULL",  "ansibrightred"),
        "plan":         ("PLAN",  "ansibrightblue"),
        "plan_approve": ("P-APR", "ansibrightmagenta"),
    }
    ms, mc = _mode_map.get(permission_mode, (permission_mode.upper()[:5], "white"))

    # -- Session title (truncated) --
    title_str = ""
    if session_title:
        title_str = session_title if len(session_title) <= 20 else session_title[:17] + "\u2026"

    # -- Message count --
    msg_str = f"#{message_count}" if message_count > 0 else ""

    # -- Steering queue indicator --
    steer_str = ""
    if steering_queue:
        try:
            qsize = steering_queue.qsize() if hasattr(steering_queue, "qsize") else 0
            if qsize > 0:
                steer_str = f"\u21e8{qsize}"
        except Exception:
            pass

    # ===================================================================
    # prompt_toolkit FormattedText path (for bottom toolbar)
    # ===================================================================
    if as_ansi:
        SEP = " \u2502 "
        parts: list[tuple[str, str]] = []
        parts.append(("", " "))

        # Model
        parts.append((f"bold {model_color}", model_short))

        # Background tasks
        if bg_indicator:
            parts.append(("", f" {bg_indicator}"))

        # Cost
        if cost_str:
            parts.append(("", SEP))
            parts.append(("", cost_str))

        # Context gauge
        parts.append(("", SEP))
        parts.append(("", f"{used_k}/{limit_k} "))
        parts.append((gauge_color, bar_chars))
        parts.append(("", f" {pct}%"))

        # Permission mode
        parts.append(("", SEP))
        parts.append((f"bold {mc}", ms))

        # Mood
        if mood_indicator:
            parts.append(("", f" {mood_indicator}"))

        # Research mode
        if research_indicator:
            parts.append(("", SEP))
            parts.append(("ansibrightcyan", research_indicator))

        # Watch mode
        if watch_indicator:
            parts.append(("", SEP))
            parts.append(("ansibrightyellow", watch_indicator))

        # Steering queue
        if steer_str:
            parts.append(("", SEP))
            parts.append(("ansibrightmagenta", steer_str))

        # Session title
        if title_str:
            parts.append(("", SEP))
            parts.append(("italic", title_str))

        # Message count
        if msg_str:
            parts.append(("", SEP))
            parts.append(("", msg_str))

        return parts

    # ===================================================================
    # Rich Text path (for console.print)
    # ===================================================================
    rich_parts: list[Text] = []

    # Model
    t = Text(model_short, style=f"bold {model_color}")
    if bg_indicator:
        t.append(f" {bg_indicator}", style="dim")
    rich_parts.append(t)

    # Cost
    if cost_str:
        rich_parts.append(Text(cost_str, style="dim"))

    # Context gauge
    gauge = Text()
    gauge.append(f"{used_k}/{limit_k} ", style="dim")
    gauge.append(bar_chars, style=gauge_color)
    gauge.append(f" {pct}%", style="dim")
    rich_parts.append(gauge)

    # Permission mode + mood
    mode_t = Text(ms, style=f"bold {mc}")
    if mood_indicator:
        mode_t.append(f" {mood_indicator}", style="dim")
    rich_parts.append(mode_t)

    # Research indicator
    if research_indicator:
        rich_parts.append(Text(research_indicator, style="cyan"))

    # Watch indicator
    if watch_indicator:
        rich_parts.append(Text(watch_indicator, style="yellow"))

    # Steering queue
    if steer_str:
        rich_parts.append(Text(steer_str, style="magenta"))

    # Session title
    if title_str:
        rich_parts.append(Text(title_str, style="italic dim"))

    # Message count
    if msg_str:
        rich_parts.append(Text(msg_str, style="dim"))

    # -- Assemble with separators --
    sep = Text("  \u2502  ", style="dim")
    result = Text("  ")
    for i, part in enumerate(rich_parts):
        if i > 0:
            result.append_text(sep)
        result.append_text(part)

    return result
