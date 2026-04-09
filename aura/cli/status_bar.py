"""Status bar for AURA CLI — model, cost, context gauge, permissions, and all indicators.

Responsive: segments are progressively hidden based on terminal width.
Priority tiers (always shown first → dropped first):
  P0 (always): model name, permission mode
  P1 (>= 100): context gauge, cost
  P2 (>= 120): session title, message count
  P3 (conditional): bg/research/watch/steering indicators — shown if content AND width permits
"""

import os

from rich.text import Text

_SPARK_CHARS = "\u2581\u2582\u2583\u2584\u2585\u2586\u2587\u2588"  # ▁▂▃▄▅▆▇█


def _build_sparkline(values: list[float], width: int = 8) -> str:
    """Render values as a Unicode sparkline."""
    if not values:
        return ""
    recent = values[-width:]
    lo, hi = min(recent), max(recent)
    span = hi - lo if hi > lo else 1.0
    chars = []
    for v in recent:
        idx = int((v - lo) / span * (len(_SPARK_CHARS) - 1))
        idx = max(0, min(len(_SPARK_CHARS) - 1, idx))
        chars.append(_SPARK_CHARS[idx])
    return "".join(chars)


def _terminal_width() -> int:
    """Return current terminal width, defaulting to 80."""
    try:
        return os.get_terminal_size().columns
    except (ValueError, OSError):
        return 80


def build_status_bar(
    *,
    model: str = "auto",
    token_used: int = 0,
    token_limit: int = 128000,
    permission_mode: str = "careful",
    cost_usd: float = 0.0,
    last_turn_cost: float = 0.0,
    bg_indicator: str = "",
    research_indicator: str = "",
    mood_indicator: str = "",       # accepted but ignored (removed from output)
    watch_indicator: str = "",
    steering_queue: object = None,
    session_title: str = "",
    message_count: int = 0,
    project_type: str = "",
    cost_history: "list[float] | None" = None,
    as_ansi: bool = False,
    _term_width: int | None = None,  # for testing
) -> "Text | list[tuple[str, str]]":
    """Build a responsive status bar with all indicators wired up.

    Segments are progressively dropped when they don't fit the terminal.

    When *as_ansi* is False (default), returns a Rich ``Text`` object.
    When *as_ansi* is True, returns a list of ``(style, text)`` tuples
    suitable for prompt_toolkit's ``FormattedText``.
    """
    width = _term_width if _term_width is not None else _terminal_width()

    # -- Model name --
    model_short = model.replace(":cloud", "").replace(":latest", "")
    if len(model_short) > 25:
        model_short = model_short[:22] + "\u2026"
    model_color = "ansibrightcyan" if model == "auto" else "ansibrightgreen"

    # -- Cost --
    cost_str = f"${cost_usd:.3f}" if cost_usd > 0.001 else ""

    # -- Turn cost (C2) --
    turn_cost_str = f"+${last_turn_cost:.4f}" if last_turn_cost > 0.0001 else ""

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

    # =================================================================
    # Build a list of (priority, segment_id, plain_width) entries.
    # Lower priority number = shown first / dropped last.
    # We build segments eagerly, then filter by what fits.
    # =================================================================
    SEP_WIDTH = 5  # "  │  " for Rich or " │ " for ansi — use max

    # Candidate segments: (priority, id, plain_text_width)
    # Priority 0 = always shown, 1 = >= 100 cols, 2 = >= 120, 3 = conditional indicators
    candidates: list[tuple[int, str, int]] = []

    # P0: model (always)
    model_w = len(model_short)
    if bg_indicator:
        model_w += 1 + len(bg_indicator)
    candidates.append((0, "model", model_w))

    # P0: permission mode (always)
    candidates.append((0, "mode", len(ms)))

    # P1: context gauge (>= 100)
    gauge_plain = f"{used_k}/{limit_k} {bar_chars} {pct}%"
    candidates.append((1, "gauge", len(gauge_plain)))

    # P1: cost (>= 100, only if nonzero)
    if cost_str:
        candidates.append((1, "cost", len(cost_str)))

    # P1: turn cost (>= 100, only if nonzero)
    if turn_cost_str:
        candidates.append((1, "turn_cost", len(turn_cost_str)))

    # P1: cost sparkline (>= 100, only if enough history)
    if cost_history and len(cost_history) >= 2:
        _sparkline = _build_sparkline(cost_history)
        candidates.append((1, "sparkline", len(_sparkline)))

    # P3: indicators — shown if they have content AND width permits
    if research_indicator:
        candidates.append((3, "research", len(research_indicator)))
    if watch_indicator:
        candidates.append((3, "watch", len(watch_indicator)))
    if steer_str:
        candidates.append((3, "steer", len(steer_str)))

    # P2: session title (>= 120)
    if title_str:
        candidates.append((2, "title", len(title_str)))

    # P2: message count (>= 120)
    if msg_str:
        candidates.append((2, "msg", len(msg_str)))

    # Sort by priority (stable — preserves insertion order within tier)
    candidates.sort(key=lambda c: c[0])

    # Progressively include segments that fit
    included: set[str] = set()
    used_width = 2  # leading padding
    for priority, seg_id, seg_w in candidates:
        # Apply minimum-width gates
        if priority >= 1 and width < 100:
            continue
        if priority >= 2 and width < 120:
            continue
        # Cost of adding: separator (if not first) + segment width
        extra = seg_w + (SEP_WIDTH if included else 0)
        if used_width + extra <= width:
            included.add(seg_id)
            used_width += extra

    # ===================================================================
    # prompt_toolkit FormattedText path (for bottom toolbar)
    # ===================================================================
    if as_ansi:
        SEP = " \u2502 "
        parts: list[tuple[str, str]] = []
        parts.append(("", " "))

        def _ansi_sep() -> None:
            parts.append(("", SEP))

        # Model (P0 — always)
        parts.append((f"bold {model_color}", model_short))
        if bg_indicator and "model" in included:
            parts.append(("", f" {bg_indicator}"))

        # Cost (P1)
        if "cost" in included:
            _ansi_sep()
            parts.append(("", cost_str))

        # Turn cost (P1)
        if "turn_cost" in included:
            parts.append(("ansibrightyellow", f" {turn_cost_str}"))

        # Cost sparkline (P1)
        if "sparkline" in included and cost_history:
            _spark = _build_sparkline(cost_history)
            _ansi_sep()
            parts.append(("ansibrightyellow", _spark))

        # Context gauge (P1)
        if "gauge" in included:
            _ansi_sep()
            parts.append(("", f"{used_k}/{limit_k} "))
            parts.append((gauge_color, bar_chars))
            parts.append(("", f" {pct}%"))

        # Permission mode (P0 — always)
        _ansi_sep()
        parts.append((f"bold {mc}", ms))

        # Research mode (P3)
        if "research" in included:
            _ansi_sep()
            parts.append(("ansibrightcyan", research_indicator))

        # Watch mode (P3)
        if "watch" in included:
            _ansi_sep()
            parts.append(("ansibrightyellow", watch_indicator))

        # Steering queue (P3)
        if "steer" in included:
            _ansi_sep()
            parts.append(("ansibrightmagenta", steer_str))

        # Session title (P2)
        if "title" in included:
            _ansi_sep()
            parts.append(("italic", title_str))

        # Message count (P2)
        if "msg" in included:
            _ansi_sep()
            parts.append(("", msg_str))

        return parts

    # ===================================================================
    # Rich Text path (for console.print)
    # ===================================================================
    rich_parts: list[Text] = []

    # Model (P0 — always)
    t = Text(model_short, style=f"bold {model_color}")
    if bg_indicator and "model" in included:
        t.append(f" {bg_indicator}", style="dim")
    rich_parts.append(t)

    # Cost (P1)
    if "cost" in included:
        cost_text = Text(cost_str, style="dim")
        # Turn cost appended inline with session cost (P1)
        if "turn_cost" in included:
            cost_text.append(f" {turn_cost_str}", style="yellow")
        rich_parts.append(cost_text)

    # Cost sparkline (P1)
    if "sparkline" in included and cost_history:
        _spark = _build_sparkline(cost_history)
        rich_parts.append(Text(_spark, style="yellow"))

    # Context gauge (P1)
    if "gauge" in included:
        gauge = Text()
        gauge.append(f"{used_k}/{limit_k} ", style="dim")
        gauge.append(bar_chars, style=gauge_color)
        gauge.append(f" {pct}%", style="dim")
        rich_parts.append(gauge)

    # Permission mode (P0 — always)
    rich_parts.append(Text(ms, style=f"bold {mc}"))

    # Research indicator (P3)
    if "research" in included:
        rich_parts.append(Text(research_indicator, style="cyan"))

    # Watch indicator (P3)
    if "watch" in included:
        rich_parts.append(Text(watch_indicator, style="yellow"))

    # Steering queue (P3)
    if "steer" in included:
        rich_parts.append(Text(steer_str, style="magenta"))

    # Session title (P2)
    if "title" in included:
        rich_parts.append(Text(title_str, style="italic dim"))

    # Message count (P2)
    if "msg" in included:
        rich_parts.append(Text(msg_str, style="dim"))

    # -- Assemble with separators --
    sep = Text("  \u2502  ", style="dim")
    result = Text("  ")
    for i, part in enumerate(rich_parts):
        if i > 0:
            result.append_text(sep)
        result.append_text(part)

    return result
