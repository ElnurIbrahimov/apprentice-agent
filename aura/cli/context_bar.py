"""Context window visibility — token counting and budget display."""
from __future__ import annotations

from typing import Dict, List


def estimate_tokens(text: str) -> int:
    """Estimate token count — delegates to the authoritative token_manager."""
    from aura.core.token_manager import estimate_tokens as _est
    return _est(text)

def estimate_messages_tokens(messages: List[Dict]) -> int:
    """Estimate total tokens across a message list — delegates to token_manager."""
    from aura.core.token_manager import estimate_messages_tokens as _est_msgs
    return _est_msgs(messages)

def format_token_count(count: int) -> str:
    """Format token count: 500, 12.4K, 1.2M."""
    if count < 1000:
        return str(count)
    elif count < 1_000_000:
        return f"{count / 1000:.1f}K"
    else:
        return f"{count / 1_000_000:.1f}M"

def get_context_limit(model_name: str) -> int:
    """Get context window limit for a model — delegates to the authoritative token_manager."""
    from aura.core.token_manager import get_context_window
    return get_context_window(model_name)

def _usage_color(pct: float) -> str:
    if pct < 0.50:
        return "green"
    elif pct < 0.80:
        return "yellow"
    else:
        return "red"

def build_context_gauge(used: int, limit: int) -> str:
    """Build compact context gauge: 'Ctx: 12.4K/128K [████░░░░] 10%'"""
    if limit <= 0:
        return f"Ctx: {format_token_count(used)}/0"
    pct = min(used / limit, 1.0)
    pct_int = int(pct * 100)
    color = _usage_color(pct)
    used_str = format_token_count(used)
    limit_str = format_token_count(limit)
    filled = int(pct * 8)
    bar = "█" * filled + "░" * (8 - filled)
    return f"[{color}]{used_str}[/{color}]/{limit_str} [{color}]{bar}[/{color}] {pct_int}%"

def build_context_breakdown(system_tokens: int, history_tokens: int, tools_tokens: int, limit: int, message_count: int = 0) -> str:
    """Build detailed context breakdown for /context command."""
    total = system_tokens + history_tokens + tools_tokens
    lines = [
        f"  System prompt:  {format_token_count(system_tokens):>8}",
        f"  Conversation:   {format_token_count(history_tokens):>8}",
        f"  Tool schemas:   {format_token_count(tools_tokens):>8}",
        "  ─────────────────────",
        f"  Total:          {format_token_count(total):>8} / {format_token_count(limit)}",
    ]
    pct = (total / limit * 100) if limit > 0 else 0
    color = _usage_color(total / limit if limit > 0 else 0)
    lines.append(f"  Usage:          [{color}]{pct:.0f}%[/{color}]")
    if pct > 80:
        lines.append(f"\n  [yellow]⚠ Context is {pct:.0f}% full — consider /compact[/yellow]")

    # Remaining messages projection (C3)
    if message_count > 0 and history_tokens > 0:
        avg_per_msg = history_tokens / message_count
        remaining = max(0, limit - total)
        est_remaining = int(remaining / avg_per_msg) if avg_per_msg > 0 else 0
        if est_remaining > 0:
            lines.append(f"  Remaining:      ~{est_remaining} more messages")
        else:
            lines.append("  [red]Context nearly full — use /compact[/red]")

    return "\n".join(lines)
