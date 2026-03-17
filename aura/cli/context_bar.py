"""Context window visibility — token counting and budget display."""
from __future__ import annotations
import json
import math
from typing import List, Dict, Optional

_CHARS_PER_TOKEN = 3.8

def estimate_tokens(text: str) -> int:
    """Estimate token count from text using char-based heuristic."""
    if not text:
        return 0
    return max(1, int(len(text) / _CHARS_PER_TOKEN))

def estimate_messages_tokens(messages: List[Dict]) -> int:
    """Estimate total tokens across a message list."""
    total = 0
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            total += estimate_tokens(content)
        total += 4  # overhead per message
        tool_calls = msg.get("tool_calls", [])
        if tool_calls:
            total += estimate_tokens(json.dumps(tool_calls))
    return total

def format_token_count(count: int) -> str:
    """Format token count: 500, 12.4K, 1.2M."""
    if count < 1000:
        return str(count)
    elif count < 1_000_000:
        return f"{count / 1000:.1f}K"
    else:
        return f"{count / 1_000_000:.1f}M"

MODEL_CONTEXT_LIMITS = {
    "default": 128_000,
    "qwen": 128_000,
    "deepseek": 128_000,
    "gemma": 128_000,
    "llama": 128_000,
    "minimax": 1_000_000,
    "chatgpt": 128_000,
    "gpt-5": 1_000_000,
}

def get_context_limit(model_name: str) -> int:
    """Get context window limit for a model."""
    model_lower = (model_name or "").lower()
    for prefix, limit in MODEL_CONTEXT_LIMITS.items():
        if prefix in model_lower:
            return limit
    return MODEL_CONTEXT_LIMITS["default"]

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

def build_context_breakdown(system_tokens: int, history_tokens: int, tools_tokens: int, limit: int) -> str:
    """Build detailed context breakdown for /context command."""
    total = system_tokens + history_tokens + tools_tokens
    lines = [
        f"  System prompt:  {format_token_count(system_tokens):>8}",
        f"  Conversation:   {format_token_count(history_tokens):>8}",
        f"  Tool schemas:   {format_token_count(tools_tokens):>8}",
        f"  ─────────────────────",
        f"  Total:          {format_token_count(total):>8} / {format_token_count(limit)}",
    ]
    pct = (total / limit * 100) if limit > 0 else 0
    color = _usage_color(total / limit if limit > 0 else 0)
    lines.append(f"  Usage:          [{color}]{pct:.0f}%[/{color}]")
    if pct > 80:
        lines.append(f"\n  [yellow]⚠ Context is {pct:.0f}% full — consider /compact[/yellow]")
    return "\n".join(lines)
