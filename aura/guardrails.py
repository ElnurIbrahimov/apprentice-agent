"""Tool loop guardrails — track failures and warn/stop based on thresholds.

Mirrors Hermes Agent's tool_loop_guardrails config:
  tool_loop_guardrails:
    warn_after:
      exact_failure: 2
      same_tool_failure: 3
      idempotent_no_progress: 2
    hard_stop_after:
      exact_failure: 5
      same_tool_failure: 8
      idempotent_no_progress: 5

Tracks per-tool-call failures during the agentic loop. When thresholds
are hit, emits warnings or stops the loop.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class GuardrailState:
    """Tracks tool call patterns for guardrail decisions."""
    # Last N tool calls (for "same_tool_failure" detection)
    recent_calls: list[dict] = field(default_factory=list)
    # Last successful tool call (for "idempotent_no_progress" detection)
    last_success_signature: Optional[str] = None
    # Consecutive idempotent calls with no progress
    idempotent_streak: int = 0
    # Per-tool failure counts
    tool_failures: dict[str, int] = field(default_factory=dict)
    # Last warn/stop time (to avoid spamming)
    last_warn_at: float = 0.0
    # Whether hard stop has been triggered
    hard_stopped: bool = False


# Module-level singleton (per-process)
_state: Optional[GuardrailState] = None


def get_state() -> GuardrailState:
    """Get or create the guardrail state singleton."""
    global _state
    if _state is None:
        _state = GuardrailState()
    return _state


def reset_state() -> None:
    """Reset the guardrail state (call at start of new agent run)."""
    global _state
    _state = GuardrailState()


def get_thresholds() -> dict:
    """Get the warn and hard_stop thresholds."""
    try:
        from aura.security_config import get_guardrail_thresholds
        return get_guardrail_thresholds()
    except ImportError:
        return {
            "warn": {"exact_failure": 2, "same_tool_failure": 3, "idempotent_no_progress": 2},
            "hard_stop": {"exact_failure": 5, "same_tool_failure": 8, "idempotent_no_progress": 5},
        }


def record_tool_call(tool_name: str, args: dict, success: bool) -> dict:
    """Record a tool call result and check guardrails.

    Returns:
        dict with 'action' (one of 'continue', 'warn', 'stop') and optional 'reason'.
    """
    state = get_state()
    thresholds = get_thresholds()
    signature = f"{tool_name}:{hash(frozenset(args.items()) if args else frozenset())}"

    # Track in recent calls
    state.recent_calls.append({
        "tool": tool_name,
        "signature": signature,
        "success": success,
        "time": time.time(),
    })
    if len(state.recent_calls) > 20:
        state.recent_calls = state.recent_calls[-20:]

    if success:
        state.tool_failures[tool_name] = 0
        if signature != state.last_success_signature:
            state.idempotent_streak = 0
        state.last_success_signature = signature
        return {"action": "continue"}

    # Failure path
    state.tool_failures[tool_name] = state.tool_failures.get(tool_name, 0) + 1
    fail_count = state.tool_failures[tool_name]

    # Check idempotent (same tool + same args consecutively)
    if state.recent_calls and state.recent_calls[-2]["signature"] == signature:
        state.idempotent_streak += 1
    else:
        state.idempotent_streak = 0

    # Check thresholds
    warn_thresholds = thresholds["warn"]
    hard_thresholds = thresholds["hard_stop"]

    # Hard stop: same tool failing too many times
    if fail_count >= hard_thresholds["same_tool_failure"]:
        state.hard_stopped = True
        return {"action": "stop", "reason": f"Tool '{tool_name}' failed {fail_count} times"}

    # Hard stop: idempotent no progress
    if state.idempotent_streak >= hard_thresholds["idempotent_no_progress"]:
        state.hard_stopped = True
        return {"action": "stop", "reason": f"Tool '{tool_name}' made {state.idempotent_streak} identical no-progress calls"}

    # Hard stop: total exact failure count
    total_failures = sum(state.tool_failures.values())
    if total_failures >= hard_thresholds["exact_failure"]:
        state.hard_stopped = True
        return {"action": "stop", "reason": f"Total tool failures reached {total_failures}"}

    # Warning: same tool failing
    if fail_count >= warn_thresholds["same_tool_failure"]:
        return {"action": "warn", "reason": f"Tool '{tool_name}' failed {fail_count} times"}

    # Warning: idempotent
    if state.idempotent_streak >= warn_thresholds["idempotent_no_progress"]:
        return {"action": "warn", "reason": f"Tool '{tool_name}' repeated {state.idempotent_streak} times without progress"}

    return {"action": "continue"}


def is_hard_stopped() -> bool:
    """Check if hard stop has been triggered."""
    return get_state().hard_stopped
