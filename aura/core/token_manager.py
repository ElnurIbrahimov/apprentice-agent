"""Context window management for the agentic loop.

Tracks token usage and auto-compacts messages when approaching
the model's context limit, preventing truncation and crashes.
"""

import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)

MODEL_CONTEXT_WINDOWS = {
    # Utility models (local)
    "nomic-embed-text:latest": 8192,
    "glm-ocr:latest": 8192,
    # Local models
    "deepseek-r1:8b": 131072,
    "qwen3:8b": 131072,
    "qwen2.5-coder:7b": 131072,
    "deepseek-coder:6.7b": 16384,
    "gemma3:4b": 131072,
    "deepcoder:1.5b": 131072,
    "qwen2:1.5b": 32768,
    "llava:latest": 4096,
    # Cloud models
    "minimax-m2.7:cloud": 1048576,
    "minimax-m2.5:cloud": 196608,
    "kimi-k2.5:cloud": 131072,
    "qwen3.5:397b-cloud": 131072,
    "qwen3.5:cloud": 131072,
    "deepseek-v3.2:cloud": 131072,
    "qwen3-coder:480b-cloud": 131072,
    "qwen3-coder-next:cloud": 131072,
    "gpt-oss:120b-cloud": 131072,
    "glm-5:cloud": 131072,
    "nemotron-3-super:cloud": 131072,
    "cogito-2.1:671b": 131072,
    "devstral-2:123b": 131072,
    "kimi-k2-thinking": 131072,
}
DEFAULT_CONTEXT_WINDOW = 128000


def estimate_tokens(text: str) -> int:
    """Rough token estimate: ~3.5 chars per token for English, ~2.5 for code, ~1.2 for CJK."""
    if not text:
        return 0
    # Detect CJK-heavy text (Chinese, Japanese, Korean)
    cjk_count = sum(1 for c in text if '\u4e00' <= c <= '\u9fff' or '\u3040' <= c <= '\u30ff' or '\uac00' <= c <= '\ud7af')
    if cjk_count / len(text) > 0.3:
        chars_per_token = 1.2  # CJK characters are roughly 1 token each
    else:
        # Heuristic: code-heavy content has more symbols
        code_indicators = text.count("{") + text.count("}") + text.count("(") + text.count(")")
        chars_per_token = 2.8 if code_indicators > len(text) * 0.02 else 3.5
    return max(1, int(len(text) / chars_per_token))


def estimate_messages_tokens(messages: list[dict]) -> int:
    """Sum token estimates across all messages including serialized tool_calls."""
    total = 0
    for msg in messages:
        content = msg.get("content", "") or ""
        total += estimate_tokens(content)
        # Account for tool_calls serialization
        tool_calls = msg.get("tool_calls")
        if tool_calls:
            total += estimate_tokens(json.dumps(tool_calls, default=str))
        # Per-message overhead (role, formatting)
        total += 4
    return total


def get_context_window(model: str) -> int:
    """Lookup context window size. Falls back to DEFAULT_CONTEXT_WINDOW."""
    if model in MODEL_CONTEXT_WINDOWS:
        return MODEL_CONTEXT_WINDOWS[model]
    # Try prefix matching for versioned models
    for known_model, window in MODEL_CONTEXT_WINDOWS.items():
        if model.startswith(known_model.split(":")[0]):
            return window
    return DEFAULT_CONTEXT_WINDOW


class ContextWindowManager:
    """Track and manage context window usage during agentic loop."""

    def __init__(self, model: str, reserve_output: int = 4096):
        self.model = model
        self.max_tokens = get_context_window(model)
        self.reserve_output = reserve_output
        self.budget = self.max_tokens - reserve_output
        self._compaction_count = 0

    def _get_thresholds(self) -> tuple[float, float]:
        """Adaptive compaction thresholds based on context window size (C4)."""
        if self.max_tokens <= 8192:
            return (0.60, 0.75)
        elif self.max_tokens <= 32768:
            return (0.70, 0.85)
        else:
            return (0.80, 0.90)

    def check_and_compact(self, messages: list[dict], brain=None) -> list[dict]:
        """Called each iteration. Returns (possibly compacted) messages.

        Strategy:
        1. Estimate current token usage
        2. If below low threshold: return unchanged
        3. If between low and high: truncate large tool results
        4. If above high threshold: summarize oldest 2/3 of messages
        """
        low_thresh, high_thresh = self._get_thresholds()
        used = estimate_messages_tokens(messages)
        pct = used / self.budget if self.budget > 0 else 0

        if pct < low_thresh:
            return messages

        if pct < high_thresh:
            logger.info(f"[ContextMgr] {pct:.0%} used — truncating large tool results")
            return self._truncate_tool_results(messages)

        # Above high threshold — aggressive compaction
        logger.info(f"[ContextMgr] {pct:.0%} used — summarizing old messages")
        compacted = self._summarize_old(messages, brain)
        self._compaction_count += 1
        return compacted

    def _truncate_tool_results(self, messages: list[dict]) -> list[dict]:
        """Shorten tool results (role=tool) over 2000 chars."""
        result = []
        for msg in messages:
            if msg.get("role") == "tool":
                content = msg.get("content", "")
                if len(content) > 2000:
                    # Keep first 800 + last 400 chars
                    truncated = content[:800] + "\n\n... (truncated) ...\n\n" + content[-400:]
                    result.append({**msg, "content": truncated})
                    continue
            result.append(msg)
        return result

    def _summarize_old(self, messages: list[dict], brain=None) -> list[dict]:
        """Keep system + last 1/3 of messages. Compress the rest into a summary note."""
        if len(messages) < 6:
            return messages

        # Keep system message(s) at the start
        system_msgs = []
        non_system = []
        for msg in messages:
            if msg.get("role") == "system":
                system_msgs.append(msg)
            else:
                non_system.append(msg)

        if len(non_system) < 4:
            return messages

        # Keep last 1/3 of non-system messages
        keep_count = max(4, len(non_system) // 3)
        old_msgs = non_system[:-keep_count]
        recent_msgs = non_system[-keep_count:]

        # Use the shared action-summary extractor from agentic_loop
        try:
            from .agentic_loop import _extract_action_summary
            summary_lines = []
            for msg in old_msgs:
                line = _extract_action_summary(msg)
                if line:
                    summary_lines.append(line)
            summary_text = "\n".join(summary_lines) if summary_lines else "(earlier conversation)"
        except ImportError:
            # Fallback: basic role + truncated content
            summary_parts = []
            for msg in old_msgs:
                role = msg.get("role", "?")
                content = msg.get("content", "") or ""
                if role == "user":
                    first_line = content.split("\n", 1)[0]
                    summary_parts.append(f"- User: {first_line[:120]}")
                elif role == "assistant":
                    tc = msg.get("tool_calls")
                    if tc:
                        names = []
                        for t in tc:
                            if isinstance(t, dict):
                                names.append(t.get("function", {}).get("name", "?"))
                            else:
                                func = getattr(t, "function", None)
                                names.append(getattr(func, "name", "?") if func else "?")
                        summary_parts.append(f"- Agent used tools: {', '.join(names)}")
                    elif content:
                        first_line = content.split("\n", 1)[0]
                        summary_parts.append(f"- Agent responded: {first_line[:100]}")
            summary_text = "\n".join(summary_parts) if summary_parts else "(earlier conversation)"

        n_compressed = len(old_msgs)
        compaction_msg = {
            "role": "system",
            "content": (
                f"[Previous conversation summary]\n"
                f"{summary_text}\n"
                f"[End summary — {n_compressed} messages compressed]"
            ),
        }

        return system_msgs + [compaction_msg] + recent_msgs

    def usage_report(self, messages: list[dict]) -> dict:
        """Return usage stats."""
        used = estimate_messages_tokens(messages)
        return {
            "used_tokens": used,
            "budget": self.budget,
            "max_tokens": self.max_tokens,
            "pct_used": round(used / self.budget * 100, 1) if self.budget > 0 else 0,
            "model": self.model,
            "compactions": self._compaction_count,
        }


class TokenAccumulator:
    """Track token counts incrementally to avoid re-scanning full history (C5)."""

    def __init__(self):
        self._total: int = 0
        self._counted: int = 0

    def update(self, messages: list[dict]) -> int:
        """Return total tokens. Only counts new messages since last call."""
        if len(messages) < self._counted:
            # History was compacted — recount everything
            self._total = estimate_messages_tokens(messages)
            self._counted = len(messages)
        elif len(messages) > self._counted:
            new_msgs = messages[self._counted:]
            self._total += estimate_messages_tokens(new_msgs)
            self._counted = len(messages)
        return self._total

    def reset(self):
        self._total = 0
        self._counted = 0


def estimate_messages_remaining(current_tokens: int, limit: int, avg_per_msg: float) -> int:
    """Estimate how many more messages fit in the context window."""
    if avg_per_msg <= 0:
        return 0
    remaining = max(0, limit - current_tokens)
    return int(remaining / avg_per_msg)
