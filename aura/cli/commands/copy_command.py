"""Copy command -- copy last response or specific code blocks to clipboard."""
from __future__ import annotations

from typing import Optional
from .common import command, TIER_BETA, TIER_EXPERIMENTAL, TIER_STABLE


@command("/copy",     "Copy last response or code block to clipboard",    tier=TIER_STABLE)


def handle_copy(agent, arg, context) -> Optional[str]:
    """Copy last response or a specific code block to clipboard.

    Usage:
        /copy          -- copy full last response
        /copy code     -- copy first code block
        /copy code N   -- copy Nth code block
    """
    from ..context import get_ctx
    from ..display import show_error, show_info

    ctx = get_ctx()
    if not ctx or not ctx.agentic_loop:
        show_error("Copy not available outside chat mode.")
        return None

    loop = ctx.agentic_loop

    # Find last assistant response
    last_response = ""
    for msg in reversed(getattr(loop, '_conversation_history', [])):
        if msg.get("role") == "assistant" and msg.get("content"):
            last_response = msg["content"]
            break

    if not last_response:
        show_error("No response to copy.")
        return None

    arg = (arg or "").strip()

    if arg.startswith("code"):
        # Extract code blocks
        import re
        blocks = re.findall(r'```(?:\w*)\n(.*?)```', last_response, re.DOTALL)
        if not blocks:
            show_error("No code blocks found in last response.")
            return None

        parts = arg.split()
        idx = 0
        if len(parts) > 1 and parts[1].isdigit():
            idx = int(parts[1]) - 1

        if idx < 0 or idx >= len(blocks):
            show_error(f"Code block {idx + 1} not found ({len(blocks)} blocks available).")
            return None

        text = blocks[idx].strip()
    else:
        text = last_response

    try:
        import pyperclip
        pyperclip.copy(text)
        show_info(f"Copied {len(text)} chars to clipboard.")
    except ImportError:
        show_error("Clipboard requires pyperclip: pip install pyperclip")
    except Exception as e:
        show_error(f"Clipboard failed: {e}")

    return None
