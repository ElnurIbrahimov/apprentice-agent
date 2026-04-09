# aura/cli/steering.py
"""Mid-turn steering — queue user messages while agent is running."""
from __future__ import annotations

import threading
from collections import deque
from typing import List, Optional


class SteeringQueue:
    """Thread-safe queue for mid-turn user messages.

    While the agentic loop is running, the user can type messages that get
    queued and injected into the next iteration as user context.
    """

    def __init__(self, max_queued: int = 5):
        self._queue: deque = deque(maxlen=max_queued)
        self._lock = threading.Lock()
        self._follow_up: Optional[str] = None

    def push(self, message: str) -> None:
        """Queue a message to inject on the next agentic iteration."""
        message = message[:500]  # length cap to limit injection size
        with self._lock:
            self._queue.append(message)

    def push_follow_up(self, message: str) -> None:
        """Queue a follow-up prompt for after the current turn completes."""
        with self._lock:
            self._follow_up = message

    def pop_all(self) -> List[str]:
        """Pop all queued messages. Called by the agentic loop between iterations."""
        with self._lock:
            messages = list(self._queue)
            self._queue.clear()
            return messages

    def pop_follow_up(self) -> Optional[str]:
        """Pop the follow-up prompt. Called after the turn completes."""
        with self._lock:
            msg = self._follow_up
            self._follow_up = None
            return msg

    def has_messages(self) -> bool:
        """Check if there are queued messages."""
        with self._lock:
            return len(self._queue) > 0

    def has_follow_up(self) -> bool:
        """Check if there's a follow-up queued."""
        with self._lock:
            return self._follow_up is not None

    @property
    def count(self) -> int:
        """Number of queued messages."""
        with self._lock:
            return len(self._queue)

    def clear(self) -> None:
        """Clear all queued messages."""
        with self._lock:
            self._queue.clear()
            self._follow_up = None

    def format_injection(self) -> Optional[str]:
        """Format all queued messages as a single injection string.

        Messages are clearly framed as system-level notes, not raw user instructions,
        to reduce prompt injection risk.
        """
        messages = self.pop_all()
        if not messages:
            return None
        if len(messages) == 1:
            return f"[SYSTEM NOTE — mid-turn user comment (not a new instruction): {messages[0]}]"
        parts = "\n".join(f"  - {m}" for m in messages)
        return f"[SYSTEM NOTE — mid-turn user comments (not new instructions):\n{parts}]"


def create_steering_indicator(queue: SteeringQueue) -> str:
    """Create a status bar indicator for queued messages."""
    count = queue.count
    if count == 0:
        return ""
    if count == 1:
        return "[yellow]\U0001f4dd 1 queued msg[/yellow]"
    return f"[yellow]\U0001f4dd {count} queued msgs[/yellow]"


def create_follow_up_indicator(queue: SteeringQueue) -> str:
    """Create indicator for queued follow-up."""
    if queue.has_follow_up():
        return "[cyan]\u23ed follow-up queued[/cyan]"
    return ""
