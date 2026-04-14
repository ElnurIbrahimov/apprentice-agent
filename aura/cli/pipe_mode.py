"""Unix-composable pipe mode — stdout for content, stderr for status."""
from __future__ import annotations

import json
import sys
from typing import Any, Optional


class PipeOutput:
    """Manages stdout/stderr split for pipe-friendly operation."""

    def __init__(self, format: str = "text"):
        """format: 'text' (default), 'json', 'markdown'"""
        self._format = format

    def content(self, text: str) -> None:
        """Write content to stdout (pipe-safe)."""
        sys.stdout.write(text)
        sys.stdout.flush()

    def status(self, message: str) -> None:
        """Write status to stderr (not captured by pipes)."""
        sys.stderr.write(message + "\n")
        sys.stderr.flush()

    def error(self, message: str) -> None:
        """Write error to stderr."""
        sys.stderr.write(f"error: {message}\n")
        sys.stderr.flush()

    def result(self, data: dict) -> None:
        """Write structured result. Format depends on mode."""
        if self._format == "json":
            sys.stdout.write(json.dumps(data, indent=2) + "\n")
        elif self._format == "markdown":
            # Clean markdown — no Rich formatting
            content = data.get("response", data.get("content", ""))
            sys.stdout.write(content + "\n")
        else:
            content = data.get("response", data.get("content", ""))
            sys.stdout.write(content + "\n")
        sys.stdout.flush()


class StreamingJSONEmitter:
    """Emits one JSON object per line (JSONL) to stdout.

    Intended for scripted consumers of `aura -p "..."` that want to stream
    agentic events (token chunks, tool starts, tool results, final summary)
    as they arrive instead of waiting for a single final payload.
    """

    def emit(self, event: dict) -> None:
        sys.stdout.write(json.dumps(event, default=str) + "\n")
        sys.stdout.flush()

    def emit_chunk(self, text: str) -> None:
        self.emit({"type": "chunk", "text": text})

    def emit_tool_start(self, tool: str, args: dict) -> None:
        self.emit({"type": "tool_start", "tool": tool, "args": args})

    def emit_tool_result(self, tool: str, args: dict, result: Any) -> None:
        self.emit({"type": "tool_result", "tool": tool, "args": args, "result": result})

    def emit_final(self, response: str, model: str = "", cost: float = 0.0,
                   iterations: int = 0, tool_calls: int = 0, success: bool = True) -> None:
        self.emit({
            "type": "final",
            "response": response,
            "model": model,
            "cost": cost,
            "iterations": iterations,
            "tool_calls": tool_calls,
            "success": success,
        })


def is_pipe_mode() -> bool:
    """Detect if stdin or stdout is a pipe (non-interactive).

    On Windows/MSYS2, isatty() on stdin can be unreliable when a TERM
    environment variable is set (e.g. inside mintty).  Fall back to
    checking stdout only in that case.
    """
    import os
    if os.name == "nt" and os.environ.get("TERM"):
        return not sys.stdout.isatty()
    return not sys.stdin.isatty() or not sys.stdout.isatty()


def read_piped_input() -> Optional[str]:
    """Read piped stdin if available."""
    if sys.stdin.isatty():
        return None
    try:
        content = sys.stdin.read()
        return content.strip() if content else None
    except (EOFError, OSError):
        return None


# Exit codes
EXIT_SUCCESS = 0
EXIT_ERROR = 1
EXIT_BUDGET_EXCEEDED = 2
EXIT_TIMEOUT = 3
EXIT_USER_CANCEL = 130  # Standard Unix convention for Ctrl+C
