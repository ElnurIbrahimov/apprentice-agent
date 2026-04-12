"""Typed internal outcomes for AgenticLoop execution."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class LoopOutcome:
    """Represents how an agentic run terminated."""

    status: str
    response: str
    error: bool = False

    @classmethod
    def completed(cls, response: str) -> "LoopOutcome":
        return cls(status="completed", response=response, error=False)

    @classmethod
    def visual_feedback(cls, response: str) -> "LoopOutcome":
        return cls(status="completed", response=response, error=False)

    @classmethod
    def budget_limit(cls, budget_usd: float) -> "LoopOutcome":
        return cls(
            status="budget_limit",
            response=f"Budget limit reached (${budget_usd:.2f}). Stopping.",
            error=True,
        )

    @classmethod
    def model_connection_error(cls, model_label: str) -> "LoopOutcome":
        return cls(
            status="model_connection_error",
            response=(
                f"Connection failed to {model_label}.\n"
                f"  - Is Ollama running? Try: ollama serve\n"
                f"  - Check your network connection."
            ),
            error=True,
        )

    @classmethod
    def model_timeout(cls, model_label: str) -> "LoopOutcome":
        return cls(
            status="model_timeout",
            response=(
                f"Request timed out for {model_label}.\n"
                f"  - The model may be overloaded or too large.\n"
                f"  - Try a smaller model with: /model <name>"
            ),
            error=True,
        )

    @classmethod
    def model_error(cls, message: str) -> "LoopOutcome":
        return cls(status="model_error", response=message, error=True)

    @classmethod
    def cancelled(cls, response: str, *, error: bool = False) -> "LoopOutcome":
        return cls(status="cancelled", response=response, error=error)

    @classmethod
    def guard_tripped(cls, response: str) -> "LoopOutcome":
        return cls(status="guard_tripped", response=response, error=True)

    @classmethod
    def max_iterations(cls, max_iterations: int, last_response: str) -> "LoopOutcome":
        return cls(
            status="max_iterations",
            response=(
                f"Reached maximum iterations ({max_iterations}). Last response:\n{last_response}"
            ),
            error=False,
        )

    def to_result_dict(self, *, iterations: int, tool_calls: int, model: str) -> dict:
        return {
            "success": not self.error,
            "status": self.status,
            "response": self.response,
            "iterations": iterations,
            "tool_calls": tool_calls,
            "model": model,
        }


@dataclass(frozen=True)
class ToolBatchResult:
    """Represents the result of processing one batch of tool calls."""

    should_break: bool = False
    outcome: LoopOutcome | None = None


@dataclass(frozen=True)
class ModelStepResult:
    """Represents one model step before tool-call execution begins."""

    status: str
    content: str = ""
    tool_calls: list[Any] | None = None
    model_used: str = ""
    delivery: str = "stream"
    outcome: LoopOutcome | None = None
    extra_messages: list[dict[str, str]] | None = None

    @classmethod
    def tool_calls_ready(
        cls,
        *,
        tool_calls: list[Any],
        content: str,
        model_used: str,
        delivery: str,
    ) -> "ModelStepResult":
        return cls(
            status="tool_calls_ready",
            content=content,
            tool_calls=tool_calls,
            model_used=model_used,
            delivery=delivery,
        )

    @classmethod
    def content_ready(
        cls,
        *,
        content: str,
        model_used: str,
        delivery: str,
    ) -> "ModelStepResult":
        return cls(
            status="content_ready",
            content=content,
            model_used=model_used,
            delivery=delivery,
        )

    @classmethod
    def retry(
        cls,
        status: str,
        *,
        extra_messages: list[dict[str, str]],
    ) -> "ModelStepResult":
        return cls(status=status, extra_messages=extra_messages)

    @classmethod
    def terminal(cls, outcome: LoopOutcome) -> "ModelStepResult":
        return cls(status="terminal", outcome=outcome)
