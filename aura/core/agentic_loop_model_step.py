"""Model-step execution and content resolution for AgenticLoop."""
from __future__ import annotations

import logging
import re
import sys
from typing import Any

from .agentic_loop_events import LoopEventEmitter
from .agentic_loop_outcomes import LoopOutcome, ModelStepResult

logger = logging.getLogger(__name__)


class ModelStepController:
    """Owns the model request and content-only decision path inside AgenticLoop.run()."""

    def __init__(self, loop: Any) -> None:
        self._loop = loop

    def request_step(
        self,
        *,
        messages: list[dict[str, Any]],
        active_tools: list[dict[str, Any]],
        step_model: str | None,
        event_emitter: Any = None,
        on_chunk: Any = None,
    ) -> ModelStepResult:
        if event_emitter is None and on_chunk is not None:
            event_emitter = LoopEventEmitter(self._loop, on_chunk=on_chunk)

        accumulated = ""
        tool_calls = None
        content = ""
        stream_error = None
        model_used = ""

        try:
            if not event_emitter or not event_emitter.listens_for("chunk"):
                model_tag = f" [{step_model.split(':')[0]}]" if step_model else ""
                sys.stdout.write(f"  \033[90m● thinking{model_tag}...\033[0m")
                sys.stdout.flush()

            for chunk_type, data in self._loop.brain.think_with_tools_stream(
                messages=messages,
                tools=active_tools,
                model_override=step_model,
            ):
                if chunk_type == "content":
                    if (not event_emitter or not event_emitter.listens_for("chunk")) and not accumulated:
                        sys.stdout.write("\r\033[K")
                        sys.stdout.write("  \033[90m● generating...\033[0m")
                        sys.stdout.flush()
                    accumulated += data
                    if event_emitter:
                        event_emitter.emit("chunk", text=data, delivery="stream")
                elif chunk_type == "tool_calls":
                    if (not event_emitter or not event_emitter.listens_for("chunk")) and not accumulated:
                        sys.stdout.write("\r\033[K")
                    tool_calls = data
                elif chunk_type == "done":
                    model_used = data.get("model", "")
                    if data.get("content"):
                        content = data["content"]
                elif chunk_type == "error":
                    stream_error = data.get("error", "Unknown stream error")
                    break

            if not event_emitter or not event_emitter.listens_for("chunk"):
                if accumulated:
                    sys.stdout.write("\r\033[K")
                    sys.stdout.flush()
                elif not tool_calls:
                    sys.stdout.write("\r\033[K")
                    sys.stdout.flush()

        except ConnectionError:
            sys.stdout.write("\r\033[K")
            sys.stdout.flush()
            model_label = step_model or "default model"
            self._loop._loop_error = True
            return ModelStepResult.terminal(LoopOutcome.model_connection_error(model_label))
        except TimeoutError:
            sys.stdout.write("\r\033[K")
            sys.stdout.flush()
            model_label = step_model or "default model"
            self._loop._loop_error = True
            return ModelStepResult.terminal(LoopOutcome.model_timeout(model_label))
        except Exception as exc:
            stream_error = str(exc)

        delivery = "stream"
        if stream_error:
            sys.stdout.write("\r\033[K")
            sys.stdout.flush()
            logger.debug(
                f"[AgenticLoop] Streaming failed ({stream_error}), falling back to blocking call"
            )
            result = self._loop.brain.think_with_tools(
                messages=messages,
                tools=active_tools,
                model_override=step_model,
            )
            if "error" in result:
                self._loop._loop_error = True
                return ModelStepResult.terminal(
                    LoopOutcome.model_error(f"Error: {result['error']}")
                )

            msg = result["message"]
            model_used = result.get("model", "")
            if isinstance(msg, dict):
                tool_calls = msg.get("tool_calls")
                content = msg.get("content", "") or ""
            else:
                tool_calls = getattr(msg, "tool_calls", None)
                content = getattr(msg, "content", "") or ""
            delivery = "blocking"
        else:
            content = content or accumulated

        content = re.sub(r"</?tool_call>|</?tool_result[^>]*>", "", content).strip()
        content = re.sub(r"\n{3,}", "\n\n", content)

        if tool_calls:
            return ModelStepResult.tool_calls_ready(
                tool_calls=tool_calls,
                content=content,
                model_used=model_used,
                delivery=delivery,
            )
        return ModelStepResult.content_ready(
            content=content,
            model_used=model_used,
            delivery=delivery,
        )

    def resolve_content_only(
        self,
        *,
        prompt: str,
        content: str,
        delivery: str,
    ) -> ModelStepResult:
        if not content:
            self._loop._empty_response_count = getattr(self._loop, "_empty_response_count", 0) + 1
            if self._loop._empty_response_count > 3:
                logger.error(
                    f"[AgenticLoop] {self._loop._empty_response_count} consecutive empty responses — aborting loop"
                )
                return ModelStepResult.terminal(
                    LoopOutcome.completed(
                        "The model failed to generate a response after multiple attempts. "
                        "Please try again with a clearer prompt."
                    )
                )
            logger.warning(
                f"[AgenticLoop] Empty response #{self._loop._empty_response_count} from model on iteration {self._loop.iteration}, nudging"
            )
            return ModelStepResult.retry(
                "retry_empty_response",
                extra_messages=[
                    {"role": "assistant", "content": ""},
                    {"role": "user", "content": "Continue. Execute the task using tools."},
                ],
            )

        thinking_phrases = (
            "let me search",
            "let me look",
            "let me find",
            "i'll search",
            "i will search",
            "let me do a",
            "let me check",
            "let me research",
            "i'll look up",
            "let me query",
            "searching for",
        )
        content_lower = content.lower().strip()
        is_thinking_without_acting = (
            self._loop.iteration <= 2
            and len(content) < 500
            and any(
                content_lower.startswith(phrase) or f"\n{phrase}" in content_lower
                for phrase in thinking_phrases
            )
            and self._loop.tool_calls_total == 0
        )
        if is_thinking_without_acting:
            nudge_count = getattr(self._loop, "_thinking_nudge_count", 0) + 1
            self._loop._thinking_nudge_count = nudge_count
            if nudge_count <= 2:
                logger.warning(
                    f"[AgenticLoop] Model is thinking without acting (nudge #{nudge_count}): '{content[:80]}...'"
                )
                return ModelStepResult.retry(
                    "retry_thinking_without_acting",
                    extra_messages=[
                        {"role": "assistant", "content": content},
                        {
                            "role": "user",
                            "content": (
                                "Don't just describe what you'll do — actually use the available tools now. "
                                "Call web_search or the appropriate tool to execute."
                            ),
                        },
                    ],
                )

        self._loop._empty_response_count = 0

        if (
            self._loop._verify_completion
            and not self._loop._verification_done
            and self._loop.iteration >= 2
            and self._loop.tool_calls_total > 0
        ):
            self._loop._verification_done = True
            incomplete_reason = self._loop._verify_task_completion(prompt, content)
            if incomplete_reason:
                logger.info("[AgenticLoop] Verification found incomplete work, continuing")
                return ModelStepResult.retry(
                    "retry_verification",
                    extra_messages=[
                        {"role": "assistant", "content": content},
                        {
                            "role": "user",
                            "content": (
                                f"[Verification check] You said you were done, but verification found: "
                                f"{incomplete_reason}\n\n"
                                f"Please complete the remaining work."
                            ),
                        },
                    ],
                )

        if delivery == "blocking":
            return ModelStepResult.terminal(LoopOutcome.completed(content))
        return ModelStepResult.content_ready(
            content=content,
            model_used="",
            delivery=delivery,
        )
