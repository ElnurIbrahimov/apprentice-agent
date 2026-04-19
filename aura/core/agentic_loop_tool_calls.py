"""Tool-call parsing, execution, and result folding for AgenticLoop."""
from __future__ import annotations

import json
import logging
from typing import Any

from .agentic_loop_events import LoopEventEmitter
from .agentic_loop_outcomes import LoopOutcome, ToolBatchResult
from .tool_executor import get_tool_pool as _get_tool_pool

logger = logging.getLogger(__name__)


class ToolCallCoordinator:
    """Owns the tool-call pipeline inside AgenticLoop.run()."""

    def __init__(self, loop: Any) -> None:
        self._loop = loop

    def append_assistant_tool_message(
        self,
        messages: list[dict[str, Any]],
        content: str,
        tool_calls: list[Any],
        event_emitter: Any = None,
        on_response: Any = None,
    ) -> None:
        if event_emitter is None and on_response is not None:
            event_emitter = LoopEventEmitter(self._loop, on_response=on_response)
        assistant_msg = {
            "role": "assistant",
            "content": content,
            "tool_calls": tool_calls,
        }
        messages.append(assistant_msg)
        if self._loop.session:
            self._loop.session.append(assistant_msg)
        if content and event_emitter:
            event_emitter.emit(
                "response",
                text=content,
                delivery="tool_call_preface",
            )

    def parse_tool_calls(
        self,
        tool_calls: list[Any],
        messages: list[dict[str, Any]],
    ) -> list[tuple[str, dict[str, Any]]]:
        parsed_calls: list[tuple[str, dict[str, Any]]] = []
        for tc in tool_calls:
            if isinstance(tc, dict):
                func = tc.get("function", {})
                tool_name = func.get("name", "")
                args = func.get("arguments", {})
            else:
                func = getattr(tc, "function", None)
                tool_name = getattr(func, "name", "") if func else ""
                args = getattr(func, "arguments", {}) if func else {}
            try:
                if isinstance(args, str):
                    args = json.loads(args)
                if args is None:
                    args = {}
            except (json.JSONDecodeError, TypeError):
                logger.warning(
                    f"[AgenticLoop] Failed to parse tool args for {tool_name}: {str(args)[:200]}"
                )
                error_result = json.dumps(
                    {
                        "error": (
                            f"Malformed arguments for {tool_name}: could not parse JSON. "
                            f"Please provide valid JSON arguments. Raw: {str(args)[:200]}"
                        )
                    }
                )
                tool_msg = {"role": "tool", "content": error_result}
                messages.append(tool_msg)
                if self._loop.session:
                    self._loop.session.append(tool_msg)
                continue
            parsed_calls.append((tool_name, args))
        return parsed_calls

    def approve_and_execute(
        self,
        parsed_calls: list[tuple[str, dict[str, Any]]],
        *,
        event_emitter: Any = None,
        on_tool_call: Any = None,
        on_tool_start: Any = None,
    ) -> list[tuple[str, dict[str, Any], str]]:
        if event_emitter is None and (on_tool_call is not None or on_tool_start is not None):
            event_emitter = LoopEventEmitter(
                self._loop,
                on_tool_call=on_tool_call,
                on_tool_start=on_tool_start,
            )
        approved: list[tuple[str, dict[str, Any], str | None]] = []
        for tool_name, args in parsed_calls:
            self._loop.tool_calls_total += 1
            resolved_name = self._loop.executor._TOOL_ALIASES.get(tool_name, tool_name).lower()
            if not self._loop.permissions.check(resolved_name, args):
                approved.append(
                    (tool_name, args, json.dumps({"error": "Permission denied by user"}))
                )
                if not event_emitter or not event_emitter.listens_for("tool_result"):
                    self._loop._show_tool_status(tool_name, args, denied=True)
            else:
                if not event_emitter or not event_emitter.listens_for("tool_result"):
                    self._loop._show_tool_status(tool_name, args)
                approved.append((tool_name, args, None))

        needs_exec = [(i, t, a) for i, (t, a, r) in enumerate(approved) if r is None]
        if event_emitter:
            for _idx, tool_name, args in needs_exec:
                event_emitter.emit("tool_start", tool_name=tool_name, tool_args=args)

        if len(needs_exec) == 1:
            idx, tool_name, args = needs_exec[0]
            result = self._loop.executor.execute(tool_name, args)
            approved[idx] = (tool_name, args, result)
        elif len(needs_exec) > 1:
            futures = {}
            pool = _get_tool_pool()
            for idx, tool_name, args in needs_exec:
                fut = pool.submit(self._loop.executor.execute, tool_name, args)
                futures[idx] = fut
            for idx, fut in futures.items():
                try:
                    result = fut.result(timeout=300)
                except Exception as exc:
                    result = json.dumps({"error": f"Tool execution failed: {exc}"})
                tool_name, args, _ = approved[idx]
                approved[idx] = (tool_name, args, result)

        return [(t, a, r or "") for t, a, r in approved]

    def collect_results(
        self,
        approved: list[tuple[str, dict[str, Any], str]],
        messages: list[dict[str, Any]],
        guard: Any,
        *,
        event_emitter: Any = None,
        on_tool_call: Any = None,
    ) -> ToolBatchResult:
        if event_emitter is None and on_tool_call is not None:
            event_emitter = LoopEventEmitter(self._loop, on_tool_call=on_tool_call)
        for tool_name, args, tool_result in approved:
            if tool_name in ("edit_file", "write_file") and not self._loop._tool_result_has_error(
                tool_result
            ):
                self._loop._edits_this_turn += 1
                self._loop._has_edits = True
                self._loop._last_tools_were_reads = False
            elif tool_name in ("read_file", "grep", "glob", "list_dir", "project_structure"):
                self._loop._last_tools_were_reads = True
            else:
                self._loop._last_tools_were_reads = False

            self._loop._track_hot_file(tool_name, args, tool_result)

            try:
                if (
                    self._loop._planner
                    and self._loop._planner.current_plan
                    and not self._loop._tool_result_has_error(tool_result)
                ):
                    if tool_name in ("edit_file", "write_file", "shell", "run_tests"):
                        result_snippet = tool_result[:100] if tool_result else ""
                        self._loop._planner.advance_step(result=result_snippet)
            except Exception as exc:
                logger.debug(f"[AgenticLoop] Planner advance failed: {exc}")

            if event_emitter:
                event_emitter.emit(
                    "tool_result",
                    tool_name=tool_name,
                    tool_args=args,
                    tool_result=tool_result,
                )

            # Context-saving: mask verbose tool outputs before they enter the LLM
            # message history. expand_observation is exempt so its response
            # reaches the LLM in full. Short outputs pass through unchanged.
            if tool_name == "expand_observation":
                llm_content = tool_result
            else:
                try:
                    from aura.memory.observation_masker import mask_tool_output
                    origin = (
                        args.get("path")
                        or args.get("pattern")
                        or args.get("command")
                        or args.get("query")
                        or ""
                    )
                    masked = mask_tool_output(
                        tool_result,
                        tool_name=tool_name,
                        origin=str(origin)[:60],
                    )
                    llm_content = masked.display
                except Exception as _mask_exc:
                    logger.debug("[AgenticLoop] observation masker failed: %s", _mask_exc)
                    llm_content = tool_result

            tool_msg = {"role": "tool", "content": llm_content}
            messages.append(tool_msg)
            if self._loop.session:
                self._loop.session.append(tool_msg)

            guard_result = guard.record(tool_name, str(args))
            if guard_result and guard_result.triggered:
                self._loop._loop_error = True
                # Surface a Rich panel so users see *why* the loop stopped,
                # not just a terse fallback message in the assistant turn.
                try:
                    from rich.panel import Panel
                    from aura.cli.display import console as _lg_console
                    files_edited = getattr(guard, "_files_edited", [])[-5:]
                    body_lines = [
                        f"[bold]Reason:[/] {guard_result.reason}",
                        f"[bold]Actions taken:[/] {guard_result.actions_taken}",
                        f"[bold]Novelty:[/] {guard_result.novelty_score:.2f}",
                    ]
                    if files_edited:
                        body_lines.append(f"[bold]Recent files:[/] {', '.join(files_edited)}")
                    body_lines.append("")
                    body_lines.append("[dim]Stopping to avoid looping. Give me a new instruction or /retry.[/]")
                    _lg_console.print(Panel.fit(
                        "\n".join(body_lines),
                        title="[yellow]Loop detected — pausing[/]",
                        border_style="yellow",
                    ))
                except Exception:
                    logger.debug("loop_guard_panel_failed", exc_info=True)
                return ToolBatchResult(
                    should_break=True,
                    outcome=LoopOutcome.guard_tripped(guard_result.fallback_message),
                )

            if self._loop._cancel_event.is_set():
                self._loop._loop_error = True
                return ToolBatchResult(
                    should_break=True,
                    outcome=LoopOutcome.cancelled(
                        f"Cancelled after {self._loop.iteration} iterations.",
                        error=True,
                    ),
                )

        return ToolBatchResult()
