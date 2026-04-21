"""Execution and post-response handling for the interactive chat session."""
from __future__ import annotations

import base64
import logging
import re
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

from ._constants import ERROR_SENTINELS as _ERROR_SENTINELS

_IMAGE_TOKEN_RE = re.compile(r"\[image:\s*([^\]\n]+?)\s*\]")

# Tool names that mutate files — snapshot before execution so /rewind works.
_EDIT_TOOL_NAMES = {"edit_file", "write_file", "patch_file", "apply_diff", "str_replace_editor"}


def _extract_edit_paths(tool_name: str, args: dict[str, Any]) -> list[str]:
    """Pull file paths out of an edit-tool's argument dict.

    Different edit tools use different arg keys; try the usual suspects and
    return a de-duplicated list. Empty list means "no snapshot, nothing to
    back up" (e.g. write_file creating a brand-new file — checkpoint records
    the non-existence and /rewind will delete it on restore).
    """
    paths: list[str] = []
    for key in ("path", "file_path", "target", "filename", "file"):
        val = args.get(key)
        if isinstance(val, str) and val:
            paths.append(val)
    files_arg = args.get("files")
    if isinstance(files_arg, list):
        for item in files_arg:
            if isinstance(item, str):
                paths.append(item)
            elif isinstance(item, dict):
                for key in ("path", "file_path"):
                    v = item.get(key)
                    if isinstance(v, str) and v:
                        paths.append(v)
    seen: set[str] = set()
    ordered: list[str] = []
    for p in paths:
        if p not in seen:
            seen.add(p)
            ordered.append(p)
    return ordered


def _extract_images_from_prompt(text: str) -> tuple[str, list[str]]:
    """Pull [image: <path>] tokens out of the prompt.

    Returns (cleaned_text, base64_images). Tokens pointing at missing files
    are silently dropped. Uses data uri-friendly raw b64 for Ollama's `images`
    message field.
    """
    if not text or "[image:" not in text:
        return text, []

    imgs: list[str] = []
    def _sub(m: re.Match) -> str:
        raw = m.group(1).strip().strip('"').strip("'")
        try:
            p = Path(raw).expanduser()
            if p.is_file():
                data = p.read_bytes()
                imgs.append(base64.b64encode(data).decode("ascii"))
                return ""
        except Exception:
            logger.debug("image_token_parse_failed", exc_info=True)
        return m.group(0)  # leave token in place if unreadable

    cleaned = _IMAGE_TOKEN_RE.sub(_sub, text).strip()
    return cleaned, imgs


class SessionExecutionController:
    """Owns the normal agent execution path for ChatSession."""

    def __init__(self, session: Any) -> None:
        self._session = session

    def run_agent(self, user_input: str) -> Optional[dict]:
        """Run the agentic loop for a user prompt."""
        import time as _exec_time

        from .display import StreamingResponse, show_error, show_response_attribution
        from aura.core.agentic_loop_events import LoopEvent

        streamer = StreamingResponse(model=self._session.current_model)
        streamer.start()
        tool_call_count = 0
        exec_start = _exec_time.monotonic()
        result: Optional[dict] = None

        # Start a background Escape listener so users can cancel in-flight
        # streaming with a single keystroke. Falls back silently on POSIX
        # where the raw-mode approach isn't available without stealing
        # prompt_toolkit's tty.
        cancel_watch_stop = self._start_escape_watchdog()

        # Outer try/finally guarantees streamer.finish() and the escape watchdog
        # release no matter how we leave — including exceptions thrown mid-stream
        # from agentic.run(). Previously finish() sat outside the try block and
        # never ran on error, leaving the Rich Live display orphaned and
        # corrupting the next turn's render.
        try:
            try:
                def _on_event(event: LoopEvent) -> None:
                    nonlocal tool_call_count
                    if event.type == "chunk":
                        streamer.chunk(str(event.payload.get("text", "")))
                    elif event.type == "tool_start":
                        tool_call_count += 1
                        streamer.pause()
                        self._handle_tool_start(
                            str(event.payload.get("tool_name", "")),
                            dict(event.payload.get("tool_args", {})),
                        )
                    elif event.type == "tool_result":
                        self._handle_tool_result(
                            str(event.payload.get("tool_name", "")),
                            dict(event.payload.get("tool_args", {})),
                            event.payload.get("tool_result"),
                        )
                        streamer.resume()
                    elif event.type == "verification_start":
                        streamer.pause()
                        mode = event.payload.get("mode", "?")
                        n = len(event.payload.get("changed_files", []) or [])
                        self._session.console.print(
                            f"  [dim cyan]verify[/dim cyan] {mode} · {n} file(s)"
                        )
                        streamer.resume()
                    elif event.type == "verification_passed":
                        streamer.pause()
                        dur = float(event.payload.get("duration_s", 0.0) or 0.0)
                        self._session.console.print(
                            f"  [green]✓ verification passed[/green] [dim]({dur:.1f}s)[/dim]"
                        )
                        streamer.resume()
                    elif event.type == "verification_failed":
                        streamer.pause()
                        dur = float(event.payload.get("duration_s", 0.0) or 0.0)
                        stages = event.payload.get("stages", []) or []
                        n_fail = sum(
                            len(s.get("failures", [])) for s in stages
                            if not s.get("success")
                        )
                        self._session.console.print(
                            f"  [red]✗ verification failed[/red] "
                            f"[dim]({dur:.1f}s, {n_fail} issue(s))[/dim]"
                        )
                        streamer.resume()
                    elif event.type == "stuck":
                        streamer.pause()
                        reason = event.payload.get("reason", "?")
                        detail = event.payload.get("details", "")
                        self._session.console.print(
                            f"  [yellow]⚠ aura thinks it's stuck[/yellow] "
                            f"[dim]({reason})[/dim]  {detail}"
                        )
                        streamer.resume()
                    elif event.type == "turn_rolled_back":
                        streamer.pause()
                        restored = int(event.payload.get("restored", 0) or 0)
                        attempted = int(event.payload.get("attempted", 0) or 0)
                        paths = event.payload.get("paths", []) or []
                        partial = bool(event.payload.get("partial", False))
                        status = (
                            "[red]partial[/red]" if partial else "[green]ok[/green]"
                        )
                        self._session.console.print(
                            f"  [yellow]↺ rolled back[/yellow] {restored}/{attempted} "
                            f"checkpoint(s) · {len(paths)} file(s) · {status}"
                        )
                        for p in paths[:5]:
                            self._session.console.print(f"      [dim]- {p}[/dim]")
                        if len(paths) > 5:
                            self._session.console.print(
                                f"      [dim]… and {len(paths) - 5} more[/dim]"
                            )
                        streamer.resume()

                # Extract [image: path] tokens and convert to base64 for vision models
                cleaned_input, _images = _extract_images_from_prompt(user_input)
                _run_kwargs: dict = {
                    "on_event": _on_event,
                    "steering_queue": self._session.steering,
                }
                if _images:
                    streamer.pause()
                    from .display import console as _img_console
                    _img_console.print(f"  [dim cyan]attached {len(_images)} image(s) for vision routing[/]")
                    streamer.resume()
                    _run_kwargs["images"] = _images
                result = self._session.agentic.run(
                    cleaned_input or user_input,
                    **_run_kwargs,
                )
            except KeyboardInterrupt:
                self._session._handle_ctrl_c_abort(streamer)
                return None
            except Exception as exc:
                streamer.pause()
                show_error(str(exc))
                return None

            # Feed per-turn stats into the streamer before finishing so the
            # summary line shows $cost and ctx% in addition to token counts.
            try:
                from .context_bar import estimate_messages_tokens, get_context_limit
                cost_delta = 0.0
                try:
                    stats = self._session.agent.brain.get_session_stats()
                    cur_cost = float(stats.get("cost_usd", 0.0) or 0.0)
                    prev_cost = float(getattr(self._session, "_last_session_cost", 0.0) or 0.0)
                    cost_delta = max(0.0, cur_cost - prev_cost)
                    self._session._last_session_cost = cur_cost
                except Exception:
                    logger.debug("Failed to compute cost delta for turn stats", exc_info=True)
                ctx_used = estimate_messages_tokens(self._session.agentic._conversation_history)
                ctx_limit = get_context_limit(self._session.current_model)
                streamer.set_turn_stats(cost_delta=cost_delta, ctx_used=ctx_used, ctx_limit=ctx_limit)
            except Exception:
                logger.debug("Failed to set turn stats on streamer", exc_info=True)

            self._session._streamer_displayed = True

            elapsed = _exec_time.monotonic() - exec_start
            if tool_call_count > 0 or elapsed > 2.0:
                iter_count = getattr(self._session.agentic, "iteration", 0)
                summary_parts = []
                if iter_count > 1:
                    summary_parts.append(f"{iter_count} steps")
                if tool_call_count > 0:
                    summary_parts.append(f"{tool_call_count} tool calls")
                show_response_attribution(
                    model=self._session.current_model,
                    elapsed=elapsed,
                    tokens=result.get("tokens", 0) if result else 0,
                )
                if summary_parts:
                    self._print_execution_summary(summary_parts)

            return result
        finally:
            if cancel_watch_stop is not None:
                try:
                    cancel_watch_stop()
                except Exception:
                    logger.debug("cancel_watch_stop_failed", exc_info=True)
            try:
                streamer.finish()
            except Exception:
                logger.debug("streamer_finish_failed", exc_info=True)

    def _start_escape_watchdog(self):
        """Start a background keyboard watcher that cancels the agentic loop on Escape.

        Windows only (msvcrt). On POSIX, returns None and the user falls back
        to Ctrl+C. Returns a stop() callable that the caller must invoke in a
        finally block, or None if no watcher was started.
        """
        import os
        if os.name != "nt":
            return None
        try:
            import msvcrt  # type: ignore[import]
        except ImportError:
            return None

        import threading

        stop_evt = threading.Event()
        session = self._session

        def _watch():
            while not stop_evt.is_set():
                try:
                    if msvcrt.kbhit():  # type: ignore[attr-defined]
                        ch = msvcrt.getwch()  # type: ignore[attr-defined]
                        # ESC is '\x1b'. Ignore anything else so we don't
                        # swallow keys that belong to prompt_toolkit's next
                        # input cycle.
                        if ch == "\x1b":
                            try:
                                session.agentic.cancel()
                                session.console.print(
                                    "\n  [red]Aborted (Esc).[/red]"
                                )
                            except Exception:
                                logger.debug("Failed to cancel agentic loop on Esc", exc_info=True)
                            return
                except Exception:
                    return
                stop_evt.wait(0.1)

        thread = threading.Thread(target=_watch, name="aura-esc-watchdog", daemon=True)
        thread.start()

        # Return a stopper that both signals the thread AND waits briefly for
        # it to exit. Without the join() there is a <=100ms race where the
        # thread can still be inside msvcrt.getwch() when streamer.finish()
        # tears down the Live display, leading to cancel-on-done or
        # console.print-on-closed-stream.
        def _stop() -> None:
            stop_evt.set()
            thread.join(timeout=0.2)
        return _stop

    def process_normal_result(self, user_input: str, result: Optional[dict]) -> bool:
        """Render and track a normal execution result. Returns True when handled successfully."""
        from .context_bar import estimate_messages_tokens, get_context_limit
        from .display import show_context_summary, show_error, show_info, show_response

        if result is None:
            show_error("No response received.")
            return False

        response_text = result.get("response", "")
        model_used = result.get("model", self._session.current_model)
        is_error = result.get("success") is False or any(
            response_text.startswith(s) for s in _ERROR_SENTINELS
        )
        if is_error:
            show_error(response_text)
            return False

        memory_count, mood, tool_count = self._build_context_summary(result)
        show_context_summary(
            memory_count=memory_count,
            mood=mood,
            model=model_used,
            tool_count=tool_count,
        )

        if response_text and not self._session._streamer_displayed:
            show_response(response_text, model=model_used, stream=False)

        self._log_activity(user_input, response_text, result)
        self._track_conversation(user_input, response_text)

        follow_up = self._session.steering.pop_follow_up()
        if follow_up and self._session._follow_up_depth < self._session._MAX_FOLLOW_UP_DEPTH:
            self._session._pending_follow_up = follow_up
            self._session._follow_up_depth += 1
        elif follow_up:
            show_info("Max auto-follow-up depth reached, dropping follow-up.")

        self._session.msg_count += 1
        if self._session.msg_count == 1 and user_input:
            self._session.session_title = user_input[:50].strip()
        # current_model is maintained by apply_model_override; no re-read needed.
        self._session.token_used = estimate_messages_tokens(
            self._session.agentic._conversation_history
        )
        self._session.token_limit = get_context_limit(self._session.current_model)

        cost_usd = 0.0
        try:
            stats = self._session.agent.brain.get_session_stats()
            cost_usd = stats.get("cost_usd", 0.0)
        except (AttributeError, TypeError, KeyError):
            logger.debug("session_stats_read_failed", exc_info=True)

        self._session._show_bar(
            model=self._session.current_model,
            project_type=self._session._project_type,
            session_title=self._session.session_title,
            message_count=self._session.msg_count,
            cost_usd=cost_usd,
            token_used=self._session.token_used,
            token_limit=self._session.token_limit,
            permission_mode=self._session.perm_mode,
        )

        if self._session.hook_mgr:
            self._session.hook_mgr.fire(
                self._session._HookEvent.POST_RESPONSE,
                {
                    "response": response_text[:500] if response_text else "",
                    "model": model_used,
                },
            )

        if self._session.speak and response_text:
            try:
                self._session.agent._speak(response_text)
            except (OSError, RuntimeError, AttributeError):
                logger.warning("tts_speak_failed", exc_info=True)

        return True

    def _handle_tool_start(self, name: str, args: dict[str, Any]) -> None:
        from .display import show_tool_call

        step = getattr(self._session.agentic, "iteration", 0)
        max_iter = getattr(self._session.agentic, "max_iterations", 0)

        # Snapshot files before edit tools run so /rewind has something to
        # restore. Without this the CheckpointManager exists but its index
        # stays empty — rewind UI would show no entries.
        if name in _EDIT_TOOL_NAMES:
            cp_mgr = getattr(self._session, "checkpoint_mgr", None)
            paths = _extract_edit_paths(name, args)
            if cp_mgr is not None and paths:
                try:
                    cp_mgr.snapshot_multi(paths, label=name)
                except Exception:
                    logger.debug("checkpoint_snapshot_failed", exc_info=True)
            # Feed the turn-scoped rollback checkpoint too. Paths already
            # captured this turn are no-ops. Cheap on repeat calls.
            agentic = getattr(self._session, "agentic", None)
            if agentic is not None and paths:
                try:
                    agentic._ensure_turn_checkpoint(paths)
                except Exception:
                    logger.debug("turn_checkpoint_snapshot_failed", exc_info=True)

        if self._session.hook_mgr:
            self._session.hook_mgr.fire(
                self._session._HookEvent.PRE_TOOL_CALL,
                {
                    "tool_name": name,
                    "tool_args": str(args)[:500],
                },
            )

        desc = args.get("path") or args.get("pattern") or args.get("query") or ""
        if not desc and "command" in args:
            desc = args["command"][:60]

        show_tool_call(name, str(desc), step=step, max_steps=max_iter, status="running")

    def _handle_tool_result(self, name: str, args: dict[str, Any], result: Any) -> None:
        from .display import show_tool_result_inline

        show_tool_result_inline(name, result)

        if self._session.hook_mgr:
            self._session.hook_mgr.fire(
                self._session._HookEvent.POST_TOOL_CALL,
                {
                    "tool_name": name,
                    "tool_args": str(args)[:500],
                },
            )
        if self._session.hook_mgr and name in ("edit_file", "write_file"):
            self._session.hook_mgr.fire(
                self._session._HookEvent.POST_EDIT,
                {
                    "tool_name": name,
                    "file_path": args.get("path", args.get("file_path", "")),
                },
            )

        if name in ("edit_file", "write_file") and getattr(
            self._session, "_auto_test_enabled", False
        ):
            try:
                test_result = self._session.agentic._run_auto_test()
                if test_result:
                    self._session.agentic._conversation_history.append(
                        {
                            "role": "user",
                            "content": f"[Auto-test failed after editing] {test_result}",
                        }
                    )
            except Exception:
                logger.debug("Failed to run auto-test after file edit", exc_info=True)

    def _print_execution_summary(self, summary_parts: list[str]) -> None:
        try:
            import os as _os

            edited_files = [f for f in getattr(self._session.agentic, "_hot_files", []) if f]
            if edited_files:
                files_display = ", ".join(_os.path.basename(f) for f in edited_files[:8])
                extra = f" (+{len(edited_files) - 8} more)" if len(edited_files) > 8 else ""
                parts_str = " \u00b7 ".join(summary_parts)
                self._session.console.print(
                    f"  [dim]Files touched: {files_display}{extra} | {parts_str}[/dim]"
                )
            else:
                self._session.console.print(f"  [dim]{' \u00b7 '.join(summary_parts)}[/dim]")
        except Exception:
            self._session.console.print(f"  [dim]{' \u00b7 '.join(summary_parts)}[/dim]")

    def _build_context_summary(self, result: dict) -> tuple[int, str, int]:
        memory_count = 0
        mood = ""
        tool_count = 0
        try:
            if hasattr(self._session.agent, "memory") and hasattr(
                self._session.agent.memory, "memories"
            ):
                memory_count = len(self._session.agent.memory.memories)
            elif hasattr(self._session.agent, "memory") and hasattr(
                self._session.agent.memory, "count"
            ):
                memory_count = self._session.agent.memory.count()
        except (TypeError, AttributeError):
            logger.debug("ctx_memory_count_failed", exc_info=True)
        try:
            if hasattr(self._session.agent, "mood") and self._session.agent.mood:
                mood = (
                    str(self._session.agent.mood.get("mood", ""))
                    if isinstance(self._session.agent.mood, dict)
                    else str(self._session.agent.mood)
                )
        except (TypeError, AttributeError):
            logger.debug("ctx_mood_read_failed", exc_info=True)
        try:
            tool_count = result.get("tool_calls", 0)
        except (TypeError, AttributeError):
            logger.debug("ctx_tool_count_failed", exc_info=True)
        return memory_count, mood, tool_count

    def _log_activity(self, user_input: str, response_text: str, result: dict) -> None:
        if self._session.activity_log:
            try:
                self._session.activity_log.log(
                    prompt=user_input,
                    response=response_text[:20000] if response_text else "",
                    model=result.get("model", ""),
                    session_id=getattr(self._session.agentic_session, "session_id", ""),
                    tool_calls=result.get("tool_calls", 0),
                )
            except (OSError, TypeError, ValueError):
                logger.debug("activity_log_write_failed", exc_info=True)

    def _track_conversation(self, user_input: str, response_text: str) -> None:
        if self._session._cm_conv_id:
            try:
                from aura.core.conversation_manager import get_conversation_manager

                cm = get_conversation_manager()
                cm.on_message_added(self._session._cm_conv_id, "user", user_input, "cli", "local")
                cm.on_message_added(
                    self._session._cm_conv_id,
                    "assistant",
                    response_text,
                    "cli",
                    "local",
                )
            except Exception:
                logger.debug("Failed to sync ConversationManager after response", exc_info=True)
