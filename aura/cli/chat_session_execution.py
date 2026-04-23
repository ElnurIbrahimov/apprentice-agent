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


_IMAGE_MAX_BYTES = 20 * 1024 * 1024  # 20 MB — vision models reject larger anyway
_IMAGE_ALLOWED_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}


def _extract_images_from_prompt(text: str) -> tuple[str, list[str]]:
    """Pull [image: <path>] tokens out of the prompt.

    Returns (cleaned_text, base64_images). Size-capped at 20 MB per image and
    extension-validated. Invalid tokens are removed from the prompt with a
    visible warning printed to the console so the user knows why the image
    didn't attach — previously they silently stayed in the prompt as literal
    text, confusing the LLM. Uses data uri-friendly raw b64 for Ollama's
    `images` message field.
    """
    if not text or "[image:" not in text:
        return text, []

    imgs: list[str] = []
    from .display import console as _img_warn_console

    def _warn(reason: str, path: str) -> None:
        try:
            _img_warn_console.print(
                f"  [dim yellow]⚠ image skipped ({reason}): {path}[/dim yellow]"
            )
        except Exception:
            logger.debug("image_warn_print_failed", exc_info=True)

    def _sub(m: re.Match) -> str:
        raw = m.group(1).strip().strip('"').strip("'")
        try:
            p = Path(raw).expanduser()
        except Exception:
            _warn("invalid path", raw)
            return ""
        if not p.is_file():
            _warn("file not found", str(p))
            return ""
        if p.suffix.lower() not in _IMAGE_ALLOWED_SUFFIXES:
            _warn(f"unsupported type {p.suffix}", str(p))
            return ""
        try:
            size = p.stat().st_size
        except OSError:
            _warn("stat failed", str(p))
            return ""
        if size > _IMAGE_MAX_BYTES:
            _warn(f"{size // (1024*1024)} MB exceeds 20 MB limit", str(p))
            return ""
        try:
            data = p.read_bytes()
        except (OSError, PermissionError):
            _warn("unreadable", str(p))
            return ""
        imgs.append(base64.b64encode(data).decode("ascii"))
        return ""

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

        # Drain deferred injections from prior turns (e.g., async auto-test
        # results that completed after the previous turn). Prepend them to
        # this user prompt so the model sees them as fresh context. Replaces
        # the prior pattern of mutating _conversation_history mid-turn from
        # the event-callback thread, which raced agentic.run().
        pending = getattr(self._session, "pending_injections", None)
        if pending:
            try:
                prefix = "\n\n".join(pending) + "\n\n---\n\n"
                user_input = prefix + (user_input or "")
            finally:
                pending.clear()

        streamer = StreamingResponse(model=self._session.current_model)
        streamer.start()
        tool_call_count = 0
        exec_start = _exec_time.monotonic()
        result: Optional[dict] = None

        # Start a background Escape listener so users can cancel in-flight
        # streaming with a single keystroke. Falls back silently on POSIX
        # where the raw-mode approach isn't available without stealing
        # prompt_toolkit's tty.
        cancel_watch_stop, esc_event = self._start_escape_watchdog()

        # Outer try/finally guarantees streamer.finish() and the escape watchdog
        # release no matter how we leave — including exceptions thrown mid-stream
        # from agentic.run(). Previously finish() sat outside the try block and
        # never ran on error, leaving the Rich Live display orphaned and
        # corrupting the next turn's render.
        #
        # The turn lock serializes this main interactive turn against bridge-
        # drained turns spawned by SessionRuntimeController.drain_channels.
        # AgenticLoop has no internal lock and shares _conversation_history,
        # so two concurrent agentic.run() calls would interleave history
        # mutations. Acquired here, released in the outer finally.
        self._session._turn_lock.acquire()
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
                logger.exception("agentic_run_failed")  # full traceback to log
                show_error(exc)  # classified user-facing message
                return None

            # Feed per-turn stats into the streamer before finishing so the
            # summary line shows $cost and ctx% in addition to token counts.
            # Cache ctx_used + cur_cost on the session so process_normal_result
            # can reuse them instead of recomputing (token estimation walks
            # the full conversation history; get_session_stats is a brain
            # call — both wasteful per turn).
            try:
                from .context_bar import estimate_messages_tokens, get_context_limit
                cost_delta = 0.0
                cur_cost = 0.0
                try:
                    stats = self._session.agent.brain.get_session_stats()
                    cur_cost = float(stats.get("cost_usd", 0.0) or 0.0)
                    prev_cost = float(getattr(self._session, "_last_session_cost", 0.0) or 0.0)
                    cost_delta = max(0.0, cur_cost - prev_cost)
                except Exception:
                    logger.debug("Failed to compute cost delta for turn stats", exc_info=True)
                # Always advance the baseline — even if the stats fetch
                # failed (cur_cost stays 0.0). Otherwise a stale baseline
                # from two turns ago inflates the next successful turn's
                # delta by the entire missed interval.
                self._session._last_session_cost = cur_cost
                ctx_used = estimate_messages_tokens(self._session.agentic._conversation_history)
                ctx_limit = get_context_limit(self._session.current_model)
                self._session._last_turn_ctx_used = ctx_used
                self._session._last_turn_cost_usd = cur_cost
                self._session._last_turn_cached = True
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
            # Release the turn lock first so a queued drain_channels can
            # pick up bridge messages during streamer teardown / watchdog
            # cleanup. Wrapped in try/except in case the acquire above
            # raised (defensive — current code can't reach here without it).
            try:
                self._session._turn_lock.release()
            except RuntimeError:
                logger.debug("turn_lock_release_skipped (already released)", exc_info=True)
            if cancel_watch_stop is not None:
                try:
                    cancel_watch_stop()
                except Exception:
                    logger.debug("cancel_watch_stop_failed", exc_info=True)
            try:
                streamer.finish()
            except Exception:
                logger.debug("streamer_finish_failed", exc_info=True)
            # Print the "Aborted (Esc)" message ONLY after Rich's Live slot
            # has been released by streamer.finish(). Printing from the
            # watchdog thread races Live and raises LiveError.
            if esc_event is not None and esc_event.is_set():
                try:
                    self._session.console.print("  [red]Aborted (Esc).[/red]")
                except Exception:
                    logger.debug("esc_abort_print_failed", exc_info=True)

    def _start_escape_watchdog(self):
        """Start a background keyboard watcher that cancels the agentic loop on Escape.

        Windows only (msvcrt). On POSIX, returns (None, None).

        Returns a tuple ``(stop_fn, esc_event)``:

        - ``stop_fn`` — callable the caller must invoke in a finally block.
        - ``esc_event`` — ``threading.Event`` that is set when ESC was
          pressed. The main thread checks this AFTER ``streamer.finish()``
          has released Rich's Live slot, then prints the "Aborted (Esc)"
          message. Printing from the watchdog thread races Rich and
          causes ``LiveError: Only one live display may be active at once``.

        Uses ``PeekConsoleInputW`` via ctypes to check the next input event
        WITHOUT consuming it, so non-ESC keys stay in the console buffer for
        prompt_toolkit's next input cycle. Without the peek, the watchdog's
        ``getwch()`` ate the user's first character on every streaming turn.
        Falls back to the legacy consume-everything behavior if the Win32
        call fails (unusual console handles, redirected stdin, etc.).
        """
        import os
        if os.name != "nt":
            return None, None
        try:
            import msvcrt  # type: ignore[import]
        except ImportError:
            return None, None

        import threading

        # ctypes shim for PeekConsoleInputW. Falls back to consume-everything
        # if initialization fails.
        _peek_available = False
        _peek_fn = None
        _std_input_h = None
        try:
            import ctypes
            from ctypes import wintypes

            class _KEY_EVENT_REC(ctypes.Structure):
                _fields_ = [
                    ("bKeyDown", wintypes.BOOL),
                    ("wRepeatCount", wintypes.WORD),
                    ("wVirtualKeyCode", wintypes.WORD),
                    ("wVirtualScanCode", wintypes.WORD),
                    ("uChar", ctypes.c_wchar),
                    ("dwControlKeyState", wintypes.DWORD),
                ]

            # Union of event-specific record types. We only read KeyEvent;
            # the other members just reserve enough bytes for the union.
            class _EVENT_UNION(ctypes.Union):
                _fields_ = [
                    ("KeyEvent", _KEY_EVENT_REC),
                    ("_pad", ctypes.c_byte * 16),
                ]

            class _INPUT_REC(ctypes.Structure):
                _fields_ = [("EventType", wintypes.WORD), ("Event", _EVENT_UNION)]

            _k32 = ctypes.windll.kernel32
            _std_input_h = _k32.GetStdHandle(wintypes.DWORD(-10))  # STD_INPUT_HANDLE
            _k32.PeekConsoleInputW.argtypes = [
                wintypes.HANDLE,
                ctypes.POINTER(_INPUT_REC),
                wintypes.DWORD,
                ctypes.POINTER(wintypes.DWORD),
            ]
            _k32.PeekConsoleInputW.restype = wintypes.BOOL
            _peek_fn = _k32.PeekConsoleInputW
            _INPUT_REC_T = _INPUT_REC
            _KEY_EVENT_TYPE = 0x0001
            _VK_ESCAPE = 0x1B
            # Validate handle is usable (GetStdHandle returns INVALID_HANDLE_VALUE == -1 cast).
            if _std_input_h and _std_input_h != wintypes.HANDLE(-1).value:
                _peek_available = True
        except Exception:
            logger.debug("peek_console_input_init_failed", exc_info=True)
            _peek_available = False

        stop_evt = threading.Event()
        esc_evt = threading.Event()
        session = self._session

        def _fire_esc() -> None:
            esc_evt.set()
            try:
                # cancel() just sets a threading.Event — thread-safe.
                # The ABORT MESSAGE is printed by the main thread after
                # Live is released.
                session.agentic.cancel()
            except Exception:
                logger.debug("Failed to cancel agentic loop on Esc", exc_info=True)

        def _watch():
            while not stop_evt.is_set():
                try:
                    if msvcrt.kbhit():  # type: ignore[attr-defined]
                        if _peek_available:
                            # Peek without consuming. If the next event is
                            # a KEY_EVENT for ESC keydown, we consume it and
                            # fire; otherwise we leave it for prompt_toolkit.
                            rec = _INPUT_REC_T()
                            n_read = wintypes.DWORD(0)
                            ok = _peek_fn(
                                _std_input_h,
                                ctypes.byref(rec),
                                1,
                                ctypes.byref(n_read),
                            )
                            if ok and n_read.value > 0:
                                is_esc = (
                                    rec.EventType == _KEY_EVENT_TYPE
                                    and bool(rec.Event.KeyEvent.bKeyDown)
                                    and rec.Event.KeyEvent.wVirtualKeyCode == _VK_ESCAPE
                                )
                                if is_esc:
                                    try:
                                        msvcrt.getwch()  # type: ignore[attr-defined]
                                    except Exception:
                                        pass
                                    _fire_esc()
                                    return
                                # Non-ESC: leave in buffer, let prompt_toolkit
                                # consume it when streaming ends.
                                # Sleep a bit longer so we don't tight-loop
                                # while a pending key waits — the peek
                                # doesn't drain it, so kbhit() stays True
                                # until prompt_toolkit reads it.
                                stop_evt.wait(0.25)
                                continue
                        else:
                            # Fallback: legacy consume-everything behavior.
                            ch = msvcrt.getwch()  # type: ignore[attr-defined]
                            if ch == "\x1b":
                                _fire_esc()
                                return
                except Exception:
                    return
                stop_evt.wait(0.1)

        thread = threading.Thread(target=_watch, name="aura-esc-watchdog", daemon=True)
        thread.start()

        def _stop() -> None:
            stop_evt.set()
            thread.join(timeout=0.2)
        return _stop, esc_evt

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
        # Reuse the cached ctx_used / cost_usd computed inside run_agent for this
        # same turn. Both are expensive (estimate_messages_tokens walks the full
        # history; get_session_stats is a brain call). The cache is marked valid
        # only when run_agent populated it this turn — fall back to fresh compute
        # otherwise (defensive; normal flow always hits the cached path).
        if getattr(self._session, "_last_turn_cached", False):
            self._session.token_used = getattr(self._session, "_last_turn_ctx_used", 0)
            cost_usd = getattr(self._session, "_last_turn_cost_usd", 0.0)
            self._session._last_turn_cached = False
        else:
            self._session.token_used = estimate_messages_tokens(
                self._session.agentic._conversation_history
            )
            cost_usd = 0.0
            try:
                stats = self._session.agent.brain.get_session_stats()
                cost_usd = stats.get("cost_usd", 0.0)
            except (AttributeError, TypeError, KeyError):
                logger.debug("session_stats_read_failed", exc_info=True)
        self._session.token_limit = get_context_limit(self._session.current_model)

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
            self._run_auto_test_async()

    def _run_auto_test_async(self) -> None:
        """Kick off auto-test on the bg_pool and defer the injection.

        Prior version polled `future.result(timeout=0.5)` in a loop up to
        180s INSIDE the event-callback thread that owns streamer rendering.
        That blocked all stream updates for the duration of the test (can
        be minutes), made Esc cancellation unreliable, and mutated
        ``_conversation_history`` mid-turn while ``agentic.run()`` was
        iterating it (data race).

        New behavior: fire-and-forget the future, register an
        ``add_done_callback`` that enqueues the failure into the session's
        ``pending_injections``. The next ``run_agent`` call drains the
        queue and prepends the failure context to the next user prompt.
        Net effect: auto-test failures show up at the START of the NEXT
        turn instead of mid-turn — coherent and race-free.
        """
        try:
            from aura.pools import bg_pool
        except Exception:
            # Fallback to old sync behavior if pool infra is missing.
            # Same race as before, but at least the test still runs.
            logger.debug("auto_test_bg_pool_unavailable", exc_info=True)
            try:
                test_result = self._session.agentic._run_auto_test()
                if test_result:
                    self._enqueue_auto_test_result(test_result)
            except Exception:
                logger.debug("Failed to run auto-test after file edit", exc_info=True)
            return

        future = bg_pool().submit(self._session.agentic._run_auto_test)

        def _on_auto_test_done(fut) -> None:
            try:
                test_result = fut.result(timeout=0)
            except Exception as exc:
                logger.debug("auto_test_failed", exc_info=True)
                test_result = f"(auto-test runner failed: {type(exc).__name__})"
            if test_result:
                self._enqueue_auto_test_result(test_result)

        future.add_done_callback(_on_auto_test_done)

    def _enqueue_auto_test_result(self, test_result: str) -> None:
        """Queue an auto-test failure for prepending to the next user turn."""
        pending = getattr(self._session, "pending_injections", None)
        if pending is None:
            # Defensive: if the session predates the queue attribute, fall
            # back to logging — better than crashing.
            logger.warning("pending_injections missing; auto-test result dropped: %s", test_result[:200])
            return
        pending.append(f"[Auto-test failed after editing]\n{test_result}")

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
