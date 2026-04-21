"""Lifecycle and runtime loop helpers for the interactive chat session."""
from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

from ._constants import ERROR_SENTINELS as _ERROR_SENTINELS


class SessionRuntimeController:
    """Owns the main runtime loop and adjacent lifecycle behaviors."""

    def __init__(self, session: Any) -> None:
        self._session = session

    def drain_channels(self) -> None:
        """Process at most one pending bridged message without blocking the UI."""
        if not self._session.bridge or not self._session.bridge.has_pending():
            return
        if not self._session._channel_lock.acquire(blocking=False):
            return

        ch_msg = self._session.bridge.get_pending_message(timeout=0)
        if ch_msg is None:
            self._session._channel_lock.release()
            return

        def _process() -> None:
            try:
                from .chat_loop import _display_channel_response

                result = self._session.agentic.run(ch_msg.text)
                response_text = result.get("response", "") if result else ""
            except Exception as exc:
                logger.debug("channel_agent_run_failed", exc_info=True)
                response_text = f"Error processing message: {exc}"
            try:
                if response_text:
                    _display_channel_response(self._session.console, ch_msg, response_text)
                    self._session.bridge.send_response(ch_msg, response_text)
            except Exception:
                logger.debug("channel_response_display_failed", exc_info=True)
            finally:
                self._session._channel_lock.release()

        import threading

        t = threading.Thread(target=_process, daemon=True, name="channel-drain")
        try:
            t.start()
        except RuntimeError:
            # Thread creation failed (OS resource exhaustion) — release lock manually
            # so subsequent drain_channels calls are not permanently blocked.
            self._session._channel_lock.release()

    def submit_background(self, user_input: str) -> None:
        """Handle the '&' prefix for background task submission.

        Routes the prompt through a sibling AgenticLoop with its own
        PermissionManager in 'careful' mode and no confirm callback — so
        any tool that would normally prompt the user (writes, shell) is
        denied without blocking the bg thread. Reads and safe shell
        commands still work. See AgenticLoop.clone_for_background().
        """
        bg_prompt = user_input[2:].strip() if user_input.startswith("& ") else user_input[1:].strip()
        if not bg_prompt:
            self._session.console.print("[dim]Usage: & <prompt>[/dim]")
            return

        def _bg_task_fn(prompt: str) -> dict[str, Any]:
            try:
                from aura.core.permissions import PermissionManager

                bg_perms = PermissionManager()
                bg_perms.set_mode("careful")
                # No confirm callback — any PROMPT-tier tool silently denies.
                bg_loop = self._session.agentic.clone_for_background(bg_perms)
                result = bg_loop.run(prompt)
                return result or {"success": False, "error": "no result"}
            except Exception as exc:
                logger.debug("background_task_failed", exc_info=True)
                return {"success": False, "error": str(exc)}

        if not self._session.bg_manager:
            self._session.console.print("[red]Background tasks are not available.[/red]")
            return
        task = self._session.bg_manager.submit(bg_prompt, _bg_task_fn)
        if task:
            self._session.console.print(f"[cyan]Background task started: {task.id}[/cyan]")
        else:
            self._session.console.print("[red]Too many background tasks running.[/red]")

    def save_session_if_initialized(self) -> None:
        if self._session._session_initialized:
            self._session.agentic_session.save()

    def run(self) -> None:
        """Run the interactive input loop."""
        from .context_bar import estimate_messages_tokens, get_context_limit
        from .display import show_error, show_help, show_info, show_response
        from .input import (
            SIGNAL_CLEAR_SCREEN,
            SIGNAL_COMMAND_PALETTE,
            SIGNAL_CYCLE_PERMS,
            SIGNAL_MODEL_PICK,
            SIGNAL_NEW_SESSION,
            SIGNAL_OPEN_EDITOR,
            SIGNAL_REWIND,
            get_input,
        )
        from .permissions_ui import is_plan_approve_mode

        all_signals = {
            SIGNAL_CLEAR_SCREEN,
            SIGNAL_NEW_SESSION,
            SIGNAL_COMMAND_PALETTE,
            SIGNAL_OPEN_EDITOR,
            SIGNAL_REWIND,
            SIGNAL_CYCLE_PERMS,
            SIGNAL_MODEL_PICK,
        }

        while True:
            self.drain_channels()

            if self._session._pending_follow_up:
                user_input = self._session._pending_follow_up
                self._session._pending_follow_up = None
                show_info(f"Follow-up: {user_input[:60]}...")
            else:
                self._session._follow_up_depth = 0
                user_input = get_input(self._session._pt_session)

            if user_input is None:
                self._handle_exit()
                break

            if user_input in all_signals:
                self._session._injected_input = None
                should_continue = self._session._handle_signal(user_input)
                if should_continue:
                    continue
                if self._session._injected_input is not None:
                    user_input = self._session._injected_input
                    self._session._injected_input = None
                else:
                    continue

            if not user_input:
                continue

            if user_input.startswith("& ") or (
                user_input.startswith("&")
                and len(user_input) > 1
                and user_input[1] != " "
            ):
                self.submit_background(user_input)
                continue

            self._send_ipc_heartbeat_if_due()

            if user_input.strip() == "?":
                show_help()
                continue

            if user_input.strip() == "/retry":
                if self._session.last_user_input:
                    show_info(f"Retrying: {self._session.last_user_input[:60]}...")
                    self._session._pending_follow_up = self._session.last_user_input
                else:
                    show_error("Nothing to retry — no previous prompt.")
                continue

            if user_input.strip() == "/channels":
                self._show_channels()
                continue

            if user_input.startswith("/"):
                self._session._dispatch_command(user_input)
                continue

            if not self._session._session_initialized:
                self._session.agentic_session.new(
                    project_root=self._session._project_root,
                    model=self._session.current_model or "auto",
                )
                self._session._session_initialized = True

            self._session.last_user_input = user_input

            if is_plan_approve_mode(self._session.perm_mode):
                result = self._session._run_plan_mode(user_input)
                if result is None:
                    continue
                self._handle_plan_result(
                    user_input,
                    result,
                    estimate_messages_tokens,
                    get_context_limit,
                    show_error,
                    show_response,
                )
                continue

            self._session._streamer_displayed = False
            result = self._session._run_agent(user_input)
            if not self._session._process_normal_result(user_input, result):
                continue

    def _handle_exit(self) -> None:
        if self._session.bridge:
            self._session.bridge.stop()
        if self._session.hook_mgr:
            self._session.hook_mgr.fire(
                self._session._HookEvent.SESSION_END,
                {"reason": "user_exit"},
            )
        self._session.console.print("\n[dim]Goodbye.[/dim]\n")

    # IPC heartbeat config — env-driven so tests and non-daemon setups can
    # opt out without code changes. The daemon publishes its token under
    # $AURA_DATA_DIR/ipc_token (default 'data/') per aura_daemon.py:619, and
    # listens on $AURA_DAEMON_PORT (default 19733) or the Windows named pipe
    # \\.\pipe\aura_daemon.
    _IPC_HEARTBEAT_INTERVAL = 30.0
    # Exponential backoff: start short, double on each consecutive failure,
    # cap at 60s. Previously a single failure muted heartbeats for 5 minutes
    # with no reconnect — if the daemon restarted during that window the CLI
    # never picked it back up. 60s keeps the reconnect window tight while
    # still avoiding a per-30s socket storm against a dead daemon.
    _IPC_BACKOFF_INITIAL = 5.0
    _IPC_BACKOFF_MAX = 60.0
    _IPC_PIPE_NAME = r"\\.\pipe\aura_daemon"

    def _send_ipc_heartbeat_if_due(self) -> None:
        import time as _t_ipc

        if os.environ.get("AURA_DAEMON_IPC", "1") == "0":
            return

        now = _t_ipc.time()
        if now - self._session._last_ipc_heartbeat <= self._IPC_HEARTBEAT_INTERVAL:
            return
        if self._session._daemon_unreachable_until > now:
            return
        self._session._last_ipc_heartbeat = now

        if self._try_ipc_send():
            # Success clears the backoff immediately — restores heartbeats on
            # a daemon that just came back up, without waiting for the old
            # 5-minute mute to expire.
            self._session._daemon_unreachable_until = 0.0
            self._session._ipc_backoff_seconds = self._IPC_BACKOFF_INITIAL
        else:
            prev = getattr(self._session, "_ipc_backoff_seconds", self._IPC_BACKOFF_INITIAL)
            backoff = min(self._IPC_BACKOFF_MAX, max(self._IPC_BACKOFF_INITIAL, prev * 2))
            self._session._ipc_backoff_seconds = backoff
            self._session._daemon_unreachable_until = now + backoff
            logger.debug("IPC heartbeat failed; backing off for %.0fs", backoff)

    def _try_ipc_send(self) -> bool:
        """Attempt to deliver one activity ping. Returns True on success."""
        import json as _json

        payload = _json.dumps({"type": "activity", "token": self._read_ipc_token()}) + "\n"
        data = payload.encode()

        # Windows: prefer the named pipe to match the daemon's primary channel.
        if os.name == "nt":
            try:
                # Open-write-close keeps us consistent with a one-shot TCP send.
                with open(self._IPC_PIPE_NAME, "wb", buffering=0) as pipe:
                    pipe.write(data)
                return True
            except OSError:
                pass  # fall back to TCP

        try:
            import socket

            port = int(os.environ.get("AURA_DAEMON_PORT", "19733"))
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(0.1)
                sock.connect(("127.0.0.1", port))
                sock.send(data)
            return True
        except (OSError, ValueError):
            return False

    @staticmethod
    def _read_ipc_token() -> str:
        """Resolve the daemon's auth token via $AURA_DATA_DIR/ipc_token."""
        try:
            data_dir = os.environ.get("AURA_DATA_DIR", "data")
            token_path = os.path.join(data_dir, "ipc_token")
            if os.path.isfile(token_path):
                with open(token_path, encoding="utf-8") as f:
                    return f.read().strip()
        except OSError:
            pass
        return ""

    def _show_channels(self) -> None:
        from .display import show_info

        if not self._session.bridge:
            show_info("No channel bridge active. Start with --channels flag.")
            return

        from rich.table import Table

        ch_table = Table(
            show_header=True,
            header_style="bold cyan",
            border_style="dim",
            padding=(0, 2),
            title="[bold]Active Channels[/bold]",
        )
        ch_table.add_column("Channel", style="cyan", width=16)
        ch_table.add_column("Status", style="white", width=12)
        ch_table.add_column("Pending", style="dim", width=10)
        for st in self._session.bridge.status():
            status_str = "[green]running[/green]" if st["running"] else "[red]stopped[/red]"
            ch_table.add_row(st["channel"], status_str, str(st["pending"]))
        self._session.console.print()
        self._session.console.print(ch_table)
        self._session.console.print()

    def _handle_plan_result(
        self,
        user_input: str,
        result: dict,
        estimate_messages_tokens: Any,
        get_context_limit: Any,
        show_error: Any,
        show_response: Any,
    ) -> None:
        response_text = result.get("response", "")
        model_used = result.get("model", self._session.current_model)
        is_error = result.get("success") is False or any(
            response_text.startswith(s) for s in _ERROR_SENTINELS
        )
        if is_error:
            show_error(response_text)
        else:
            if response_text:
                show_response(response_text, model=model_used, stream=False)

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
                logger.debug("Failed to sync ConversationManager after plan response", exc_info=True)

        self._session.msg_count += 1
        if self._session.msg_count == 1 and user_input:
            self._session.session_title = user_input[:50].strip()
        # current_model is maintained by apply_model_override; no re-read needed.
        self._session.token_used = estimate_messages_tokens(
            self._session.agentic._conversation_history
        )
        self._session.token_limit = get_context_limit(self._session.current_model)
        self._session._show_bar(
            model=self._session.current_model,
            project_type=self._session._project_type,
            session_title=self._session.session_title,
            message_count=self._session.msg_count,
            token_used=self._session.token_used,
            token_limit=self._session.token_limit,
            permission_mode=self._session.perm_mode,
        )
