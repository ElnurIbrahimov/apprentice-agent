"""Keyboard signal handling for the interactive chat session."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
import os


@dataclass
class SignalHandlingResult:
    """Outcome of handling a prompt-toolkit pseudo-input signal."""

    should_continue_loop: bool
    injected_input: Optional[str] = None


class SessionSignalController:
    """Owns keyboard signal handling for ChatSession."""

    def __init__(self, session: Any) -> None:
        self._session = session

    def handle(self, user_input: str) -> Optional[SignalHandlingResult]:
        """Handle a signal and return its effect on the main loop."""
        from .chat_loop import _rewind_picker
        from .context_bar import get_context_limit
        from .display import show_info
        from .input import (
            SIGNAL_CLEAR_SCREEN,
            SIGNAL_COMMAND_PALETTE,
            SIGNAL_CYCLE_PERMS,
            SIGNAL_MODEL_PICK,
            SIGNAL_NEW_SESSION,
            SIGNAL_OPEN_EDITOR,
            SIGNAL_REWIND,
        )
        from .permissions_ui import cycle_permission_mode, get_mode_description

        if user_input == SIGNAL_CLEAR_SCREEN:
            self._session.console.clear()
            self._refresh_bar()
            return SignalHandlingResult(should_continue_loop=True)

        if user_input == SIGNAL_NEW_SESSION:
            if hasattr(self._session.agentic, "session") and self._session.agentic.session:
                self._session.agentic.session.save()
            self._session.agentic._conversation_history.clear()
            if self._session.checkpoint_mgr:
                self._session.checkpoint_mgr.clear()
            self._session.msg_count = 0
            self._session.token_used = 0
            self._session.console.print("[dim]New session started[/dim]")
            self._refresh_bar()
            return SignalHandlingResult(should_continue_loop=True)

        if user_input == SIGNAL_COMMAND_PALETTE:
            from .command_palette import build_palette, open_palette, record_usage
            from .input import SLASH_COMMANDS as _palette_cmds

            items = build_palette(_palette_cmds)
            selected = open_palette(items, self._session.console)
            if selected:
                record_usage(selected)
                return SignalHandlingResult(
                    should_continue_loop=False,
                    injected_input=selected,
                )
            return SignalHandlingResult(should_continue_loop=True)

        if user_input == SIGNAL_OPEN_EDITOR:
            import subprocess as _sp
            import tempfile

            editor = os.environ.get("EDITOR", "notepad" if os.name == "nt" else "nano")
            with tempfile.NamedTemporaryFile(suffix=".md", delete=False, mode="w") as f:
                f.write("")
                tmp_path = f.name
            try:
                _sp.call([editor, tmp_path])
                # Force UTF-8 — the default encoding follows the locale, so
                # on a non-UTF-8 Windows console (e.g., cp1252) any non-ASCII
                # editor content (Azerbaijani, accented chars, smart quotes)
                # raises UnicodeDecodeError mid-session.
                edited = Path(tmp_path).read_text(
                    encoding="utf-8", errors="replace"
                ).strip()
            except (FileNotFoundError, OSError, UnicodeDecodeError) as exc:
                self._session.console.print(f"[red]Editor failed: {exc}[/red]")
                edited = ""
            finally:
                # Windows: editor may still hold the file → PermissionError.
                # Don't crash the session; log for cleanup at next startup.
                try:
                    Path(tmp_path).unlink(missing_ok=True)
                except (PermissionError, OSError):
                    try:
                        from aura.cli.hooks import logger as _log
                    except Exception:
                        import logging as _l
                        _log = _l.getLogger("aura.cli.chat_session_signals")
                    _log.debug("editor_tmp_unlink_failed path=%s", tmp_path, exc_info=True)
                    # Prior versions appended the unlink-failed path to
                    # ~/.aura/tmp_cleanup.txt for a "next startup sweep" that
                    # was never implemented. The file grew monotonically with
                    # no consumer. Orphan editor temp files are rare and
                    # small, so we just accept them silently.
            if not edited:
                return SignalHandlingResult(should_continue_loop=True)
            return SignalHandlingResult(
                should_continue_loop=False,
                injected_input=edited,
            )

        if user_input == SIGNAL_CYCLE_PERMS:
            self._session.perm_mode = cycle_permission_mode(self._session.perm_mode)
            self._session.console.print(
                f"[dim]{get_mode_description(self._session.perm_mode)}[/dim]"
            )
            self._session._show_perm_banner(self._session.perm_mode)
            self._session.permissions.set_mode(self._session.perm_mode)
            self._refresh_bar()
            return SignalHandlingResult(should_continue_loop=True)

        if user_input == SIGNAL_REWIND:
            if self._session.checkpoint_mgr:
                _rewind_picker(self._session.checkpoint_mgr, self._session.console)
            else:
                self._session.console.print("[dim]No checkpoint manager available[/dim]")
            return SignalHandlingResult(should_continue_loop=True)

        if user_input == SIGNAL_MODEL_PICK:
            from .model_picker import pick_model

            # current_model is authoritative (maintained by apply_model_override).
            choice = pick_model(self._session.console, self._session.current_model or "auto")
            if choice:
                self._session.apply_model_override(choice)
                if choice == "auto":
                    show_info("Model set to auto-routing")
                else:
                    show_info(f"Model set to {choice}")
            self._session.token_limit = get_context_limit(self._session.current_model)
            self._refresh_bar()
            return SignalHandlingResult(should_continue_loop=True)

        return None

    def _refresh_bar(self) -> None:
        self._session._show_bar(
            model=self._session.current_model,
            project_type=self._session._project_type,
            session_title=self._session.session_title,
            message_count=self._session.msg_count,
            token_used=self._session.token_used,
            token_limit=self._session.token_limit,
            permission_mode=self._session.perm_mode,
        )
