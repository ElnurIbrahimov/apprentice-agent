"""Status and diagnostics helpers for the interactive chat session."""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def show_startup_diagnostics(console: Any) -> None:
    """Show quick warnings if Ollama or cloud key are missing.

    The Ollama reachability HEAD probe runs in a background daemon
    thread so a slow or unreachable host doesn't stall the interactive
    prompt by up to 2s on every startup. The warning still prints when
    the probe fails, just asynchronously.
    """
    import os as _os
    import threading as _threading

    if not _os.environ.get("OLLAMA_API_KEY"):
        console.print(
            "  [yellow]⚠ OLLAMA_API_KEY not set — cloud models unavailable. "
            "Set it in .env[/yellow]"
        )

    def _probe() -> None:
        try:
            import urllib.request

            host = _os.environ.get("OLLAMA_HOST", "http://localhost:11434")
            req = urllib.request.Request(host, method="HEAD")
            urllib.request.urlopen(req, timeout=0.3)
        except Exception:
            try:
                console.print(
                    "  [yellow]⚠ Ollama not running — start with: "
                    "ollama serve[/yellow]"
                )
            except Exception:
                logger.debug("startup_ollama_probe_print_failed", exc_info=True)

    _threading.Thread(target=_probe, daemon=True, name="aura-ollama-probe").start()


class SessionStatusController:
    """Owns CLI status bar rendering and live indicators."""

    def __init__(
        self,
        *,
        console: Any,
        cli_ctx: Any,
        steering_queue: Any,
        create_background_indicator: Any,
        create_research_indicator: Any,
        create_mood_indicator: Any,
    ) -> None:
        self.console = console
        self._cli_ctx = cli_ctx
        self._steering = steering_queue
        self._create_background_indicator = create_background_indicator
        self._create_research_indicator = create_research_indicator
        self._create_mood_indicator = create_mood_indicator
        self._mood_cache: dict[str, Any] = {"state": {}, "ts": 0.0}

    def show_permission_banner(self, mode: str) -> None:
        from .permissions_ui import get_mode_indicator

        self.console.print(f"  {get_mode_indicator(mode)}")
        self.console.print()

    def show_bar(self, **kwargs: Any) -> None:
        from .display import show_status_bar

        # mood_indicator was previously computed and passed through, but the
        # status bar stopped rendering it. Dropped from the tuple now that
        # build_status_bar no longer accepts the kwarg.
        bg_ind, res_ind, _mood_unused, watch_ind = self._phase3_indicators()
        show_status_bar(
            bg_indicator=bg_ind,
            research_indicator=res_ind,
            watch_indicator=watch_ind,
            steering_queue=self._steering,
            **kwargs,
        )

    def _phase3_indicators(self) -> tuple[str, str, str, str]:
        import time as _t

        background_indicator = (
            self._create_background_indicator(self._cli_ctx.bg_manager)
            if self._cli_ctx.bg_manager
            else ""
        )
        research_indicator = (
            self._create_research_indicator(self._cli_ctx.research_ctx)
            if self._cli_ctx.research_ctx
            else ""
        )
        mood_indicator = ""
        now = _t.time()
        if now - self._mood_cache["ts"] > 5.0:
            try:
                from aura.emotion.alma_engine import get_alma_engine

                engine = get_alma_engine()
                emotional_state = engine.get_emotional_state() if engine else {}
                self._mood_cache["state"] = emotional_state
                self._mood_cache["ts"] = now
            except Exception:
                logger.debug("mood_cache_update_failed", exc_info=True)
        if self._mood_cache["state"]:
            mood_indicator = self._create_mood_indicator(self._mood_cache["state"])
        watch_indicator = ""
        if self._cli_ctx.file_watcher:
            from .watch_mode import create_watch_indicator

            watch_indicator = create_watch_indicator(self._cli_ctx.file_watcher)
        return background_indicator, research_indicator, mood_indicator, watch_indicator
