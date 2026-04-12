"""Status and diagnostics helpers for the interactive chat session."""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def show_startup_diagnostics(console: Any) -> None:
    """Show quick warnings if Ollama or cloud key are missing."""
    import os as _os

    if not _os.environ.get("OLLAMA_API_KEY"):
        console.print(
            "  [yellow]\u26a0 OLLAMA_API_KEY not set \u2014 cloud models unavailable. "
            "Set it in .env[/yellow]"
        )

    # Quick Ollama reachability check (2s timeout)
    try:
        import urllib.request

        host = _os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        req = urllib.request.Request(host, method="HEAD")
        urllib.request.urlopen(req, timeout=2)
    except Exception:
        console.print(
            "  [yellow]\u26a0 Ollama not running \u2014 start with: "
            "ollama serve[/yellow]"
        )


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

        bg_ind, res_ind, mood_ind, watch_ind = self._phase3_indicators()
        show_status_bar(
            bg_indicator=bg_ind,
            research_indicator=res_ind,
            mood_indicator=mood_ind,
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
