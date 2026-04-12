"""Application lifespan and background runtime helpers for the AURA API."""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from api.services.websocket_hub import websocket_hub
from api.utils import get_agent_service

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from fastapi import FastAPI


def _schedule_on_loop(loop: asyncio.AbstractEventLoop, coroutine) -> None:
    """Schedule a coroutine from any thread onto the API event loop."""

    if loop.is_closed():
        return
    loop.call_soon_threadsafe(loop.create_task, coroutine)


async def _wait_for_agent_ready(logger: logging.Logger) -> None:
    """Wait briefly for background agent initialization to finish."""

    for _ in range(30):
        await asyncio.sleep(2)
        try:
            if get_agent_service().is_ready:
                return
        except Exception:
            logger.debug("agent_ready_check_failed", exc_info=True)

    logger.warning("[API] Agent not ready after 60s, starting proactive system anyway")


async def _start_proactive_runtime(app: "FastAPI", logger: logging.Logger) -> None:
    """Start long-lived background services after the app boots."""

    await _wait_for_agent_ready(logger)

    daemon = None
    try:
        from aura.proactive.gateway_daemon import get_gateway_daemon
        from aura.proactive.monitors.system_monitor import SystemMonitor

        daemon = get_gateway_daemon()
        proactive_loop = asyncio.get_running_loop()

        def _on_proactive_message(message) -> None:
            logger.info("[Proactive] %s: %s...", message.action.value, message.content[:80])
            try:
                _schedule_on_loop(
                    proactive_loop,
                    websocket_hub.broadcast_proactive_message(message),
                )
            except Exception as exc:
                logger.debug("[Proactive] WebSocket push failed: %s", exc)

        daemon.set_notification_callback(_on_proactive_message)
        await daemon.start()

        try:
            from aura.consciousness.intrinsic_motivation import get_intrinsic_motivation

            loop = asyncio.get_running_loop()
            intrinsic_motivation = get_intrinsic_motivation()
            await loop.run_in_executor(None, intrinsic_motivation.run_motivation_cycle)
            logger.info("[API] Intrinsic Motivation engine: first cycle complete")
        except Exception as exc:
            logger.warning("[API] Motivation engine startup cycle failed: %s", exc)

        system_monitor = SystemMonitor(event_bus=daemon.event_bus, poll_interval=60.0)
        await system_monitor.start()

        is_headless = bool(os.environ.get("AURA_HEADLESS")) or (
            sys.platform != "win32" and not os.environ.get("DISPLAY")
        )

        screen_monitor = None
        if is_headless:
            logger.info("[API] ScreenMonitor skipped (headless mode)")
        else:
            try:
                from aura.proactive.monitors.screen_monitor import ScreenMonitor

                screen_monitor = ScreenMonitor(
                    event_bus=daemon.event_bus,
                    poll_interval=10.0,
                )
                await screen_monitor.start()
                logger.info("[API] ScreenMonitor started")
            except Exception as exc:
                logger.warning("[API] ScreenMonitor failed to start: %s", exc)

        calendar_monitor = None
        try:
            from aura.proactive.monitors.calendar_monitor import get_calendar_monitor

            calendar_monitor = get_calendar_monitor(event_bus=daemon.event_bus)
            await calendar_monitor.start()
            logger.info("[API] CalendarMonitor started")
        except Exception as exc:
            logger.warning("[API] CalendarMonitor failed to start: %s", exc)

        workflow_detector = None
        if is_headless:
            logger.info("[API] WorkflowDetector skipped (headless mode)")
        else:
            try:
                from aura.proactive.monitors.workflow_detector import get_workflow_detector

                workflow_detector = get_workflow_detector(event_bus=daemon.event_bus)
                await workflow_detector.start()
                logger.info("[API] WorkflowDetector started")
            except Exception as exc:
                logger.warning("[API] WorkflowDetector failed to start: %s", exc)

        app.state.proactive_daemon = daemon
        app.state.system_monitor = system_monitor
        app.state.screen_monitor = screen_monitor
        app.state.calendar_monitor = calendar_monitor
        app.state.workflow_detector = workflow_detector

        active_components = ["GatewayDaemon", "SystemMonitor", "CalendarMonitor"]
        if screen_monitor:
            active_components.append("ScreenMonitor")
        if workflow_detector:
            active_components.append("WorkflowDetector")

        logger.info(
            "[API] Proactive system started (%s)%s",
            " + ".join(active_components),
            " [headless]" if is_headless else "",
        )
        logger.info("[API] SQLite persistence active for proactive subsystem")
    except Exception as exc:
        logger.warning("[API] Proactive system failed to start: %s", exc)

    try:
        from aura.services.voice_presence import get_voice_presence

        voice_service = get_voice_presence()
        voice_service.start()
        app.state.voice_presence = voice_service
        logger.info("[API] VoicePresenceService started (kokoro)")
    except Exception as exc:
        logger.warning("[API] VoicePresenceService failed to start: %s", exc)
        app.state.voice_presence = None

    try:
        from api.routes.idle_behaviors import init_idle_presence

        init_idle_presence()
    except Exception as exc:
        logger.warning("[API] Idle presence init failed: %s", exc)

    try:
        from aura.consciousness.self_improvement import get_self_improvement_engine

        get_self_improvement_engine().start()
        logger.info("[API] Self-Improvement Engine started")
    except Exception as exc:
        logger.warning("[API] Self-Improvement Engine failed to start: %s", exc)

    try:
        from aura.hands.manager import get_hand_manager

        hands_manager = get_hand_manager()
        hands_loop = asyncio.get_running_loop()
        hands_manager.set_event_loop(hands_loop)

        def _on_hand_result(result) -> None:
            try:
                _schedule_on_loop(
                    hands_loop,
                    websocket_hub.broadcast_hand_event(result.to_dict()),
                )
            except Exception as exc:
                logger.debug("[API] Hand WS broadcast failed: %s", exc)

            try:
                from aura.messaging.telegram_bot import notify_hand_result

                notify_hand_result(result)
            except Exception:
                logger.debug("[API] Telegram hand notification failed", exc_info=True)

        hands_manager.set_notify_callback(_on_hand_result)
        logger.info("[API] HandManager notification callback wired (with approval loop)")

        try:
            if daemon is not None and hasattr(daemon, "event_bus") and daemon.event_bus:
                hands_manager.set_event_bus(daemon.event_bus)
                logger.info("[API] HandManager event bus wired")
        except Exception as exc:
            logger.debug("[API] Hand event bus wiring failed: %s", exc)
    except Exception as exc:
        logger.debug("[API] HandManager callback wiring failed: %s", exc)


async def _cancel_startup_task(app: "FastAPI", logger: logging.Logger) -> None:
    """Cancel the deferred startup task if it is still running."""

    try:
        task = getattr(app.state, "proactive_startup_task", None)
        if task and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            logger.info("[API] Proactive startup task cancelled")
    except Exception as exc:
        logger.warning("[API] Error cancelling startup task: %s", exc)


async def _stop_proactive_runtime(app: "FastAPI", logger: logging.Logger) -> None:
    """Stop proactive monitors and background services."""

    try:
        for attr_name in (
            "proactive_daemon",
            "system_monitor",
            "screen_monitor",
            "calendar_monitor",
            "workflow_detector",
        ):
            component = getattr(app.state, attr_name, None)
            if component is not None:
                await component.stop()
        logger.info("[API] Proactive system stopped")
    except Exception as exc:
        logger.warning("[API] Proactive shutdown error: %s", exc)


def _stop_voice_presence(app: "FastAPI", logger: logging.Logger) -> None:
    """Stop voice presence if it was started."""

    try:
        voice_presence = getattr(app.state, "voice_presence", None)
        if voice_presence is not None:
            voice_presence.stop()
            logger.info("[API] VoicePresenceService stopped")
    except Exception as exc:
        logger.warning("[API] Voice shutdown error: %s", exc)


def _stop_self_improvement(logger: logging.Logger) -> None:
    """Stop the self-improvement engine."""

    try:
        from aura.consciousness.self_improvement import get_self_improvement_engine

        get_self_improvement_engine().stop()
        logger.info("[API] Self-Improvement Engine stopped")
    except Exception as exc:
        logger.warning("[API] Self-Improvement Engine stop failed: %s", exc)


def _stop_idle_presence(logger: logging.Logger) -> None:
    """Stop idle presence background tasks."""

    try:
        from aura.consciousness.idle_presence import get_idle_presence_engine

        get_idle_presence_engine().stop_background_tasks()
        logger.info("[API] Idle Presence Engine stopped")
    except Exception as exc:
        logger.warning("[API] Idle Presence Engine stop failed: %s", exc)


def _close_proactive_persistence(logger: logging.Logger) -> None:
    """Close the proactive persistence database."""

    try:
        from aura.proactive.persistence import get_persistence

        get_persistence().close()
        logger.info("[API] Proactive persistence closed")
    except Exception as exc:
        logger.warning("[API] Persistence shutdown error: %s", exc)


def create_lifespan(logger: logging.Logger):
    """Build the FastAPI lifespan handler with shared startup/shutdown logic."""

    @asynccontextmanager
    async def lifespan(app: "FastAPI") -> AsyncIterator[None]:
        logger.info("[API] Starting AURA Web API...")

        loop = asyncio.get_running_loop()
        loop.set_default_executor(ThreadPoolExecutor(max_workers=20))
        logger.info("[API] Thread pool set to 20 workers")

        try:
            get_agent_service().start_background_init()
            logger.info("[API] Agent initialization started in background")
        except Exception as exc:
            logger.error("[API] Agent initialization failed: %s", exc)
            logger.warning(
                "[API] Server running without agent - install missing dependencies",
            )

        app.state.proactive_startup_task = asyncio.create_task(
            _start_proactive_runtime(app, logger),
        )

        yield

        logger.info("[API] Shutting down AURA Web API...")
        await _cancel_startup_task(app, logger)
        await _stop_proactive_runtime(app, logger)
        _stop_voice_presence(app, logger)
        _stop_self_improvement(logger)
        _stop_idle_presence(logger)
        _close_proactive_persistence(logger)

    return lifespan
