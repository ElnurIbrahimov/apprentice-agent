"""
Screen Monitor - Monitors screen content and application changes.

Uses Screenpipe when available for rich screen context,
falls back to basic window tracking otherwise.

Events generated:
- app_switch: User switched to a different application
- window_change: Active window title changed
- content_detected: Relevant content detected on screen
- idle_detected: User appears to be idle
"""

import asyncio
import logging
import time
from datetime import datetime
from typing import List, Optional, Dict, Any

from .base_monitor import BaseMonitor
from ..event_bus import Event, EventPriority

logger = logging.getLogger(__name__)

# Try to import screenpipe client
try:
    from apprentice_agent.tools.screenpipe import ScreenpipeClient
    SCREENPIPE_AVAILABLE = True
except ImportError:
    SCREENPIPE_AVAILABLE = False


class ScreenMonitor(BaseMonitor):
    """
    Monitor for screen content and application changes.

    Features:
    - Tracks active application and window
    - Detects app switches
    - Integrates with Screenpipe for OCR content
    - Detects user idle state
    """

    def __init__(
        self,
        event_bus=None,
        poll_interval: float = 3.0,
        idle_threshold: float = 60.0,  # Seconds before considered idle
        use_screenpipe: bool = True
    ):
        """
        Initialize screen monitor.

        Args:
            event_bus: EventBus to publish to
            poll_interval: Seconds between polls
            idle_threshold: Seconds of inactivity before idle event
            use_screenpipe: Whether to use Screenpipe integration
        """
        super().__init__(event_bus, poll_interval)

        self._idle_threshold = idle_threshold
        self._use_screenpipe = use_screenpipe and SCREENPIPE_AVAILABLE

        # State tracking
        self._last_app: Optional[str] = None
        self._last_window: Optional[str] = None
        self._last_activity: datetime = datetime.now()
        self._is_idle: bool = False
        self._screenpipe: Optional['ScreenpipeClient'] = None

        # Content keywords to watch for
        self._watch_keywords: List[str] = []

        logger.info(f"[ScreenMonitor] Initialized (screenpipe={self._use_screenpipe})")

    @property
    def source(self) -> str:
        return "screen"

    def set_watch_keywords(self, keywords: List[str]) -> None:
        """
        Set keywords to watch for in screen content.

        Args:
            keywords: Keywords to detect
        """
        self._watch_keywords = [kw.lower() for kw in keywords]
        logger.debug(f"[ScreenMonitor] Watching for: {self._watch_keywords}")

    async def _on_start(self) -> None:
        """Initialize Screenpipe if available."""
        if self._use_screenpipe:
            try:
                self._screenpipe = ScreenpipeClient()
                # Test connection
                status = self._screenpipe.health_check()
                if not status.get("healthy"):
                    logger.warning("[ScreenMonitor] Screenpipe not healthy, falling back")
                    self._screenpipe = None
                    self._use_screenpipe = False
                else:
                    logger.info("[ScreenMonitor] Screenpipe connected")
            except Exception as e:
                logger.warning(f"[ScreenMonitor] Screenpipe init failed: {e}")
                self._screenpipe = None
                self._use_screenpipe = False

    async def _on_stop(self) -> None:
        """Cleanup Screenpipe connection."""
        self._screenpipe = None

    async def _poll(self) -> List[Event]:
        """Poll for screen events."""
        events = []

        # Get current window info
        current_app, current_window = await self._get_active_window()

        # Check for app switch
        if current_app and current_app != self._last_app:
            events.append(self._create_app_switch_event(
                self._last_app,
                current_app,
                current_window
            ))
            self._last_app = current_app
            self._last_activity = datetime.now()

        # Check for window change (within same app)
        elif current_window and current_window != self._last_window:
            events.append(self._create_window_change_event(
                current_app,
                self._last_window,
                current_window
            ))
            self._last_window = current_window
            self._last_activity = datetime.now()

        # Update last window
        self._last_window = current_window

        # Check for content keywords (if Screenpipe available)
        if self._screenpipe and self._watch_keywords:
            content_events = await self._check_screen_content()
            events.extend(content_events)

        # Check for idle state
        idle_event = self._check_idle()
        if idle_event:
            events.append(idle_event)

        return events

    async def _get_active_window(self) -> tuple[Optional[str], Optional[str]]:
        """
        Get the currently active window.

        Returns:
            (app_name, window_title) or (None, None) if unavailable
        """
        # Try Screenpipe first
        if self._screenpipe:
            try:
                # Query recent screen content
                results = self._screenpipe.search(
                    limit=1,
                    content_type="ocr"
                )
                if results and len(results) > 0:
                    latest = results[0]
                    app = latest.get("app_name", "Unknown")
                    title = latest.get("window_name", "")
                    return (app, title)
            except Exception as e:
                logger.debug(f"[ScreenMonitor] Screenpipe query failed: {e}")

        # Fallback: Try platform-specific APIs
        try:
            return await self._get_active_window_native()
        except Exception as e:
            logger.debug(f"[ScreenMonitor] Native window query failed: {e}")

        return (None, None)

    async def _get_active_window_native(self) -> tuple[Optional[str], Optional[str]]:
        """
        Get active window using native platform APIs.

        Returns:
            (app_name, window_title)
        """
        import sys

        if sys.platform == "win32":
            try:
                import ctypes
                from ctypes import wintypes

                user32 = ctypes.windll.user32

                # Get foreground window
                hwnd = user32.GetForegroundWindow()
                if not hwnd:
                    return (None, None)

                # Get window title
                length = user32.GetWindowTextLengthW(hwnd) + 1
                title = ctypes.create_unicode_buffer(length)
                user32.GetWindowTextW(hwnd, title, length)

                # Get process name
                pid = wintypes.DWORD()
                user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))

                import psutil
                try:
                    process = psutil.Process(pid.value)
                    app_name = process.name().replace(".exe", "")
                except:
                    app_name = "Unknown"

                return (app_name, title.value)

            except ImportError:
                pass

        elif sys.platform == "darwin":
            try:
                import subprocess
                script = '''
                tell application "System Events"
                    set frontApp to name of first application process whose frontmost is true
                    set frontWindow to ""
                    try
                        set frontWindow to name of front window of first application process whose frontmost is true
                    end try
                end tell
                return frontApp & "|" & frontWindow
                '''
                result = subprocess.run(
                    ["osascript", "-e", script],
                    capture_output=True,
                    text=True
                )
                if result.returncode == 0:
                    parts = result.stdout.strip().split("|")
                    return (parts[0], parts[1] if len(parts) > 1 else "")
            except Exception:
                pass

        elif sys.platform.startswith("linux"):
            try:
                import subprocess
                # Try xdotool
                result = subprocess.run(
                    ["xdotool", "getactivewindow", "getwindowname"],
                    capture_output=True,
                    text=True
                )
                if result.returncode == 0:
                    title = result.stdout.strip()
                    # Try to get app name from WM_CLASS
                    wid = subprocess.run(
                        ["xdotool", "getactivewindow"],
                        capture_output=True,
                        text=True
                    ).stdout.strip()
                    class_result = subprocess.run(
                        ["xprop", "-id", wid, "WM_CLASS"],
                        capture_output=True,
                        text=True
                    )
                    app_name = "Unknown"
                    if class_result.returncode == 0:
                        # Parse WM_CLASS
                        parts = class_result.stdout.split('"')
                        if len(parts) >= 4:
                            app_name = parts[3]
                    return (app_name, title)
            except Exception:
                pass

        return (None, None)

    async def _check_screen_content(self) -> List[Event]:
        """
        Check screen content for watched keywords.

        Returns:
            List of content_detected events
        """
        events = []

        if not self._screenpipe or not self._watch_keywords:
            return events

        try:
            # Search for each keyword
            for keyword in self._watch_keywords[:5]:  # Limit to avoid spam
                results = self._screenpipe.search(
                    query=keyword,
                    limit=1,
                    content_type="ocr"
                )
                if results:
                    result = results[0]
                    # Only trigger if recent (within last poll interval)
                    timestamp = result.get("timestamp")
                    if timestamp:
                        # Check if this is a new detection
                        events.append(self.create_event(
                            "content_detected",
                            {
                                "keyword": keyword,
                                "app_name": result.get("app_name"),
                                "window_name": result.get("window_name"),
                                "text_preview": result.get("text", "")[:200]
                            },
                            priority=EventPriority.LOW
                        ))
        except Exception as e:
            logger.debug(f"[ScreenMonitor] Content check failed: {e}")

        return events

    def _check_idle(self) -> Optional[Event]:
        """
        Check if user is idle.

        Returns:
            idle_detected event if newly idle, None otherwise
        """
        seconds_since_activity = (datetime.now() - self._last_activity).total_seconds()

        if seconds_since_activity >= self._idle_threshold:
            if not self._is_idle:
                self._is_idle = True
                return self.create_event(
                    "idle_detected",
                    {
                        "idle_seconds": seconds_since_activity,
                        "last_app": self._last_app,
                        "last_window": self._last_window
                    },
                    priority=EventPriority.LOW
                )
        else:
            self._is_idle = False

        return None

    def _create_app_switch_event(
        self,
        from_app: Optional[str],
        to_app: str,
        window_title: Optional[str]
    ) -> Event:
        """Create app_switch event."""
        return self.create_event(
            "app_switch",
            {
                "from_app": from_app,
                "to_app": to_app,
                "window_title": window_title
            },
            priority=EventPriority.LOW
        )

    def _create_window_change_event(
        self,
        app_name: Optional[str],
        from_window: Optional[str],
        to_window: str
    ) -> Event:
        """Create window_change event."""
        return self.create_event(
            "window_change",
            {
                "app_name": app_name,
                "from_window": from_window,
                "to_window": to_window
            },
            priority=EventPriority.BACKGROUND
        )

    def record_activity(self) -> None:
        """Record user activity (call this on keyboard/mouse input)."""
        self._last_activity = datetime.now()
        self._is_idle = False
