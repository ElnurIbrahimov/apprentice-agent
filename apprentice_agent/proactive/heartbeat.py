"""
HeartbeatMonitor - Proactive System for AURA v3.0

Background system that:
- Periodically checks for things to notify user about
- Monitors system state
- Generates proactive suggestions
- Manages notification queue

Makes AURA feel "alive" by initiating contextual interactions.
"""

import json
import logging
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Callable, Any
from enum import Enum
from queue import Queue, Empty

logger = logging.getLogger(__name__)


class NotificationPriority(Enum):
    """Priority levels for notifications."""
    LOW = 1       # Nice to know
    MEDIUM = 2    # Should see soon
    HIGH = 3      # Should see now
    URGENT = 4    # Interrupt if needed


@dataclass
class Notification:
    """A proactive notification from AURA."""
    message: str
    priority: NotificationPriority = NotificationPriority.MEDIUM
    category: str = "general"
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    action_hint: Optional[str] = None  # Suggested action
    expires_at: Optional[str] = None   # When this becomes stale
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["priority"] = self.priority.value
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "Notification":
        data["priority"] = NotificationPriority(data.get("priority", 2))
        return cls(**data)

    def is_expired(self) -> bool:
        """Check if notification has expired."""
        if self.expires_at:
            try:
                expiry = datetime.fromisoformat(self.expires_at)
                return datetime.now() > expiry
            except ValueError:
                pass
        return False


@dataclass
class Check:
    """A registered check that runs periodically."""
    name: str
    callback: Callable[[], Optional[Notification]]
    interval_seconds: int
    last_run: Optional[datetime] = None
    enabled: bool = True


class HeartbeatMonitor:
    """
    Background monitor that generates proactive notifications.

    Features:
    - Periodic check system
    - Notification queue management
    - Built-in common checks
    - Extensible with custom checks
    """

    # Built-in check intervals (seconds)
    DEFAULT_INTERVALS = {
        "session_greeting": 0,         # Run once at start
        "idle_check": 300,             # Every 5 minutes
        "time_awareness": 1800,        # Every 30 minutes
        "memory_reminder": 3600,       # Every hour
    }

    def __init__(
        self,
        data_dir: Optional[str] = None,
        check_interval: int = 60
    ):
        """
        Initialize the heartbeat monitor.

        Args:
            data_dir: Directory for storing state
            check_interval: Base interval for running checks (seconds)
        """
        if data_dir is None:
            data_dir = Path(__file__).parent.parent / "data"

        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.state_file = self.data_dir / "heartbeat_state.json"
        self.check_interval = check_interval

        # Notification queue
        self.notifications: Queue[Notification] = Queue()
        self.notification_history: List[Notification] = []

        # Registered checks
        self.checks: Dict[str, Check] = {}

        # State
        self.running = False
        self._thread: Optional[threading.Thread] = None
        self._last_user_activity: datetime = datetime.now()
        self._session_start: datetime = datetime.now()

        # Load state
        self._load_state()

        # Register built-in checks
        self._register_builtin_checks()

        logger.info("HeartbeatMonitor initialized")

    def _load_state(self) -> None:
        """Load monitor state from file."""
        if self.state_file.exists():
            try:
                data = json.loads(self.state_file.read_text(encoding="utf-8"))
                # Could restore notification history, last activity, etc.
            except (json.JSONDecodeError, KeyError):
                pass

    def _save_state(self) -> None:
        """Save monitor state to file."""
        try:
            state = {
                "last_activity": self._last_user_activity.isoformat(),
                "session_start": self._session_start.isoformat(),
                "notification_count": len(self.notification_history)
            }
            self.state_file.write_text(json.dumps(state, indent=2), encoding="utf-8")
        except IOError as e:
            logger.error(f"Error saving heartbeat state: {e}")

    def _register_builtin_checks(self) -> None:
        """Register default checks."""

        # Session greeting (runs once)
        def session_greeting():
            hour = datetime.now().hour
            if 5 <= hour < 12:
                greeting = "Good morning!"
            elif 12 <= hour < 17:
                greeting = "Good afternoon!"
            elif 17 <= hour < 21:
                greeting = "Good evening!"
            else:
                greeting = "Hey there, working late?"

            return Notification(
                message=f"{greeting} I'm here whenever you need me.",
                priority=NotificationPriority.LOW,
                category="greeting",
                expires_at=(datetime.now() + timedelta(minutes=5)).isoformat()
            )

        # Idle check
        def idle_check():
            idle_time = (datetime.now() - self._last_user_activity).total_seconds()
            if idle_time > 600:  # 10 minutes idle
                return Notification(
                    message="Still here if you need anything...",
                    priority=NotificationPriority.LOW,
                    category="presence",
                    expires_at=(datetime.now() + timedelta(minutes=30)).isoformat()
                )
            return None

        # Time awareness
        def time_awareness():
            hour = datetime.now().hour
            # Late night check
            if 23 <= hour or hour < 5:
                session_hours = (datetime.now() - self._session_start).total_seconds() / 3600
                if session_hours > 2:
                    return Notification(
                        message="It's getting late. Remember to take breaks!",
                        priority=NotificationPriority.MEDIUM,
                        category="wellbeing",
                        action_hint="Consider wrapping up soon"
                    )
            return None

        self.register_check("session_greeting", session_greeting, 0)  # Once
        self.register_check("idle_check", idle_check, 300)
        self.register_check("time_awareness", time_awareness, 1800)

    def register_check(
        self,
        name: str,
        callback: Callable[[], Optional[Notification]],
        interval_seconds: int
    ) -> None:
        """
        Register a new check.

        Args:
            name: Unique name for the check
            callback: Function that returns Notification or None
            interval_seconds: How often to run (0 = once)
        """
        self.checks[name] = Check(
            name=name,
            callback=callback,
            interval_seconds=interval_seconds
        )

    def unregister_check(self, name: str) -> bool:
        """Remove a registered check."""
        if name in self.checks:
            del self.checks[name]
            return True
        return False

    def record_activity(self) -> None:
        """Record user activity (call this when user sends message)."""
        self._last_user_activity = datetime.now()

    def _run_check(self, check: Check) -> Optional[Notification]:
        """Run a single check safely."""
        try:
            result = check.callback()
            check.last_run = datetime.now()
            return result
        except Exception as e:
            logger.error(f"Check '{check.name}' failed: {e}")
            return None

    def _should_run_check(self, check: Check) -> bool:
        """Determine if a check should run now."""
        if not check.enabled:
            return False

        # One-time checks
        if check.interval_seconds == 0:
            return check.last_run is None

        # Periodic checks
        if check.last_run is None:
            return True

        elapsed = (datetime.now() - check.last_run).total_seconds()
        return elapsed >= check.interval_seconds

    def run_checks(self) -> List[Notification]:
        """
        Run all due checks and return any notifications.

        Returns:
            List of new notifications
        """
        new_notifications = []

        for check in self.checks.values():
            if self._should_run_check(check):
                notification = self._run_check(check)
                if notification and not notification.is_expired():
                    self.notifications.put(notification)
                    self.notification_history.append(notification)
                    new_notifications.append(notification)

        # Trim history
        self.notification_history = self.notification_history[-100:]

        return new_notifications

    def get_pending_notifications(
        self,
        min_priority: NotificationPriority = NotificationPriority.LOW
    ) -> List[Notification]:
        """
        Get pending notifications above threshold.

        Args:
            min_priority: Minimum priority to include

        Returns:
            List of notifications (removes from queue)
        """
        notifications = []

        while True:
            try:
                notification = self.notifications.get_nowait()
                if (
                    notification.priority.value >= min_priority.value
                    and not notification.is_expired()
                ):
                    notifications.append(notification)
            except Empty:
                break

        return sorted(notifications, key=lambda n: n.priority.value, reverse=True)

    def add_notification(
        self,
        message: str,
        priority: NotificationPriority = NotificationPriority.MEDIUM,
        category: str = "custom",
        action_hint: Optional[str] = None
    ) -> None:
        """
        Add a notification manually.

        Args:
            message: Notification message
            priority: Priority level
            category: Category for filtering
            action_hint: Suggested action
        """
        notification = Notification(
            message=message,
            priority=priority,
            category=category,
            action_hint=action_hint
        )
        self.notifications.put(notification)
        self.notification_history.append(notification)

    def _monitor_loop(self) -> None:
        """Background monitoring loop with exponential backoff on errors."""
        error_count = 0
        max_errors = 5

        while self.running:
            try:
                self.run_checks()
                self._save_state()
                error_count = 0  # Reset on success
                time.sleep(self.check_interval)
            except Exception as e:
                error_count += 1
                logger.error(f"Monitor loop error ({error_count}/{max_errors}): {e}")

                if error_count >= max_errors:
                    logger.critical("Too many monitor errors, stopping proactive system")
                    self.running = False
                    break

                # Exponential backoff: 10, 20, 40, 80, 160 seconds
                backoff = min(160, 10 * (2 ** (error_count - 1)))
                time.sleep(backoff)

    def start(self) -> None:
        """Start background monitoring."""
        if self.running:
            return

        self.running = True
        self._session_start = datetime.now()
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        logger.info("HeartbeatMonitor started")

    def stop(self) -> None:
        """Stop background monitoring."""
        self.running = False
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("HeartbeatMonitor stopped")

    def get_status(self) -> Dict[str, Any]:
        """Get monitor status."""
        return {
            "running": self.running,
            "checks_registered": len(self.checks),
            "pending_notifications": self.notifications.qsize(),
            "total_notifications": len(self.notification_history),
            "session_duration_min": round(
                (datetime.now() - self._session_start).total_seconds() / 60, 1
            ),
            "idle_seconds": round(
                (datetime.now() - self._last_user_activity).total_seconds(), 0
            )
        }


if __name__ == "__main__":
    print("=" * 60)
    print("HeartbeatMonitor - Proactive System Test")
    print("=" * 60)

    monitor = HeartbeatMonitor()

    # Run checks immediately
    print("\n--- Running checks ---")
    notifications = monitor.run_checks()

    print(f"Generated {len(notifications)} notifications:")
    for n in notifications:
        print(f"  [{n.priority.name}] {n.message}")

    # Get pending
    print("\n--- Pending notifications ---")
    pending = monitor.get_pending_notifications()
    for n in pending:
        print(f"  [{n.category}] {n.message}")
        if n.action_hint:
            print(f"    Hint: {n.action_hint}")

    # Add custom notification
    print("\n--- Adding custom notification ---")
    monitor.add_notification(
        "Remember to commit your changes!",
        priority=NotificationPriority.MEDIUM,
        category="reminder",
        action_hint="Run 'git commit'"
    )

    # Status
    print("\n--- Status ---")
    status = monitor.get_status()
    for k, v in status.items():
        print(f"  {k}: {v}")

    print("\n" + "=" * 60)
    print("Test complete!")
