"""Background scheduler daemon for notifications.

Run this separately from the main agent to handle scheduled tasks.
Usage: python -m apprentice_agent.scheduler
"""

import json
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import psutil

try:
    from winotify import Notification, audio
    WINOTIFY_AVAILABLE = True
except ImportError:
    WINOTIFY_AVAILABLE = False
    print("Warning: winotify not installed. Notifications will be logged only.")


# Paths
BASE_DIR = Path(__file__).parent.parent
TASKS_FILE = BASE_DIR / "data" / "scheduled_tasks.json"
LOGS_DIR = BASE_DIR / "logs" / "notifications"

# Ensure directories exist
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Setup logging
log_file = LOGS_DIR / f"{datetime.now().strftime('%Y-%m-%d')}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file, encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class NotificationScheduler:
    """Background daemon for processing scheduled notifications."""

    def __init__(self, check_interval: int = 30):
        """Initialize scheduler.

        Args:
            check_interval: Seconds between checks (default 30)
        """
        self.check_interval = check_interval
        self.running = False
        self._condition_cooldown = {}  # Prevent spam for conditional alerts

    def _load_tasks(self) -> dict:
        """Load tasks from JSON file."""
        try:
            if TASKS_FILE.exists():
                with open(TASKS_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Failed to load tasks: {e}")
        return {"reminders": [], "scheduled": [], "conditional": []}

    def _save_tasks(self, tasks: dict) -> bool:
        """Save tasks to JSON file."""
        try:
            TASKS_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(TASKS_FILE, "w", encoding="utf-8") as f:
                json.dump(tasks, f, indent=4)
            return True
        except IOError as e:
            logger.error(f"Failed to save tasks: {e}")
            return False

    def _send_notification(self, title: str, message: str, task_type: str = "reminder"):
        """Send a Windows toast notification.

        Args:
            title: Notification title
            message: Notification body
            task_type: Type of task for icon selection
        """
        logger.info(f"NOTIFICATION [{task_type}]: {title} - {message}")

        if not WINOTIFY_AVAILABLE:
            return

        try:
            toast = Notification(
                app_id="Aura - Apprentice Agent",
                title=title,
                msg=message,
                duration="short"
            )

            # Set audio based on type
            if task_type == "conditional":
                toast.set_audio(audio.LoopingAlarm, loop=False)
            else:
                toast.set_audio(audio.Default, loop=False)

            toast.show()
        except Exception as e:
            logger.error(f"Failed to show notification: {e}")

    def _check_reminders(self, tasks: dict) -> bool:
        """Check and fire due reminders.

        Args:
            tasks: Current tasks dict

        Returns:
            True if tasks were modified
        """
        now = datetime.now()
        modified = False
        remaining = []

        for reminder in tasks.get("reminders", []):
            try:
                fire_at = datetime.fromisoformat(reminder["fire_at"])
                if now >= fire_at:
                    # Fire the reminder
                    self._send_notification(
                        "Reminder",
                        reminder["message"],
                        "reminder"
                    )
                    modified = True
                    logger.info(f"Fired reminder: {reminder['id']} - {reminder['message']}")
                else:
                    remaining.append(reminder)
            except (KeyError, ValueError) as e:
                logger.error(f"Invalid reminder format: {e}")
                # Keep malformed reminders for manual cleanup
                remaining.append(reminder)

        tasks["reminders"] = remaining
        return modified

    def _check_scheduled(self, tasks: dict) -> bool:
        """Check and fire scheduled notifications.

        Args:
            tasks: Current tasks dict

        Returns:
            True if any notification was fired
        """
        now = datetime.now()
        current_time = now.strftime("%H:%M")
        current_weekday = now.weekday()  # 0=Monday, 6=Sunday
        fired = False

        for scheduled in tasks.get("scheduled", []):
            try:
                task_time = scheduled["time"]
                repeat = scheduled.get("repeat", "daily")

                # Check if time matches (within 1 minute window)
                if task_time != current_time:
                    continue

                # Check if we already fired this minute
                last_fired = scheduled.get("last_fired_date")
                today = now.strftime("%Y-%m-%d")
                if last_fired == today:
                    continue

                # Check repeat pattern
                should_fire = False
                if repeat == "daily":
                    should_fire = True
                elif repeat == "weekdays":
                    should_fire = current_weekday < 5  # Mon-Fri
                elif repeat == "weekly":
                    # Fire on the same weekday as created
                    created = datetime.fromisoformat(scheduled["created_at"])
                    should_fire = current_weekday == created.weekday()

                if should_fire:
                    self._send_notification(
                        f"Scheduled ({repeat})",
                        scheduled["message"],
                        "scheduled"
                    )
                    scheduled["last_fired_date"] = today
                    fired = True
                    logger.info(f"Fired scheduled: {scheduled['id']} - {scheduled['message']}")

            except (KeyError, ValueError) as e:
                logger.error(f"Invalid scheduled format: {e}")

        return fired

    def _check_conditional(self, tasks: dict) -> bool:
        """Check and fire conditional notifications.

        Args:
            tasks: Current tasks dict

        Returns:
            True if any notification was fired
        """
        fired = False
        now = datetime.now()

        # Get system stats
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            ram_percent = psutil.virtual_memory().percent
            disk_percent = psutil.disk_usage('/').percent
        except Exception as e:
            logger.error(f"Failed to get system stats: {e}")
            return False

        stats = {
            "cpu": cpu_percent,
            "ram": ram_percent,
            "disk": disk_percent
        }

        for conditional in tasks.get("conditional", []):
            try:
                task_id = conditional["id"]
                condition_type = conditional["condition_type"]
                threshold = conditional["threshold"]
                message = conditional["message"]

                current_value = stats.get(condition_type, 0)

                # Check if threshold exceeded
                if current_value > threshold:
                    # Check cooldown (5 minute cooldown between alerts)
                    last_triggered = self._condition_cooldown.get(task_id)
                    if last_triggered:
                        elapsed = (now - last_triggered).total_seconds()
                        if elapsed < 300:  # 5 minutes
                            continue

                    self._send_notification(
                        f"Alert: {condition_type.upper()} High",
                        f"{message}\n\nCurrent: {current_value:.1f}% (Threshold: {threshold}%)",
                        "conditional"
                    )
                    self._condition_cooldown[task_id] = now
                    conditional["last_triggered"] = now.isoformat()
                    fired = True
                    logger.info(f"Fired conditional: {task_id} - {condition_type}={current_value:.1f}%>{threshold}%")

            except (KeyError, ValueError) as e:
                logger.error(f"Invalid conditional format: {e}")

        return fired

    def run_once(self) -> None:
        """Run a single check cycle."""
        tasks = self._load_tasks()

        modified = False
        modified |= self._check_reminders(tasks)
        modified |= self._check_scheduled(tasks)
        modified |= self._check_conditional(tasks)

        if modified:
            self._save_tasks(tasks)

    def run(self) -> None:
        """Run the scheduler daemon loop."""
        logger.info(f"Starting notification scheduler (check every {self.check_interval}s)")
        logger.info(f"Tasks file: {TASKS_FILE}")
        logger.info(f"Log file: {log_file}")

        self.running = True

        try:
            while self.running:
                self.run_once()
                time.sleep(self.check_interval)
        except KeyboardInterrupt:
            logger.info("Scheduler stopped by user")
        except Exception as e:
            logger.error(f"Scheduler error: {e}")
            raise

    def stop(self) -> None:
        """Stop the scheduler daemon."""
        self.running = False
        logger.info("Scheduler stopping...")


def main():
    """Main entry point for scheduler daemon."""
    print("=" * 60)
    print("Aura - Notification Scheduler")
    print("=" * 60)
    print()
    print(f"Tasks file: {TASKS_FILE}")
    print(f"Log directory: {LOGS_DIR}")
    print(f"Winotify available: {WINOTIFY_AVAILABLE}")
    print()
    print("Press Ctrl+C to stop")
    print()

    scheduler = NotificationScheduler(check_interval=30)
    scheduler.run()


if __name__ == "__main__":
    main()
