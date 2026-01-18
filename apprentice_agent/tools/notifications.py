"""Smart Notifications tool for reminders, scheduled tasks, and conditional alerts."""

import json
import re
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional


class NotificationTool:
    """Tool for managing reminders, scheduled notifications, and conditional alerts."""

    name = "notifications"
    description = "Set reminders, schedule notifications, and create conditional alerts"

    # Path to scheduled tasks file
    TASKS_FILE = Path(__file__).parent.parent.parent / "data" / "scheduled_tasks.json"

    def __init__(self):
        """Initialize notification tool."""
        self._ensure_tasks_file()

    def _ensure_tasks_file(self):
        """Ensure the tasks file and directory exist."""
        self.TASKS_FILE.parent.mkdir(parents=True, exist_ok=True)
        if not self.TASKS_FILE.exists():
            self._save_tasks({"reminders": [], "scheduled": [], "conditional": []})

    def _load_tasks(self) -> dict:
        """Load tasks from JSON file."""
        try:
            with open(self.TASKS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {"reminders": [], "scheduled": [], "conditional": []}

    def _save_tasks(self, tasks: dict) -> bool:
        """Save tasks to JSON file."""
        try:
            with open(self.TASKS_FILE, "w", encoding="utf-8") as f:
                json.dump(tasks, f, indent=4)
            return True
        except IOError:
            return False

    def _generate_id(self) -> str:
        """Generate a short unique ID."""
        return uuid.uuid4().hex[:8]

    def _parse_time_str(self, time_str: str) -> Optional[datetime]:
        """Parse relative time string like 'in 30 minutes', 'in 2 hours'.

        Args:
            time_str: Relative time string

        Returns:
            datetime when the reminder should fire, or None if parse fails
        """
        time_str = time_str.lower().strip()

        # Remove "in" prefix if present
        if time_str.startswith("in "):
            time_str = time_str[3:]

        # Parse patterns: "30 minutes", "2 hours", "1 hour", "5 mins"
        patterns = [
            (r"(\d+)\s*(?:minutes?|mins?)", "minutes"),
            (r"(\d+)\s*(?:hours?|hrs?)", "hours"),
            (r"(\d+)\s*(?:seconds?|secs?)", "seconds"),
            (r"(\d+)\s*(?:days?)", "days"),
        ]

        for pattern, unit in patterns:
            match = re.search(pattern, time_str)
            if match:
                value = int(match.group(1))
                now = datetime.now()
                if unit == "minutes":
                    return now + timedelta(minutes=value)
                elif unit == "hours":
                    return now + timedelta(hours=value)
                elif unit == "seconds":
                    return now + timedelta(seconds=value)
                elif unit == "days":
                    return now + timedelta(days=value)

        return None

    def _parse_time_of_day(self, time_str: str) -> Optional[str]:
        """Parse time of day string like '9:00 AM', '14:30', '9am'.

        Args:
            time_str: Time of day string

        Returns:
            Normalized time string in HH:MM format, or None if parse fails
        """
        time_str = time_str.strip().upper()

        # Try HH:MM AM/PM format
        match = re.match(r"(\d{1,2}):(\d{2})\s*(AM|PM)?", time_str)
        if match:
            hour = int(match.group(1))
            minute = int(match.group(2))
            ampm = match.group(3)

            if ampm == "PM" and hour != 12:
                hour += 12
            elif ampm == "AM" and hour == 12:
                hour = 0

            return f"{hour:02d}:{minute:02d}"

        # Try simple format like "9am", "2pm"
        match = re.match(r"(\d{1,2})\s*(AM|PM)", time_str)
        if match:
            hour = int(match.group(1))
            ampm = match.group(2)

            if ampm == "PM" and hour != 12:
                hour += 12
            elif ampm == "AM" and hour == 12:
                hour = 0

            return f"{hour:02d}:00"

        # Try 24-hour format
        match = re.match(r"(\d{1,2}):(\d{2})", time_str)
        if match:
            hour = int(match.group(1))
            minute = int(match.group(2))
            if 0 <= hour <= 23 and 0 <= minute <= 59:
                return f"{hour:02d}:{minute:02d}"

        return None

    def add_reminder(self, message: str, time_str: str) -> dict:
        """Add a one-time reminder.

        Args:
            message: Reminder message
            time_str: Relative time like "in 30 minutes", "in 2 hours"

        Returns:
            dict with success status and task info
        """
        if not message:
            return {"success": False, "error": "No message provided"}

        fire_time = self._parse_time_str(time_str)
        if not fire_time:
            return {
                "success": False,
                "error": f"Could not parse time: {time_str}. Try 'in 30 minutes' or 'in 2 hours'"
            }

        task_id = self._generate_id()
        task = {
            "id": task_id,
            "message": message,
            "fire_at": fire_time.isoformat(),
            "created_at": datetime.now().isoformat(),
            "type": "reminder"
        }

        tasks = self._load_tasks()
        tasks["reminders"].append(task)
        self._save_tasks(tasks)

        return {
            "success": True,
            "task_id": task_id,
            "message": message,
            "fire_at": fire_time.strftime("%Y-%m-%d %H:%M:%S"),
            "fire_in": time_str,
            "response": f"Reminder set for {fire_time.strftime('%H:%M:%S')}: {message}"
        }

    def add_scheduled(self, message: str, time_of_day: str, repeat: str = "daily") -> dict:
        """Add a recurring scheduled notification.

        Args:
            message: Notification message
            time_of_day: Time like "9:00 AM", "14:30"
            repeat: Repeat pattern - "daily", "weekdays", "weekly"

        Returns:
            dict with success status and task info
        """
        if not message:
            return {"success": False, "error": "No message provided"}

        normalized_time = self._parse_time_of_day(time_of_day)
        if not normalized_time:
            return {
                "success": False,
                "error": f"Could not parse time: {time_of_day}. Try '9:00 AM' or '14:30'"
            }

        repeat = repeat.lower()
        if repeat not in ["daily", "weekdays", "weekly"]:
            return {
                "success": False,
                "error": f"Invalid repeat option: {repeat}. Use 'daily', 'weekdays', or 'weekly'"
            }

        task_id = self._generate_id()
        task = {
            "id": task_id,
            "message": message,
            "time": normalized_time,
            "repeat": repeat,
            "created_at": datetime.now().isoformat(),
            "type": "scheduled"
        }

        tasks = self._load_tasks()
        tasks["scheduled"].append(task)
        self._save_tasks(tasks)

        return {
            "success": True,
            "task_id": task_id,
            "message": message,
            "time": normalized_time,
            "repeat": repeat,
            "response": f"Scheduled {repeat} notification at {normalized_time}: {message}"
        }

    def add_condition(self, message: str, condition_type: str, threshold: int) -> dict:
        """Add a conditional notification that fires when system condition is met.

        Args:
            message: Notification message
            condition_type: "cpu", "ram", or "disk"
            threshold: Percentage threshold (0-100)

        Returns:
            dict with success status and task info
        """
        if not message:
            return {"success": False, "error": "No message provided"}

        condition_type = condition_type.lower()
        if condition_type not in ["cpu", "ram", "disk"]:
            return {
                "success": False,
                "error": f"Invalid condition type: {condition_type}. Use 'cpu', 'ram', or 'disk'"
            }

        try:
            threshold = int(threshold)
            if not 0 <= threshold <= 100:
                raise ValueError()
        except (ValueError, TypeError):
            return {
                "success": False,
                "error": f"Invalid threshold: {threshold}. Must be 0-100"
            }

        task_id = self._generate_id()
        task = {
            "id": task_id,
            "message": message,
            "condition_type": condition_type,
            "threshold": threshold,
            "created_at": datetime.now().isoformat(),
            "type": "conditional",
            "last_triggered": None
        }

        tasks = self._load_tasks()
        tasks["conditional"].append(task)
        self._save_tasks(tasks)

        return {
            "success": True,
            "task_id": task_id,
            "message": message,
            "condition": f"{condition_type} > {threshold}%",
            "response": f"Conditional alert set: notify when {condition_type.upper()} exceeds {threshold}%"
        }

    def list_tasks(self) -> dict:
        """List all scheduled tasks.

        Returns:
            dict with all tasks organized by type
        """
        tasks = self._load_tasks()

        total = (
            len(tasks.get("reminders", [])) +
            len(tasks.get("scheduled", [])) +
            len(tasks.get("conditional", []))
        )

        if total == 0:
            return {
                "success": True,
                "total": 0,
                "message": "No scheduled tasks",
                "tasks": tasks
            }

        # Format tasks for display
        formatted = []

        for reminder in tasks.get("reminders", []):
            fire_at = datetime.fromisoformat(reminder["fire_at"])
            formatted.append(
                f"[{reminder['id']}] REMINDER at {fire_at.strftime('%H:%M:%S')}: {reminder['message']}"
            )

        for scheduled in tasks.get("scheduled", []):
            formatted.append(
                f"[{scheduled['id']}] SCHEDULED {scheduled['repeat']} at {scheduled['time']}: {scheduled['message']}"
            )

        for conditional in tasks.get("conditional", []):
            formatted.append(
                f"[{conditional['id']}] CONDITIONAL {conditional['condition_type']}>{conditional['threshold']}%: {conditional['message']}"
            )

        return {
            "success": True,
            "total": total,
            "reminders": len(tasks.get("reminders", [])),
            "scheduled": len(tasks.get("scheduled", [])),
            "conditional": len(tasks.get("conditional", [])),
            "tasks": tasks,
            "formatted": "\n".join(formatted),
            "message": f"Found {total} scheduled task(s)"
        }

    def remove_task(self, task_id: str) -> dict:
        """Remove a task by ID.

        Args:
            task_id: The task ID to remove

        Returns:
            dict with success status
        """
        if not task_id:
            return {"success": False, "error": "No task ID provided"}

        tasks = self._load_tasks()
        found = False

        for task_type in ["reminders", "scheduled", "conditional"]:
            for i, task in enumerate(tasks.get(task_type, [])):
                if task.get("id") == task_id:
                    tasks[task_type].pop(i)
                    found = True
                    break
            if found:
                break

        if not found:
            return {
                "success": False,
                "error": f"Task not found: {task_id}"
            }

        self._save_tasks(tasks)
        return {
            "success": True,
            "removed_id": task_id,
            "message": f"Removed task {task_id}"
        }

    def clear_all(self) -> dict:
        """Remove all scheduled tasks.

        Returns:
            dict with success status
        """
        tasks = self._load_tasks()
        total = (
            len(tasks.get("reminders", [])) +
            len(tasks.get("scheduled", [])) +
            len(tasks.get("conditional", []))
        )

        self._save_tasks({"reminders": [], "scheduled": [], "conditional": []})

        return {
            "success": True,
            "removed_count": total,
            "message": f"Cleared {total} task(s)"
        }

    def execute(self, action: str, **kwargs) -> dict:
        """Execute a notification action.

        Args:
            action: Action to perform
            **kwargs: Additional arguments

        Returns:
            dict with action result
        """
        action_lower = action.lower()

        # List tasks
        if "list" in action_lower or "show" in action_lower:
            return self.list_tasks()

        # Clear all
        if "clear" in action_lower and "all" in action_lower:
            return self.clear_all()

        # Remove task
        if "remove" in action_lower or "cancel" in action_lower or "delete" in action_lower:
            task_id = kwargs.get("task_id") or self._extract_task_id(action)
            if task_id:
                return self.remove_task(task_id)
            return {"success": False, "error": "No task ID specified"}

        # Add conditional
        if "condition" in action_lower or "when" in action_lower or "alert" in action_lower:
            message = kwargs.get("message")
            condition_type = kwargs.get("condition_type")
            threshold = kwargs.get("threshold")
            # Try to extract from action string
            if not all([message, condition_type, threshold]):
                extracted = self._extract_condition(action)
                message = message or extracted.get("message")
                condition_type = condition_type or extracted.get("condition_type")
                threshold = threshold or extracted.get("threshold")
            if message and condition_type and threshold:
                return self.add_condition(message, condition_type, threshold)
            return {"success": False, "error": "Missing condition parameters (message, condition_type, threshold)"}

        # Add scheduled
        if "every" in action_lower or "daily" in action_lower or "schedule" in action_lower:
            message = kwargs.get("message")
            time_of_day = kwargs.get("time_of_day") or kwargs.get("time")
            repeat = kwargs.get("repeat", "daily")
            # Try to extract from action string
            if not time_of_day:
                extracted = self._extract_schedule(action)
                message = message or extracted.get("message")
                time_of_day = extracted.get("time")
                repeat = extracted.get("repeat", repeat)
            if message and time_of_day:
                return self.add_scheduled(message, time_of_day, repeat)
            return {"success": False, "error": "Missing schedule parameters (message, time_of_day)"}

        # Add reminder (default)
        if "remind" in action_lower or "in " in action_lower:
            message = kwargs.get("message")
            time_str = kwargs.get("time_str") or kwargs.get("time")
            # Try to extract from action string
            if not time_str:
                extracted = self._extract_reminder(action)
                message = message or extracted.get("message")
                time_str = extracted.get("time")
            if message and time_str:
                return self.add_reminder(message, time_str)
            return {"success": False, "error": "Missing reminder parameters (message, time)"}

        return {"success": False, "error": f"Unknown notification action: {action}"}

    def _extract_task_id(self, action: str) -> Optional[str]:
        """Extract task ID from action string."""
        # Look for 8-character hex ID
        match = re.search(r'\b([a-f0-9]{8})\b', action.lower())
        return match.group(1) if match else None

    def _extract_reminder(self, action: str) -> dict:
        """Extract reminder info from action string."""
        result = {}

        # Look for time pattern
        time_match = re.search(r'in\s+(\d+\s*(?:minutes?|mins?|hours?|hrs?|seconds?|secs?|days?))', action, re.IGNORECASE)
        if time_match:
            result["time"] = "in " + time_match.group(1)

        # Look for message (often in quotes or after "to" or "remind me")
        msg_patterns = [
            r'"([^"]+)"',
            r"'([^']+)'",
            r'remind(?:\s+me)?\s+(?:to\s+)?(.+?)(?:\s+in\s+\d+|\s*$)',
            r'to\s+(.+?)(?:\s+in\s+\d+|\s*$)',
        ]
        for pattern in msg_patterns:
            match = re.search(pattern, action, re.IGNORECASE)
            if match:
                result["message"] = match.group(1).strip()
                break

        return result

    def _extract_schedule(self, action: str) -> dict:
        """Extract schedule info from action string."""
        result = {"repeat": "daily"}

        # Look for time
        time_match = re.search(r'(?:at\s+)?(\d{1,2}(?::\d{2})?\s*(?:am|pm)?)', action, re.IGNORECASE)
        if time_match:
            result["time"] = time_match.group(1)

        # Look for repeat pattern
        if "weekday" in action.lower():
            result["repeat"] = "weekdays"
        elif "weekly" in action.lower() or "every week" in action.lower():
            result["repeat"] = "weekly"

        # Look for message
        msg_patterns = [
            r'"([^"]+)"',
            r"'([^']+)'",
        ]
        for pattern in msg_patterns:
            match = re.search(pattern, action)
            if match:
                result["message"] = match.group(1).strip()
                break

        return result

    def _extract_condition(self, action: str) -> dict:
        """Extract condition info from action string."""
        result = {}

        # Look for condition type
        if "cpu" in action.lower():
            result["condition_type"] = "cpu"
        elif "ram" in action.lower() or "memory" in action.lower():
            result["condition_type"] = "ram"
        elif "disk" in action.lower():
            result["condition_type"] = "disk"

        # Look for threshold
        threshold_match = re.search(r'(\d+)\s*%', action)
        if threshold_match:
            result["threshold"] = int(threshold_match.group(1))

        # Look for message
        msg_patterns = [
            r'"([^"]+)"',
            r"'([^']+)'",
        ]
        for pattern in msg_patterns:
            match = re.search(pattern, action)
            if match:
                result["message"] = match.group(1).strip()
                break

        return result


# Singleton instance
notification_tool = NotificationTool()
