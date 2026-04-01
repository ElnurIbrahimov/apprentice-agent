"""Hooks / Event System for AURA.

Allows users to register event-driven actions:
"When X happens, do Y."

Event types: schedule, file_modified, system_alert, clipboard_changed, keyword_on_screen
Action types: notify, speak, run_tool, log
"""

import json
import logging
import os
import subprocess
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class HooksManager:
    """Manages event-driven hooks that trigger actions automatically.

    Usage:
        hooks = HooksManager(tools=agent.tools)
        hooks.register("schedule", "09:00", "notify", "Good morning!")
        hooks.start_background(interval=15)
    """

    STORAGE_FILE = "hooks.json"

    def __init__(self, tools: Optional[Dict[str, Any]] = None, data_dir: str = None):
        """Initialize the HooksManager.

        Args:
            tools: Agent tool registry (for run_tool actions)
            data_dir: Directory to store hooks.json (default: data/)
        """
        self.tools = tools or {}
        self._hooks: Dict[str, dict] = {}
        self._hooks_lock = threading.Lock()
        self._background_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._last_clipboard: Optional[str] = None
        self._file_mtimes: Dict[str, float] = {}

        # Storage path
        if data_dir:
            self._data_dir = Path(data_dir)
        else:
            self._data_dir = Path(__file__).parent.parent / "data"
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._storage_path = self._data_dir / self.STORAGE_FILE

        # Load existing hooks
        self._load_hooks()

    def register(self, event: str, condition: str, action: str, action_args: str = "") -> str:
        """Register a new hook.

        Args:
            event: Event type (schedule, file_modified, system_alert, clipboard_changed, keyword_on_screen)
            condition: Event-specific condition (e.g., "09:00" for schedule, file path for file_modified)
            action: Action type (notify, speak, run_tool, log)
            action_args: Arguments for the action

        Returns:
            Hook ID string
        """
        hook_id = str(uuid.uuid4())[:8]
        hook = {
            "id": hook_id,
            "event": event,
            "condition": condition,
            "action": action,
            "action_args": action_args,
            "created": datetime.now().isoformat(),
            "last_triggered": None,
            "trigger_count": 0,
        }
        with self._hooks_lock:
            self._hooks[hook_id] = hook

        # Initialize file mtime tracking if needed
        if event == "file_modified" and os.path.exists(condition):
            self._file_mtimes[condition] = os.path.getmtime(condition)

        self._save_hooks()
        logger.info(f"[HOOKS] Registered hook {hook_id}: {event}:{condition} -> {action}")
        return hook_id

    def unregister(self, hook_id: str) -> bool:
        """Remove a hook by ID.

        Args:
            hook_id: The hook ID to remove

        Returns:
            True if removed, False if not found
        """
        with self._hooks_lock:
            if hook_id in self._hooks:
                del self._hooks[hook_id]
                self._save_hooks()
                logger.info(f"[HOOKS] Unregistered hook {hook_id}")
                return True
            return False

    def list_hooks(self) -> List[dict]:
        """List all registered hooks.

        Returns:
            List of hook dicts
        """
        with self._hooks_lock:
            return list(self._hooks.values())

    def check_events(self):
        """Check all event sources and trigger matching hooks."""
        now = datetime.now()

        with self._hooks_lock:
            hooks_snapshot = list(self._hooks.items())
        for hook_id, hook in hooks_snapshot:
            try:
                triggered = False
                event = hook["event"]

                if event == "schedule":
                    triggered = self._check_schedule(hook, now)
                elif event == "file_modified":
                    triggered = self._check_file(hook)
                elif event == "system_alert":
                    triggered = self._check_system(hook)
                elif event == "clipboard_changed":
                    triggered = self._check_clipboard(hook)
                elif event == "keyword_on_screen":
                    triggered = self._check_screen(hook)

                if triggered:
                    self._execute_action(hook)
                    with self._hooks_lock:
                        hook["last_triggered"] = now.isoformat()
                        hook["trigger_count"] += 1
                    self._save_hooks()

            except Exception as e:
                logger.debug(f"[HOOKS] Error checking hook {hook_id}: {e}")

    def start_background(self, interval: int = 15):
        """Start background event checking thread.

        Args:
            interval: Seconds between checks
        """
        if self._background_thread and self._background_thread.is_alive():
            return

        self._stop_event.clear()

        def _check_loop():
            while not self._stop_event.is_set():
                try:
                    self.check_events()
                except Exception as e:
                    logger.debug(f"[HOOKS] Background check error: {e}")
                self._stop_event.wait(timeout=interval)

        self._background_thread = threading.Thread(target=_check_loop, daemon=True, name="hooks-bg")
        self._background_thread.start()
        logger.info(f"[HOOKS] Background checking started (interval={interval}s)")

    def stop_background(self):
        """Stop the background checking thread."""
        self._stop_event.set()
        if self._background_thread:
            self._background_thread.join(timeout=5)
            self._background_thread = None
        logger.info("[HOOKS] Background checking stopped")

    # ---- Event Checkers ----

    def _check_schedule(self, hook: dict, now: datetime) -> bool:
        """Check if schedule condition matches current time.

        Condition format: "HH:MM" or "HH:MM:day_of_week" (0=Mon, 6=Sun)
        """
        condition = hook["condition"]
        parts = condition.split(":")

        if len(parts) < 2:
            return False

        try:
            target_hour = int(parts[0])
            target_minute = int(parts[1])
        except ValueError:
            return False

        # Check day of week if specified
        if len(parts) >= 3:
            try:
                target_dow = int(parts[2])
                if now.weekday() != target_dow:
                    return False
            except ValueError:
                pass

        # Match hour and minute (within the check interval window)
        if now.hour == target_hour and now.minute == target_minute:
            # Prevent re-triggering within same minute
            last = hook.get("last_triggered")
            if last:
                last_dt = datetime.fromisoformat(last)
                if (now - last_dt).total_seconds() < 60:
                    return False
            return True

        return False

    def _check_file(self, hook: dict) -> bool:
        """Check if a watched file has been modified."""
        filepath = hook["condition"]
        if not os.path.exists(filepath):
            return False

        current_mtime = os.path.getmtime(filepath)
        last_mtime = self._file_mtimes.get(filepath, 0)

        if current_mtime > last_mtime and last_mtime > 0:
            self._file_mtimes[filepath] = current_mtime
            return True

        self._file_mtimes[filepath] = current_mtime
        return False

    def _check_system(self, hook: dict) -> bool:
        """Check system resource alerts (CPU/RAM/disk).

        Condition format: "cpu>80" or "ram>90" or "disk>95"
        """
        condition = hook["condition"].lower()

        try:
            import psutil
        except ImportError:
            return False

        try:
            if "cpu>" in condition:
                threshold = float(condition.split(">")[1])
                return psutil.cpu_percent(interval=0.1) > threshold
            elif "ram>" in condition:
                threshold = float(condition.split(">")[1])
                return psutil.virtual_memory().percent > threshold
            elif "disk>" in condition:
                threshold = float(condition.split(">")[1])
                return psutil.disk_usage("/").percent > threshold
        except (ValueError, IndexError):
            pass

        return False

    def _check_clipboard(self, hook: dict) -> bool:
        """Check if clipboard content has changed and matches condition."""
        try:
            import pyperclip
            current = pyperclip.paste()
        except (ImportError, Exception):
            return False

        if current == self._last_clipboard:
            return False

        self._last_clipboard = current

        # Condition can be a keyword to match, or "*" for any change
        condition = hook["condition"]
        if condition == "*":
            return True
        return condition.lower() in current.lower()

    def _check_screen(self, hook: dict) -> bool:
        """Check if a keyword appears on screen using ScreenReaderTool OCR."""
        keyword = hook["condition"]

        # Rate limit: max once per 60 seconds for screen checks
        last = hook.get("last_triggered")
        if last:
            last_dt = datetime.fromisoformat(last)
            if (datetime.now() - last_dt).total_seconds() < 60:
                return False

        try:
            if "screen_reader" in self.tools:
                result = self.tools["screen_reader"].read_screen()
                if result.get("success") and keyword.lower() in result.get("text", "").lower():
                    return True
        except Exception:
            pass

        return False

    # ---- Action Executors ----

    def _execute_action(self, hook: dict):
        """Execute the action associated with a triggered hook."""
        action = hook.get("action")
        if not action:
            return
        args = hook.get("action_args", "")

        logger.info(f"[HOOKS] Triggered: {hook['event']}:{hook['condition']} -> {action} {args}")

        if action == "notify":
            self._action_notify(args, hook)
        elif action == "speak":
            self._action_speak(args)
        elif action == "run_tool":
            self._action_run_tool(args)
        elif action == "log":
            self._action_log(args, hook)
        else:
            logger.warning(f"[HOOKS] Unknown action type: {action}")

    def _action_notify(self, message: str, hook: dict):
        """Show a desktop notification."""
        timestamp = datetime.now().strftime("%H:%M")
        logger.debug(f"\n  [HOOK {hook['id']}] {timestamp}: {message}")
        # Try system notification
        try:
            if os.name == 'nt':
                # Windows toast notification - message passed via env var to avoid injection
                ps_script = (
                    "[void][System.Reflection.Assembly]::LoadWithPartialName('System.Windows.Forms');"
                    "$n = New-Object System.Windows.Forms.NotifyIcon;"
                    "$n.Icon = [System.Drawing.SystemIcons]::Information;"
                    "$n.Visible = $true;"
                    "$n.ShowBalloonTip(5000, 'AURA Hook', $env:HOOK_MSG, 'Info')"
                )
                _SENSITIVE_ENV_KEYS = frozenset({
                    "OLLAMA_API_KEY", "E2B_API_KEY", "TAVILY_API_KEY",
                    "BRAVE_API_KEY", "FIRECRAWL_API_KEY", "TELEGRAM_BOT_TOKEN",
                    "CHATGPT_REFRESH_TOKEN",
                })
                safe_env = {k: v for k, v in os.environ.items() if k not in _SENSITIVE_ENV_KEYS}
                safe_env["HOOK_MSG"] = str(message)
                subprocess.run(
                    ["powershell", "-Command", ps_script],
                    env=safe_env,
                    timeout=5
                )
            else:
                subprocess.run(["notify-send", "AURA Hook", str(message)], timeout=5, check=False)
        except Exception:
            pass

    def _action_speak(self, text: str):
        """Speak text using TTS."""
        try:
            from aura.services.voice_presence import get_voice_presence
            vps = get_voice_presence()
            if vps._enabled:
                vps.speak(text, block=False)
            else:
                logger.debug(f"\n  [HOOK TTS] {text}")
        except Exception:
            logger.debug(f"\n  [HOOK TTS] {text}")

    def _action_run_tool(self, tool_spec: str):
        """Run a tool with given arguments.

        Format: "tool_name:method:arg1,arg2"
        """
        parts = tool_spec.split(":", maxsplit=2)
        if not parts:
            return

        tool_name = parts[0]
        if tool_name not in self.tools:
            logger.warning(f"[HOOKS] Tool not found: {tool_name}")
            return

        tool = self.tools[tool_name]
        method = parts[1] if len(parts) > 1 else "run"
        args_str = parts[2] if len(parts) > 2 else ""

        try:
            func = getattr(tool, method, None)
            if func and callable(func):
                if args_str:
                    result = func(args_str)
                else:
                    result = func()
                logger.info(f"[HOOKS] Tool {tool_name}.{method} result: {str(result)[:200]}")
        except Exception as e:
            logger.error(f"[HOOKS] Tool execution error: {e}")

    def _action_log(self, message: str, hook: dict):
        """Log a message to the hooks log file."""
        log_file = self._data_dir / "hooks_log.txt"
        timestamp = datetime.now().isoformat()
        entry = f"[{timestamp}] [{hook['event']}:{hook['condition']}] {message}\n"
        try:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(entry)
        except Exception as e:
            logger.error(f"[HOOKS] Log write error: {e}")

    # ---- Persistence ----

    def _save_hooks(self):
        """Save hooks to JSON file (atomic write via temp file + rename)."""
        try:
            import tempfile
            with self._hooks_lock:
                data = {"hooks": dict(self._hooks)}
            dir_path = self._storage_path.parent
            fd, tmp_path = tempfile.mkstemp(dir=str(dir_path), suffix=".tmp")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)
                os.replace(tmp_path, str(self._storage_path))
            except BaseException:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise
        except Exception as e:
            logger.error(f"[HOOKS] Save error: {e}")

    def _load_hooks(self):
        """Load hooks from JSON file."""
        if not self._storage_path.exists():
            return

        try:
            with open(self._storage_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._hooks = data.get("hooks", {})

            # Re-initialize file mtime tracking
            for hook in self._hooks.values():
                if hook["event"] == "file_modified" and os.path.exists(hook["condition"]):
                    self._file_mtimes[hook["condition"]] = os.path.getmtime(hook["condition"])

            logger.info(f"[HOOKS] Loaded {len(self._hooks)} hooks from disk")
        except (json.JSONDecodeError, IOError, KeyError, TypeError) as e:
            logger.error(f"[HOOKS] Load error: {e}")
            self._hooks = {}
