#!/usr/bin/env python3
# D:/Aura/aura_daemon.py
"""
AURA Living Daemon — always-on background service.

Extends the existing HooksManager 15s loop into a proper
tiered daemon with screen monitoring, event bus, and proactive intelligence.

Run directly:    python aura_daemon.py
Install service: python aura_daemon.py --install
"""

import os

os.environ["TQDM_DISABLE"] = "1"

import argparse
import json
import logging
import signal
import sys
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional

# Headless mode — skip screen/GUI features on servers without a display
HEADLESS = bool(os.environ.get("AURA_HEADLESS")) or (
    sys.platform != "win32" and not os.environ.get("DISPLAY")
)

# On headless servers, prevent imports of GUI/audio modules from crashing.
# We install import hooks that return stub modules for known Windows/desktop-only packages.
if HEADLESS:
    _HEADLESS_STUBS = {
        "mss", "mss.tools", "pyperclip", "pyautogui",
        "sounddevice", "pyttsx3", "comtypes", "pycaw",
        "winotify", "screen_brightness_control",
        "win32gui", "win32con", "win32api", "win32process",
        "pystray", "PIL.Image",
    }

    import importlib.abc
    import importlib.machinery
    import types

    class _HeadlessStubFinder(importlib.abc.MetaPathFinder):
        """Return a stub loader for desktop-only modules in headless mode."""
        def find_spec(self, fullname, path, target=None):
            if fullname in _HEADLESS_STUBS or fullname.split(".")[0] in _HEADLESS_STUBS:
                return importlib.machinery.ModuleSpec(fullname, _HeadlessStubLoader())
            return None

    class _HeadlessStubLoader(importlib.abc.Loader):
        """Return a dummy module that raises ImportError on attribute access."""
        def create_module(self, spec):
            mod = types.ModuleType(spec.name)
            mod.__spec__ = spec
            mod.__loader__ = self
            mod.__path__ = []  # Allow submodule imports
            mod._is_headless_stub = True
            return mod

        def exec_module(self, module):
            pass

    sys.meta_path.insert(0, _HeadlessStubFinder())


# PID lock — prevent double-launch
PID_FILE = Path.home() / ".aura_daemon.pid"

# Log file — on servers, also write to /opt/aura/logs if available
_log_dir = Path(os.getenv("AURA_DATA_DIR", "data")).parent / "logs"
if _log_dir.exists():
    LOG_FILE = _log_dir / "aura_daemon.log"
else:
    LOG_FILE = Path.home() / ".aura_daemon.log"

from logging.handlers import RotatingFileHandler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [DAEMON] %(levelname)s %(name)s %(message)s",
    handlers=[
        RotatingFileHandler(LOG_FILE, maxBytes=10_000_000, backupCount=5, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


class AuraDaemon:
    """
    The living daemon.

    Tick schedule:
      5s   — screen hash check (cheap: perceptual hash only)
      30s  — hooks evaluation, system health, notifications
      5min — idle check, dream trigger consideration
      3 AM — full dream + memory consolidation
    """

    TICK_SCREEN   = 5     # seconds
    TICK_HOOKS    = 30
    TICK_IDLE     = 300   # 5 minutes
    TICK_PROACTIVE = 1800 # 30 minutes — ProactiveAwarenessEngine full analysis
    DREAM_HOUR    = 3     # 3 AM
    GEPA_HOUR     = 4     # 4 AM (after dream completes)
    GEPA_INTERVAL_DAYS = 7  # run weekly
    GEPA_MIN_SKILLS = 3   # skip if fewer than this many skills

    def __init__(self):
        self._running = False
        self._agent = None
        self._agent_load_error: Optional[str] = None
        self._agent_ready = threading.Event()
        self._load_agent_thread: Optional[threading.Thread] = None
        self._last_agent_attempt: float = 0.0  # monotonic time of last load attempt
        self._agent_retry_interval: float = 60.0  # seconds between retry attempts
        self._last_screen_hash = None
        self._last_activity = time.monotonic()
        self._event_bus = EventBus()
        self._proactive = ProactiveEngine(self._event_bus)
        self._ipc = IPCServer(self._event_bus)
        self._headless = HEADLESS
        self._start_time = time.time()

        # Screen monitoring — skip entirely on headless servers (uses shared bg_pool)
        self._screen_pool_enabled = not self._headless
        self._screen_tick_pending = False

        # Tick tracking (monotonic for interval measurement)
        self._last_hooks_tick = 0.0
        self._last_idle_tick = 0.0
        self._last_proactive_tick = 0.0
        self._proactive_running = False
        self._last_dream_date = None
        self._last_gepa_date = None
        self._gepa_running = False

        # Persist dream state across restarts
        self._state_file = Path(os.getenv("AURA_DATA_DIR", "data")) / "daemon_state.json"
        self._load_daemon_state()

    def start(self):
        """Start the daemon."""
        self._write_pid()
        self._running = True
        logger.info("AURA daemon starting... (headless=%s, pid=%d)", self._headless, os.getpid())

        # Load agent (lazy — don't block startup)
        self._load_agent_thread = threading.Thread(target=self._load_agent, daemon=True)
        self._load_agent_thread.start()

        # Start IPC server (CLI connects here)
        try:
            self._ipc.start()
        except Exception as e:
            logger.error("IPC server failed to start: %s", e)
            # Non-fatal — daemon can still run without IPC

        # Main loop
        try:
            self._run_loop()
        except KeyboardInterrupt:
            pass
        except Exception as e:
            logger.critical("Daemon main loop crashed: %s\n%s", e, traceback.format_exc())
        finally:
            self.stop()

    def stop(self):
        self._running = False
        try:
            self._event_bus.shutdown()
        except Exception:
            logger.debug("event_bus_shutdown_failed", exc_info=True)
        try:
            self._ipc.stop()
        except Exception:
            logger.debug("ipc_stop_failed", exc_info=True)
        self._remove_pid()
        logger.info("AURA daemon stopped.")

    def _run_loop(self):
        """Main heartbeat loop."""
        if self._headless:
            logger.info("Headless mode -- screen monitoring disabled")
        while self._running:
            now = time.monotonic()

            try:
                # Retry agent load if it failed and enough time has passed
                if (self._agent is None
                        and self._agent_ready.is_set()
                        and now - self._last_agent_attempt >= self._agent_retry_interval):
                    logger.info("Agent is None after failed load -- retrying _load_agent")
                    self._agent_ready.clear()
                    self._load_agent_thread = threading.Thread(target=self._load_agent, daemon=True)
                    self._load_agent_thread.start()

                # 5s: screen monitoring (non-blocking, skip if previous tick still running)
                # Skipped entirely in headless mode (no display)
                if self._screen_pool_enabled and not self._screen_tick_pending:
                    self._screen_tick_pending = True
                    from aura.pools import bg_submit
                    bg_submit(self._tick_screen_wrapper)

                # 30s: hooks + system health
                if now - self._last_hooks_tick >= self.TICK_HOOKS:
                    self._tick_hooks()
                    self._last_hooks_tick = now

                # 5min: idle check
                if now - self._last_idle_tick >= self.TICK_IDLE:
                    self._tick_idle()
                    self._last_idle_tick = now

                # 30min: proactive awareness full analysis
                if now - self._last_proactive_tick >= self.TICK_PROACTIVE:
                    self._tick_proactive()
                    self._last_proactive_tick = now

                # 3 AM: full dream
                self._check_dream_time()

                # 4 AM weekly: GEPA skill evolution
                self._check_gepa_time()

            except Exception as e:
                logger.error("Tick error (non-fatal): %s", e)

            time.sleep(self.TICK_SCREEN)

    def _tick_screen_wrapper(self):
        """Wrapper that clears the pending flag after _tick_screen completes."""
        try:
            self._tick_screen()
        finally:
            self._screen_tick_pending = False

    def _tick_screen(self):
        """Check screen for changes. ~2ms."""
        if not self._agent:
            return
        try:
            screen_tool = self._agent.tools.get("screenpipe") or self._agent.tools.get("screenshot")
            if not screen_tool:
                return

            # Try perceptual hash comparison
            new_hash = None
            if hasattr(screen_tool, 'get_screen_hash'):
                new_hash = screen_tool.get_screen_hash()
            elif hasattr(screen_tool, 'take_screenshot'):
                result = screen_tool.take_screenshot()
                new_hash = result.get("hash") if result else None

            if new_hash and new_hash != self._last_screen_hash:
                self._last_screen_hash = new_hash
                self._event_bus.emit("screen:changed", {"hash": new_hash})

                # Check for errors on screen
                if hasattr(screen_tool, 'detect_errors'):
                    errors = screen_tool.detect_errors()
                    if errors:
                        self._event_bus.emit("screen:error_detected", {"errors": errors})

        except Exception as e:
            logger.debug("Screen tick failed: %s", e)

    def _tick_hooks(self):
        """Run hooks evaluation."""
        if not self._agent:
            return
        try:
            if hasattr(self._agent, 'hooks') and self._agent.hooks:
                self._agent.hooks._check_triggers()
        except Exception as e:
            logger.debug("Hooks tick failed: %s", e)

    def _tick_idle(self):
        """Check idle state, maybe trigger light dream."""
        idle_secs = time.monotonic() - self._last_activity
        if idle_secs >= 1800:  # 30 minutes idle
            logger.info("Idle for %.0fmin -- triggering light dream", idle_secs / 60)
            self._event_bus.emit("daemon:idle", {"idle_seconds": idle_secs})
            if self._agent and hasattr(self._agent, 'neurodream') and self._agent.neurodream:
                threading.Thread(
                    target=self._safe_light_sleep,
                    daemon=True
                ).start()

    def _safe_light_sleep(self):
        """Run light sleep with error handling."""
        try:
            self._agent.neurodream.light_sleep()
        except Exception as e:
            logger.error("Light sleep failed: %s", e)

    def _check_dream_time(self):
        """Trigger full dream at 3 AM (once per day)."""
        now = datetime.now()
        today = now.date()
        if (now.hour == self.DREAM_HOUR and
                self._last_dream_date != today and
                self._agent is not None):
            self._last_dream_date = today
            self._save_daemon_state()
            logger.info("3 AM -- starting full dream cycle")
            self._event_bus.emit("daemon:dream_start", {})
            threading.Thread(target=self._run_full_dream, daemon=True).start()

    def _tick_proactive(self):
        """Trigger proactive awareness full analysis in background.

        Cheap guardrails: skip if no agent, skip if already running, skip if the
        world model hasn't accumulated at least 3 projects (noise floor).
        """
        if self._proactive_running or self._agent is None:
            return
        self._proactive_running = True
        logger.debug("[ProactiveAwareness] tick -- dispatching full analysis")
        threading.Thread(target=self._run_proactive_analysis, daemon=True).start()

    def _run_proactive_analysis(self):
        """Run ProactiveAwarenessEngine.run_full_analysis() off the main loop."""
        try:
            # Noise-floor guard: don't burn LLM calls on an empty world model.
            try:
                from aura.consciousness.world_model import get_world_model
                wm = get_world_model()
                projects = wm.get_all_projects() if hasattr(wm, "get_all_projects") else []
                if len(projects) < 3:
                    logger.debug(
                        "[ProactiveAwareness] skipped -- only %d projects tracked",
                        len(projects),
                    )
                    return
            except Exception as e:
                logger.debug("[ProactiveAwareness] noise-floor check failed: %s", e)

            from aura.consciousness.proactive_awareness import (
                get_proactive_awareness_engine,
            )
            engine = get_proactive_awareness_engine()
            insights = engine.run_full_analysis()
            logger.info(
                "[ProactiveAwareness] generated %d insight(s)",
                len(insights) if insights else 0,
            )
            self._event_bus.emit("daemon:proactive_insights", {
                "count": len(insights) if insights else 0,
                "titles": [getattr(i, "title", "?") for i in (insights or [])[:3]],
            })
        except ImportError as e:
            logger.debug("[ProactiveAwareness] engine unavailable: %s", e)
        except Exception as e:
            logger.error("[ProactiveAwareness] analysis failed: %s", e)
        finally:
            self._proactive_running = False

    def _check_gepa_time(self):
        """Trigger GEPA skill evolution at GEPA_HOUR, at most once per GEPA_INTERVAL_DAYS."""
        if self._gepa_running or self._agent is None:
            return
        now = datetime.now()
        today = now.date()
        if now.hour != self.GEPA_HOUR:
            return
        if self._last_gepa_date is not None:
            days_since = (today - self._last_gepa_date).days
            if days_since < self.GEPA_INTERVAL_DAYS:
                return
        # Mark immediately so re-entry within this hour is prevented
        self._gepa_running = True
        self._last_gepa_date = today
        self._save_daemon_state()
        logger.info("GEPA window reached -- starting weekly skill evolution")
        self._event_bus.emit("daemon:gepa_start", {})
        threading.Thread(target=self._run_gepa, daemon=True).start()

    def _run_gepa(self):
        """Run GEPA evolution in background. Guarded by _gepa_running flag."""
        try:
            # Preflight: skip if skill library is too small
            try:
                from aura_skill_library.skill_store import SkillStore
                store = SkillStore(
                    storage_path=str(Path(os.getenv("AURA_DATA_DIR", "data")).parent / "aura_data" / "skill_library")
                )
                skill_count = len(store.index) if store.index else 0
                if skill_count < self.GEPA_MIN_SKILLS:
                    logger.info(
                        "GEPA skipped -- only %d skills in library (need >= %d)",
                        skill_count, self.GEPA_MIN_SKILLS,
                    )
                    self._event_bus.emit("daemon:gepa_skipped", {"reason": "too_few_skills", "count": skill_count})
                    return
            except Exception as e:
                logger.warning("GEPA skill-count preflight failed, proceeding anyway: %s", e)

            from aura.evolution.runner import run_evolution
            result = run_evolution(skill_ids=None, dry_run=False)
            logger.info("GEPA complete: %s", json.dumps(result, default=str)[:500])
            self._event_bus.emit("daemon:gepa_complete", result)
        except ImportError as e:
            logger.warning("GEPA module not available: %s", e)
        except Exception as e:
            logger.error("GEPA failed: %s\n%s", e, traceback.format_exc())
            self._event_bus.emit("daemon:gepa_failed", {"error": str(e)})
        finally:
            self._gepa_running = False

    def _run_full_dream(self):
        """Run full dream + memory consolidation."""
        try:
            from aura.dream import run_dream_mode
            run_dream_mode()
            logger.info("Dream cycle complete")
            self._event_bus.emit("daemon:dream_complete", {})
        except ImportError as e:
            logger.warning("Dream module not available: %s", e)
        except Exception as e:
            logger.error("Dream failed: %s\n%s", e, traceback.format_exc())

        # Run DreamConsolidator pipeline (cluster -> summarize -> prune -> report)
        try:
            from aura.dream import get_dream_consolidator
            consolidator = get_dream_consolidator()
            # Get default user_id from agent if available
            user_id = "default_user"
            try:
                if self._agent and hasattr(self._agent, "user_id"):
                    user_id = self._agent.user_id or user_id
            except Exception:
                logger.debug("dream_user_id_lookup_failed", exc_info=True)
            consolidator.run_cycle_background(user_id=user_id)
            logger.info("DreamConsolidator cycle started in background (user=%s)", user_id)
        except ImportError as e:
            logger.warning("DreamConsolidator not available: %s", e)
        except Exception as e:
            logger.error("DreamConsolidator failed: %s", e)

    def _load_agent(self):
        """Load ApprenticeAgent in background -- doesn't block daemon start.

        On headless servers, catches ALL import errors gracefully so
        missing GUI/audio modules don't crash the daemon.
        """
        self._last_agent_attempt = time.monotonic()
        try:
            sys.path.insert(0, str(Path(__file__).parent))
            logger.info("Loading ApprenticeAgent (fast_init=True)...")
            from aura.agent import ApprenticeAgent
            self._agent = ApprenticeAgent(fast_init=True)
            tools_loaded = len(self._agent.tools) if hasattr(self._agent, 'tools') else 0
            logger.info("Agent loaded successfully (%d tools)", tools_loaded)
            self._agent_load_error = None
            self._event_bus.emit("daemon:agent_ready", {})

            # Wire IdlePresenceEngine with agent reference for Hands
            try:
                from aura.consciousness.idle_presence import get_idle_presence_engine
                idle = get_idle_presence_engine()
                idle.set_agent(self._agent)
            except Exception as e:
                logger.debug("IdlePresence agent wiring failed: %s", e)

            # Register and start Autonomous Hands
            self._start_hands()
        except ImportError as e:
            self._agent_load_error = f"ImportError: {e}"
            logger.error("Agent load failed (missing module): %s\n%s", e, traceback.format_exc())
            if self._headless:
                logger.warning(
                    "This may be a desktop-only module. The daemon will continue "
                    "running without the agent. Fix by installing the missing package "
                    "or ensuring it is excluded in requirements-server.txt."
                )
        except Exception as e:
            self._agent_load_error = str(e)
            logger.error("Agent load failed: %s\n%s", e, traceback.format_exc())
        finally:
            self._agent_ready.set()

    def _start_hands(self):
        """Register and start Autonomous Hands scheduler."""
        try:
            from aura.hands.collector import CollectorHand
            from aura.hands.guardian import GuardianHand
            from aura.hands.manager import get_hand_manager
            from aura.hands.memory_hand import MemoryHand
            from aura.hands.researcher import ResearcherHand

            manager = get_hand_manager()
            if not manager.list_hands():
                manager.register(ResearcherHand())
                manager.register(GuardianHand())
                manager.register(MemoryHand())
                manager.register(CollectorHand())

            def _get_idle_seconds():
                return time.monotonic() - self._last_activity

            def _get_drive_urgencies():
                try:
                    from aura.consciousness.intrinsic_motivation import get_intrinsic_motivation
                    engine = get_intrinsic_motivation()
                    if engine:
                        return engine.get_drives_summary()
                except Exception:
                    pass
                return {}

            manager.start_scheduler(
                brain=self._agent.brain,
                tools=getattr(self._agent, 'tools', {}),
                get_idle_seconds=_get_idle_seconds,
                get_drive_urgencies=_get_drive_urgencies,
            )
            logger.info("HandManager scheduler started with %d hands", len(manager.list_hands()))
        except Exception as e:
            logger.warning("HandManager startup failed: %s", e)

    def get_status(self) -> dict:
        """Return daemon status for health checks."""
        return {
            "running": self._running,
            "headless": self._headless,
            "uptime_seconds": int(time.time() - self._start_time),
            "agent_loaded": self._agent is not None,
            "agent_error": self._agent_load_error,
            "agent_tools": len(self._agent.tools) if self._agent and hasattr(self._agent, 'tools') else 0,
            "pid": os.getpid(),
        }

    def record_activity(self):
        """Call this when user interacts -- resets idle timer."""
        self._last_activity = time.monotonic()
        if self._agent and hasattr(self._agent, 'neurodream') and self._agent.neurodream:
            self._agent.neurodream.record_activity()

    # -- Daemon state persistence --------------------------------------------

    def _load_daemon_state(self):
        """Load persisted daemon state (last dream/GEPA dates)."""
        try:
            if self._state_file.exists():
                data = json.loads(self._state_file.read_text(encoding="utf-8"))
                from datetime import date as date_cls
                last_dream = data.get("last_dream_date")
                if last_dream:
                    self._last_dream_date = date_cls.fromisoformat(last_dream)
                last_gepa = data.get("last_gepa_date")
                if last_gepa:
                    self._last_gepa_date = date_cls.fromisoformat(last_gepa)
        except Exception as e:
            logger.debug("Failed to load daemon state: %s", e)

    def _save_daemon_state(self):
        """Persist daemon state to disk."""
        try:
            self._state_file.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "last_dream_date": self._last_dream_date.isoformat() if self._last_dream_date else None,
                "last_gepa_date": self._last_gepa_date.isoformat() if self._last_gepa_date else None,
            }
            self._state_file.write_text(json.dumps(data), encoding="utf-8")
        except Exception as e:
            logger.debug("Failed to save daemon state: %s", e)

    # -- PID management ------------------------------------------------------

    def _write_pid(self):
        try:
            PID_FILE.write_text(str(os.getpid()))
        except OSError as e:
            logger.error("Failed to write PID file: %s", e)

    def _remove_pid(self):
        try:
            PID_FILE.unlink(missing_ok=True)
        except OSError as e:
            logger.warning("Failed to remove PID file: %s", e)

    @staticmethod
    def is_running() -> bool:
        """Check if daemon is running. Handles TOCTOU race on PID file."""
        try:
            pid_text = PID_FILE.read_text().strip()
            pid = int(pid_text)
        except (FileNotFoundError, ValueError, OSError):
            return False

        # Verify process is actually alive
        try:
            import psutil
            try:
                proc = psutil.Process(pid)
                return proc.is_running() and proc.status() != psutil.STATUS_ZOMBIE
            except psutil.NoSuchProcess:
                return False
            except psutil.AccessDenied:
                return True  # Process exists but we can't inspect it
            except psutil.ZombieProcess:
                return False
        except ImportError:
            pass

        # Fallback: platform-specific process check
        try:
            if sys.platform == "win32":
                import subprocess
                result = subprocess.run(
                    ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                    capture_output=True, text=True, timeout=3,
                )
                return str(pid) in result.stdout
            else:
                # Linux/macOS: check /proc or send signal 0
                os.kill(pid, 0)
                return True
        except (ProcessLookupError, PermissionError):
            return False
        except Exception:
            logger.debug("pid_check_failed", exc_info=True)
            return False


class EventBus:
    """Simple in-process pub/sub event bus."""

    def __init__(self):
        self._handlers: dict = {}
        self._lock = threading.Lock()
        from aura.pools import bg_pool
        self._pool = bg_pool()

    def emit(self, event_type: str, data: dict = None):
        data = data or {}
        with self._lock:
            handlers = list(self._handlers.get(event_type, []))
            # Also check wildcard patterns like "screen:*"
            prefix = event_type.split(":")[0] + ":*"
            handlers += list(self._handlers.get(prefix, []))

        for handler in handlers:
            try:
                self._pool.submit(handler, event_type, data)
            except Exception as e:
                logger.debug("EventBus handler error: %s", e)

    def subscribe(self, pattern: str, handler):
        with self._lock:
            if pattern not in self._handlers:
                self._handlers[pattern] = []
            self._handlers[pattern].append(handler)

    def shutdown(self):
        """Clear all event handlers. Does NOT shut down the shared bg_pool()."""
        with self._lock:
            self._handlers.clear()


class ProactiveEngine:
    """
    Scores events and decides whether to surface proactive suggestions.
    Score 0.6+ triggers a notification/message.
    Rate-limited: 2-minute cooldown between proactive messages.
    """

    THRESHOLD = 0.6
    COOLDOWN = 120  # seconds

    EVENT_SCORES = {
        "screen:error_detected": 0.9,
        "screen:changed": 0.2,
        "daemon:idle": 0.3,
    }

    def __init__(self, event_bus: EventBus):
        self._event_bus = event_bus
        self._last_proactive = 0.0  # monotonic timestamp
        self._cooldown_lock = threading.Lock()
        event_bus.subscribe("screen:error_detected", self._on_event)
        event_bus.subscribe("daemon:agent_ready", self._on_agent_ready)

    def _on_event(self, event_type: str, data: dict):
        score = self.EVENT_SCORES.get(event_type, 0.1)
        if score >= self.THRESHOLD:
            self._maybe_surface(event_type, data, score)

    def _on_agent_ready(self, event_type: str, data: dict):
        logger.info("Proactive engine: agent ready, monitoring active")

    def _maybe_surface(self, event_type: str, data: dict, score: float):
        now = time.monotonic()
        with self._cooldown_lock:
            if now - self._last_proactive < self.COOLDOWN:
                return
            self._last_proactive = now
        self._event_bus.emit("proactive:suggestion", {
            "trigger": event_type,
            "score": score,
            "data": data,
        })
        logger.info("Proactive suggestion triggered by %s (score=%.2f)", event_type, score)


class IPCServer:
    """
    Named pipe IPC -- CLI connects here to send messages to daemon.
    Falls back to TCP on localhost:19733 if named pipe unavailable.
    """

    PIPE_NAME = r"\\.\pipe\aura_daemon"
    TCP_PORT = 19733

    def __init__(self, event_bus: EventBus):
        self._event_bus = event_bus
        self._thread = None
        self._running = False
        self._auth_token = self._generate_auth_token()

    def _generate_auth_token(self) -> str:
        """Generate a random auth token and write it to a restricted file."""
        import secrets
        import stat
        token = secrets.token_hex(32)
        token_path = Path(os.getenv("AURA_DATA_DIR", "data")) / "ipc_token"
        token_path.parent.mkdir(parents=True, exist_ok=True)
        token_path.write_text(token, encoding="utf-8")
        try:
            token_path.chmod(stat.S_IRUSR | stat.S_IWUSR)  # 0o600
        except OSError:
            # Windows: POSIX permissions not supported — use icacls to restrict
            if sys.platform == "win32":
                try:
                    import subprocess
                    # Remove inherited permissions, grant only current user
                    subprocess.run(
                        ["icacls", str(token_path), "/inheritance:r",
                         "/grant:r", f"{os.getenv('USERNAME', 'OWNER')}:(R,W)"],
                        capture_output=True, timeout=5,
                    )
                except Exception as e:
                    logger.warning("IPC token file permissions could not be restricted: %s", e)
        logger.info("IPC auth token written to %s", token_path)
        return token

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False

    def _serve(self):
        """Try named pipe first, fall back to TCP."""
        try:
            self._serve_tcp()
        except OSError as e:
            # Port already in use — likely stale PID file, non-fatal
            logger.warning("IPC server could not bind (port in use?): %s", e)
        except Exception as e:
            logger.error("IPC server failed: %s", e)

    def _serve_tcp(self):
        import socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
            srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            srv.bind(("127.0.0.1", self.TCP_PORT))
            srv.listen(5)
            srv.settimeout(1.0)
            logger.info("IPC listening on TCP 127.0.0.1:%d", self.TCP_PORT)
            while self._running:
                try:
                    conn, _ = srv.accept()
                    threading.Thread(
                        target=self._handle_client, args=(conn,), daemon=True
                    ).start()
                except socket.timeout:
                    continue

    # Allowlist of valid IPC command types
    ALLOWED_IPC_TYPES = {
        "message", "chat", "query", "ping", "status",
        "activity", "stop", "dream", "config",
    }

    def _handle_client(self, conn):
        import socket
        try:
            conn.settimeout(10.0)
            chunks = []
            while True:
                try:
                    chunk = conn.recv(4096)
                except socket.timeout:
                    logger.debug("IPC client recv timed out")
                    break
                if not chunk:
                    break
                chunks.append(chunk)
                if b"\n" in chunk or len(b"".join(chunks)) > 65536:
                    break
            data = b"".join(chunks).decode("utf-8").strip()
            if data:
                msg = json.loads(data)
                # Validate auth token before processing any command
                import hmac
                msg_token = msg.get("token", "")
                if not msg_token or not self._auth_token or not hmac.compare_digest(msg_token, self._auth_token):
                    logger.warning("IPC rejected: invalid auth token")
                    conn.send(json.dumps({"status": "error", "reason": "invalid auth token"}).encode())
                    return
                msg_type = msg.get("type", "message")
                if msg_type not in self.ALLOWED_IPC_TYPES:
                    logger.warning("IPC rejected unknown type: %s", msg_type)
                    conn.send(json.dumps({"status": "error", "reason": "unknown command type"}).encode())
                    return
                self._event_bus.emit(f"ipc:{msg_type}", msg)
                conn.send(json.dumps({"status": "ok"}).encode())
        except Exception as e:
            logger.debug("IPC client error: %s", e)
        finally:
            conn.close()


def install_service():
    """Install AURA daemon as Windows service via NSSM."""
    import subprocess
    nssm = "nssm"
    script = str(Path(__file__).resolve())
    python = sys.executable

    cmds = [
        [nssm, "install", "AuraDaemon", python, script],
        [nssm, "set", "AuraDaemon", "AppDirectory", str(Path(__file__).parent)],
        [nssm, "set", "AuraDaemon", "Start", "SERVICE_AUTO_START"],
        [nssm, "set", "AuraDaemon", "AppStdout", str(LOG_FILE)],
        [nssm, "set", "AuraDaemon", "AppStderr", str(LOG_FILE)],
    ]
    for cmd in cmds:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Failed: {' '.join(cmd)}\n{result.stderr}")
            return False
        print(f"OK: {' '.join(cmd[1:3])}")
    print("\nService installed. Start with: nssm start AuraDaemon")
    return True


def main():
    parser = argparse.ArgumentParser(description="AURA Living Daemon")
    parser.add_argument("--install", action="store_true", help="Install as Windows service")
    parser.add_argument("--status", action="store_true", help="Check daemon status")
    args = parser.parse_args()

    if args.status:
        running = AuraDaemon.is_running()
        print(f"Daemon: {'RUNNING' if running else 'STOPPED'}")
        sys.exit(0)

    if args.install:
        success = install_service()
        sys.exit(0 if success else 1)

    if AuraDaemon.is_running():
        print("Daemon already running.")
        sys.exit(1)

    daemon = AuraDaemon()

    # Wire OS signals to daemon shutdown (replaces duplicated startup logic)
    def _signal_handler(signum, frame):
        daemon._running = False

    signal.signal(signal.SIGTERM, _signal_handler)
    if hasattr(signal, "SIGINT"):
        signal.signal(signal.SIGINT, _signal_handler)

    # Use daemon.start() — no duplication of _write_pid, _load_agent, _ipc.start
    daemon.start()


if __name__ == "__main__":
    main()
