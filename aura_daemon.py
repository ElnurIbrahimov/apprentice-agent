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

import sys
import time
import json
import signal
import logging
import threading
import argparse
from pathlib import Path
from datetime import datetime

# PID lock — prevent double-launch
PID_FILE = Path.home() / ".aura_daemon.pid"
LOG_FILE = Path.home() / ".aura_daemon.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [DAEMON] %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
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
    DREAM_HOUR    = 3     # 3 AM

    def __init__(self):
        self._running = False
        self._agent = None
        self._last_screen_hash = None
        self._last_activity = time.time()
        self._event_bus = EventBus()
        self._proactive = ProactiveEngine(self._event_bus)
        self._ipc = IPCServer(self._event_bus)

        # Tick tracking
        self._last_hooks_tick = 0.0
        self._last_idle_tick = 0.0
        self._last_dream_date = None

    def start(self):
        """Start the daemon."""
        self._write_pid()
        self._running = True
        logger.info("AURA daemon starting...")

        # Load agent (lazy — don't block startup)
        threading.Thread(target=self._load_agent, daemon=True).start()

        # Start IPC server (CLI connects here)
        self._ipc.start()

        # Main loop
        try:
            self._run_loop()
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()

    def stop(self):
        self._running = False
        self._ipc.stop()
        self._remove_pid()
        logger.info("AURA daemon stopped.")

    def _run_loop(self):
        """Main heartbeat loop."""
        while self._running:
            now = time.time()

            # 5s: screen monitoring
            self._tick_screen()

            # 30s: hooks + system health
            if now - self._last_hooks_tick >= self.TICK_HOOKS:
                self._tick_hooks()
                self._last_hooks_tick = now

            # 5min: idle check
            if now - self._last_idle_tick >= self.TICK_IDLE:
                self._tick_idle()
                self._last_idle_tick = now

            # 3 AM: full dream
            self._check_dream_time()

            time.sleep(self.TICK_SCREEN)

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
            logger.debug(f"Screen tick failed: {e}")

    def _tick_hooks(self):
        """Run hooks evaluation."""
        if not self._agent:
            return
        try:
            if hasattr(self._agent, 'hooks') and self._agent.hooks:
                self._agent.hooks._check_triggers()
        except Exception as e:
            logger.debug(f"Hooks tick failed: {e}")

    def _tick_idle(self):
        """Check idle state, maybe trigger light dream."""
        idle_secs = time.time() - self._last_activity
        if idle_secs >= 1800:  # 30 minutes idle
            logger.info(f"Idle for {idle_secs/60:.0f}min — triggering light dream")
            self._event_bus.emit("daemon:idle", {"idle_seconds": idle_secs})
            if self._agent and hasattr(self._agent, 'neurodream') and self._agent.neurodream:
                threading.Thread(
                    target=self._agent.neurodream.light_sleep,
                    daemon=True
                ).start()

    def _check_dream_time(self):
        """Trigger full dream at 3 AM (once per day)."""
        now = datetime.now()
        today = now.date()
        if (now.hour == self.DREAM_HOUR and
                self._last_dream_date != today and
                self._agent is not None):
            self._last_dream_date = today
            logger.info("3 AM — starting full dream cycle")
            self._event_bus.emit("daemon:dream_start", {})
            threading.Thread(target=self._run_full_dream, daemon=True).start()

    def _run_full_dream(self):
        """Run full dream + memory consolidation."""
        try:
            from aura.dream import run_dream_mode
            run_dream_mode()
            logger.info("Dream cycle complete")
            self._event_bus.emit("daemon:dream_complete", {})
        except Exception as e:
            logger.error(f"Dream failed: {e}")

    def _load_agent(self):
        """Load ApprenticeAgent in background — doesn't block daemon start."""
        try:
            sys.path.insert(0, str(Path(__file__).parent))
            from aura.agent import ApprenticeAgent
            self._agent = ApprenticeAgent(fast_init=True)
            logger.info("Agent loaded successfully")
            self._event_bus.emit("daemon:agent_ready", {})
        except Exception as e:
            logger.error(f"Agent load failed: {e}")

    def record_activity(self):
        """Call this when user interacts — resets idle timer."""
        self._last_activity = time.time()
        if self._agent and hasattr(self._agent, 'neurodream') and self._agent.neurodream:
            self._agent.neurodream.record_activity()

    # ── PID management ──────────────────────────────────────────────────────

    def _write_pid(self):
        PID_FILE.write_text(str(os.getpid()))

    def _remove_pid(self):
        PID_FILE.unlink(missing_ok=True)

    @staticmethod
    def is_running() -> bool:
        if not PID_FILE.exists():
            return False
        pid = int(PID_FILE.read_text().strip())
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False


class EventBus:
    """Simple in-process pub/sub event bus."""

    def __init__(self):
        self._handlers: dict = {}
        self._lock = threading.Lock()

    def subscribe(self, pattern: str, handler):
        with self._lock:
            if pattern not in self._handlers:
                self._handlers[pattern] = []
            self._handlers[pattern].append(handler)

    def emit(self, event_type: str, data: dict = None):
        data = data or {}
        with self._lock:
            handlers = list(self._handlers.get(event_type, []))
            # Also check wildcard patterns like "screen:*"
            prefix = event_type.split(":")[0] + ":*"
            handlers += list(self._handlers.get(prefix, []))

        for handler in handlers:
            try:
                threading.Thread(
                    target=handler, args=(event_type, data), daemon=True
                ).start()
            except Exception as e:
                logger.debug(f"EventBus handler error: {e}")


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
        self._last_proactive = 0.0
        event_bus.subscribe("screen:error_detected", self._on_event)
        event_bus.subscribe("daemon:agent_ready", self._on_agent_ready)

    def _on_event(self, event_type: str, data: dict):
        score = self.EVENT_SCORES.get(event_type, 0.1)
        if score >= self.THRESHOLD:
            self._maybe_surface(event_type, data, score)

    def _on_agent_ready(self, event_type: str, data: dict):
        logger.info("Proactive engine: agent ready, monitoring active")

    def _maybe_surface(self, event_type: str, data: dict, score: float):
        now = time.time()
        if now - self._last_proactive < self.COOLDOWN:
            return
        self._last_proactive = now
        self._event_bus.emit("proactive:suggestion", {
            "trigger": event_type,
            "score": score,
            "data": data,
        })
        logger.info(f"Proactive suggestion triggered by {event_type} (score={score:.2f})")


class IPCServer:
    """
    Named pipe IPC — CLI connects here to send messages to daemon.
    Falls back to TCP on localhost:19733 if named pipe unavailable.
    """

    PIPE_NAME = r"\\.\pipe\aura_daemon"
    TCP_PORT = 19733

    def __init__(self, event_bus: EventBus):
        self._event_bus = event_bus
        self._thread = None
        self._running = False

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
        except Exception as e:
            logger.error(f"IPC server failed: {e}")

    def _serve_tcp(self):
        import socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
            srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            srv.bind(("127.0.0.1", self.TCP_PORT))
            srv.listen(5)
            srv.settimeout(1.0)
            logger.info(f"IPC listening on TCP 127.0.0.1:{self.TCP_PORT}")
            while self._running:
                try:
                    conn, _ = srv.accept()
                    threading.Thread(
                        target=self._handle_client, args=(conn,), daemon=True
                    ).start()
                except socket.timeout:
                    continue

    def _handle_client(self, conn):
        try:
            data = conn.recv(4096).decode("utf-8").strip()
            if data:
                msg = json.loads(data)
                self._event_bus.emit(f"ipc:{msg.get('type', 'message')}", msg)
                conn.send(json.dumps({"status": "ok"}).encode())
        except Exception:
            pass
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
    signal.signal(signal.SIGTERM, lambda s, f: daemon.stop())
    daemon.start()


if __name__ == "__main__":
    main()
