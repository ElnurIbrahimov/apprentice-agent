# AURA Daemon Architecture: The "Living" System

**Date:** 2026-02-28
**Status:** Design Document
**Goal:** Make AURA a continuously running, proactive presence on Elnur's machine.

---

## 1. Executive Summary

AURA currently runs only when invoked (CLI chat, goal mode, or API server). This design turns AURA into an **always-on daemon** that starts at boot, monitors the environment, acts proactively, dreams during idle time, and communicates with any CLI/GUI/API client through a shared event bus.

The architecture has four layers:

```
+---------------------------------------------------+
|              CLI / GUI / API Clients               |  <-- User-facing
+---------------------------------------------------+
|               IPC Layer (Named Pipe)               |  <-- Communication
+---------------------------------------------------+
|              Event Bus (Pub/Sub)                   |  <-- Routing
+---------------------------------------------------+
|            Daemon Core (Heartbeat Loop)            |  <-- Always running
|  +--------+  +---------+  +-------+  +---------+  |
|  | Screen |  | System  |  | File  |  | Schedule|  |
|  |Monitor |  |Monitor  |  |Watch  |  | Timer   |  |
|  +--------+  +---------+  +-------+  +---------+  |
|  +--------+  +---------+  +-------+              |
|  | Dream  |  | Hooks   |  | Notif |              |
|  |Engine  |  |Manager  |  |Checker|              |
|  +--------+  +---------+  +-------+              |
+---------------------------------------------------+
```

---

## 2. What Already Exists (Inventory)

Before designing, here is what AURA already has that the daemon will orchestrate:

| Component | File | What It Does | Daemon Role |
|---|---|---|---|
| **ScreenReaderTool** | `aura/tools/screen_reader.py` | OCR via Florence-2, `ChangeDetector` with PIL diff, `watch()` for keywords | Screen monitor sensor |
| **ScreenpipeClient** | `aura/tools/screenpipe.py` | Queries Screenpipe REST API (localhost:3030), has `has_significant_change()`, privacy filtering, perceptual hashing | Primary screen sensor (if Screenpipe running) |
| **ScreenshotTool** | `aura/tools/screenshot.py` | Captures screenshots via `mss` | On-demand capture for events |
| **SystemControlTool** | `aura/tools/system_control.py` | Volume, brightness, app launch, GPU/CPU/RAM via `psutil` and `nvidia-smi` | System health sensor |
| **NotificationTool** | `aura/tools/notifications.py` | Reminders, scheduled tasks, conditional alerts. JSON-persisted. | Schedule/alert source |
| **HooksManager** | `aura/hooks.py` | Event-driven hooks (schedule, file_modified, system_alert, clipboard_changed, keyword_on_screen). Already has `start_background()` with a daemon thread on a 15s loop. | Already a mini-daemon. Becomes a subsystem. |
| **NeuroDreamEngine** | `aura/tools/neurodream.py` | Sleep phases (light/deep/REM), memory replay, pattern abstraction, creative synthesis. Has `check_idle_trigger()`, runs in background thread, neural oscillator. | Dream/consolidation subsystem |
| **DreamMode** | `aura/dream.py` | Analyzes metacognition logs, generates insights, A-MEM consolidation. | Nightly consolidation task |
| **MemoryConsolidator** | `aura_episodic_memory/consolidation.py` | Importance decay, episode merging, garbage collection, summary generation. | Scheduled maintenance |
| **FastAPI + WebSocket** | `api/main.py`, `api/routes/chat.py` | HTTP/WS API with lifespan, active WebSocket tracking, streaming. | IPC alternative / GUI bridge |

**Key observation:** HooksManager is already a daemon thread. NeuroDream has idle detection and background threading. The pieces exist -- they just need a central orchestrator.

---

## 3. Boot & Lifecycle: How AURA Starts and Stays Alive

### 3.1 Service Installation Strategy

**Primary approach: NSSM (Non-Sucking Service Manager)**

NSSM is the right choice for AURA on Windows because:
- It runs before any user logs in (true service, not Task Scheduler)
- Auto-restarts on crash (configurable restart delay)
- Logs stdout/stderr to files automatically
- Zero code changes needed -- just wraps the Python process
- Already proven for Python services in production

```bash
# One-time installation (run as admin)
nssm install AuraDaemon "C:\Users\asus\AppData\Local\Programs\Python\Python312\python.exe"
nssm set AuraDaemon AppParameters "D:\Aura\aura_daemon.py"
nssm set AuraDaemon AppDirectory "D:\Aura"
nssm set AuraDaemon DisplayName "AURA Daemon"
nssm set AuraDaemon Description "AURA autonomous AI assistant background service"
nssm set AuraDaemon Start SERVICE_AUTO_START
nssm set AuraDaemon AppStdout "D:\Aura\logs\daemon_stdout.log"
nssm set AuraDaemon AppStderr "D:\Aura\logs\daemon_stderr.log"
nssm set AuraDaemon AppRotateFiles 1
nssm set AuraDaemon AppRotateBytes 10485760  # 10MB rotation
nssm set AuraDaemon AppRestartDelay 5000  # 5s restart on crash
nssm start AuraDaemon
```

**Fallback: Task Scheduler** (if user doesn't want admin/service overhead)

```bash
# Register via schtasks for user login trigger
schtasks /create /tn "AuraDaemon" /tr "python D:\Aura\aura_daemon.py" /sc onlogon /rl highest
```

**Fallback 2: Startup folder shortcut** (simplest, least reliable)

### 3.2 Process Lifecycle

```
Boot/Login
    |
    v
aura_daemon.py starts
    |
    v
[1] Load config
[2] Acquire PID lock (prevent double-launch)
[3] Initialize logging
[4] Start event bus
[5] Start sensors (screen, system, file, schedule)
[6] Start IPC server (named pipe)
[7] Enter heartbeat loop
    |
    +---> Running... (indefinite)
    |
    v (on shutdown signal)
[8] Graceful shutdown
    - Stop sensors
    - Flush event queue
    - Run emergency consolidation
    - Release PID lock
    - Exit 0
```

### 3.3 PID Lock (Single Instance)

```python
# D:\Aura\data\daemon.pid
import os, sys

PID_FILE = "D:/Aura/data/daemon.pid"

def acquire_lock():
    """Ensure only one daemon instance runs."""
    if os.path.exists(PID_FILE):
        with open(PID_FILE) as f:
            old_pid = int(f.read().strip())
        # Check if process still running
        try:
            os.kill(old_pid, 0)  # Signal 0 = check existence
            print(f"Daemon already running (PID {old_pid}). Exiting.")
            sys.exit(1)
        except OSError:
            pass  # Stale PID file, safe to proceed

    with open(PID_FILE, "w") as f:
        f.write(str(os.getpid()))

def release_lock():
    if os.path.exists(PID_FILE):
        os.remove(PID_FILE)
```

---

## 4. The Heartbeat Loop

The heartbeat is the daemon's central tick. It runs every **N seconds** and orchestrates all subsystems.

### 4.1 Tick Intervals (Tiered)

Not everything needs to check at the same rate. The heartbeat uses tiered intervals:

| Tier | Interval | What Runs | CPU Impact |
|------|----------|-----------|------------|
| **Fast** | 5s | Screen change detection (hash only), IPC message drain | Negligible (<1%) |
| **Medium** | 30s | Hooks check, notification check, system health snapshot | Low (~1-2%) |
| **Slow** | 5 min | Idle detection, dream trigger check, memory health | Negligible |
| **Hourly** | 60 min | Full system report, log rotation | Negligible |
| **Nightly** | 3:00 AM | Full dream cycle, memory consolidation, A-MEM cleanup | Moderate (1-2 min) |

### 4.2 Heartbeat Implementation

```python
import asyncio
import time

class Heartbeat:
    """Central daemon tick with tiered scheduling."""

    def __init__(self, event_bus):
        self.bus = event_bus
        self.last_activity = time.time()
        self.running = True

        # Counters for tiered execution
        self._tick_count = 0
        self._last_medium = 0
        self._last_slow = 0
        self._last_hourly = 0

    async def run(self):
        """Main heartbeat loop."""
        while self.running:
            now = time.time()
            self._tick_count += 1

            # FAST tier (every tick = 5s)
            await self._fast_tick()

            # MEDIUM tier (every 30s)
            if now - self._last_medium >= 30:
                await self._medium_tick()
                self._last_medium = now

            # SLOW tier (every 5 min)
            if now - self._last_slow >= 300:
                await self._slow_tick()
                self._last_slow = now

            # HOURLY tier
            if now - self._last_hourly >= 3600:
                await self._hourly_tick()
                self._last_hourly = now

            await asyncio.sleep(5)  # Base tick interval

    async def _fast_tick(self):
        """5-second checks: screen hash, IPC drain."""
        self.bus.emit("tick:fast", {"tick": self._tick_count})

    async def _medium_tick(self):
        """30-second checks: hooks, notifications, system health."""
        self.bus.emit("tick:medium", {"tick": self._tick_count})

    async def _slow_tick(self):
        """5-minute checks: idle detection, dream trigger."""
        idle_seconds = time.time() - self.last_activity
        self.bus.emit("tick:slow", {
            "idle_seconds": idle_seconds,
            "tick": self._tick_count
        })

    async def _hourly_tick(self):
        """Hourly: system report, log rotation."""
        self.bus.emit("tick:hourly", {"tick": self._tick_count})

    def record_activity(self):
        """Called when any user interaction is detected."""
        self.last_activity = time.time()
```

---

## 5. Screen Monitoring: Efficient Change Detection

### 5.1 The Problem

Full OCR + LLM analysis every second would kill CPU/GPU. The solution is a **three-tier escalation**:

```
Tier 1: Perceptual Hash (5s) -- is anything visually different?
   |
   | (yes, hash distance > threshold)
   v
Tier 2: Region Diff (immediate) -- what changed? how much?
   |
   | (significant: > 15% pixel change)
   v
Tier 3: OCR + Semantic Analysis (on-demand) -- what does it mean?
```

### 5.2 Tier 1: Perceptual Hash Comparison

Uses `imagehash.dhash` on downscaled screenshots. Costs ~2ms per frame.

```python
import mss
import imagehash
from PIL import Image
from io import BytesIO

class ScreenSensor:
    """Efficient screen change detection using perceptual hashing."""

    def __init__(self, hash_size=8, change_threshold=12):
        self._sct = mss.mss()
        self._last_hash = None
        self._last_screenshot = None
        self.hash_size = hash_size
        self.change_threshold = change_threshold  # Hamming distance

    def check_change(self) -> dict:
        """Fast hash-based screen change detection.

        Returns:
            {"changed": bool, "distance": int, "screenshot": Image or None}
        """
        # Capture primary monitor
        monitor = self._sct.monitors[1]
        raw = self._sct.grab(monitor)

        # Convert to PIL and downscale for hashing (very fast)
        img = Image.frombytes("RGB", raw.size, raw.rgb)
        small = img.resize((64, 64))  # Tiny image for hash

        # Compute dHash
        current_hash = imagehash.dhash(small, hash_size=self.hash_size)

        if self._last_hash is None:
            self._last_hash = current_hash
            self._last_screenshot = img
            return {"changed": False, "distance": 0, "screenshot": None}

        distance = current_hash - self._last_hash
        changed = distance > self.change_threshold

        result = {
            "changed": changed,
            "distance": distance,
            "screenshot": img if changed else None
        }

        self._last_hash = current_hash
        if changed:
            self._last_screenshot = img

        return result
```

**Why this works:**
- dHash on a 64x64 image takes ~2ms
- Hamming distance comparison is a single integer subtraction
- Only when distance exceeds threshold do we proceed to heavier analysis
- Clock updates, cursor blinks, and minor UI animations produce distance < 5
- App switches, new content, error dialogs produce distance > 15

### 5.3 Tier 2: Region Diff (When Hash Changes)

When Tier 1 detects a change, compute a pixel-level diff to understand *what* changed:

```python
import numpy as np
from PIL import ImageChops

def compute_region_diff(old_img, new_img, grid=(4, 4)):
    """Divide screen into grid and find which regions changed.

    Returns list of (row, col, change_ratio) for changed regions.
    """
    w, h = new_img.size
    cell_w, cell_h = w // grid[0], h // grid[1]

    diff = ImageChops.difference(old_img.convert("RGB"), new_img.convert("RGB"))
    diff_array = np.array(diff)

    changed_regions = []
    for row in range(grid[1]):
        for col in range(grid[0]):
            region = diff_array[
                row * cell_h:(row + 1) * cell_h,
                col * cell_w:(col + 1) * cell_w
            ]
            change_ratio = float(np.mean(region > 30))
            if change_ratio > 0.05:  # 5% of pixels in region changed
                changed_regions.append((row, col, change_ratio))

    return changed_regions
```

### 5.4 Tier 3: OCR + Semantic Analysis (When Significant Change)

Only when region diff shows > 15% change do we invoke OCR (Florence-2 or pytesseract) and optionally ask an LLM to interpret the screen.

This tier also integrates with **Screenpipe** when available. If Screenpipe is running (localhost:3030), the daemon uses its continuous OCR stream instead of doing its own -- Screenpipe is already optimized for 24/7 capture.

```python
# In the daemon, the screen sensor checks Screenpipe first:
def get_screen_context(self):
    """Get screen context, preferring Screenpipe if available."""
    if self.screenpipe and self.screenpipe.is_available():
        return self.screenpipe.get_screen_context_filtered(
            minutes=1,
            only_if_changed=True
        )
    else:
        # Fallback to built-in OCR
        return self.screen_reader.read_screen()
```

### 5.5 Screenpipe Integration Detail

The existing `ScreenpipeClient` already has:
- `has_significant_change()` with content hashing + perceptual hashing
- `get_screen_context_filtered()` with privacy filtering and delta detection
- Error indicator scanning (traceback, 404, 500, crash, etc.)

The daemon wraps this as a sensor that emits events to the bus.

---

## 6. Event System: What Triggers What

### 6.1 Event Bus Design

A lightweight in-process pub/sub. No external dependencies (no Redis, no Kafka -- this is a single-machine daemon).

```python
import asyncio
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List

@dataclass
class Event:
    """An event in the AURA daemon."""
    type: str                    # e.g., "screen:changed", "system:cpu_high"
    data: Dict[str, Any]         # Event payload
    timestamp: str = ""          # ISO timestamp
    source: str = ""             # Which sensor emitted it
    priority: int = 5            # 1 (critical) to 10 (info)

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


class EventBus:
    """In-process async event bus for the daemon."""

    def __init__(self):
        self._subscribers: Dict[str, List[Callable]] = defaultdict(list)
        self._queue: asyncio.Queue = None  # Set in async context
        self._history: List[Event] = []    # Last 1000 events
        self._max_history = 1000

    def subscribe(self, event_pattern: str, handler: Callable):
        """Subscribe to events matching a pattern.

        Patterns:
            "screen:changed"       -- exact match
            "screen:*"             -- all screen events
            "*"                    -- all events
        """
        self._subscribers[event_pattern].append(handler)

    def emit(self, event_type: str, data: dict = None, source: str = "", priority: int = 5):
        """Emit an event to all matching subscribers."""
        event = Event(
            type=event_type,
            data=data or {},
            source=source,
            priority=priority
        )

        self._history.append(event)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]

        # Match subscribers
        for pattern, handlers in self._subscribers.items():
            if self._matches(pattern, event_type):
                for handler in handlers:
                    try:
                        result = handler(event)
                        if asyncio.iscoroutine(result):
                            asyncio.create_task(result)
                    except Exception as e:
                        pass  # Log but don't crash

    def _matches(self, pattern: str, event_type: str) -> bool:
        if pattern == "*":
            return True
        if pattern.endswith(":*"):
            prefix = pattern[:-2]
            return event_type.startswith(prefix + ":")
        return pattern == event_type

    def get_recent(self, event_type: str = None, limit: int = 50) -> List[Event]:
        """Get recent events, optionally filtered by type."""
        if event_type:
            return [e for e in self._history if e.type == event_type][-limit:]
        return self._history[-limit:]
```

### 6.2 Event Catalog

All events the daemon can produce and consume:

```
SCREEN EVENTS
    screen:changed              -- Visual hash threshold exceeded
    screen:error_detected       -- Error/traceback/crash text found on screen
    screen:app_switched         -- User switched to a different application
    screen:keyword_found        -- Watched keyword appeared (from hooks)
    screen:idle                 -- Screen hasn't changed for X minutes

SYSTEM EVENTS
    system:cpu_high             -- CPU usage > threshold (default 85%)
    system:ram_high             -- RAM usage > threshold (default 90%)
    system:disk_high            -- Disk usage > threshold (default 95%)
    system:gpu_hot              -- GPU temp > threshold (default 85C)
    system:battery_low          -- Battery < 20% (laptops)
    system:process_crash        -- Monitored process exited unexpectedly

FILE EVENTS
    file:modified               -- Watched file changed (from hooks)
    file:created                -- New file in watched directory
    file:large_download         -- Large file appeared in Downloads

SCHEDULE EVENTS
    schedule:reminder_due       -- A reminder's fire_at time reached
    schedule:recurring_due      -- A recurring scheduled notification fires
    schedule:nightly            -- 3 AM nightly maintenance window

CLIPBOARD EVENTS
    clipboard:changed           -- Clipboard content changed
    clipboard:keyword_match     -- Clipboard matches a watched keyword

USER EVENTS (from IPC)
    user:message                -- User sent a message via CLI/GUI
    user:command                -- User issued a slash command
    user:activity               -- Any user interaction (resets idle timer)

DAEMON EVENTS
    daemon:started              -- Daemon process started
    daemon:heartbeat            -- Periodic heartbeat signal
    daemon:shutting_down        -- Graceful shutdown initiated
    daemon:error                -- Internal daemon error

DREAM EVENTS
    dream:idle_triggered        -- Idle threshold reached, dream starting
    dream:phase_changed         -- Sleep phase transition (light/deep/REM)
    dream:insight_generated     -- REM phase produced a novel insight
    dream:complete              -- Dream cycle finished
    dream:consolidation_done    -- Memory consolidation completed

PROACTIVE EVENTS (daemon -> user)
    proactive:suggestion        -- AURA has a suggestion for the user
    proactive:error_help        -- AURA detected an error and wants to help
    proactive:reminder          -- Proactive reminder based on context
    proactive:summary           -- Periodic activity summary
```

### 6.3 Event Flow Example: Error Detected on Screen

```
1. [ScreenSensor] 5s tick -> dHash comparison -> distance=47 (big change)
2. [ScreenSensor] Tier 2 region diff -> top-right quadrant, 62% changed
3. [ScreenSensor] Tier 3 OCR -> "Traceback (most recent call last): ..."
4. [ScreenSensor] emits -> screen:error_detected {text: "Traceback...", app: "VS Code"}
5. [EventBus] routes to ProactiveEngine subscriber
6. [ProactiveEngine] checks: is user in a coding app? YES
7. [ProactiveEngine] analyzes error with fast LLM (qwen2.5-coder:7b)
8. [ProactiveEngine] emits -> proactive:error_help {suggestion: "...", severity: "medium"}
9. [IPCServer] pushes to connected CLI client as notification
10. [CLI] shows: "[AURA] Detected a TypeError in your terminal. The issue is..."
```

---

## 7. Proactive Response Engine

### 7.1 What Triggers Proactive Action

Not every event should produce a proactive response. The engine uses a scoring system:

```python
class ProactiveEngine:
    """Decides when and how AURA should proactively intervene."""

    # Minimum score to trigger proactive response (0-1)
    INTERVENTION_THRESHOLD = 0.6

    def score_event(self, event: Event) -> float:
        """Score how much this event warrants proactive intervention."""
        score = 0.0

        # Error on screen while coding -> high value
        if event.type == "screen:error_detected":
            score += 0.7
            if "traceback" in event.data.get("text", "").lower():
                score += 0.2

        # App switch to browser after long coding session -> maybe needs a break
        if event.type == "screen:app_switched":
            score += 0.1

        # System resource critical
        if event.type in ("system:cpu_high", "system:ram_high"):
            score += 0.5

        # User idle for a long time -> suggest dream or break
        if event.type == "screen:idle":
            idle_min = event.data.get("idle_minutes", 0)
            if idle_min > 30:
                score += 0.4

        # Cooldown: don't spam the user
        # (check time since last proactive message)

        return min(1.0, score)
```

### 7.2 Response Types

| Trigger | Response | Delivery |
|---------|----------|----------|
| Error on screen | Analyze + suggest fix | Toast notification + IPC |
| CPU/RAM critical | Identify heavy process, suggest action | Toast notification |
| Idle > 30 min | "Want me to run a dream cycle?" | Toast + IPC (if connected) |
| Idle > 2 hours | Auto-trigger dream (no prompt) | Silent, log only |
| Long coding session (>2h) | "Consider taking a break" | Toast (once) |
| Reminder due | "Reminder: [message]" | Toast + TTS (if enabled) |
| Recurring schedule | Daily briefing, etc. | Toast + IPC |
| File change in watched dir | "File X was modified" | IPC only |

### 7.3 Notification Delivery

The daemon uses Windows toast notifications for proactive alerts when no CLI is connected. When a CLI is connected via IPC, it pushes messages there instead.

```python
def deliver_proactive(self, message: str, priority: int = 5):
    """Deliver a proactive message to the user."""
    # Try IPC first (if CLI is connected)
    if self.ipc_server.has_clients():
        self.ipc_server.push_notification({
            "type": "proactive",
            "message": message,
            "timestamp": datetime.now().isoformat()
        })
    else:
        # Fall back to Windows toast
        self._show_toast("AURA", message)

    # Also speak if TTS is enabled and priority is high
    if priority <= 3 and self.tts_enabled:
        self._speak(message)
```

---

## 8. IPC: Daemon <-> CLI Communication

### 8.1 Protocol Choice: Named Pipes

Named pipes are the right IPC mechanism for AURA on Windows because:
- Native to Windows (no extra dependencies)
- Fast (kernel-level, no network stack)
- Works across processes on same machine
- Python `asyncio` supports them via `loop.create_pipe_connection`
- No port conflicts (unlike TCP sockets)

Pipe name: `\\.\pipe\aura_daemon`

### 8.2 Message Protocol

JSON-line protocol over the named pipe. Each message is a single JSON object terminated by `\n`.

```python
# Message format
{
    "id": "msg_abc123",          # Unique message ID
    "type": "request|response|notification|event",
    "method": "chat|command|status|subscribe",  # For requests
    "data": { ... },             # Payload
    "timestamp": "2026-02-28T14:30:00",
    "reply_to": "msg_xyz789"     # For responses (links to request ID)
}
```

### 8.3 IPC Server (Daemon Side)

```python
import asyncio
import json

PIPE_NAME = r'\\.\pipe\aura_daemon'

class IPCServer:
    """Named pipe server for daemon <-> client communication."""

    def __init__(self, event_bus):
        self.bus = event_bus
        self._clients = []

    async def start(self):
        """Start listening for client connections."""
        while True:
            # Windows named pipe via asyncio (using proactor event loop)
            reader, writer = await asyncio.open_named_pipe_server(PIPE_NAME)
            client = IPCClient(reader, writer)
            self._clients.append(client)
            asyncio.create_task(self._handle_client(client))

    async def _handle_client(self, client):
        """Handle messages from a connected client."""
        try:
            while True:
                line = await client.reader.readline()
                if not line:
                    break
                msg = json.loads(line.decode())
                await self._route_message(client, msg)
        finally:
            self._clients.remove(client)
            client.writer.close()

    async def _route_message(self, client, msg):
        """Route incoming message to appropriate handler."""
        msg_type = msg.get("type")
        method = msg.get("method")

        if msg_type == "request":
            if method == "chat":
                # Forward to agent for processing
                response = await self._handle_chat(msg["data"])
                await client.send({"type": "response", "reply_to": msg["id"], "data": response})

            elif method == "status":
                # Return daemon status
                status = self._get_status()
                await client.send({"type": "response", "reply_to": msg["id"], "data": status})

            elif method == "subscribe":
                # Client wants to receive events
                client.subscriptions.update(msg["data"].get("events", []))

    def has_clients(self) -> bool:
        return len(self._clients) > 0

    async def push_notification(self, data: dict):
        """Push notification to all connected clients."""
        msg = {"type": "notification", "data": data}
        for client in self._clients:
            try:
                await client.send(msg)
            except Exception:
                pass
```

### 8.4 IPC Client (CLI Side)

```python
class IPCDaemonClient:
    """Client for connecting to the AURA daemon."""

    def __init__(self):
        self.reader = None
        self.writer = None

    async def connect(self):
        """Connect to the daemon's named pipe."""
        self.reader, self.writer = await asyncio.open_named_pipe_client(PIPE_NAME)

    async def chat(self, message: str) -> str:
        """Send a chat message and get response."""
        msg_id = str(uuid.uuid4())[:8]
        await self._send({"id": msg_id, "type": "request", "method": "chat", "data": {"message": message}})
        response = await self._recv()
        return response["data"]

    async def get_status(self) -> dict:
        """Get daemon status."""
        msg_id = str(uuid.uuid4())[:8]
        await self._send({"id": msg_id, "type": "request", "method": "status", "data": {}})
        return (await self._recv())["data"]
```

### 8.5 Fallback: TCP Socket

If named pipes cause issues (some Python versions on Windows have limited asyncio pipe support), fall back to a TCP socket on `localhost:19733` (AURA in phone keypad numbers). Same JSON-line protocol.

### 8.6 Also: FastAPI WebSocket (Already Exists)

The existing FastAPI server at `api/main.py` already has WebSocket support. The daemon can optionally start this server too, giving the GUI (web frontend) a communication channel. This means three IPC paths:

1. **Named Pipe** -- CLI <-> Daemon (primary, fastest)
2. **WebSocket** -- GUI <-> Daemon (via existing FastAPI)
3. **HTTP REST** -- External tools <-> Daemon (via existing FastAPI)

---

## 9. Dream/Consolidation Integration

### 9.1 When Does Dreaming Happen?

```
Trigger                         | Action
--------------------------------|----------------------------------
User idle > 30 min              | NeuroDream.check_idle_trigger()
                                | If true, start light sleep cycle
User idle > 2 hours             | Full dream cycle (light+deep+REM)
3:00 AM nightly                 | Full dream + DreamMode analysis
                                | + MemoryConsolidator.run_full()
                                | + A-MEM consolidation
User says "dream" or "/dream"   | Manual trigger, full cycle
System low resources at night   | Trigger via NeuroDream's LOW_RESOURCES
```

### 9.2 Dream Pipeline in the Daemon

```python
class DreamScheduler:
    """Schedules and runs dream/consolidation tasks."""

    def __init__(self, neurodream, dream_mode, consolidator, event_bus):
        self.neurodream = neurodream
        self.dream_mode = dream_mode
        self.consolidator = consolidator
        self.bus = event_bus

        # Subscribe to relevant events
        self.bus.subscribe("tick:slow", self.on_slow_tick)
        self.bus.subscribe("schedule:nightly", self.on_nightly)
        self.bus.subscribe("user:command", self.on_user_command)

    async def on_slow_tick(self, event):
        """Check idle trigger every 5 minutes."""
        idle_seconds = event.data.get("idle_seconds", 0)

        if idle_seconds > 7200:  # 2 hours
            if self.neurodream.current_phase == SleepPhase.AWAKE:
                self.bus.emit("dream:idle_triggered", {"idle_minutes": idle_seconds / 60})
                self.neurodream.begin_sleep_cycle(trigger="idle")

        elif idle_seconds > 1800:  # 30 minutes
            if self.neurodream.check_idle_trigger():
                self.bus.emit("dream:idle_triggered", {"idle_minutes": idle_seconds / 60})
                self.neurodream.begin_sleep_cycle(trigger="idle")

    async def on_nightly(self, event):
        """Full nightly consolidation."""
        # Phase 1: NeuroDream full cycle
        self.neurodream.begin_sleep_cycle(trigger="scheduled")

        # Phase 2: DreamMode analysis (metacognition logs)
        result = self.dream_mode.dream()

        # Phase 3: Episodic memory consolidation
        if self.consolidator:
            self.consolidator.run_full_consolidation()

        self.bus.emit("dream:consolidation_done", {
            "insights": result.get("insights", []),
            "consolidation": result.get("consolidation", {})
        })
```

### 9.3 Dream Wake-Up

When the user becomes active during a dream cycle, `NeuroDreamEngine.record_activity()` already handles interruption -- it sets the interrupt flag and transitions to WAKING phase. The daemon just needs to forward user activity:

```python
# In heartbeat, when any user:activity event fires:
bus.subscribe("user:activity", lambda e: neurodream.record_activity())
```

---

## 10. The Complete Daemon: `aura_daemon.py`

### 10.1 Entry Point Structure

```python
"""
AURA Daemon - Always-on background service.

Usage:
    python aura_daemon.py              # Run in foreground (development)
    python aura_daemon.py --install    # Install as Windows service via NSSM
    python aura_daemon.py --status     # Check if daemon is running
    python aura_daemon.py --stop       # Stop the daemon
"""

import asyncio
import signal
import sys
import logging
from pathlib import Path

# Ensure AURA is importable
sys.path.insert(0, str(Path(__file__).parent))

from aura.daemon.heartbeat import Heartbeat
from aura.daemon.event_bus import EventBus
from aura.daemon.ipc_server import IPCServer
from aura.daemon.sensors import ScreenSensor, SystemSensor, FileSensor, ScheduleSensor
from aura.daemon.proactive import ProactiveEngine
from aura.daemon.dream_scheduler import DreamScheduler
from aura.daemon.pid_lock import acquire_lock, release_lock

logger = logging.getLogger("aura.daemon")


class AuraDaemon:
    """The main daemon orchestrator."""

    def __init__(self):
        self.bus = EventBus()
        self.heartbeat = Heartbeat(self.bus)
        self.ipc = IPCServer(self.bus)

        # Sensors
        self.screen_sensor = ScreenSensor(self.bus)
        self.system_sensor = SystemSensor(self.bus)
        self.file_sensor = FileSensor(self.bus)
        self.schedule_sensor = ScheduleSensor(self.bus)

        # Engines
        self.proactive = ProactiveEngine(self.bus, self.ipc)
        self.dream = DreamScheduler(self.bus)

        # Wire events to sensors
        self.bus.subscribe("tick:fast", self.screen_sensor.on_tick)
        self.bus.subscribe("tick:medium", self.system_sensor.on_tick)
        self.bus.subscribe("tick:medium", self.schedule_sensor.on_tick)
        self.bus.subscribe("tick:medium", self.file_sensor.on_tick)

        # Wire proactive engine to screen/system events
        self.bus.subscribe("screen:error_detected", self.proactive.on_event)
        self.bus.subscribe("system:cpu_high", self.proactive.on_event)
        self.bus.subscribe("system:ram_high", self.proactive.on_event)

    async def start(self):
        """Start all daemon components."""
        logger.info("AURA Daemon starting...")
        self.bus.emit("daemon:started", {"pid": os.getpid()})

        await asyncio.gather(
            self.heartbeat.run(),
            self.ipc.start(),
        )

    async def stop(self):
        """Graceful shutdown."""
        logger.info("AURA Daemon shutting down...")
        self.bus.emit("daemon:shutting_down", {})
        self.heartbeat.running = False
        # Run emergency consolidation if mid-dream
        if self.dream.neurodream and self.dream.neurodream.current_phase != SleepPhase.AWAKE:
            self.dream.neurodream.wake_up("daemon_shutdown")


def main():
    acquire_lock()

    # Windows requires ProactorEventLoop for named pipes
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

    daemon = AuraDaemon()

    loop = asyncio.new_event_loop()

    # Handle shutdown signals
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, lambda: asyncio.create_task(daemon.stop()))
        except NotImplementedError:
            # Windows doesn't support add_signal_handler for all signals
            signal.signal(sig, lambda s, f: asyncio.create_task(daemon.stop()))

    try:
        loop.run_until_complete(daemon.start())
    finally:
        release_lock()
        loop.close()


if __name__ == "__main__":
    main()
```

### 10.2 File Structure for the Daemon Package

```
aura/
  daemon/
    __init__.py
    heartbeat.py          # Heartbeat class (tiered tick loop)
    event_bus.py           # EventBus class (pub/sub)
    ipc_server.py          # IPCServer (named pipe + TCP fallback)
    ipc_client.py          # IPCDaemonClient (for CLI to connect)
    pid_lock.py            # PID file locking
    proactive.py           # ProactiveEngine (scoring, delivery)
    dream_scheduler.py     # DreamScheduler (idle/nightly triggers)
    sensors/
      __init__.py
      screen.py            # ScreenSensor (dhash + Screenpipe bridge)
      system.py            # SystemSensor (CPU/RAM/GPU/disk)
      file.py              # FileSensor (watched directories)
      schedule.py          # ScheduleSensor (notifications + hooks)
      clipboard.py         # ClipboardSensor (change detection)
    config.py              # Daemon-specific config (intervals, thresholds)
aura_daemon.py             # Entry point (lives at project root)
```

---

## 11. Configuration

```python
# aura/daemon/config.py

class DaemonConfig:
    """Daemon configuration with sensible defaults."""

    # Heartbeat intervals (seconds)
    TICK_FAST = 5          # Screen hash check
    TICK_MEDIUM = 30       # Hooks, notifications, system health
    TICK_SLOW = 300        # Idle detection, dream trigger
    TICK_HOURLY = 3600     # System report, log rotation

    # Screen monitoring
    SCREEN_HASH_SIZE = 8           # dHash size (8 = 64-bit hash)
    SCREEN_CHANGE_THRESHOLD = 12   # Hamming distance for "changed"
    SCREEN_DIFF_THRESHOLD = 0.15   # Pixel diff ratio for "significant"
    SCREEN_OCR_COOLDOWN = 10       # Min seconds between OCR runs

    # Proactive engine
    PROACTIVE_THRESHOLD = 0.6      # Min score to trigger intervention
    PROACTIVE_COOLDOWN = 120       # Min seconds between proactive messages
    PROACTIVE_ENABLED = True       # Master switch

    # Dream/consolidation
    IDLE_DREAM_MINUTES = 30        # Minutes idle before light dream
    IDLE_FULL_DREAM_MINUTES = 120  # Minutes idle before full dream
    NIGHTLY_HOUR = 3               # 3 AM for nightly consolidation
    DREAM_ENABLED = True           # Master switch

    # System monitoring thresholds
    CPU_ALERT_PERCENT = 85
    RAM_ALERT_PERCENT = 90
    DISK_ALERT_PERCENT = 95
    GPU_TEMP_ALERT_C = 85

    # IPC
    PIPE_NAME = r'\\.\pipe\aura_daemon'
    TCP_FALLBACK_PORT = 19733      # AURA on phone keypad
    IPC_TIMEOUT = 30               # Seconds

    # Logging
    LOG_DIR = "D:/Aura/logs/daemon"
    LOG_LEVEL = "INFO"
    LOG_ROTATE_MB = 10
```

---

## 12. CLI Integration

The existing CLI (`main.py`) gets a small addition: try to connect to the daemon first. If the daemon is running, the CLI becomes a thin client that communicates via IPC. If the daemon is not running, the CLI works as it does today (standalone).

```python
# In main.py, at the start of run_chat_mode():
async def try_daemon_connection():
    """Try to connect to running daemon."""
    try:
        client = IPCDaemonClient()
        await asyncio.wait_for(client.connect(), timeout=2)
        return client
    except Exception:
        return None

# In chat loop:
daemon_client = asyncio.run(try_daemon_connection())
if daemon_client:
    print("[Connected to AURA daemon]")
    # Use daemon_client.chat() instead of local agent
else:
    print("[Standalone mode - daemon not running]")
    # Use local agent as before
```

---

## 13. Resource Budget

The daemon must be lightweight. Target resource usage when idle:

| Resource | Budget | How |
|----------|--------|-----|
| CPU | < 1% average | Hash comparison is ~2ms per 5s tick |
| RAM | < 100MB | No models loaded until dream time |
| Disk I/O | Minimal | JSON writes only on events, not every tick |
| GPU | 0% when awake | GPU only used during dream (OCR/LLM) |
| Network | 0 | Everything local (Screenpipe is localhost) |

During dream cycles: CPU may spike to 10-20%, GPU may be used for Florence-2 OCR. This only happens during idle periods.

---

## 14. Implementation Priority

### Phase 1: Skeleton (Day 1)
1. `aura/daemon/__init__.py` + package structure
2. `EventBus` (pub/sub core)
3. `Heartbeat` (tiered tick loop)
4. `PID lock`
5. `aura_daemon.py` entry point
6. Basic logging

### Phase 2: Sensors (Day 2)
1. `ScreenSensor` (dhash change detection, Screenpipe bridge)
2. `SystemSensor` (wrap existing SystemControlTool)
3. `ScheduleSensor` (wrap existing NotificationTool + HooksManager)
4. Wire sensors to heartbeat ticks

### Phase 3: IPC (Day 3)
1. `IPCServer` (named pipe with TCP fallback)
2. `IPCDaemonClient` (for CLI)
3. Message protocol
4. CLI integration (auto-connect to daemon)

### Phase 4: Proactive Engine (Day 4)
1. Event scoring
2. Error detection response
3. Toast notification delivery
4. Cooldown/rate limiting

### Phase 5: Dream Integration (Day 5)
1. `DreamScheduler` (idle + nightly triggers)
2. Wire NeuroDream engine
3. Wire DreamMode + MemoryConsolidator
4. Nightly cron via schedule sensor

### Phase 6: Service Installation (Day 6)
1. NSSM install script
2. Task Scheduler fallback
3. Service management commands (`--install`, `--stop`, `--status`)
4. Log rotation

---

## 15. Open Questions / Decisions Needed

1. **Screenpipe dependency**: Should the daemon require Screenpipe, or treat it as optional with built-in fallback? (Currently designed as optional.)

2. **LLM for proactive analysis**: When the daemon detects an error on screen, should it call a local model (qwen2.5-coder:7b) to analyze it, or just show the raw error text? Local model adds ~2s latency but much better suggestions.

3. **Notification aggressiveness**: How often should AURA proactively message? Currently set to 2-minute cooldown. Too frequent = annoying. Too rare = feels dead.

4. **Privacy**: The daemon sees everything on screen. Should there be a "do not disturb" mode or privacy zones beyond what Screenpipe already filters?

5. **Multi-user**: The current design is single-user. The PID lock and pipe name are hardcoded. Fine for Elnur's machine, but worth noting.

---

## Research Sources

- [Python Windows Service with win32serviceutil](https://dev.to/demola12/building-a-robust-windows-service-in-python-with-win32serviceutil-part-13-1k6k)
- [NSSM for Python Services on Windows](https://www.mssqltips.com/sqlservertip/7325/how-to-run-a-python-script-windows-service-nssm/)
- [Servy - NSSM Alternative](https://github.com/aelassas/servy)
- [Python Perceptual Hashing with imagehash](https://github.com/JohannesBuchner/imagehash)
- [dHash for Fast Image Comparison](https://benhoyt.com/writings/duplicate-image-detection/)
- [asyncio Event Bus Pattern](https://www.joeltok.com/posts/2021-03-building-an-event-bus-in-python/)
- [Lahja - Multi-process Event Bus](https://github.com/ethereum/lahja)
- [aiopubsub - asyncio Pub/Sub](https://pypi.org/project/aiopubsub/)
- [Graphite - Event-Driven AI Agent Framework](https://medium.com/binome/introduction-to-graphite-an-event-driven-ai-agent-framework-540478130cd2)
- [CrewAI Event-Driven Flows](https://www.kdnuggets.com/top-7-python-frameworks-for-ai-agents)
- [Proactive AI Assistants (CHI 2025)](https://dl.acm.org/doi/10.1145/3706598.3714002)
- [Proactive AI Assistant Patterns](https://www.saner.ai/blogs/best-proactive-ai-assistants)
