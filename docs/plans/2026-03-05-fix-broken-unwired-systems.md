# Fix Broken & Unwired Systems Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix 9 specific broken or unwired systems identified by audit — episodic memory import crash, classifier bug, vision chain, test mismatches, and wire Active Inference, World Model, GatewayDaemon, Proto-AGI API, and FluxMind API.

**Architecture:** Fixes are independent and ordered by risk (lowest to highest). Tests first, then broken imports, then unwired systems, then new routes. No new abstractions — wire what already exists.

**Tech Stack:** Python, FastAPI, pymdp, qdrant-client, sentence-transformers

---

## Task 1 — Fix SENTENCE_TRANSFORMERS_AVAILABLE in memory_store.py

**Files:**
- Modify: `aura_episodic_memory/memory_store.py` (after QDRANT_AVAILABLE block, ~line 35)

**Context:** `aura_episodic_memory/__init__.py` imports `SENTENCE_TRANSFORMERS_AVAILABLE` from `memory_store.py` but the flag is never defined there. This crashes the entire episodic memory module on import.

**Step 1: Add the flag to memory_store.py**

After the `QDRANT_AVAILABLE` try/except block (around line 35), add:

```python
# Check sentence-transformers availability (lazy-loaded via _shared_models)
try:
    import sentence_transformers  # noqa: F401
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
```

**Step 2: Verify the import works**

```bash
cd D:\Aura
python -c "from aura_episodic_memory import EpisodicMemoryStore, SENTENCE_TRANSFORMERS_AVAILABLE; print('OK, ST_AVAILABLE=', SENTENCE_TRANSFORMERS_AVAILABLE)"
```

Expected: `OK, ST_AVAILABLE= True`

**Step 3: Commit**

```bash
git add aura_episodic_memory/memory_store.py
git commit -m "fix: export SENTENCE_TRANSFORMERS_AVAILABLE from memory_store"
```

---

## Task 2 — Fix strategy_bandit default returns DEBUG instead of ANALYSIS

**Files:**
- Modify: `aura/consciousness/strategy_bandit.py` (in `classify()` method, after the for-loop)

**Context:** `classify("Hello, how are you?")` returns `DEBUG` instead of `ANALYSIS`. The default `best_category = ProblemCategory.ANALYSIS` before the loop is correct, but something in the keyword matching is scoring DEBUG > 0 for ambiguous queries. The explicit fix: add a guard after the loop so score=0 always returns ANALYSIS.

**Step 1: Read the current classify() method**

```bash
cd D:\Aura
grep -n "def classify\|best_score\|best_category\|return best" aura/consciousness/strategy_bandit.py
```

**Step 2: Add explicit zero-score guard after the for-loop**

Find the `classify()` method. After the for-loop, before the final return, add:

```python
        # Explicit default: if nothing scored, fall back to ANALYSIS
        if best_score == 0:
            return ProblemCategory.ANALYSIS
        return best_category
```

Remove any existing bare `return best_category` at the end of the method (replace it with the above block).

**Step 3: Run the test**

```bash
cd D:\Aura
python -m pytest tests/test_strategy_bandit.py::TestProblemClassifier::test_classify_default -v
```

Expected: PASSED

**Step 4: Commit**

```bash
git add aura/consciousness/strategy_bandit.py
git commit -m "fix: strategy_bandit classify() defaults to ANALYSIS when no keywords match"
```

---

## Task 3 — Add local llava fallback to vision model chain

**Files:**
- Modify: `aura/config.py` (MODEL_VISION_CHAIN definition, ~line 158)

**Context:** `MODEL_VISION_CHAIN` is cloud-only. If Ollama Pro is down, vision fails completely. Test `test_chain_contains_llava_fallback` verifies a local fallback exists.

**Step 1: Edit config.py**

Find `MODEL_VISION_CHAIN` and add `"llava:latest"` as the last entry:

```python
MODEL_VISION_CHAIN = [
    "qwen3-vl:235b-cloud",             # Primary: dedicated VL model
    "kimi-k2.5:cloud",                 # Fallback: multimodal capable
    "gemini-3-flash-preview:cloud",    # Fallback: Gemini supports vision
    "llava:latest",                    # Local fallback: works offline
]
```

**Step 2: Run the test**

```bash
cd D:\Aura
python -m pytest tests/test_vision.py::TestConfigIntegration::test_chain_contains_llava_fallback -v
```

Expected: PASSED

**Step 3: Commit**

```bash
git add aura/config.py
git commit -m "fix: add local llava fallback to vision model chain"
```

---

## Task 4 — Fix voice presence tests (pyttsx3 → Kokoro mocks)

**Files:**
- Modify: `tests/test_voice_presence.py`
- Read first: `aura/services/voice_presence.py` (to find the Kokoro import path)

**Context:** `VoicePresenceService` was refactored from pyttsx3 to Kokoro ONNX TTS, but tests still mock `pyttsx3.init`. Four tests fail because the mock target no longer exists.

**Step 1: Read voice_presence.py to find Kokoro imports**

```bash
cd D:\Aura
head -60 aura/services/voice_presence.py
grep -n "kokoro\|pyttsx3\|import" aura/services/voice_presence.py | head -30
```

**Step 2: Update mock targets in test_voice_presence.py**

For each test that mocks `pyttsx3.init`, replace with the correct Kokoro mock path found in Step 1. Pattern:

```python
# BEFORE (broken):
@patch("pyttsx3.init")
def test_speak_queues(self, mock_tts):
    ...

# AFTER (fixed) — use whatever Kokoro class voice_presence.py imports:
@patch("aura.services.voice_presence.KokoroPipeline")  # adjust to actual import
def test_speak_queues(self, mock_kokoro):
    mock_instance = MagicMock()
    mock_kokoro.return_value = mock_instance
    ...
```

**Step 3: Run the failing tests**

```bash
cd D:\Aura
python -m pytest tests/test_voice_presence.py::TestVoicePresenceService::test_block_mode tests/test_voice_presence.py::TestVoicePresenceService::test_emotion_params tests/test_voice_presence.py::TestVoicePresenceService::test_get_status tests/test_voice_presence.py::TestVoicePresenceService::test_speak_queues -v
```

Expected: All 4 PASSED

**Step 4: Commit**

```bash
git add tests/test_voice_presence.py
git commit -m "fix: update voice presence tests to mock Kokoro instead of pyttsx3"
```

---

## Task 5 — Fix auth tests (enable auth flag in test setup)

**Files:**
- Modify: `tests/test_auth.py`

**Context:** Tests set `AURA_API_KEY` but auth requires `AURA_API_AUTH_ENABLED=true` separately. Setting the key alone does nothing — both env vars must be set before importing the app.

**Step 1: Add AURA_API_AUTH_ENABLED to test_auth.py**

At the top of `tests/test_auth.py`, before `from api.main import app`:

```python
import os
import pytest
from fastapi.testclient import TestClient

# Both vars required: key sets the secret, enabled flag activates middleware
os.environ["AURA_API_KEY"] = "test-key-123"
os.environ["AURA_API_AUTH_ENABLED"] = "true"

from api.main import app
# ... rest of file unchanged
```

**Step 2: Run the failing tests**

```bash
cd D:\Aura
python -m pytest tests/test_auth.py::test_proactive_start_requires_auth tests/test_auth.py::test_memory_recalls_requires_auth -v
```

Expected: Both PASSED

**Step 3: Run full test suite to confirm no regressions**

```bash
cd D:\Aura
python -m pytest tests/ -v 2>&1 | tail -20
```

Expected: 8 failures → 0 or fewer failures

**Step 4: Commit**

```bash
git add tests/test_auth.py
git commit -m "fix: auth tests must set AURA_API_AUTH_ENABLED=true alongside AURA_API_KEY"
```

---

## Task 6 — Wire GatewayDaemon in aura_daemon.py

**Files:**
- Modify: `aura_daemon.py`
- Read first: `aura/proactive/gateway_daemon.py` (GatewayDaemon.__init__ signature)

**Context:** `GatewayDaemon` (with `ActiveInferenceEngine` / Free Energy Principle) is imported in gateway_daemon.py but never instantiated anywhere. `aura_daemon.py` uses a simpler `ProactiveEngine` instead. The daemon should start `GatewayDaemon` as a background thread alongside the existing ProactiveEngine.

**Step 1: Read GatewayDaemon.__init__ signature**

```bash
cd D:\Aura
grep -n "def __init__\|def start\|def stop" aura/proactive/gateway_daemon.py | head -20
```

**Step 2: Add GatewayDaemon startup in AuraDaemon**

In `aura_daemon.py`, find the `AuraDaemon.__init__` or `start()` method. Add after the ProactiveEngine is set up:

```python
# Start the GatewayDaemon (Active Inference proactive center)
try:
    from aura.proactive.gateway_daemon import GatewayDaemon
    self._gateway_daemon = GatewayDaemon(
        notification_callback=self._on_proactive_message
    )
    self._gateway_daemon.start()
    logger.info("[Daemon] GatewayDaemon started (Active Inference enabled)")
except Exception as e:
    logger.warning(f"[Daemon] GatewayDaemon unavailable: {e}")
    self._gateway_daemon = None
```

Note: Adjust the `GatewayDaemon.__init__` kwargs to match the actual signature found in Step 1.

**Step 3: Add shutdown in AuraDaemon.stop()**

In the daemon's shutdown/stop method, add:

```python
if self._gateway_daemon:
    try:
        self._gateway_daemon.stop()
    except Exception:
        pass
```

**Step 4: Verify daemon starts without error**

```bash
cd D:\Aura
python -c "
from aura_daemon import AuraDaemon
import time
d = AuraDaemon()
print('Daemon created OK, gateway_daemon=', d._gateway_daemon is not None)
"
```

Expected: `Daemon created OK, gateway_daemon= True`

**Step 5: Commit**

```bash
git add aura_daemon.py
git commit -m "feat: wire GatewayDaemon (Active Inference) into aura_daemon startup"
```

---

## Task 7 — Wire WorldModel to ProactiveAwareness

**Files:**
- Modify: `aura/agent.py` (wherever ProactiveAwareness is instantiated)
- Read first: `aura/consciousness/proactive_awareness.py` (__init__ signature)
- Read first: `aura/consciousness/world_model.py` (get_world_model() function)

**Context:** `ProactiveAwareness.__init__` expects a `world_model` parameter but is never passed one. `get_world_model()` is a singleton factory in world_model.py. The fix is to import and pass it at instantiation time.

**Step 1: Find ProactiveAwareness instantiation in agent.py**

```bash
cd D:\Aura
grep -n "ProactiveAwareness\|proactive_awareness" aura/agent.py | head -20
grep -n "def __init__" aura/consciousness/proactive_awareness.py | head -5
```

**Step 2: Read ProactiveAwareness.__init__ signature**

```bash
cd D:\Aura
sed -n '$(grep -n "class ProactiveAwareness" aura/consciousness/proactive_awareness.py | head -1 | cut -d: -f1),+30p' aura/consciousness/proactive_awareness.py
```

Or just read lines around the class definition.

**Step 3: Import get_world_model at top of agent.py (or locally)**

Find the ProactiveAwareness import in agent.py. Near it add (if not already imported):

```python
from .consciousness.world_model import get_world_model
```

**Step 4: Pass world_model when instantiating ProactiveAwareness**

Find the line that creates `ProactiveAwareness(...)` in agent.py. Change it to:

```python
self.proactive_awareness = ProactiveAwareness(
    world_model=get_world_model(),
    # ... other existing kwargs unchanged
)
```

**Step 5: Verify agent initializes without error**

```bash
cd D:\Aura
python -c "
from aura.consciousness.proactive_awareness import ProactiveAwareness
from aura.consciousness.world_model import get_world_model
wm = get_world_model()
pa = ProactiveAwareness(world_model=wm)
print('ProactiveAwareness OK, world_model=', pa.world_model is not None)
"
```

Expected: `ProactiveAwareness OK, world_model= True`

**Step 6: Commit**

```bash
git add aura/agent.py
git commit -m "fix: pass WorldModel instance to ProactiveAwareness at init"
```

---

## Task 8 — Add /api/proto-agi routes

**Files:**
- Create: `api/routes/proto_agi.py`
- Modify: `api/main.py` (import + include_router)

**Context:** Frontend calls `/api/proto-agi/status`, `/api/proto-agi/start`, `/api/proto-agi/stop`. Agent has `self.proto_agi` with `start_proto_agi()`, `stop_proto_agi()`, `proto_agi.is_running`, `proto_agi.cycle_count`.

**Step 1: Create api/routes/proto_agi.py**

```python
"""API endpoints for Proto-AGI autonomous core."""

import logging
from fastapi import APIRouter

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/proto-agi", tags=["proto-agi"])


def _get_agent():
    from api.services.agent_service import agent_service
    return agent_service.agent


@router.get("/status")
async def get_proto_agi_status():
    """Get Proto-AGI status and cycle count."""
    try:
        agent = _get_agent()
        if agent is None or agent.proto_agi is None:
            return {"available": False, "running": False, "cycle_count": 0}
        pa = agent.proto_agi
        return {
            "available": True,
            "running": getattr(pa, "is_running", False),
            "cycle_count": getattr(pa, "cycle_count", 0),
            "mode": getattr(pa, "mode", "unknown"),
        }
    except Exception as e:
        logger.error(f"[ProtoAGI] Status error: {e}")
        return {"available": False, "running": False, "cycle_count": 0}


@router.post("/start")
async def start_proto_agi(cycle_interval: float = 60.0):
    """Start the Proto-AGI autonomous loop."""
    try:
        agent = _get_agent()
        if agent is None:
            return {"success": False, "error": "Agent not ready"}
        agent.start_proto_agi(cycle_interval=cycle_interval)
        return {"success": True, "cycle_interval": cycle_interval}
    except Exception as e:
        logger.error(f"[ProtoAGI] Start error: {e}")
        return {"success": False, "error": str(e)}


@router.post("/stop")
async def stop_proto_agi():
    """Stop the Proto-AGI autonomous loop."""
    try:
        agent = _get_agent()
        if agent is None:
            return {"success": False, "error": "Agent not ready"}
        agent.stop_proto_agi()
        return {"success": True}
    except Exception as e:
        logger.error(f"[ProtoAGI] Stop error: {e}")
        return {"success": False, "error": str(e)}
```

**Step 2: Register in api/main.py**

In `api/main.py`, find the imports block (line ~30):

```python
from api.routes import proto_agi as proto_agi_routes
```

Find the `app.include_router(...)` block and add:

```python
app.include_router(proto_agi_routes.router)
```

**Step 3: Verify the endpoint exists**

```bash
cd D:\Aura
python -c "
from api.main import app
routes = [r.path for r in app.routes]
proto_routes = [r for r in routes if 'proto' in r]
print('Proto-AGI routes:', proto_routes)
"
```

Expected: `Proto-AGI routes: ['/api/proto-agi/status', '/api/proto-agi/start', '/api/proto-agi/stop']`

**Step 4: Commit**

```bash
git add api/routes/proto_agi.py api/main.py
git commit -m "feat: add /api/proto-agi status/start/stop endpoints"
```

---

## Task 9 — Add /api/fluxmind routes

**Files:**
- Create: `api/routes/fluxmind.py`
- Modify: `api/main.py` (import + include_router)

**Context:** Frontend calls `/api/fluxmind`. FluxMind is in `agent.tools["fluxmind"]` with methods `is_available()`, `get_status()`, and command handling via `_handle_fluxmind_command()`.

**Step 1: Read FluxMind tool interface**

```bash
cd D:\Aura
grep -n "def \|is_available\|get_status\|class Flux" aura/tools/reflexion.py aura/tools/synapseforge.py 2>/dev/null | head -10
# Actually find the fluxmind tool file:
ls aura/tools/ | grep -i flux
grep -n "class FluxMind\|def is_available\|def get_status\|def query\|def step" aura/tools/*.py 2>/dev/null | grep -i flux | head -20
```

**Step 2: Create api/routes/fluxmind.py**

```python
"""API endpoints for FluxMind calibrated reasoning engine."""

import logging
from fastapi import APIRouter
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/fluxmind", tags=["fluxmind"])


def _get_agent():
    from api.services.agent_service import agent_service
    return agent_service.agent


@router.get("/status")
async def get_fluxmind_status():
    """Get FluxMind availability and status."""
    try:
        agent = _get_agent()
        if agent is None or "fluxmind" not in getattr(agent, "tools", {}):
            return {"available": False, "loaded": False}
        tool = agent.tools["fluxmind"]
        available = tool.is_available() if hasattr(tool, "is_available") else False
        status = tool.get_status() if hasattr(tool, "get_status") else {}
        return {"available": available, "loaded": True, **status}
    except Exception as e:
        logger.error(f"[FluxMind] Status error: {e}")
        return {"available": False, "loaded": False}


class FluxMindQueryRequest(BaseModel):
    message: str


@router.post("/query")
async def fluxmind_query(request: FluxMindQueryRequest):
    """Run a query through FluxMind reasoning engine."""
    try:
        agent = _get_agent()
        if agent is None:
            return {"success": False, "error": "Agent not ready"}
        result = agent._handle_fluxmind_command(request.message)
        if result is None:
            result = agent.run(f"fluxmind: {request.message}")
        return {"success": True, "result": result}
    except Exception as e:
        logger.error(f"[FluxMind] Query error: {e}")
        return {"success": False, "error": str(e)}
```

**Step 3: Register in api/main.py**

```python
from api.routes import fluxmind as fluxmind_routes
# ...
app.include_router(fluxmind_routes.router)
```

**Step 4: Verify endpoints**

```bash
cd D:\Aura
python -c "
from api.main import app
routes = [r.path for r in app.routes]
print('FluxMind routes:', [r for r in routes if 'fluxmind' in r])
"
```

Expected: `FluxMind routes: ['/api/fluxmind/status', '/api/fluxmind/query']`

**Step 5: Commit**

```bash
git add api/routes/fluxmind.py api/main.py
git commit -m "feat: add /api/fluxmind status and query endpoints"
```

---

## Verification

After all 9 tasks:

```bash
cd D:\Aura

# Run full test suite — target: 0 pre-existing failures remain
python -m pytest tests/ -v 2>&1 | tail -15

# Episodic memory
python -c "from aura_episodic_memory import EpisodicMemoryStore, SENTENCE_TRANSFORMERS_AVAILABLE; print('Episodic OK:', SENTENCE_TRANSFORMERS_AVAILABLE)"

# Strategy bandit
python -c "from aura.consciousness.strategy_bandit import ProblemClassifier, ProblemCategory; c = ProblemClassifier(); print(c.classify('hello') == ProblemCategory.ANALYSIS)"

# Vision chain has local fallback
python -c "from aura.config import Config; print('llava in chain:', 'llava' in ' '.join(Config.MODEL_VISION_CHAIN))"

# API routes registered
python -c "from api.main import app; paths = [r.path for r in app.routes]; print([p for p in paths if 'proto' in p or 'fluxmind' in p])"
```
