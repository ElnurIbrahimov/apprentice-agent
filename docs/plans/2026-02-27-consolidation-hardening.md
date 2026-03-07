# AURA Consolidation & Hardening — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Merge the duplicate `aura/` and `aura/` systems into one coherent codebase, wire up all cognitive systems that currently exist but never run, and harden everything until 356/356 tests pass with no warnings.

**Architecture:** `aura/` becomes the single source of truth. The simpler `aura/` duplicates are deleted. What's unique in `aura/` (messaging connectors, PatternProphet) moves into `aura/`. All proactive systems that were dead code get wired into the startup sequence.

**Tech Stack:** Python 3.10+, FastAPI, Ollama, ChromaDB, Qdrant, Kuzu, asyncio, pytest

**Baseline:** 355/356 tests pass. 1 pre-existing failure in `test_neurodream.py::test_enter_sleep` (not caused by our work).

---

## PHASE 1: Dependencies & Baseline Lock

### Task 1.1: Install missing calendar dependencies

**Files:** `requirements.txt`

**Step 1: Install missing packages**
```bash
cd D:/apprentice-agent
pip install icalendar recurring-ical-events
```

**Step 2: Verify installation**
```bash
python -c "import icalendar; import recurring_ical_events; print('OK')"
```
Expected: `OK`

**Step 3: Add to requirements.txt**

Add to `requirements.txt`:
```
icalendar>=5.0.0
recurring-ical-events>=2.1.0
```

**Step 4: Run full test suite to confirm baseline**
```bash
cd D:/apprentice-agent && python -m pytest tests/ -q --tb=short 2>&1 | tail -10
```
Expected: `355 passed, 1 failed` (same as before — neurodream pre-existing)

**Step 5: Commit**
```bash
git add requirements.txt
git commit -m "deps: add icalendar and recurring-ical-events for calendar monitor"
```

---

### Task 1.2: Fix the pre-existing NeuroDream test failure

**Files:**
- Read: `tests/test_neurodream.py` (find `test_enter_sleep`)
- Modify: `aura/tools/neurodream.py`

**Step 1: Read the failing test to understand what it expects**
```bash
cd D:/apprentice-agent && python -m pytest tests/test_neurodream.py::TestNeuroDreamEngine::test_enter_sleep -v --tb=long 2>&1
```

**Step 2: Read the neurodream sleep entry code**
Read `aura/tools/neurodream.py` around the `enter_sleep` / sleep state transition logic.

**Step 3: Fix the assertion**
The fix will be in `neurodream.py` — adjust the sleep state entry so the test assertion is satisfied without breaking other NeuroDream tests.

**Step 4: Run NeuroDream tests only**
```bash
cd D:/apprentice-agent && python -m pytest tests/test_neurodream.py -v --tb=short 2>&1
```
Expected: All pass

**Step 5: Run full suite**
```bash
cd D:/apprentice-agent && python -m pytest tests/ -q --tb=no 2>&1 | tail -5
```
Expected: `356 passed`

**Step 6: Commit**
```bash
git commit -m "fix: resolve pre-existing NeuroDream test_enter_sleep assertion"
```

---

## PHASE 2: Merge `aura/` into `aura/`

### Task 2.1: Move PatternProphet into aura

**Files:**
- Source: `aura/patterns/pattern_prophet.py`
- Destination: `aura/patterns/pattern_prophet.py`
- Create: `aura/patterns/__init__.py`
- Modify: `aura/agent.py` (update PatternProphet import)
- Modify: `aura/engine.py` (update import to point to new location)

**Step 1: Create the patterns directory and move the file**
```bash
mkdir -p D:/apprentice-agent/aura/patterns
cp D:/apprentice-agent/aura/patterns/pattern_prophet.py D:/apprentice-agent/aura/patterns/pattern_prophet.py
echo "from .pattern_prophet import PatternProphet" > D:/apprentice-agent/aura/patterns/__init__.py
```

**Step 2: Update import in `aura/engine.py`**

Change:
```python
from .patterns import PatternProphet
```
To:
```python
from aura.patterns import PatternProphet
```

**Step 3: Add PatternProphet to ApprenticeAgent** (if not already wired)

In `aura/agent.py`, find where `AURAEngine` is initialized and where patterns are used. Add direct access to PatternProphet from `aura/patterns/`.

**Step 4: Run tests**
```bash
cd D:/apprentice-agent && python -m pytest tests/ -q --tb=no 2>&1 | tail -5
```
Expected: `356 passed`

**Step 5: Commit**
```bash
git commit -m "refactor: move PatternProphet from aura/ to aura/patterns/"
```

---

### Task 2.2: Move messaging connectors into aura

**Files:**
- Source: `aura/messaging/` (entire directory)
- Destination: `aura/messaging/`
- Modify: `run_telegram.py`
- Modify: `run_messaging.py`

**Step 1: Copy messaging directory**
```bash
cp -r D:/apprentice-agent/aura/messaging D:/apprentice-agent/aura/messaging
```

**Step 2: Update imports in `run_telegram.py`**

Change:
```python
from aura.messaging.telegram_bot import TelegramBot
from aura.messaging.config import TELEGRAM_CONFIG
```
To:
```python
from aura.messaging.telegram_bot import TelegramBot
from aura.messaging.config import TELEGRAM_CONFIG
```

**Step 3: Update imports in `run_messaging.py`** (same pattern)

**Step 4: Update `telegram_bot.py` to use ApprenticeAgent instead of AURAEngine**

In `aura/messaging/telegram_bot.py`, find the handler that processes messages and replace `AURAEngine.process_input()` / `process_response()` calls with `ApprenticeAgent.chat_stream()`.

The current pattern (roughly):
```python
context = self.aura.process_input(message)
response = self.llm.generate(context)
final = self.aura.process_response(response, context)
```

Becomes:
```python
full_response = ""
for chunk in self.agent.chat_stream(message):
    full_response += chunk
```

**Step 5: Pass agent instance into TelegramBot constructor**

In `run_telegram.py`:
```python
from aura import ApprenticeAgent
agent = ApprenticeAgent()
bot = TelegramBot(agent=agent, config=TELEGRAM_CONFIG)
```

**Step 6: Test syntax**
```bash
cd D:/apprentice-agent && python -c "from aura.messaging.telegram_bot import TelegramBot; print('OK')"
```

**Step 7: Run full tests**
```bash
cd D:/apprentice-agent && python -m pytest tests/ -q --tb=no 2>&1 | tail -5
```

**Step 8: Commit**
```bash
git commit -m "refactor: move messaging connectors to aura/messaging/"
```

---

### Task 2.3: Replace AURAEngine duplicate calls with ALMA/unified-memory directly

**Files:**
- Modify: `aura/agent.py` (lines ~4699, ~5098, ~5301, ~5479, ~5691)

**Background:** `ApprenticeAgent` calls `self.aura.process_input()` and `self.aura.process_response()` at chat time. These go through the simple `aura/emotion/` and `aura/memory/` instead of the sophisticated ALMA and unified memory that `aura/` already has.

**Step 1: Read what `AURAEngine.process_input()` and `process_response()` actually do**

Read `aura/engine.py` lines 172 onward. Document what each method returns/does:
- `process_input()`: records interaction in emotional engine, gets context (mood, tone, memory snippets, thinking prefix)
- `process_response()`: humanizes the response

**Step 2: Identify ALMA equivalents already in aura**

The equivalent calls that already exist in `aura/`:
- `alma_engine.process_stimulus()` → replaces `emotional_engine.process_interaction()`
- `unified_memory.get_context()` → replaces `markdown_store.get_context_for_llm()`
- `aura/humanize/` → replaces `aura/humanize/`

**Step 3: Create a thin compatibility wrapper in `aura/agent.py`**

Find `if self.aura_enabled and self.aura:` blocks. Replace the `self.aura.process_input()` call so it uses `self.brain`'s emotion state (ALMA) instead. Example:

```python
# Before:
aura_context = self.aura.process_input(message)

# After — use ALMA directly:
aura_context = {
    "mood": self._get_alma_mood(),
    "tone": self._get_alma_tone(),
    "memory_context": self._get_memory_context(message),
}
```

Add helper methods `_get_alma_mood()`, `_get_alma_tone()`, `_get_memory_context()` that read from existing ALMA engine.

**Step 4: Make `self.aura` optional/None after verifying nothing breaks**

Set `self.aura_enabled = False` in config temporarily and run tests to see what breaks.

**Step 5: Run full tests**
```bash
cd D:/apprentice-agent && python -m pytest tests/ -q --tb=short 2>&1 | tail -15
```
Expected: 356 passed

**Step 6: Commit**
```bash
git commit -m "refactor: replace AURAEngine wrapper with direct ALMA/unified-memory calls"
```

---

### Task 2.4: Delete duplicate aura/ subsystems

**Only do this AFTER Tasks 2.1-2.3 are complete and all tests pass.**

**Files to delete:**
- `aura/llm/` (duplicate of `aura/aura_ollama_client.py`)
- `aura/memory/` (duplicate of `aura/memory/`)
- `aura/emotion/` (duplicate of `aura/emotion/`)
- `aura/humanize/` (duplicate of `aura/humanize/`)
- `aura/soul/` (duplicate of `aura/soul/`)
- `aura/thinking/` (duplicate of `aura/thinking/`)
- `aura/proactive/heartbeat.py` (replaced by gateway daemon in Phase 3)
- `aura/fast_path.py` (duplicate of `aura/fast_path.py`)
- `aura/engine.py` (the wrapper we're removing)
- `aura/core/` (check if context_builder.py has unique logic first)
- `aura/patterns/` (moved to `aura/patterns/` in Task 2.1)

**Step 1: Verify no remaining imports from aura/ in the codebase**
```bash
grep -rn "from aura\." D:/apprentice-agent --include="*.py" | grep -v __pycache__ | grep -v "aura_data\|aura_episodic\|aura_knowledge\|aura_skill\|aura_life"
```
Expected: Only `run_telegram.py` and `run_messaging.py` (which we already fixed)

**Step 2: Read `aura/core/context_builder.py` to check for unique logic**

If it has unique logic, move it to `aura/`. If it duplicates something, note what.

**Step 3: Delete the duplicate directories**
```bash
rm -rf D:/apprentice-agent/aura/llm
rm -rf D:/apprentice-agent/aura/memory
rm -rf D:/apprentice-agent/aura/emotion
rm -rf D:/apprentice-agent/aura/humanize
rm -rf D:/apprentice-agent/aura/soul
rm -rf D:/apprentice-agent/aura/thinking
rm -rf D:/apprentice-agent/aura/patterns
rm -rf D:/apprentice-agent/aura/fast_path.py
rm -rf D:/apprentice-agent/aura/engine.py
rm -rf D:/apprentice-agent/aura/core
```

Keep: `aura/messaging/` as backup until `aura/messaging/` is confirmed working.

**Step 4: Run full tests**
```bash
cd D:/apprentice-agent && python -m pytest tests/ -q --tb=short 2>&1 | tail -15
```
Expected: 356 passed — any new failures mean we missed an import, fix them.

**Step 5: Commit**
```bash
git commit -m "refactor: delete duplicate aura/ subsystems — aura/ is now sole source of truth"
```

---

## PHASE 3: Wire the Proactive Gateway

### Task 3.1: Install and verify proactive dependencies

**Step 1: Verify pymdp is installed (Active Inference)**
```bash
python -c "import pymdp; print('pymdp OK')"
```

**Step 2: Read gateway_daemon.py to understand startup API**
Read `aura/proactive/gateway_daemon.py` — find `get_gateway_daemon()` function signature and what `start=True` does.

**Step 3: Read api/main.py startup section**
Read `D:/apprentice-agent/api/main.py` lines 60-160 to see where systems are initialized.

---

### Task 3.2: Start Gateway Daemon on app startup

**Files:**
- Modify: `api/main.py`

**Step 1: Find the lifespan/startup section in `api/main.py`**

Look for `@app.on_event("startup")` or `@asynccontextmanager async def lifespan(app)`.

**Step 2: Add gateway daemon startup**

In the startup sequence, after global workspace and self-improvement are initialized, add:

```python
# Start proactive gateway daemon
try:
    from aura.proactive.gateway_daemon import get_gateway_daemon
    gateway = get_gateway_daemon(start=True)
    logger.info(f"[OK] Proactive gateway started (state: {gateway.state.value})")
except Exception as e:
    logger.warning(f"[WARN] Proactive gateway failed to start: {e}")
```

**Step 3: Verify startup doesn't crash**
```bash
cd D:/apprentice-agent && python -c "
from api.main import app
print('Import OK')
"
```

**Step 4: Run tests**
```bash
cd D:/apprentice-agent && python -m pytest tests/ -q --tb=no 2>&1 | tail -5
```
Expected: 356 passed

**Step 5: Commit**
```bash
git commit -m "feat: start proactive gateway daemon on app startup"
```

---

### Task 3.3: Start System and Calendar monitors

**Files:**
- Modify: `api/main.py`
- Verify: `aura/proactive/monitors/system_monitor.py`
- Verify: `aura/proactive/monitors/calendar_monitor.py`

**Step 1: Read both monitor files to find their start() API**

Read `aura/proactive/monitors/system_monitor.py` — find `SystemMonitor` class and `start()` method.
Read `aura/proactive/monitors/calendar_monitor.py` — find `get_calendar_monitor()` and start.

**Step 2: Add monitor startup in `api/main.py`**

After gateway startup:
```python
# Start system monitor
try:
    from aura.proactive.monitors.system_monitor import SystemMonitor
    system_monitor = SystemMonitor(event_bus=gateway.event_bus)
    system_monitor.start()
    logger.info("[OK] System monitor started")
except Exception as e:
    logger.warning(f"[WARN] System monitor failed: {e}")

# Start calendar monitor (requires icalendar)
try:
    from aura.proactive.monitors.calendar_monitor import get_calendar_monitor
    cal_monitor = get_calendar_monitor()
    cal_monitor.start()
    logger.info("[OK] Calendar monitor started")
except Exception as e:
    logger.warning(f"[WARN] Calendar monitor failed: {e}")
```

**Step 3: Verify Event Bus has publishers**

Read `aura/proactive/event_bus.py` — verify SystemMonitor actually publishes events to it.

**Step 4: Run tests**
```bash
cd D:/apprentice-agent && python -m pytest tests/ -q --tb=no 2>&1 | tail -5
```
Expected: 356 passed

**Step 5: Commit**
```bash
git commit -m "feat: start system and calendar monitors in proactive gateway"
```

---

## PHASE 4: Expose Missing Cognitive Tools

### Task 4.1: Add MetacogGuardian to tool registry

**Files:**
- Read: `aura/tools/metacog_guardian.py` (find class name + `execute()` interface)
- Modify: `aura/agent.py` (tool loading section)

**Step 1: Find tool loading section in agent.py**
```bash
grep -n "self.tools\[" D:/apprentice-agent/aura/agent.py | head -30
```

**Step 2: Read metacog_guardian.py for its interface**

Find: class name, constructor args, and how `execute()` is called.

**Step 3: Add to tool loading**

In the tool loading section of `agent.py`, add:
```python
try:
    from aura.tools.metacog_guardian import MetacogGuardian
    self.tools['metacog_guardian'] = MetacogGuardian()
    logger.debug("[LOADED] MetacogGuardian")
except Exception as e:
    logger.debug(f"[SKIP] MetacogGuardian: {e}")
```

**Step 4: Write a test**
```python
# In tests/test_tool_loading.py (create if doesn't exist)
def test_metacog_guardian_loads():
    from aura import ApprenticeAgent
    agent = ApprenticeAgent()
    assert 'metacog_guardian' in agent.tools
```

**Step 5: Run test**
```bash
cd D:/apprentice-agent && python -m pytest tests/test_tool_loading.py -v
```

**Step 6: Run full suite**
```bash
cd D:/apprentice-agent && python -m pytest tests/ -q --tb=no 2>&1 | tail -5
```

**Step 7: Commit**
```bash
git commit -m "feat: expose MetacogGuardian, CognitiveTheater, SynapseForge in tool registry"
```

Note: Do CognitiveTheater and SynapseForge in the same commit — same pattern.

---

### Task 4.2: Add CognitiveTheater and SynapseForge (same pattern as 4.1)

**Files:**
- `aura/tools/cognitive_theater.py`
- `aura/tools/synapseforge.py`

Follow exact same pattern as Task 4.1. Key thing to check for SynapseForge: it may need a `tools` reference passed to it (since it creates new tools).

---

## PHASE 5: Code Quality Sweep

### Task 5.1: Fix datetime.utcnow() deprecation warnings

**Step 1: Find all occurrences**
```bash
grep -rn "datetime.utcnow()" D:/apprentice-agent/aura D:/apprentice-agent/aura D:/apprentice-agent/api --include="*.py" | grep -v __pycache__
```

**Step 2: Replace all occurrences**

Replace pattern: `datetime.utcnow()` → `datetime.now(datetime.UTC)`

Also check imports — some files may do `from datetime import datetime` and then call `datetime.utcnow()`. The fix is:
```python
# Before:
from datetime import datetime
now = datetime.utcnow()

# After:
from datetime import datetime, timezone
now = datetime.now(timezone.utc)
```

**Step 3: Run full suite — verify no new failures**
```bash
cd D:/apprentice-agent && python -m pytest tests/ -q --tb=no -W error::DeprecationWarning 2>&1 | tail -10
```
Expected: 356 passed, 0 DeprecationWarnings from utcnow

**Step 4: Commit**
```bash
git commit -m "fix: replace deprecated datetime.utcnow() with datetime.now(timezone.utc)"
```

---

### Task 5.2: Fix Soul config interface

**Files:**
- Read: `aura/soul/soul_loader.py` (or `aura/soul/soul_loader.py` before merge)
- Find all callers using `.get()` on the SoulConfig object

**Step 1: Find the SoulConfig class definition**
```bash
grep -n "class SoulConfig" D:/apprentice-agent/aura/soul/soul_loader.py D:/apprentice-agent/aura/soul/soul_loader.py 2>/dev/null
```

**Step 2: Add dict-like interface to SoulConfig**

```python
class SoulConfig:
    # ... existing code ...

    def get(self, key, default=None):
        return getattr(self, key, default)

    def __getitem__(self, key):
        return getattr(self, key)

    def to_dict(self):
        return {k: v for k, v in vars(self).items() if not k.startswith('_')}
```

**Step 3: Run tests**
```bash
cd D:/apprentice-agent && python -m pytest tests/ -q --tb=no 2>&1 | tail -5
```

**Step 4: Commit**
```bash
git commit -m "fix: add dict-like interface to SoulConfig for .get() compatibility"
```

---

### Task 5.3: Fix custom tool security false positives

**Files:**
- `data/custom/bmi_calculator.py` (or wherever rejected tools live)
- `data/custom/temperature_converter.py`
- Read: `aura/tools/validation.py` (find the security scanner)

**Step 1: Read the rejected tool files**

Check if `urllib` is actually needed. If it's just `from urllib.parse import quote` for encoding (safe), add to whitelist. If it's making network requests, flag for review.

**Step 2: Fix the validation whitelist**

In `aura/tools/validation.py`, find the pattern that flags `urllib`. If the pattern is overly broad (e.g., flags `urllib.parse` which is safe), narrow it:

```python
# Overly broad (flags safe uses):
BLOCKED_PATTERNS = [r'urllib']

# Better (only flag network urllib):
BLOCKED_PATTERNS = [r'urllib\.request', r'urllib\.urlopen']
```

**Step 3: Run tests**
```bash
cd D:/apprentice-agent && python -m pytest tests/ -q --tb=no 2>&1 | tail -5
```

**Step 4: Commit**
```bash
git commit -m "fix: narrow urllib security pattern to only flag network requests, not urllib.parse"
```

---

### Task 5.4: Deduplicate skill library index

**Files:**
- Read/Modify: skill index file in `aura_data/skill_library/`

**Step 1: Find the index file**
```bash
find D:/apprentice-agent/aura_data/skill_library -name "*.json" -o -name "index*" 2>/dev/null
```

**Step 2: Load and deduplicate**
```python
import json
with open('path/to/index.json') as f:
    index = json.load(f)
# Remove duplicate "Python Code Reviewer"
# Keep the one with more complete data
```

**Step 3: Write back and verify**
```bash
cd D:/apprentice-agent && python -c "from aura_skill_library import SkillLibrary; sl = SkillLibrary(); print(len(sl.list_skills()), 'skills')"
```

**Step 4: Commit**
```bash
git commit -m "fix: deduplicate Python Code Reviewer in skill library index"
```

---

## PHASE 6: Final Verification

### Task 6.1: Full test suite with all warnings as errors

```bash
cd D:/apprentice-agent && python -m pytest tests/ -v --tb=short -W error::DeprecationWarning 2>&1 | tail -30
```
Expected: 356 passed, 0 failures, 0 deprecation warnings

### Task 6.2: Verify proactive systems actually start

```bash
cd D:/apprentice-agent && python -c "
from aura.proactive.gateway_daemon import get_gateway_daemon
import time
gw = get_gateway_daemon(start=True)
time.sleep(2)
print('State:', gw.state.value)
print('Running:', gw._running)
gw.stop()
print('Stopped cleanly')
"
```
Expected: State: observing (or idle), Running: True, Stopped cleanly

### Task 6.3: Verify cognitive tools are all accessible

```bash
cd D:/apprentice-agent && python -c "
from aura import ApprenticeAgent
a = ApprenticeAgent()
required = ['metacog_guardian', 'cognitive_theater', 'synapseforge', 'reflexion', 'neurodream', 'worldsim']
for t in required:
    status = 'OK' if t in a.tools else 'MISSING'
    print(f'{t}: {status}')
"
```
Expected: All `OK`

### Task 6.4: Verify no duplicate imports from aura/

```bash
grep -rn "from aura\." D:/apprentice-agent --include="*.py" | grep -v __pycache__ | grep -v "aura_data\|aura_episodic\|aura_knowledge\|aura_skill\|aura_life\|aura_ollama"
```
Expected: Empty (or only `aura/messaging/` if not yet deleted)

### Task 6.5: Final commit and tag

```bash
git add -A
git commit -m "chore: AURA consolidation complete — single architecture, all systems wired"
git tag v2.0-consolidated
```

---

## Summary of Changes

| Phase | What Changes | Risk |
|-------|-------------|------|
| 1 | +icalendar deps, fix neurodream test | Low |
| 2 | Move PatternProphet + messaging, delete duplicates | Medium |
| 3 | Wire proactive gateway on startup | Medium |
| 4 | Expose 3 cognitive tools | Low |
| 5 | datetime fix, SoulConfig fix, security fix | Low |
| 6 | Verification pass | None |

**Total expected test delta:** 355 → 356 (fix 1 pre-existing failure, add no new failures)
