# Full Audit Fix Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix all 85 findings from the 5-agent Sonnet audit covering architecture, security, AI/ML correctness, performance, and integration.

**Architecture:** All fixes are surgical — no feature additions. Tasks are ordered by severity and dependency (critical cloud routing first, then security, then correctness, then perf, then cleanup). Each task is self-contained in one file or one logical group.

**Tech Stack:** Python 3.12, FastAPI, asyncio, threading, ollama SDK, `aura/config.py` Config class

---

## Task 1: Fix cloud host URL + cloud suffix detection (brain.py)

**Files:**
- Modify: `aura/brain.py:185` (cloud host URL)
- Modify: `aura/brain.py:243` (suffix check)

**Context:**
All cloud inference is currently broken because `OLLAMA_CLOUD_HOST` points to the marketing website instead of the API. Additionally the `_get_client_for_model` method only checks `endswith("-cloud")` but all models in Config use `:cloud` tag format (e.g. `gemini-3-flash-preview:cloud`), so every cloud model call falls through to the local client.

**Step 1: Read the current code**

Read `D:/Aura/aura/brain.py` lines 183–250.

**Step 2: Fix the cloud host URL**

Find:
```python
OLLAMA_CLOUD_HOST = "https://ollama.com"
```
Replace with:
```python
OLLAMA_CLOUD_HOST = "https://api.ollama.com"
```

**Step 3: Fix the cloud suffix check**

Find (around line 243):
```python
if model.endswith("-cloud"):
```
Replace with:
```python
if model.endswith(("-cloud", ":cloud")):
```

**Step 4: Verify**

```bash
cd D:/Aura && python -c "from aura.brain import OllamaBrain, OLLAMA_CLOUD_HOST; print('host:', OLLAMA_CLOUD_HOST); print('OK')"
```
Expected:
```
host: https://api.ollama.com
OK
```

---

## Task 2: Fix _select_model dead cloud branch (brain.py)

**Files:**
- Modify: `aura/brain.py:1481–1560` (`_select_model` method)

**Context:**
`_select_model` uses `getattr(Config, 'MODEL_VISION_CLOUD', Config.MODEL_VISION)` etc. — but `MODEL_VISION_CLOUD`, `MODEL_CODE_CLOUD`, `MODEL_REASON_CLOUD` do not exist on Config. The getattr always returns the fallback default which is already the correct cloud model. This means the `use_cloud` branch is dead code and the `_is_complex_query` check conflates "should I use cloud?" with "should I escalate to System 2?", causing long-but-simple queries to hit the 397B reason model.

Read `D:/Aura/aura/brain.py` lines 1481–1560 first.

**Step 1: Fix the vision/code/identity branches**

The fix: remove the ternary `if use_cloud else` pattern entirely. All Config model attrs are already cloud models. The routing logic should be:
- vision queries → `Config.get_model("vision")`
- code queries → `Config.get_model("code")`
- identity queries → `Config.MODEL_FAST`

Find patterns like:
```python
return getattr(Config, 'MODEL_VISION_CLOUD', Config.MODEL_VISION) if use_cloud else Config.MODEL_VISION
```
Replace with:
```python
return Config.get_model("vision")
```

Find:
```python
return getattr(Config, 'MODEL_CODE_CLOUD', Config.MODEL_CODE) if use_cloud else Config.MODEL_CODE
```
Replace with:
```python
return Config.get_model("code")
```

Find (general reason cloud fallbacks):
```python
return getattr(Config, 'MODEL_REASON_CLOUD', Config.MODEL_REASON) if use_cloud else Config.MODEL_REASON
model = getattr(Config, 'MODEL_REASON_CLOUD', Config.MODEL_REASON) if use_cloud else Config.MODEL_REASON
```
Replace with:
```python
return Config.get_model("reason")
model = Config.get_model("reason")
```

**Step 2: Fix _is_complex_query in _should_escalate_to_system2 dead code**

Find in `_should_escalate_to_system2` (around line 1459):
```python
if len(prompt.split()) > 50:
```
Remove this block — it's unreachable dead code. `_is_complex_query` already checks this.

**Step 3: Fix code_patterns — remove false positives**

Find (around line 1513–1516):
```python
code_patterns = [
    ...
    'print(', 'import ', 'def ', 'for ', 'while ', 'python',
```
Change `'for ', 'while ',` — remove those two. Replace the list with word-boundary safe patterns. Simply delete those two entries:
```python
code_patterns = [
    ...
    'print(', 'import ', 'def ', 'python',
```

**Step 4: Verify**

```bash
cd D:/Aura && python -c "from aura.brain import OllamaBrain; print('OK')"
```
Expected: `OK`

---

## Task 3: Fix warmup to use correct client (brain.py)

**Files:**
- Modify: `aura/brain.py:271–285` (`_warmup_models` method)

**Context:**
`_warmup_models` calls `self.client.generate(model=Config.MODEL_FAST, ...)` where `self.client` is the local Ollama client. But `Config.MODEL_FAST` is `gemini-3-flash-preview:cloud` — a cloud model. This always silently fails, wasting up to 10 seconds of startup time.

**Step 1: Read the current warmup method**

Read `D:/Aura/aura/brain.py` lines 271–290.

**Step 2: Fix to skip warmup for cloud models**

Replace the `_warmup_models` body. Cloud models don't need keepalive warming — they're remote. Only warm local models.

Find the `_warmup_models` method and replace its body:
```python
def _warmup_models(self) -> None:
    """Warm up local Ollama models with a keep-alive ping. Skipped for cloud models."""
    models_to_warm = [
        m for m in [Config.MODEL_FAST, Config.MODEL_REASON, Config.MODEL_CODE]
        if not m.endswith(("-cloud", ":cloud"))
    ]
    if not models_to_warm:
        logger.info("[BRAIN] All models are cloud-hosted — skipping warmup")
        return
    for model in models_to_warm:
        try:
            call_with_timeout(
                lambda m=model: self.client.generate(model=m, prompt="", keep_alive="30m"),
                timeout=WARMUP_TIMEOUT,
                default=None
            )
            logger.info(f"[BRAIN] Warmed up local model: {model}")
        except Exception as e:
            logger.warning(f"[BRAIN] Warmup failed for {model}: {e}")
```

**Step 3: Verify**

```bash
cd D:/Aura && python -c "from aura.brain import OllamaBrain; print('OK')"
```

---

## Task 4: Fix _save_history outside lock + TOCTOU double-lock (brain.py)

**Files:**
- Modify: `aura/brain.py:1097–1121` (think method)
- Modify: `aura/brain.py:1339–1365` (think_stream method)

**Context:**
Two performance/correctness bugs in the hot path:
1. `_save_history()` (full JSON disk write) is called *inside* `_history_lock`, serializing all concurrent requests. Fix: copy data inside lock, write outside.
2. Two consecutive `with self._history_lock:` blocks create a TOCTOU race where another thread can modify `conversation_history` between the two acquisitions. Fix: merge into one lock block.

**Step 1: Read the affected sections**

Read `D:/Aura/aura/brain.py` lines 1093–1130 and 1335–1375.

**Step 2: Fix think() — merge two lock blocks into one, move disk write outside**

Find in `think()`:
```python
if use_history:
    with self._history_lock:
        self.conversation_history.append({"role": "user", "content": prompt})
        self.conversation_history.append({"role": "assistant", "content": assistant_message})
        self._save_history()
...
with self._history_lock:
    recent = list(self.conversation_history[-6:]) if use_history else [
```

Replace with a single block that captures both the append and the slice, then writes to disk after releasing the lock:
```python
if use_history:
    with self._history_lock:
        self.conversation_history.append({"role": "user", "content": prompt})
        self.conversation_history.append({"role": "assistant", "content": assistant_message})
        recent = list(self.conversation_history[-6:])
        history_snapshot = list(self.conversation_history)  # copy for save
    self._save_history_snapshot(history_snapshot)  # disk write outside lock
else:
    recent = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": assistant_message},
    ]
```

Note: `_save_history_snapshot` is a new private method that accepts a pre-copied list. See Step 4.

**Step 3: Apply the same fix to think_stream()**

Find the equivalent two-block pattern in `think_stream()` (lines 1339–1365) and apply the identical transformation.

**Step 4: Add _save_history_snapshot method**

After `_save_history`, add:
```python
def _save_history_snapshot(self, history: list) -> None:
    """Save a pre-copied history list to disk (called OUTSIDE _history_lock)."""
    try:
        self.history_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.history_file, 'w', encoding='utf-8') as f:
            json.dump(history[-self.MAX_HISTORY_LENGTH:], f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"[BRAIN] Failed to save history snapshot: {e}")
```

**Step 5: Verify**

```bash
cd D:/Aura && python -c "from aura.brain import OllamaBrain; print('OK')"
```

---

## Task 5: Fix ACTION_MODE_MODELS + GET /api/init + duplicate /api/mood

**Files:**
- Modify: `D:/Aura/api/services/agent_service.py:164–210` (ACTION_MODE_MODELS)
- Modify: `D:/Aura/api/routes/status.py:152–165` (GET /api/init)
- Modify: `D:/Aura/api/routes/status.py:167–185` (duplicate /api/mood)

### Part A — Fix ACTION_MODE_MODELS

**Context:** The model names use stale identifiers (`devstral-small-2:24b-cloud`, `deepseek-v3.1`, `glm-4.7`, `kimi-k2.5-cloud` with wrong separator). Must match VERIFIED_CLOUD_MODELS format (`name:cloud` suffix with colon).

Read `D:/Aura/api/services/agent_service.py` lines 163–213.

Replace the `ACTION_MODE_MODELS` dict:
```python
ACTION_MODE_MODELS = {
    "search": {
        "preferred": "gemini-3-flash-preview:cloud",
        "fallbacks": ["nemotron-3-nano:30b-cloud", "kimi-k2.5:cloud"],
        "description": "Quick web search"
    },
    "research": {
        "preferred": "qwen3.5:397b-cloud",
        "fallbacks": ["cogito-2.1:671b-cloud", "deepseek-v3.2:cloud"],
        "description": "Comprehensive research"
    },
    "agent": {
        "preferred": "kimi-k2.5:cloud",
        "fallbacks": ["devstral-2:123b-cloud", "deepseek-v3.2:cloud"],
        "description": "Autonomous task execution"
    },
    "code": {
        "preferred": "qwen3-coder:480b-cloud",
        "fallbacks": ["devstral-2:123b-cloud", "qwen3-coder-next:cloud"],
        "description": "Code generation and analysis"
    },
    "vision": {
        "preferred": "qwen3-vl:235b-cloud",
        "fallbacks": ["kimi-k2.5:cloud", "gemini-3-flash-preview:cloud"],
        "description": "Image analysis"
    },
    "deep_research": {
        "preferred": "kimi-k2-thinking:cloud",
        "fallbacks": ["qwen3.5:397b-cloud", "cogito-2.1:671b-cloud"],
        "description": "Multi-source deep research with page reading"
    },
    "swarm": {
        "preferred": "qwen3.5:397b-cloud",
        "fallbacks": ["cogito-2.1:671b-cloud", "kimi-k2.5:cloud"],
        "description": "Multi-agent parallel collaboration"
    }
}
```

Also fix `_is_model_available()` if it checks `endswith("-cloud")` only — change to `endswith(("-cloud", ":cloud"))`.

### Part B — Fix GET /api/init

**Context:** `GET /api/init` reads `app.state.init_state` which is never written. Always returns `ready: false`.

Read `D:/Aura/api/routes/status.py` lines 152–165.

Replace the handler to read from `agent_service.is_ready` directly:
```python
@router.get("/api/init", response_model=InitStatus)
async def get_init_status(request: Request) -> InitStatus:
    """Get agent initialization status."""
    try:
        from api.services.agent_service import agent_service
        if agent_service.is_ready:
            return InitStatus(ready=True, progress="complete")
        elif agent_service._initializing:
            return InitStatus(ready=False, progress="initializing")
        else:
            return InitStatus(ready=False, progress="not_started")
    except Exception as e:
        return InitStatus(ready=False, progress=f"error: {e}")
```

### Part C — Remove duplicate /api/mood stub from status.py

**Context:** `GET /api/mood` exists in both `features.py` (working, reads from ALMA) and `status.py` (hardcoded stub). The status.py version is unreachable (features.py registered first) and should be removed.

Read `D:/Aura/api/routes/status.py` around lines 167–185.

Find the `GET /mood` route with a hardcoded `MoodState(emotion="curious", ...)` response and delete the entire route handler and its model definition if only used there.

**Verify:**
```bash
cd D:/Aura && python -c "from api.routes.status import router; from api.services.agent_service import agent_service; print('OK')"
```

---

## Task 6: Fix API security defaults (main.py, middleware.py)

**Files:**
- Modify: `D:/Aura/api/main.py:82–91` (proactive callback thread safety)
- Modify: `D:/Aura/api/main.py:295–302` (rate limit independent of auth)
- Modify: `D:/Aura/api/main.py:356–365` (default binding)
- Modify: `D:/Aura/api/middleware.py:56–61` (remove API key from query param)

### Part A — Fix proactive callback asyncio.create_task in sync context

**Context:** `_on_proactive_message` is a sync callback called from the daemon's background thread. `asyncio.create_task()` inside a sync function called from a non-event-loop thread raises RuntimeError.

Read `D:/Aura/api/main.py` lines 78–95.

Before `daemon.set_notification_callback`, capture the running loop. Then use `loop.call_soon_threadsafe`:
```python
_loop = asyncio.get_running_loop()

def _on_proactive_message(msg):
    logger.info(f"[Proactive] {msg.action.value}: {msg.content[:80]}...")
    try:
        from api.routes.chat import broadcast_proactive_message
        _loop.call_soon_threadsafe(
            _loop.create_task,
            broadcast_proactive_message(msg)
        )
    except Exception as e:
        logger.debug(f"[Proactive] WebSocket push failed: {e}")

daemon.set_notification_callback(_on_proactive_message)
```

### Part B — Make rate limiting independent of auth

**Context:** Rate limit is only enabled when `_auth_enabled=True`. Without this, the `/api/shell/run` endpoint has no DoS protection by default.

Read `D:/Aura/api/main.py` lines 295–302.

Find:
```python
app.add_middleware(
    RateLimitMiddleware,
    requests_per_minute=getattr(_auth_cfg, 'API_RATE_LIMIT', 60),
    enabled=_auth_enabled,
)
```
Replace with:
```python
app.add_middleware(
    RateLimitMiddleware,
    requests_per_minute=getattr(_auth_cfg, 'API_RATE_LIMIT', 60),
    enabled=True,   # Always rate-limit regardless of auth setting
)
```

### Part C — Change default binding to 127.0.0.1

**Context:** Server binds to `0.0.0.0` by default — exposes all endpoints to the local network without authentication.

Read `D:/Aura/api/main.py` lines 354–366.

Find:
```python
uvicorn.run("api.main:app", host="0.0.0.0", port=8000, ...)
```
Replace with:
```python
_host = os.environ.get("AURA_HOST", "127.0.0.1")
uvicorn.run("api.main:app", host=_host, port=8000, ...)
```

### Part D — Remove API key from query param

**Context:** API key in query params gets logged in server logs and browser history.

Read `D:/Aura/api/middleware.py` lines 54–65.

Find:
```python
api_key = request.headers.get("X-API-Key") or request.query_params.get("api_key")
```
Replace with:
```python
api_key = request.headers.get("X-API-Key")
```

Update the 401 error message to no longer mention the query param option:
```python
content={"detail": "Missing API key. Provide X-API-Key header."}
```

**Verify:**
```bash
cd D:/Aura && python -c "from api.main import app; print('OK')"
```

---

## Task 7: Fix asyncio.get_event_loop() → get_running_loop() in all API routes

**Files:**
- Modify: All files in `D:/Aura/api/routes/` that contain `asyncio.get_event_loop()`

**Context:** Python 3.12 deprecates `asyncio.get_event_loop()` when called inside a running async context. There are 93 occurrences across API route files. All must be changed to `asyncio.get_running_loop()`.

**Step 1: Find all affected files**

```bash
cd D:/Aura && grep -rl "get_event_loop" api/routes/ api/services/ 2>/dev/null
```

**Step 2: For each file, replace all occurrences**

```bash
cd D:/Aura
for f in $(grep -rl "get_event_loop" api/routes/ api/services/ 2>/dev/null); do
  sed -i 's/asyncio\.get_event_loop()/asyncio.get_running_loop()/g' "$f"
  echo "Fixed: $f"
done
```

**Step 3: Verify no get_event_loop remains in async route handlers**

```bash
cd D:/Aura && grep -rn "get_event_loop" api/routes/ api/services/ 2>/dev/null | head -5
```
Expected: no results (or only in non-async helper functions that are called from sync threads — those should remain `get_event_loop`).

**Step 4: Verify imports**

```bash
cd D:/Aura && python -c "from api.routes import features, status, tools_new, chat; print('OK')"
```
Expected: `OK`

---

## Task 8: Fix MCTS stale evaluation + KG extractor interface + fix code routing

**Files:**
- Modify: `D:/Aura/aura/tools/mcts_reasoning.py:526–527` (stale MCTS caching)
- Modify: `D:/Aura/aura/tools/kg_extractor.py:109–115` (generate call signature)
- Brain.py code routing already fixed in Task 2

### Part A — Fix MCTS stale node evaluation

**Context:** `_evaluate()` returns the cached `avg_value` after first evaluation, preventing value convergence. MCTS trees need re-evaluation to converge.

Read `D:/Aura/aura/tools/mcts_reasoning.py` lines 517–570.

Find:
```python
def _evaluate(self, node: MCTSNode) -> float:
    ...
    if node.state == NodeState.EVALUATED and node.visits > 0:
        return node.avg_value
```

Remove those two lines (the early-return cache). Leave the rest of `_evaluate` intact. Nodes will now be re-evaluated on each visit, allowing value estimates to converge.

### Part B — Fix KG extractor generate call

**Context:** `self.llm.generate(prompt, model=Config.get_model("fast"))` may fail with TypeError if `OllamaBrain.generate` has a different signature. This silently degrades KG to regex-only.

**Step 1:** Read `D:/Aura/aura/tools/kg_extractor.py` lines 105–125.

**Step 2:** Read `D:/Aura/aura/brain.py` and search for `def generate` to find the actual signature.

```bash
cd D:/Aura && grep -n "def generate\|def quick_generate\|def think" aura/brain.py | head -10
```

**Step 3:** Update the call in `kg_extractor.py` to match the actual signature. If `OllamaBrain.generate(prompt, model=...)` is not the right form, change it to `self.llm.think(prompt)` or `self.llm._quick_generate(prompt, model=Config.get_model("fast"))` — whichever exists on `OllamaBrain`. If it fails try calling with just the prompt.

**Verify:**
```bash
cd D:/Aura && python -c "from aura.tools.kg_extractor import KnowledgeExtractor; print('OK')"
```

---

## Task 9: Fix SSRF in API tester + urllib allowlist + exception leakage + dead aura_ollama_client

**Files:**
- Modify: `D:/Aura/aura/tools/api_tester.py:82–105` (SSRF protection)
- Modify: `D:/Aura/aura/agent.py:32–45` (urllib allowlist)
- Modify: `D:/Aura/api/routes/tools_new.py` (exception leakage — pervasive)
- Delete: `D:/Aura/aura/aura_ollama_client.py` (dead code)

### Part A — Fix SSRF in api_tester.py

**Context:** The `request()` method accepts any URL with no SSRF protection. Attackers can probe internal services or cloud metadata endpoints.

Read `D:/Aura/aura/tools/api_tester.py` lines 82–105.

After the existing URL validation block (after `if not urlparse(url).netloc:`), add SSRF protection:

```python
# SECURITY: Block SSRF targets — private/loopback ranges and metadata endpoints
import ipaddress
_parsed = urlparse(url)
_hostname = _parsed.hostname or ""
_ssrf_blocked = [
    "169.254.169.254",  # AWS/GCP/Azure metadata
    "metadata.google.internal",
    "169.254.170.2",    # ECS metadata
]
if _hostname in _ssrf_blocked:
    return {"success": False, "error": "Blocked: metadata endpoint"}
try:
    _ip = ipaddress.ip_address(_hostname)
    if _ip.is_private or _ip.is_loopback or _ip.is_link_local:
        return {"success": False, "error": "Blocked: private/loopback IP addresses not allowed"}
except ValueError:
    pass  # hostname is a DNS name, not an IP — allow
```

### Part B — Fix urllib allowlist

**Context:** `"urllib"` in `ALLOWED_TOOL_IMPORTS` permits `urllib.request` which can make outbound HTTP from custom tool code.

Read `D:/Aura/aura/agent.py` lines 32–45.

Find:
```python
"urllib",  # urllib.parse is safe (URL encoding/decoding)
```
Replace with:
```python
"urllib.parse",  # Only URL encoding/decoding — urllib.request is not allowed
```

Also check the `FORBIDDEN_PATTERNS` section nearby (around line 43) — verify `"urllib.request"` is already in the forbidden list. If not, add it:
```python
"urllib.request", "urllib.urlopen",
```

### Part C — Fix exception leakage in API routes

**Context:** `except Exception as e: return {"success": False, "error": str(e)}` leaks internal error messages to callers.

This pattern appears 80+ times. The fix: log the full error internally, return generic message externally.

Add a helper at the top of `D:/Aura/api/routes/tools_new.py` (after imports):
```python
def _safe_error(e: Exception, context: str = "") -> str:
    """Log full error internally, return generic message to client."""
    logger.error(f"[API] {context}: {e}", exc_info=True)
    return "Internal error — check server logs"
```

Then do a targeted replace for the most sensitive routes (shell, code execution, file operations) — not every route needs this if it's a simple read. For the shell executor route and code executor route specifically, replace:
```python
return {"success": False, "error": str(e)}
```
With:
```python
return {"success": False, "error": _safe_error(e, "shell_run")}
```
(adapt context string per route)

For routes that deal with AI content (web_search results, clipboard reads), keep `str(e)` since those errors are useful to the client.

### Part D — Delete dead aura_ollama_client.py

```bash
cd D:/Aura && rm aura/aura_ollama_client.py
```

Verify nothing imports it:
```bash
cd D:/Aura && grep -rn "aura_ollama_client\|OllamaClient" aura/ api/ main.py 2>/dev/null | grep -v ".pyc"
```
Expected: no results (or only historical references in docs).

If any file imports it, update that import to use `OllamaBrain` instead.

**Verify all:**
```bash
cd D:/Aura && python -c "
from aura.tools.api_tester import APITesterTool
from aura.agent import ALLOWED_TOOL_IMPORTS
assert 'urllib.parse' in ALLOWED_TOOL_IMPORTS
assert 'urllib' not in ALLOWED_TOOL_IMPORTS
print('OK')
"
```

---

## Task 10: Fix CognitiveTheater/MirrorMind silent degradation + model staleness

**Files:**
- Modify: `D:/Aura/aura/tools/cognitive_theater.py:124–145` (`_get_client`, `_call_llm`)
- Modify: `D:/Aura/aura/tools/mirrormind.py:107–140` (`__init__`, `_get_client`, `_call_llm`)

**Context:** When cloud model is set but `OLLAMA_API_KEY` is missing, both tools silently return empty strings, causing the agent to use degraded (empty) deliberation/critique results without knowing they failed. Also both freeze their model at construction time instead of reading Config dynamically.

**Step 1: Read current code**

Read `D:/Aura/aura/tools/cognitive_theater.py` lines 115–160.
Read `D:/Aura/aura/tools/mirrormind.py` lines 100–145.

**Step 2: Fix CognitiveTheater _call_llm to return a marked failure**

Find the `_call_llm` method. Change the `None` client early return:
```python
def _call_llm(self, prompt: str) -> str:
    client = self._get_client()
    if client is None:
        return "__NO_CLIENT__"   # Caller can detect this
    ...
```

**Step 3: Fix CognitiveTheater deliberate() to detect and surface failure**

Find where `_call_llm` result is used in the deliberation loop. Add:
```python
raw = self._call_llm(prompt)
if raw == "__NO_CLIENT__":
    logger.warning("[CognitiveTheater] No cloud client — deliberation skipped")
    return None  # Caller receives None, not empty dict
```

**Step 4: Apply the same pattern to MirrorMind**

In `MirrorMind._call_llm`:
```python
if client is None:
    return "__NO_CLIENT__"
```

In the critique/improve methods, detect `"__NO_CLIENT__"` and return early with the original unchanged response.

**Step 5: Fix model staleness — read Config dynamically**

In `CognitiveTheater._get_client()`, instead of using `self.model` (frozen at init), read:
```python
model = self.model or Config.get_model("fast")
if model.endswith(("-cloud", ":cloud")):
    ...
```
This way if `self.model is None`, it always reads the current Config value.

Apply the same to `MirrorMind`.

**Verify:**
```bash
cd D:/Aura && python -c "
from aura.tools.cognitive_theater import CognitiveTheater
from aura.tools.mirrormind import MirrorMind
ct = CognitiveTheater()
mm = MirrorMind()
print('CT model:', ct.model)
print('MM model:', mm.model)
print('OK')
"
```

---

## Task 11: Performance — parallelized model validation + connection reuse + deque + executor offload

**Files:**
- Modify: `D:/Aura/aura/config.py:86–200` (`validate_models_on_startup`)
- Modify: `D:/Aura/aura/agent.py:328–336` (history deque)
- Modify: `D:/Aura/aura/brain.py:1104–1110` (record_chat_outcome blocking)

### Part A — Parallelize model validation + connection reuse

**Context:** `validate_models_on_startup` fires 6 sequential blocking HTTP calls on startup (18 in worst case). These can run concurrently. Also, `validate_model` creates a new TCP connection per call — add a shared Session.

Read `D:/Aura/aura/config.py` lines 55–200.

**Step 1:** Add a module-level shared session at the top of `config.py` (after imports):
```python
_validation_session = None

def _get_validation_session():
    global _validation_session
    if _validation_session is None:
        import requests
        _validation_session = requests.Session()
    return _validation_session
```

**Step 2:** In `validate_model`, change `requests.get(...)` to:
```python
_get_validation_session().get(f"{host}/api/tags", timeout=5)
```

**Step 3:** In `validate_models_on_startup`, parallelize the 6 role checks:
```python
@classmethod
def validate_models_on_startup(cls) -> dict:
    from concurrent.futures import ThreadPoolExecutor, as_completed
    roles = [
        ("fast", cls.MODEL_FAST, cls.MODEL_FAST_CHAIN),
        ("reason", cls.MODEL_REASON, cls.MODEL_REASON_CHAIN),
        ("code", cls.MODEL_CODE, cls.MODEL_CODE_CHAIN),
        ("vision", cls.MODEL_VISION, cls.MODEL_VISION_CHAIN),
        ("think", cls.MODEL_THINK, cls.MODEL_THINK_CHAIN),
        ("longctx", cls.MODEL_LONGCTX, cls.MODEL_LONGCTX_CHAIN),
    ]
    results = {}
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = {
            pool.submit(get_best_available_model, preferred, fallbacks): role
            for role, preferred, fallbacks in roles
        }
        for future in as_completed(futures):
            role = futures[future]
            try:
                results[role] = future.result()
            except Exception:
                results[role] = None
    # Apply results
    with cls._config_lock:
        if results.get("fast"): cls.MODEL_FAST = results["fast"]
        if results.get("reason"): cls.MODEL_REASON = results["reason"]
        if results.get("code"): cls.MODEL_CODE = results["code"]
        if results.get("vision"): cls.MODEL_VISION = results["vision"]
        if results.get("think"): cls.MODEL_THINK = results["think"]
        if results.get("longctx"): cls.MODEL_LONGCTX = results["longctx"]
    return results
```

### Part B — Use deque for agent history

**Context:** `self._history = self._history[-MAX_HISTORY_SIZE:]` creates a new list copy on every truncation. `deque(maxlen=N)` does this in O(1).

Read `D:/Aura/aura/agent.py` lines 325–340.

Find:
```python
self._history.append(item)
if len(self._history) > MAX_HISTORY_SIZE:
    self._history = self._history[-MAX_HISTORY_SIZE:]
```
Replace with:
```python
self._history.append(item)  # deque handles maxlen automatically
```

And change the initialization from:
```python
self._history = []
```
To:
```python
from collections import deque
self._history = deque(maxlen=MAX_HISTORY_SIZE)
```

### Part C — Offload record_chat_outcome to executor

**Context:** `get_self_improvement_engine().record_chat_outcome(...)` blocks `think()` return after the LLM responds.

Read `D:/Aura/aura/brain.py` lines 1104–1115.

Find:
```python
from aura.consciousness.self_improvement import get_self_improvement_engine
get_self_improvement_engine().record_chat_outcome(...)
```
Wrap in executor submit:
```python
try:
    _SHARED_EXECUTOR.submit(
        get_self_improvement_engine().record_chat_outcome,
        # pass the args it needs
    )
except Exception:
    pass
```

Read the exact arguments needed from the existing call before making this change.

**Verify:**
```bash
cd D:/Aura && python -c "from aura.config import Config; print('OK')"
cd D:/Aura && python -c "from aura.agent import ApprenticeAgent; print('OK')" 2>&1 | tail -3
```

---

## Task 12: Fix AURA_ENV in Config + /docs in production + ALMA trigger wiring

**Files:**
- Modify: `D:/Aura/aura/config.py` (add AURA_ENV)
- Modify: `D:/Aura/api/main.py:328–340` (disable docs in production using Config)
- Modify: `D:/Aura/api/routes/status.py:243–267` (wire POST /api/mood/trigger to ALMA)
- Modify: `D:/Aura/api/middleware.py:25–32` (remove /docs from PUBLIC_PATHS in production)

### Part A — Add AURA_ENV to Config

Read `D:/Aura/aura/config.py` around the feature flags section (line 284+).

Add near other env-based config:
```python
AURA_ENV: str = os.getenv("AURA_ENV", "development")  # "development" or "production"

@classmethod
def is_production(cls) -> bool:
    return cls.AURA_ENV == "production"
```

### Part B — Disable /docs in production

Read `D:/Aura/api/main.py` lines 325–345.

The FastAPI `app` is instantiated with `docs_url="/docs"` by default. To disable in production:

Find where `app = FastAPI(...)` is created. Add:
```python
_is_production = os.environ.get("AURA_ENV") == "production"
app = FastAPI(
    ...
    docs_url=None if _is_production else "/docs",
    redoc_url=None if _is_production else "/redoc",
    openapi_url=None if _is_production else "/openapi.json",
)
```

Also remove `/docs`, `/redoc`, `/openapi.json` from `PUBLIC_PATHS` in `middleware.py` — they should not be publicly accessible even when enabled:
```python
PUBLIC_PATHS = {
    "/",
    "/health",
    "/api/status",
    # Docs paths require auth in production; excluded here
}
```

Wait — if auth is disabled by default, removing from PUBLIC_PATHS would block them even in dev. A better fix: keep them in PUBLIC_PATHS but add a conditional in `main.py` to set `docs_url=None` in production mode. Just add the `None` trick to the FastAPI constructor.

### Part C — Wire POST /api/mood/trigger to ALMA

**Context:** `POST /api/mood/trigger` accepts emotion/intensity but doesn't call `alma_engine.trigger_emotion()`.

Read `D:/Aura/api/routes/status.py` lines 243–270.

Find the stub handler and replace its body:
```python
@router.post("/api/mood/trigger")
async def trigger_mood(request: MoodTriggerRequest) -> MoodState:
    """Trigger an emotion in the ALMA engine."""
    try:
        from aura.emotion.alma_engine import alma_engine, trigger_emotion
        trigger_emotion(request.emotion, intensity=request.intensity)
        # Return updated state
        state = alma_engine.get_current_state()
        return MoodState(
            emotion=state.get("dominant_emotion", "neutral"),
            confidence=int(state.get("confidence", 0.5) * 100),
            valence=state.get("valence", 0.0),
            arousal=state.get("arousal", 0.0),
        )
    except ImportError:
        raise HTTPException(status_code=503, detail="ALMA engine not available")
    except Exception as e:
        logger.warning(f"[API] Mood trigger failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to trigger emotion")
```

**Verify all:**
```bash
cd D:/Aura && python -c "
from aura.config import Config
print('AURA_ENV:', Config.AURA_ENV)
from api.main import app
print('docs_url:', app.docs_url)
print('OK')
"
```

---

## Final Verification

After all 12 tasks, run the full import + sanity check:

```bash
cd D:/Aura && python -c "
from aura.brain import OllamaBrain, OLLAMA_CLOUD_HOST
from aura.config import Config, validate_model
from aura.agent import ApprenticeAgent, ALLOWED_TOOL_IMPORTS
from aura.tools.mcts_reasoning import MCTSReasoning
from aura.tools.cognitive_theater import CognitiveTheater
from aura.tools.mirrormind import MirrorMind
from aura.tools.api_tester import APITesterTool
from aura.tools.kg_extractor import KnowledgeExtractor
from api.main import app
from api.services.agent_service import ACTION_MODE_MODELS

# Critical assertions
assert OLLAMA_CLOUD_HOST == 'https://api.ollama.com', f'Wrong host: {OLLAMA_CLOUD_HOST}'
assert 'urllib.parse' in ALLOWED_TOOL_IMPORTS, 'urllib.parse not in allowlist'
assert 'urllib' not in ALLOWED_TOOL_IMPORTS, 'bare urllib still in allowlist'

# Cloud model validation
result = validate_model('gemini-3-flash-preview:cloud')
print('Cloud validate result:', result)

# ACTION_MODE_MODELS format check
for mode, cfg in ACTION_MODE_MODELS.items():
    pref = cfg['preferred']
    assert pref.endswith(':cloud'), f'{mode} preferred model missing :cloud suffix: {pref}'

print('All assertions passed')
print('Fast model:', Config.MODEL_FAST)
print('AURA_ENV:', Config.AURA_ENV)
"
```

Expected output:
```
Cloud validate result: False   (no OLLAMA_API_KEY in test env — correct)
All assertions passed
Fast model: gemini-3-flash-preview:cloud
AURA_ENV: development
```
