# Bug Sweep — Remaining Issues Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix the remaining 13 medium/high bugs found in the audit, covering stale model refs, race conditions, security holes, and trivial cleanup.

**Architecture:** All fixes are surgical — no refactors, no new features. Each task targets one file or one logical group within a file.

**Tech Stack:** Python 3.12, `threading.Lock`, `uuid`, `pathlib.Path`, existing `Config` class

---

## Bug Reference Table

| # | File | Severity | Problem |
|---|------|----------|---------|
| B12 | `aura/brain.py:251-264` | High | `fallback_map` keys use stale model names (`deepseek-v3.1`, `qwen3-next`, `glm-4.7`, `devstral-small-2`) that never match |
| B14 | `aura/brain.py:1118-1122` | Medium | Unbounded `threading.Thread` spawn per LLM call for world model extraction |
| B20 | `aura/brain.py:388` | Medium | Conversation ID uses 6-hex MD5 of predictable inputs — collision risk |
| B22 | `aura/brain.py:811-813` | Medium | `compact_history()` prepends a system message without stripping existing system messages from `old_messages` |
| B6  | `aura/brain.py:1096-1099` | High | `conversation_history.append` + `_save_history()` called without lock — concurrent calls can corrupt `history.json` |
| B13 | `aura/tools/voice_manager.py:154` | High | `_unload_ollama_models` tries to unload `qwen2:1.5b`, `llama3:8b`, `llava` — none exist in cloud-only setup |
| B15 | `aura/agent.py:1467` | Medium | Fast-path system prompt says "You run locally using Ollama with the Llama and Qwen models" — factually wrong |
| B18 | `aura/agent.py:1106` | Medium | Custom tool `file` path loaded without checking it stays inside the project directory — path traversal risk |
| B16 | `aura/config.py:68-78` | Medium | `validate_model()` makes HTTP request to localhost for cloud models — causes ~20 blocking calls at startup, always fails |
| B23 | `aura/config.py:340` | Low | `API_CORS_ORIGINS` defaults to `"*"` — open CORS in production |
| B17 | `aura/tools/web_search.py:43-51` | Medium | `RateLimiter.wait_if_needed()` reads/writes `last_call_time` without lock — concurrent calls bypass rate limit |
| B21 | `aura/fast_path.py:62-66` | Medium | `known_commands` includes `"mood"`, `"feeling"`, `"how are you"` — file's own docstring says NOT to intercept these |
| B7  | `aura/tools/shell_executor.py:220-234` | High | `cd` handler updates session CWD without first running the injection check |
| B8  | `aura/tools/shell_executor.py:61-68` | High | `"set"` in `ALLOWED_COMMANDS_PREFIX` allows `setx` (sets Windows registry env vars permanently) |
| B19 | `aura/tools/shell_executor.py:103` | Medium | `self._sessions` dict accessed/mutated without lock across concurrent requests |
| B24 | `aura/tools/clipboard_history.py:96` | Low | `subprocess.Popen(["clip"], shell=True)` — unnecessary `shell=True` on Windows |
| B25 | `requirements.txt:52` | Low | `cryptography>=41.0.0,<43.0.0` blocks security patches in v43+ |

---

## Task 1: brain.py — Stale fallback_map keys

**File:** `aura/brain.py:250-261`

**Current broken code:**
```python
fallback_map = {
    "deepseek-v3.1": Config.MODEL_REASON,   # wrong — config now has deepseek-v3.2
    "cogito-2.1": Config.MODEL_REASON,
    "qwen3-next": Config.MODEL_REASON,       # wrong — config has qwen3-coder-next
    "devstral-2": Config.MODEL_CODE,
    "glm-4.7": Config.MODEL_CODE,            # not in VERIFIED_CLOUD_MODELS
    "devstral-small-2": Config.MODEL_CODE,   # not in VERIFIED_CLOUD_MODELS
    "qwen3-vl": Config.MODEL_VISION,
    "kimi-k2.5": Config.MODEL_VISION,
    "gpt-oss": Config.MODEL_REASON,
    "qwen3-coder": Config.MODEL_CODE,
}
```

**Fix:** Update keys to match the base names from `VERIFIED_CLOUD_MODELS` in `config.py`. Add missing entries (`kimi-k2-thinking`, `minimax-m2.5`, `qwen3.5`). Remove entries for models no longer in use (`glm-4.7`, `devstral-small-2`).

**Fixed code:**
```python
fallback_map = {
    "deepseek-v3.2": Config.MODEL_REASON,
    "cogito-2.1": Config.MODEL_REASON,
    "qwen3.5": Config.MODEL_REASON,
    "kimi-k2-thinking": Config.MODEL_THINK,
    "minimax-m2.5": Config.MODEL_LONGCTX,
    "devstral-2": Config.MODEL_CODE,
    "qwen3-coder-next": Config.MODEL_CODE,
    "qwen3-coder": Config.MODEL_CODE,
    "qwen3-vl": Config.MODEL_VISION,
    "kimi-k2.5": Config.MODEL_FAST,
    "gpt-oss": Config.MODEL_REASON,
    "gemini-3-flash-preview": Config.MODEL_FAST,
    "nemotron-3-nano": Config.MODEL_FAST,
}
```

**Steps:**
1. Read `aura/brain.py` lines 248–265 to confirm exact current text
2. Replace the `fallback_map` dict body with the fixed version above
3. Verify: `cd D:/Aura && python -c "from aura.brain import OllamaBrain; print('OK')"`
4. Expected: `OK`

---

## Task 2: brain.py — Conversation ID collision + compact_history duplicate system message

**File:** `aura/brain.py`

### Part A — Conversation ID (lines 384–389)

**Current broken code:**
```python
def _generate_conversation_id(self) -> str:
    """Generate a unique conversation ID."""
    import hashlib
    timestamp = int(time.time())
    random_part = hashlib.md5(f"{timestamp}{os.getpid()}{id(self)}".encode()).hexdigest()[:6]
    return f"conv_{timestamp}_{random_part}"
```

**Fix:** Use `uuid4` instead of MD5 of predictable inputs:
```python
def _generate_conversation_id(self) -> str:
    """Generate a unique conversation ID."""
    import uuid
    return f"conv_{int(time.time())}_{uuid.uuid4().hex[:8]}"
```

### Part B — compact_history duplicate system message (lines 781–813)

**Bug:** When `compact_history()` runs, it takes `old_messages = history[:split_point]`. If a previous compaction already stored a `{"role": "system", ...}` entry as the first message, that system message gets included in `old_messages` and passed to the summarization prompt as if it were a user/assistant turn. This produces garbled summaries and, after re-injection, duplicate system messages.

**Fix:** Filter out system messages from `old_messages` before building the summary prompt. Change line 787:
```python
# BEFORE
old_messages = history[:split_point]

# AFTER
old_messages = [m for m in history[:split_point] if m.get("role") != "system"]
```

**Steps:**
1. Read `aura/brain.py` lines 384–390 (conversation_id) and 781–790 (compact_history split)
2. Apply Part A fix to `_generate_conversation_id`
3. Apply Part B fix to the `old_messages` assignment in `compact_history`
4. Verify: `cd D:/Aura && python -c "from aura.brain import OllamaBrain; print('OK')"`

---

## Task 3: brain.py — Thread safety for history writes + bounded WM extraction threads

**File:** `aura/brain.py`

### Part A — Add `_history_lock` to prevent concurrent `history.json` corruption (Bug B6)

The `think()` method appends to `self.conversation_history` and calls `self._save_history()` without any lock. A second concurrent call can interleave.

**Step 1:** Find `__init__` of `OllamaBrain`. Read lines around where `self.conversation_history` is initialized to find the exact line.

**Step 2:** Add lock initialization in `__init__`, right after `self.conversation_history = ...` is set:
```python
self._history_lock = threading.Lock()
```

**Step 3:** Find the section in `think()` around lines 1095–1099:
```python
if use_history:
    self.conversation_history.append({"role": "user", "content": prompt})
    self.conversation_history.append({"role": "assistant", "content": assistant_message})
    self._save_history()
```
Wrap it:
```python
if use_history:
    with self._history_lock:
        self.conversation_history.append({"role": "user", "content": prompt})
        self.conversation_history.append({"role": "assistant", "content": assistant_message})
        self._save_history()
```

**Step 4:** Also protect the world model extraction's read of `conversation_history`. Around line 1114:
```python
recent = self.conversation_history[-6:] if use_history else [...]
```
Change to:
```python
with self._history_lock:
    recent = list(self.conversation_history[-6:]) if use_history else [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": assistant_message},
    ]
```

### Part B — Cap world model extraction thread spawning (Bug B14)

**Current code (lines 1118–1122):**
```python
threading.Thread(
    target=_run_world_model_extraction,
    args=(conv_id, list(recent)),
    daemon=True, name="wm-extract",
).start()
```

**Fix:** Use the module-level `_SHARED_EXECUTOR` thread pool if it exists, otherwise keep the daemon thread but make the name unique:
```python
try:
    _SHARED_EXECUTOR.submit(_run_world_model_extraction, conv_id, list(recent))
except Exception:
    threading.Thread(
        target=_run_world_model_extraction,
        args=(conv_id, list(recent)),
        daemon=True,
        name=f"wm-extract-{conv_id[:8]}",
    ).start()
```

First check if `_SHARED_EXECUTOR` exists in the file. If it does not, just add the unique name to the daemon thread (remove the `_SHARED_EXECUTOR` path).

**Steps:**
1. Read `OllamaBrain.__init__` — search for `conversation_history` init line
2. Add `self._history_lock = threading.Lock()` after it
3. Wrap the history append + save block in `think()` with `_history_lock`
4. Wrap the `recent = self.conversation_history[-6:]` read with `_history_lock`
5. Search for `_SHARED_EXECUTOR` in brain.py — use it if found, otherwise just make thread name unique
6. Verify: `python -c "from aura.brain import OllamaBrain; print('OK')"`

---

## Task 4: voice_manager.py — Remove stale local model unload list

**File:** `aura/tools/voice_manager.py:154`

**Current broken code:**
```python
models_to_unload = ["qwen2:1.5b", "llama3:8b", "llava"]
```

**Fix:** Derive from `Config.get_all_models()` to always reflect the current active models:
```python
from ..config import Config
models_to_unload = list(Config.get_all_models().values())
```

**Steps:**
1. Read `aura/tools/voice_manager.py` lines 1–20 to check if Config is already imported
2. Add `from ..config import Config` if missing
3. Replace the hardcoded list with `list(Config.get_all_models().values())`
4. Verify: `python -c "from aura.tools.voice_manager import VoiceManager; print('OK')"` (or whatever the class is named)

---

## Task 5: agent.py — Stale system prompt + custom tool path traversal

**File:** `aura/agent.py`

### Part A — System prompt stale text (line 1467)

**Current:**
```python
- You run locally using Ollama with the Llama and Qwen models
```

**Fix:**
```python
- You run on Ollama cloud models (Pro subscription) — fast, reasoning, code, vision, and thinking specialists
```

### Part B — Custom tool path traversal (lines 1105–1108)

**Current:**
```python
tool_file = Path(tool_entry.get("file", ""))
if not tool_file.exists():
    continue
```

**Fix:** Resolve the path and verify it stays within the Aura project directory before opening:
```python
# SECURITY: Resolve path and verify it stays within the project tools directory
tools_base = Path(__file__).parent.resolve()
tool_file = Path(tool_entry.get("file", ""))
try:
    tool_file_resolved = tool_file.resolve()
except Exception:
    continue
if not str(tool_file_resolved).startswith(str(tools_base)):
    logger.warning(f"[SECURITY] Custom tool path outside project directory: {tool_file}")
    continue
if not tool_file_resolved.exists():
    continue
tool_file = tool_file_resolved
```

**Steps:**
1. Read `aura/agent.py` lines 1462–1470 (system prompt) and 1103–1115 (custom tool loading)
2. Apply Part A: replace the stale "locally using Ollama with the Llama and Qwen models" line
3. Apply Part B: add path resolution + boundary check before the `tool_file.exists()` check
4. Verify: `python -c "from aura.agent import ApprenticeAgent; print('OK')"`

---

## Task 6: config.py — Cloud shortcut in validate_model + CORS default

**File:** `aura/config.py`

### Part A — Cloud shortcut in `validate_model` (lines 58–79) — fixes ~100s startup delay

**Current code always makes HTTP request even for cloud models:**
```python
def validate_model(model_name: str, ollama_host: str = None) -> bool:
    import requests
    host = ollama_host or os.getenv("OLLAMA_HOST", "http://localhost:11434")
    try:
        response = requests.get(f"{host}/api/tags", timeout=5)
        ...
```

**Fix:** Add early return for cloud models — if the name ends in `-cloud` and `OLLAMA_API_KEY` is set, the model is available:
```python
def validate_model(model_name: str, ollama_host: str = None) -> bool:
    # Cloud models are available if OLLAMA_API_KEY is set — no local check needed
    if model_name.endswith("-cloud"):
        return bool(os.getenv("OLLAMA_API_KEY"))

    import requests
    host = ollama_host or os.getenv("OLLAMA_HOST", "http://localhost:11434")
    try:
        ...  # rest unchanged
```

### Part B — CORS default (line 340)

**Current:**
```python
API_CORS_ORIGINS: str = os.getenv("AURA_CORS_ORIGINS", "*")
```

**Fix:**
```python
API_CORS_ORIGINS: str = os.getenv("AURA_CORS_ORIGINS", "http://localhost:3000,http://localhost:5173")
```

**Steps:**
1. Read `aura/config.py` lines 58–80 and line 340
2. Add the early-return cloud check at the top of `validate_model`
3. Update the CORS default
4. Verify: `python -c "from aura.config import Config, validate_model; print(validate_model('gemini-3-flash-preview:cloud'))"`
   - If `OLLAMA_API_KEY` is set in env: should print `True`
   - If not set: should print `False` (and startup will use fallbacks — that's correct)

---

## Task 7: web_search.py + fast_path.py + clipboard + requirements.txt — Scattered quick fixes

### Part A — RateLimiter thread safety (`aura/tools/web_search.py:38-51`)

**Fix:** Add a `threading.Lock()` to `RateLimiter` and acquire it in `wait_if_needed`:
```python
class RateLimiter:
    def __init__(self, calls_per_minute: int = 30):
        self.calls_per_minute = calls_per_minute
        self.min_interval = 60.0 / calls_per_minute
        self.last_call_time = 0.0
        self._lock = threading.Lock()   # ADD THIS

    def wait_if_needed(self):
        with self._lock:                # ADD THIS
            now = time.time()
            elapsed = now - self.last_call_time
            if elapsed < self.min_interval:
                sleep_time = self.min_interval - elapsed
                logger.debug(f"[RATE_LIMIT] Sleeping {sleep_time:.2f}s")
                time.sleep(sleep_time)
            self.last_call_time = time.time()
```

Check if `threading` is already imported in web_search.py; add it if not.

### Part B — fast_path.py known_commands cleanup (`aura/fast_path.py:62-66`)

Remove conversational phrases from `known_commands`:
```python
# BEFORE
known_commands = {
    "status", "state", "mood", "feeling", "how are you",
    "memory", "memories", "remember", "what do you remember",
    "help", "commands", "?"
}

# AFTER
known_commands = {
    "status", "state",
    "memory", "memories", "remember", "what do you remember",
    "help", "commands", "?"
}
```
(`"mood"`, `"feeling"`, `"how are you"` removed — these are conversational and should go to the LLM)

### Part C — clipboard shell=True (`aura/tools/clipboard_history.py:96`)

```python
# BEFORE
process = subprocess.Popen(["clip"], stdin=subprocess.PIPE, shell=True)

# AFTER
process = subprocess.Popen(["clip"], stdin=subprocess.PIPE, shell=False)
```

### Part D — cryptography version cap (`requirements.txt:52`)

```
# BEFORE
cryptography>=41.0.0,<43.0.0

# AFTER
cryptography>=41.0.0
```

**Steps:**
1. Read each file at the affected lines to confirm exact current state
2. Apply Part A (add `_lock` to RateLimiter, wrap `wait_if_needed` — check threading import)
3. Apply Part B (remove mood/feeling/how-are-you from known_commands)
4. Apply Part C (shell=False)
5. Apply Part D (remove cryptography upper cap)
6. Verify each module imports: `python -c "from aura.tools.web_search import search; from aura.fast_path import FastPath; print('OK')"`

---

## Task 8: shell_executor.py — Sessions lock + setx fix + cd path injection

**File:** `aura/tools/shell_executor.py`

### Part A — Add `_sessions_lock` to `ShellExecutorTool.__init__` (Bug B19)

```python
def __init__(self):
    self._sessions: Dict[str, ShellSession] = {}
    self._is_windows = sys.platform == "win32"
    self._sessions_lock = threading.Lock()   # ADD
```

Then wrap every `self._sessions` access in the following methods with `with self._sessions_lock:`:
- `_get_session()` — wrap the read/write of `self._sessions`
- `_cleanup_sessions()` — wrap the iteration/deletion
- `close_session()` — wrap the `del self._sessions[...]`
- `list_sessions()` — wrap the read

Check if `threading` is imported at the top of shell_executor.py; add if missing.

### Part B — Remove `"set"` from ALLOWED_COMMANDS_PREFIX (Bug B8)

`"set"` allows `setx` through because the prefix check uses `startswith`. Remove it:

```python
# In ALLOWED_COMMANDS_PREFIX list (around line 68), remove:
"set", "export",
# Keep "export" only if it's needed for Linux — but "set" must go
```

Actually keep `"export"` (Linux env var syntax), remove only `"set"`:
```python
# BEFORE (line 68 area)
"set", "export", "which", ...
# AFTER
"export", "which", ...
```

### Part C — Run injection check before cd handler (Bug B7)

The `cd` handler at lines 220–234 runs before `_validate_command` is called on the full command. Add validation before the cd logic executes:

Read `run()` method to find where `_validate_command` is called vs where the cd match happens. The fix: call `_validate_command(command)` at the top of `_execute_in_session()` (or wherever `run()` delegates), before the `cd_match` block.

If `_validate_command` is already called before this code path, add an additional path-boundary check in the cd handler itself:
```python
if cd_match:
    target = cd_match.group(1).strip().strip('"').strip("'")
    target_path = (Path(working_dir) / target).resolve()
    # SECURITY: Verify target is not trying to escape to system directories
    target_str = str(target_path)
    blocked_roots = ["C:\\Windows", "C:\\System32", "/etc", "/sys", "/proc"]
    if any(target_str.startswith(root) for root in blocked_roots):
        return {"success": False, "error": f"Access denied: {target_path}", "exit_code": 1, "session_id": session.id}
    if target_path.exists() and target_path.is_dir():
        session.cwd = str(target_path)
        ...
```

**Steps:**
1. Read `aura/tools/shell_executor.py` lines 1–10 (imports), 60–72 (ALLOWED_COMMANDS_PREFIX), 96–105 (`__init__`), and all `_sessions` access methods
2. Check if `threading` is imported; add if not
3. Apply Part A: add `_sessions_lock` and wrap all `_sessions` accesses
4. Apply Part B: remove `"set"` from ALLOWED_COMMANDS_PREFIX
5. Apply Part C: add path boundary check in the cd handler
6. Verify: `python -c "from aura.tools.shell_executor import ShellExecutorTool; print('OK')"`

---

## Final Verification

After all 8 tasks:

```bash
cd D:/Aura
python -c "
from aura.brain import OllamaBrain
from aura.agent import ApprenticeAgent
from aura.config import Config, validate_model
from aura.tools.voice_manager import VoiceManager
from aura.tools.web_search import search
from aura.fast_path import FastPath
from aura.tools.shell_executor import ShellExecutorTool
from aura.tools.clipboard_history import ClipboardHistory
print('All modules import cleanly')
print('Cloud model validates fast:', validate_model('gemini-3-flash-preview:cloud'))
"
```
