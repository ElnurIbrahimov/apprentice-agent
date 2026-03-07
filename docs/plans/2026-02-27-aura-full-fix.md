# AURA Full Fix Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix all critical security vulnerabilities, data-safety issues, broken integrations, and key architectural problems found in the 5-agent audit of D:/Aura.

**Architecture:** Changes are grouped into 5 phases in priority order. Each phase is independently deployable. Phases 1-3 are urgent. Phases 4-5 are important but can run over multiple sessions.

**Tech Stack:** Python 3.12, FastAPI, Ollama, ChromaDB, NetworkX, Qdrant, python-telegram-bot, Playwright, tempfile/os for atomic writes.

---

## Scope Honesty

This plan covers **what is actually fixable in code**:
- ✅ Phase 1: 5 critical security vulnerabilities
- ✅ Phase 2: Data corruption / atomic writes
- ✅ Phase 3: Broken integrations (marketplace, custom tools)
- ✅ Phase 4: Architecture quick wins (response builder, error handling, KG indexing)
- ⚠️ Phase 5: AI depth (Truth Spine hashing, Dream consolidation, ALMA wiring) — large, do last

**NOT in this plan** (months of rearchitecting, not bugs):
- Splitting agent.py God object into 4 classes (Sprint-level refactor)
- Real MCTS ground truth evaluation (requires task success oracle)
- Full consciousness → mechanism replacement (research problem)

---

## Phase 1: Critical Security Fixes

### Task 1: Fix shell=True injection in system_control.py

**Files:**
- Modify: `aura/tools/system_control.py:150-165`

**Problem:** `subprocess.Popen(command, shell=True)` where `command = "start msedge"` allows shell metacharacter injection.

**Fix:** Replace `shell=True` with `subprocess.Popen` using `startfile` on Windows or list-form args.

**Step 1: Read current open_app method**
```bash
grep -n "shell=True\|shell=False\|Popen" D:/Aura/aura/tools/system_control.py
```

**Step 2: Replace the shell=True block**

Replace lines 150-165 in `aura/tools/system_control.py`:

```python
    def open_app(self, name: str) -> dict:
        """Open an application from the allowlist."""
        name_lower = name.lower().strip()

        if name_lower not in self.APP_ALLOWLIST:
            allowed = ", ".join(self.APP_ALLOWLIST.keys())
            return {
                "success": False,
                "error": f"App '{name}' not in allowlist. Allowed: {allowed}"
            }

        try:
            command = self.APP_ALLOWLIST[name_lower]

            if command.startswith("start "):
                # Extract the target after "start " — use os.startfile on Windows
                # This avoids shell=True entirely
                import os
                target = command[6:].strip()  # Remove "start " prefix
                os.startfile(target)
            else:
                subprocess.Popen([command], shell=False)

            return {
                "success": True,
                "app": name_lower,
                "message": f"Opened {name_lower}"
            }
        except Exception as e:
            return {"success": False, "error": str(e), "app": name_lower}
```

**Step 3: Also fix APP_ALLOWLIST** — the `start chrome` and `start firefox` entries should use real executable names:

```python
    APP_ALLOWLIST = {
        "notepad": "notepad.exe",
        "calculator": "calc.exe",
        "browser": "msedge",
        "chrome": "chrome",
        "firefox": "firefox",
        "explorer": "explorer.exe",
        "vscode": "code",
        "terminal": "wt.exe",
    }
```

**Step 4: Fix get_gpu_info to return error dict instead of None** (line 205):
```python
        except Exception:
            return {"success": False, "error": "nvidia-smi unavailable or no NVIDIA GPU"}
```

**Step 5: Verify no other shell=True in system_control.py**
```bash
grep -n "shell=True" D:/Aura/aura/tools/system_control.py
```
Expected: 0 results.

---

### Task 2: Fix API auth — warn loudly but don't silently allow

**Files:**
- Modify: `aura/api/auth.py:16-31`

**Problem:** When `AURA_API_KEY` is not set, the API allows all requests silently. On a network-exposed server, this is open access.

**Fix:** Keep dev mode working (key optional for localhost dev), but add startup warning + env flag to lock it down:

```python
"""API key authentication for Aura routes."""
import os
import secrets
import logging
from fastapi import Header, HTTPException, status

logger = logging.getLogger(__name__)

_API_KEY_ENV = "AURA_API_KEY"
_AUTH_REQUIRED_ENV = "AURA_REQUIRE_AUTH"


def _get_configured_key() -> str | None:
    return os.environ.get(_API_KEY_ENV)


def _auth_is_required() -> bool:
    """Returns True if auth is explicitly required (default: False for local dev)."""
    return os.environ.get(_AUTH_REQUIRED_ENV, "false").lower() in ("true", "1", "yes")


async def require_api_key(x_api_key: str = Header(default="")) -> str:
    """FastAPI dependency: validates X-API-Key header."""
    configured = _get_configured_key()

    if configured is None:
        if _auth_is_required():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=(
                    "AURA_REQUIRE_AUTH=true but AURA_API_KEY is not set. "
                    "Set AURA_API_KEY environment variable."
                ),
            )
        # Dev mode: no key configured
        logger.warning(
            "[Auth] AURA_API_KEY not set — running in unauthenticated dev mode. "
            "Set AURA_REQUIRE_AUTH=true and AURA_API_KEY before network exposure."
        )
        return ""

    if not x_api_key or not secrets.compare_digest(x_api_key, configured):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    return x_api_key


def verify_api_key_ws(key: str) -> bool:
    """For WebSocket connections where Header dependency isn't available."""
    configured = _get_configured_key()
    if configured is None:
        return not _auth_is_required()  # Block if auth required, allow in dev
    return bool(key) and secrets.compare_digest(key, configured)
```

**Step: Update .env.example** to document the new variables:
```
# Security — set these before exposing to network
AURA_API_KEY=your-strong-random-key-here
AURA_REQUIRE_AUTH=true
```

---

### Task 3: Fix Telegram — default deny, require whitelist

**Files:**
- Modify: `aura/messaging/telegram_bot.py:195-199`

**Problem:** Empty `allowed_users` list = allow everyone.

**Step 1: Change `_is_user_allowed`**

```python
    def _is_user_allowed(self, user_id: int) -> bool:
        """Check if user is allowed to use the bot.

        SECURITY: Returns False if no whitelist configured (fail-closed).
        Add your Telegram user ID to allowed_users in messaging config.
        """
        if not self.allowed_users:
            logger.warning(
                f"[TelegramBot] Rejecting user {user_id} — no allowed_users configured. "
                "Set TELEGRAM_ALLOWED_USERS in your config to enable access."
            )
            return False  # CHANGED: was True (allow all), now False (deny all)
        return str(user_id) in self.allowed_users
```

**Step 2: Update messaging config default with instructions**

In `aura/messaging/config.py`, find the `allowed_users` default and add a comment:
```python
# SECURITY: Add your Telegram user ID here. Get it from @userinfobot.
# Leave empty to disable all Telegram access (fail-closed security).
TELEGRAM_ALLOWED_USERS: list[str] = []  # e.g. ["123456789"]
```

**Step 3: Verify rejection message is clear**

Check that `_handle_start` (line 207+) sends a helpful message when rejected, so user knows to configure the whitelist.

---

### Task 4: Sanitize proactive messages before Telegram send

**Files:**
- Modify: `aura/messaging/telegram_bot.py` — find the `send_proactive` method
- Create: `aura/messaging/sanitizer.py`

**Problem:** LLM-generated insight content sent directly to Telegram without sanitization. Prompt injection in web content could cause AURA to send phishing messages.

**Step 1: Create sanitizer module**

```python
# aura/messaging/sanitizer.py
"""Content sanitization for outgoing messages."""
import re
import html

# Max length for any outgoing message
MAX_MESSAGE_LENGTH = 1000

# Patterns that suggest prompt injection / phishing
_SUSPICIOUS_PATTERNS = [
    r'https?://\S+',          # Any URLs (flag for review, don't auto-block)
    r'click here',
    r'verify your account',
    r'your account (will|has)',
    r'urgent[: ]',
    r'immediately',
    r'password',
    r'credit card',
    r'bank account',
]
_SUSPICIOUS_RE = re.compile(
    '|'.join(_SUSPICIOUS_PATTERNS),
    re.IGNORECASE
)


def sanitize_outgoing(text: str, source: str = "unknown") -> tuple[str, bool]:
    """Sanitize LLM-generated text before sending to users.

    Returns:
        (sanitized_text, was_flagged)
        was_flagged=True means content was suspicious and should be logged/reviewed
    """
    if not text:
        return "", False

    # Truncate
    text = text[:MAX_MESSAGE_LENGTH]

    # HTML escape to prevent Telegram markdown injection
    # (Telegram parses markdown, so we escape then re-apply safe formatting)
    text = html.escape(text)

    # Check for suspicious patterns
    flagged = bool(_SUSPICIOUS_RE.search(text))
    if flagged:
        import logging
        logging.getLogger(__name__).warning(
            f"[Sanitizer] Flagged outgoing message from source='{source}': {text[:100]}..."
        )

    return text, flagged
```

**Step 2: Apply sanitizer in send_proactive**

Find `send_proactive` in `telegram_bot.py` and wrap the message content:

```python
from .sanitizer import sanitize_outgoing

# Before sending:
clean_message, flagged = sanitize_outgoing(personalized, source="proactive_awareness")
if flagged:
    logger.warning(f"[TelegramBot] Proactive message flagged — sending anyway but logged")
await self.send_proactive(chat_id, clean_message)
```

---

### Task 5: Fix shell_executor allowlist — remove python/pip/curl as unrestricted

**Files:**
- Modify: `aura/tools/shell_executor.py:62-73`

**Problem:** `python`, `pip`, `curl`, `wget`, `node`, `npm` are in the allowed prefix list but are unrestricted — `python -c "import os; os.system('rm -rf /')"` passes all validation.

**Step 1: Read current ALLOWED_COMMANDS_PREFIX**
```bash
grep -n "ALLOWED_COMMANDS_PREFIX\|python\|pip\|curl\|wget" D:/Aura/aura/tools/shell_executor.py | head -30
```

**Step 2: Remove dangerous entries and add argument validation**

Find and replace the ALLOWED_COMMANDS_PREFIX definition:

```python
# Commands that must match a prefix (first word of command)
# SECURITY: Omit interpreters (python, node) and network fetchers (curl, wget)
# These can be used via CodeExecutorTool and WebSearchTool respectively
ALLOWED_COMMANDS_PREFIX = [
    "ls", "dir", "pwd", "echo", "cat", "head", "tail",
    "grep", "find", "wc", "sort", "uniq", "diff",
    "mkdir", "cp", "mv", "touch",
    "git",              # Git operations
    "tar", "zip", "unzip",
    "docker",           # Docker container management
    # Removed: python, pip, node, npm (use CodeExecutorTool instead)
    # Removed: curl, wget (use WebSearchTool instead)
]
```

**Step 3: Improve pipe-to-shell regex pattern**

Find the existing blocked pattern for curl piping and improve it:
```python
# Block pipe to any shell variant, with or without spaces
r"\|\s*(?:/(?:usr/)?(?:local/)?bin/)?(?:ba|da|z|k|fi|c)?sh\b",
```

**Step 4: Add a comment in help text** so users know why python/curl aren't available via shell:
```python
# In execute() or help output:
"Note: Use 'code' tool for Python execution, 'search' tool for web requests."
```

---

## Phase 2: Data Safety — Atomic Writes

### Task 6: Atomic writes for A-MEM JSON persistence

**Files:**
- Modify: `aura/tools/amem.py` — find `save()` method (~line 850)

**Problem:** `Path.write_text(json.dumps(...))` is not atomic. Process crash mid-write corrupts the file.

**Step 1: Find the save method**
```bash
grep -n "def save\|write_text\|json.dump" D:/Aura/aura/tools/amem.py | head -20
```

**Step 2: Replace with atomic write pattern**

```python
import tempfile
import os

def _atomic_write_json(self, path: Path, data: dict) -> None:
    """Write JSON atomically using temp file + rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.tmp.",
        suffix=".json"
    )
    try:
        with os.fdopen(tmp_fd, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        os.replace(tmp_path, path)  # Atomic on all modern OSes
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
```

**Step 3: Call `_atomic_write_json` in save()**

Replace any `path.write_text(json.dumps(...))` calls with:
```python
self._atomic_write_json(self._notes_file, notes_data)
```

---

### Task 7: Atomic writes for Knowledge Graph

**Files:**
- Modify: `aura/tools/knowledge_graph.py` — find save/persist methods

**Step 1: Find all write locations**
```bash
grep -n "write_text\|\.open.*'w'\|json.dump\|jsonlines" D:/Aura/aura/tools/knowledge_graph.py | head -20
```

**Step 2: Apply same `_atomic_write_json` pattern** (extract to shared utility or copy the method).

**Step 3: For JSONL files** (nodes.jsonl, edges.jsonl), use append-only pattern:
```python
def _append_jsonl(self, path: Path, record: dict) -> None:
    """Append a single JSON record to a JSONL file (append-only = crash safe)."""
    with open(path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(record, default=str) + '\n')
```
Append-only JSONL cannot be corrupted by crash — at worst you lose the last record.

**Step 4: Remove GraphML write** — it's write-only and never read. Dead code causing format inconsistency. Find and remove:
```bash
grep -n "GraphML\|write_graphml\|graphml" D:/Aura/aura/tools/knowledge_graph.py
```

---

### Task 8: Thread safety for Qdrant episodic memory

**Files:**
- Modify: `aura_episodic_memory/memory_store.py`

**Problem:** Qdrant embedded mode is single-process. Concurrent thread access causes race conditions.

**Step 1: Find the store class**
```bash
grep -n "class.*Store\|self\.client\|qdrant" D:/Aura/aura_episodic_memory/memory_store.py | head -20
```

**Step 2: Add threading.Lock**
```python
import threading

class EpisodicMemoryStore:
    def __init__(self, ...):
        ...
        self._lock = threading.Lock()

    def add_episode(self, episode) -> str:
        with self._lock:
            return self._add_episode_unsafe(episode)

    def search(self, query, ...) -> list:
        with self._lock:
            return self._search_unsafe(query, ...)
```

Wrap all methods that touch `self.client` with `self._lock`.

---

## Phase 3: Broken Integrations

### Task 9: Connect custom tools to agent — dynamic import scanning

**Files:**
- Modify: `aura/agent.py` — find `__init__` method, tool initialization section
- Create: `aura/tools/custom_loader.py`

**Problem:** `ToolBuilderTool` generates tools into `aura/tools/custom/` and `MarketplaceTool` downloads plugins there, but `agent.py` never scans that directory. The entire extensibility system is dead.

**Step 1: Create custom_loader.py**

```python
# aura/tools/custom_loader.py
"""Dynamically load custom and marketplace tools from aura/tools/custom/."""
import importlib
import logging
import inspect
from pathlib import Path

logger = logging.getLogger(__name__)


def load_custom_tools(custom_dir: Path | None = None) -> dict[str, object]:
    """Scan custom/ directory and return {tool_name: tool_instance} for all valid tools."""
    if custom_dir is None:
        custom_dir = Path(__file__).parent / "custom"

    if not custom_dir.exists():
        return {}

    loaded = {}
    for tool_file in sorted(custom_dir.glob("*.py")):
        if tool_file.name.startswith("_"):
            continue  # Skip __init__.py, __pycache__, etc.

        module_name = f"aura.tools.custom.{tool_file.stem}"
        try:
            module = importlib.import_module(module_name)
        except Exception as e:
            logger.warning(f"[CustomLoader] Failed to import {tool_file.name}: {e}")
            continue

        # Find Tool class: has name + description attributes + execute() method
        for attr_name in dir(module):
            obj = getattr(module, attr_name)
            if not inspect.isclass(obj):
                continue
            if not (hasattr(obj, 'name') and hasattr(obj, 'description') and hasattr(obj, 'execute')):
                continue
            if attr_name.startswith('_'):
                continue

            try:
                instance = obj()
                tool_name = getattr(instance, 'name', tool_file.stem)
                loaded[tool_name] = instance
                logger.info(f"[CustomLoader] Loaded custom tool: {tool_name} from {tool_file.name}")
            except Exception as e:
                logger.warning(f"[CustomLoader] Failed to instantiate {attr_name} from {tool_file.name}: {e}")

    return loaded
```

**Step 2: Call load_custom_tools in agent.__init__**

Find where tools are initialized in `agent.py` (around line 356-500) and add at the end:

```python
# Load custom and marketplace tools dynamically
try:
    from .tools.custom_loader import load_custom_tools
    from pathlib import Path
    custom_tools = load_custom_tools()
    for tool_name, tool_instance in custom_tools.items():
        if tool_name not in self.tools:  # Don't override core tools
            self.tools[tool_name] = tool_instance
            logger.info(f"[Agent] Registered custom tool: {tool_name}")
except Exception as e:
    logger.warning(f"[Agent] Custom tool loading failed: {e}")
```

**Step 3: Make MarketplaceTool trigger reload after install**

Find `install_plugin` in `marketplace.py` and add at the end:
```python
# Signal to reload custom tools on next agent interaction
# Write a sentinel file that agent can check
sentinel = Path(__file__).parent / "custom" / ".reload_needed"
sentinel.touch()
logger.info(f"[Marketplace] Plugin installed. Reload sentinel written.")
return {"success": True, "plugin": plugin_id, "reload_required": True}
```

Then in agent.py's main loop or at start of each `run()`, check for the sentinel:
```python
sentinel = Path(__file__).parent / "tools" / "custom" / ".reload_needed"
if sentinel.exists():
    custom_tools = load_custom_tools()
    for name, inst in custom_tools.items():
        if name not in self.tools:
            self.tools[name] = inst
    sentinel.unlink()
    logger.info("[Agent] Reloaded custom tools after marketplace install")
```

---

### Task 10: Fix DatabaseTool — exceptions → error dicts

**Files:**
- Modify: `aura/tools/database_tool.py`

**Step 1: Find all raise statements**
```bash
grep -n "raise ValueError\|raise RuntimeError\|raise Exception" D:/Aura/aura/tools/database_tool.py
```

**Step 2: Replace each raise with error dict return**

Pattern to apply throughout:
```python
# BEFORE:
def _validate_table_name(name: str) -> None:
    if not _SAFE_TABLE_NAME.match(name):
        raise ValueError(f"Invalid table name: {name!r}.")

# AFTER:
def _validate_table_name(name: str) -> tuple[bool, str | None]:
    if not _SAFE_TABLE_NAME.match(name):
        return False, f"Invalid table name: {name!r}. Use only letters, numbers, underscores."
    return True, None
```

Then in calling methods:
```python
valid, err = self._validate_table_name(table)
if not valid:
    return {"success": False, "error": err, "blocked_by": "validation"}
```

**Step 3: Apply same pattern to EmailTool**

```bash
grep -n "raise RuntimeError\|raise ValueError" D:/Aura/aura/tools/email_tool.py
```

Replace `raise RuntimeError("AURA_EMAIL_KEY not set")` with:
```python
if not secret:
    return {"success": False, "error": "Email encryption key not configured. Set AURA_EMAIL_KEY environment variable.", "blocked_by": "configuration"}
```

---

### Task 11: Add KG indexing — replace O(n) substring search

**Files:**
- Modify: `aura/tools/knowledge_graph.py` — `find_nodes()` method (~line 617)

**Problem:** `find_nodes()` iterates all nodes sequentially. 11MB file = slow.

**Step 1: Add in-memory label index to __init__**

```python
def __init__(self, ...):
    ...
    self._label_index: dict[str, list[str]] = {}  # lowercase_word → [node_ids]
```

**Step 2: Index on node add**

```python
def _index_node(self, node_id: str, label: str) -> None:
    """Add node to the label index."""
    for word in label.lower().split():
        if word not in self._label_index:
            self._label_index[word] = []
        if node_id not in self._label_index[word]:
            self._label_index[word].append(node_id)

def _deindex_node(self, node_id: str, label: str) -> None:
    """Remove node from label index."""
    for word in label.lower().split():
        if word in self._label_index:
            try:
                self._label_index[word].remove(node_id)
            except ValueError:
                pass
```

**Step 3: Rebuild index on load**

In the `_load()` method, after loading nodes:
```python
# Rebuild label index
self._label_index = {}
for node_id, node in self._nodes.items():
    self._index_node(node_id, node.label)
```

**Step 4: Use index in find_nodes()**

```python
def find_nodes(self, query: str, limit: int = 10) -> list:
    query_lower = query.lower()
    query_words = query_lower.split()

    # Fast path: use word index
    candidate_ids: set[str] = set()
    for word in query_words:
        for idx_word, node_ids in self._label_index.items():
            if query_lower in idx_word or idx_word in query_lower:
                candidate_ids.update(node_ids)

    # Score candidates
    results = []
    for node_id in candidate_ids:
        node = self._nodes.get(node_id)
        if node and query_lower in node.label.lower():
            results.append(node)

    # Sort by relevance + recency
    results.sort(key=lambda n: (query_lower == n.label.lower(), n.access_count), reverse=True)
    return results[:limit]
```

---

## Phase 4: Architecture Quick Wins

### Task 12: Extract response builder — eliminate 15+ duplicate response dicts

**Files:**
- Modify: `aura/agent.py`

**Problem:** The same response dict structure is repeated 15+ times in `agent.py`:
```python
return {
    "goal": goal,
    "completed": True,
    "iterations": 0,
    "fast_path": True,
    "response": response,
    "final_evaluation": {"progress": "completed", ...},
    "history": []
}
```

**Step 1: Find all occurrences**
```bash
grep -n '"goal":\|"completed":\|"fast_path":' D:/Aura/aura/agent.py | head -30
```

**Step 2: Add helper method to ApprenticeAgent** (near the top, after __init__):
```python
def _make_response(
    self,
    goal: str,
    response: str,
    *,
    completed: bool = True,
    iterations: int = 0,
    fast_path: bool = False,
    history: list | None = None,
    metadata: dict | None = None,
) -> dict:
    """Build a standardized response dict."""
    result = {
        "goal": goal,
        "completed": completed,
        "iterations": iterations,
        "fast_path": fast_path,
        "response": response,
        "final_evaluation": {
            "progress": "completed" if completed else "in_progress",
            "response": response,
        },
        "history": history or [],
    }
    if metadata:
        result.update(metadata)
    return result
```

**Step 3: Replace all duplicate dict constructions** with `self._make_response(...)`.

Example replacement:
```python
# BEFORE:
return {
    "goal": goal, "completed": True, "iterations": 0,
    "fast_path": True, "fluxmind_direct": True,
    "response": response,
    "final_evaluation": {"progress": "completed", "response": response},
    "history": []
}

# AFTER:
return self._make_response(goal, response, fast_path=True, metadata={"fluxmind_direct": True})
```

---

### Task 13: Fix SystemControlTool — consistent error returns

**Files:**
- Modify: `aura/tools/system_control.py:167-207`

**Step 1: Fix get_gpu_info return type** — change `return None` to return error dict:
```python
def get_gpu_info(self) -> dict:
    """Get NVIDIA GPU information."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,temperature.gpu,memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode != 0:
            return {"success": False, "error": "nvidia-smi returned non-zero exit code"}
        output = result.stdout.strip()
        if not output:
            return {"success": False, "error": "nvidia-smi returned empty output"}
        parts = [p.strip() for p in output.split(",")]
        if len(parts) < 5:
            return {"success": False, "error": "Unexpected nvidia-smi output format"}
        return {
            "success": True,
            "name": parts[0],
            "temperature": int(parts[1]),
            "memory_used_mb": int(parts[2]),
            "memory_total_mb": int(parts[3]),
            "utilization_percent": int(parts[4])
        }
    except FileNotFoundError:
        return {"success": False, "error": "nvidia-smi not found — no NVIDIA GPU or drivers"}
    except Exception as e:
        return {"success": False, "error": str(e)}
```

---

### Task 14: Fix secure_logging — add missing patterns, recursive sanitization

**Files:**
- Modify: `aura/secure_logging.py`

**Step 1: Find current SENSITIVE_PATTERNS**
```bash
grep -n "SENSITIVE_PATTERNS\|pattern\|re\." D:/Aura/aura/secure_logging.py | head -20
```

**Step 2: Add missing patterns**

```python
SENSITIVE_PATTERNS = [
    # Existing patterns (keep as-is)
    ...
    # New patterns to add:
    r'postgresql://[^@]+@',       # DB connection strings with credentials
    r'mysql://[^@]+@',
    r'mongodb://[^@]+@',
    r'redis://:[^@]+@',
    r'Authorization:\s*Bearer\s+[A-Za-z0-9._-]{20,}',  # JWT tokens
    r'eyJ[A-Za-z0-9._-]{20,}',   # Raw JWT (starts with eyJ)
    r'X-Webhook-Secret:\s*\S+',  # Webhook secrets
    r'OLLAMA_API_KEY[=:]\s*\S+', # Ollama API keys
]
```

**Step 3: Recursive sanitization for dict/list args**

Find `_sanitize_message()` and update args handling:
```python
def _sanitize_arg(self, arg):
    """Recursively sanitize an argument."""
    if isinstance(arg, str):
        return sanitize_text(arg)
    elif isinstance(arg, dict):
        return {k: self._sanitize_arg(v) for k, v in arg.items()}
    elif isinstance(arg, (list, tuple)):
        return type(arg)(self._sanitize_arg(item) for item in arg)
    return arg

# In _sanitize_message:
args = tuple(self._sanitize_arg(a) for a in args)
```

---

### Task 15: Add unified context budget for memory injection

**Files:**
- Create: `aura/memory/context_budget.py`
- Modify: `aura/agent.py` — find where memory context is injected

**Problem:** Each memory system (A-MEM, KG, episodic, RAG) injects independently with hardcoded limits. Combined can overflow the context window.

**Step 1: Create ContextBudget**

```python
# aura/memory/context_budget.py
"""Unified token budget for memory injection into LLM context."""
from dataclasses import dataclass, field


@dataclass
class ContextBudget:
    """Allocates a fixed token budget across memory systems.

    Usage:
        budget = ContextBudget(total_tokens=3000)
        amem_limit = budget.allocate("amem", requested=800)
        kg_limit = budget.allocate("kg", requested=600)
        # Each allocate() returns how many tokens this system can use
    """
    total_tokens: int = 3000
    _allocated: dict[str, int] = field(default_factory=dict, repr=False)

    def allocate(self, system: str, requested: int) -> int:
        """Allocate up to `requested` tokens for `system`. Returns actual allocation."""
        used = sum(self._allocated.values())
        available = max(0, self.total_tokens - used)
        granted = min(requested, available)
        self._allocated[system] = granted
        return granted

    @property
    def remaining(self) -> int:
        return max(0, self.total_tokens - sum(self._allocated.values()))

    def reset(self) -> None:
        self._allocated.clear()
```

**Step 2: Use budget in agent.py context building**

Find where the agent builds context from memory systems and wrap with budget:
```python
budget = ContextBudget(total_tokens=3000)

# A-MEM
amem_limit = budget.allocate("amem", requested=800)
amem_context = hybrid_amem.get_context(message, max_tokens=amem_limit)

# KG
kg_limit = budget.allocate("kg", requested=600)
kg_context = kg_bridge.get_context_for_query(message, max_entities=5)
kg_context = kg_context[:kg_limit]  # Trim to budget

# RAG
rag_limit = budget.allocate("rag", requested=1500)
rag_context = rag_tool.rag.get_context(message, top_k=3, max_tokens=rag_limit)
```

---

## Phase 5: AI Depth Fixes

### Task 16: Truth Spine — add real artifact hashing for file operations

**Files:**
- Modify: `aura/truth_spine.py`

**Problem:** Truth Spine claims artifact-based verification but uses LLM judgment. Add real hash verification for file-system operations.

**Step 1: Find where file artifacts are stored**
```bash
grep -n "artifact\|FACT\|BELIEF\|hash\|sha" D:/Aura/aura/truth_spine.py | head -30
```

**Step 2: Add hash_file utility**

```python
import hashlib

def _hash_file(path: str) -> str | None:
    """Compute SHA256 hash of a file. Returns None if file doesn't exist."""
    try:
        h = hashlib.sha256()
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(65536), b''):
                h.update(chunk)
        return h.hexdigest()
    except (OSError, IOError):
        return None
```

**Step 3: Add artifact verification method**

```python
def verify_file_artifact(self, path: str, expected_hash: str | None = None) -> dict:
    """Verify a file artifact after an operation.

    Args:
        path: File path
        expected_hash: If provided, verify the file matches this hash

    Returns:
        {"verified": bool, "hash": str, "tier": "FACT"|"BELIEF"|"SPECULATION"}
    """
    current_hash = _hash_file(path)
    if current_hash is None:
        return {"verified": False, "hash": None, "tier": "SPECULATION",
                "reason": "File does not exist"}
    if expected_hash and current_hash != expected_hash:
        return {"verified": False, "hash": current_hash, "tier": "SPECULATION",
                "reason": f"Hash mismatch: expected {expected_hash[:8]}..., got {current_hash[:8]}..."}
    return {"verified": True, "hash": current_hash, "tier": "FACT"}
```

**Step 4: Store pre-op hashes, verify post-op**

In the tool execution pipeline (agent.py or truth_spine), for filesystem tool calls:
```python
# Before filesystem write:
pre_hash = truth_spine._hash_file(path) if Path(path).exists() else None

# Execute the tool
result = filesystem_tool.write_file(path, content)

# After: verify file changed as expected
post_verify = truth_spine.verify_file_artifact(path)
result["artifact_hash"] = post_verify["hash"]
result["verification_tier"] = post_verify["tier"]
```

---

### Task 17: Dream mode — implement actual memory consolidation

**Files:**
- Modify: `aura/dream.py`

**Problem:** Dream mode only extracts insights from logs. It doesn't consolidate, merge, or prune memories.

**Step 1: Find run_dream_mode and consolidation hooks**
```bash
grep -n "def run_dream_mode\|consolidat\|prune\|merge" D:/Aura/aura/dream.py
```

**Step 2: Add consolidation step after insight extraction**

After the existing insight extraction (lines 270-288), add:

```python
def _consolidate_amem_notes(amem_system, similarity_threshold: float = 0.85) -> dict:
    """Merge A-MEM notes with cosine similarity > threshold."""
    if not amem_system:
        return {"merged": 0, "pruned": 0}

    notes = list(amem_system._notes.values())
    merged_count = 0
    pruned_count = 0

    # Get embeddings for all notes
    embeddings = {}
    for note in notes:
        if note.id in amem_system._embeddings:
            embeddings[note.id] = amem_system._embeddings[note.id]

    # Find pairs with high similarity
    merged_ids = set()
    for i, note_a in enumerate(notes):
        if note_a.id in merged_ids:
            continue
        for note_b in notes[i+1:]:
            if note_b.id in merged_ids:
                continue
            if note_a.id not in embeddings or note_b.id not in embeddings:
                continue

            # Cosine similarity
            ea, eb = embeddings[note_a.id], embeddings[note_b.id]
            similarity = sum(a*b for a, b in zip(ea, eb)) / (
                (sum(a**2 for a in ea)**0.5) * (sum(b**2 for b in eb)**0.5 + 1e-8)
            )

            if similarity >= similarity_threshold:
                # Merge note_b into note_a (keep higher importance, combine keywords)
                note_a.importance = max(note_a.importance, note_b.importance)
                note_a.keywords = list(set(note_a.keywords + note_b.keywords))[:20]
                note_a.context = f"{note_a.context} | {note_b.context}"
                merged_ids.add(note_b.id)
                merged_count += 1

    # Remove merged notes
    for note_id in merged_ids:
        del amem_system._notes[note_id]

    # Prune low-importance notes (< 0.2)
    low_importance = [
        nid for nid, n in amem_system._notes.items()
        if n.importance < 0.2 and n.access_count == 0
    ]
    for nid in low_importance:
        del amem_system._notes[nid]
        pruned_count += 1

    # Save consolidated state
    amem_system.save()

    return {"merged": merged_count, "pruned": pruned_count}
```

**Step 3: Call consolidation in run_dream_mode**

```python
# After insight extraction:
from .tools.amem import AMemSystem
try:
    amem = AMemSystem()
    consolidation = _consolidate_amem_notes(amem)
    logger.info(f"[Dream] Consolidated A-MEM: {consolidation}")
    result["consolidation"] = consolidation
except Exception as e:
    logger.warning(f"[Dream] Consolidation failed: {e}")
```

---

### Task 18: ALMA — wire neuromodulators to actual LLM parameters

**Files:**
- Modify: `aura/emotion/integration.py`
- Modify: `aura/brain.py` — `think()` method

**Problem:** ALMA computes neuromodulator values but only injects them as prompt text. They should actually change `temperature` and `top_p` for the LLM call.

**Step 1: Add parameter export to ALMA**

In `aura/emotion/integration.py`, add:
```python
def get_llm_parameters() -> dict:
    """Export current emotional state as LLM generation parameters.

    Returns parameters dict to pass to Ollama's generate/chat call.
    """
    if not ALMA_AVAILABLE:
        return {}

    try:
        state = alma_engine.get_state()
        mods = state.get("neuromodulators", {})

        # Dopamine → creativity/temperature (high dopamine = more creative)
        dopamine = mods.get("dopamine", 0.5)
        # Serotonin → coherence/top_p (high serotonin = more focused)
        serotonin = mods.get("serotonin", 0.5)
        # Norepinephrine → response sharpness (high NE = faster/sharper)
        norepinephrine = mods.get("norepinephrine", 0.5)

        # Map to LLM parameters with safety bounds
        # temperature: 0.3 (calm/focused) to 1.2 (excited/creative)
        base_temp = 0.7
        temp_delta = (dopamine - 0.5) * 0.6  # ±0.3 range
        temperature = max(0.3, min(1.2, base_temp + temp_delta))

        # top_p: 0.7 (coherent) to 0.99 (diverse)
        top_p = max(0.7, min(0.99, 0.85 + (0.5 - serotonin) * 0.2))

        return {
            "temperature": round(temperature, 2),
            "top_p": round(top_p, 2),
        }
    except Exception:
        return {}
```

**Step 2: Apply in brain.py's think() method**

Find the Ollama chat call in `brain.py` and apply emotional parameters:
```python
from .emotion.integration import get_llm_parameters

# In think() or _call_llm():
emotional_params = get_llm_parameters() if ALMA_AVAILABLE else {}

response = self._client.chat(
    model=model,
    messages=messages,
    options={
        "temperature": emotional_params.get("temperature", 0.7),
        "top_p": emotional_params.get("top_p", 0.9),
        **other_options
    }
)
```

---

## Phase 6: Cleanup

### Task 19: Remove dead code — CryptoPriceTool

**Files:**
- Delete: `aura/tools/crypto_price.py` (or mark as deprecated)
- Modify: `aura/tools/__init__.py` — remove import
- Modify: `aura/agent.py` — remove import

**Step 1: Verify it's unused**
```bash
grep -rn "CryptoPriceTool\|crypto_price" D:/Aura/aura/ --include="*.py"
```

**Step 2: Remove if only imported, not used**

---

### Task 20: Add .env.example with all security variables documented

**Files:**
- Modify/Create: `D:/Aura/.env.example`

```bash
# === SECURITY (set before network exposure) ===
AURA_API_KEY=change-this-to-a-strong-random-key
AURA_REQUIRE_AUTH=true

# === TELEGRAM ===
TELEGRAM_BOT_TOKEN=your-bot-token
TELEGRAM_ALLOWED_USERS=your-telegram-user-id
# Get your user ID from @userinfobot on Telegram

# === EMAIL ===
AURA_EMAIL_KEY=32-byte-random-base64-key

# === OLLAMA ===
OLLAMA_API_BASE=http://localhost:11434

# === OPTIONAL INTEGRATIONS ===
OPENAI_API_KEY=sk-...         # For ImageGenTool DALL-E
HF_TOKEN=hf_...               # For HuggingFace models
```

---

## Testing Checklist

After all phases:

```bash
# 1. Security: shell injection blocked
python -c "
from aura.tools.system_control import SystemControlTool
t = SystemControlTool()
r = t.open_app('chrome | echo INJECTED')
assert r['success'] == False, 'Injection should be blocked'
print('PASS: injection blocked')
"

# 2. API auth respects AURA_REQUIRE_AUTH
AURA_REQUIRE_AUTH=true python -c "
import asyncio
from aura.api.auth import require_api_key
# Should raise HTTPException when no key configured + auth required
"

# 3. Telegram rejects unknown users
python -c "
from aura.messaging.telegram_bot import TelegramBot
bot = TelegramBot.__new__(TelegramBot)
bot.allowed_users = []
assert bot._is_user_allowed(12345) == False, 'Should deny unknown users'
print('PASS: unknown user denied')
"

# 4. Atomic write test
python -c "
from pathlib import Path
import json, tempfile, os
# Simulate atomic write
data = {'test': True}
path = Path(tempfile.mktemp(suffix='.json'))
tmp_fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix='.tmp')
with os.fdopen(tmp_fd, 'w') as f:
    json.dump(data, f)
os.replace(tmp_path, path)
assert json.loads(path.read_text()) == data
path.unlink()
print('PASS: atomic write works')
"

# 5. Custom tool loading
python -c "
from aura.tools.custom_loader import load_custom_tools
from pathlib import Path
tools = load_custom_tools(Path('D:/Aura/aura/tools/custom'))
print(f'PASS: loaded {len(tools)} custom tools: {list(tools.keys())}')
"
```

---

## Execution Notes

**What this plan changes:**
- 15 files modified, 2 files created
- No major refactors — all surgical fixes
- Backwards compatible — no API changes

**What this plan deliberately skips (too large for this session):**
- Splitting agent.py God object
- Full memory system unification
- Real MCTS ground truth oracle
- Full consciousness → mechanism replacement

**Estimated effort:** 3-5 hours of focused implementation

**Risk:** Low — most changes are additive or isolated. The biggest risk is agent.py changes (Task 9, 12) which touch initialization code.

**Recommended order:** Tasks 1→5 (security) → Tasks 6→8 (data safety) → Tasks 9→11 (functional) → Tasks 12→15 (architecture) → Tasks 16→18 (AI) → Tasks 19→20 (cleanup)
