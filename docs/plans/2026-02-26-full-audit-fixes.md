# Full Audit Fixes Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix all 74 findings from the 6-agent parallel audit of the Aura agent codebase — 26 criticals, 37 highs, 5 mediums, 6 coverage gaps.

**Architecture:** Fixes are organized into 8 batches by domain. Each batch is independent and can be executed sequentially. All fixes are surgical — no refactors beyond what's needed to fix the bug. Every fix is followed by a syntax check and targeted test where applicable.

**Tech Stack:** Python 3.10+, FastAPI, asyncio, threading, ChromaDB, numpy, sqlite3

---

## Batch A — Security: Injection & Code Execution (C-1 through C-7, C-25, C-26)

*9 criticals. Fixes injection vectors across shell, SQL, IMAP, and dynamic code execution.*

---

### Task A-1: Fix shell_executor.py command injection (C-1)

**Files:**
- Modify: `D:\apprentice-agent\aura\tools\shell_executor.py`

**Step 1: Read the file to find the execute/run method and allowlist**

Read `shell_executor.py`, focus on lines 100–240. Identify where `subprocess.run` or `subprocess.Popen` is called with `shell=True` and where the allowlist check happens.

**Step 2: Add metacharacter sanitization AFTER the allowlist check**

Find the method that builds and dispatches the shell command. After the allowlist check passes, add this block before the subprocess call:

```python
import re
SHELL_METACHAR_RE = re.compile(r'[&|;`$(){}[\]<>!\\]')

def _contains_shell_injection(cmd: str) -> bool:
    """Detect shell metacharacters and dangerous flags."""
    if SHELL_METACHAR_RE.search(cmd):
        return True
    # Block -c flag on interpreter-like commands
    tokens = cmd.split()
    if len(tokens) >= 2 and tokens[1] in ('-c', '/c', '-e', '-enc'):
        return True
    return False
```

Then in the execute method, after the allowlist check:
```python
if _contains_shell_injection(command):
    return {"success": False, "output": "", "error": "Command contains disallowed characters or flags", "exit_code": 1}
```

**Step 3: Remove cmd.exe and powershell.exe from ALLOWED_COMMANDS (same file, also fixes C-6 for shell_executor)**

Find `ALLOWED_COMMANDS` or `ALLOWED_COMMANDS_PREFIX` list. Remove or comment out entries for `cmd`, `cmd.exe`, `powershell`, `powershell.exe`.

**Step 4: Syntax check**
```bash
python -c "import ast; ast.parse(open('D:/apprentice-agent/aura/tools/shell_executor.py', encoding='utf-8').read()); print('OK')"
```

---

### Task A-2: Fix system_control.py cmd.exe/powershell in allowlist (C-6)

**Files:**
- Modify: `D:\apprentice-agent\aura\tools\system_control.py`

**Step 1: Read the file, find the app allowlist and open_app method (around lines 140–160)**

**Step 2: Remove cmd.exe and powershell.exe from the allowlist**

Find the dict/list that maps app names to executables. Remove or comment:
```python
# "cmd": "cmd.exe",       # REMOVED: privilege escalation vector
# "powershell": "powershell.exe",  # REMOVED: privilege escalation vector
```

**Step 3: Change shell=True to shell=False for non-start commands**

Find `subprocess.Popen(command, shell=True)` calls for hardcoded executables. Change to:
```python
subprocess.Popen([command], shell=False)
```

**Step 4: Syntax check**
```bash
python -c "import ast; ast.parse(open('D:/apprentice-agent/aura/tools/system_control.py', encoding='utf-8').read()); print('OK')"
```

---

### Task A-3: Fix database_tool.py SQL injection + path traversal (C-2, C-3)

**Files:**
- Modify: `D:\apprentice-agent\aura\tools\database_tool.py`

**Step 1: Read the file — focus on `_get_db_path()`, `query()`, `execute()`, `import_csv()` methods**

**Step 2: Fix path traversal in `_get_db_path()`**

Find `_get_db_path()`. Replace the current logic with a strict sandbox check:

```python
def _get_db_path(self, db: str) -> str:
    DB_DIR = Path(__file__).parent.parent.parent / "data" / "databases"
    DB_DIR.mkdir(parents=True, exist_ok=True)

    p = Path(db)
    # Only allow paths that resolve within DB_DIR
    try:
        resolved = (DB_DIR / p.name).resolve()  # use .name only to strip any directory components
        if not str(resolved).startswith(str(DB_DIR.resolve())):
            raise ValueError(f"Database path outside sandbox: {db}")
    except Exception:
        raise ValueError(f"Invalid database path: {db}")

    if resolved.suffix not in (".db", ".sqlite", ".sqlite3"):
        resolved = resolved.with_suffix(".db")
    return str(resolved)
```

**Step 3: Fix SQL injection in `query()` / `execute()`**

Find the `query()` or `execute()` method. Add a SQL statement allowlist check before execution:

```python
ALLOWED_SQL_VERBS = {"SELECT", "PRAGMA", "EXPLAIN"}
BLOCKED_SQL_VERBS = {"DROP", "DELETE", "UPDATE", "INSERT", "CREATE", "ALTER", "ATTACH", "DETACH", "REPLACE"}

def _check_sql_safety(self, sql: str) -> None:
    """Raise ValueError if SQL contains destructive statements."""
    first_word = sql.strip().split()[0].upper() if sql.strip() else ""
    if first_word in BLOCKED_SQL_VERBS:
        raise ValueError(f"SQL verb '{first_word}' is not allowed. Only SELECT and PRAGMA are permitted.")
```

Call `self._check_sql_safety(sql)` at the top of `query()` before executing.

**Step 4: Fix `import_csv()` path traversal**

Find `import_csv()`. Add sandbox validation for `csv_path`:

```python
DATA_DIR = Path(__file__).parent.parent.parent / "data"

def import_csv(self, csv_path: str, db: str = "default", table: str = "imported") -> dict:
    resolved = Path(csv_path).resolve()
    data_dir = DATA_DIR.resolve()
    if not str(resolved).startswith(str(data_dir)):
        return {"success": False, "error": f"CSV path must be within the data directory"}
    # rest of existing method...
```

**Step 5: Syntax check**
```bash
python -c "import ast; ast.parse(open('D:/apprentice-agent/aura/tools/database_tool.py', encoding='utf-8').read()); print('OK')"
```

---

### Task A-4: Fix tool_builder.py arbitrary code execution (C-4)

**Files:**
- Modify: `D:\apprentice-agent\aura\tools\tool_builder.py`

**Step 1: Read the file — focus on `_scan_for_dangerous_code()`, the code generation method, and `test_tool()` (around lines 198–221, 440–451)**

**Step 2: Replace regex scan with AST-based safety check**

Find `_scan_for_dangerous_code()`. Replace its body with an AST walk:

```python
import ast

BLOCKED_BUILTINS = {"eval", "exec", "compile", "__import__", "open", "breakpoint"}
BLOCKED_MODULES = {"subprocess", "os", "sys", "shutil", "socket", "ctypes", "importlib", "pickle"}

def _scan_for_dangerous_code(self, code: str) -> tuple[bool, str]:
    """AST-based safety check for generated tool code."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return True, f"Syntax error in generated code: {e}"

    for node in ast.walk(tree):
        # Block dangerous builtins
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in BLOCKED_BUILTINS:
                return True, f"Blocked builtin: {node.func.id}"
            if isinstance(node.func, ast.Attribute) and node.func.attr in BLOCKED_BUILTINS:
                return True, f"Blocked builtin: {node.func.attr}"
        # Block dangerous imports
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
            for name in names:
                if name and name.split(".")[0] in BLOCKED_MODULES:
                    return True, f"Blocked module import: {name}"
    return False, ""
```

**Step 3: Fix `test_tool()` registry path validation**

Find `test_tool()`. Before `subprocess.run([sys.executable, str(test_file)], ...)`, add:

```python
CUSTOM_TESTS_DIR = Path(__file__).parent.parent.parent / "data" / "custom_tool_tests"
test_file = Path(tool_entry.get("test_file", ""))
try:
    resolved_test = test_file.resolve()
    resolved_tests_dir = CUSTOM_TESTS_DIR.resolve()
    if not str(resolved_test).startswith(str(resolved_tests_dir)):
        return {"success": False, "error": "Test file path outside sandbox"}
except Exception as e:
    return {"success": False, "error": f"Invalid test file path: {e}"}
```

**Step 4: Syntax check**
```bash
python -c "import ast; ast.parse(open('D:/apprentice-agent/aura/tools/tool_builder.py', encoding='utf-8').read()); print('OK')"
```

---

### Task A-5: Fix synapseforge.py arbitrary code execution (C-5)

**Files:**
- Modify: `D:\apprentice-agent\aura\tools\synapseforge.py`

**Step 1: Read the file — focus on `execute_tool()` and `_validate_code()` (around lines 518–553)**

**Step 2: Add AST-based pre-validation before subprocess execution**

Add the same `_scan_for_dangerous_code` logic (copy from tool_builder fix above, or import if refactored). Before the `subprocess.run([sys.executable, str(temp_path)], ...)` call:

```python
# AST safety check before execution
dangerous, reason = self._scan_for_dangerous_code(code)
if dangerous:
    return {"success": False, "error": f"Generated code failed safety check: {reason}"}
```

**Step 3: Add `BLOCKED_MODULES` import filter to `_validate_code()`**

Similarly protect `_validate_code()` with the same AST walker.

**Step 4: Syntax check**
```bash
python -c "import ast; ast.parse(open('D:/apprentice-agent/aura/tools/synapseforge.py', encoding='utf-8').read()); print('OK')"
```

---

### Task A-6: Fix hooks.py command injection (C-7)

**Files:**
- Modify: `D:\apprentice-agent\aura\hooks.py`

**Step 1: Read the file — find lines around 333 and 336 where notifications are sent**

**Step 2: Fix PowerShell injection (line ~333)**

Find the PowerShell notification block. Change from f-string interpolation to a subprocess list call:

```python
# BEFORE (vulnerable):
# subprocess.run(["powershell", "-Command", f'$n.ShowBalloonTip(5000, "AURA Hook", "{message}", "Info")'], ...)

# AFTER (safe):
ps_script = (
    f"[void][System.Reflection.Assembly]::LoadWithPartialName('System.Windows.Forms');"
    f"$n = New-Object System.Windows.Forms.NotifyIcon;"
    f"$n.Icon = [System.Drawing.SystemIcons]::Information;"
    f"$n.Visible = $true;"
    f"$n.ShowBalloonTip(5000, 'AURA Hook', $env:HOOK_MSG, 'Info')"
)
subprocess.run(
    ["powershell", "-Command", ps_script],
    env={**os.environ, "HOOK_MSG": str(message)},
    timeout=5
)
```

**Step 3: Fix Linux shell injection (line ~336)**

Change `os.system(f'notify-send "AURA Hook" "{message}" ...')` to:
```python
subprocess.run(["notify-send", "AURA Hook", str(message)], timeout=5, check=False)
```

**Step 4: Syntax check**
```bash
python -c "import ast; ast.parse(open('D:/apprentice-agent/aura/hooks.py', encoding='utf-8').read()); print('OK')"
```

---

### Task A-7: Fix email_tool.py IMAP injection (C-25)

**Files:**
- Modify: `D:\apprentice-agent\aura\tools\email_tool.py`

**Step 1: Read the file — find lines around 467 and 316**

**Step 2: Add IMAP string sanitizer helper**

Near the top of the class or as a module-level function:
```python
def _sanitize_imap_string(value: str) -> str:
    """Strip characters that could break IMAP search string grammar."""
    return value.replace('"', '').replace('\\', '').replace('\r', '').replace('\n', '').strip()
```

**Step 3: Apply sanitizer at line ~467 and ~316**

At line ~467:
```python
safe_query = _sanitize_imap_string(query)
status, messages = mail.search(None, f'(OR SUBJECT "{safe_query}" BODY "{safe_query}")')
```

At line ~316 (wherever `since_date` is interpolated into an IMAP string):
```python
safe_date = _sanitize_imap_string(since_date)
status, messages = mail.search(None, f'(SINCE "{safe_date}")')
```

**Step 4: Syntax check**
```bash
python -c "import ast; ast.parse(open('D:/apprentice-agent/aura/tools/email_tool.py', encoding='utf-8').read()); print('OK')"
```

---

### Task A-8: Fix code_executor.py run_math injection (C-26)

**Files:**
- Modify: `D:\apprentice-agent\aura\tools\code_executor.py`

**Step 1: Read the file — find `run_math()` around line 247**

**Step 2: Pre-validate expression as AST expression before wrapping**

```python
def run_math(self, expression: str) -> dict:
    # Validate the expression is a pure math expression before wrapping
    try:
        tree = ast.parse(expression.strip(), mode='eval')
    except SyntaxError as e:
        return {"success": False, "output": "", "error": f"Invalid expression syntax: {e}"}

    # Only allow numeric/math nodes
    ALLOWED_NODES = (
        ast.Expression, ast.BinOp, ast.UnaryOp, ast.BoolOp,
        ast.Constant, ast.Num,  # Num for older Python compat
        ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod, ast.Pow,
        ast.FloorDiv, ast.BitAnd, ast.BitOr, ast.BitXor,
        ast.LShift, ast.RShift, ast.Invert, ast.Not, ast.UAdd, ast.USub,
        ast.Compare, ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
    )
    for node in ast.walk(tree):
        if not isinstance(node, ALLOWED_NODES):
            # Allow Call only for math builtins
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    MATH_FUNCS = {"abs", "round", "min", "max", "sum", "pow", "int", "float"}
                    if node.func.id not in MATH_FUNCS:
                        return {"success": False, "output": "", "error": f"Function '{node.func.id}' not allowed in math expressions"}
                elif isinstance(node.func, ast.Attribute):
                    # Allow math.sqrt etc
                    if not (isinstance(node.func.value, ast.Name) and node.func.value.id == "math"):
                        return {"success": False, "output": "", "error": "Only math.* functions allowed"}
            else:
                return {"success": False, "output": "", "error": f"Expression contains disallowed construct: {type(node).__name__}"}

    code = f"import math\nprint({expression})"
    return self.execute(code)
```

**Step 5: Syntax check**
```bash
python -c "import ast; ast.parse(open('D:/apprentice-agent/aura/tools/code_executor.py', encoding='utf-8').read()); print('OK')"
```

---

## Batch B — Authentication (C-8)

*Add API key authentication to all unprotected routes.*

---

### Task B-1: Create API key authentication dependency

**Files:**
- Create: `D:\apprentice-agent\api\auth.py`
- Modify: `D:\apprentice-agent\api\routes\proactive.py`
- Modify: `D:\apprentice-agent\api\routes\multi_agent.py`
- Modify: `D:\apprentice-agent\api\routes\self_improvement.py`
- Modify: `D:\apprentice-agent\api\routes\memory.py`
- Modify: `D:\apprentice-agent\api\routes\chat.py` (WebSocket auth)

**Step 1: Create `api/auth.py`**

```python
"""API key authentication for Aura routes."""
import os
import logging
from fastapi import Header, HTTPException, status

logger = logging.getLogger(__name__)

_API_KEY_ENV = "AURA_API_KEY"


def _get_configured_key() -> str | None:
    return os.environ.get(_API_KEY_ENV)


async def require_api_key(x_api_key: str = Header(default="")) -> str:
    """FastAPI dependency: validates X-API-Key header."""
    configured = _get_configured_key()
    if configured is None:
        # Key not configured — log warning but allow (dev mode)
        logger.warning(
            "[Auth] AURA_API_KEY not set. Running in unauthenticated dev mode. "
            "Set AURA_API_KEY env var before exposing to network."
        )
        return ""
    if not x_api_key or x_api_key != configured:
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
        return True  # dev mode
    return key == configured
```

**Step 2: Add `Depends(require_api_key)` to all route files**

For each of the four route files (`proactive.py`, `multi_agent.py`, `self_improvement.py`, `memory.py`):

At the top, add:
```python
from api.auth import require_api_key
from fastapi import Depends
```

On every `@router.get(...)`, `@router.post(...)`, `@router.delete(...)` decorator, add:
```python
@router.post("/start", dependencies=[Depends(require_api_key)])
```

Or add it at router creation level to protect all routes at once:
```python
router = APIRouter(prefix="/api/proactive", dependencies=[Depends(require_api_key)])
```

Use the router-level approach (one line per router file) since it protects all current and future routes.

**Step 3: Syntax check all modified files**
```bash
for f in api/auth.py api/routes/proactive.py api/routes/multi_agent.py api/routes/self_improvement.py api/routes/memory.py; do
  python -c "import ast; ast.parse(open('D:/apprentice-agent/$f', encoding='utf-8').read()); print('$f OK')"
done
```

---

## Batch C — Multi-User Isolation (C-9, C-10, H-17, H-18)

*Fix shared singletons and flat storage so users can't access each other's data.*

---

### Task C-1: Fix multi-agent orchestrator singleton (C-9)

**Files:**
- Modify: `D:\apprentice-agent\api\routes\multi_agent.py`

**Step 1: Read `multi_agent.py` — find `_orchestrator` global and `get_orchestrator()` function**

**Step 2: Replace global singleton with per-session dict**

```python
import threading
_orchestrators: dict[str, "MultiAgentOrchestrator"] = {}
_orch_lock = threading.Lock()

def get_orchestrator(session_id: str) -> "MultiAgentOrchestrator":
    with _orch_lock:
        if session_id not in _orchestrators:
            _orchestrators[session_id] = MultiAgentOrchestrator()
        return _orchestrators[session_id]
```

**Step 3: Update all route handlers to pass `session_id`**

Add `session_id: str = Query(default="default")` parameter to each route handler. Pass to `get_orchestrator(session_id)`.

For the `/clear` endpoint, change to clear only the calling session's orchestrator:
```python
@router.post("/clear")
async def clear_history(session_id: str = Query(default="default")):
    with _orch_lock:
        if session_id in _orchestrators:
            del _orchestrators[session_id]
    return {"status": "cleared", "session_id": session_id}
```

**Step 4: Syntax check**
```bash
python -c "import ast; ast.parse(open('D:/apprentice-agent/api/routes/multi_agent.py', encoding='utf-8').read()); print('OK')"
```

---

### Task C-2: Fix proactive daemon singleton (C-10)

**Files:**
- Modify: `D:\apprentice-agent\api\routes\proactive.py`

**Step 1: Read the file — find `_daemon` global and `_get_daemon()` function**

**Step 2: Replace with session-keyed dict (same pattern as C-1)**

```python
_daemons: dict[str, "GatewayDaemon"] = {}
_daemon_lock = threading.Lock()

async def _get_daemon(session_id: str = "default") -> "GatewayDaemon":
    with _daemon_lock:
        if session_id not in _daemons:
            _daemons[session_id] = GatewayDaemon()
        return _daemons[session_id]
```

Update all route handlers to accept and pass `session_id: str = Query(default="default")`.

**Step 3: Syntax check**
```bash
python -c "import ast; ast.parse(open('D:/apprentice-agent/api/routes/proactive.py', encoding='utf-8').read()); print('OK')"
```

---

### Task C-3: Namespace notifications and calendar storage by user (H-17)

**Files:**
- Modify: `D:\apprentice-agent\aura\tools\notifications.py`
- Modify: `D:\apprentice-agent\aura\tools\calendar_tool.py`

**Step 1: Read both files — find the `TASKS_FILE` and `calendar_events.json` path constants**

**Step 2: Make file paths accept a `user_id` parameter**

In `notifications.py`, find the class `__init__` or wherever `TASKS_FILE` is used. Change the hardcoded path to:

```python
def __init__(self, user_id: str = "default"):
    self.user_id = user_id
    data_dir = Path(__file__).parent.parent.parent / "data" / "users" / user_id
    data_dir.mkdir(parents=True, exist_ok=True)
    self.tasks_file = data_dir / "scheduled_tasks.json"
```

Do the same in `calendar_tool.py` for `calendar_events.json`.

**Step 3: Update tool `execute()` methods to accept optional `user_id` kwarg**

In the `execute(self, action, **kwargs)` method, extract `user_id = kwargs.get("user_id", "default")` and pass it to the constructor or method calls.

**Step 4: Syntax check both files**
```bash
python -c "import ast; ast.parse(open('D:/apprentice-agent/aura/tools/notifications.py', encoding='utf-8').read()); print('notifications OK')"
python -c "import ast; ast.parse(open('D:/apprentice-agent/aura/tools/calendar_tool.py', encoding='utf-8').read()); print('calendar OK')"
```

---

### Task C-4: Scope memory tracker by session (H-18)

**Files:**
- Modify: `D:\apprentice-agent\api\routes\memory.py`

**Step 1: Read the file — find `_tracker` global and `GET /recalls/recent` + `POST /recalls/record`**

**Step 2: Replace global tracker with session-keyed dict**

```python
_trackers: dict[str, "MemoryRecallTracker"] = {}
_tracker_lock = threading.Lock()

def _get_tracker(session_id: str) -> "MemoryRecallTracker":
    with _tracker_lock:
        if session_id not in _trackers:
            _trackers[session_id] = MemoryRecallTracker()
        return _trackers[session_id]
```

Update route handlers to accept and pass `session_id: str = Query(default="default")`.

**Step 3: Syntax check**
```bash
python -c "import ast; ast.parse(open('D:/apprentice-agent/api/routes/memory.py', encoding='utf-8').read()); print('OK')"
```

---

## Batch D — Memory Persistence (C-11, C-12, C-13, C-14, H-14, H-15)

*Fix data that is silently lost on restart.*

---

### Task D-1: Fix truth_spine.py store_fact() dedup + thread lock (C-11, C-12)

**Files:**
- Modify: `D:\apprentice-agent\aura\truth_spine.py`

**Step 1: Read the file — find `VerifiedMemory.__init__`, `store_fact()`, `_rank_by_relevance()`**

**Step 2: Add thread lock to `__init__`**

In `VerifiedMemory.__init__`, add:
```python
import threading
self._lock = threading.Lock()
```

**Step 3: Add dedup to `store_fact()` (mirrors store_belief/store_speculation)**

In `store_fact()`, after `if not verification.is_verified:` check, add:
```python
h = self._content_hash(content)
for trace in self.traces.values():
    if trace.tier == MemoryTier.FACT and trace.reasoning and f"hash:{h}" in trace.reasoning:
        trace.last_accessed = time.time()
        trace.access_count = getattr(trace, 'access_count', 0) + 1
        return trace
self._prune_if_needed()
```

And update the trace creation to embed the hash in `reasoning`:
```python
reasoning=f"Verified: {verification.reasoning} [hash:{h}]",
```

**Step 4: Protect all public methods with `self._lock`**

Wrap `store_fact()`, `store_belief()`, `store_speculation()`, `retrieve_facts()`, `retrieve_beliefs()`, `retrieve_all()`, `_rank_by_relevance()` with `with self._lock:`.

**Step 5: Save access metadata after `_rank_by_relevance()` mutations**

At the end of `_rank_by_relevance()`, after all mutations, add:
```python
self._save()
```
(This is already inside the lock from Step 4 so no double-lock issue.)

**Step 6: Syntax check**
```bash
python -c "import ast; ast.parse(open('D:/apprentice-agent/aura/truth_spine.py', encoding='utf-8').read()); print('OK')"
```

---

### Task D-2: Fix AMEMSystem.update() not persisting (C-13)

**Files:**
- Modify: `D:\apprentice-agent\aura\tools\amem.py`

**Step 1: Read the file — find `AMEMSystem.update()` method (~line 598)**

**Step 2: Add `self.save()` call at the end of `update()`**

Find the end of the `update()` method body. Before the `return` statement, add:
```python
# Persist updated note to disk
self.save()
```

Note: `self.save()` does a full rewrite of the JSONL file, which handles the duplicate-entry problem that `_append_note` would create.

**Step 3: Syntax check**
```bash
python -c "import ast; ast.parse(open('D:/apprentice-agent/aura/tools/amem.py', encoding='utf-8').read()); print('OK')"
```

---

### Task D-3: Fix KnowledgeGraphTool mutation persistence (C-14)

**Files:**
- Modify: `D:\apprentice-agent\aura\tools\knowledge_graph.py`

**Step 1: Read the file — find `update_node()`, `strengthen_edge()`, `weaken_edge()`, `invalidate_edge()`, `supersede_edge()`, `get_node()`**

**Step 2: Add `self.save()` after mutating state in each method**

For each of these methods, at the end of the method body (after state mutation, before return), add:
```python
self.save()
```

For `get_node()` which increments `access_count` and `last_accessed`: these are lightweight mutations. Add `self.save()` at the end. If this causes performance issues, a dirty flag + periodic save can be added later — but correctness first.

**Step 3: Syntax check**
```bash
python -c "import ast; ast.parse(open('D:/apprentice-agent/aura/tools/knowledge_graph.py', encoding='utf-8').read()); print('OK')"
```

---

### Task D-4: Fix HybridAMEMSystem cross-system link persistence (H-14)

**Files:**
- Modify: `D:\apprentice-agent\aura\tools\hybrid_amem.py`

**Step 1: Read the file — find `__init__`, `_link_amem_to_kg()`, `consolidate()`**

**Step 2: Add link map file path in `__init__`**

```python
self._links_file = self.data_dir / "cross_system_links.json"
```

**Step 3: Load links on init**

After setting `self._amem_to_kg = {}` and `self._kg_to_amem = {}`, add:
```python
if self._links_file.exists():
    try:
        data = json.loads(self._links_file.read_text(encoding="utf-8"))
        self._amem_to_kg = data.get("amem_to_kg", {})
        self._kg_to_amem = data.get("kg_to_amem", {})
    except Exception:
        pass
```

**Step 4: Save links after `_link_amem_to_kg()`**

At the end of `_link_amem_to_kg()`, add:
```python
try:
    self._links_file.write_text(
        json.dumps({"amem_to_kg": self._amem_to_kg, "kg_to_amem": self._kg_to_amem}, indent=2),
        encoding="utf-8"
    )
except Exception as e:
    logger.warning(f"[HybridAMEM] Failed to save cross-system links: {e}")
```

**Step 5: Syntax check**
```bash
python -c "import ast; ast.parse(open('D:/apprentice-agent/aura/tools/hybrid_amem.py', encoding='utf-8').read()); print('OK')"
```

---

### Task D-5: Fix LocalRAG embedding search broken after restart (H-15)

**Files:**
- Modify: `D:\apprentice-agent\aura\tools\local_rag.py`

**Step 1: Read the file — find `_load_index()`, `search()`, and how `chunk.embedding` is used in search**

**Step 2: Track which chunks have embeddings by index**

The root problem: after load, `chunk.embedding` is `None` for all chunks, but `self.embeddings` (the numpy array) contains the data. `search()` uses `chunk.embedding` as the guard, so it never matches.

Fix: Add a `chunk_has_embedding` bool list that mirrors `self.chunks`:

In `_load_index()`, after loading chunks and embeddings, add:
```python
# Mark which chunks have a corresponding row in self.embeddings
n_emb = len(self.embeddings) if self.embeddings is not None else 0
self._chunk_has_embedding = [False] * len(self.chunks)
for i in range(min(n_emb, len(self.chunks))):
    self._chunk_has_embedding[i] = True
```

In `search()`, replace `if chunk.embedding and emb_idx < len(self.embeddings):` with:
```python
if self._chunk_has_embedding[idx] and emb_idx < len(self.embeddings):
```
where `idx` is the current chunk's position in `self.chunks`.

Make sure `_chunk_has_embedding` is initialized in `__init__` as `[]` and updated whenever embeddings are added during indexing.

**Step 3: Syntax check**
```bash
python -c "import ast; ast.parse(open('D:/apprentice-agent/aura/tools/local_rag.py', encoding='utf-8').read()); print('OK')"
```

---

## Batch E — Cognition Crashes (C-15, C-16, C-17, C-18, C-19)

---

### Task E-1: Fix MCTS log(0) crash (C-15)

**Files:**
- Modify: `D:\apprentice-agent\aura\tools\mcts_reasoning.py`

**Step 1: Read the file — find the UCB1 formula around line 148**

**Step 2: Guard `math.log` call with `max(1, ...)`**

Find `math.log(self.parent.visits)`. Replace with:
```python
math.log(max(1, self.parent.visits))
```

Also check if `self.visits + 1` is used in the denominator (it should be, to avoid div-by-zero). If it's just `self.visits`, change to `self.visits + 1`.

**Step 3: Syntax check**
```bash
python -c "import ast; ast.parse(open('D:/apprentice-agent/aura/tools/mcts_reasoning.py', encoding='utf-8').read()); print('OK')"
```

---

### Task E-2: Fix MCTS unbounded recursion (C-16)

**Files:**
- Modify: `D:\apprentice-agent\aura\tools\mcts_reasoning.py`

**Step 1: Find `backpropagate()` and `_get_all_nodes()` methods**

**Step 2: Convert `backpropagate()` to iterative**

```python
def backpropagate(self, value: float) -> None:
    """Iterative backpropagation up the tree."""
    node = self
    while node is not None:
        node.visits += 1
        node.value += value
        node = node.parent
```

**Step 3: Convert `_get_all_nodes()` to iterative**

```python
def _get_all_nodes(self) -> list:
    """Iterative BFS to collect all nodes."""
    result = []
    queue = [self]
    while queue:
        node = queue.pop(0)
        result.append(node)
        queue.extend(node.children)
    return result
```

**Step 4: Fix dead evaluation cache (H — finding #6 from consciousness audit)**

Find `_evaluate()`. Change the cache condition from `node.state == NodeState.EVALUATED and node.visits > 0` to:
```python
if node.state == NodeState.EVALUATED and node.value != 0.0:
    return node.value
```

**Step 5: Fix pruned nodes selectable (H — finding #7)**

Find `_select()`. Before the UCB1 loop, filter out pruned children:
```python
valid_children = [c for c in node.children if c.state != NodeState.PRUNED]
if not valid_children:
    return node
# use valid_children instead of node.children for UCB1
```

**Step 6: Syntax check**
```bash
python -c "import ast; ast.parse(open('D:/apprentice-agent/aura/tools/mcts_reasoning.py', encoding='utf-8').read()); print('OK')"
```

---

### Task E-3: Fix self_improvement.py false success on exception (C-17)

**Files:**
- Modify: `D:\apprentice-agent\aura\consciousness\self_improvement.py`

**Step 1: Read the file — find `_enhanced_practice()` around line 425**

**Step 2: Change `True` to `False` in the exception handler**

Find the `except Exception` block in `_enhanced_practice()`. Change:
```python
# BEFORE:
_record_outcome(domain, True, ...)
return ("practice attempt...", True)

# AFTER:
_record_outcome(domain, False, f"practice attempt failed for {domain}: {e}")
return (f"practice attempt for {domain} failed: LLM unavailable", False)
```

**Step 3: Fix Brain() instantiated on every call (H — finding #13)**

Find where `Brain()` is called inside `_enhanced_practice()`. Replace with a cached instance:

In `__init__` of the class:
```python
self._practice_brain = None
self._practice_brain_lock = threading.Lock()
```

Then in `_enhanced_practice()`, replace `brain = Brain()` with:
```python
with self._practice_brain_lock:
    if self._practice_brain is None:
        self._practice_brain = Brain()
    brain = self._practice_brain
```

**Step 4: Syntax check**
```bash
python -c "import ast; ast.parse(open('D:/apprentice-agent/aura/consciousness/self_improvement.py', encoding='utf-8').read()); print('OK')"
```

---

### Task E-4: Fix reflexion.py file handle leak and Windows 1-byte lock (C-18, C-19)

**Files:**
- Modify: `D:\apprentice-agent\aura\tools\reflexion.py`

**Step 1: Read the file — find the `file_lock` context manager around lines 57–81**

**Step 2: Fix file handle leak on lock failure**

Find the `except (ImportError, OSError)` block. Remove the second `f = open(...)` call from inside the except block. `f` is already open — just yield it:

```python
@contextmanager
def file_lock(filepath, mode='r', encoding='utf-8'):
    f = None
    try:
        f = open(filepath, mode, encoding=encoding)
        # Try to lock
        try:
            if sys.platform == 'win32':
                import msvcrt
                file_size = max(1, os.path.getsize(filepath))
                msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, file_size)
            else:
                import fcntl
                fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (ImportError, OSError) as e:
            logger.warning(f"File locking unavailable: {e}")
            # f is already open, proceed without lock
        yield f
    finally:
        if f is not None:
            try:
                if sys.platform == 'win32':
                    import msvcrt
                    try:
                        file_size = max(1, os.path.getsize(filepath))
                        msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, file_size)
                    except Exception:
                        pass
                else:
                    import fcntl
                    try:
                        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                    except Exception:
                        pass
            finally:
                f.close()
```

**Step 3: Fix IndexError on empty `past_lessons` (H — finding #10)**

Find the return statement at the end of the retry loop (around line 421). Change:
```python
new_reflection=past_lessons[-1].reflection if past_lessons else None
```

**Step 4: Syntax check**
```bash
python -c "import ast; ast.parse(open('D:/apprentice-agent/aura/tools/reflexion.py', encoding='utf-8').read()); print('OK')"
```

---

## Batch F — Race Conditions & Thread Safety (C-20, C-21, C-22, C-23, C-24, H-16)

---

### Task F-1: Fix agent_service.py _initializing race (C-20)

**Files:**
- Modify: `D:\apprentice-agent\api\services\agent_service.py`

**Step 1: Read the file — find `start_background_init()` around line 324**

**Step 2: Wrap check-and-set in existing `_agent_lock`**

```python
def start_background_init(self):
    with self._agent_lock:
        if self._agent is not None or self._initializing:
            return
        self._initializing = True
    # Start background thread outside the lock
    thread = threading.Thread(target=self._background_init, daemon=True)
    thread.start()
```

**Step 3: Fix `brain` unbound in finally block (C-22)**

Find `chat_stream()`. Before the `try:` block, add:
```python
brain = None
effective_model = None
```

In the `finally:` block, change:
```python
if effective_model and brain is not None:
    with self._agent_lock:
        brain.set_model_override(None)
```

**Step 4: Syntax check**
```bash
python -c "import ast; ast.parse(open('D:/apprentice-agent/api/services/agent_service.py', encoding='utf-8').read()); print('OK')"
```

---

### Task F-2: Fix RateLimitMiddleware concurrent mutation (C-21)

**Files:**
- Modify: `D:\apprentice-agent\api\middleware.py`

**Step 1: Read the file — find `RateLimitMiddleware` class, `dispatch()`, `_cleanup_old_entries()`**

**Step 2: Add asyncio.Lock**

In `__init__`:
```python
self._lock = asyncio.Lock()
```

**Step 3: Wrap the entire critical section in `dispatch()` with `async with self._lock:`**

```python
async def dispatch(self, request: Request, call_next):
    client_ip = request.client.host if request.client else "unknown"
    now = time.time()

    async with self._lock:
        # Filter old requests
        window_start = now - self.window_seconds
        self._requests[client_ip] = [
            t for t in self._requests[client_ip] if t > window_start
        ]

        if len(self._requests[client_ip]) >= self.max_requests:
            return JSONResponse(status_code=429, content={"detail": "Rate limit exceeded"})

        self._requests[client_ip].append(now)

        # Periodic cleanup
        if now - self._last_cleanup > 60:
            self._cleanup_old_entries(now)
            self._last_cleanup = now

    return await call_next(request)
```

**Step 4: Syntax check**
```bash
python -c "import ast; ast.parse(open('D:/apprentice-agent/api/middleware.py', encoding='utf-8').read()); print('OK')"
```

---

### Task F-3: Fix ALMAEngine class-level variables (C-23, C-24)

**Files:**
- Modify: `D:\apprentice-agent\aura\emotion\alma_engine.py`

**Step 1: Read the file — find `__init__`, and lines 951–952 + 1020–1022**

**Step 2: Move class-level variables to instance variables in `__init__`**

Find where `_weather_cache`, `_weather_cache_time`, `_last_interaction_time`, `_success_streak`, `_last_drift_time` are declared at class level. Remove those class-level declarations. In `__init__`, add:

```python
self._weather_cache: Optional[Dict[str, Any]] = None
self._weather_cache_time: float = 0.0
self._last_interaction_time: float = 0.0
self._success_streak: int = 0
self._last_drift_time: float = 0.0
```

**Step 3: Fix non-atomic reads in `get_response_style_prompt` (C-24)**

Find `get_response_style_prompt()`. It makes two separate locked calls to `get_emotional_state()` and `get_response_modulation()`. Consolidate into one locked snapshot:

```python
def get_response_style_prompt(self) -> str:
    with self._lock:
        state = self.get_emotional_state()  # reentrant — safe inside lock
        modulation = self.get_response_modulation()  # also reentrant
    # now use state and modulation to build prompt
    ...
```

**Step 4: Syntax check**
```bash
python -c "import ast; ast.parse(open('D:/apprentice-agent/aura/emotion/alma_engine.py', encoding='utf-8').read()); print('OK')"
```

---

### Task F-4: Fix UnifiedMemory._init_backends race (H-16)

**Files:**
- Modify: `D:\apprentice-agent\aura\memory\unified_memory.py`

**Step 1: Read the file — find `__init__`, `_init_backends()`, `get_unified_memory()`**

**Step 2: Add lock to `__init__`**

```python
import threading
self._init_lock = threading.Lock()
```

**Step 3: Wrap `_init_backends()` with double-checked locking**

```python
def _init_backends(self):
    if self._backends_checked:
        return
    with self._init_lock:
        if self._backends_checked:  # re-check after acquiring lock
            return
        self._backends_checked = True
        # ... rest of init code
```

**Step 4: Fix `get_unified_memory()` singleton**

```python
_unified_memory_lock = threading.Lock()

def get_unified_memory() -> UnifiedMemory:
    global _unified_memory_instance
    if _unified_memory_instance is None:
        with _unified_memory_lock:
            if _unified_memory_instance is None:
                _unified_memory_instance = UnifiedMemory()
    return _unified_memory_instance
```

**Step 5: Syntax check**
```bash
python -c "import ast; ast.parse(open('D:/apprentice-agent/aura/memory/unified_memory.py', encoding='utf-8').read()); print('OK')"
```

---

### Task F-5: Fix GlobalWorkspace double-start race (H — finding #11 from consciousness audit)

**Files:**
- Modify: `D:\apprentice-agent\aura\consciousness\global_workspace.py`

**Step 1: Read the file — find `start()` method**

**Step 2: Wrap `_running` check inside `self._lock`**

```python
def start(self):
    with self._lock:
        if self._running:
            return
        self._running = True
    # start thread outside the lock
    self._thread = threading.Thread(target=self._run_cognitive_cycle, daemon=True)
    self._thread.start()
```

**Step 3: Syntax check**
```bash
python -c "import ast; ast.parse(open('D:/apprentice-agent/aura/consciousness/global_workspace.py', encoding='utf-8').read()); print('OK')"
```

---

### Task F-6: Fix NeuroDream deadlock in enter_sleep (H — emotion audit finding #9)

**Files:**
- Modify: `D:\apprentice-agent\aura\tools\neurodream.py`

**Step 1: Read the file — find `enter_sleep()` around lines 350–391**

**Step 2: Move `thread.start()` outside the lock**

```python
def enter_sleep(self, trigger: str = "manual") -> Dict[str, Any]:
    thread = None
    with self._phase_lock:
        # all state checks and session init here
        ...
        thread = threading.Thread(target=self._run_sleep_cycle, daemon=True)
        self._sleep_thread = thread
    # Start AFTER releasing _phase_lock
    if thread:
        thread.start()
    return {"success": True, ...}
```

**Step 3: Syntax check**
```bash
python -c "import ast; ast.parse(open('D:/apprentice-agent/aura/tools/neurodream.py', encoding='utf-8').read()); print('OK')"
```

---

## Batch G — Remaining High Severity Fixes

---

### Task G-1: Fix path traversal in chat.py attachments (H-1)

**Files:**
- Modify: `D:\apprentice-agent\api\routes\chat.py`

**Step 1: Read the file — find `process_attachments()` around lines 97–129**

**Step 2: Add UPLOAD_DIR bounds check**

Find where `file_path = attachment.get("path")` is used and the file is opened. Add before the `open()`:

```python
import os
UPLOAD_DIR_RESOLVED = Path(UPLOAD_DIR).resolve()

file_path_resolved = Path(file_path).resolve()
if not str(file_path_resolved).startswith(str(UPLOAD_DIR_RESOLVED)):
    logger.warning(f"[Attachments] Path traversal blocked: {file_path}")
    continue  # skip this attachment
```

**Step 3: Fix disconnected WebSockets not cleaned up (H-7)**

Find `_broadcast_json()`. In the inner `except Exception: pass` block, add `unregister_websocket(ws)`:

```python
for ws in targets:
    try:
        await ws.send_json(payload)
    except Exception:
        unregister_websocket(ws)
```

**Step 4: Fix `asyncio.get_event_loop()` deprecated calls (M-4)**

Find all `asyncio.get_event_loop()` calls inside async functions. Replace with `asyncio.get_running_loop()`.

**Step 5: Syntax check**
```bash
python -c "import ast; ast.parse(open('D:/apprentice-agent/api/routes/chat.py', encoding='utf-8').read()); print('OK')"
```

---

### Task G-2: Fix path traversal in pdf_reader, research_tool, calendar_tool (H-3, H-4, H-9)

**Files:**
- Modify: `D:\apprentice-agent\aura\tools\pdf_reader.py`
- Modify: `D:\apprentice-agent\aura\tools\research_tool.py`
- Modify: `D:\apprentice-agent\aura\tools\calendar_tool.py`

**Step 1: Fix pdf_reader.py — add sandbox before fitz.open()**

Find `info()` and `extract_text()`. Before `fitz.open(pdf_path)`, add:
```python
DOCS_DIR = Path(__file__).parent.parent.parent / "data"
resolved = Path(path).resolve()
docs_dir_resolved = DOCS_DIR.resolve()
if not str(resolved).startswith(str(docs_dir_resolved)):
    return {"success": False, "error": "Path not within allowed data directory"}
```

**Step 2: Fix research_tool.py — reject unknown categories**

Find `_resolve_category()`. Change the fallback:
```python
def _resolve_category(self, category: str) -> str:
    cat = category.lower().strip()
    resolved = CATEGORIES.get(cat)
    if resolved is None:
        raise ValueError(f"Unknown category: {category!r}. Allowed: {list(CATEGORIES.keys())}")
    return resolved
```

**Step 3: Fix calendar_tool.py `import_ics()` path traversal**

Find `import_ics()`. Add before `open(ics_path)`:
```python
ALLOWED_DIRS = [Path.home() / "Downloads", Path.home() / "Documents", Path(__file__).parent.parent.parent / "data"]
resolved = Path(path).resolve()
if not any(str(resolved).startswith(str(d.resolve())) for d in ALLOWED_DIRS):
    return {"success": False, "error": "ICS path not in an allowed directory"}
```

**Step 4: Syntax check all three**
```bash
for f in "aura/tools/pdf_reader.py" "aura/tools/research_tool.py" "aura/tools/calendar_tool.py"; do
  python -c "import ast; ast.parse(open('D:/apprentice-agent/$f', encoding='utf-8').read()); print('$f OK')"
done
```

---

### Task G-3: Fix identity.py and alma_engine.py non-atomic writes (H-8, H-9)

**Files:**
- Modify: `D:\apprentice-agent\aura\identity.py`
- Modify: `D:\apprentice-agent\aura\emotion\alma_engine.py`

**Step 1: Fix identity.py `save_identity()`**

Find `save_identity()`. Replace `open(IDENTITY_FILE, "w", ...)` with atomic write:

```python
import os, tempfile

def save_identity(data: dict) -> bool:
    try:
        dir_ = IDENTITY_FILE.parent
        dir_.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(dir=dir_, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4)
            os.replace(tmp_path, IDENTITY_FILE)
        except Exception:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
            raise
        return True
    except (IOError, OSError):
        return False
```

**Step 2: Fix alma_engine.py `_save_state()`**

Find `_save_state()`. Replace `self.state_file.write_text(...)` with:

```python
import os, tempfile

def _save_state(self):
    try:
        state = { ... }  # existing state dict construction
        dir_ = self.state_file.parent
        fd, tmp_path = tempfile.mkstemp(dir=dir_, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2)
            os.replace(tmp_path, self.state_file)
        except Exception:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
            raise
    except Exception as e:
        logger.error(f"[ALMA] Failed to save state: {e}")
```

**Step 3: Syntax check both**
```bash
python -c "import ast; ast.parse(open('D:/apprentice-agent/aura/identity.py', encoding='utf-8').read()); print('identity OK')"
python -c "import ast; ast.parse(open('D:/apprentice-agent/aura/emotion/alma_engine.py', encoding='utf-8').read()); print('alma OK')"
```

---

### Task G-4: Fix secure_logging.py traceback sanitization (H-10)

**Files:**
- Modify: `D:\apprentice-agent\aura\secure_logging.py`

**Step 1: Read the file — find `SanitizingFormatter.format()` around lines 104–119**

**Step 2: Apply sanitization to the fully-formatted string (including traceback)**

```python
def format(self, record: logging.LogRecord) -> str:
    # Sanitize the message and args first (existing logic)
    if record.msg and isinstance(record.msg, str):
        record.msg = self._sanitize(record.msg)
    if record.args:
        if isinstance(record.args, dict):
            record.args = {k: self._sanitize(str(v)) if isinstance(v, str) else v
                          for k, v in record.args.items()}
        elif isinstance(record.args, tuple):
            record.args = tuple(self._sanitize(str(a)) if isinstance(a, str) else a
                               for a in record.args)

    # Format (this includes exc_info traceback appended at end)
    formatted = super().format(record)

    # Sanitize the entire formatted string to catch traceback secrets
    return self._sanitize(formatted)
```

Where `self._sanitize(text)` is the existing sanitization method (or `sanitize_text(text)`).

**Step 3: Syntax check**
```bash
python -c "import ast; ast.parse(open('D:/apprentice-agent/aura/secure_logging.py', encoding='utf-8').read()); print('OK')"
```

---

### Task G-5: Fix router.py unbounded cache + IndexError crash (H-11, H-12)

**Files:**
- Modify: `D:\apprentice-agent\aura\multi_agent\router.py`

**Step 1: Read the file — find `_intent_cache`, `route()`, and `_score_specialists()`**

**Step 2: Replace unbounded dict with bounded OrderedDict**

```python
from collections import OrderedDict

MAX_CACHE_SIZE = 500

# In __init__:
self._intent_cache: OrderedDict = OrderedDict()

# Add cache setter method:
def _cache_set(self, key: str, value) -> None:
    if len(self._intent_cache) >= MAX_CACHE_SIZE:
        self._intent_cache.popitem(last=False)  # evict oldest
    self._intent_cache[key] = value
```

Replace all `self._intent_cache[key] = value` with `self._cache_set(key, value)`.

**Step 3: Guard against empty specialists in `route()`**

After `scores = self._score_specialists(query_lower)`, add:
```python
if not scores:
    return RoutingDecision(
        agents=["analyst"],
        mode=CollaborationMode.SINGLE,
        reasoning="No specialists available",
        confidence=0.0
    )
```

**Step 4: Syntax check**
```bash
python -c "import ast; ast.parse(open('D:/apprentice-agent/aura/multi_agent/router.py', encoding='utf-8').read()); print('OK')"
```

---

### Task G-6: Fix email encryption insecure key + plaintext fallback (H-19)

**Files:**
- Modify: `D:\apprentice-agent\aura\tools\email_tool.py`

**Step 1: Read lines 34–62 of the file**

**Step 2: Require secret from environment variable**

Replace the deterministic key derivation:

```python
import os

def _derive_encryption_key(self) -> bytes:
    """Derive Fernet key from a user-provided secret."""
    secret = os.environ.get("AURA_EMAIL_KEY")
    if not secret:
        raise RuntimeError(
            "AURA_EMAIL_KEY environment variable is not set. "
            "Set it to a strong random secret before storing email credentials."
        )
    import hashlib, base64
    key_bytes = hashlib.pbkdf2_hmac("sha256", secret.encode(), b"aura-email-v1", 100_000)
    return base64.urlsafe_b64encode(key_bytes)
```

**Step 3: Remove the plaintext fallback**

Find the `except ImportError` block that silently stores credentials in plaintext. Replace with:
```python
except ImportError:
    raise RuntimeError(
        "The 'cryptography' package is required to store email credentials. "
        "Install it with: pip install cryptography"
    )
```

**Step 4: Syntax check**
```bash
python -c "import ast; ast.parse(open('D:/apprentice-agent/aura/tools/email_tool.py', encoding='utf-8').read()); print('OK')"
```

---

### Task G-7: Pin VCS dependency in requirements.txt (H-20)

**Files:**
- Modify: `D:\apprentice-agent\requirements.txt`

**Step 1: Find the unpinned VCS line**

```
moshi @ git+https://github.com/NVIDIA/personaplex.git#subdirectory=moshi
```

**Step 2: Get the current HEAD commit SHA**

```bash
git ls-remote https://github.com/NVIDIA/personaplex.git HEAD | cut -f1
```

**Step 3: Pin to that SHA**

```
moshi @ git+https://github.com/NVIDIA/personaplex.git@<SHA_FROM_STEP_2>#subdirectory=moshi
```

---

### Task G-8: Fix MirrorMind biased scores (H — consciousness finding #12)

**Files:**
- Modify: `D:\apprentice-agent\aura\tools\mirrormind.py`

**Step 1: Read the file — find `_parse_critique()` around line 162**

**Step 2: Normalize weights over actually-returned dimensions**

Find the weighted sum calculation. Replace:
```python
# BEFORE: missing dimensions default to 0.5, biasing score
weighted_sum = sum(dimension_scores.get(dim, 0.5) * weight for dim, weight in self.DIMENSIONS.items())

# AFTER: normalize over present dimensions only
total_weight = sum(self.DIMENSIONS[dim] for dim in dimension_scores if dim in self.DIMENSIONS)
if total_weight > 0:
    weighted_sum = sum(
        dimension_scores[dim] * self.DIMENSIONS[dim]
        for dim in dimension_scores
        if dim in self.DIMENSIONS
    ) / total_weight
else:
    weighted_sum = 0.5  # fallback if no dimensions parsed
```

**Step 3: Syntax check**
```bash
python -c "import ast; ast.parse(open('D:/apprentice-agent/aura/tools/mirrormind.py', encoding='utf-8').read()); print('OK')"
```

---

### Task G-9: Fix InnerMonologue Lock → RLock + singleton thread-safety (H — findings #14, #17)

**Files:**
- Modify: `D:\apprentice-agent\aura\tools\inner_monologue.py`

**Step 1: Read the file — find `__init__` (line ~191), `get_monologue()` (line ~696)**

**Step 2: Change `threading.Lock()` to `threading.RLock()`**

```python
self._lock = threading.RLock()  # was threading.Lock()
```

**Step 3: Add double-checked locking to `get_monologue()` singleton**

```python
_monologue_lock = threading.Lock()

def get_monologue() -> InnerMonologueTool:
    global _monologue_instance
    if _monologue_instance is None:
        with _monologue_lock:
            if _monologue_instance is None:
                _monologue_instance = InnerMonologueTool()
    return _monologue_instance
```

**Step 4: Syntax check**
```bash
python -c "import ast; ast.parse(open('D:/apprentice-agent/aura/tools/inner_monologue.py', encoding='utf-8').read()); print('OK')"
```

---

### Task G-10: Fix scheduler conditional cooldown not persisted (H-8 from core audit)

**Files:**
- Modify: `D:\apprentice-agent\aura\scheduler.py`

**Step 1: Read the file — find `_check_conditional()` around line 239–250**

**Step 2: Store `last_triggered` in the persistent task dict**

In `_check_conditional()`, when a condition fires, update the persistent `conditional["last_triggered"]` (which is already loaded from disk each run) with the current timestamp:

```python
conditional["last_triggered"] = now.isoformat()
modified = True  # ensure save() is called
```

Then change the cooldown check from using `_condition_cooldown` dict (in-memory) to reading `conditional.get("last_triggered")`:

```python
last_triggered = conditional.get("last_triggered")
if last_triggered:
    try:
        last_dt = datetime.fromisoformat(last_triggered)
        cooldown_secs = conditional.get("cooldown_minutes", 60) * 60
        if (now - last_dt).total_seconds() < cooldown_secs:
            continue  # still in cooldown
    except (ValueError, TypeError):
        pass
```

Remove or keep `_condition_cooldown` for in-process deduplication only (as a cache on top of the persistent check).

**Step 3: Syntax check**
```bash
python -c "import ast; ast.parse(open('D:/apprentice-agent/aura/scheduler.py', encoding='utf-8').read()); print('OK')"
```

---

### Task G-11: Fix exceptions.py shadowing built-ins (H-9 from core audit)

**Files:**
- Modify: `D:\apprentice-agent\aura\exceptions.py`

**Step 1: Read the file — find `TimeoutError` and `MemoryError` class definitions**

**Step 2: Rename to avoid shadowing**

```python
# BEFORE:
class TimeoutError(AURAException): ...
class MemoryError(AURAException): ...

# AFTER:
class AURATimeoutError(AURAException): ...
class AURAMemoryError(AURAException): ...

# Add backward-compat aliases if needed (search for usages first)
TimeoutError = AURATimeoutError  # only if no callers catch the builtin
```

**Step 3: Search all files for `except TimeoutError` and `except MemoryError` to identify callers that need updating**

```bash
grep -r "except TimeoutError\|except MemoryError\|raise TimeoutError\|raise MemoryError" D:/apprentice-agent --include="*.py" -l
```

Update those files to use `AURATimeoutError` / `AURAMemoryError` where they mean the custom exception, and use the built-in where they mean OS/asyncio exceptions.

**Step 4: Syntax check**
```bash
python -c "import ast; ast.parse(open('D:/apprentice-agent/aura/exceptions.py', encoding='utf-8').read()); print('OK')"
```

---

### Task G-12: Fix AMEMSystem _reembed_orphans TOCTOU (H — memory finding #9)

**Files:**
- Modify: `D:\apprentice-agent\aura\tools\amem.py`

**Step 1: Read the file — find `_reembed_orphans` closure around line 1216–1249**

**Step 2: Move `_notes.get()` inside the lock**

```python
for _nid in _ids:
    _emb = _self._embed(...)  # compute embedding outside lock (CPU/GPU work)
    if _emb is None:
        continue
    with _self._lock:
        _note = _self._notes.get(_nid)  # moved inside lock
        if _note is None:
            continue  # note was deleted since we started
        _self._embeddings[_nid] = _emb
        _note.has_embedding = True
        # ChromaDB upsert inside lock
        ...
        _reembedded += 1
```

**Step 3: Syntax check**
```bash
python -c "import ast; ast.parse(open('D:/apprentice-agent/aura/tools/amem.py', encoding='utf-8').read()); print('OK')"
```

---

## Batch H — Medium Severity + Coverage Gaps

---

### Task H-1: Fix reasoning_tree_tool.py IndexError (H — consciousness finding #8)

**Files:**
- Modify: `D:\apprentice-agent\aura\tools\reasoning_tree_tool.py`

**Step 1: Read the file — find `explore_options()` around line 196**

**Step 2: Guard `options[0]` access**

```python
if not options:
    return {"success": False, "error": "MCTS search produced no options. Check LLM availability."}
best = options[0]
```

**Step 3: Syntax check**
```bash
python -c "import ast; ast.parse(open('D:/apprentice-agent/aura/tools/reasoning_tree_tool.py', encoding='utf-8').read()); print('OK')"
```

---

### Task H-2: Fix strategy_bandit total_reward unbounded negative (M-1)

**Files:**
- Modify: `D:\apprentice-agent\aura\consciousness\strategy_bandit.py`

**Step 1: Read the file — find `record_user_feedback()` around line 633**

**Step 2: Clamp `total_reward`**

Find the line `total_reward = total_reward + reward_delta`. Change to:
```python
total_reward = max(0.0, total_reward + reward_delta)
```

**Step 3: Syntax check**
```bash
python -c "import ast; ast.parse(open('D:/apprentice-agent/aura/consciousness/strategy_bandit.py', encoding='utf-8').read()); print('OK')"
```

---

### Task H-3: Fix UnifiedMemory O(n²) dedup (H-6 from memory audit)

**Files:**
- Modify: `D:\apprentice-agent\aura\memory\unified_memory.py`

**Step 1: Read the file — find dedup block in `query()` around lines 165–177**

**Step 2: Replace with O(n) dict-based dedup**

```python
# Replace the dedup loop with:
best: dict[str, any] = {}
for r in all_results:
    h = r.content_hash
    if h not in best or r.score > best[h].score:
        best[h] = r
deduped = list(best.values())
```

**Step 3: Syntax check**
```bash
python -c "import ast; ast.parse(open('D:/apprentice-agent/aura/memory/unified_memory.py', encoding='utf-8').read()); print('OK')"
```

---

### Task H-4: Fix remaining asyncio.get_event_loop() deprecations (M-4)

**Files:**
- Modify: `D:\apprentice-agent\api\routes\multi_agent.py`
- Modify: `D:\apprentice-agent\api\routes\memory.py`
- Modify: `D:\apprentice-agent\api\main.py`

**Step 1: Find all occurrences**

```bash
grep -n "get_event_loop" D:/apprentice-agent/api/routes/multi_agent.py D:/apprentice-agent/api/routes/memory.py D:/apprentice-agent/api/main.py
```

**Step 2: Replace all `asyncio.get_event_loop()` inside async functions with `asyncio.get_running_loop()`**

Each line found: change `asyncio.get_event_loop()` → `asyncio.get_running_loop()`.

**Step 3: Syntax check all three**
```bash
for f in "api/routes/multi_agent.py" "api/routes/memory.py" "api/main.py"; do
  python -c "import ast; ast.parse(open('D:/apprentice-agent/$f', encoding='utf-8').read()); print('$f OK')"
done
```

---

### Task H-5: Fix proactive.py daemon start race + pending_messages lock (H-8, M-5 from tests audit)

**Files:**
- Modify: `D:\apprentice-agent\api\routes\proactive.py`

**Step 1: Read the file — find `/start` endpoint (line ~123) and test-message endpoint**

**Step 2: Add asyncio.Lock to `/start` endpoint**

```python
_start_lock = asyncio.Lock()

@router.post("/start", dependencies=[Depends(require_api_key)])
async def start_daemon(session_id: str = Query(default="default")):
    async with _start_lock:
        daemon = await _get_daemon(session_id)
        if daemon.state.value == "running":
            return {"status": "already_running"}
        asyncio.create_task(run_daemon(daemon))
    return {"status": "started"}
```

**Step 3: Protect `_pending_messages` with asyncio.Queue**

Find where `daemon._pending_messages.append(message)` is called in the test-message endpoint. If `_pending_messages` is a plain list in the daemon class, change it to `asyncio.Queue` or protect appends with `asyncio.Lock`. At minimum, wrap the append in the route handler:

```python
# In the test-message endpoint:
async with _start_lock:  # reuse or create dedicated lock
    daemon._pending_messages.append(message)
```

**Step 4: Syntax check**
```bash
python -c "import ast; ast.parse(open('D:/apprentice-agent/api/routes/proactive.py', encoding='utf-8').read()); print('OK')"
```

---

### Task H-6: Add basic test coverage for critical paths

**Files:**
- Create: `D:\apprentice-agent\tests\test_auth.py`
- Create: `D:\apprentice-agent\tests\test_memory_persistence.py`
- Create: `D:\apprentice-agent\tests\test_injection_guards.py`

**test_auth.py — verify unauthenticated requests are blocked:**
```python
import os
import pytest
from fastapi.testclient import TestClient

# Set API key before importing app
os.environ["AURA_API_KEY"] = "test-key-123"

from api.main import app

client = TestClient(app)

def test_proactive_start_requires_auth():
    response = client.post("/api/proactive/start")
    assert response.status_code == 401

def test_proactive_start_with_valid_key():
    response = client.post("/api/proactive/start", headers={"X-API-Key": "test-key-123"})
    assert response.status_code != 401

def test_memory_recalls_requires_auth():
    response = client.get("/api/memory/recalls/recent")
    assert response.status_code == 401
```

**test_memory_persistence.py — verify fixes from Batch D:**
```python
import pytest
import tempfile
from pathlib import Path

def test_store_fact_dedup():
    """store_fact() should not create duplicate entries for same content."""
    from aura.truth_spine import VerifiedMemory, VerificationResult
    with tempfile.TemporaryDirectory() as tmpdir:
        vm = VerifiedMemory(data_dir=Path(tmpdir))
        v = VerificationResult(is_verified=True, confidence=0.9, reasoning="test", evidence=[])
        t1 = vm.store_fact("The sky is blue", v, "test")
        t2 = vm.store_fact("The sky is blue", v, "test")
        assert t1.trace_id == t2.trace_id  # same trace returned, not duplicated
        assert len(vm.traces) == 1

def test_knowledge_graph_edge_weight_persists(tmp_path):
    """strengthen_edge() should persist to disk."""
    from aura.tools.knowledge_graph import KnowledgeGraphTool
    kg = KnowledgeGraphTool(data_dir=tmp_path)
    n1 = kg.add_node("test_node_a", {"label": "A"})
    n2 = kg.add_node("test_node_b", {"label": "B"})
    e = kg.add_edge(n1, n2, "related")
    original_weight = kg.edges[e].weight
    kg.strengthen_edge(e, 0.2)
    # Reload from disk
    kg2 = KnowledgeGraphTool(data_dir=tmp_path)
    assert kg2.edges[e].weight > original_weight
```

**test_injection_guards.py — verify injection fixes:**
```python
def test_shell_executor_blocks_chaining():
    from aura.tools.shell_executor import ShellExecutor
    executor = ShellExecutor()
    result = executor.execute("echo hello && del important.txt")
    assert result["success"] == False
    assert "disallowed" in result["error"].lower()

def test_database_tool_blocks_drop():
    from aura.tools.database_tool import DatabaseTool
    db = DatabaseTool()
    result = db.query("DROP TABLE users")
    assert result["success"] == False

def test_run_math_blocks_import():
    from aura.tools.code_executor import CodeExecutor
    ex = CodeExecutor()
    result = ex.run_math("__import__('os').system('echo pwned')")
    assert result["success"] == False
```

**Step 1: Create the three test files with the code above**

**Step 2: Run tests to verify they pass**
```bash
cd D:/apprentice-agent && python -m pytest tests/test_auth.py tests/test_memory_persistence.py tests/test_injection_guards.py -v 2>&1 | head -60
```

Expected: tests that cover the fixed behaviors should pass; pre-existing failures are acceptable as long as the targeted assertions pass.

---

## Final Verification

After all 8 batches are complete, run a full syntax check on all modified files:

```bash
cd D:/apprentice-agent
python -m py_compile \
  aura/tools/shell_executor.py \
  aura/tools/system_control.py \
  aura/tools/database_tool.py \
  aura/tools/tool_builder.py \
  aura/tools/synapseforge.py \
  aura/hooks.py \
  aura/tools/email_tool.py \
  aura/tools/code_executor.py \
  api/auth.py \
  api/routes/proactive.py \
  api/routes/multi_agent.py \
  api/routes/self_improvement.py \
  api/routes/memory.py \
  api/routes/chat.py \
  aura/truth_spine.py \
  aura/tools/amem.py \
  aura/tools/knowledge_graph.py \
  aura/tools/hybrid_amem.py \
  aura/tools/local_rag.py \
  aura/tools/mcts_reasoning.py \
  aura/tools/reasoning_tree_tool.py \
  aura/consciousness/self_improvement.py \
  aura/tools/reflexion.py \
  api/services/agent_service.py \
  api/middleware.py \
  aura/emotion/alma_engine.py \
  aura/memory/unified_memory.py \
  aura/consciousness/global_workspace.py \
  aura/tools/neurodream.py \
  api/routes/chat.py \
  aura/tools/pdf_reader.py \
  aura/tools/research_tool.py \
  aura/tools/calendar_tool.py \
  aura/identity.py \
  aura/secure_logging.py \
  aura/multi_agent/router.py \
  aura/tools/mirrormind.py \
  aura/tools/inner_monologue.py \
  aura/scheduler.py \
  aura/exceptions.py \
  aura/consciousness/strategy_bandit.py \
  && echo "All files compile OK"
```

Run targeted tests:
```bash
python -m pytest tests/test_auth.py tests/test_memory_persistence.py tests/test_injection_guards.py -v
```
