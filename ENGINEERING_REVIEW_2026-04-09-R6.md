# Engineering Review — 2026-04-09 Round 6

**Scope:** Security hardening, input validation, thread safety, dead code removal — addressing R5 cataloged items  
**Baseline:** 1338 passed / 1 failed (pre-existing WebSocket test)  
**Result:** 1338 passed / 1 failed — zero regressions  
**Method:** Targeted fixes of R5-cataloged security, reliability, and dead code issues

---

## 1. Project-Wide Issues Found and Fixed

| # | Severity | File | Issue |
|---|----------|------|-------|
| 1 | **CRITICAL** | `chatgpt_login.py` | Hardcoded production API key (`AURA_API_KEY = "i-L5Sh..."`) in committed source file |
| 2 | **HIGH** | `aura/security/tool_validator.py` | `aura.tools.*` import allowlist permits custom tools to import `shell_executor`, `code_executor`, `browser` — bypasses all code safety |
| 3 | **HIGH** | `aura_daemon.py` | IPC auth token file permissions (`chmod 0o600`) silently fails on Windows — token is world-readable |
| 4 | **HIGH** | `aura/brain.py` | `think_with_tools_stream()` missing circuit breaker check — streaming tool calls bypass broken circuit |
| 5 | **MEDIUM** | `api/routes/tools_new.py` | 9 Pydantic request models with unbounded string fields (email body, SQL, task descriptions, etc.) |
| 6 | **MEDIUM** | `api/routes/tools_new.py` | `ShellRunRequest.session_id` accepts any string — no format validation |
| 7 | **MEDIUM** | `api/routes/tools_new.py` | `calendar_remove` path parameter `event_id` passed unvalidated to tool |
| 8 | **MEDIUM** | `api/routes/chat.py` | WebSocket `hand_name` field passed unvalidated to `HandManager.send_command()` |
| 9 | **MEDIUM** | `api/routes/chat.py` | WebSocket `surface` field accepts any string — stored in conversation records |
| 10 | **MEDIUM** | `aura/memory/kg_contradiction.py` | `_contradictions` list accessed from multiple threads without lock |
| 11 | **MEDIUM** | `api/routes/memory.py` | 3 endpoints call blocking SQLite operations directly on async event loop |
| 12 | **LOW** | `aura/tools/__init__.py` | 9 dead AMEM/Sesame stubs (all `= None`, never imported) |
| 13 | **LOW** | `api/routes/features.py` | ~370 lines of dead AMEM/hybrid-memory endpoints (always returned "not available") |

---

## 2. Bugs and Risks Fixed

| File | Root Cause | Fix |
|------|-----------|-----|
| `chatgpt_login.py:27` | Production API key hardcoded in source. Anyone with repo access can authenticate to live server. | Replaced with `os.environ.get("AURA_API_KEY")` + fail-fast if unset |
| `aura/security/tool_validator.py:19` | `ALLOWED_TOOL_IMPORTS` includes `aura.tools` as prefix, which permits `aura.tools.shell_executor`, `aura.tools.code_executor`, etc. Custom tools can import these and execute arbitrary commands. | Added `_BLOCKED_TOOL_SUBMODULES` set (8 dangerous modules) checked before the allowlist in `_is_import_allowed()` |
| `aura_daemon.py:620-623` | `chmod(0o600)` on IPC token file silently fails on Windows. Token file remains world-readable. | Added Windows fallback using `icacls` to restrict to current user only |
| `aura/brain.py:997` | `think_with_tools_stream()` lacks circuit breaker check. When the breaker is open, `think()` and `think_stream()` correctly return degraded responses, but tool-stream calls still attempt (and fail) LLM calls. | Added `_check_think_circuit_breaker()` at top of method, yields error tuple if open |
| `api/routes/memory.py:311,322,340` | `get_recent_memories()`, `search_memories()`, `add_memory()` call blocking SQLite operations directly on the async event loop thread, stalling all concurrent requests. | Wrapped all three in `loop.run_in_executor(None, ...)` |
| `aura/memory/kg_contradiction.py:132` | `_contradictions` list is a shared mutable data structure accessed from `check_for_contradictions`, `supersede`, and `get_*` methods without thread synchronization. | Added `threading.Lock()` and wrapped all 4 access sites |

---

## 3. Security and Reliability Improvements

### Hardcoded Credential Removal (Critical)
`chatgpt_login.py`: The hardcoded API key has been replaced with an environment variable read. The script now fails immediately with a clear error message if `AURA_API_KEY` is not set. **The exposed key must still be rotated server-side.**

### Tool Import Allowlist Hardening (Security)
`aura/security/tool_validator.py`: Added a blocklist of 8 dangerous `aura.tools.*` submodules that custom tools cannot import, even though `aura.tools` is in the allowlist. This blocks:
- `shell_executor` — arbitrary command execution
- `code_executor` — Python code execution
- `browser` — web automation
- `database_tool` — SQL access
- `code_edit` — file modification
- `deploy_tool` — deployment operations
- `system_control`, `windows_control` — OS control

### IPC Token File Permissions (Windows Hardening)
`aura_daemon.py`: When `chmod(0o600)` fails on Windows, the daemon now attempts `icacls` to restrict the token file to the current user only, preventing other local users from reading the IPC auth token.

### Input Validation (9 Pydantic Models)
`api/routes/tools_new.py`: Added `Field(max_length=...)` constraints to all unbounded string fields across 9 request models:
- `AddEventRequest`: title (500), start/end (64), description (5000), location (1000)
- `AddCardRequest`: front/back (10000), deck (200)
- `SendEmailRequest`: to/subject (500), body (100000), cc/bcc (500)
- `ShellRunRequest`: session_id pattern-validated (`^[a-zA-Z0-9_\-]{1,64}$`)
- `AddTaskRequest`: title (500), description (5000), priority (20)
- `UpdateTaskRequest`: task_id (128), title (500), description (5000)
- `SQLQueryRequest`: sql (10000), db (200)
- `CSVImportRequest`: csv_path (512), table (200)
- `TranscribeRequest`: file_path (512), language (10)
- `SaveResearchRequest`: title (500), content (100000)

### Path Parameter Validation
`api/routes/tools_new.py`: `calendar_remove(event_id)` now validates the path parameter against `^[a-zA-Z0-9_\-\.]{1,128}$` before passing to the calendar tool.

### WebSocket Field Validation
`api/routes/chat.py`:
- `hand_name` in `hand_command` messages validated against `^[a-zA-Z0-9_\-]{1,64}$`
- `surface` field whitelisted to `{"web", "telegram", "extension", "cli", "miniapp"}`

---

## 4. Dead Code, Duplication, and Consolidation

| What | Where | Lines Removed | Why Safe |
|------|-------|---------------|----------|
| AMEM/Sesame stubs | `aura/tools/__init__.py:17-26` | 10 | All `= None`, never imported by any module (`grep` confirmed) |
| AMEM endpoints (9) | `api/routes/features.py:847-1131` | ~285 | `get_amem()` always returns `None` — all endpoints return empty/error. Replaced by `/api/memory/*` |
| Hybrid memory endpoints (3) | `api/routes/features.py:1139-1221` | ~83 | `agent.tools.get('hybrid_amem')` always `None` — dead since AMEM removal |
| AMEM model classes (4) | `api/routes/features.py:847-891` | ~45 | Used only by removed endpoints |
| `get_amem` import | `api/routes/features.py:11` | 1 | No longer referenced after endpoint removal |
| Proto-AGI section | `api/routes/features.py:1134-1148` | ~15 | Empty section with only a comment |

**Total: ~440 lines of confirmed dead code removed.**

---

## 5. Refactors Performed and Why

| Refactor | File | Benefit |
|----------|-------|---------|
| Circuit breaker pattern completion | `brain.py` | All three LLM call paths (`think`, `think_stream`, `think_with_tools_stream`) now consistently check the circuit breaker. Previously only `think` did, creating an inconsistency where streaming calls could bypass the protection. |
| Async wrapping of blocking endpoints | `memory.py` | Three endpoints that called SQLite directly on the event loop are now properly wrapped in `run_in_executor`, preventing event loop starvation under concurrent load. |

---

## 6. Performance Improvements

| Optimization | File | Impact |
|-------------|------|--------|
| Removed ~440 lines of dead code | `features.py`, `__init__.py` | Faster module load, smaller attack surface, less code to maintain |
| `run_in_executor` for memory endpoints | `memory.py` | Prevents event loop blocking — concurrent requests no longer serialize behind SQLite operations |

---

## 7. Tests Added or Updated

No new test files added this round. All 1338 existing tests pass. The pre-existing WebSocket test failure (`test_websocket_chat_protocol`) remains unchanged — it's a test infrastructure issue, not a code bug.

---

## 8. Documentation Updated

| File | Change |
|------|--------|
| `api/routes/features.py` | Added tombstone comment explaining AMEM removal and pointing to `/api/memory/*` |
| `ENGINEERING_REVIEW_2026-04-09-R6.md` | This report |

---

## 9. Remaining Risks, Ambiguities, and Recommended Next Steps

### Must-Do (Immediate)
1. **Rotate the exposed API key** — The key `i-L5Sh...` was in committed source. Even though the code now reads from env, the key in git history must be considered compromised. Rotate it on the server.

### Should-Do (Next Round)
1. **Unify auth flags** — `AURA_REQUIRE_AUTH` and `AURA_API_AUTH_ENABLED` control different layers and default differently. Should be one flag.
2. **Fix `X-Forwarded-For` rate limit bypass** — Use rightmost IP (proxy-added) instead of leftmost (attacker-controlled).
3. **Route background LLM calls to `bg_pool`** — `_quick_generate`, `compact_history`, world-model extraction contend with user calls in `llm_pool(4)`.
4. **Fix `inner_thoughts_engine.py` silent exceptions** — 9 consecutive `except Exception: pass` blocks mask errors.

### Could-Do (Future)
1. Add tests for `tools_new.py` endpoints (calendar, flashcard, shell, email)
2. Install `pytest-timeout` to enable test timeouts
3. Wire `sanitize_outgoing()` into Telegram response path
4. Archive stale roadmap files to `docs/archive/`
5. `build.py:18` — Replace `shell=True` with list form for `subprocess.run`

---

## 10. Change Summary

### Files Modified (11 files)
| File | Changes |
|------|---------|
| `chatgpt_login.py` | Removed hardcoded API key, read from env with fail-fast |
| `aura/security/tool_validator.py` | Added `_BLOCKED_TOOL_SUBMODULES` blocklist, updated `_is_import_allowed()` |
| `aura_daemon.py` | Added Windows `icacls` fallback for IPC token file permissions |
| `aura/brain.py` | Added circuit breaker check to `think_with_tools_stream()` |
| `api/routes/tools_new.py` | Added `Field(max_length=...)` to 9 request models, validated `event_id` path param |
| `api/routes/chat.py` | Validated `hand_name` and whitelisted `surface` in WebSocket handler |
| `aura/memory/kg_contradiction.py` | Added `_contradictions_lock` to all 4 access sites |
| `api/routes/memory.py` | Wrapped 3 blocking endpoints in `run_in_executor` |
| `aura/tools/__init__.py` | Removed 9 dead AMEM/Sesame stubs |
| `api/routes/features.py` | Removed ~430 lines of dead AMEM/hybrid-memory endpoints + models |
| `ENGINEERING_REVIEW_2026-04-09-R6.md` | This report |

### Public Behavior Changes
- **Removed endpoints:** `/api/amem/*` (9 endpoints) and `/api/hybrid-memory/*` (3 endpoints). These always returned empty/error since AMEM was removed. No frontend references them.
- **Stricter validation:** Requests with oversized strings or invalid formats that previously were accepted will now be rejected with 422 Validation Error. This is a security improvement, not a regression.
- All other changes are internal (bug fixes, thread safety, cleanup).
