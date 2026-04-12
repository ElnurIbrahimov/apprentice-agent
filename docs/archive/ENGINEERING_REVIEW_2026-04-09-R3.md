> SUPERSEDED 2026-04-13. Current source of truth: D:/Aura/CURRENT_STATE.md

# Engineering Review — 2026-04-09 Round 3

**Scope:** Full-project audit of Aura v4.7.0 (~200+ Python files, 1338 tests)  
**Methodology:** Automated analysis (ruff, pytest) + 5 parallel deep-exploration agents + manual code review of all critical paths  
**Baseline:** 1337 tests passing, 1646 ruff errors, 6 runtime crashers, 7 dangling async tasks, 3 critical security gaps

---

## 1. Project-Wide Issues Found

### Confirmed Issues (Fixed)
- **6 runtime crashers** (F821/syntax): Undefined names in 5 files + f-string syntax error that would cause NameError/SyntaxError at runtime
- **7 dangling asyncio tasks** (RUF006): `create_task`/`ensure_future` results not stored — tasks silently GC'd before completion
- **1 closure-over-loop-variable bug** (B023): WebSocket streaming thread captured loop variables unsafely
- **1 shared pool shutdown bug**: EventBus.shutdown() killed the shared `bg_pool()`, breaking background tasks for the entire application
- **1 stale test assertion**: Command registry count was 57 but actual was 58
- **102 empty f-strings** (F541): `f"string"` with no placeholders — unnecessary overhead
- **8 unused exception variables** (F841): `except Exception as e:` where `e` was never used
- **1 parameter shadowing import** (F811): `field` parameter shadowed `dataclasses.field` import
- **1 incomplete env passthrough**: Shell executor missing Windows env vars needed for npm/node tools

### Lower-Confidence Concerns (Not Changed)
- **40 unused local variables** (F841): Most appear intentional (computed but not yet used) or are in complex functions where removal needs manual review
- **84 mutable class defaults** (RUF012): `list`/`dict` class attributes — most are intentionally mutable shared state, not accidental
- **98 raise-without-from** (B904): `raise X` inside `except` blocks — lower priority, doesn't affect behavior
- **17 B023 closure warnings** in `api/routes/chat.py`: Most are in `await` contexts where the loop doesn't continue before the closure executes

### Security Assessment
- **Authentication**: Solid. Timing-safe comparison (`secrets.compare_digest`), fail-closed when misconfigured, WebSocket auth handled separately
- **Shell execution**: Well-sandboxed. Injection detection, command allowlisting, env sanitization, sandbox routing for dangerous commands
- **File system**: Proper sandbox enforcement with symlink resolution, null byte blocking, and race condition mitigation
- **Code execution**: Three-tier sandbox (Monty → E2B → subprocess) with AST validation before all tiers
- **SQL**: Parameterized queries throughout. FTS5 input sanitized
- **Uploads**: Filename sanitization, extension whitelist, streaming size enforcement, UUID storage names
- **API keys**: Not in git, `.env` in `.gitignore`, weak key detection in config validation
- **No `os.system()` calls**, no `shell=True` without validation

---

## 2. Bugs and Risks Fixed

### Runtime Crashers (would cause NameError/SyntaxError when hit)

| File | Issue | Fix |
|------|-------|-----|
| `api/routes/multi_model.py:78` | `HTTPException` not imported | Added to FastAPI import |
| `api/routes/status.py:217` | `os` not imported | Added `import os` |
| `api/routes/tools_new.py:479` | `os` not imported | Added `import os` |
| `aura/cli/commands/tool_commands.py:50` | `get_ctx` undefined | Added `from aura.cli.context import get_ctx` |
| `aura/tools/crypto_price.py:147` | f-string syntax error (`{{` dict inside f-string) | Fixed with space: `{ {dict}.get(...)}` |
| `aura/brain.py:510` | `atexit` re-imported (shadowed module-level import) | Removed redundant local import |

### Dangling Async Tasks (silently cancelled by GC)

Created `fire_and_forget()` utility in `aura/pools.py` using the standard Python pattern (`_background_tasks` set + `add_done_callback(discard)`). Applied to 7 locations:

- `aura/agent.py:1470` — gateway daemon shutdown
- `aura/channels/channel_bridge.py:161` — channel stop
- `aura/core/conversation_manager.py:342` — async event listeners
- `aura/hands/manager.py:174` — action trace broadcast
- `aura/messaging/telegram/mixins/misc.py:757` — hand execution
- `aura/messaging/telegram/mixins/research.py:90` — progress updates
- `aura/messaging/telegram/mixins/scheduling.py:106` — reminder send

### Closure Bug in WebSocket Streaming

`api/routes/chat.py`: `stream_worker()` captured `message`, `model_override`, `action_mode`, `routing_opts`, `loop`, and `chunk_queue` from the enclosing WebSocket loop iteration. Bound them as default arguments to prevent cross-iteration capture.

### Shared Pool Destruction Bug

`aura_daemon.py:EventBus.shutdown()` called `self._pool.shutdown(wait=False)` on the shared `bg_pool()`. This would kill background tasks for the entire application when the EventBus was shut down. Fixed to only clear handlers without touching the shared pool.

---

## 3. Security and Reliability Improvements

### SSRF IPv4-mapped IPv6 Bypass (CRITICAL)
`aura/security/ssrf_guard.py`: IPv4-mapped IPv6 addresses like `::ffff:127.0.0.1` completely bypassed all SSRF checks. Python's `ipaddress.IPv6Address.is_loopback` returns `False` for these, and IPv4 network checks don't match IPv6 types. **Fix:** Added recursive check via `addr.ipv4_mapped` — if an IPv6 address has an IPv4-mapped form, it's validated against the IPv4 rules.

### Tool Validator Wildcard Import (CRITICAL)
`aura/security/tool_validator.py`: The `ALLOWED_TOOL_IMPORTS` set contained `"aura"` which allowed custom tools to import ANY aura module — including `aura.security.tool_signing` (read signing keys), `aura.config` (read API keys), `aura.hands.manager` (invoke autonomous actions). **Fix:** Changed to `"aura.tools"` and rewrote import validation to use prefix matching (`_is_import_allowed()`). Now `import aura.tools.browser` works but `import aura.security` is blocked.

### Tool Validator Missing setattr Block (CRITICAL)
`aura/security/tool_validator.py`: `setattr` was not in the blocked call list, allowing custom tools to monkey-patch live objects (e.g., disable security checks, modify API keys). **Fix:** Added `"setattr"` to the blocked calls set.

### API CORS Fallback Hardened
`api/main.py`: When config loading failed, CORS silently fell back to `allow_origins=["*"]` (full wildcard). **Fix:** Fallback now defaults to `localhost:5173` only, with a WARNING-level log.

### API Auth Middleware Failure Visibility
`api/main.py`: Auth/rate-limit middleware setup failure was logged at WARNING level and silently continued with no protection. **Fix:** Changed to ERROR level with `exc_info=True` and clear "server may be unprotected" message.

### UnifiedMemory numpy Guard
`aura/memory/unified_memory.py`: `_get_embedding()` would crash with `AttributeError` if numpy was not installed (np was None). **Fix:** Added `if np is None: return None` guard.

### Shell Executor Environment Passthrough
Added missing Windows environment variables to `_SAFE_ENV_KEYS` in `aura/tools/shell_executor.py`:
- `APPDATA`, `LOCALAPPDATA`, `PROGRAMFILES`, `PROGRAMFILES(X86)`, `WINDIR`, `USERNAME`, `HOMEDRIVE`, `HOMEPATH` — required for npm/node and many Windows tools
- `LC_ALL`, `LC_CTYPE`, `PYTHONIOENCODING`, `TERM` — locale/encoding

Without these, shell commands run by the agent would fail silently or produce encoding errors on Windows.

### GC Protection for Background Tasks
The `fire_and_forget()` pattern prevents Python's garbage collector from collecting unfinished async tasks — a well-known source of "silent failures" in asyncio applications.

---

## 4. Dead Code and Cleanup

| Change | Count | Details |
|--------|-------|---------|
| Empty f-strings removed | 102 | `f"literal"` → `"literal"` across 4 files |
| Unused exception vars removed | 8 | `except Exception as e:` → `except Exception:` |
| Unused loop vars cleaned | 3 | `loop = asyncio.get_running_loop()` → `asyncio.get_running_loop()` |
| Shadowing parameter renamed | 1 | `field` → `field_name` in task_manager.py |

---

## 5. Refactors Performed

### `fire_and_forget()` Utility (aura/pools.py)
Centralized the "background async task with GC protection" pattern into a single utility. This replaces 7 ad-hoc `create_task`/`ensure_future` calls with a consistent, safe pattern. The utility lives in `pools.py` which already manages thread pools, making it the natural home for async task lifecycle management.

### No major structural refactors
The codebase architecture is sound. The mixin decomposition, pool consolidation, and module boundaries are well-structured. No architectural changes were needed.

---

## 6. Performance Improvements

No speculative performance changes made. The codebase doesn't show obvious hot-path performance issues. The 102 empty f-string removals avoid trivially unnecessary string formatting overhead but this is negligible.

---

## 7. Tests

- **Before:** 1337 passed, 2 failed
- **After:** 1338 passed, 1 failed
- **Fixed:** `test_registry_count` assertion updated from 57 → 58 (stale after new command was added)
- **Remaining failure:** `test_websocket_chat_protocol` — requires running server, integration test, pre-existing

---

## 8. Documentation

No README or documentation changes needed. The codebase is well-documented with clear module docstrings and inline comments where appropriate.

---

## 9. Remaining Risks, Ambiguities, and Recommended Next Steps

### Critical Open Items (from deep exploration agents)

**Security — Still Open:**
1. **`code_agent.py` AST sandbox escape** — MRO chain access to builtins cannot be fully prevented via AST checking. Real fix requires E2B or Docker isolation. (Carried from R1/R2)
2. **`tool_signing.py` HMAC fallback** — signing key == verification key when PyNaCl is not installed. Any process with key file access can forge signatures.
3. **`tool_signing.py` algorithm field** — `.sig` files contain an attacker-controlled `algorithm` field that determines verification path. A downgrade or DoS attack is possible.
4. **`ssrf_guard.py` DNS resolution** — creates a new `ThreadPoolExecutor` per DNS call. Under load this exhausts OS thread limits.
5. **`taint_tracker.py` enforcement** — taint tracking is advisory-only. No enforcement mechanism forces callers to check before writing to sinks.
6. **`api/main.py` missing security headers** — no CSP, HSTS, X-Frame-Options, X-Content-Type-Options on any response.

**Reliability — Still Open:**
7. **`agent.shutdown()` never called in CLI mode** — KG and skill library may not flush on exit. Only `brain.close()` is registered via atexit.
8. **`bot.py` `_last_exchange` dict** — initialized but never written to; reads always return `{}`. Silent correctness bug in Telegram responses.
9. **`telegram_store.py` `reaction_feedback` table** — unbounded growth, no cleanup logic (unlike group_messages and inline_cache which are capped).
10. **`telegram_store.py` `migrate_from_json`** — runs on every restart, silently overwrites timestamps via INSERT OR REPLACE.
11. **`store.py` `search_semantic`** — loads ALL embeddings into RAM for linear scan. Will OOM on large databases.
12. **`brain.py` `_load_history`** — catches `(json.JSONDecodeError, IOError)` but not `UnicodeDecodeError`. Corrupted UTF-8 in history file would crash init.
13. **`config.py` OLLAMA_HOST uppercase scheme** — `HTTP://host` (uppercase) fails the startswith check and silently reverts to localhost.
14. **`hands/manager.py` `request_approval()`** — blocks thread for 60s, can exhaust the 3-thread pool.

**Code Quality — Still Open:**
15. **40 unused local variables** (F841) — need manual review
16. **98 raise-without-from-except** (B904) — `raise HTTPException()` inside `except` blocks
17. **pools.py docstring** — says llm_pool has 12 workers but actual is 4

### Recommended Next Steps (Priority Order)
1. **Install PyNaCl** for Ed25519 tool signing (eliminates HMAC fallback weakness)
2. **Pin algorithm server-side** in tool_signing.py (don't read from untrusted .sig files)
3. **Add security headers middleware** to api/main.py (CSP, HSTS, X-Frame-Options)
4. **Register `agent.shutdown()` via atexit** in main.py for CLI mode
5. **Add UnicodeDecodeError catch** to brain.py `_load_history()`
6. **Cap `reaction_feedback` table** in telegram_store.py (like group_messages)
7. **Replace per-call ThreadPoolExecutor** in ssrf_guard.py with shared pool
8. **Add vector index** to memory store for semantic search (sqlite-vss or FAISS)

---

## 10. Change Summary

### Files Modified (by this review)

**Critical security fixes:**
- `aura/security/ssrf_guard.py` — IPv4-mapped IPv6 SSRF bypass closed
- `aura/security/tool_validator.py` — wildcard `aura` import restricted to `aura.tools`, `setattr` blocked, prefix-based import check
- `api/main.py` — CORS fallback hardened, auth failure logging elevated
- `aura/memory/unified_memory.py` — numpy None guard

**Core fixes (runtime crashers):**
- `api/routes/multi_model.py` — added missing `HTTPException` import
- `api/routes/status.py` — added missing `os` import
- `api/routes/tools_new.py` — added missing `os` import
- `aura/cli/commands/tool_commands.py` — added missing `get_ctx` import
- `aura/tools/crypto_price.py` — fixed f-string syntax error
- `aura/brain.py` — removed duplicate `atexit` import

**Async task safety:**
- `aura/pools.py` — added `fire_and_forget()` utility
- `aura/agent.py` — use `fire_and_forget()`
- `aura/channels/channel_bridge.py` — use `fire_and_forget()`
- `aura/core/conversation_manager.py` — use `fire_and_forget()`
- `aura/hands/manager.py` — use `fire_and_forget()`
- `aura/messaging/telegram/mixins/misc.py` — use `fire_and_forget()`
- `aura/messaging/telegram/mixins/research.py` — use `fire_and_forget()`
- `aura/messaging/telegram/mixins/scheduling.py` — use `fire_and_forget()`

**Bug fixes:**
- `api/routes/chat.py` — bound closure variables in stream_worker
- `aura_daemon.py` — EventBus.shutdown() no longer kills shared pool
- `aura/tools/shell_executor.py` — added missing env vars for Windows tools
- `aura/tools/task_manager.py` — renamed shadowing parameter

**Cleanup (auto-fixed by ruff):**
- 102 empty f-strings across multiple files
- 8 unused exception variables across multiple files

**Test fixes:**
- `tests/cli/test_command_registry.py` — updated stale count assertion

### Public Behavior Changes
- **Shell executor**: Now passes additional Windows env vars (`APPDATA`, `LOCALAPPDATA`, etc.) to child processes. This is a **fix** — tools that depend on these vars will now work correctly.
- **EventBus.shutdown()**: No longer kills the shared bg_pool. Other subsystems using bg_pool will continue to function after EventBus shutdown.
- All other changes are internal (no public API changes).

### Ruff Error Reduction
- **Before:** 1646 errors
- **After:** 1496 errors
- **Reduced by:** 150 errors (9.1%)
- **Critical errors (F821) remaining:** 2 (both are string type annotations, not runtime errors)