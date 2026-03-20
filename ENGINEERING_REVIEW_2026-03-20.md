# Engineering Review — 2026-03-20

**Scope**: Full-project audit across all subsystems
**Test suite**: 690 passed, 0 failed (up from 689/1)
**Files modified**: 12
**Approach**: Surgical fixes preserving existing behavior; no rewrites

---

## 1. Project-Wide Issues Found

### Confirmed and Fixed (19 issues)

**Critical Security (7)**:
- Permission tier comparison used string ordering (`"auto" < "blocked" < "prompt"`) — project configs could escalate permissions from PROMPT to AUTO
- Shell allowlist bypass: `python -c "os.system(...)"` and `node -e "..."` passed all checks since only the command name was validated, not arguments
- Path traversal via `startswith()` bypass on custom tool loading — sibling directories like `tools_evil/` passed the check
- WebSocket auth defaulted to disabled (`"false"`) while HTTP auth defaulted to enabled (`"true"`) — unauthenticated WebSocket access
- PKCE verifier returned over HTTP in OAuth login response — defeats the purpose of PKCE
- ChatGPT login endpoint had no auth dependency
- Code sandbox exposed `__aura_stdout_cap__`/`__aura_stderr_cap__` (real `sys.stderr/stdout`) to user code before cleanup
- CLI `/git` and `/diff` commands passed arbitrary args to subprocess with no subcommand allowlist — git `-c` flag enables arbitrary execution

**Critical Bugs (4)**:
- `self.monologue` called without None-guard in `run()`, `_react_step()`, `_react_step_code()` — `AttributeError` crash when inner_monologue tool not loaded
- `think_stream()` appended error fallback string to conversation history — polluting context for subsequent LLM calls
- Provider-prefixed models (e.g., `anthropic:claude-*`) silently routed to local Ollama when API key missing — confusing "model not found" errors
- IPC activity signal sent without auth token — daemon silently rejected all activity signals, causing idle dreams during active use

**Reliability (5)**:
- Background `write_text` futures discarded — disk I/O failures silently lost conversation history
- `_update_snapshot` in WorldModel used shared `.tmp` path — concurrent writes corrupted snapshot
- `_update_snapshot` called outside `_lock` in `run_maintenance` — race with concurrent mutations
- Double-checked lock in `config.py` had outer check without lock — potential duplicate session creation
- `_save_interaction_count` in ALMA engine read-modify-write not protected by any lock

**Logic/Cleanup (3)**:
- `consecutive_failures` counter reset to 0 at threshold 2, making outer loop's threshold 3 unreachable
- `_log_buffer_lock` was plain `Lock` but re-entered from `_save_state` and `close()` — potential deadlock
- `assess_capabilities()` mutated shared `_capabilities` dict without holding `_lock`
- Unused `build_tool_prompt` import in `_think_with_tools_chatgpt`
- Stale test assertion (`test_render_collapsed` expected "expand" text removed from implementation)

### Identified But Not Fixed (documented for future)

**Important — requires design decision**:
- SSRF via DNS rebinding in API tester (`validate_url_scheme` resolves at validation time, HTTP request resolves again)
- `chat()` in `agent_service.py` holds `_agent_lock` for entire LLM response (30-60s blocking)
- `supersede_belief` in WorldModel does non-atomic two-phase DB writes (crash leaves orphaned beliefs)
- `_log_state_change` opens a new SQLite connection on every call inside the lock (inefficient under load)
- `_react_step_code` sends LLM code to executor without `validate_custom_tool_code` (code agent path unguarded)
- `_temporal_grounding` deletes dream queue file non-atomically (concurrent session data loss)
- KG extraction queue `flush()` called from 3 paths, only 1 holds `_kg_queue_lock`

**Low priority / ambiguous**:
- `delete_conversation` TOCTOU: directory deleted before index updated (crash → orphaned index entry)
- WebSocket `_broadcast_json` uses `threading.Lock` in async context (fragile but currently safe)
- `record_recall` endpoint accepts unbounded `memories` list (DoS potential)
- `_handle_agent_command` in CLI bypasses `PermissionManager.check()` for sub-agent spawning

---

## 2. Bugs and Risks Fixed

| # | File | Root Cause | Fix |
|---|------|-----------|-----|
| 1 | `permissions.py:88` | String comparison `"auto" > "prompt"` is False (lexicographic), so escalation check was inverted | Added `_severity()` method returning int ordering; changed comparison to `new_tier._severity() < current._severity()` |
| 2 | `tools_new.py:373-380` | Shell allowlist only checked command name, not arguments | Added `_INTERPRETER_EXEC_FLAGS` dict and post-allowlist check blocking `-c`, `-e`, `--eval` etc. on interpreter commands |
| 3 | `agent.py:1304` | `startswith(str(tools_base))` matched sibling dirs | Replaced with `Path.relative_to()` which raises `ValueError` on escape |
| 4 | `auth.py:65` | WS auth default `"false"` vs HTTP default `"true"` | Changed WS default to `"true"` to match HTTP |
| 5 | `routes/auth.py:100-103` | PKCE verifier returned in HTTP response body | Store verifier server-side keyed by state; removed from response |
| 6 | `routes/auth.py:68` | Login endpoint missing auth dependency | Added `dependencies=[Depends(require_api_key)]` |
| 7 | `code_executor.py:309` | Wrapper internals (`__aura_stdout_cap__`) accessible to user code | Added `del _io, _redirect_stdout, _redirect_stderr` before user code executes |
| 8 | `main.py:1444-1470` | `/git` and `/diff` passed unvalidated args to subprocess | Added safe-subcommand allowlist for `/git`, flag blocking for `/diff` |
| 9 | `agent.py:1659-1660` | `self.monologue` is `None` when tool not loaded | Wrapped all bare `.monologue` calls in `if self.monologue:` guards |
| 10 | `brain.py:2254-2268` | Error fallback appended to `conversation_history` | Added `_all_models_failed` flag; skip history append when set |
| 11 | `brain.py:393-404` | Provider-prefixed model fell through to local Ollama | Added explicit fallback to `MODEL_FAST` with cloud client preference |
| 12 | `main.py:843-849` | IPC activity message had no `token` field | Read `data/ipc_token` and include in message |
| 13 | `agent.py:2186-2193` | `consecutive_failures` reset to 0, preventing outer threshold check | Removed reset; counter now accumulates to trigger abort |
| 14 | `brain.py:1009,1031` | Background write futures discarded | Wrapped in `_bg_write` with try/except + warning log |

---

## 3. Security and Reliability Improvements

| Area | Change |
|------|--------|
| **Permission escalation** | Integer-based severity comparison prevents project configs from weakening permissions |
| **Shell execution** | Interpreter exec-flag blocking closes the `python -c` / `node -e` bypass vector |
| **Custom tool loading** | `Path.relative_to()` is immune to sibling-directory prefix attacks |
| **WebSocket auth** | Consistent default (enabled) across HTTP and WS — no unauthenticated WS access |
| **OAuth PKCE** | Verifier stays server-side; login endpoint requires auth |
| **Code sandbox** | Wrapper internals deleted from namespace before user code runs |
| **CLI commands** | Read-only git subcommand allowlist; dangerous flag blocking |
| **Snapshot persistence** | `tempfile.mkstemp` prevents concurrent snapshot corruption |
| **Lock safety** | `_log_buffer_lock` upgraded to RLock; `assess_capabilities` now under lock; save counter protected |
| **API endpoints** | Input clamping on `days`, `limit` parameters |

---

## 4. Dead Code, Duplication, and Consolidation

| Change | Rationale |
|--------|-----------|
| Removed unused `build_tool_prompt` import from `_think_with_tools_chatgpt` | Dead import at function scope |
| Fixed stale test assertion in `test_render_collapsed` | Test expected "expand" text removed from `render_collapsed()` implementation |

Note: Larger dead-code cleanup (MarkdownStore, unused memory backends, dead consciousness modules) was identified but **not removed** — purpose is unclear and removal risk exceeds benefit without owner confirmation.

---

## 5. Refactors Performed

| Refactor | Benefit |
|----------|---------|
| `assess_capabilities` split into public wrapper + `_assess_capabilities_unlocked` | Thread-safe public API while avoiding double-lock overhead for internal callers |
| `_update_snapshot` switched to `tempfile.mkstemp` | Eliminates shared `.tmp` path race condition |
| Background write wrapped in named function | Enables error logging on I/O failure |

---

## 6. Performance Improvements

| Change | Rationale |
|--------|-----------|
| Removed outer check from double-checked lock in `config.py` | Correctness fix that also simplifies the code path; the lock contention is negligible since `_get_validation_session` is called rarely |

No speculative performance optimizations were made.

---

## 7. Tests Added or Updated

| Test | Change |
|------|--------|
| `test_render_collapsed` | Updated assertion to match actual `render_collapsed` output (was asserting removed "expand" text) |
| Full suite | 690 passed, 0 failed (previously 689/1) |

---

## 8. Documentation Updated

This engineering review document serves as the documentation update. No README or inline comment changes were needed — all fixes are self-documenting through code comments added at the fix sites.

---

## 9. Remaining Risks, Ambiguities, and Recommended Next Steps

### High Priority (recommend fixing soon)

1. **SSRF DNS rebinding in API tester** — `validate_url_scheme` resolves hostname at check time, HTTP client resolves again at request time. Fix requires binding to pre-resolved IP or OS-level DNS blocklist.

2. **`chat()` lock contention** — `AgentService.chat()` holds `_agent_lock` for the full LLM response duration, blocking all other operations. Should mirror `chat_stream()` pattern (hold lock only for setup/teardown).

3. **`_react_step_code` unguarded** — LLM-generated code in code-agent mode bypasses `validate_custom_tool_code`. Should add the same validation as `_handle_direct_code`.

4. **Non-atomic `supersede_belief`** — Two separate DB writes in different connections; crash between them orphans beliefs. Should use single transaction.

5. **KG flush race** — `flush()` called from 3 code paths, only 1 holds `_kg_queue_lock`. Either all callers should acquire the lock, or flush should be idempotent.

### Medium Priority

6. **Dream queue file deletion** — `_temporal_grounding` reads then deletes without file-level lock. Use atomic rename-to-consume pattern.
7. **`_log_state_change`** opens new SQLite connection per call inside the lock. Pass connection as parameter.
8. **Agent.py size** — 5086 lines. Consider splitting tool registration, ReAct loop, and tool execution into separate modules.
9. **300+ silent exception swallows** — Many `except Exception: pass` blocks hide real issues. Audit and add logging.

### Owner Decisions Needed

10. **Dead memory backends** — MarkdownStore, several consciousness modules are wired but never instantiated. Confirm before removal.
11. **`_handle_agent_command`** bypasses `PermissionManager` — Is this intentional for CLI power users, or a gap?
12. **WebSocket auth default** — Changed from `"false"` to `"true"`. If existing deployments rely on unauthenticated WS access, they need to set `AURA_API_AUTH_ENABLED=false` explicitly.

---

## 10. Change Summary

### Files Modified (12)

| File | Changes | Classification |
|------|---------|---------------|
| `aura/core/permissions.py` | Fixed permission tier comparison | Security fix |
| `api/auth.py` | Aligned WS/HTTP auth defaults | Security fix |
| `api/routes/auth.py` | Added auth to login, removed PKCE verifier from response | Security fix |
| `api/routes/tools_new.py` | Interpreter exec-flag blocking, input validation clamping | Security fix, reliability |
| `aura/agent.py` | Monologue None-guards, path traversal fix, consecutive_failures fix | Bug fix, security fix |
| `aura/brain.py` | Provider fallback, stream error history, background write safety, unused import | Bug fix, reliability, cleanup |
| `aura/tools/code_executor.py` | Sandbox namespace cleanup before user code | Security fix |
| `main.py` | Git subcommand allowlist, IPC token inclusion | Security fix, bug fix |
| `aura/consciousness/world_model.py` | Snapshot tempfile, lock-protected maintenance | Reliability fix |
| `aura/consciousness/metacognition.py` | Thread-safe assess_capabilities | Reliability fix |
| `aura/emotion/alma_engine.py` | RLock upgrade, locked save counter | Reliability fix |
| `aura/config.py` | Fixed double-checked lock | Reliability fix |
| `tests/cli/test_disclosure.py` | Updated stale assertion | Test fix |

### Public Behavior Changes

1. **WebSocket auth now defaults to enabled** — deployments relying on unauthenticated WS must set `AURA_API_AUTH_ENABLED=false`
2. **`/git` CLI command restricted to read-only subcommands** — write operations (add, commit, push) blocked
3. **ChatGPT login endpoint now requires API key** — unauthenticated callers will get 401
4. **Shell endpoint blocks `python -c`, `node -e`** — previously allowed interpreter code execution
5. **PKCE verifier no longer returned in login response** — clients that consumed it must be updated to use the callback flow
