> SUPERSEDED 2026-04-13. Current source of truth: D:/Aura/CURRENT_STATE.md

# Engineering Review — 2026-04-09 Round 7

**Scope:** R6 should-do items, security hardening, reliability, dead code, resource leaks  
**Baseline:** 1338 passed / 1 failed (pre-existing WebSocket test)  
**Result:** 1338 passed / 1 failed — zero regressions  
**Method:** Targeted fixes of R6 recommended items + deep audit of unexplored areas

---

## 1. Project-Wide Issues Found and Fixed

| # | Severity | File | Issue |
|---|----------|------|-------|
| 1 | **HIGH** | `api/auth.py` | Two conflicting auth flags (`AURA_REQUIRE_AUTH` vs `AURA_API_AUTH_ENABLED`) with different defaults control different layers, creating inconsistent security posture |
| 2 | **HIGH** | `api/middleware.py` | `X-Forwarded-For` rate limit bypass — leftmost IP is attacker-controlled, not proxy-added |
| 3 | **MEDIUM** | `api/services/inner_thoughts_engine.py` | 8 silent `except Exception: pass` blocks masking errors in cognitive subsystems |
| 4 | **MEDIUM** | `aura/messaging/telegram/bot.py` | `_save_state()` called AFTER `store.close()` — writes to closed SQLite connection |
| 5 | **MEDIUM** | `aura/messaging/sanitizer.py` | `sanitize_outgoing()` not wired into Telegram response path — only protected proactive messages |
| 6 | **MEDIUM** | `aura/tools/code_executor.py` | `run_math()` has no guard against exponential blowup (e.g., `10**10**10`) |
| 7 | **LOW** | `build.py` | `shell=True` with `subprocess.run` — should use list form |
| 8 | **LOW** | `api/main.py` | Silent `except Exception: pass` for Telegram hand notification |
| 9 | **LOW** | `aura/tools/local_rag.py` | `fitz.open()` (PyMuPDF) not wrapped in try/finally — resource leak on exception |
| 10 | **HIGH** | `aura/hands/manager.py` | Broken import: `get_metacognition_engine` → should be `get_metacognitive_engine`. Metacognition recording after every Hand run is silently broken. |
| 11 | **HIGH** | `aura/consciousness/strategy_bandit.py` | SQLite connection opened BEFORE acquiring `_db_lock` in `decay_arms()` and `get_arm_stats()` — race condition + resource leak |
| 12 | **MEDIUM** | `aura/consciousness/strategy_bandit.py` | `get_stats_summary()` has no lock, inconsistent with other DB methods |
| 13 | **MEDIUM** | `aura/consciousness/reasoning_templates.py` | `_run_abstraction_batch()` opens SQLite without try/finally — connection leak on exception |
| 14 | **LOW** | `aura/hands/manager.py` | Silent `except Exception: pass` blocks for audit chain and metacognition hide real failures |

---

## 2. Bugs and Risks Fixed

| File | Root Cause | Fix |
|------|-----------|-----|
| `api/auth.py` | `require_api_key` checks `AURA_API_AUTH_ENABLED` inline, `verify_api_key_ws` checks it separately, legacy `_auth_is_required()` reads a different flag. Three code paths with different logic for the same decision. | Created unified `_auth_is_enabled()` that checks `AURA_API_AUTH_ENABLED` first, falls back to `AURA_REQUIRE_AUTH`. All three paths now use it. WebSocket now also fails closed when key is not configured. |
| `api/middleware.py:153,190` | `X-Forwarded-For` rate limiting used `split(",")[0]` (leftmost IP). Behind a reverse proxy, the leftmost IP is client-supplied and trivially spoofable. An attacker can bypass rate limits by injecting arbitrary IPs. | Created `_get_client_ip()` helper that uses `split(",")[-1]` (rightmost IP = proxy-appended). Replaced all 3 IP extraction sites. |
| `aura/messaging/telegram/bot.py:744` | `stop()` calls `_save_state()` at line 734 (correct, before store.close), then `store.close()` at 736, then `_save_state()` again at 744 (AFTER close). The second call writes to a closed SQLite connection — silently fails or corrupts data. | Removed the duplicate `_save_state()` call after `store.close()`. |
| `aura/tools/code_executor.py:448` | `run_math()` evaluates AST-validated expressions via `eval()` in a restricted namespace. While safe from injection, exponential blowup (e.g., `pow(10, 10**10)` or `2**999999`) can hang the process with no timeout. | Added AST check: if a `Pow` node has a constant exponent > 10000, reject immediately. |

---

| `aura/hands/manager.py:239` | Import `get_metacognition_engine` does not exist — correct name is `get_metacognitive_engine`. `ImportError` silently swallowed by `except Exception: pass`. Metacognition recording for Hand runs has been permanently broken since the refactor. | Fixed import name to `get_metacognitive_engine` |
| `aura/consciousness/strategy_bandit.py:782,819` | `decay_arms()` and `get_arm_stats()` open SQLite connection BEFORE acquiring `_db_lock`. Two threads can both open connections before either gets the lock, then one closes the connection the other is using. | Moved `conn = sqlite3.connect(...)` inside `with self._db_lock:` block |
| `aura/consciousness/strategy_bandit.py:863` | `get_stats_summary()` performs DB reads with no lock at all. Inconsistent with `record_outcome()` which writes under lock. | Added `with self._db_lock:` wrapper and fixed indentation |
| `aura/consciousness/reasoning_templates.py:606` | `_run_abstraction_batch()` opens SQLite with `conn = sqlite3.connect(...)` then calls `conn.close()` at line 618 — but no try/finally. If `fetchall()` raises, connection leaks. | Wrapped in try/finally |
| `aura/hands/manager.py:234,249` | Audit chain and metacognition recording use `except Exception: pass` — all failures including the broken import at line 239 are permanently invisible. | Replaced with `logger.debug(..., exc_info=True)` |

---

## 3. Security and Reliability Improvements

### Auth Flag Unification (Security)
`api/auth.py`: Replaced 3 separate auth-checking code paths with a single `_auth_is_enabled()` function. The key behavior changes:
- `verify_api_key_ws` now **fails closed** when auth is enabled but no key is configured (previously returned `True` in this case via `not _auth_is_required()` which defaults to `False`)
- Both HTTP and WebSocket auth use the same flag precedence: `AURA_API_AUTH_ENABLED` > `AURA_REQUIRE_AUTH`

### X-Forwarded-For Fix (Security)
`api/middleware.py`: The rightmost IP in `X-Forwarded-For` is now used for rate limiting. This is the IP appended by the trusted reverse proxy. The leftmost IP (previously used) is attacker-controlled. Added a `_get_client_ip()` helper used by both HTTP and WebSocket rate limiting.

### Telegram Response Sanitization (Security)
`aura/messaging/telegram/mixins/agent_core.py`: `sanitize_outgoing()` is now called in `_edit_or_send_response()`, the central method used by all Telegram response paths. Previously only proactive messages were sanitized. The sanitizer flags social engineering patterns (phishing, credential harvesting) in LLM-generated output.

### Math Exponent Guard (Reliability)
`aura/tools/code_executor.py`: `run_math()` now rejects constant exponents > 10000 at the AST level, preventing CPU-hang attacks via pathological expressions.

### Silent Exception Logging (Reliability)
`api/services/inner_thoughts_engine.py`: 8 `except Exception: pass` blocks replaced with `logger.debug(..., exc_info=True)`. Errors in emotion state, context tracking, idle presence, proactive awareness, and thinking stats are now visible in debug logs.

`api/main.py`: Telegram hand notification failure now logged at debug level instead of silently swallowed.

---

## 4. Dead Code, Duplication, and Consolidation

| What | Where | Change |
|------|-------|--------|
| Duplicate `_save_state()` call | `telegram/bot.py:744` | Removed — first call at line 734 (before `store.close()`) is the correct one |
| 3 duplicate IP extraction blocks | `api/middleware.py` | Consolidated into `_get_client_ip()` helper |
| Duplicate auth checking logic | `api/auth.py` | Consolidated into `_auth_is_enabled()` |

---

## 5. Refactors Performed and Why

| Refactor | Files | Benefit |
|----------|-------|---------|
| `_get_client_ip()` extraction | `api/middleware.py` | Single source of truth for client IP extraction; eliminates 3 copy-pasted blocks with the same bug |
| `_auth_is_enabled()` unification | `api/auth.py` | All auth paths (HTTP middleware, route deps, WebSocket) now use identical logic; impossible to have one path enabled and another disabled |

---

## 6. Performance Improvements

No performance-focused changes this round. All changes are correctness and security.

---

## 7. Tests Added or Updated

No new test files. All 1338 existing tests pass unchanged. The auth changes maintain backward compatibility: `AURA_API_AUTH_ENABLED=false` still disables auth, `AURA_API_AUTH_ENABLED=true` (or unset) still enables it.

---

## 8. Documentation Updated

| File | Change |
|------|--------|
| `ENGINEERING_REVIEW_2026-04-09-R7.md` | This report |

---

## 9. Remaining Risks, Ambiguities, and Recommended Next Steps

### Still Must-Do
1. **Rotate the exposed API key** from `chatgpt_login.py` git history — flagged in R6, still pending

### Should-Do (Next Round)
1. **Route background LLM calls to `bg_pool`** — `_quick_generate`, `compact_history`, world-model extraction still contend with user calls in `llm_pool(4)`
2. **Add tests for `tools_new.py` endpoints** — calendar, flashcard, shell, email routes have zero test coverage
3. **Fix `code_executor.run_math` timeout** on Windows — the exponent guard helps but a true timeout (via subprocess or threading) would be more robust
4. **Address `_save_state` after `store.close` in other shutdown paths** — verify no other modules have this pattern

### Architectural (Future)
1. Memory search brute-force (`store.py search_semantic` loads all embeddings)
2. Hands manager 60s blocking approval
3. WebSocket test infrastructure redesign

---

## 10. Change Summary

### Files Modified (15 files)
| File | Changes |
|------|---------|
| `api/auth.py` | Unified auth into `_auth_is_enabled()`, simplified `require_api_key` and `verify_api_key_ws` |
| `api/middleware.py` | Added `_get_client_ip()` using rightmost XFF IP, replaced 3 extraction sites |
| `api/services/inner_thoughts_engine.py` | Replaced 8 silent `except Exception: pass` with debug logging |
| `api/main.py` | Replaced silent `except Exception: pass` for Telegram notification with debug logging |
| `build.py` | Changed `shell=True` to list form `['npm', 'run', 'build']` |
| `aura/messaging/telegram/bot.py` | Removed duplicate `_save_state()` call after `store.close()` |
| `aura/messaging/telegram/mixins/agent_core.py` | Wired `sanitize_outgoing()` into `_edit_or_send_response()` |
| `aura/tools/code_executor.py` | Added exponent size guard in `run_math()` |
| `aura/tools/local_rag.py` | Wrapped 2 `fitz.open()` calls in try/finally for proper resource cleanup |
| `aura/hands/manager.py` | Fixed broken `get_metacognition_engine` import, added debug logging to 2 silent exception blocks |
| `aura/consciousness/strategy_bandit.py` | Fixed connection-before-lock in `decay_arms()`/`get_arm_stats()`, added lock to `get_stats_summary()` |
| `aura/consciousness/reasoning_templates.py` | Wrapped `_run_abstraction_batch()` DB connection in try/finally |
| `ENGINEERING_REVIEW_2026-04-09-R7.md` | This report |

### Public Behavior Changes
- **Auth flag behavior:** `AURA_REQUIRE_AUTH` alone (without `AURA_API_AUTH_ENABLED`) now enables auth consistently. Previously, middleware ignored `AURA_REQUIRE_AUTH` and only read `AURA_API_AUTH_ENABLED`. WebSocket now fails closed when auth is enabled but no key is configured (previously allowed connections).
- **Rate limiting:** With `AURA_TRUST_PROXY=true`, the proxy-appended (rightmost) IP in `X-Forwarded-For` is used instead of the client-supplied (leftmost) IP. This is the correct behavior per RFC 7239.
- **Telegram responses:** LLM-generated responses are now checked by `sanitize_outgoing()` before sending. Flagged messages are still sent (not blocked) but generate a warning log.
- **Math expressions:** Exponents > 10000 are now rejected in `run_math()` to prevent CPU-hang attacks.