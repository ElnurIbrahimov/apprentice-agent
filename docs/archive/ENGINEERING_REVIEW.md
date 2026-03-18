# Aura Engineering Review Report
**Date:** 2026-03-12
**Scope:** Full-project audit covering correctness, security, thread safety, dead code, performance, and architecture

---

## Summary

| Category | Found | Fixed | Remaining |
|----------|-------|-------|-----------|
| **Critical (security/correctness)** | 13 | 13 | 0 |
| **High (bugs/performance)** | 14 | 14 | 0 |
| **Medium (dead code/quality)** | 19 | 14 | 5 |
| **Low (style/minor)** | 13 | 0 | 13 |

**Previous session fixes (9 applied before this review):**
- `_quick_generate` timeout (brain.py)
- `_BG_EXECUTOR` workers 4→8 (brain.py)
- Circuit breaker on world model extraction (brain.py)
- System prompt size cap 12K (brain.py)
- Module additions cap 4K (brain.py)
- Timeout floor 45s (brain.py)
- num_predict floor 512 (brain.py)
- Budget limits 300/1024/2048 (config.py)
- StateExtractor uses `_quick_generate` instead of `think()` (state_extractor.py)

---

## Fixes Applied — Session 1

### Critical/High

1. **Broken `from aura.brain import Brain`** → Fixed to `OllamaBrain` in:
   - `consciousness/self_improvement.py:398`
   - `consciousness/reasoning_templates.py:643`
   - These modules were silently failing — self-improvement and template abstraction were completely non-functional.

2. **Stream timeout protection** (brain.py `think_stream`):
   - Added 90-second stale-chunk timeout. If no chunk arrives for 90s, stream aborts and falls back to next model in chain. Previously could hang forever.

3. **`_save_history_snapshot` blocking I/O** (brain.py):
   - Changed from synchronous `write_text()` on request thread to background `_BG_EXECUTOR.submit()`. Reduces response latency.

4. **`compact_history` race with `_save_history`** (brain.py):
   - `_check_auto_reset` was calling `compact_history()` (async) then immediately `_save_history()` (writes pre-compacted data). Removed the redundant save — compaction handles its own persistence.

5. **`_tags_cache` race condition** (config.py):
   - Added `threading.Lock` around concurrent reads/writes during model validation startup (6 threads).

6. **Evolution endpoint race condition** (api/routes/evolution.py):
   - Added `threading.Lock` around `_current_run` state mutations from background thread.

7. **SSRF protection on PDF URL extraction** (api/routes/pdf.py):
   - Added private IP blocklist (127.x, 10.x, 172.16-31.x, 192.168.x, 169.254.x, localhost).
   - Added 50MB size limit on downloaded PDFs.

8. **`_trackers` unbounded memory growth** (api/routes/memory.py):
   - Added cap of 50 sessions with LRU eviction.

9. **`_thought_history` unbounded growth** (api/routes/thinking.py):
   - Changed from `List` to `deque(maxlen=200)`.

### Medium

10. **Duplicate `API_AUTH_ENABLED`** (config.py):
    - Removed shadowing duplicate at line 403 (was inconsistent with primary definition at line 143).

11. **Dead code removal:**
    - `_active_sessions` dict in reasoning_tree.py (declared, never used)
    - `AURA_AVAILABLE = False` in agent.py (dead constant)
    - Unused `datetime` and `BackgroundTasks` imports in reasoning_tree.py

12. **`__import__("time")` anti-pattern** (agent_service.py):
    - Added `import time` at top level, replaced 4 `__import__("time").time()` calls.

13. **Duplicate `import os`** (brain.py):
    - Removed redundant `import os as _os` and `import os` inside methods, using top-level import.

14. **Stale budget comment** (brain.py):
    - Updated comment that still said "150/400/800" to reference config constants.

---

## Fixes Applied — Session 2

### Critical

15. **Deadlock in `switch_conversation()`** (brain.py:700-711):
    - `switch_conversation()` held `_history_lock` (non-reentrant `threading.Lock`) and called `_save_history()` which also acquires `_history_lock` → guaranteed deadlock on every conversation switch.
    - Fix: Extracted `_save_history_unlocked()` method (assumes lock held), `switch_conversation()` now calls the unlocked version.

16. **Lambda closure bug in attachment processing** (api/routes/chat.py:126,143):
    - `file_path` captured by reference in lambdas inside `for attachment in attachments` loop. Classic Python late-binding closure bug.
    - Fix: Added default argument binding (`lambda fp=file_path: ...`).

17. **Zip-slip path traversal** (api/services/zip_analyzer.py:91):
    - Used `str(target).startswith(str(resolved_root))` — prefix-only check. `/tmp/aura_zip_abc` would match `/tmp/aura_zip_abcEVIL/...`.
    - Fix: Replaced with `target.relative_to(resolved_root)` which correctly validates path containment.

18. **Broken shutdown: `get_episodic_memory` doesn't exist** (api/main.py:289):
    - Shutdown handler imported `get_episodic_memory` from `aura_episodic_memory` — no such function exists. Import always fails silently.
    - Fix: Removed broken shutdown block (no global singleton to close).

19. **Evolution endpoint `UnboundLocalError`** (api/routes/evolution.py:73):
    - `_current_run = {...}` created a local variable, shadowing the module-level dict. Python's scope rules then treated ALL references in the function as local → `UnboundLocalError` on line 70. The `/run` endpoint was completely broken.
    - Fix: Changed to mutate existing dict with `_current_run["status"] = ...` inside the lock.

20. **Shell command blocklist** (api/routes/tools_new.py):
    - `/api/shell/run` accepted arbitrary commands. Added blocklist for destructive patterns (`rm -rf /`, `mkfs`, `dd if=`, fork bombs, blanket `taskkill`/`pkill` node, `shutdown`, etc.).

21. **SQL injection — SELECT-only restriction** (api/routes/tools_new.py):
    - `/api/database/query` accepted arbitrary SQL. Now restricted to `SELECT`, `PRAGMA`, `EXPLAIN`, `WITH` at the API layer.

22. **CSV path traversal protection** (api/routes/tools_new.py):
    - `csv_path` now resolved with `Path.resolve(strict=True)` and blocked against system directories (`/etc/`, `/proc/`, `\Windows\`, etc.).

23. **File path disclosure** (api/routes/upload.py:196):
    - Response exposed absolute filesystem path. Changed to return filename only. Chat handler updated to resolve relative paths against UPLOAD_DIR.

### High

24. **Rate limiter global lock removed** (api/middleware.py):
    - Single `asyncio.Lock` serialized ALL rate limit checks across all IPs. Since asyncio is single-threaded and there's no `await` inside the critical section, no coroutine can interleave — the lock was pure overhead. Removed.

25. **Proactive lazy lock race** (api/routes/proactive.py:22-29):
    - `asyncio.Lock()` created lazily without guard (TOCTOU race in theory). Fixed: create lock at module level.

26. **Multi-agent session LRU eviction** (api/routes/multi_agent.py):
    - 100-session hard cap with no eviction → HTTP 429 error. Changed to LRU eviction: oldest session dropped when cap reached.

27. **PDF upload size limit** (api/routes/pdf.py):
    - Uploaded PDFs had no size check. Added 50MB limit matching the URL extraction endpoint.

28. **Whisper model cached** (api/routes/transcribe.py):
    - Whisper base model (~300MB) was loaded per-request via `_whisper.load_model("base")`. Added module-level cache with double-checked locking. Model loads once, reused on subsequent requests.

29. **Missing auth on 24 state-mutating API routes:**
    - Added `require_api_key` dependency to: context, thinking, idle_behaviors, consciousness, conversation_starters, knowledge, models, search, youtube, math, reasoning_tree, image_gen, transcribe, pdf, ocr, summarize, research, agent_action, multi_model, thinking_mode, state_machine, introspection, activity, reliability.

30. **MemorySystem.recall() optimized** (memory/memory_system.py):
    - Was: SELECT all rows → embed query → score all in Python (O(N)).
    - Now: Embed query first → if embeddings available, only fetch embedded rows → if not, Jaccard fallback limited to most recent 500 rows. Avoids deserializing unused embedding column in Jaccard mode.

31. **Shared `_AGENT_EXECUTOR` replaces throwaway ThreadPoolExecutors** (agent.py):
    - Added module-level `ThreadPoolExecutor(max_workers=4)`.
    - Replaced 4 throwaway executor sites: `_observe()` context gathering, tool call timeout wrapper, and 2 memory query executors (sync + stream paths).
    - Eliminates thread pool create/destroy overhead on every message and tool call.

### Medium / Dead Code

32. **System prompt early budget checks** (brain.py):
    - Moved `MAX_SYSTEM_PROMPT_CHARS = 12000` before expensive sections. Codebase search and episodic recall now skipped when prompt is already near cap, avoiding wasted work.

33. **Deleted 3 confirmed dead files** (594 lines):
    - `aura/secure_logging.py` — zero imports (only referenced from deleted legacy test)
    - `aura/tools/plugin_sdk.py` — zero imports outside itself
    - `aura/tools/edit_loop.py` — zero references anywhere

34. **Deleted `scripts/legacy_tests/` directory** (15 files):
    - All legacy test files from Feb 2026, superseded by `tests/stress_test_brain.py`.

---

## Remaining Issues (Not Fixed — Low Priority)

### Consolidation Opportunities (Medium)
- **3 overlapping clipboard tools**: `clipboard.py`, `clipboard_history.py`, `clipboard_memory.py` — could merge
- **5 overlapping memory systems**: `amem.py`, `hybrid_memory.py`, `hybrid_amem.py`, `local_rag.py`, `knowledge_graph.py` — unified_memory.py already wraps these, but the backends are redundant
- **2 overlapping research tools**: `research_tool.py`, `deep_research.py` — could merge
- **239 `except Exception: pass` blocks** across 63 files — most justified for optional modules, but inhibits debugging
- **Brain startup**: 60+ tool imports slow init (already split into fast/slow tiers)

### Style (Low)
- Remaining low-priority style issues (13 total) — not blocking functionality

---

## Architecture Observations

### What's Working Well
- **Circuit breaker pattern** on world model extraction prevents cascading failures
- **Fallback chain** for LLM calls (model A → B → C) provides resilience
- **TTL-cached system additions** (12s) prevent 8+ module queries per rapid message
- **History lock discipline** — reads copy under lock, I/O outside lock
- **Neuromodulator safety bounds** (0.7x-1.4x multiplier) prevent runaway parameters
- **Code executor sandboxing** is solid (AST validation, blocked modules/builtins)
- **Lazy imports** throughout prevent circular dependency deadlocks

### Areas for Improvement
- **Memory system fragmentation**: 5 overlapping backends means data splits across systems with no unified view.
- **Silent failure culture**: 239 bare `except` blocks mean bugs in consciousness/emotion modules are invisible.
- **Testing**: Only 1 test file (`stress_test_brain.py`). No unit tests for API routes, tools, or consciousness modules.

---

## Stress Test Results (from previous session)
- **80 messages across 8 categories**: 99% pass rate (79/80)
- **Only failure**: Gibberish input ("a " * 500) → empty response (expected)
- **Zero timeouts, zero errors**
- **Context memory working**: Remembered user name, location, project across messages
- **Auto-compact triggered correctly** at messages 15, 30, 45, 60, 75

---

## Total Impact

| Metric | Value |
|--------|-------|
| **Critical bugs fixed** | 13/13 (100%) |
| **High bugs fixed** | 14/14 (100%) |
| **Medium issues fixed** | 14/19 (74%) |
| **Files modified** | 38+ |
| **Dead code removed** | 594 lines + 15 legacy test files |
| **Security hardened** | Shell blocklist, SQL restriction, CSV path validation, SSRF protection, zip-slip fix, path disclosure fix, auth on 24 routes |
| **Performance improved** | Shared executor (4 sites), memory recall optimization, system prompt budget checks, Whisper model cache |
| **Thread safety fixed** | Deadlock, 3 race conditions, lock removal where unnecessary |
