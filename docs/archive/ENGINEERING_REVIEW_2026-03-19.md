# Aura Engineering Review Report
**Date:** 2026-03-19
**Scope:** Full-project audit — 10 parallel agents covering ~177K lines across ~830 Python files
**Prior Review:** 2026-03-18 (28 issues found/fixed, 445 tests passing)
**Test Results:** 658 passed, 0 failures, 0 regressions after fixes

---

## Executive Summary

This review found **~120 distinct issues** across 10 subsystems. The project is ambitious and architecturally rich but has significant gaps in security, data integrity, and dead infrastructure. The most concerning patterns are:

1. **Security theater** — Shell blocklists, AST-based sandbox checks, and prompt injection filters that can all be trivially bypassed
2. **Memory system divergence** — UnifiedMemory (primary) and MemoryRetriever write to different backends on every chat; 2 systems are dead code (see section 15)
3. **Dead infrastructure** — Life modeling, strategy bandit, and several consciousness modules are built but never wired in (GEPA evolution now wired in — see section 16)
4. **~300+ silent exception swallows** across the codebase hiding real bugs
5. **Thread safety gaps** in core state management (A-MEM, MetacognitiveEngine) — note: MarkdownStore is dead code (section 15)

---

## Summary Table

| Category | Critical | High | Medium | Low | Total |
|----------|----------|------|--------|-----|-------|
| **Security** | 5 | 10 | 8 | 3 | 26 |
| **Bugs / Logic** | 4 | 8 | 12 | 6 | 30 |
| **Thread Safety** | 1 | 2 | 6 | 4 | 13 |
| **Dead Code** | 0 | 2 | 3 | 12 | 17 |
| **Performance** | 0 | 3 | 4 | 3 | 10 |
| **Error Handling** | 0 | 0 | 5 | 4 | 9 |
| **Architecture** | 0 | 1 | 4 | 3 | 8 |
| **Data Integrity** | 2 | 2 | 3 | 1 | 8 |
| **Total** | **12** | **28** | **45** | **36** | **121** |

---

## 1. CRITICAL Issues (Must Fix)

### SEC-C1: Shell endpoint is fundamentally insecure
**File:** `api/routes/tools_new.py:340-411`
**Subsystem:** API
The `/api/shell/run` endpoint uses a blocklist to prevent dangerous commands. The blocklist is trivially bypassed via base64 encoding, variable indirection, backticks, path-based invocation (`/usr/bin/curl`), and PowerShell Core (`pwsh`). **A blocklist approach cannot secure arbitrary shell execution.**
**Fix:** Replace with a strict allowlist of permitted commands with argument validation, or sandbox via containers.

### SEC-C2: Sandbox AST check is bypassable
**File:** `aura/sandbox/executor.py:299-341`
**Subsystem:** Sandbox
The Python sandbox relies on AST analysis that can be bypassed with string concatenation (`getattr(__builtins__, 'op'+'en')`), computed attribute access, and dynamic import tricks. The subprocess runs with the full Python environment.
**Fix:** Use `--isolated` flag, restrict `PYTHONPATH`, or containerize execution.

### SEC-C3: Email attachment path traversal
**File:** `aura/tools/email_tool.py:1138-1141, 1179-1182`
**Subsystem:** Tools
Attachment filenames from emails are used directly in file paths. A malicious email with filename `../../.ssh/authorized_keys` could write files outside the target directory.
**Fix:** `filepath = target_dir / Path(filename).name` + validate resolved path is within target_dir.

### SEC-C4: Scaffold writes AI-generated files to arbitrary paths
**File:** `aura/tools/scaffold.py:1527-1535`
**Subsystem:** Tools
`scaffold_custom()` takes LLM-generated JSON with file paths and writes them via `_write_files()` with no path validation. Paths like `../../../.bashrc` could escape the target directory.
**Fix:** Validate each resolved path is a child of `base` using `resolved.relative_to(base)`.

### SEC-C5: ChatGPT set-token endpoint has no authentication
**File:** `api/routes/auth.py:30`
**Subsystem:** API
`POST /api/auth/chatgpt/set-token` accepts a refresh token without any auth. Anyone who can reach the API can overwrite the ChatGPT OAuth token.
**Fix:** Add `Depends(require_api_key)` to this endpoint.

### BUG-C1: Scheduled episodic memory consolidation calls nonexistent method
**File:** `aura_episodic_memory/memory_store.py:655`
**Subsystem:** Memory
`consolidator.consolidate()` is called but `MemoryConsolidator` has no such method. Correct method is `run_full_consolidation()`. The 24-hour auto-consolidation has been silently crashing every time.
**Fix:** Change to `consolidator.run_full_consolidation()`.

### BUG-C2: A-MEM Qdrant point IDs are non-deterministic across restarts
**File:** `aura/tools/amem.py:445, 773, 818`
**Subsystem:** Memory
Uses `abs(hash(note.id)) % (2**63)` — Python's hash is randomized per process (since 3.3). After restart, delete/update operations target wrong Qdrant point IDs, causing phantom data and failed deletes.
**Fix:** Use `int(hashlib.md5(note.id.encode()).hexdigest()[:15], 16)` for deterministic IDs.

### BUG-C3: `random.triangular` argument order is wrong in life modeling
**File:** `aura_life_modeling/scenario.py:44`
**Subsystem:** Life Modeling
`rng.triangular(self.min_value, self.most_likely, self.max_value)` — Python's `random.triangular(low, high, mode)` expects `(min, max, mode)` not `(min, mode, max)`. All scenario simulations produce incorrect distributions.
**Fix:** `rng.triangular(self.min_value, self.max_value, self.most_likely)`

### BUG-C4: SelfImprovementEngine calls nonexistent `get_drive_levels()`
**File:** `aura/consciousness/self_improvement.py:697`
**Subsystem:** Consciousness
Calls `im.get_drive_levels()` but the actual method is `get_drives_summary()`. The competence-urgency gate is permanently disabled because the `AttributeError` is silently caught.
**Fix:** Change to `im.get_drives_summary()`.

### BUG-C5: EmotionActionBridge instantiated when import failed (None callable)
**File:** `aura/proactive/gateway_daemon.py:122`
**Subsystem:** Proactive
`EmotionActionBridge()` is called unconditionally, but the import sets it to `None` on failure. This crashes the daemon on startup when the emotion module is missing.
**Fix:** `self.emotion_action_bridge = EmotionActionBridge() if EmotionActionBridge else None`

### BUG-C6: ALMA Engine module-level instantiation blocks import
**File:** `aura/emotion/alma_engine.py:1422`
**Subsystem:** Emotion
`alma_engine = ALMAEngine()` runs at import time, creating directories, reading files, and registering atexit handlers. If any dependency fails, the entire emotion module is broken.
**Fix:** Use lazy singleton via existing `get_alma_engine()`.

### BUG-C7: SelfImprovementEngine monkey-patches MetacognitiveEngine
**File:** `aura/consciousness/self_improvement.py:317-328`
**Subsystem:** Consciousness
`_install_strategy_override()` replaces `mc._run_strategy` at runtime. Fragile — if singleton is recreated or method signature changes, it silently breaks.
**Fix:** Use a proper strategy registry or composition pattern.

---

## 2. HIGH Issues (Should Fix)

### Security (HIGH)

| ID | File | Issue |
|----|------|-------|
| SEC-H1 | `api/routes/tools_new.py:743-762` | CSV import path traversal — blocklist is incomplete |
| SEC-H2 | `api/routes/tools_new.py:786-806` | Audio transcribe path — same issue |
| SEC-H3 | `api/routes/tools_new.py:588-613` | SSRF DNS rebinding bypass in API tester |
| SEC-H4 | `api/routes/feed.py:182-206` | Feed `item_id` path traversal via `../` |
| SEC-H5 | `aura/auth/chatgpt_oauth.py:147-158` | OAuth tokens stored as plaintext JSON |
| SEC-H6 | `aura/multi_user/identity_core.py:329` | User ID used unsanitized in file paths |
| SEC-H7 | `aura_daemon.py:566-612` | IPC server accepts commands without auth |
| SEC-H8 | `aura_skill_library/skill_learner.py:205-246` | Unsanitized LLM-generated skill procedures |
| SEC-H9 | `aura/tools/scaffold.py:1592` | `shell=True` in subprocess call |
| SEC-H10 | `aura/tools/code_executor.py:334-340` | Subprocess inherits full environment (API keys) |

### Bugs (HIGH)

| ID | File | Issue |
|----|------|-------|
| BUG-H1 | `aura/agent.py:795` | `self.tools["inner_monologue"]` KeyError if tool failed to load |
| BUG-H2 | `aura/agent.py:963` | Missing `import os` at module level — NameError in fast_init |
| BUG-H3 | `aura_life_modeling/simulation_engine.py:18-20` | Mesa v2 API incompatibility (crashes at runtime) |
| BUG-H4 | `aura_life_modeling/simulation_engine.py:187-188` | Quit-job scenario permanently zeros income |
| BUG-H5 | `api/services/agent_service.py:906-912` | Model override not cleared in chat_stream |
| BUG-H6 | `api/services/agent_service.py:466-474` | Orchestrator closure captures stale model |
| BUG-H7 | `aura/narrative_self.py:41` | `threading.Lock()` instead of `RLock()` — deadlock risk |
| BUG-H8 | `aura/consciousness/world_model.py:1279-1307` | Nested SQLite connections inside lock |

### Data Integrity (HIGH)

| ID | File | Issue |
|----|------|-------|
| DAT-H1 | `aura/markdown_store.py:176-198` | Race condition — read-modify-write without locking |
| DAT-H2 | `aura/tools/amem.py:1409-1415` | JSONL append not atomic — crash = data loss |
| DAT-H3 | Memory systems (multiple) | Two user profile stores diverge silently |

---

## 3. MEDIUM Issues (Fix When Practical)

### Security (MEDIUM)
- `api/routes/tools_new.py:682-698` — SQL WITH prefix can bypass in non-SQLite DBs
- `api/routes/tools_new.py:670-762` — DB parameter unsanitized
- `aura/multi_user/manager.py:231` — Session ID uses MD5 (guessable)
- `api/routes/auth.py:61` — Error messages leak internal details
- `aura/multi_agent/router.py:217-227` — Prompt injection filter trivially bypassable
- `aura/channels/discord_adapter.py:208-246` — DMs open to all users
- `api/routes/multi_agent.py:180` — No rate limiting on expensive multi-agent endpoint
- `aura/sandbox/executor.py:54` — `rm -rf` pattern too narrow

### Bugs (MEDIUM)
- `aura/agent.py:3878-3924` — `chat_stream()` bypasses Strategy Bandit entirely
- `aura/agent.py:3162` — Broken f-string in generated prime number code
- `aura/brain.py:1232-1244` — `list_conversations` mutates cached index
- `aura/agent.py:1555-1825` — `run()` skips `_finalize_chat` post-processing
- `aura/evolution/proposers.py:259` — Merge candidate ID collision
- `aura_skill_library/skill_executor.py:274` — Crash if feedback callback returns non-dict
- `aura_life_modeling/life_state.py:48-50` — Runway calculation double-counts emergency fund
- `aura_episodic_memory/temporal_parser.py:86` — Stale `base_time` for long-running processes
- `aura_episodic_memory/mcp_tools.py:554` — Search cache never invalidated
- `aura/proactive/event_bus.py:113-117` — Lazy lock init race condition
- `aura/proactive/gateway_daemon.py:662-666` — Catches BaseException (too broad)
- `aura/multi_agent/orchestrator.py:171` — Throwaway ThreadPoolExecutor per call

### Thread Safety (MEDIUM)
- `aura/tools/amem.py:473` — Evolution runs outside lock with stale data
- `aura/consciousness/metacognition.py` — No locking at all
- `aura/emotion/action_bridge.py:58-155` — Shared mutable action objects
- `aura/consciousness/strategy_bandit.py` — SQLite connections not pooled
- `aura/proactive/gateway_daemon.py:1631` — Accumulator internals accessed without lock
- `aura/proactive/gateway_daemon.py:339` — Fire-and-forget event bus subscription

---

## 4. Dead Code & Dead Infrastructure

### Dead Infrastructure (built but never wired in)
| System | Status | Action |
|--------|--------|--------|
| **GEPA Evolution** (`aura/evolution/`) | **Now wired in** — see section 16 | /evolve command, API endpoint, proactive suggestions, idle checks |
| **Life Modeling** (`aura_life_modeling/`) | MCP tools defined but not registered | Register with Aura's tool registry |
| **Strategy Bandit** | Selects strategies but all modules except CoT are None | Either implement or remove non-CoT strategies |
| **Prompt Evolution** | Infrastructure present but disabled by default | Wire in or remove |
| **Conditional/failure impacts** in life sim | Data model fields exist but never applied | Implement in `step()` or remove |

### Dead Methods
| File | Method | Notes |
|------|--------|-------|
| `agent.py` | `_handle_emotional_message()` | Superseded by ALMA |
| `agent.py` | `_detect_git_action()`, `_detect_monologue_action()`, `_detect_knowledge_graph_action()` | Superseded by `_handle_*_command()` variants |
| `agent.py` | `record_user_feedback()` | No-op with TODO |
| `brain.py` | `_default_system_prompt()` | Never referenced |
| `memory_retriever.py` | `_embed_text()` | Never called |
| `knowledge_graph/graph_database.py` | `_escape_string()` | Superseded by parameterized queries |

### Dead Files/Directories
- `aura_skills/` — Empty directory
- `scripts/migrate_memory.py:251` — References nonexistent `ep_store.get_recent()` method
- `patch_telegram_auth.py` — One-shot patch script (should be merged into source)

---

## 5. Memory System Architecture Problem (SUPERSEDED — see section 15)

> **Note:** This section was based on static code analysis. Runtime tracing revealed a different picture. See **section 15** for the corrected architecture.

**Original analysis (partially incorrect)** — listed 7 memory systems with overlapping purposes:

| # | System | Backend | Status |
|---|--------|---------|--------|
| 1 | MarkdownStore | Flat .md files | Active, writes `data/memory/` |
| 2 | MemoryRetriever | Flat .md files + delegates to UnifiedMemory | Active, also writes `data/memory/` |
| 3 | MemorySystem (DEPRECATED) | SQLite (agent_memory.db) | Deprecated, still exists |
| 4 | A-MEM (Zettelkasten) | JSONL + NPZ + Qdrant | Active |
| 5 | Episodic Memory | Qdrant | Active |
| 6 | Knowledge Graph (Kuzu) | Kuzu embedded DB | Active |
| 7 | Knowledge Graph (NetworkX) | In-memory + JSON | Active, independent from #6 |
| 8 | Unified MemoryStore | SQLite (aura_memory.db) + FTS5 | Intended consolidation target |

**Key problem:** Systems 1+2 both write to `data/memory/user_profile.md`. System 8 writes to its own SQLite DB. User facts learned via one system are invisible to others. Two Qdrant instances run independently (A-MEM and Episodic). Two knowledge graphs with no sync mechanism.

**Recommendation:** Complete the migration to system 8 (Unified MemoryStore) or establish write-through from active systems to a single source of truth. Remove deprecated system 3.

---

## 6. Silent Exception Pattern (~300+ instances)

The project has approximately 300+ instances of `except Exception` that either `pass` or log at `debug` level. Worst offenders by file:

| File | Count | Impact |
|------|-------|--------|
| `email_tool.py` | ~39 | IMAP failures invisible |
| `browser.py` | ~34 | Browser automation errors hidden |
| `calendar_tool.py` | ~31 | Calendar sync failures invisible |
| `brain.py` | ~25 | Subsystem failures masked |
| `agent.py` | ~40 | Tool loading, memory, consciousness failures hidden |
| `windows_control.py` | ~25 | OS integration errors hidden |

**Recommendation:** For each subsystem, differentiate between "not yet initialized" (debug) and "errored after initialization" (warning). Add a `_consecutive_failures` counter that escalates to warning after N failures.

---

## 7. Performance Hotspots

| Issue | File | Impact |
|-------|------|--------|
| System prompt rebuilt on every `think()` call | `brain.py:1758-1903` | CodebaseIndex opens/closes SQLite per call |
| Pre-response emotional appraisal = LLM call per message | `agent.py:3307` | 3 LLM calls minimum per `chat()` |
| Regex compiled per keyword per message (~150 keywords) | `agent.py:1403` | CPU waste on every message |
| `get_node()` calls `save()` (full disk write) on every access | `tools/knowledge_graph.py:352` | Severe I/O bottleneck |
| World model snapshot written under lock on every mutation | `consciousness/world_model.py:1624` | Serializes all WM operations behind disk I/O |
| New OllamaBrain created per prompt evolution stage | `consciousness/prompt_evolution.py:395` | 3 unnecessary brain instantiations |
| Embedding model loaded in hot path on first call | `consciousness/reasoning_templates.py:162` | Blocks first conversation |
| New ThreadPoolExecutor per multi-agent parallel call | `multi_agent/orchestrator.py:171` | Unbounded thread creation |

---

## 8. Recommended Fix Priority

### Phase 1: Security & Crash Fixes (do first)
1. Guard EmotionActionBridge instantiation (BUG-C5)
2. Fix `consolidator.consolidate()` → `run_full_consolidation()` (BUG-C1)
3. Fix A-MEM hash to be deterministic (BUG-C2)
4. Add auth to `/api/auth/chatgpt/set-token` (SEC-C5)
5. Sanitize email attachment filenames (SEC-C3)
6. Validate scaffold file paths (SEC-C4)
7. Add `import os` to agent.py top level (BUG-H2)
8. Fix `self.tools["inner_monologue"]` to `.get()` (BUG-H1)
9. Fix `random.triangular` argument order (BUG-C3)
10. Fix `get_drive_levels()` → `get_drives_summary()` (BUG-C4)

### Phase 2: Data Integrity & Thread Safety
1. Add threading locks to MarkdownStore and MemoryRetriever
2. Fix A-MEM evolution to hold lock during note modification
3. Add RLock to NarrativeSelf (replace Lock)
4. Add locking to MetacognitiveEngine
5. Clear model override in chat_stream's finally block
6. Fix TemporalParser to use `datetime.now()` at parse time

### Phase 3: Performance
1. Pre-compile `_TOOL_KEYWORDS` into single regex
2. Cache CodebaseIndex instance instead of creating per call
3. Add dirty tracking + debounced saves to knowledge graph
4. Debounce world model snapshot writes
5. Share OllamaBrain instance across prompt evolution stages

### Phase 4: Dead Code Cleanup
1. Delete MarkdownStore — confirmed dead code, never instantiated (section 15)
2. Delete MemorySystem (deprecated) — confirmed dead code, never instantiated (section 15)
3. Remove dead agent methods (`_detect_*_action`, `_handle_emotional_message`, etc.)
4. ~~Remove or wire in GEPA evolution system~~ — **DONE** (section 16)
5. Clean up Strategy Bandit — remove strategies with no backing module
6. Merge `patch_telegram_auth.py` into source

---

## 9. Remaining Risks & Ambiguities

1. **Shell endpoint** — Even with an allowlist, local shell execution is inherently risky. Consider if this feature is necessary.
2. **Sandbox** — The AST-based check + local subprocess approach cannot provide real isolation. Consider E2B-only or containerized execution.
3. **Dual knowledge graphs** — NetworkX (tool-only) and Kuzu (wired into UnifiedMemory). Kuzu is the active system; NetworkX is tool-only and independent (see section 15).
4. **Memory consolidation** — MarkdownStore and MemorySystem are dead code (being deleted). Remaining divergence is UnifiedMemory vs MemoryRetriever .md files (being fixed with write-through). See section 15.
5. **`skills/community/` directory** — ~100+ files from Claude Code community skills. Unclear if these are used by Aura or orphaned.
6. **Mesa v2 compatibility** — Life modeling module uses Mesa v1 API. If Mesa v2 is installed, the module crashes.

---

## 10. What's Working Well

- **Authentication middleware** — Properly implemented with public path exemptions and production mode detection
- **Attachment path validation** in chat.py — Uses `resolve(strict=True)` + `relative_to()` correctly
- **Upload filename sanitization** — Thorough
- **Zip-slip protection** — Correct `relative_to()` check
- **PDF SSRF protection** — DNS resolution + IP validation is solid
- **Hook execution security** — `shell=False` with `shlex.split` and clamped timeout
- **LINE webhook verification** — Proper HMAC-SHA256 with timing-safe comparison
- **World model architecture** — Well-designed SQLite + JSON snapshot with audit logging
- **ALMA emotional engine** — Sophisticated PAD space with well-structured emotion transitions
- **Event bus pattern** — Clean pub-sub with channel-based routing
- **GEPA evolution** — Architecturally solid even if not yet wired in

---

## 11. Fixes Applied in This Review

All fixes verified with 658 passing tests, 0 regressions.

### Security Fixes (8)

| # | File | Fix |
|---|------|-----|
| 1 | `api/routes/auth.py` | Added `Depends(require_api_key)` to ChatGPT set-token endpoint |
| 2 | `aura/tools/email_tool.py` | Sanitized attachment filenames via `Path(filename).name` + resolve validation in both Gmail and IMAP downloaders |
| 3 | `aura/tools/scaffold.py` | Path traversal guard in `_write_files()` — rejects `..` and paths outside base directory |
| 4 | `aura/tools/scaffold.py` | Removed `shell=True` from `_auto_install()` subprocess — now uses `shlex.split` + `shell=False` |
| 5 | `aura/tools/code_executor.py` | Sanitized subprocess environment — only passes PATH, HOME, TEMP, etc. (strips API keys/tokens) |
| 6 | `aura/auth/chatgpt_oauth.py` | Set 0600 file permissions on saved token file |
| 7 | `aura/multi_user/identity_core.py` | Sanitized user_id in file path construction (strips non-alphanumeric chars) |
| 8 | `api/routes/feed.py` | Added `_validate_feed_id()` regex guard against path traversal in feed item_id |

### Bug Fixes (7)

| # | File | Fix |
|---|------|-----|
| 1 | `aura_episodic_memory/memory_store.py:655` | `consolidator.consolidate()` → `consolidator.run_full_consolidation()` (nonexistent method) |
| 2 | `aura/tools/amem.py` (5 locations) | Replaced `abs(hash(...))` with `_deterministic_point_id()` using MD5 — survives process restarts |
| 3 | `aura/agent.py:795` | `self.tools["inner_monologue"]` → `self.tools.get("inner_monologue")` + null guard |
| 4 | `aura/agent.py:3` | Added `import os` at module level — fixes NameError in fast_init mode |
| 5 | `aura_life_modeling/scenario.py:44` | Fixed `random.triangular(min, mode, max)` → `random.triangular(min, max, mode)` |
| 6 | `aura/consciousness/self_improvement.py:696` | `im.get_drive_levels()` → `im.get_drives_summary()` (nonexistent method) |
| 7 | `aura/proactive/gateway_daemon.py:122` | Guarded `EmotionActionBridge()` — now checks `if EmotionActionBridge else None` |

### Thread Safety / Data Integrity Fixes (3)

| # | File | Fix |
|---|------|-----|
| 1 | `aura/markdown_store.py` | Added `threading.RLock()` to `__init__`, wrapped `add_entry()` and `update_section()` in lock |
| 2 | `aura/narrative_self.py:42` | `threading.Lock()` → `threading.RLock()` to prevent deadlock |
| 3 | `aura_episodic_memory/temporal_parser.py:86` | `base_time` is now a property that returns `datetime.now()` at parse time instead of stale cached value |

### Credential / Config Fixes (3)

| # | File | Fix |
|---|------|-----|
| 1 | `deploy/setup_env_keys.sh` | **CRITICAL** — Removed 4 hardcoded API keys (Ollama, Tavily, Brave, Firecrawl). Script now requires keys via env vars. **YOU MUST ROTATE ALL 4 KEYS — they were committed to git history.** |
| 2 | `deploy/fix_server.sh:257` | Removed hardcoded server IP `89.167.107.134` — replaced with placeholder |
| 3 | `aura.cmd` | Fixed stale path pointing to `C:\Users\asus\apprentice-agent\` → now uses `D:\Aura\main.py` |

---

## 12. Additional Findings from Agents 9 & 10

### Test Coverage Gaps (Agent 9)
- **brain.py** (~2500 lines) has ZERO unit tests — only a manual stress test
- **agent.py** (~5266 lines) has ZERO unit tests — integration tests manually replicate its internals
- **API routes** — only auth (3 tests) and WebSocket are covered; ~30 route files untested
- **Emotion system** (ALMA, EvoEmo, Theory of Mind) — zero direct tests
- No tests for concurrent access, malformed LLM responses, disk full, or model fallback chains

### Dead Code (Agent 9)
- `aura/tools/edit_test_fix.py` — zero references anywhere
- `aura/channels/line_adapter.py`, `signal_adapter.py` — never imported by channel_manager
- `aura/tools/firecrawl_tool.py`, `obsidian_tool.py` — only lazy-registered, never called
- `docs/plans/always_on_context_engine.py` — a plan doc stored as .py
- 20+ empty directories in `aura_skills/`, `aura_data/`
- `data/worldmodel.db.bak` — stale backup committed to repo

### Web / Deploy (Agent 10)
- **4 API keys were hardcoded in `deploy/setup_env_keys.sh`** — FIXED (see above)
- **Server IP hardcoded in `deploy/fix_server.sh`** — FIXED
- Web UI has no Content Security Policy meta tag
- Extension `newtab.js` and `background.js` hardcode `http://localhost:8000`
- Docker memory limit (4GB) insufficient for ML model loading
- Nginx config uses `YOUR_DOMAIN` placeholder without prompting

---

## 13. Second Pass Fixes (Phase 2-5)

Performed after the initial 21 fixes. 38 files modified total, 658 tests still passing.

### Security Fixes (Second Pass)

| # | File | Fix | Category |
|---|------|-----|----------|
| 1 | `api/routes/tools_new.py` | CSV import + audio transcribe: restricted file paths to allowed directories with resolve+relative_to validation | SEC-H1, SEC-H2 |
| 2 | `api/routes/tools_new.py` | SSRF: added `socket.getaddrinfo()` DNS resolution for non-IP hostnames, validating resolved IPs against private ranges | SEC-H3 |
| 3 | `api/routes/auth.py` | Replaced all raw `str(e)` error responses with generic messages; detailed errors logged server-side only | MEDIUM |
| 4 | `aura/multi_user/manager.py` | Session ID: replaced `hashlib.md5(time+user)` with `secrets.token_hex(12)` | MEDIUM |
| 5 | `aura_daemon.py` | IPC server: added random auth token generated at startup, written to restricted file, required in every message | SEC-H7 |
| 6 | `aura/tools/database_tool.py` | Always apply read-only SQLite authorizer (not just for SELECT); block multi-statement queries via semicolon check | MEDIUM |
| 7 | `aura/sandbox/executor.py` | Expanded `rm -rf` pattern to catch `/anything`, `-fr` variant, and `rm -rf *` | MEDIUM |

### Bug Fixes (Second Pass)

| # | File | Fix | Category |
|---|------|-----|----------|
| 1 | `api/services/agent_service.py` | Clear model_override in `chat_stream` finally block (was only cleared in `chat()`) | BUG-H5 |
| 2 | `api/services/agent_service.py` | Initialize `self._orchestrator = None` in `__init__`; pass model dynamically to avoid stale closure | BUG-H6 |
| 3 | `aura/brain.py` | `list_conversations`: copy entry dicts before mutating `is_active` to avoid cache corruption | MEDIUM |
| 4 | `aura/agent.py` | Fixed broken f-string in generated prime number code | MEDIUM |
| 5 | `aura/evolution/engine.py` | Initialize `iteration = 0` before loop to prevent NameError when `max_iterations=0` | MEDIUM |
| 6 | `aura_skill_library/skill_executor.py` | Added `isinstance(feedback, dict)` check before `.get()` calls | MEDIUM |
| 7 | `aura_life_modeling/life_state.py` | Simplified runway_months formula; added docstring noting the double-count ambiguity | MEDIUM |

### Thread Safety Fixes (Second Pass)

| # | File | Fix | Category |
|---|------|-----|----------|
| 1 | `aura/consciousness/metacognition.py` | Added `threading.RLock()`, wrapped state mutation methods | MEDIUM |
| 2 | `aura/tools/amem.py` | `_evolve_related` now snapshots data under lock, does LLM outside lock, re-acquires lock for writes | MEDIUM |
| 3 | `aura/proactive/gateway_daemon.py` | Stored event bus subscription task ref + done_callback for error logging | MEDIUM |
| 4 | `aura/proactive/event_bus.py` | Initialize asyncio.Lock eagerly in `__init__` instead of lazy init | MEDIUM |

### Reliability Fixes (Second Pass)

| # | File | Fix | Category |
|---|------|-----|----------|
| 1 | `aura/emotion/alma_engine.py` | Replaced module-level `ALMAEngine()` with lazy singleton via `get_alma_engine()` | BUG-C6 |
| 2 | `aura/emotion/__init__.py` | Updated re-exports to use lazy getter | BUG-C6 |
| 3 | `aura_daemon.py` | Added agent load retry (60s cooldown between attempts) | RELIABILITY |
| 4 | `aura/proactive/gateway_daemon.py` | Changed `BaseException` catch to `Exception` + explicit `SystemExit`/`KeyboardInterrupt`/`CancelledError` handling | MEDIUM |
| 5 | `aura_episodic_memory/mcp_tools.py` | Added 60-second TTL to search cache entries | MEDIUM |

### Performance Fixes (Second Pass)

| # | File | Fix | Category |
|---|------|-----|----------|
| 1 | `aura/agent.py` | Pre-compiled `_TOOL_KEYWORDS` into single combined regex pattern (was ~150 separate regex compilations per message) | PERFORMANCE |
| 2 | `aura/tools/knowledge_graph.py` | Removed `save()` call from `get_node()` (reads should not trigger full disk writes) | PERFORMANCE |
| 3 | `aura/multi_agent/orchestrator.py` | Shared `ThreadPoolExecutor(max_workers=8)` on instance instead of creating/destroying per call | PERFORMANCE |

### Dead Code Cleanup (Second Pass)

| # | File | What | Category |
|---|------|------|----------|
| 1 | `aura/agent.py` | Removed 5 dead methods: `_handle_emotional_message`, `_detect_git_action`, `_detect_monologue_action`, `_detect_knowledge_graph_action`, `record_user_feedback` (~170 lines) | CLEANUP |
| 2 | `aura/brain.py` | Removed dead `_default_system_prompt()` static method | CLEANUP |
| 3 | `aura/memory_retriever.py` | Removed dead `_embed_text()` method | CLEANUP |
| 4 | `aura_knowledge_graph/graph_database.py` | Removed dead `_escape_string()` method | CLEANUP |
| 5 | `aura/tools/edit_test_fix.py` | **Deleted** — zero references anywhere in codebase | CLEANUP |

---

## 14. Complete Change Summary

### Total fixes across both passes: **~50 distinct issues resolved**

| Category | Pass 1 | Pass 2 | Total |
|----------|--------|--------|-------|
| Security | 9 | 7 | 16 |
| Bug fixes | 7 | 7 | 14 |
| Thread safety | 3 | 4 | 7 |
| Reliability | 0 | 5 | 5 |
| Performance | 0 | 3 | 3 |
| Dead code cleanup | 0 | 5 | 5 |
| Credential/config | 3 | 0 | 3 |

### Files touched: 38
### Net line change: -141 lines (459 added, 600 removed)
### Test results: 658 passed, 0 failures, 0 regressions

### Remaining items requiring owner decision:
1. **ROTATE API KEYS** — 4 keys were in git history (Ollama, Tavily, Brave, Firecrawl)
2. **Shell endpoint architecture** — blocklist approach is fundamentally insecure; needs allowlist or removal
3. **Sandbox architecture** — AST check alone cannot provide real isolation
4. **Memory system consolidation** — see corrected architecture in section 15 below
5. **Dead infrastructure** — life modeling, strategy bandit: wire in or remove? (GEPA evolution now wired in — see section 16)
6. **Mesa v2 compatibility** — life modeling uses Mesa v1 API
7. **~200+ remaining silent exception swallows** — need systematic escalation pass
8. **Test coverage** — brain.py, agent.py, API routes, emotion system have zero unit tests

---

## 15. Memory Architecture — Corrected Understanding

**Replaces section 5 ("Memory System Architecture Problem").** The original audit counted "7 overlapping memory systems" based on static code analysis. Runtime tracing revealed a different picture — some systems are dead code, some are tool-only, and UnifiedMemory is the clear primary system.

### Actual Runtime Architecture

| # | System | Backend | Runtime Status |
|---|--------|---------|----------------|
| 1 | **UnifiedMemory** (SQLite+FTS5) | `aura_memory.db` + FTS5 full-text + Kuzu KG | **PRIMARY** — reads and writes on every chat message |
| 2 | **MemoryRetriever** | Flat .md files in `data/memory/` | **Active parallel writer** — creates .md files alongside UnifiedMemory, causing divergence |
| 3 | **MarkdownStore** | Flat .md files | **DEAD CODE** — never instantiated at runtime. Being deleted. |
| 4 | **MemorySystem (deprecated)** | SQLite (`agent_memory.db`) | **DEAD CODE** — never instantiated at runtime. Being deleted. |
| 5 | **Episodic Memory** | Qdrant | **Active reader** — auto-recalled into system prompt via `quick_recall()` |
| 6 | **A-MEM (Zettelkasten)** | JSONL + NPZ + Qdrant | **Tool-only** — only queried when LLM explicitly calls the A-MEM tool. NOT auto-queried. |
| 7 | **KG Kuzu** | Kuzu embedded DB | **Active** — wired into UnifiedMemory as graph retrieval channel + entity extraction on every chat |
| 8 | **KG NetworkX** | In-memory + JSON | **Tool-only** — separate from Kuzu, only used when LLM calls the knowledge graph tool |
| 9 | **ContextEngine** | Delegates to UnifiedMemory | **Broken recall path** — calls `.recall()` but UnifiedMemory exposes `.query()`. Being fixed. |

### Data Flow Per Chat Message

```
USER MESSAGE
    │
    ├──► WRITE PATH
    │       │
    │       ├─► UnifiedMemory.store()          ← PRIMARY write (SQLite+FTS5)
    │       │       └─► Kuzu KG entity extraction  ← auto-triggered on every store
    │       │
    │       ├─► MemoryRetriever.save()         ← parallel .md write (DIVERGENCE SOURCE)
    │       │                                     (being fixed: write-through to UnifiedMemory only)
    │       │
    │       └─► Episodic Memory.record()       ← stores episode for later recall
    │
    └──► READ PATH
            │
            ├─► UnifiedMemory.query()          ← PRIMARY read (RRF + reranker retrieval)
            │       ├─► FTS5 full-text channel
            │       ├─► Embedding similarity channel
            │       └─► Kuzu KG graph channel
            │
            ├─► Episodic Memory.quick_recall() ← auto-injected into system prompt
            │
            ├─► ContextEngine.recall()         ← BROKEN: calls .recall() not .query()
            │                                     (being fixed to call .query())
            │
            ├─╌ A-MEM                          ← NOT auto-queried (tool-only, on LLM request)
            └─╌ KG NetworkX                    ← NOT auto-queried (tool-only, on LLM request)
```

### Key Corrections from Original Section 5

1. **"7 overlapping memory systems" is wrong.** At runtime, only 3 systems actively read/write on every chat: UnifiedMemory, MemoryRetriever, and Episodic Memory.
2. **MarkdownStore is dead code.** Thread safety fixes applied in this review (section 11) were unnecessary — the class is never instantiated. Being deleted.
3. **MemorySystem (deprecated) is dead code.** Never instantiated. Being deleted.
4. **A-MEM is not an "overlapping system"** — it is a tool the LLM can choose to invoke, like web search. It does not auto-query on every message.
5. **The real divergence problem** is between UnifiedMemory (SQLite) and MemoryRetriever (.md files) — both write on every chat but to different backends. Fix: make MemoryRetriever write-through to UnifiedMemory only.
6. **ContextEngine had a broken recall path** — it called `.recall()` on UnifiedMemory which exposes `.query()`. This meant context-based retrieval was silently failing. Being fixed.
7. **Kuzu KG is NOT a separate system** — it is wired into UnifiedMemory as one of three retrieval channels (FTS5, embedding, graph). The original audit listed it as independent.

---

## 16. GEPA Evolution — Now Wired In

The original audit (section 4) listed GEPA Evolution as "dead infrastructure — built but never wired in." This has been corrected. GEPA is now accessible through multiple entry points, though it remains user-controlled (not autonomous).

### What Was Added

| Entry Point | How It Works |
|-------------|-------------|
| **`/evolve` command** | Manual trigger in chat — user types `/evolve` to start a skill evolution cycle |
| **API endpoint** | Programmatic trigger via HTTP for external tools or scheduled jobs |
| **Proactive suggestion** | When a skill has >5 uses and <60% success rate, Aura suggests running evolution on it |
| **Idle presence check** | Every 6 hours, scans for weak skills and surfaces them as evolution candidates |

### How It Works

1. GEPA (Genetic Evolution of Pareto-optimal Agents) uses Pareto evolution to improve skill implementations
2. When triggered, it evaluates the target skill's historical performance, generates variant implementations, and tests them
3. Successful variants replace the original skill implementation

### Important: NOT Autonomous

Evolution is **never** triggered automatically. All paths require user approval:
- `/evolve` requires the user to type the command
- API endpoint requires an authenticated request
- Proactive suggestions are surfaced as messages — the user must confirm before evolution runs
- Idle checks only create suggestions, not actions

This is a deliberate design choice. Autonomous self-modification without user consent would be unsafe.
