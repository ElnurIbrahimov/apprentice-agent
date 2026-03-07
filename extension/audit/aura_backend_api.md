# AURA Backend API — Full Hardening Audit

**Date:** 2026-03-07
**Auditor:** Claude (production hardening pass)
**Scope:** `D:/Aura/api/main.py`, `D:/Aura/api/routes/*.py`, `D:/Aura/api/services/*.py`

---

## Summary

20 distinct issues found and fixed across 14 files. Issues span four categories: blocking async calls (critical), security/input validation (security), hardcoded URLs / env var naming (code quality), and missing HTTP status checks (correctness).

---

## Issues Found and Fixed

### 1. BLOCKING SYNC CALL IN ASYNC HANDLER — `transcribe.py`

**File:** `D:/Aura/api/routes/transcribe.py`
**Lines (before fix):** 37–39
**Severity:** Critical — blocks the entire event loop for potentially 30–120s per Whisper transcription.

`whisper.load_model()` and `model.transcribe()` are both pure-CPU blocking operations. Called directly inside an `async def` handler, they blocked the event loop for every other request during transcription.

**Fix:** Wrapped in `loop.run_in_executor(None, _run_whisper)`.

Also added a **100 MB upload size guard** (`HTTPException 413`) to prevent memory exhaustion from oversized audio files.

---

### 2. BLOCKING SYNC CALL IN ASYNC HANDLER — `pdf.py`

**File:** `D:/Aura/api/routes/pdf.py`
**Lines (before fix):** 25–27, 62–64
**Severity:** Critical — `pdfplumber.open()` + `extract_text()` are synchronous I/O + CPU operations.

Both `extract_upload` and `extract_url` called `pdfplumber.open()` directly in `async def` handlers.

**Fix:** Extracted `_extract(raw)` / `_extract_bytes(raw)` sync helpers, wrapped both in `loop.run_in_executor(None, ...)`.

---

### 3. BLOCKING SYNC TOOL CALLS IN ASYNC HANDLERS — `reasoning_tree.py`

**File:** `D:/Aura/api/routes/reasoning_tree.py`
**Lines (before fix):** `think_deeply` at line ~92, `explore_options` at line ~141
**Severity:** Critical — MCTS reasoning can run for 5–60+ seconds. Calling `tool.think_deeply()` and `tool.explore_options()` synchronously in `async def` handlers blocks the entire server for the duration.

**Fix:** Both calls wrapped in `loop.run_in_executor(None, lambda: ...)`.

---

### 4. BLOCKING SYNC TOOL CALLS IN ASYNC HANDLERS — `introspection.py`

**File:** `D:/Aura/api/routes/introspection.py`
**Lines (before fix):** `analyze_query` (~line 66), `pre_check` (~line 94), `wrap_response` (~line 120)
**Severity:** Critical — Each invokes `brain.think()` (LLM call) synchronously.

**Fix:** All three wrapped in `loop.run_in_executor(None, lambda: ...)`. Added `import asyncio` at top.

---

### 5. HARDCODED `http://localhost:11434` — `models.py`

**File:** `D:/Aura/api/routes/models.py`
**Line (before fix):** `host = os.getenv("OLLAMA_HOST", "http://localhost:11434")`
**Severity:** Code quality + correctness — Used wrong env var name (`OLLAMA_HOST` instead of `OLLAMA_BASE_URL` used by all other routes).

**Fix:** Changed to `os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")`.

Also fixed: `_ollama_models()` was a **blocking `requests.get()`** called directly from an async route handler with no `run_in_executor`. Replaced with `_ollama_models_async()` using `httpx.AsyncClient`, eliminating both the blocking call and the `requests` dependency.

---

### 6. HARDCODED `http://localhost:11434` — `multi_model.py`

**File:** `D:/Aura/api/routes/multi_model.py`
**Line (before fix):** `OLLAMA_BASE = "http://localhost:11434"`
**Severity:** Code quality — not configurable via environment.

**Fix:** `OLLAMA_BASE = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")`. Added `import os`.

---

### 7. HARDCODED `http://localhost:11434` — `youtube.py`

**File:** `D:/Aura/api/routes/youtube.py`
**Line (before fix):** `OLLAMA_URL = "http://localhost:11434/api/generate"`
**Severity:** Code quality.

**Fix:** `OLLAMA_URL = _os.getenv("OLLAMA_BASE_URL", "http://localhost:11434") + "/api/generate"`.

---

### 8. HARDCODED `http://localhost:11434` — `summarize.py`

**File:** `D:/Aura/api/routes/summarize.py`
**Line (before fix):** `OLLAMA_URL = "http://localhost:11434/api/generate"`
**Severity:** Code quality.

**Fix:** `OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434") + "/api/generate"`.

---

### 9. WRONG ENV VAR NAME `OLLAMA_HOST` — `agent_service.py`

**File:** `D:/Aura/api/services/agent_service.py`
**Line (before fix):** `ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")`
**Severity:** Code quality — inconsistent with all other routes that use `OLLAMA_BASE_URL`.

**Fix:** `os.getenv("OLLAMA_BASE_URL", os.getenv("OLLAMA_HOST", "http://localhost:11434"))` — reads `OLLAMA_BASE_URL` first, falls back to `OLLAMA_HOST` for backward compatibility, then defaults.

---

### 10. MISSING `raise_for_status()` — `agent_action.py`

**File:** `D:/Aura/api/routes/agent_action.py`
**Line (before fix):** After `await c.post(...)`, `r.json()` called without checking status.
**Severity:** Correctness — a 4xx/5xx Ollama response would cause an uninformative `KeyError` or return empty string instead of a meaningful error.

**Fix:** Added `r.raise_for_status()` before `r.json()`.

---

### 11. MISSING `raise_for_status()` — `math.py`

**File:** `D:/Aura/api/routes/math.py`
**Line (before fix):** Same pattern as above.
**Severity:** Correctness.

**Fix:** Added `r.raise_for_status()` before `r.json()`.

---

### 12. DEPRECATED `asyncio.get_event_loop()` — `youtube.py`

**File:** `D:/Aura/api/routes/youtube.py`
**Line (before fix):** `loop = asyncio.get_event_loop()` inside `summarize_youtube`
**Severity:** Code quality + correctness — `get_event_loop()` is deprecated in Python 3.10+ and may return a different loop in some contexts.

**Fix:** Changed to `asyncio.get_running_loop()`.

---

### 13. DEPRECATED `asyncio.get_event_loop()` — `research.py`

**File:** `D:/Aura/api/routes/research.py`
**Line (before fix):** `loop = asyncio.get_event_loop()` inside `_run_search`
**Severity:** Code quality + correctness.

**Fix:** Changed to `asyncio.get_running_loop()`.

---

### 14. DEPRECATED `asyncio.get_event_loop()` — `search.py`

**File:** `D:/Aura/api/routes/search.py`
**Line (before fix):** `loop = asyncio.get_event_loop()` inside `web_search`
**Severity:** Code quality + correctness.

**Fix:** Changed to `asyncio.get_running_loop()`.

---

### 15. NO INPUT LENGTH LIMIT — `ocr.py` (`image_b64`)

**File:** `D:/Aura/api/routes/ocr.py`
**Severity:** Security / DoS — an attacker can POST an arbitrarily large base64 string. Decoding a 500 MB payload allocates 375 MB RAM and blocks the process indefinitely.

**Fix:** Added guard: if `len(image_b64) > 20 * 1024 * 1024` raise `HTTPException(400, ...)`. 20 MB base64 ≈ 15 MB decoded, sufficient for high-res OCR.

---

### 16. NO INPUT LENGTH LIMIT — `summarize.py` (`SummarizeRequest.text`)

**File:** `D:/Aura/api/routes/summarize.py`
**Severity:** Security / DoS + prompt injection surface.

**Fix:** Added Pydantic `Field(..., max_length=500_000)` on `text`, `max_length=2048` on `url`, `max_length=500` on `title`.

---

### 17. NO INPUT LENGTH LIMIT — `knowledge.py` (`SaveRequest.text`)

**File:** `D:/Aura/api/routes/knowledge.py`
**Severity:** Security / DoS — text is stored in Qdrant episodic memory and later retrieved into LLM context windows; unbounded input is a prompt injection risk.

**Fix:** Added `Field(..., max_length=50_000)` on `text`, `max_length=2048` on `url`, `max_length=500` on `title`.

---

### 18. NO INPUT LENGTH LIMIT — `research.py` (`ResearchRequest.query`)

**File:** `D:/Aura/api/routes/research.py`
**Severity:** Security — query is passed verbatim to Tavily and then to an LLM. Unbounded length enables prompt injection.

**Fix:** Added `Field(..., max_length=1000)` on `query`.

---

### 19. URL SCHEME INJECTION — `tools_new.py` (`APITestRequest.url`)

**File:** `D:/Aura/api/routes/tools_new.py`
**Severity:** Security — the API tester endpoint proxied arbitrary URLs without scheme validation. An attacker could submit `file:///etc/passwd`, `ftp://...`, or `gopher://...` SSRF vectors.

**Fix:** Added `@field_validator("url")` that rejects any URL not starting with `http://` or `https://`. Also added `@field_validator("method")` to restrict HTTP methods to a safe allowlist, and bounded `body` to 1 MB and `timeout` to 1–120s.

---

### 20. NO INPUT LENGTH LIMIT — `agent_action.py` (`prompt`), `math.py` (`problem`), `multi_agent.py` (`message`), `multi_model.py` (`message`), `image_gen.py` (`prompt`)

**Files:** Multiple
**Severity:** Security / DoS — all these fields are passed directly to LLMs. Unbounded length enables token-flooding attacks and prompt injection.

**Fixes Applied:**
- `agent_action.py`: `prompt` capped at 32,000 chars (explicit 400 check).
- `math.py`: `problem` capped at 4,000 chars (explicit 400 check).
- `multi_agent.py`: `MultiAgentChatRequest.message` → `Field(..., max_length=32_000)`.
- `multi_model.py`: `CompareRequest.message` → `Field(..., max_length=8000)`.
- `image_gen.py`: `prompt` → `Field(..., max_length=1000)`, `negative_prompt` → `max_length=500`, `steps` bounded `ge=1, le=150`. Also extracted `COMFY` URL from env var `COMFY_BASE_URL`.

---

### 21. CONVERSATION SEARCH NO LIMIT — `chat.py`

**File:** `D:/Aura/api/routes/chat.py`
**Line (before fix):** `search_messages(q: str, limit: int = 20)`
**Severity:** DoS — no cap on `q` length, no cap on `limit`. With thousands of conversations, iteration could be unbounded.

**Fix:** Added `len(q) > 500` guard returning early, and `limit = min(limit, 100)` cap.

---

### 22. MUTABLE DEFAULT ARGUMENT — `proactive.py`

**File:** `D:/Aura/api/routes/proactive.py`
**Line (before fix):** `payload: Dict[str, Any] = None` (type hint inconsistency)
**Severity:** Code quality.

**Fix:** Changed to `payload: Optional[Dict[str, Any]] = None` (consistent with the `Optional` type semantics).

---

### 23. HARDCODED `http://localhost:8188` — `image_gen.py`

**File:** `D:/Aura/api/routes/image_gen.py`
**Severity:** Code quality.

**Fix:** `COMFY = os.getenv("COMFY_BASE_URL", "http://localhost:8188")`.

---

## Files NOT Changed

- `main.py` — no issues; already uses env vars, proper lifespan management, and async patterns throughout.
- `services/zip_analyzer.py` — synchronous but only called via `run_in_executor` from `chat.py`; zip-slip protection already in place.
- `services/agent_service.py` — only one env var fix applied (OLLAMA_HOST → OLLAMA_BASE_URL).
- `services/inner_thoughts_engine.py` — runs in a background daemon thread, not in async context; no blocking-in-async issue.
- `routes/features.py`, `routes/memory.py`, `routes/context.py`, `routes/thinking.py`, `routes/idle_behaviors.py`, `routes/conversation_starters.py`, `routes/proactive.py` (beyond the mutable default fix), `routes/multi_agent.py` (beyond message limit), `routes/activity.py`, `routes/state_machine.py`, `routes/thinking_mode.py`, `routes/self_improvement.py` — all already used `run_in_executor` correctly.

---

## Remaining Flags (Not Fixed — Require Architectural Decisions)

1. **CORS is `allow_origins=["*"]` in dev** (`main.py:325`) — intentional (documented in comment); WebSocket requires this. Flag for review when deploying to production.

2. **`/api/knowledge/save` has no auth** (`knowledge.py:68`) — the file comment states "No auth required (localhost-only, extension origin)". Acceptable if this is truly localhost-only, but should be documented clearly.

3. **`/api/introspection/config` exposes internal model thresholds** (`introspection.py:211`) — protected by `agent_service.agent` check but no explicit auth dependency. Low risk on local deployment.

4. **N+1 in `search_messages`** (`chat.py:334`) — iterates all conversations and calls `get_conversation_messages()` for each one. The fix added a `limit=min(limit, 100)` cap, but the underlying pattern is still O(conversations × messages). A proper fix requires a full-text search index in the persistence layer.

5. **`_session_results` dict in `reasoning_tree.py`** — shared global state with a 100-entry eviction, but eviction only removes the **oldest key by insertion order** (Python 3.7+ dict ordering). Under concurrent load this is safe but may evict active sessions. Consider a proper TTL cache.
