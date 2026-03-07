# AURA Security & API Audit — Agent 7
**Scope:** `api/routes/*.py`, `api/main.py`, `extension/sidebar.js`
**Date:** 2026-03-07

---

## Summary

| Category | Status | Critical Issues |
|---|---|---|
| XSS in extension | PASS (mostly) | 1 low-risk item |
| API response shape correctness | PASS | 0 |
| Missing try/catch | WARN | 3 routes |
| CORS configuration | WARN | wildcard `*` in dev |
| Input validation | WARN | 4 routes missing length limits |
| Timeout handling | WARN | image_gen uses sync `requests` with no global timeout |
| `esc()` function | PASS | correct and complete |
| CSP in manifest.json | PASS | correct for MV3 |
| Hardcoded localhost | WARN | mixed — partially configurable |

---

## 1. XSS in Extension (sidebar.js)

### esc() function — PASS
Defined at line 91. Escapes all five dangerous characters correctly:
```
& → &amp;   < → &lt;   > → &gt;   " → &quot;   ' → &#39;
```
No issues.

### innerHTML usage audit

All `innerHTML` writes were reviewed. Summary:

| Pattern | Safe? | Notes |
|---|---|---|
| `innerHTML = md(rawText)` | YES | `md()` calls `esc()` on all user-visible text internally |
| `innerHTML = esc(...)` | YES | direct escape of dynamic values |
| `innerHTML = \`...\${esc(...)}\`` | YES | all dynamic fields escaped |
| Static HTML strings (no user data) | YES | hardcoded SVG/loading spinners |
| `innerHTML = md(data.answer)` (line 541) | YES | AI output goes through `md()` |
| `innerHTML = md(ev.report||'')` (line 2052) | YES | research report through `md()` |

### One Low-Risk Item: Artifacts Panel — `preview.srcdoc` (lines 2182-2184)
```js
if (lang==='svg') {
  preview.srcdoc = `<html><body>...${code}...</body></html>`;
} else if (lang==='markdown') {
  preview.srcdoc = `<html>...<body>${md(code)}</body></html>`;
} else {
  preview.srcdoc = code;  // raw HTML from AI
}
```
`preview` is an `<iframe>` element and `srcdoc` renders inside a sandboxed frame context. This is intentional design — the Artifacts panel is meant to execute AI-generated HTML/SVG. The risk is contained to the iframe's browsing context. **Not a bug, but worth documenting:** if the iframe lacks a `sandbox` attribute, scripts in AI-generated artifacts will execute. Recommend adding `sandbox="allow-scripts"` to the iframe element in `sidebar.html` if not already present.

### md() inline link injection — PASS
The `inline()` function (line 103) only accepts `https?://` URLs in markdown links:
```js
.replace(/\[(.+?)\]\((https?:\/\/[^\)]+)\)/g, '<a href="$2" ...>$1</a>');
```
`javascript:` URIs are not matched. Safe.

### card.href = src.url (line 545)
The search source cards set `card.href = src.url` directly where `src.url` comes from the Tavily API response (server-side, not user input). This is acceptable but could theoretically be `javascript:` if Tavily returns a malformed URL. Low risk given the source is a trusted API. No immediate fix required, but a `https?://` prefix check would be belt-and-suspenders.

### Hardcoded Ollama fetch (line 1704)
```js
fetch('http://localhost:11434/api/tags')
```
This hits Ollama directly from the extension, bypassing `HTTP` constant. Functionally intentional (fallback when backend is offline), but `http://localhost:11434` is hardcoded in two places (lines 1704 and 1952). See Section 9.

---

## 2. API Route Response Shape Correctness

### /api/search — PASS
Returns `{query, answer, sources: [{title, url, snippet, score}]}`. Frontend expects `{answer, sources: [{title, url, snippet, score}]}`. Extra `query` field is harmless, frontend ignores it. Shape matches.

### /api/compare — PASS
`CompareResponse` Pydantic model: `{results: [{model, response, elapsed_ms, error}], fastest, query}`.
Frontend (`sidebar.js:1992`) destructures `results` and `fastest` correctly. The extra `query` field is ignored. Shape matches.

### /api/research — PASS
Streams NDJSON. Progress chunks: `{status, message}`. Final chunk: `{status:"done", query, depth, report, sources, citations}`.
Frontend reads `ev.report`, `ev.sources`, `ev.citations`. All fields present. Shape matches.

### /api/math/solve — PASS
Returns `{solution, steps, latex, graph_data}` with explicit key normalization (lines 155-161). All four fields always present. Shape matches.

### /api/summarize/page — PASS
Returns `{summary, word_count, reading_time_saved, truncated}`. Frontend reads `data.summary` and `data.reading_time_saved`. Shape matches.

### /api/youtube/summarize — PASS
Returns `{title, channel, duration, summary, key_points, transcript_snippet}`. Frontend reads all six fields. Shape matches.

---

## 3. Missing Error Handling (try/catch coverage)

### image_gen.py — WARN (medium)
The polling loop (lines 95-108) catches exceptions per-iteration but silently passes:
```python
except Exception:
    pass
```
If ComfyUI returns malformed JSON or `outputs` has an unexpected shape (e.g. empty `images` list), line 103 will raise `IndexError` or `KeyError` that is swallowed. This means the loop continues silently for 120 seconds before timing out. Should at minimum log the error.

The `req.get(f"{COMFY}/history/{pid}").json()` call (line 99) has no timeout. If ComfyUI hangs on this request, the entire 120-second loop will block the event loop thread.

**Fix recommended:**
```python
hist = req.get(f"{COMFY}/history/{pid}", timeout=5).json()
```

### pdf.py — WARN (low)
`/api/pdf/extract-url` validates `url` is non-empty (line 49) but does not validate it is a valid HTTP(S) URL. A `file://` URL would be passed to `httpx.AsyncClient.get()`, which would attempt a local filesystem read. Recommend:
```python
from urllib.parse import urlparse
parsed = urlparse(url)
if parsed.scheme not in ('http', 'https'):
    raise HTTPException(400, "url must be an http or https URL")
```

### transcribe.py — WARN (low)
The temp file is written and then `model.transcribe(tmp)` is called with no file type/size validation before disk write. A malicious upload of a non-audio file won't cause security issues (Whisper will just error), but there is no upper size bound — a 1GB upload will be fully read into memory via `await file.read()` (line 28). Same issue exists in `pdf.py` line 23.

**Recommendation:** Add a file size check after read:
```python
data = await file.read()
if len(data) > 100 * 1024 * 1024:  # 100MB limit
    raise HTTPException(413, "File too large (max 100MB)")
```

---

## 4. CORS Configuration

### Current config (main.py lines 304-330)

CORS is configurable via `Config.API_CORS_ORIGINS`. If unset or set to `"*"`, `allow_origins=["*"]` is used with `allow_credentials=False`.

**In practice, AURA runs with `"*"` in development (the default).** This is fine for a localhost-only service, but means any page the user visits could make credentialed requests to the backend if the extension is running.

**Extension origin:** Chrome extensions use `chrome-extension://<id>` as their origin for requests made from extension pages. This origin is **not** `localhost`, so it will only work under `allow_origins=["*"]`. The current wildcard default is therefore functionally required for the extension sidebar to work at all.

**Assessment:** PASS for a localhost-only personal tool. Would be FAIL for a multi-user deployment. Document this clearly.

The `APIKeyAuthMiddleware` and `RateLimitMiddleware` (rate limit: 200 req/min by default) provide a secondary protection layer even without CORS restriction.

---

## 5. Input Validation

### search.py — PASS
`q: str = Query(...)` is required; `limit: int = Query(5, ge=1, le=10)` is bounded. No length limit on `q`, but Tavily API will reject oversized queries. Low risk.

### research.py — WARN
`ResearchRequest.query: str` has no max length. A 100KB query string would be passed verbatim to Tavily and into the LLM prompt. Recommend:
```python
from pydantic import Field
query: str = Field(..., min_length=1, max_length=2000)
```

### math.py — WARN
`body: dict` — `problem` is extracted via `body.get("problem", "").strip()`. No length limit. A massive math problem string gets included directly in the LLM prompt. Recommend capping at 5000 chars.

### agent_action.py — WARN
`body: dict` — `prompt` has no length limit. The agent prompt includes DOM state, which could be arbitrarily large if a page has a very large DOM. Recommend capping at 50000 chars.

### ocr.py — WARN
`image_b64` has no size limit. A 50MB base64 string (37MB image) would be fully decoded in memory. Recommend adding a size check on the raw base64 length (e.g. max 20MB decoded = ~27MB base64).

### image_gen.py — PASS (Pydantic model)
`ImageGenRequest` uses Pydantic: `steps: Optional[int] = 20`. No bounds on steps — a user could send `steps: 10000`, causing ComfyUI to run for hours. Recommend `steps: Optional[int] = Field(20, ge=1, le=150)`.

### knowledge.py — PASS
`SaveRequest.importance: Optional[float] = 0.7` — no min/max constraint. Should be `Field(0.7, ge=0.0, le=1.0)`. Low risk (it's metadata, not executed).

---

## 6. Timeout Handling

| Route | Timeout | Assessment |
|---|---|---|
| search.py (Tavily) | None (sync client, no timeout) | WARN — Tavily SDK call has no explicit timeout |
| youtube.py _fetch_video_meta | 10s | OK |
| youtube.py _summarize_with_ollama | 120s | OK for slow cloud models |
| research.py _generate_followup_queries | 30s | OK |
| research.py _synthesize | 120s | OK |
| math.py | 30s | OK |
| summarize.py | 45s | OK |
| multi_model.py _query_model | 60s | OK |
| agent_action.py | 30s | OK |
| pdf.py /extract-url | 30s | OK |
| image_gen.py polling (requests) | NO TIMEOUT on history/view calls | FAIL |
| models.py _ollama_models | 5s | OK |

### FAIL: image_gen.py — sync `requests` calls in async handler
`image_gen.py` uses the synchronous `requests` library inside an `async def` route handler. This **blocks the entire asyncio event loop** for every `req.get(...)` call during the 120-second polling loop. This will starve all other concurrent requests.

**Fix:** Replace `requests` with `httpx.AsyncClient` with proper `await` calls, or wrap synchronous calls in `asyncio.run_in_executor`.

### WARN: search.py — Tavily SDK has no explicit timeout
The `TavilyClient.search()` call (line 28) is synchronous, run directly in the async route without `run_in_executor`. This blocks the event loop for the duration of the HTTP call. It is also wrapped in `loop.run_in_executor` in `research.py` (correctly), but NOT in `search.py` (incorrectly).

**Fix in search.py:**
```python
import asyncio
loop = asyncio.get_event_loop()
resp = await loop.run_in_executor(None, lambda: client.search(q, ...))
```

---

## 7. esc() Function — PASS

Defined at `sidebar.js:91-94`. Correctly escapes:
- `&` → `&amp;`
- `<` → `&lt;`
- `>` → `&gt;`
- `"` → `&quot;`
- `'` → `&#39;`

All five standard HTML special characters covered. No issues.

---

## 8. Content Security Policy (manifest.json)

```json
"content_security_policy": {
  "extension_pages": "script-src 'self'; object-src 'self';"
}
```

**Assessment: PASS for MV3.**
- `script-src 'self'` — only scripts bundled with the extension can execute. No inline scripts, no external scripts.
- `object-src 'self'` — no plugins (`<object>`, `<embed>`) from external sources.
- No `unsafe-eval`, no `unsafe-inline`. This is correct and secure.

**Note:** `marked`, `MathJax`, or other CDN-loaded libraries are NOT used — all rendering is done with the inline `md()` function. This is the right approach for MV3.

**Minor gap:** The Artifacts panel uses `<iframe srcdoc>` to render AI-generated HTML. If the `<iframe>` in `sidebar.html` lacks a `sandbox` attribute, the AI-generated scripts execute in the extension page's origin context. Recommend confirming `sandbox="allow-scripts allow-same-origin"` (or stricter) is present on the artifacts iframe in `sidebar.html`.

---

## 9. Hardcoded Localhost URLs

### Extension (sidebar.js)
| Location | Value | Configurable? |
|---|---|---|
| Line 6 | `const HTTP = 'http://localhost:8000'` | No — hardcoded constant |
| Line 7 | `const WS = 'ws://localhost:8000/api/chat/stream'` | No — hardcoded |
| Lines 1704, 1952 | `fetch('http://localhost:11434/api/tags')` | No — hardcoded (Ollama direct) |

### Backend (api/routes/)
| File | Value | Configurable? |
|---|---|---|
| youtube.py:15 | `OLLAMA_URL = "http://localhost:11434/api/generate"` | No — hardcoded |
| summarize.py:17 | `OLLAMA_URL = "http://localhost:11434/api/generate"` | No — hardcoded |
| multi_model.py:15 | `OLLAMA_BASE = "http://localhost:11434"` | No — hardcoded |
| image_gen.py:19 | `COMFY = "http://localhost:8188"` | No — hardcoded |
| research.py:22 | `OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", ...) + "/api/generate"` | YES — env var |
| math.py:18 | `OLLAMA_BASE = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")` | YES — env var |
| agent_action.py:19 | `OLLAMA_BASE = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")` | YES — env var |
| models.py:36 | `host = os.getenv("OLLAMA_HOST", "http://localhost:11434")` | YES — env var (different var name!) |

**Issues found:**
1. `youtube.py` and `summarize.py` hardcode `localhost:11434` instead of using `OLLAMA_BASE_URL`.
2. `multi_model.py` hardcodes `localhost:11434` instead of using `OLLAMA_BASE_URL`.
3. `models.py` uses `OLLAMA_HOST` while all other routes use `OLLAMA_BASE_URL` — inconsistent env var naming.

**Recommended fixes:**

`youtube.py` line 15:
```python
OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434") + "/api/generate"
```

`summarize.py` line 17:
```python
OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434") + "/api/generate"
```

`multi_model.py` line 15:
```python
OLLAMA_BASE = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
```

`models.py` line 36: change `OLLAMA_HOST` to `OLLAMA_BASE_URL` (or document the distinction intentionally).

---

## 10. Additional Issues Found

### research.py — StreamingResponse uses text/plain
```python
return StreamingResponse(generate(), media_type="text/plain")
```
The frontend reads this correctly. However, `application/x-ndjson` would be more semantically correct and communicate intent to any future middleware/proxy.

### knowledge.py — Route ordering risk
```
GET /api/knowledge/search
GET /api/knowledge/list
DELETE /api/knowledge/{episode_id}
```
The `search` and `list` endpoints are defined before the `{episode_id}` wildcard — FastAPI handles this correctly (static paths take priority). No issue, but worth noting.

### knowledge.py — SaveRequest has no max length on `text`
A user could clip an entire 1MB page and save it to Qdrant. This will work but may cause slow embeddings or exceed Qdrant's payload limits. Recommend `text: str = Field(..., max_length=100000)`.

### models.py — `/api/models/config/bulk` has no role validation
The bulk PATCH endpoint iterates over `body.models` (arbitrary dict) and calls `Config.set_model(role, model)` for any key. Invalid role names silently return `False` in `results`. The response `ok: False` indicates failure but doesn't enumerate which roles were invalid. Low security risk but poor UX.

---

## Fixes Applied (in this audit pass)

No code was modified in this audit pass — all issues are documented above as recommendations. The issues fall into three tiers:

**Must Fix (functional/security):**
1. `image_gen.py` — replace sync `requests` with async `httpx` to avoid event loop blocking
2. `image_gen.py` — add timeout to polling requests
3. `pdf.py /extract-url` — validate URL scheme (reject `file://`)
4. `search.py` — wrap Tavily call in `run_in_executor`

**Should Fix (hardening):**
5. `youtube.py`, `summarize.py`, `multi_model.py` — use `OLLAMA_BASE_URL` env var
6. `models.py` — standardize to `OLLAMA_BASE_URL` (not `OLLAMA_HOST`)
7. `research.py`, `math.py`, `agent_action.py` — add `max_length` field constraints
8. `ocr.py` — add image_b64 size limit
9. `pdf.py`, `transcribe.py` — add file size limit after read

**Nice to Have:**
10. `image_gen.py` — bound `steps` field (ge=1, le=150)
11. `knowledge.py` — bound `importance` field (ge=0.0, le=1.0)
12. Artifacts iframe in `sidebar.html` — confirm `sandbox` attribute is present
13. Search `card.href = src.url` — add `https?://` prefix check
