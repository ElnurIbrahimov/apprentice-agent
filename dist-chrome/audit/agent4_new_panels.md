# Agent 4 Audit: Compare, Deep Research, Math, Artifacts, Page Summary

**Date:** 2026-03-07
**Agent:** 4 of 5 (functional testing)
**Files audited:**
- `D:/Aura/extension/sidebar.js`
- `D:/Aura/extension/sidebar.html`
- `D:/Aura/api/routes/multi_model.py`
- `D:/Aura/api/routes/research.py`
- `D:/Aura/api/routes/math.py`
- `D:/Aura/api/routes/summarize.py`

---

## COMPARE PANEL

### Check results

| Check | Status | Notes |
|-------|--------|-------|
| `initComparePanel()` called on `switchPanel('compare')` | PASS | Line 249: `else if (name === 'compare') initComparePanel();` |
| Chip building and COMPARE_DEFAULT_MODELS pre-selected | PASS | `buildChips()` adds `.on` class and inserts into `compareSelectedModels` for default models |
| `runCompare()` POSTs to `/api/compare` with `{message, models}` | PASS | Line 2040: correct fetch with JSON body |
| Loading skeleton while waiting | PASS | Skeletons appended per model before fetch resolves |
| Result cards render with model name, timing, response | PASS | Cards built from `data.results` array |
| Fastest badge highlight | PASS | `isFastest = r.model === data.fastest && !r.error` adds `.fastest` class |
| "Send to Chat" pre-fills `#inp` | PASS (fixed) | Was missing `autoH()` call — fixed |
| All/Clear buttons | PASS | Both iterate `.cmp-chip` elements correctly |
| `/api/compare` returns `{results, fastest}` with `elapsed_ms` | PASS | `multi_model.py` returns `CompareResponse` with all three fields |

### Bugs found and fixed

**BUG 1** — `compareInitialized` never set on total fetch failure
When both Ollama and the backend `/api/models/available` are unreachable, the error handler showed "No models" text but left `compareInitialized = false`. Every subsequent `switchPanel('compare')` would trigger a new pair of fetch attempts.
**Fix:** Set `compareInitialized = true` in the innermost `.catch()` (line ~2010).

**BUG 2** — "Send to Chat" did not resize the textarea
After setting `inp.value` the `#inp` textarea stayed at its original height. `autoH()` was not called.
**Fix:** Added `autoH()` call before `switchPanel('chat')` in the `btn2` click listener (line ~2054).

---

## DEEP RESEARCH PANEL

### Check results

| Check | Status | Notes |
|-------|--------|-------|
| Depth buttons (Quick/Standard/Deep) toggle `resDepth` | PASS | `querySelectorAll('.res-d')` event listeners at line ~2001 |
| `res-go` POSTs to `/api/research` (streaming) | PASS | Fetch with body `{query, depth, model}` |
| Streaming NDJSON parser (line-by-line) | PASS | ReadableStream reader with `buf.split('\n')` accumulator pattern |
| Status updates per phase | PASS | `ev.message` shown for non-done status events |
| Source cards with clickable links | PASS | `res-src` divs with `<a>` tags using `target="_blank"` |
| Report rendered via `md()` | PASS | `resultEl.innerHTML = md(ev.report||'')` |
| Button re-enables after completion | PASS | `finally { goBtn.disabled=false; }` |
| Backend endpoint returns correct NDJSON | PASS | `research.py` yields JSON lines for searching/analyzing/writing/done phases |

### Bugs found

None.

---

## MATH PANEL

### Check results

| Check | Status | Notes |
|-------|--------|-------|
| Mode buttons toggle `mathMode` | PASS | `querySelectorAll('.math-m')` listeners |
| `math-go` POSTs to `/api/math/solve` | PASS | `{problem, mode, model}` body |
| `math-solution`, `math-latex`, `math-steps` render | PASS | All three elements populated/shown conditionally |
| Ctrl+Enter submits | PASS | `e.ctrlKey||e.metaKey` check at line ~2159 |
| `#math-result` shown as `display:flex` | PASS | `resultEl.style.display='flex'` (not `display:block`) at line ~2154 |
| Backend returns `{solution, steps, latex, graph_data}` | PASS | `math.py` normalizes and returns all four keys |

### Bugs found

None.

---

## ARTIFACTS PANEL

### Check results

| Check | Status | Notes |
|-------|--------|-------|
| `art-go` POSTs to `/api/chat` with right prompt | PASS | Sends `{message: fullPrompt, stream: false}`; `ChatRequest` accepts `message` field |
| Code fence stripping | PASS (fixed) | Regex improved |
| Iframe preview via `srcdoc` | PASS | `preview.srcdoc = code` for HTML; wrapped for SVG/Markdown |
| Preview/Code tabs toggle | PASS | `.art-tab` listeners toggle `.on` on `#art-preview` and `#art-code` |
| Copy Code | PASS | `navigator.clipboard.writeText(artCode)` |
| Send to Chat sets `pendingCtx` | PASS | Sets `pendingCtx = { text: artCode, title: 'Artifact', ... }` and calls `showCtx()` |
| `/api/chat` response field | PASS | JS reads `data.response||data.message` covering both field names |

### Bugs found and fixed

**BUG 3** — Code fence stripping regex didn't handle language names with hyphens
Original regex: `/^```[\w]*\n?/` — `\w` matches `[a-zA-Z0-9_]` but not `-`. Language identifiers like `html-css`, `js-module`, or those with dots (e.g., `.jsx`) would not be stripped.
Additionally, the closing fence regex `/\n?```$/` did not handle Windows `\r\n` line endings.
**Fix:** Updated both regexes to `[\w\-\.]` and added `\r?\n?` for CRLF tolerance.

---

## PAGE SUMMARY PANEL

### Check results

| Check | Status | Notes |
|-------|--------|-------|
| Uses `GET_PAGE_CONTENT` message | PASS | `summarizeCurrentPage()` at line ~1269: `ext.runtime.sendMessage({ type: 'GET_PAGE_CONTENT' }, ...)` |
| POSTs to `/api/summarize/page` | PASS | Fetch with `{text, url, title, format, model}` |
| Format buttons (Bullets/Paragraph/TL;DR) work | PASS | `.sum-fmt` listeners update `summaryFormat`; `data-fmt` values match `PROMPTS` keys in `summarize.py` |
| `summaryText` persists for Copy and Send to Chat | PASS | Set from `data.summary`; used in both `sum-copy` and `sum-to-chat` handlers |
| Word count and reading time badges shown | PASS | `sumWc.textContent` and `sumRt.textContent` populated from `data.word_count` / `data.reading_time_saved` |
| Badges hidden until result | PASS | `sumBadges.classList.add('on')` only after success |
| `summaryText` reset on new summarize | PASS | `summaryText = ''` at line ~1265 before fetch |

### Bugs found

None.

---

## BACKEND API CHECKS

### `/api/compare` (multi_model.py)

- Returns `CompareResponse` with `results: List[ModelResult]`, `fastest: str`, `query: str`. PASS.
- `ModelResult` has `elapsed_ms: int` — matches what JS reads (`r.elapsed_ms`). PASS.
- `fastest` field contains the model name of the fastest non-error result. PASS.

### `/api/research` (research.py)

- Returns `StreamingResponse` with `media_type="text/plain"` (NDJSON).
- Each line is valid JSON with `status` field.
- Final line has `status: "done"` with `report`, `sources`, `citations`. PASS.

### `/api/math/solve` (math.py)

- Accepts `{problem, mode, model}`.
- Returns `{solution, steps, latex, graph_data}` with normalization for missing keys. PASS.
- JSON extraction handles markdown-fenced LLM output via `_extract_json()`. PASS.

### `/api/summarize/page` (summarize.py)

- Accepts `{text, url, title, format, model}`.
- Returns `{summary, word_count, reading_time_saved, truncated}`. PASS.
- `format` values `"bullets"`, `"paragraph"`, `"tldr"` — match HTML `data-fmt` attributes. PASS.

---

## FIXES APPLIED

### `D:/Aura/extension/sidebar.js`

1. **`compareInitialized` on error** (~line 2010): Added `compareInitialized = true` in the innermost catch of `initComparePanel()` to prevent infinite refetch loop on error.

2. **Compare "Send to Chat" textarea resize** (~line 2054): Added `autoH()` call before `switchPanel('chat')` so the pre-filled textarea expands to show content.

3. **Artifacts code fence regex** (~line 2200): Changed stripping regex from `/^```[\w]*\n?/` to `/^```[\w\-\.]*\r?\n?/` and closing from `/\n?```$/` to `/\r?\n?```[\w\-\.]*\s*$/` to handle hyphenated language names and CRLF line endings.

---

## SUMMARY

- **Panels verified:** 5 of 5
- **Bugs found:** 3
- **Bugs fixed:** 3
- **Backend endpoints verified:** 4 of 4
- **All panel flows are functional** after fixes applied.
