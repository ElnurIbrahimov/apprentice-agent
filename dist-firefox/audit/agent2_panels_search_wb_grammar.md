# Agent 2 Audit Report — Search, Wisebase, Grammar, Write, Translate Panels

**Date:** 2026-03-07
**Files audited:**
- `D:/Aura/extension/sidebar.js`
- `D:/Aura/extension/sidebar.html`
- `D:/Aura/api/routes/search.py`
- `D:/Aura/api/routes/knowledge.py`

---

## SEARCH PANEL

### Checklist

| Item | Status | Notes |
|------|--------|-------|
| Uses HTTP fetch (not WebSocket) | PASS | `doSearch()` uses `fetch(searchUrl)` |
| Hits `/api/search?q=...` | PASS | URL: `` `${HTTP}/api/search?q=${encodeURIComponent(q)}&limit=5` `` |
| Renders source cards with title, snippet, URL | FAIL (fixed) | Title and URL were rendered; snippet was silently dropped |
| Enter key triggers search | PASS | `keydown` listener on `search-inp` checks `e.key === 'Enter'` |
| Loading state while fetching | PASS | `searchLoading.classList.add('on')` before fetch, `.remove('on')` after |
| Search errors shown clearly | PASS | catch block sets `searchEmpty` text to `'Search error: ' + err.message` |

### Bug Found and Fixed

**Bug:** Source cards omitted the `snippet` field.

The API (`search.py`) returns `snippet: r.get("content", "")[:200]` for each result. The frontend card template only rendered `src-card-n` (index), `src-card-t` (title), and `src-card-u` (URL). The snippet was never displayed.

**Fix applied in `sidebar.js` (~line 521):**
Added a conditional `src-card-s` div between title and URL that renders `src.snippet` when present.

**Fix applied in `sidebar.html`:**
Added `.src-card-s` CSS rule: `font-size:11px`, `color:var(--mu)`, 2-line clamp.

---

## WISEBASE PANEL

### Checklist

| Item | Status | Notes |
|------|--------|-------|
| `loadWisebase()` calls `/api/knowledge/list` on panel open | PASS | `switchPanel('wisebase')` calls `loadWisebase()`, which fetches `/api/knowledge/list?limit=20` |
| Search calls `/api/knowledge/search?q=...` | PASS | When query is non-empty, fetches `/api/knowledge/search?q=...&limit=20` |
| Delete buttons call `DELETE /api/knowledge/{id}` | PASS | `fetch(..., { method: 'DELETE' })` using `encodeURIComponent(item.episode_id)` |
| Clicking card sets `pendingCtx` and switches to chat | PASS | Click handler sets `pendingCtx = { text, title, url }`, calls `showCtx()` and `switchPanel('chat')` |
| Empty state shown when no clips exist | PASS | `renderWbCards([])` renders `wb-empty` div with "No saved clips found." |

**No bugs found.**

---

## GRAMMAR PANEL

### Checklist

| Item | Status | Notes |
|------|--------|-------|
| 3 mode buttons toggle properly | PASS | `.gr-mode` buttons remove/add `.on` class; `grMode` state updates |
| Check button sends via WebSocket | PASS | `ws.send(JSON.stringify(...))` in `gr-btn` click handler |
| Splits on `---CHANGES---` to show diff | FAIL (fixed) | Off-by-one: was slicing at `sep + 14` instead of `sep + 13` |
| Inline diff renders (red strikethrough / green additions) | PASS | `renderWordDiff()` wraps deletions in `.gr-del` and insertions in `.gr-ins`; CSS has correct styles |

### Bug Found and Fixed

**Bug:** `---CHANGES---` separator slice was off by one.

The string `---CHANGES---` is exactly 13 characters. The code used `fullText.slice(sep + 14)` to get the text after the separator, which would skip one extra character from the changes content — causing the first character of the changes text to be dropped on every grammar check.

**Fix applied in `sidebar.js` (~line 936):**
```js
// Before (wrong):
const changesRaw = fullText.slice(sep + 14).trim();

// After (correct):
const changesRaw = fullText.slice(sep + 13).trim();
```

---

## WRITE PANEL

### Checklist

| Item | Status | Notes |
|------|--------|-------|
| Compose/Improve modes work | PASS | `.wtab` buttons toggle `writeTab`; prompt differs for each mode |
| Output streams into result area | PASS | `activeStream` type `'write'` renders via `el.innerHTML = md(rawText)` on each chunk |
| Copy button present | FAIL (fixed) | No copy button existed for the write result |

### Bug Found and Fixed

**Bug:** No copy button for write output.

The `#write-foot-l` div in the HTML was empty — no copy button. After a stream completed, there was no way to copy the generated text.

**Fix applied in `sidebar.js`:** Added `onDone` callback to the write panel's `activeStream` object. After the stream finishes, a "Copy" button is dynamically appended to `#write-result`. It copies `rawText` to clipboard and shows a brief "Copied!" confirmation with green color, then reverts.

---

## TRANSLATE PANEL

### Checklist

| Item | Status | Notes |
|------|--------|-------|
| Language selection works | PASS | `#tr-from` and `#tr-to` `<select>` elements; values read in `trBtn` click handler |
| Translation streams properly | PASS | Uses WebSocket `activeStream` type `'translate'`; chunks write to `trOut.textContent` |
| OCR-to-translate works | PASS | `#ocr-translate` click sets `$('tr-inp').value = lastOcrText` then `switchPanel('translate')` |

**No bugs found.**

---

## Summary of Changes Made

| File | Change |
|------|--------|
| `sidebar.js` | **Search:** Added `src.snippet` rendering in source card HTML (with conditional guard) |
| `sidebar.js` | **Grammar:** Fixed off-by-one in `---CHANGES---` separator slice: `sep + 14` → `sep + 13` |
| `sidebar.js` | **Write:** Added `onDone` callback that appends a functional copy button to `#write-result` after streaming completes |
| `sidebar.html` | **Search:** Added `.src-card-s` CSS class for snippet text styling |

---

## No Changes Needed

- Wisebase panel: fully functional
- Translate panel: fully functional (including OCR integration)
- Grammar diff rendering: correct (`.gr-del` / `.gr-ins` CSS, `renderWordDiff` logic)
- Search loading/error states: correct
- Wisebase empty state: correct
- Grammar mode button toggling: correct
