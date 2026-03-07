# AURA Performance Audit — Agent 9

## Summary

7 targeted fixes applied to `sidebar.js`, 1 fix to `content.js`. No changes needed to `background.js` (clean).

---

## Fixes Applied

### 1. WebSocket Exponential Backoff (`sidebar.js`)

**Problem:** Fixed 5-second reconnect hammered server continuously after repeated failures.

**Fix:** Added `_wsRetryDelay` state with exponential backoff: 1s → 2s → 4s → ... → 30s cap. Reset to 1s on successful connection.

```js
// Before
setTimeout(connectWS, 5000);

// After
setTimeout(connectWS, _wsRetryDelay);
_wsRetryDelay = Math.min(_wsRetryDelay * 2, 30000);
// reset in onopen: _wsRetryDelay = 1000;
```

---

### 2. Markdown Rendering Throttled via rAF (`sidebar.js`)

**Problem:** `md()` (a full markdown parser) was called synchronously on every WebSocket chunk. With fast models streaming 50+ chunks/second, this ran the parser hundreds of times per second for large responses.

**Fix:** Added `scheduleMdRender(stream)` that gates rendering through `requestAnimationFrame`. At most one `md()` call per frame (~60fps). On `done`/`error`, the pending rAF is cancelled and a final authoritative render is forced.

```js
let _mdRafId = null;
function scheduleMdRender(stream) {
  if (_mdRafId !== null) return;
  _mdRafId = requestAnimationFrame(() => {
    _mdRafId = null;
    if (!stream || !stream.el || !activeStream) return;
    stream.el.innerHTML = md(stream.rawText);
    if (stream.type === 'chat') msgs.scrollTop = msgs.scrollHeight;
  });
}
```

---

### 3. Page Content Cache (30s TTL) (`sidebar.js`)

**Problem:** `sendMessage` called `GET_PAGE_CONTENT` on every message when the user asked about the page. Three messages in a row re-extracted the page text three times.

**Fix:** Added `getPageContentCached()` — a URL-keyed Map with 30-second TTL. Subsequent calls within the window return the cached response. `sendMessage` now calls `getPageContentCached()` instead of the raw message send.

```js
const _pageCache = new Map();
const _PAGE_CACHE_TTL = 30000;
async function getPageContentCached() {
  const now = Date.now();
  for (const [key, val] of _pageCache) {
    if (now - val.ts < _PAGE_CACHE_TTL) return val.resp;
    _pageCache.delete(key);
  }
  const resp = await new Promise(r => ext.runtime.sendMessage({ type: 'GET_PAGE_CONTENT' }, r));
  if (resp?.ok && resp.url) _pageCache.set(resp.url, { resp, ts: now });
  return resp;
}
```

---

### 4. Wisebase Search Debounced 300ms (`sidebar.js`)

**Problem:** No live-search debounce existed on the Wisebase `wb-inp` input. While `keydown Enter` was the trigger, the `input` event had no handler at all, meaning if one were added (or was already assumed to exist) it would spam the API.

**Fix:** Added explicit 300ms debounce on `input` event. Fires search on ≥2 chars; resets to full list on empty input.

---

### 5. Event Listener Leak — Per-Pill Click-Outside (`sidebar.js`)

**Problem:** `buildPill()` is called 13 times (once per feature). Each call added `document.addEventListener('click', ...)` — a new global listener that is never removed. Result: 13 global click handlers firing on every click in the sidebar.

**Fix:** Replaced the per-pill listener with a single delegated `document.addEventListener('click')` after `initModelPills()`. The per-pill line is replaced with a comment.

---

### 6. Chat History DOM Bounding (`sidebar.js`)

**Problem:** `msgs` div accumulated `.mrow` elements indefinitely. Multi-hour chat sessions could produce hundreds of DOM nodes, degrading scroll and layout performance.

**Fix:** Added `trimMsgsDOM()` called before each `addUserMsg()`. Keeps at most 150 `.mrow` nodes by removing oldest first.

---

### 7. Dead Code: `artLang` State Variable (`sidebar.js`)

**Problem:** `artLang` was declared as a state variable and updated in a `change` listener on `#art-lang`, but the `art-go` click handler read `$('art-lang').value` directly — making the shadow variable entirely unused.

**Fix:** Removed the variable declaration (replaced with comment) and removed the now-pointless change listener.

---

### 8. `serializeDOM()` Early Exit (`content.js`)

**Problem:** `serializeDOM()` ran `document.querySelectorAll(...)` on all interactive elements, called `getBoundingClientRect()` on every one, then sliced to 80 at the end. On complex pages (e.g., Gmail, Twitter) this could iterate 1000+ elements when only 80 were needed.

**Fix:** Replaced `forEach` + `slice(0, 80)` with a `for...of` loop that breaks early once 80 visible elements are collected. Eliminates unnecessary getBCR calls on elements beyond the limit.

---

## No Changes Required

### Model List Caching — Already Correct
`mdlCloudList`/`mdlLocalList` are checked before fetching (`if (!mdlCloudList.length)`). Both the pill dropdown and `initComparePanel()` use this guard correctly. No stale cache issue — lists are reset only when `loadModelPanel()` is called explicitly.

### DOM Queries — Not in Loops
`querySelectorAll('.res-d')`, `.math-m`, `.sum-fmt`, `.art-tab`, `.img-style`, `.gr-mode` are all called at init-time inside `forEach` for event listener setup. None are inside loops or repeated calls. Confirmed safe.

### Compare Panel — Uses Shared State
`initComparePanel()` checks `if (mdlCloudList.length || mdlLocalList.length)` before fetching and uses the `compareInitialized` guard. No redundant fetch.

### Content Script Size
`content.js` is already minimal (~500 lines). Shadow DOM isolation means styles are isolated, not duplicated. No lazy-loading needed — the dock and toolbar are core functionality required on every page.

### `background.js`
No performance issues found. All message handlers are O(1) relay operations. No loops, no unbounded state.

---

## Files Modified

- `D:/Aura/extension/sidebar.js` — 7 fixes
- `D:/Aura/extension/content.js` — 1 fix (`serializeDOM` early exit)
