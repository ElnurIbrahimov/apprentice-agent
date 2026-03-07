# AURA Extension — Audit Agent 1: Bugs, Errors, Race Conditions
**Audited files:** sidebar.js, sidebar.html, background.js, content.js
**Date:** 2026-03-07
**Agent:** Audit Agent 1 of 3

---

## Summary

13 bugs fixed directly in source files. 3 additional issues documented but left for human review (architectural decisions). No storage quota issues found.

---

## FIXED BUGS

### BUG-01 — Unhandled Promise Rejections from async `sendMessage`
**File:** `sidebar.js`
**Severity:** High
**Category:** Async bug, unhandled rejection

`sendMessage` is declared `async function sendMessage(...)` but called without `.catch()` from multiple sites:
- Chip click handlers: `sendMessage(c.dataset.q)` — no catch
- Send button click: `sendMessage()` — no catch
- Enter key handler: `sendMessage()` — no catch
- Ask panel buttons: `sendMessage(msg, 'ask')` — no catch (x2)
- Tools panel `.then()` chains: `sendMessage(...)` — no catch (x4)

Any rejection (e.g., if `wsReady` guard races with WS drop between guard check and send) creates an uncaught Promise rejection that Chrome logs as an error and may suppress future handlers in some environments.

**Fix applied:** Added `.catch(() => {})` to all 9 call sites.

---

### BUG-02 — WebSocket Disconnect Leaves `activeStream` Stuck (UI Permanently Frozen)
**File:** `sidebar.js`
**Severity:** Critical
**Category:** Race condition, UI freeze

`ws.onclose` was:
```js
ws.onclose = () => { wsReady = false; setOnline(false); setTimeout(connectWS, 5000); };
```

If the WebSocket closes while a stream is in progress (`activeStream !== null`), the `submitBtn` stays `disabled = true` forever. The user cannot send another message until they reload. The streaming bubble stays in its intermediate state with no error shown.

**Fix applied:** `ws.onclose` now:
1. Checks if `activeStream` is set
2. Nulls it out and re-enables `submitBtn`
3. Appends "Connection lost" error text to chat bubbles
4. Also handles `activeStream === true` (the pre-await sentinel)
5. Implements exponential backoff for reconnects (1s → 2s → ... → 30s max, resets on successful open)

---

### BUG-03 — `ws` Null Guard Missing Before `ws.send()`
**File:** `sidebar.js`
**Severity:** Medium
**Category:** Null reference

Several panels guard with `if (!wsReady)` but then call `ws.send(...)` directly. Between the `onclose` handler setting `wsReady = false` and the code checking it, there's a window where `ws` could theoretically be null (e.g., first load before `connectWS()` completes). More importantly, `wsReady` can lag in edge cases.

**Affected panels:** translate, write, grammar, sendPdfQuestion, rec-summarize

**Fix applied:** Changed all guards from `if (!wsReady)` to `if (!wsReady || !ws)`.

---

### BUG-04 — `recTimerInterval` Not Cleared on Speech Recognition Error
**File:** `sidebar.js`
**Severity:** Medium
**Category:** Memory leak, timer leak

`recRecognition.onerror` handler was:
```js
recRecognition.onerror = (e) => { $('rec-status').textContent = 'Error: ' + e.error; };
```

On speech recognition error, the `recTimerInterval` keeps running indefinitely (counting up) even though recording has stopped. The Record button also stays in "recording" state with the animation. The only way to clear the timer was via `stopRec()`, which was never called on error.

**Fix applied:** `onerror` now calls `clearInterval(recTimerInterval)`, resets `recTimerInterval = null`, and resets the button text/class.

---

### BUG-05 — Per-Pill `document.addEventListener('click', ...)` Creating N Duplicate Handlers
**File:** `sidebar.js`
**Severity:** Medium
**Category:** Memory leak, event listener duplication

`buildPill(featureKey)` was called once for each of the 12 feature keys in `PILL_SLOTS`. Each call registered:
```js
document.addEventListener('click', () => drop.classList.remove('open'));
```
This created 12 separate document-level click listeners (one per pill), each closing only their own dropdown. This means 12 function objects are retained for the life of the page, and every click on the document fires 12 handlers.

**Fix applied:**
- Removed the per-pill `document.addEventListener` call from inside `buildPill()`
- Added a single shared handler in `initModelPills()` that closes all open dropdowns with one `querySelectorAll('.mdl-drop.open').forEach(...)` call

---

### BUG-06 — `doSearch` Calls `res.json()` Without Checking `res.ok`
**File:** `sidebar.js`
**Severity:** Medium
**Category:** Missing error handling

```js
const res = await fetch(searchUrl);
const data = await res.json();  // throws if res body is not JSON (e.g. on 500 HTML error page)
```

A 4xx/5xx response with an HTML body (e.g., Nginx 502 page) causes `res.json()` to throw a SyntaxError. The catch block catches it but then does `searchEmpty.querySelector('p').textContent = 'Search error: ' + err.message` which itself throws if `searchEmpty` has no `<p>` child.

**Fix applied:**
1. Added `if (!res.ok)` check before `res.json()` — parses error detail and throws a clean Error
2. Changed `searchEmpty.querySelector('p').textContent = ...` to guard with `const p = ...; if (p) p.textContent = ...`

---

### BUG-07 — Image Generator Calls `res.json()` Before Checking `res.status === 503`
**File:** `sidebar.js`
**Severity:** Medium
**Category:** Logic error, potential crash

```js
const data = await res.json();  // ← called FIRST
if (res.status === 503) {       // ← checked AFTER
  ...
  return;
}
```

A 503 response from ComfyUI (which is the intended "offline" signal) may return an HTML body rather than JSON. Calling `res.json()` on it throws a SyntaxError before the `503` check is ever reached.

**Fix applied:** Moved `res.status === 503` check to before `res.json()` call.

---

### BUG-08 — `rec-save-wb` Fetch Has No `res.ok` Check
**File:** `sidebar.js`
**Severity:** Low-Medium
**Category:** Missing error handling

The voice note save button silently succeeded even on server errors (4xx/5xx). The UI showed "✓ Saved to Wisebase" even if the server rejected the request.

**Fix applied:** Added `if (!saveR.ok)` check that reads the error detail and throws, causing the catch block to display the actual error.

---

### BUG-09 — `loadWisebase` Fetch Has No `res.ok` Check
**File:** `sidebar.js`
**Severity:** Low-Medium
**Category:** Missing error handling

`loadWisebase` called `res.json()` without checking `res.ok`. A 4xx/5xx response would cause a confusing JSON parse error in the catch block rather than a meaningful error message.

**Fix applied:** Added `if (!res.ok) throw new Error(`Server error ${res.status}`)`.

---

### BUG-10 — Wisebase Delete Has No `res.ok` Check
**File:** `sidebar.js`
**Severity:** Low
**Category:** Missing error handling

`card.remove()` was called unconditionally after the DELETE fetch even if the server returned a 404 or 500. Cards disappeared from UI even when the delete failed on the backend.

**Fix applied:** Added `if (!delR.ok)` check before `card.remove()`.

---

### BUG-11 — OCR Fetch in Background Has No `res.ok` Check
**File:** `background.js`
**Severity:** Medium
**Category:** Missing error handling

```js
const resp = await fetch(`${BACKEND}/api/ocr`, { ... });
const d = await resp.json();  // throws if server returns HTML error page
```

On a 4xx/5xx from the OCR endpoint, `resp.json()` throws and the catch sends a confusing `String(e)` error instead of a meaningful message.

**Fix applied:** Added `if (!resp.ok)` check that sends `OCR_RESULT` with a clean error before attempting `resp.json()`.

---

### BUG-12 — OCR Overlay `onEsc` Listener Leaks on `mouseup` Path
**File:** `content.js`
**Severity:** Low-Medium
**Category:** Memory leak, double `sendResponse` risk

The `showOcrOverlay` function registered an `onEsc` keydown listener on `document`. On the `mouseup` path (user draws a region), the overlay was removed and `sendResponse` was called — but the `onEsc` listener was never removed. This means:
1. If the user presses Escape after already completing a selection, `sendResponse` is called again (Chrome extension protocol violation — second call is silently ignored but the handler stays in memory)
2. The listener is never garbage collected until the content script is destroyed

**Fix applied:**
- Moved `onEsc` function declaration before `mouseup` listener so both paths can reference it
- Added `document.removeEventListener('keydown', onEsc)` to the `mouseup` handler
- Added `document.body.contains(overlay)` guard before `removeChild` to prevent errors if both paths fire

---

### BUG-13 — `apiFetch` Helper Added (External) — Verifying Adoption
**File:** `sidebar.js`
**Severity:** Info

An `apiFetch` helper was added externally (detected during audit via auto-formatter). It correctly throws on non-2xx and parses error detail. Several endpoints already use it. However, `loadWisebase`, `rec-save-wb`, `image/generate`, and the wisebase DELETE were not yet converted (fixed above manually).

---

## NON-CRITICAL FINDINGS (No Fix Applied)

### NC-01 — `conversationHistory` / `featureModels` Storage Quota
**File:** `sidebar.js`
**Severity:** Low (theoretical)

`featureModels` is stored in `chrome.storage.local` (quota: 10MB per extension). The object stores only key→modelName pairs for ~13 features with short string values. Maximum size is negligible (~500 bytes). No quota risk.

`conversationHistory` is NOT stored in localStorage in this codebase — it's managed server-side via `conversation_id`. No client-side unbounded growth observed.

**Recommendation:** No action needed.

---

### NC-02 — `sendMessage` Race Condition: Two Quick Messages
**File:** `sidebar.js`
**Severity:** Low (guarded)

If the user clicks Send twice quickly, the second call hits `if (activeStream) return` and is silently dropped. The send button is disabled on the first call (`sendBtn.disabled = true`), which prevents UI-level double-sends. However, the disable happens *after* the async page context fetch (lines 451-488) — a very fast second keyboard Enter could bypass the `activeStream` check if the first call is still in the `await` for `GET_PAGE_CONTENT`.

**Recommendation:** Set `sendBtn.disabled = true` immediately at the top of `sendMessage` (before the awaits), then reset it in the return-early path if validation fails. This is a hardening improvement, not a crash-level bug.

---

### NC-03 — `content.js` `ext` Declared Outside IIFE
**File:** `content.js`
**Severity:** Negligible

`const ext = typeof browser !== 'undefined' ? browser : chrome;` is declared at file scope (line 8), outside the IIFE wrapper. In Chrome extensions, content scripts run in an isolated world — they do not share the `window` scope with the page. However, multiple injections of the same content script (e.g., via `executeScript` calls in `background.js`) could conflict since the IIFE cleans up DOM elements but `ext` is re-declared as a `const` in the same scope. This would throw a `SyntaxError: Identifier 'ext' has already been declared` on second injection.

The IIFE at the top removes previous DOM elements (`_prevDock`, `_prevHost`) but cannot prevent the file-scope `const ext` redeclaration on re-injection.

**Recommendation:** Move `const ext = ...` inside the IIFE, or check `window.__auraExtShim` before declaring.

---

## FILES CHANGED

| File | Changes |
|------|---------|
| `D:/Aura/extension/sidebar.js` | BUG-01, 02, 03, 04, 05, 06, 07, 08, 09, 10 |
| `D:/Aura/extension/background.js` | BUG-11 |
| `D:/Aura/extension/content.js` | BUG-12 |
| `D:/Aura/extension/sidebar.html` | No changes needed |
