# AURA Extension Audit — background.js / content.js / sidebar.html / manifests

**Date:** 2026-03-07
**Auditor:** Claude Sonnet 4.6 (automated deep audit)
**Status:** All issues fixed.

---

## background.js

### BUG FIXED — Missing `moz-extension://` in protected-URL block
**File:** `background.js`, `GET_PAGE_CONTENT` handler
**Problem:** The guard that blocks extension injection into protected pages checked for `chrome://`, `chrome-extension://`, `about:`, and `edge://` — but not `moz-extension://`. On Firefox, attempting to inject into extension pages via `moz-extension://` URLs would throw and leave `sendResponse` uncalled.
**Fix:** Added `moz-extension://` to the protected-URL check.

### BUG FIXED — No timeout on `scripting.executeScript` (sendResponse could hang forever)
**File:** `background.js`, `GET_PAGE_CONTENT` handler, scripting fallback path
**Problem:** When `content.js` wasn't running on a tab, the code fell back to `scripting.executeScript`. If the script execution stalled (e.g., frozen tab, browser permission lag), `sendResponse` was never called. The message channel held open by `return true` would leak indefinitely.
**Fix:** Added a 10-second `setTimeout` fallback that calls `sendResponse({ ok: false, error: 'Extraction timed out' })` if neither `.then()` nor `.catch()` fires in time. Both paths use a `responded` flag to prevent double-firing.

### VERIFIED OK — SIDEBAR_READY handler
Delivers `pendingQuery` (with action/url/title) OR `pendingPanelSwitch` correctly. PDF/YT detection messages are broadcast directly (not stored/replayed), which is the correct pattern.

### VERIFIED OK — All `return true` placements
Every case that calls `sendResponse` asynchronously has `return true`. `OCR_START` does not use `sendResponse` — it uses separate `runtime.sendMessage` calls — so its `return false` is correct.

### VERIFIED OK — Agent relay handlers
`AGENT_DOM`, `AGENT_EXEC`, `AGENT_NAV` all have `return true` at line 271.

---

## content.js

### BUG FIXED — `execAction()` missing `selectOption` action
**File:** `content.js`, `execAction()` function
**Problem:** The browser agent could send `{ action: 'selectOption', selector: '...', value: '...' }` to interact with `<select>` dropdowns, but `execAction()` had no handler for it — it would fall through to `'Unknown action'` error.
**Fix:** Added a `selectOption` case that finds the matching `<option>` by value or text, sets `el.value`, and dispatches a `change` event.

### VERIFIED OK — `serializeDOM()` graceful on empty pages
Returns `[]` when no interactive elements exist — no crash.

### VERIFIED OK — OCR overlay cleanup
Both the Esc path and mouseup path properly call `document.removeEventListener('keydown', onEsc)` and remove the overlay from `document.body`. No event listener leak.

### VERIFIED OK — Shadow DOM toolbar isolation
The toolbar is inside a Shadow DOM (`attachShadow({ mode: 'open' })`). Page CSS cannot pierce Shadow DOM. The dock is injected as a direct `document.body` child with all styles applied via JS `Object.assign(el.style, ...)` — not classes — so page CSS has no selectors to target it. Isolation is adequate.

### NOTE — Dock does not auto-hide on scroll
The dock is always visible at `right:0; top:50%` and does not hide when the user scrolls. This is a design decision, not a bug. If auto-hide-on-scroll is wanted, a `scroll` event listener on `window` would need to be added.

---

## sidebar.html — HTML

### BUG FIXED — `#ocr-actions` CSS/HTML mismatch
**Problem:** The CSS defined `#ocr-actions { display: flex }` (always visible), but the HTML had `style="display:none"`. There was no `#ocr-actions.on { display: flex }` rule. If sidebar.js tried to toggle it via `.classList.add('on')`, it would fail silently.
**Fix:**
- Changed CSS to `#ocr-actions { display: none }` + added `#ocr-actions.on { display: flex }`.
- Removed `style="display:none"` from HTML (now CSS handles default state).

### BUG FIXED — Duplicate `display` property in `#sum-badges` CSS rule
**Problem:** `#sum-badges` had `display:flex; ... display:none` in the same rule. The second declaration overrides the first, so `display:flex` was dead code. While functionally harmless (`.on { display:flex }` restores it), it was misleading.
**Fix:** Reordered to `display:none; ... /* no duplicate */`.

### BUG FIXED — Duplicate `display` property in `#sum-actions` CSS rule
**Same problem as #sum-badges.**
**Fix:** Same fix.

### FIXED — Missing `aria-label` on icon-only buttons
The following buttons had no accessible name (screen readers would announce them with no label):
- `#send` → added `aria-label="Send message"`
- `#search-btn` → added `aria-label="Search"`
- `#yt-summarize-btn` → added `aria-label="Summarize video"`
- `#mdl-reload` → added `aria-label="Reload models"`
- `#rail-clear` → added `aria-label="Clear chat"`
- `#btn-new` already had `title="New conversation"` → added matching `aria-label`

### FIXED — `#yt-results.on` CSS rule missing
**Problem:** `#yt-results` uses `display:none` inline in HTML and is shown by JS. There was no `.on { display: flex }` CSS rule to go with the pattern used by all other panels.
**Fix:** Added `#yt-results.on { display: flex; flex-direction: column }`.

### FIXED — Narrow (320px) media query missing coverage for search/translate/artifacts panels
The existing `@media (max-width:320px)` block only explicitly targeted chat, YouTube, research, and math content. Added padding/margin overrides for `#search-hdr`, `#search-results`, `#tr-langs`, `#sum-go`, `#sum-result`, `#sum-badges`, `#sum-actions`, `#art-inp`, and `#art-code` to prevent overflow.

### SECURITY FIX — Artifacts iframe sandbox
**File:** `sidebar.html`, `#art-preview` iframe
**Problem:** `sandbox="allow-scripts"` was present (good — blocks same-origin access, preventing AI-generated code from touching extension APIs or storage). However, generated HTML demos often include forms and modal dialogs (alert/confirm). Without `allow-forms` and `allow-modals`, form submissions are blocked silently and JS modals throw errors, breaking common artifact demos.
**Fix:** Updated to `sandbox="allow-scripts allow-forms allow-modals"`.
- `allow-same-origin` intentionally NOT added — this remains the critical security boundary preventing iframe content from accessing extension APIs, cookies, or storage.
- `allow-popups` intentionally NOT added — prevents new windows/tabs from AI-generated code.

### VERIFIED — All 20 panels present with correct IDs
All 20 `<div class="panel" id="panel-X">` elements exist and every rail `<button data-panel="X">` maps to a real panel div. No orphaned buttons or missing panels.

### VERIFIED — All panels have overflow control
All panels have `overflow:hidden` at the `.panel` level (via `.panel { overflow:hidden }`) and individual scrollable children have `overflow-y:auto`. No content should escape panel bounds.

### NOTE — No panel-level loading indicators on several panels
Panels like `#panel-grammar`, `#panel-translate`, `#panel-write`, `#panel-ocr` do not have dedicated loading spinners — they rely on button `:disabled` state and status text elements. This is acceptable but noted.

---

## manifest.json (Chrome)

### VERIFIED OK
All required permissions present: `sidePanel`, `contextMenus`, `storage`, `activeTab`, `tabs`, `scripting`.
`host_permissions` includes `<all_urls>` — required for content extraction on arbitrary pages.
No unused permissions identified.

---

## manifest.firefox.json

### BUG FIXED — Missing `scripting` permission
**Problem:** `scripting` was in Chrome manifest but not Firefox manifest. The `executeScript` fallback in `background.js` (used when a tab was opened before the extension loaded) would throw a permission error on Firefox.
**Fix:** Added `"scripting"` to `permissions` array.

### BUG FIXED — Missing `<all_urls>` in `host_permissions`
**Problem:** Firefox manifest only allowed `http://localhost/*` and `http://127.0.0.1/*`. Content scripts are declared for `<all_urls>` in `content_scripts.matches`, but without the matching `host_permissions`, Firefox would silently refuse to inject content scripts on most pages. Page content extraction, selection toolbar, and the floating dock would all fail.
**Fix:** Added `"<all_urls>"` to `host_permissions`.

---

## Summary of All Changes

| File | Change |
|------|--------|
| `background.js` | Added `moz-extension://` to protected URL list |
| `background.js` | Added 10s timeout + `responded` flag to `executeScript` fallback |
| `content.js` | Added `selectOption` action to `execAction()` |
| `manifest.firefox.json` | Added `scripting` permission |
| `manifest.firefox.json` | Added `<all_urls>` host permission |
| `sidebar.html` CSS | Fixed duplicate `display` in `#sum-badges` rule |
| `sidebar.html` CSS | Fixed duplicate `display` in `#sum-actions` rule |
| `sidebar.html` CSS | Changed `#ocr-actions` to `display:none` default + added `.on{display:flex}` |
| `sidebar.html` CSS | Added `#yt-results.on { display:flex; flex-direction:column }` |
| `sidebar.html` CSS | Extended `@media (max-width:320px)` to cover more panels |
| `sidebar.html` HTML | Added `aria-label` to 6 icon-only buttons |
| `sidebar.html` HTML | Removed redundant `style="display:none"` from `#ocr-actions` |
| `sidebar.html` HTML | Updated `#art-preview` sandbox to `allow-scripts allow-forms allow-modals` |
