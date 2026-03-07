# Agent 5 Audit Report — Infrastructure, Dock, Browser Agent
**Date:** 2026-03-07
**Files audited:** content.js, background.js, sidebar.js, sidebar.html, manifest.json

---

## Summary

3 bugs found and fixed. All other infrastructure checks passed.

---

## BUGS FIXED

### Bug 1 — content.js: `__auraToolbarMounted` guard never checked (FIXED)

**File:** `D:/Aura/extension/content.js`

**Problem:** `window.__auraToolbarMounted` was set to `true` but never read before executing. If the content script ran twice (e.g., Chrome injecting it on extension reload + a cached run), both instances would execute, creating duplicate dock and toolbar elements.

**Fix:** Added `if (window.__auraToolbarMounted) return;` at the top of the IIFE, before the existing element-removal logic.

```js
// Before (broken):
(function () {
  const _prevDock = document.getElementById('aura-dock-host');
  ...
  window.__auraToolbarMounted = true;

// After (fixed):
(function () {
  if (window.__auraToolbarMounted) return;
  window.__auraToolbarMounted = true;
  const _prevDock = document.getElementById('aura-dock-host');
  ...
```

---

### Bug 2 — content.js: `showToast` crashes / mispositioning when toolbar is hidden (FIXED)

**File:** `D:/Aura/extension/content.js`

**Problem:** `showToast()` always read `toolbar.style.top` to position the toast. When the dock's "Save" button triggers a toast, the selection toolbar is not visible and its `top` is unset (`''`), causing `parseInt('')` to return `NaN`, positioning the toast at `NaN + 40 = NaN` (invisible). Also `toolbar.style.left` would be empty.

**Fix:** Added a condition to fall back to top-center positioning when the toolbar is not visible.

```js
function showToast(message, durationMs = 2000) {
  toast.textContent = message;
  toast.classList.add('visible');
  if (toolbar.classList.contains('visible') && toolbar.style.top) {
    toast.style.top = (parseInt(toolbar.style.top) + 40) + 'px';
    toast.style.left = toolbar.style.left;
  } else {
    toast.style.top = '20px';
    toast.style.left = Math.round(window.innerWidth / 2 - 100) + 'px';
  }
  setTimeout(() => toast.classList.remove('visible'), durationMs);
}
```

---

### Bug 3 — sidebar.js + sidebar.html: Browser Agent panel missing model pill slot (FIXED)

**Files:** `D:/Aura/extension/sidebar.js`, `D:/Aura/extension/sidebar.html`

**Problem:** `FEATURE_DEFS` in sidebar.js includes the `agent` key (Browser Agent), but `PILL_SLOTS` did not contain `agent: 'mdl-agent'`. Additionally, the Browser Agent panel HTML had no `<span id="mdl-agent">` slot. This meant `initModelPills()` silently skipped the agent feature — users could not select a model for the Browser Agent from the panel header.

**Fix (sidebar.js):** Added `agent: 'mdl-agent'` to `PILL_SLOTS`:
```js
const PILL_SLOTS = {
  ...
  youtube: 'mdl-youtube', research: 'mdl-research', math: 'mdl-math',
  agent: 'mdl-agent',   // <-- added
};
```

**Fix (sidebar.html):** Changed the agent panel header `div` from plain `<h2>` to the standard pattern with a model pill slot:
```html
<!-- Before -->
<div class="tool-hdr"><h2>Browser Agent</h2></div>

<!-- After -->
<div class="tool-hdr" style="display:flex;align-items:center;justify-content:space-between;padding-bottom:12px">
  <h2 style="margin:0">Browser Agent</h2>
  <span id="mdl-agent"></span>
</div>
```

---

## CHECKS PASSED

### FLOATING DOCK (content.js)

| Check | Result |
|-------|--------|
| `dockHost` appended to `document.body` | PASS — line 163 |
| All 5 dock buttons present (Chat, Search, This Page, Translate, Save) | PASS — `dock-chat`, `dock-search`, `dock-thispage`, `dock-translate`, `dock-save` |
| `mouseenter`/`mouseleave` expand/collapse dock | PASS — JS event listeners on `dockHost` set `maxHeight` and `opacity` |
| "This Page" button reads page text and opens Ask panel | PASS — reads `document.body.innerText` and sends `OPEN_WITH_TEXT` |
| "Save" button saves selection to `/api/knowledge/save` | PASS — sends `SAVE_KNOWLEDGE` to background which POSTs to backend |
| Dock appears on every page | PASS — `content_scripts` matches `<all_urls>`, run_at `document_idle` |
| `__auraToolbarMounted` guard | FIXED (Bug 1) |

### ASK AURA PANEL (sidebar.js)

| Check | Result |
|-------|--------|
| `PREFILL_TEXT` populates `ask-sel-txt` | PASS — line 800 |
| `PREFILL_TEXT` populates `ask-src` | PASS — line 801 |
| 5 action buttons fire `sendMessage` with correct prefix | PASS — `data-prompt` + `pendingCtx.text`, switches to chat, calls `sendMessage(msg, 'ask')` |
| `ask-send` button works | PASS — line 815 |
| Enter key on `ask-inp` works | PASS — line 824-826 |
| Context menu "Ask AURA" triggers panel | PASS — background stores pending, sends `PREFILL_TEXT` on `SIDEBAR_READY` |

### BROWSER AGENT PANEL (sidebar.js)

| Check | Result |
|-------|--------|
| `agent-start` calls `runAgentLoop()` | PASS — line 1522-1531 |
| `agent-stop` sets `agentRunning = false` | PASS — line 1534-1538 |
| `runAgentLoop` gets DOM via `AGENT_DOM` → background → content | PASS — `msgBg({ type: 'AGENT_DOM' })` → background `AGENT_DOM` case → `GET_DOM` to content |
| POSTs to `/api/agent/action` | PASS — line 1486-1490 |
| Executes actions via `AGENT_EXEC` | PASS — `msgBg({ type: 'AGENT_EXEC', action })` |
| `AGENT_NAV` updates tab URL | PASS — background `AGENT_NAV` case calls `ext.tabs.update()` |
| 15-step limit enforced | PASS — `agentStep >= 15` check at top of `runAgentLoop` |
| `agent-log` shows step-by-step progress | PASS — `logAgent()` appends `.agent-step` divs |

### RAIL NAVIGATION (sidebar.html)

| Check | Result |
|-------|--------|
| All 20 rail buttons have `data-panel` | PASS — chat, ask, search, wisebase, translate, grammar, write, rec, ocr, pdf, summary, image, agent, youtube, compare, research, math, artifacts, models, tools |
| All `data-panel` values match actual panel IDs | PASS — each `data-panel="X"` has `id="panel-X"` in HTML |
| `switchPanel()` shows correct panel and marks `.on` | PASS — removes `active` from all panels, adds to `panel-${name}`, toggles `.on` on matching `.rbtn[data-panel]` |
| Rail scrollable when too many buttons | PASS — `#rail` has `overflow-y:auto; scrollbar-width:none` |
| Separators (`rsep`) display correctly | PASS — `.rsep` styled as `width:20px; height:1px` lines between groups |

### CONTEXT MENU (background.js + manifest.json)

| Check | Result |
|-------|--------|
| `contextMenus` permission in manifest | PASS — `"permissions": ["sidePanel","contextMenus","storage","activeTab","tabs","scripting"]` |
| Background creates "Ask AURA" context menu | PASS — `ext.contextMenus.create({ id: 'ask-aura', ... })` in `onInstalled` |
| Clicking opens sidebar and prefills Ask panel | PASS — stores pending query, opens side panel, delivered on `SIDEBAR_READY` |

### MODEL PILLS (sidebar.js)

| Check | Result |
|-------|--------|
| `initModelPills()` injects pills into all PILL_SLOTS | PASS — iterates all entries, calls `buildPill(key)`, appends to `$(slotId)` |
| All slot IDs exist in HTML | PASS after fix — `mdl-chat`, `mdl-search`, `mdl-translate`, `mdl-write`, `mdl-grammar`, `mdl-ask`, `mdl-pdf`, `mdl-voice`, `mdl-summary`, `mdl-youtube`, `mdl-research`, `mdl-math`, `mdl-agent` (added) |
| `featureModels` persists to `chrome.storage.local` | PASS — `ext.storage.local.set({ featureModels })` on every `pick()` call |

### GENERAL INFRASTRUCTURE

| Check | Result |
|-------|--------|
| `const HTTP = 'http://localhost:8000'` | PASS — `sidebar.js` line 6, `background.js` line 9 (as `BACKEND`) |
| `const WS = 'ws://localhost:8000/api/chat/stream'` | PASS — `sidebar.js` line 7 |
| `ext = typeof browser !== 'undefined' ? browser : chrome` at top of each JS file | PASS — line 8 in `content.js`, line 7 in `background.js`, line 10 in `sidebar.js` |
| `esc()` function exists | PASS — `sidebar.js` lines 91-94 |
| `md()` function exists | PASS — `sidebar.js` lines 106-141 |

---

## NO-CHANGE DECISIONS

- **`aura-host` on `documentElement` vs `body`**: The shadow DOM selection toolbar host uses `documentElement`. The `dockHost` uses `document.body` as required. The shadow host on `documentElement` is intentional — it ensures the toolbar renders even before `body` is fully available. Not a bug.
- **`dock-thispage` uses inline `innerText` instead of `GET_PAGE_CONTENT` message**: Functionally equivalent and more efficient (already in content script context). Sends `OPEN_WITH_TEXT` which achieves the same result.
- **`OPEN_WITH_TEXT` when sidebar already open**: If the sidebar is already visible when a dock button is clicked, the pending query is stored but not immediately delivered (requires re-SIDEBAR_READY). This is a pre-existing architectural limitation shared across all dock actions. No fix applied.
