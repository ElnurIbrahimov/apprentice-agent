# Agent 1 Audit: Chat Panel & WebSocket Infrastructure

## Files Audited
- `D:/Aura/extension/sidebar.js`
- `D:/Aura/extension/sidebar.html`
- `D:/Aura/extension/background.js`
- `D:/Aura/api/routes/chat.py`
- `D:/Aura/api/auth.py`

---

## Bugs Found and Fixed

### Bug 1 — Race condition in `sendMessage`: `activeStream` guard after async awaits

**File:** `sidebar.js`

**Problem:** `sendMessage` is `async` and awaits two network calls (GET_PAGE_CONTENT and GET_CURRENT_TAB) before setting `activeStream` and disabling `sendBtn`. A second call to `sendMessage` (e.g. double-click, chip click while tab fetch is in flight) would pass the `if (activeStream) return` guard and both calls would proceed to `ws.send()` simultaneously, creating two AI bubbles and corrupting the stream.

**Fix:** Moved `sendBtn.disabled = true` and `activeStream = true` (sentinel) to immediately after the initial guards, BEFORE any `await`. Added a post-await re-check of `wsReady` and `ws.readyState` to bail cleanly if WS dropped during the fetch.

---

### Bug 2 — `clearAll` / "New conversation" doesn't tell the backend

**File:** `sidebar.js`

**Problem:** `clearAll()` only cleared the UI (`msgs` DOM) and reset `conversationId = null`, but never notified the backend. The server's agent maintained the old conversation history for the WebSocket connection, so "New conversation" / clear button was purely cosmetic — the next message would still get context from the previous conversation.

**Fix:** Added `fetch('/api/chat/clear', { method: 'POST' })` at the end of `clearAll()` when WS is connected. This calls the existing `/api/chat/clear` endpoint which clears the agent's conversation history.

---

### Bug 3 — `clearAll` didn't abort an in-progress stream

**File:** `sidebar.js`

**Problem:** If `clearAll()` was called while a message was streaming (via the trash button or new conversation button), `activeStream` was left set. The `onmessage` handler would continue writing chunks into a DOM node that had been removed from the document. Worse, `finalizeStream()` would eventually run and try to re-enable `sendBtn` — but the entire chat UI had been wiped, which could leave `sendBtn` stuck disabled if `activeStream.submitBtn` was somehow different.

**Fix:** Added stream abort at the top of `clearAll()` — if `activeStream` is a stream object (not the sentinel), it re-enables `submitBtn` and nulls `activeStream`. If it's the pre-await sentinel (`true`), it nulls it and re-enables `sendBtn`.

---

### Bug 4 — `finalizeStream` and `onmessage` didn't guard against the sentinel value

**File:** `sidebar.js`

**Problem:** After introducing the sentinel (`activeStream = true` as a lock), the existing `if (!activeStream) return` guards in `onmessage` and `finalizeStream` would NOT catch the sentinel (since `true` is truthy). The `onmessage` handler would proceed to `const { type, el } = activeStream` which destructures `true`, giving `type = undefined` and `el = undefined`. This would cause silent failures or errors when processing any WS messages that arrived during the page-context await window.

**Fix:** Changed `if (!activeStream) return` to `if (!activeStream || activeStream === true) return` in both `onmessage` and `finalizeStream`.

---

### Bug 5 — Arrow `<span>` inside model pill had `className = 'mdl-pill'`

**File:** `sidebar.js`

**Problem:** In `buildPill()`, the chevron arrow span inside the pill button was assigned `arrow.className = 'mdl-pill'`. The `mdl-pill` class is the full button style (sets `display:inline-flex`, `background:rgba(124,58,237,.1)`, `border:1px solid`, `border-radius:20px`, `padding`, etc.). This caused the arrow span — which lives inside the button — to render as a visually nested pill with its own border, background and padding, making the model picker look broken.

**Fix:** Replaced `arrow.className = 'mdl-pill'` with an inline style `display:inline-flex;align-items:center;margin-left:1px;opacity:.55` that achieves proper alignment without inheriting button styles.

---

## Items Verified as Correct (No Fix Needed)

1. **WebSocket URL & connection**: `ws://localhost:8000/api/chat/stream` matches the backend route `@router.websocket("/stream")` under prefix `/api/chat`. Correct.

2. **WebSocket auth**: `verify_api_key_ws` checks `AURA_API_AUTH_ENABLED` which defaults to `false`. No API key required for local dev. WS connection with no headers works correctly.

3. **Reconnect on disconnect**: `ws.onclose` calls `setTimeout(connectWS, 5000)`. The `ws.onerror` fires before `ws.onclose`; the browser always fires both, so reconnect is handled. Note added in the `onerror` handler comment for clarity.

4. **`connectWS` double-connect guard**: `if (ws && ws.readyState <= 1) return` correctly prevents creating a new WebSocket when one is already connecting (0) or open (1).

5. **Streaming accumulation**: `activeStream.rawText += d.content || ''` correctly accumulates chunks. `md(activeStream.rawText)` is called on each chunk for the chat panel.

6. **`finalizeStream` re-enables button**: Captures `stream` before nulling `activeStream`, then calls `stream.submitBtn.disabled = false`. Correct.

7. **`m-think`, `m-research`, `m-page` buttons**: All three exist in HTML with correct IDs. All three event listeners are wired in JS at lines 407-419. They toggle class `on` correctly.

8. **`mdl-chat` slot**: `<span id="mdl-chat"></span>` exists in HTML inside `#modes`. `initModelPills()` appends the pill into it. Correct.

9. **`rail-clear` button**: Exists in HTML as `<button class="rbtn" id="rail-clear">`. `$('rail-clear').addEventListener('click', clearAll)` is wired at line 259.

10. **`btn-new` button**: Exists in HTML header. `$('btn-new').addEventListener('click', clearAll)` wired at line 509.

11. **Empty state show/hide**: `hideEmpty()` sets `display:none`. `clearAll()` appends empty back and sets `display:''` which reverts to CSS default `display:flex`. Correct.

12. **`PAGE_KEYWORDS` regex**: `/\b(this page|this site|...)\b/i` — uses word boundaries and case-insensitive flag. Correctly matches natural language page-context requests.

13. **`conversationId` persistence within a session**: Multi-turn chat works because the backend maintains conversation context per WebSocket connection. `conversationId` (null) is sent each message; the server auto-uses its current conversation. Works correctly for the common case.

14. **Model passed to `ws.send()`**: `model: getModel(modelKey || 'chat')` — returns the feature-specific model from `featureModels` storage, or `null` if not set. The backend accepts `null` and uses its default. Correct.

15. **`thinkingMode` + `deepResearch` prepend correctly**: Both prepend instruction strings to `full` (the context-enriched message). Correct.

16. **`loadPage()` called from `m-page` button**: Fire-and-forget call is appropriate — it's async but the promise result isn't needed synchronously. Shows `sysmsg` feedback inside itself.

---

### Bug 6 — `ws.onclose` didn't handle the pre-await sentinel value

**File:** `sidebar.js`

**Problem:** The existing `ws.onclose` handler correctly cleaned up an active stream on disconnect. However after introducing the `activeStream = true` sentinel (Bug 1 fix), if WS dropped during the pre-await window, `activeStream === true`, and the handler would do `const s = true`, then `s.submitBtn` and `s.el` would both be `undefined`. The `sendBtn` would stay disabled forever.

**Fix:** Added a sentinel check at the top of the `onclose` cleanup block — if `activeStream === true`, null it and re-enable `sendBtn` directly. Otherwise proceed with the normal stream object cleanup.

---

## Summary

6 bugs fixed:
1. Race condition — `activeStream` lock moved before async awaits
2. Clear/New conversation didn't reset backend history
3. Active stream not aborted when chat cleared
4. Sentinel value not guarded in `onmessage` and `finalizeStream`
5. Arrow span in model pill had wrong CSS class causing visual corruption
6. `ws.onclose` didn't handle the pre-await sentinel — `sendBtn` would stay disabled if WS dropped during page context fetch
