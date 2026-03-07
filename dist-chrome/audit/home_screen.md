# AURA Chat Home Screen — Implementation Report

**Date:** 2026-03-07
**Files modified:** `extension/sidebar.html`, `extension/sidebar.js`

---

## What was built

A Sider-style welcome home screen that replaces the blank chat panel on first open.

### Structure

`#chat-home` is inserted as a flex sibling **above** `#msgs` inside `#panel-chat`, between the context bar and the messages div. It contains:

- `#home-avatar` — purple gradient circle with "A"
- `#home-title` — "Hi, I'm AURA"
- `#home-sub` — "How can I assist you today?"
- `#home-chips` — 2x2 grid of action buttons: Summarize page, Deep Research, Explain selection, Write something

Each chip has an inline 14x14 stroked SVG icon.

---

## Show/hide mechanism

Visibility is controlled via a CSS class `.at-home` on `#panel-chat` (not inline `display:` toggling on `#chat-home` itself). This avoids fighting specificity wars with the CSS rule that collapses `#msgs`.

- `#panel-chat.at-home #msgs` → `flex:0; overflow:hidden; min-height:0` (collapsed)
- `#panel-chat:not(.at-home) #chat-home` → `display:none !important` (hidden)

`#panel-chat` starts with class `at-home` in HTML so the home screen is visible on first open.

### `updateHomeState()` (new JS function)

Checks `msgs.querySelector('.mrow')`. If a message row exists, removes `.at-home`; otherwise adds it back.

Called from:
- `addUserMsg()` — after appending the first user message
- `clearAll()` — after wiping messages (restores home)
- `switchPanel('chat')` — when returning to chat tab

---

## Chip actions

| Chip | Action |
|------|--------|
| Summarize page | `sendMessage('Summarize this page for me')` |
| Deep Research | `switchPanel('research')` |
| Explain selection | `switchPanel('chat')` + focus input + set placeholder |
| Write something | `switchPanel('write')` |

---

## Old `#empty` state

Kept intact but initialised with `style="display:none"` in HTML (was previously shown by default). `clearAll()` no longer restores it to visible — home screen handles the empty state visually. The old `.chip` elements still work via their existing `document.querySelectorAll('.chip')` handler.

---

## Animation

`@keyframes homeIn` — 350ms cubic-bezier ease, slides up 16px from opacity 0.

---

## CSS variables used

All from existing AURA theme: `--p`, `--ub2`, `--tx`, `--mu`, `--b1`, `--b2`. No new variables introduced.
