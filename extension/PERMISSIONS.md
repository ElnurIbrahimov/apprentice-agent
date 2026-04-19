# AURA Extension — Permission Justifications

This document explains why each declared permission is needed. If/when this
extension is ever submitted to the Chrome Web Store, these justifications are
the reviewer-facing rationale.

## `permissions`

| Permission | Used by | Why |
|------------|---------|-----|
| `sidePanel` | `background.ts` | Open the Aura UI in the Chrome side panel. |
| `contextMenus` | `background.ts` | Right-click actions (summarize selection, ask about link). |
| `storage` | `background.ts`, `offscreen.ts`, sidebar store | Save backend URL, API key, conversation history, prefs. |
| `activeTab` | content script | Access current tab for page capture / summarization on user action. |
| `tabs` | `background.ts` | Query the active tab to route hotkeys and omnibox queries. |
| `scripting` | build pipeline | Content-script injection. |
| `unlimitedStorage` | conversation history | Conversations + lifelog can exceed 5 MB quickly. |
| `topSites` | `newtab.ts:274` | New Tab override shows user's frequently visited sites. |
| `offscreen` | `background.ts`, `offscreen.ts` | Persistent WebSocket in offscreen document (survives SW suspension). |
| `notifications` | `background.ts` | Surface proactive / approval events when sidebar is closed. |
| **`debugger`** | `agentLoop.ts`, `cdp.ts` | **Power Mode / Hands agent only.** Used to drive Chrome DevTools Protocol for clicks, typing, screenshots, and DOM inspection during a user-initiated autonomous task. Never attaches without explicit user action. |

## `optional_permissions`

| Permission | Why |
|------------|-----|
| `tabCapture` | Voice + screen recording features; requested at feature use. |

## `host_permissions`

- `<all_urls>` — content script runs on every page at `document_idle` for
  ghost-bar, stuck-detector, page-capture, and ambient-surface features.
  Respects user opt-out via sidebar settings.

## Content Security Policy

- `wasm-unsafe-eval` — required by `workers/ai-worker.ts`, which loads
  Transformers.js (`Xenova/all-MiniLM-L6-v2`) for on-device text embeddings.
  No JS eval; WASM-only.
