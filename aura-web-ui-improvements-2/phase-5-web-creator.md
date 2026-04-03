# Phase 5: Web Creator

**Effort:** 3-4 days
**Impact:** Full website building capability in the web UI
**Depends on:** Phase 4 (Artifacts panel provides the iframe preview foundation)

---

## The Vision

A dedicated tab or mode where users describe a website → Aura builds it with live preview → users iterate via chat → export or deploy. Same as the extension's WebCreatorPanel but adapted for the web UI's wider layout.

---

## 5A. Web Creator Tab

### Layout
New tab (7th or 8th) or sub-mode within the Artifacts panel:

```
┌────────────────────────────────────────────────────┐
│ [Templates] [Device] [Theme] [Export] [Share] [Deploy] │
├──────────────────────┬─────────────────────────────┤
│ Chat / Instructions  │ Live Preview (iframe)        │
│                      │                              │
│ [Message list]       │ ┌──────────────────────┐    │
│                      │ │                      │    │
│                      │ │   Rendered Website   │    │
│                      │ │                      │    │
│                      │ └──────────────────────┘    │
│                      │                              │
│ [Input bar]          │ [File Tree] (project mode)  │
├──────────────────────┴─────────────────────────────┤
│ [Code Editor — collapsible]                         │
└────────────────────────────────────────────────────┘
```

### Two Modes
1. **Single Page** — generates one HTML file (quick, simple)
2. **Project** — multi-file with VirtualFS, file tree (same as extension Phase 3)

---

## 5B. Features to Port from Extension

### Core (must-have)
- Chat-to-HTML generation with streaming preview
- 7 single-page templates (Landing, Portfolio, Blog, Dashboard, Login, Pricing, 404)
- 5 multi-file project templates (Landing, Portfolio, Dashboard, React App, API+Frontend, Vite+React)
- Element selection mode (click element → send as context)
- Version history with undo/redo (20 versions)
- Device preview: Desktop / Tablet / Mobile
- Theme editor (color picker for primary/secondary/accent/background/text)
- Export: Download HTML, Download ZIP, Copy code, Open in CodeSandbox, Open in StackBlitz

### Phase 4-6 Features (already built for extension)
- CodeMirror 6 code editor (Phase 2 — port to web)
- Diff view when AI regenerates (accept/reject)
- Multi-file VirtualFS + FileTree + parseFileOperations (Phase 3)
- Agent Build Mode with plan → approve → execute (Phase 4)
- WebContainer for npm projects (Phase 5)
- Share via backend + GitHub Pages deploy (Phase 6)
- Visual AI feedback with quality score (Phase 7)
- Component Library + Design Tokens (Phase 8)

### Adaptation for Web UI
The web UI has more screen space than the extension sidebar. Take advantage:
- Side-by-side chat + preview (not stacked)
- Wider code editor panel (collapsible)
- File tree can be always visible (not a toggle)
- Better template picker (card grid with previews, not just icon buttons)

---

## 5C. Template Picker

Full-screen or modal template selection on first visit:

```
┌────────────────────────────────────────────────┐
│ Start a new project                             │
│                                                 │
│ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ │
│ │      │ │      │ │      │ │      │ │      │ │
│ │ Land │ │ Port │ │ Blog │ │ Dash │ │ React│ │
│ │ Page │ │folio │ │      │ │board │ │ App  │ │
│ └──────┘ └──────┘ └──────┘ └──────┘ └──────┘ │
│                                                 │
│ Or describe what you want to build:             │
│ ┌──────────────────────────────────────────┐   │
│ │ Build me a...                            │   │
│ └──────────────────────────────────────────┘   │
│                                   [Start] │
└────────────────────────────────────────────────┘
```

Each template card shows:
- Preview thumbnail (static screenshot)
- Template name
- File count (for multi-file templates)
- Framework badge (HTML, React, etc.)

---

## 5D. Shared Utilities

These utilities already exist in `extension-src/src/utils/` and can be shared:
- `virtualFS.ts` — copy to `web/src/utils/`
- `parseFileOperations.ts` — copy to `web/src/utils/`
- `buildPlan.ts` — copy to `web/src/utils/`
- `StreamingPreviewController` — copy or extract to shared package
- `deployUtils.ts` — copy to `web/src/utils/`
- `designTokens.ts` — copy to `web/src/utils/`
- `componentLibrary.ts` — adapt (use localStorage instead of chrome.storage.local)

### Storage Adapter
Extension uses `chrome.storage.local`. Web uses `localStorage`. Create an adapter:
```typescript
// web/src/utils/storage.ts
export const storage = {
  get: (key: string) => JSON.parse(localStorage.getItem(key) || 'null'),
  set: (key: string, value: any) => localStorage.setItem(key, JSON.stringify(value)),
  remove: (key: string) => localStorage.removeItem(key),
};
```

---

## 5E. Backend Integration

The web UI connects to the same backend. All endpoints already exist:
- `POST /api/generate/raw` — raw LLM generation for HTML
- `POST /api/agent/build` — agent-powered multi-file build
- `POST /api/agent/build/plan` — build planning
- `POST /api/share` — shareable links
- `WS /api/artifacts/stream` — live artifact updates from agent builds

No backend changes needed.

---

## Definition of Done — Phase 5
- [ ] Web Creator tab with side-by-side chat + preview
- [ ] Single-page and project modes
- [ ] All 12 templates available with card picker
- [ ] Chat-to-HTML streaming generation with live preview
- [ ] Element selection for targeted edits
- [ ] Version history (20 versions, undo/redo)
- [ ] Device preview (desktop/tablet/mobile)
- [ ] Theme editor with color pickers
- [ ] Code editor (CodeMirror 6 or Shiki-based viewer)
- [ ] Multi-file: VirtualFS, FileTree, file operations
- [ ] Export: HTML, ZIP, CodeSandbox, StackBlitz
- [ ] Share + Deploy buttons (reuse Phase 6 backend)
- [ ] Agent Build Mode with plan/approve/execute
- [ ] Mobile-responsive layout (stacked on narrow screens)
