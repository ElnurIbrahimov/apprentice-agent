# Phase 4: Artifacts Panel

**Effort:** 3 days
**Impact:** Brings the extension's most impressive feature to the web — live rendering of HTML, React, SVG, Mermaid, Charts

---

## The Vision

When Aura generates a UI component, a chart, a diagram, or an interactive page, it renders live in a preview panel next to the chat — just like Claude Artifacts. Users see the result immediately, can iterate, and export.

---

## 4A. Artifacts Panel Component

### Layout
Two-column layout when artifacts are active:
```
┌─────────────────────────────────────────────┐
│ [Sidebar] │ [Chat]          │ [Artifacts]   │
│           │ Messages        │ Live Preview  │
│           │ Input           │ Code View     │
│           │                 │ Controls      │
└─────────────────────────────────────────────┘
```

- Chat column shrinks from 100% to ~55% when artifacts open
- Artifacts panel takes ~45%
- Resizable divider between them
- Artifacts panel can be closed (back to full-width chat)
- On mobile: artifacts take full screen with a tab toggle (Chat/Preview)

### New Component: `web/src/components/ArtifactsPanel.tsx`

**Supported types** (same as extension):
- `html` — full HTML pages
- `react` — JSX/TSX with ESM imports via esm.sh
- `svg` — inline SVG
- `mermaid` — diagrams via mermaid.js
- `chart` — Chart.js visualizations
- `markdown` — rendered markdown
- `css` — CSS with preview

### Core Rendering: `buildSrcdoc(type, code)`
Port the `buildSrcdoc()` function from the extension's ArtifactsPanel:
- HTML: inject error handler + console interceptor, auto-detect Tailwind classes → add CDN
- React: detect imports → build ESM import map via esm.sh, React 19 + common libraries
- SVG: minimal wrapper
- Mermaid: load mermaid.js 11 from CDN, dark/light theme
- Chart.js: load from CDN
- Markdown: inline converter

### iframe Sandbox
```html
<iframe
  srcdoc={buildSrcdoc(type, code)}
  sandbox="allow-scripts allow-same-origin"
  style={{ width: '100%', height: '100%', border: 'none' }}
/>
```

### Error Handling
- iframe `window.onerror` and `unhandledrejection` → `postMessage` to parent
- Parent displays errors in a console panel below the preview
- Error count badge on the console toggle

---

## 4B. Artifact Detection in Chat

### Auto-Detection
When the AI responds with a code block, detect if it's an artifact:

```typescript
function detectArtifactType(code: string, language: string): ArtifactType | null {
  if (language === 'html' && code.includes('<!DOCTYPE') || code.includes('<html')) return 'html';
  if (language === 'jsx' || language === 'tsx') return 'react';
  if (language === 'svg' && code.includes('<svg')) return 'svg';
  if (language === 'mermaid' || code.trim().startsWith('graph') || code.trim().startsWith('sequenceDiagram')) return 'mermaid';
  if (code.includes('new Chart(') || code.includes('Chart.js')) return 'chart';
  return null;
}
```

### Trigger
When an artifact is detected in a message:
1. Show a "Preview" button on the code block
2. Clicking opens the Artifacts panel with the code rendered
3. Or: auto-open the panel if user has "Auto-preview artifacts" enabled in settings

### Manual Trigger
User can click any code block → "Open as Artifact" to preview it.

---

## 4C. View Modes

**Preview** — iframe rendering (default)
**Code** — syntax-highlighted source code (Shiki from Phase 3)
**Split** — side-by-side preview + code

Toggle buttons in the artifacts toolbar.

---

## 4D. Artifact Controls

Toolbar at the top of the artifacts panel:
- **Type selector**: HTML / React / SVG / Mermaid / Chart / Markdown
- **View mode**: Preview / Code / Split
- **Copy code** button
- **Download** button (as .html, .svg, .md, etc.)
- **Open in new tab** — window.open with the rendered HTML
- **Device preview**: Desktop / Tablet (768px) / Mobile (375px)
- **Console toggle** with error count badge
- **Close panel** (X)

---

## 4E. Iteration

When the artifacts panel is open and the user sends a new message:
1. The AI receives the current artifact code as context
2. The AI can generate an updated version
3. The preview updates in real-time (streaming)

### Version History
- Keep last 10 versions
- Undo/Redo buttons in toolbar
- Click to compare (before/after)

---

## 4F. Quick Start Templates

Same as extension — quick buttons to generate common artifacts:
- Webpage, Chart, Mind Map, Flowchart, Game, Slides

Each sends a pre-built prompt to the AI with the appropriate artifact type.

---

## 4G. Streaming Preview

Port `StreamingPreviewController` from the extension:
- Debounce rendering (250ms, 1s max-wait)
- Check safe HTML boundaries (don't render inside unclosed tags)
- Show loading skeleton during initial generation

---

## Definition of Done — Phase 4
- [ ] Artifacts panel renders in a right column alongside chat
- [ ] HTML, React, SVG, Mermaid, Chart.js, and Markdown all render correctly
- [ ] React artifacts use ESM import maps via esm.sh (React 19 + common libs)
- [ ] Auto-detect artifact type from code blocks in chat
- [ ] "Preview" button on artifact-compatible code blocks
- [ ] View modes: Preview, Code, Split
- [ ] Device preview: Desktop, Tablet, Mobile
- [ ] Console panel with error capture from iframe
- [ ] Streaming preview with debounced rendering
- [ ] Version history with undo/redo (10 versions)
- [ ] Download and Copy buttons
- [ ] Quick start templates
- [ ] Resizable panel divider
- [ ] Mobile: full-screen artifact with Chat/Preview tab toggle
- [ ] Light theme compatible
