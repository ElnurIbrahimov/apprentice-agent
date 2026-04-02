# Aura Extension Roadmap — Web Builder, Artifacts & Code Creation

State-of-the-art upgrade plan for Aura's creation panels.

## Phases

| Phase | Focus | Files |
|-------|-------|-------|
| 1 | Quick Wins — Error Recovery, Persistence, Gallery | [phase-1-quick-wins.md](phase-1-quick-wins.md) |
| 2 | Code Editor Upgrade — CodeMirror 6 + Diff View | [phase-2-code-editor.md](phase-2-code-editor.md) |
| 3 | Multi-File Projects + Virtual Filesystem | [phase-3-multi-file.md](phase-3-multi-file.md) |
| 4 | Agent Build Mode — Aura's Differentiator | [phase-4-agent-build-mode.md](phase-4-agent-build-mode.md) |
| 5 | WebContainer Runtime — Full Node.js in Browser | [phase-5-webcontainer.md](phase-5-webcontainer.md) |
| 6 | Deploy & Share | [phase-6-deploy.md](phase-6-deploy.md) |
| 7 | Visual AI Feedback Loop | [phase-7-visual-feedback.md](phase-7-visual-feedback.md) |
| 8 | Cross-Panel Workflows + Component Library | [phase-8-cross-panel.md](phase-8-cross-panel.md) |

## Current State (Baseline)

### What Exists Today
- **WebCreatorPanel** (59K) — Chat-to-HTML website builder, single-file, iframe preview, element selection, version history, theme editor
- **ArtifactsPanel** (58K) — Generate + Live modes, 7 artifact types (HTML/React/SVG/Mermaid/Chart/Markdown/CSS), ESM import maps via esm.sh, WebSocket live sync
- **CodePanel** (37K) — Python REPL via Pyodide WASM, backend fallback, variable inspector, Fix Error button
- **CapturePanel** (56K) — Clone existing web pages via screenshot + HTML extraction
- **SlidesPanel** — AI slide deck generator
- **scaffold.py** — 6 project templates (Next.js, React-Vite, FastAPI, Express, Static, Chrome Extension)
- **tool_builder.py** — Runtime tool creation with VOYAGER-style composition

### Key Architecture
- All previews use `<iframe srcdoc="...">` with error/console interception via postMessage
- Code editing uses plain `<textarea>` + highlighted `<pre>` overlay (no real editor)
- Streaming via `StreamingPreviewController` (250ms debounce, 1s max-wait)
- Version history via `useVersionHistory` hook (20 versions, chrome.storage.local)
- Backend: `/api/generate/raw` (SSE, direct Ollama, bypasses agent pipeline)
- Live mode: WebSocket `/api/artifacts/stream` broadcasts from agentic_loop file writes

### Key Files
```
extension-src/src/panels/WebCreatorPanel.tsx
extension-src/src/panels/ArtifactsPanel.tsx
extension-src/src/panels/CodePanel.tsx
extension-src/src/panels/CapturePanel.tsx
extension-src/src/utils/StreamingPreviewController.ts
extension-src/src/utils/useVersionHistory.ts
extension-src/src/utils/PyodideExecutor.ts
extension-src/src/utils/highlighter.ts
extension-src/src/workers/pyodide-worker.ts
extension-src/src/store.ts
api/routes/generate.py
api/routes/artifacts.py
api/routes/code.py
aura/tools/scaffold.py
aura/tools/tool_builder.py
aura/sandbox/executor.py
```
