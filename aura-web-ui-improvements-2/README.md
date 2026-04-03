# Aura Web UI Improvements — Roadmap v2

State-of-the-art upgrade plan for Aura's web dashboard (`web/`).

## Phases

| Phase | Focus | Files |
|-------|-------|-------|
| 1 | Fix Broken Things — Light theme, regenerate, syntax highlighting, mobile drawer | [phase-1-fix-broken.md](phase-1-fix-broken.md) |
| 2 | Chat UX Upgrades — Message editing, retry, export, keyboard shortcuts | [phase-2-chat-ux.md](phase-2-chat-ux.md) |
| 3 | Code Syntax Highlighting + Code Interpreter | [phase-3-code.md](phase-3-code.md) |
| 4 | Artifacts Panel — Render HTML, React, SVG, Mermaid, Charts | [phase-4-artifacts.md](phase-4-artifacts.md) |
| 5 | Web Creator — Chat-to-website with live preview | [phase-5-web-creator.md](phase-5-web-creator.md) |
| 6 | Image Generation + Media Panel | [phase-6-image-gen.md](phase-6-image-gen.md) |
| 7 | Performance + Polish — Reduce polling, drop framer-motion, responsive fixes | [phase-7-performance.md](phase-7-performance.md) |
| 8 | Mobile-First Overhaul — PWA, native feel, offline support | [phase-8-mobile.md](phase-8-mobile.md) |

## Current State (Baseline)

### What Works Well
- Chat: streaming, citations, tool traces, attachments, 7 action modes, model compare, voice input
- Sidebar: emotion panel, thought stream, system stats, conversations with search/date grouping
- Advanced tab: MCTS reasoning tree, NeuroDream, A-MEM Zettelkasten memory
- Monitoring: ThoughtStream, AuraPanel with energy/warmth bars
- Settings: 38 AI provider key management, ALMA personality sliders
- Design: animated dark theme, mood-reactive CSS variables, mesh gradient background
- Mobile baseline: swipe gestures, safe areas, touch targets, 100dvh

### What's Broken
- Light theme: toggle exists, zero CSS rules — does nothing
- Regenerate button: console.log stub
- showThinking setting: wired but never consumed
- Mobile swipe drawer: gesture works, body is placeholder text
- Code blocks: no syntax highlighting (monochrome)
- CitationsPanel toggle disappears when open

### What's Missing (Extension has it)
- Artifacts panel (HTML/React/SVG/Mermaid/Chart rendering)
- Web Creator (chat-to-website builder)
- Code Interpreter (Pyodide + backend Python execution)
- Image generation UI
- Message editing + retry
- Conversation export
- Component library, design tokens, cross-panel workflows

### Key Files
```
web/src/App.tsx                          — Shell layout, 6 tabs
web/src/store/chatStore.ts               — All volatile chat state
web/src/store/settingsStore.ts           — Theme, font, behavior toggles
web/src/hooks/useWebSocket.ts            — WS connection + message handling
web/src/hooks/usePolling.ts              — Staggered REST polling manager
web/src/hooks/useMoodTheme.ts            — Live emotion → CSS variable mapping
web/src/components/ChatContainer.tsx     — Main chat surface
web/src/components/MessageBubble.tsx     — Individual message rendering
web/src/components/MessageInput.tsx      — Input bar with attachments, modes
web/src/components/Sidebar.tsx           — Left sidebar with panels
web/src/components/SettingsPage.tsx       — Provider keys, appearance, ALMA
web/src/components/ThoughtStream.tsx      — Inner monologue feed
web/src/components/ReasoningTreePanel.tsx — MCTS reasoning
web/src/components/NeuroDreamPanel.tsx    — Dream engine
web/src/components/AMEMPanel.tsx          — A-MEM memory
web/src/components/ActivityTimeline.tsx   — Live event feed
web/src/components/ToolsPanel.tsx         — Tool discovery
web/src/components/CodeBlock.tsx          — Code rendering (no highlighting)
web/src/index.css                        — Tailwind + custom CSS vars
web/package.json                         — React 18, Zustand 4, Vite 5
```
