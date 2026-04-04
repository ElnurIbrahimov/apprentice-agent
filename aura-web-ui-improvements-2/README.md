# Aura Web UI Improvements — Roadmap v2

**Last updated:** 2026-04-04

---

## Extension vs Web UI — Gap Analysis

The browser extension has **26 panels**. The web UI only has a fraction of these.

### What the Extension Has vs Web UI

| Extension Panel | Web UI Equivalent | Status |
|----------------|-------------------|--------|
| **ChatPanel** | ChatContainer + MessageBubble + MessageInput | ✅ Done |
| **CodePanel** | CodeInterpreter (Pyodide + Ask AI) | ✅ Done |
| **WebCreatorPanel** | WebCreator (live streaming, model selector) | ✅ Done |
| **ImagePanel** | ImageGenPanel (ComfyUI + SVG fallback) | ✅ Done |
| **ArtifactsPanel** | ArtifactsPanel | ✅ Done |
| **ToolsPanel** | ToolsPanel | ✅ Done |
| **SettingsPanel** | SettingsPage + SettingsModal | ✅ Done |
| **ModelsPanel** | ModelsPanel (dedicated tab) | ✅ Done |
| **ComparePanel** | ComparePanel (dedicated tab) | ✅ Done |
| **SearchPanel** | SearchPanel | ✅ Done |
| **ResearchPanel** | ResearchPanel | ✅ Done |
| **AgentPanel** | AgentPanel | ✅ Done |
| **AskPanel** | AskPanel | ✅ Done |
| **WritePanel** | WritePanel | ✅ Done |
| **TranslatePanel** | TranslatePanel | ✅ Done |
| **GrammarPanel** | GrammarPanel | ✅ Done |
| **SummaryPanel** | SummaryPanel | ✅ Done |
| **MathPanel** | MathPanel | ✅ Done |
| **PdfPanel** | PdfPanel | ✅ Done |
| **OcrPanel** | OcrPanel | ✅ Done |
| **SlidesPanel** | SlidesPanel | ✅ Done |
| **VoicePanel** | VoicePanel | ✅ Done |
| **RecordPanel** | RecordPanel | ✅ Done |
| **YoutubePanel** | YoutubePanel | ✅ Done |
| **CapturePanel** | CapturePanel | ✅ Done |
| **WisebasePanel** | WisebasePanel | ✅ Done |

**Score: 26/26 done**

### Web UI Has But Extension Doesn't

These are web-only features not in the extension:

| Web UI Component | Description |
|-----------------|-------------|
| ReasoningTreePanel | MCTS reasoning tree visualization |
| NeuroDreamPanel | Dream engine visualization |
| AMEMPanel | A-MEM Zettelkasten memory |
| ThoughtStream | Inner monologue feed |
| AuraPanel | Energy/warmth/mood bars |
| ActivityTimeline | Live event feed |
| HandsDashboard | Hands (tool execution) dashboard |
| EmotionPanel | Emotion state display |
| ProactiveDaemonPanel | Proactive awareness display |
| ContextHeatmap | Code context heatmap |
| IdleBehaviorPanel | Idle behavior visualization |
| MotivationDrivesPanel | Motivation drives display |
| DreamJournal | Dream journal |
| SystemStatsPanel | System resource stats |

---

## Phases

### Phase 1: Fix Broken Things ✅ COMPLETED (2026-04-04)

All items verified and either already done or fixed:

- [x] Light theme (full CSS overrides, blob/grain softening, scrollbar styling)
- [x] Regenerate button (wired to re-send user message)
- [x] showThinking toggle (ChatContainer respects setting)
- [x] Mobile swipe drawer (renders ConversationList)
- [x] CitationsPanel toggle (always visible when citations exist)
- [x] Emotion constants extracted to shared utils
- [x] Code block syntax highlighting reacts to theme changes

### Phase 2: Chat UX Upgrades ⚠️ PARTIALLY DONE

| Feature | Status | Notes |
|---------|--------|-------|
| Message editing | ✅ Done | Edit mode in MessageBubble |
| Regenerate/retry | ✅ Done | Context menu + button |
| Conversation export | ❌ Missing | Need export to JSON/MD |
| Keyboard shortcuts (Ctrl+K, etc.) | ⚠️ Partial | Focus shortcut exists, need more |
| Message reactions/feedback | ❌ Missing | Thumbs up/down for RLHF |
| Copy message | ✅ Done | Context menu |
| Share message | ✅ Done | Web Share API |

### Phase 3: Code Syntax Highlighting + Interpreter ✅ COMPLETED

- [x] Shiki-based syntax highlighting (highlightCode util)
- [x] Code Interpreter with Pyodide (client-side Python)
- [x] AI code generation ("Ask AI" button)
- [x] Model selector for code gen
- [x] Example quick-start snippets

### Phase 4: Artifacts Panel ✅ COMPLETED

- [x] HTML/CSS/JS rendering in sandboxed iframe
- [x] Device size preview (desktop/tablet/mobile)
- [x] Code view with syntax highlighting
- [x] Split view mode

### Phase 5: Web Creator ✅ COMPLETED

- [x] Chat-to-website builder
- [x] Live code streaming during generation
- [x] Template picker (7 templates)
- [x] Version history (undo/redo)
- [x] Model selector
- [x] Download HTML / Open in new tab
- [x] Device preview modes

### Phase 6: Image Generation ✅ COMPLETED

- [x] ComfyUI integration (when available)
- [x] SVG fallback via LLM (when no GPU)
- [x] Model selector
- [x] Style presets (9 styles)
- [x] History with localStorage persistence
- [x] Download and copy

### Phase 7: Performance + Polish ✅ COMPLETED

- [x] Reduce polling overhead — Staggered polling implemented
- [x] Drop framer-motion — Removed from deps
- [x] Lazy load heavy components — React.lazy on all 26 tabs
- [x] Responsive mobile fixes — dvh, touch targets, fullscreen layouts
- [x] Model routing speed — Skip LLM classifier for conversation

### Phase 8: Mobile-First Overhaul ✅ COMPLETED

- [x] PWA manifest + service worker
- [x] Offline support / cache-first strategy
- [x] Native-feel gestures (swipe tabs, pull-to-refresh)
- [x] Bottom sheet for model selection
- [x] Haptic feedback (vibration API)
- [x] Install prompt

---

## Phase 9: Missing Extension Panels ✅ COMPLETED (2026-04-04)

All 19 remaining extension panels built and integrated.

Built panels:
- SearchPanel, ResearchPanel, AgentPanel
- WritePanel, TranslatePanel, GrammarPanel, SummaryPanel, AskPanel
- MathPanel, PdfPanel, OcrPanel, VoicePanel
- SlidesPanel, RecordPanel, YoutubePanel, CapturePanel, WisebasePanel
- ModelsPanel, ComparePanel

---

## Web Creator Features

8 new features added to WebCreator:

1. **Streaming generation** — Real-time code stream during generation
2. **Model selector** — Choose model per generation
3. **Template system** — 7 pre-built starting points (blank, portfolio, blog, docs, ecommerce, dashboard, landing)
4. **Version history** — Full undo/redo with version browser
5. **Device preview** — Desktop, tablet, mobile viewports
6. **Download HTML** — Export standalone HTML file
7. **Open in new tab** — Live preview in separate window
8. **Live code editing** — Edit generated code in-panel with live preview

---

## Session 2026-04-04 Changelog

**Massive phase completion: 26/26 panels now feature-complete.**

### Panels Built (19 new)

**Tier 1 (High-value generative):**
- SearchPanel — Web search with streaming results
- ResearchPanel — Multi-source deep research
- AgentPanel — Autonomous task execution
- WritePanel — Long-form writing with drafting
- TranslatePanel — Multi-language translation

**Tier 2 (Content tools):**
- SummaryPanel — Text/URL/doc summarization
- GrammarPanel — Grammar/style/tone analysis
- AskPanel — Quick Q&A interface
- MathPanel — Math problem solver (Pyodide + LLM)
- PdfPanel — PDF parsing + analysis

**Tier 3 (Specialized):**
- OcrPanel — Image text extraction
- VoicePanel — TTS + voice input
- SlidesPanel — Presentation builder
- RecordPanel — Audio recording + transcription
- YoutubePanel — YouTube transcript + analysis
- CapturePanel — Screenshot + annotation
- WisebasePanel — Knowledge base browser
- ModelsPanel — Model explorer (latency, cost, capability)
- ComparePanel — Side-by-side model comparison

### WebCreator Features (8 new)

- Streaming code generation with real-time display
- Per-generation model selector
- 7 template starters
- Full version history (undo/redo)
- Multi-device preview modes
- HTML export + open in new tab
- In-panel code editor with live preview

### Shared Utilities Created

- `GenerativePanel` base component — Reusable chat + streaming + model selector
- `systemPrompts` object — Prompt templates for all 19 panels
- `outputFormatters` — Specialized renderers (code, markdown, tables, etc.)
- `panelConfig` — Unified panel metadata (icon, label, description, shortcut)
- Theme color mappings — Emotion-aware UI theming
- Citation/reference rendering — Unified source attribution

### Security & Performance

- XSS prevention in all generative outputs
- Rate-limiting awareness (model/tool-specific backoff)
- Streaming cancellation on unmount
- Reduced re-renders via `React.memo` + normalized stores
- Service worker for offline state awareness

### Mobile Polish

- Touch-friendly spacing on all panels
- Swipe gestures for panel switching
- Full-height responsive layouts (dvh)
- Bottom-sheet model selector (mobile)
- Haptic feedback integration

### Bug Fixes

- Fixed modal overlay z-index conflicts
- Resolved streaming timeout on slow networks
- Fixed scroll-to-bottom in long outputs
- Corrected citation panel visibility logic
- Fixed theme persistence across tab switches

---

## Key Files

```
web/src/App.tsx                          — Shell layout, 26 tabs
web/src/store/chatStore.ts               — Chat state (Zustand)
web/src/store/settingsStore.ts           — Theme, font, behavior
web/src/hooks/useWebSocket.ts            — WS connection + messages
web/src/hooks/usePolling.ts              — REST polling manager
web/src/hooks/useMoodTheme.ts            — Emotion → CSS mapping
web/src/hooks/useGenerativeStream.ts     — Streaming generation (new)
web/src/components/ChatContainer.tsx     — Main chat surface
web/src/components/MessageBubble.tsx     — Message rendering
web/src/components/MessageInput.tsx      — Input bar + model selector
web/src/components/WebCreator.tsx        — Chat-to-website builder
web/src/components/CodeInterpreter.tsx   — Python REPL + AI gen
web/src/components/ImageGenPanel.tsx     — Image gen (ComfyUI/SVG)
web/src/components/ArtifactsPanel.tsx    — HTML/React/SVG preview
web/src/components/SettingsPage.tsx      — Provider keys, appearance
web/src/components/GenerativePanel.tsx   — Base component for all tools (new)
web/src/components/SearchPanel.tsx       — Web search
web/src/components/ResearchPanel.tsx     — Deep research
web/src/components/AgentPanel.tsx        — Task automation
web/src/components/WritePanel.tsx        — Writing assistant
web/src/components/TranslatePanel.tsx    — Multi-language
web/src/components/GrammarPanel.tsx      — Grammar/style
web/src/components/SummaryPanel.tsx      — Text summarization
web/src/components/AskPanel.tsx          — Quick Q&A
web/src/components/MathPanel.tsx         — Math solver
web/src/components/PdfPanel.tsx          — PDF analysis
web/src/components/OcrPanel.tsx          — Image text extraction
web/src/components/VoicePanel.tsx        — TTS + voice input
web/src/components/SlidesPanel.tsx       — Presentation builder
web/src/components/RecordPanel.tsx       — Audio recording
web/src/components/YoutubePanel.tsx      — Video analysis
web/src/components/CapturePanel.tsx      — Screenshot tool
web/src/components/WisebasePanel.tsx     — Knowledge base
web/src/components/ModelsPanel.tsx       — Model explorer
web/src/components/ComparePanel.tsx      — Model comparison
web/src/utils/systemPrompts.ts           — All panel prompts (new)
web/src/utils/outputFormatters.ts        — Panel output renderers (new)
web/src/utils/panelConfig.ts             — Panel metadata (new)
web/src/index.css                        — Tailwind + themes + tokens
```
