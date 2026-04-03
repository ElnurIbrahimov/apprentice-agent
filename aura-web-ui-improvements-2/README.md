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
| **ModelsPanel** | Model dropdown in chat input | ⚠️ Partial (no dedicated panel) |
| **ComparePanel** | ModelCompare component exists | ⚠️ Partial (not in main tabs) |
| **SearchPanel** | — | ❌ **Missing** |
| **ResearchPanel** | — | ❌ **Missing** |
| **AgentPanel** | — | ❌ **Missing** |
| **AskPanel** | — | ❌ **Missing** |
| **WritePanel** | — | ❌ **Missing** |
| **TranslatePanel** | — | ❌ **Missing** |
| **GrammarPanel** | — | ❌ **Missing** |
| **SummaryPanel** | — | ❌ **Missing** |
| **MathPanel** | — | ❌ **Missing** |
| **PdfPanel** | — | ❌ **Missing** |
| **OcrPanel** | — | ❌ **Missing** |
| **SlidesPanel** | — | ❌ **Missing** |
| **VoicePanel** | — | ❌ **Missing** |
| **RecordPanel** | — | ❌ **Missing** |
| **YoutubePanel** | — | ❌ **Missing** |
| **CapturePanel** | — | ❌ **Missing** |
| **WisebasePanel** | — | ❌ **Missing** |

**Score: 7/26 done, 2 partial, 17 missing**

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

### Phase 7: Performance + Polish 🔄 IN PROGRESS

| Item | Status | Notes |
|------|--------|-------|
| Reduce polling overhead | ⚠️ Partial | Staggered polling exists |
| Drop framer-motion | ✅ Done | Not in deps |
| Lazy load heavy components | ✅ Done | React.lazy for all tabs |
| Responsive mobile fixes | ⚠️ Partial | dvh, touch targets done |
| Model routing speed | ✅ Fixed | Skip LLM classifier for conversation |

### Phase 8: Mobile-First Overhaul ❌ NOT STARTED

- [ ] PWA manifest + service worker
- [ ] Offline support / cache-first strategy
- [ ] Native-feel gestures (swipe tabs, pull-to-refresh)
- [ ] Bottom sheet for model selection
- [ ] Haptic feedback (vibration API)
- [ ] Install prompt

---

## Phase 9: Missing Extension Panels (NEW — Largest Gap)

**This is the biggest remaining gap.** 17 extension panels have no web equivalent.

### Priority Tier 1 — High Value (use existing backend endpoints)

| Panel | What It Does | Backend Ready? | Effort |
|-------|-------------|----------------|--------|
| **SearchPanel** | Web search with Brave/Tavily | ✅ Yes (web_search tool) | Low |
| **ResearchPanel** | Deep multi-source research | ✅ Yes (deep_research tool) | Medium |
| **AgentPanel** | Autonomous multi-step tasks | ✅ Yes (agent mode) | Medium |
| **WritePanel** | Long-form writing assistant | ✅ Yes (generate/raw) | Low |
| **TranslatePanel** | Multi-language translation | ✅ Yes (generate/raw) | Low |

### Priority Tier 2 — Medium Value

| Panel | What It Does | Backend Ready? | Effort |
|-------|-------------|----------------|--------|
| **SummaryPanel** | Summarize text/URLs/docs | ✅ Yes (generate/raw) | Low |
| **GrammarPanel** | Grammar/style checker | ✅ Yes (generate/raw) | Low |
| **MathPanel** | Math problem solver | ⚠️ Partial (Pyodide + LLM) | Low |
| **PdfPanel** | PDF reading/analysis | ✅ Yes (pdf_reader tool) | Medium |
| **VoicePanel** | Text-to-speech / voice chat | ⚠️ Partial (TTS endpoint) | Medium |

### Priority Tier 3 — Nice to Have

| Panel | What It Does | Backend Ready? | Effort |
|-------|-------------|----------------|--------|
| **OcrPanel** | Image text extraction | ⚠️ Partial (vision models) | Medium |
| **SlidesPanel** | Presentation builder | ❌ No | High |
| **RecordPanel** | Audio recording + transcription | ❌ No | High |
| **YoutubePanel** | YouTube video analysis | ⚠️ Partial (transcript tool) | Medium |
| **CapturePanel** | Screenshot capture + annotation | ⚠️ Partial (screenshot tool) | Medium |
| **WisebasePanel** | Knowledge base management | ✅ Yes (unified memory) | Medium |
| **AskPanel** | Quick Q&A (similar to chat) | ✅ Yes (generate/raw) | Low |

### Implementation Strategy

Most Tier 1 and Tier 2 panels can be built as thin frontends over `/api/generate/raw` with custom system prompts — the same pattern used by WebCreator. They don't need new backend endpoints.

Suggested approach:
1. Create a shared `GenerativePanel` base component (chat + streaming + model selector)
2. Each panel is a thin wrapper with its own system prompt, UI chrome, and output formatter
3. Add panels as sub-tabs under a new "Tools" or "Assistants" section

---

## Key Files

```
web/src/App.tsx                          — Shell layout, 5 tabs
web/src/store/chatStore.ts               — Chat state (Zustand)
web/src/store/settingsStore.ts           — Theme, font, behavior
web/src/hooks/useWebSocket.ts            — WS connection + messages
web/src/hooks/usePolling.ts              — REST polling manager
web/src/hooks/useMoodTheme.ts            — Emotion → CSS mapping
web/src/components/ChatContainer.tsx     — Main chat surface
web/src/components/MessageBubble.tsx     — Message rendering
web/src/components/MessageInput.tsx      — Input bar + model selector
web/src/components/WebCreator.tsx        — Chat-to-website builder
web/src/components/CodeInterpreter.tsx   — Python REPL + AI gen
web/src/components/ImageGenPanel.tsx     — Image gen (ComfyUI/SVG)
web/src/components/ArtifactsPanel.tsx    — HTML/React/SVG preview
web/src/components/SettingsPage.tsx      — Provider keys, appearance
web/src/index.css                        — Tailwind + themes + tokens
```
