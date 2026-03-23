# AURA Web UI

React frontend for the AURA agent. Built with Vite, TypeScript, Tailwind CSS, and Zustand.

## Setup

```bash
npm install
npm run dev
```

Runs at `http://localhost:5173`. Requires the AURA API server on port 8000 — start it first:

```bash
# In the project root
python run_web.py
```

The Vite dev server proxies all `/api` requests to `http://127.0.0.1:8000` automatically.

## Build for production

```bash
npm run build
```

Output in `dist/`. Can be served statically or via `npm run preview`.

In production (`AURA_ENV=production`), the FastAPI server serves the built React app from `web/dist/` directly.

## Stack

- **React 18** + TypeScript
- **Vite 5** — dev server + bundler
- **Tailwind CSS 3** — styling (dark glassmorphic theme)
- **Zustand** — global state management
- **react-markdown** — message rendering with citation support
- **@heroicons/react** — icons

## Components (37)

| Component | Purpose |
|-----------|---------|
| `App.tsx` | Root layout, tab routing, WebSocket lifecycle |
| `ChatContainer.tsx` | Message thread, scroll management, research progress, citations panel |
| `MessageBubble.tsx` | Individual message with markdown, inline citation badges, tool traces |
| `MessageInput.tsx` | Text input, file attach, action mode selector (search/research/agent/swarm/compare), model picker, voice input |
| `ResearchProgress.tsx` | Live research progress display — plan, search, source, finding, synthesis stages with animations |
| `CitationsPanel.tsx` | Collapsible sidebar with grouped citations, bidirectional hover highlight with inline badges |
| `Sidebar.tsx` | Navigation, conversation list, panel switching |
| `ConversationList.tsx` | Session history with rename and search |
| `ConversationStarterPanel.tsx` | Suggested conversation starters |
| `FleetDashboard.tsx` | Multi-agent task tracker for parallel sub-agents |
| `ProactiveCard.tsx` | Real-time push notification cards from proactive daemon |
| `ToolTrace.tsx` | Tool call trace visualization |
| `EmotionPanel.tsx` | Live neuromodulator state (ALMA engine) |
| `MemoryRecallIndicator.tsx` | Shows when memory is being recalled |
| `MemoryIndicator.tsx` | Memory status indicator |
| `MoodIndicator.tsx` | Current mood display |
| `InnerThoughtsPanel.tsx` | AURA's visible thinking / scratchpad |
| `ThoughtStream.tsx` | Real-time thought stream |
| `NeuroDreamPanel.tsx` | Dream mode / consolidation status |
| `ReasoningTreePanel.tsx` | Reasoning tree visualization (UCB1 scoring) |
| `AMEMPanel.tsx` | Memory browser |
| `AuraPanel.tsx` | Agent state overview |
| `AuraBreathingAvatar.tsx` | Animated avatar with mood-based breathing |
| `ContextHeatmap.tsx` | Context relevance scoring visualization |
| `SystemStatsPanel.tsx` | API latency, token usage, model info |
| `ToolsPanel.tsx` | Tool catalog with categories |
| `ActivityTimeline.tsx` | Recent agent activity log |
| `IdleBehaviorPanel.tsx` | Idle behavior visualization |
| `MotivationDrivesPanel.tsx` | Intrinsic motivation drives |
| `ProactiveDaemonPanel.tsx` | Proactive daemon status |
| `DreamJournal.tsx` | Dream journal entries |
| `ThinkingAboutTeaser.tsx` | "Thinking about..." preview |
| `ModelCompare.tsx` | Multi-model comparison view |
| `AttachmentPreview.tsx` | File attachment preview |
| `BottomSheet.tsx` | Mobile bottom sheet |
| `SettingsModal.tsx` | API key, model preferences, UI options |
| `Toast.tsx` | Notifications |
| `ErrorBoundary.tsx` | React error boundary |

## State

Global state managed by Zustand (`src/store/`):

- `chatStore` — messages, sessions, loading state, research progress, citations, fleet data, connection status, mood, available models, tool status
- `settingsStore` — API key, preferences, theme, font size, sound

## WebSocket

The UI maintains a WebSocket connection to `/api/chat/stream` for real-time streaming. Reconnects automatically with exponential backoff (1s → 30s cap, max 10 attempts). Heartbeat ping every 30 seconds. 90-second response timeout watchdog.

Message types received from server:

| Type | Purpose |
|------|---------|
| `chunk` | Streaming response token — appended to current message |
| `done` | Stream complete — marks message finished, plays TTS if enabled |
| `error` | Error message |
| `stopped` | Generation was stopped by user |
| `tool_status` | "Using tool X..." shimmer indicator |
| `citations` | Structured citation data attached to message |
| `tool_trace` | Tool call trace appended to message |
| `proactive` | Push notification from proactive daemon |
| `research_progress` | Live research stage updates (plan/search/source/finding/synthesis) |
| `pong` | Heartbeat response (silently discarded) |

## Action Modes

The message input supports multiple action modes sent to the backend:

- `search` — AI-powered web search
- `research` — Quick research
- `deep_research` — Deep research with live progress streaming
- `agent` — Full agentic mode
- `swarm` — Multi-agent fleet mode
- `compare` — Multi-model comparison (uses REST, not WebSocket)
