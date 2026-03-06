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

## Stack

- **React 18** + TypeScript
- **Vite 5** — dev server + bundler
- **Tailwind CSS 3** — styling
- **Zustand** — global state
- **react-markdown** — message rendering
- **@heroicons/react** — icons

## Component Map

| Component | Purpose |
|-----------|---------|
| `App.tsx` | Root layout, WebSocket lifecycle |
| `ChatContainer.tsx` | Message thread, scroll management |
| `MessageBubble.tsx` | Individual message with markdown rendering |
| `MessageInput.tsx` | Text input, file attach, send |
| `Sidebar.tsx` | Navigation and panel switching |
| `ConversationList.tsx` | Session history |
| `EmotionPanel.tsx` | Live neuromodulator state (ALMA) |
| `MemoryRecallIndicator.tsx` | Shows when memory is being recalled |
| `InnerThoughtsPanel.tsx` | AURA's scratchpad / visible thinking |
| `ThoughtStream.tsx` | Real-time thought stream |
| `NeuroDreamPanel.tsx` | Dream mode / consolidation status |
| `ProtoAGIPanel.tsx` | Autonomous drive loop status |
| `GuardianPanel.tsx` | Metacognition guardian pre-flight results |
| `ReasoningTreePanel.tsx` | Explicit reasoning tree visualization |
| `AMEMPanel.tsx` | Associative memory browser |
| `AuraPanel.tsx` | Agent state overview |
| `SystemStatsPanel.tsx` | API latency, token usage, model info |
| `ContextHeatmap.tsx` | Context relevance scoring visualization |
| `ActivityTimeline.tsx` | Recent agent activity log |
| `SettingsModal.tsx` | API key, model preferences, UI options |
| `Toast.tsx` | Notifications |

## State

Global state managed by Zustand (`src/store/`):

- `chatStore` — messages, sessions, loading state
- `agentStore` — agent state, emotion levels, memory status
- `settingsStore` — API key, preferences

## WebSocket

The UI maintains a WebSocket connection to `/api/ws/{session_id}` for real-time streaming. Reconnects automatically on disconnect.

Message types received from server:

- `token` — streaming response token
- `thought` — inner thought / scratchpad update
- `emotion` — neuromodulator state update
- `memory_recall` — memory retrieval event
- `done` — stream complete
- `error` — error message
