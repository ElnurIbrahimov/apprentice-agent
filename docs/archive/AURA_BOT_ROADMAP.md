> SUPERSEDED 2026-04-13. Current source of truth: D:/Aura/CURRENT_STATE.md

# AURA Bot Roadmap — Telegram, Messaging & Cross-Surface Integration

## Current State (v4.5.0, March 24 2026)

### What the Telegram Bot Can Do Now
- Full ReAct agent loop with tools (not just brain.think)
- `/research <topic>` — deep research with live progress + citations
- `/search <query>` — web search with formatted results
- `/summarize` — URL fetch + summarize, or summarize replied text
- `/code <python>` — sandboxed code execution with plot support
- `/model` / `/model <name>` — list/switch models
- `/compare <prompt>` — parallel 3-model comparison
- `/image <prompt>` — image generation
- Voice messages → Whisper transcription → AI response
- Photo messages → vision model analysis
- Document upload (PDF/text) → extraction → 30min Q&A context
- Inline mode (`@Aura828Bot` in any chat)
- Rich formatting (MarkdownV2, citation links, message splitting)
- Progress streaming (live-editing messages during research/tool use)
- `/status` `/mood` `/memory` `/forget` `/help`

### Other Surfaces
- **CLI** — full agent with 42 slash commands, 6 modes, Rich terminal UI
- **Web UI** — React dashboard, 5 tabs, 37 components, WebSocket streaming
- **Browser Extension** — 25 panels, floating UI, SERP answers, Gmail compose
- **WhatsApp** — basic chat via Baileys bridge (no tools, no commands)

### Problem
Each surface has its own isolated conversation. Can't start on Telegram, continue on web. No cross-surface awareness.

---

## Phase 1: More Messaging Channels (steal from OpenClaw)

OpenClaw supports 20+ channels. We don't need all of them. Priority by actual usefulness:

| Channel | Priority | Why | Protocol |
|---|---|---|---|
| **Discord** | High | Developer communities, always-on servers | Gateway WebSocket (no public IP) |
| **Signal** | High | Privacy-focused users | signal-cli bridge |
| **Slack** | High | Work/team access | Socket Mode (no public IP) |
| **Matrix** | Medium | Open protocol, self-hosted communities | matrix-nio SDK |
| **iMessage** | Medium | iOS users (Mac server required) | BlueBubbles bridge |
| **Microsoft Teams** | Low | Enterprise | Bot Framework SDK |
| **IRC** | Low | Niche but simple | Plain socket |

### Architecture
Already have `MessageRouter` in `aura/messaging/router.py` with normalized inbound/outbound models. Each new channel = one adapter file implementing the same interface. Pattern:

```
New message on Discord
  → DiscordAdapter.handle_incoming()
    → MessageRouter.route()
      → agent.run(goal) [full ReAct loop]
    → DiscordAdapter.send_response()
```

### Per-Channel Features
Every channel should support (where the platform allows):
- Text messages → full agent loop
- Voice/audio → transcription → AI response
- Images → vision model analysis
- Documents → extraction → Q&A context
- Commands (/ prefix or platform equivalent)
- Progress updates (edit messages or send status)
- Inline/slash commands (Discord, Slack, Telegram)

---

## Phase 2: Cross-Surface Conversation Sync

### Goal
Start a conversation on Telegram, continue on web UI, check results on the extension. All surfaces see the same conversation history.

### Architecture

**Shared Conversation Store:**
- Already exists at `data/conversations/<id>/history.json`
- Add a `ConversationManager` singleton that all surfaces read/write through
- Each surface gets a `surface_id` tag on messages (telegram, web, extension, cli)
- Messages from any surface appear in the shared history

**Real-Time Sync via WebSocket Hub:**
- Web UI already has WebSocket at `/api/chat/stream`
- Add a broadcast system: when any surface adds a message, push to all connected surfaces
- Telegram gets updates via `broadcast_proactive_message()`
- Extension gets updates via its existing WebSocket connection
- CLI gets updates via a new event listener (or polling)

**Session Binding:**
- Telegram user → bound to a conversation ID
- `/session new` — start fresh conversation
- `/session list` — show recent conversations
- `/session <id>` — switch to existing conversation
- Web UI sidebar shows which conversations have Telegram activity (and vice versa)

**Implementation Order:**
1. `ConversationManager` with surface tagging
2. Telegram commands for session management
3. Web UI shows cross-surface messages with source badges
4. Extension sidebar shows same
5. CLI event listener for remote messages

---

## Phase 3: CLI Bridge (like Claude Code Channels)

### Goal
Telegram messages appear in a running CLI session. Respond from CLI, reply goes back to Telegram.

### How Claude Code Does It
- MCP server plugin runs alongside Claude Code
- Telegram long-polls → pushes events into the session as `<channel source="telegram">`
- Claude processes with full local filesystem access
- Replies via the channel's `reply` tool → appears in Telegram

### How Aura Should Do It
Aura already has an MCP server (`aura/core/mcp_server.py`). Add a "channels" mode:

```
aura --channels telegram
```

This would:
1. Start the normal CLI session
2. Also start the Telegram bot in a background thread
3. Incoming Telegram messages appear in the CLI as `[Telegram] user: message`
4. The CLI agent processes them with full local tool access
5. Responses are sent back to Telegram AND shown in CLI
6. User can type in CLI too — both inputs feed the same agent

**Key difference from current Telegram bot:** The CLI bridge gives Telegram access to LOCAL tools (filesystem, git, code execution on your machine), not just the server-side tools.

### Extension Bridge
Same pattern for the browser extension:
```
aura --channels extension
```
- Extension connects to the local CLI session via WebSocket
- Extension actions (capture page, summarize, etc.) feed into the CLI agent
- CLI has access to extension-captured content + local filesystem

---

## Phase 4: Steal from OpenClaw

### Webhook Triggers
External services push tasks to Aura via Telegram or any channel:
- GitHub CI failure → Aura gets notified, investigates, posts findings
- Server alert (high CPU, disk full) → Aura diagnoses
- Calendar event starting → Aura prepares meeting notes
- Email received (via Resend/Gmail webhook) → Aura summarizes

**Implementation:** Add `/webhook` command + API endpoint that accepts webhook payloads and routes them to the appropriate conversation.

### Scheduled Tasks from Chat
- "Remind me in 2 hours to check the deployment"
- "Every morning at 9am, summarize my GitHub notifications"
- "Check this URL every hour and tell me if it changes"

**Implementation:** Wire the existing `task_scheduler` tool to Telegram commands:
- `/remind <time> <message>` — one-shot reminder
- `/schedule <cron> <task>` — recurring task
- `/tasks` — list active schedules
- `/cancel <id>` — cancel a schedule

### Self-Modifying Skills from Chat
OpenClaw lets the agent write/update its own skills. Aura has GEPA for evolutionary optimization, but not runtime skill creation from chat.

- `/learn` — after a successful interaction, extract a reusable skill
- `/skill create <name>` — manually define a new skill procedure
- `/skill improve <name>` — trigger GEPA evolution on a specific skill
- Uses the existing `SkillLearner` + `SkillStore` + GEPA infrastructure

### Multi-Agent from Chat
- `/agent research <topic>` — spawn a research sub-agent
- `/agent code <task>` — spawn a coding sub-agent
- `/fleet <goal>` — full parallel decomposition (already in CLI, wire to Telegram)
- Show sub-agent progress via edited messages

---

## Phase 5: Platform-Specific Power Features

### Telegram-Specific
- **Telegram Mini App** — web app inside Telegram for rich UI (charts, dashboards, code editor)
- **Telegram Payments** — accept payments for premium features via Telegram's built-in payment system
- **Telegram Groups** — Aura as a group assistant (respond when mentioned, moderate, summarize threads)
- **Stickers/GIFs** — Aura sends contextual reactions
- **Location sharing** → local recommendations, weather, nearby info

### Discord-Specific
- **Slash commands** with autocomplete
- **Thread management** — each research task in its own thread
- **Voice channels** — join and transcribe meetings
- **Embeds** — rich formatted responses with images, fields, footers

### Slack-Specific
- **App Home** — dashboard tab in Slack
- **Workflow Builder** integration
- **Channel summaries** — "what happened in #dev while I was away?"

---

## Priority Order

| Phase | Effort | Impact | Status |
|---|---|---|---|
| Phase 2: Cross-surface sync | High | Very High | **DONE** (March 24, 2026) |
| Phase 3: CLI bridge | Medium | High | **DONE** (March 24, 2026) |
| Phase 4: Webhooks + scheduled tasks | Medium | High | **DONE** (March 24, 2026) |
| Phase 4: Self-modifying skills from chat | Low | Medium | **DONE** (March 24, 2026) |
| Phase 1: Discord + Slack channels | Medium | Medium | When needed |
| Phase 5: Platform-specific features | Varies | Nice-to-have | **DONE** (March 24, 2026) |

---

## Competitive Landscape

| Feature | Aura | OpenClaw | Claude Code Channels |
|---|---|---|---|
| Channels | 2 (Telegram, WhatsApp) | 20+ | 2 (Telegram, Discord) |
| Local tool access from chat | **Yes (--channels bridge)** | Yes (local gateway) | Yes (MCP plugin) |
| Memory | Deep (UnifiedMemory + KG + FadeMem) | Basic persistence | Session-only |
| Emotions/personality | Full ALMA + OCEAN | None | None |
| Reasoning depth | MCTS + Strategy Bandit + multi-agent | Basic LLM | Claude's built-in |
| Self-improvement | GEPA Pareto evolution | Self-modifying skills | None |
| Research | Deep research with citations + progress | Basic web search | Claude's built-in |
| Vision from chat | Photo → vision model | No | No (text only) |
| Code from chat | /code with sandbox + plots | Shell execution | Full Claude Code |
| Cross-surface sync | **ConversationManager + WebSocket broadcast** | Single gateway hub | Cowork + Dispatch |
| Marketplace | GitHub plugin registry | ClawHub (50+ skills) | Claude plugins |
| Price | Free (self-hosted) | Free (self-hosted) | Pro/Max subscription |

**Aura's moat:** Cognitive depth (memory, emotions, reasoning, self-improvement) that no other bot has.
**Aura's gap:** Channel breadth and cross-surface sync — fixable with engineering, not architectural innovation.

---

*Last updated: March 24, 2026*
*Session: Phase 2 (cross-surface sync) + Phase 3 (CLI bridge) implemented*

---

## What Was Built (March 24, 2026)

### Phase 2: Cross-Surface Conversation Sync
- **ConversationManager** (`aura/core/conversation_manager.py`) — singleton, surface bindings, message attribution, event broadcast
- **Surface tagging** — every message records which surface sent it (web, telegram, cli, extension)
- **Telegram /session commands** — `/session new|list|<id>|sync` for conversation management
- **WebSocket broadcast** — `conv_sync` events pushed to all connected clients in real-time
- **API endpoints** — `GET /sync/status`, `GET /conversations/:id/messages` with surface attribution
- **Cross-surface forwarding** — messages from one surface can forward to others bound to the same conversation

### Phase 3: CLI Bridge
- **ChannelBridge** (`aura/channels/channel_bridge.py`) — thread-safe message queue, background event loop, adapter management
- **TelegramChannel** (`aura/channels/telegram_channel.py`) — lightweight relay that queues messages for CLI agent
- **ExtensionChannel** (`aura/channels/extension_channel.py`) — WebSocket server on localhost:9828
- **Channel display** (`aura/channels/display.py`) — Rich terminal rendering with channel panels and notifications
- **CLI integration** — `aura --channels telegram extension`, `/channels` command, messages drain between interactions
- **Key feature**: Telegram/extension gets access to LOCAL tools (filesystem, git, code execution on your machine)