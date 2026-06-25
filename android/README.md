# Aura Android

A native Android superapp — Kotlin + Jetpack Compose, cloud-powered, your data stays private.

This is a port of the Python [Aura](https://github.com/ElnurIbrahimov/apprentice-agent) assistant to a native Android app. The goal is the full Aura capability surface: brain, agentic loop, 21 tools, memory stack, voice I/O, multi-agent routing, proactive monitors — all powered by cloud LLM providers.

## Status

**v0.1.0 — Day 7** (see `.hermes/plans/2026-06-25_161811-aura-android-superapp.md` for the plan)

What's working today:
- 21 tools (web search, location, calendar, contacts, tasks, reminders, share, notifications, biometric, app launcher, system volume, photo library, network state, battery state, DND, remember, recall, web fetch, system volume, get current time, camera)
- Memory stack (Room + FTS5 + vector cosine + 14-day FadeMem decay + heuristic write-gate classifier)
- Task subsystem (Room-backed CRUD with WorkManager-scheduled reminder notifications)
- Agentic loop (ReAct-style with tool dispatch, abort, conversation state)
- 5 cloud LLM providers (Ollama Cloud, DeepSeek, Anthropic, OpenAI, OpenAI-compat)
- 4-tab UI (Home greeting, Chat with voice+text, Memory browser, Settings)
- Voice input (push-to-talk STT) and output (auto-TTS for agent responses)
- Proactive: WorkManager daily morning brief + 5-minute calendar event monitor
- 35 unit tests pass

## Quick start (sideload on a real device)

### Prerequisites
- Android 8.0+ (API 26+)
- ~100MB free storage
- A cloud LLM API key (Ollama Cloud is free: https://ollama.com/settings/keys)

### Build the APK
```bash
# From D:\Aura\android
./gradlew :app:assembleDebug
# APK lands at: app/build/outputs/apk/debug/app-debug.apk
```

### Install on your phone
```bash
adb install app/build/outputs/apk/debug/app-debug.apk
```
Or transfer the APK to the phone and tap it (enable "Install from unknown sources" in Settings → Apps).

### First-run setup
1. Open Aura.
2. Tap the **Settings** tab.
3. Paste your API key (Ollama Cloud is recommended — it has a generous free tier).
4. Tap the **Chat** tab.
5. Tap the model name in the header → pick a model.
6. Type or tap the mic icon to start talking.

### Permissions the app will request
- **Internet** — for cloud LLMs and web search
- **Microphone** — for voice input (first time only)
- **Location** — for the location_now tool + geofence proactive (v1.5)
- **Calendar** — for calendar_read/write
- **Contacts** — for contacts_search
- **Notifications** — for posting reminder + morning brief
- **Foreground service** — for the morning brief job

## Architecture (TL;DR)

```
┌─────────────────────┐
│ Compose UI (4 tabs) │   Home · Chat · Memory · Settings
└──────────┬──────────┘
           │ StateFlow
┌──────────▼──────────┐
│ ChatViewModel etc.  │   Hilt @HiltViewModel, owns text/TTS streams
└──────────┬──────────┘
           │
┌──────────▼─────────────────────────────────────┐
│ MemoryAugmentedAgenticLoop                      │   ReAct, recall, dispatch, auto-store
└─────┬───────────┬──────────────┬───────────────┘
      │           │              │
┌─────▼─────┐ ┌───▼──────┐  ┌────▼──────────────┐
│Brain      │ │Memory    │  │ToolRegistry      │  21 tools, Hilt singletons
└─────┬─────┘ │(Room+vec)│  └────┬──────────────┘
      │       └──────────┘       │
┌─────▼──────────────────────────▼───────────────┐
│ Provider SDK (5 cloud providers)                  │   Ollama-compat SSE, Anthropic
└─────────────────────────────────────────────────┘
```

## Tool catalog (21)

| Tool | What it does | Risk |
|---|---|---|
| `web_search` | DuckDuckGo HTML search | READ_ONLY |
| `post_notification` | System notification | WRITE_LOCAL |
| `location_now` | Last-known GPS | PRIVACY |
| `share` | Android share sheet | WRITE_LOCAL |
| `calendar_read` | Next N days of events | PRIVACY |
| `calendar_write` | Create event | PRIVACY |
| `contacts_search` | Find contact by name | PRIVACY |
| `set_reminder` | Schedule notification | WRITE_LOCAL |
| `get_current_time` | Local time helper | READ_ONLY |
| `remember` | Store a fact in memory | WRITE_LOCAL |
| `recall` | Search memory | READ_ONLY |
| `launch_app` | Open app or URL | WRITE_LOCAL |
| `system_volume` | Get/set audio streams | WRITE_LOCAL |
| `photo_library` | List recent photos | PRIVACY |
| `biometric_prompt` | Face/fingerprint gate | WRITE_LOCAL |
| `camera_capture` | Open camera | WRITE_LOCAL |
| `battery_state` | Battery level + charging | READ_ONLY |
| `network_state` | Connection type | READ_ONLY |
| `dnd_mode` | Do-Not-Disturb | WRITE_LOCAL |
| `manage_tasks` | Create/list/complete/delete tasks | WRITE_LOCAL |
| `notification_list` | Read active notifications | PRIVACY |

## Build

```bash
./gradlew :app:assembleDebug          # debug APK
./gradlew :app:assembleRelease        # release APK (needs signing)
./gradlew :aura-core:testDebugUnitTest  # unit tests
./gradlew :app:assembleDebug connectedAndroidTest  # androidTests (needs device)
```

## Project layout

```
android/
├── app/                  # :app module — Compose UI, ViewModels, NavGraph
│   ├── src/main/kotlin/com/aura/
│   │   ├── ui/           # screens, components, theme
│   │   ├── di/           # Hilt module (AppModule)
│   │   └── MainActivity, AuraApp
├── aura-core/            # :aura-core library — all logic
│   ├── src/main/kotlin/com/aura/
│   │   ├── agent/        # Brain, AgenticLoop, Conversation
│   │   ├── providers/    # Provider SDK
│   │   ├── memory/       # Room + vector + decay
│   │   ├── tools/        # 21 tools
│   │   ├── tasks/        # Room tasks
│   │   ├── voice/        # STT + TTS
│   │   └── proactive/    # Morning brief + calendar monitor
│   └── src/test/         # 32 unit tests
└── docs/                 # Architecture notes
```

## Source of truth

The plan lives at `.hermes/plans/2026-06-25_161811-aura-android-superapp.md`. Each day's commit has a message documenting what shipped.
