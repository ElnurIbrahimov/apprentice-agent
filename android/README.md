# Aura Android

![CI](https://github.com/ElnurIbrahimov/aura-android/actions/workflows/ci.yml/badge.svg)

A native Android superapp — a full port of the Python [Aura](https://github.com/ElnurIbrahimov/apprentice-agent) assistant to Kotlin/Compose. Single-user, offline-first storage, all brains in the cloud.

This is my personal copy. The plan lives at `.hermes/plans/`.

## Status

**v0.10.2** (versionCode 3).

- 36 tools (web search x3, vision, image gen, deep research, firecrawl fetch, knowledge graph, weather, translate, timer, SMS, email, biometric prompt, and phone-native tools)
- Memory stack (Room + 384-dim cloud embeddings + 6-signal RRF retrieval + 14-day FadeMem with access-frequency decay + heuristic WriteGate)
- Knowledge graph (Room-backed, 11 node types, 18 edge types, LLM-extracted per turn)
- Hands (user-defined automation macros, persisted, triggerable by phrase)
- Tasks (Room-backed, manageable in-app and via tool)
- Agentic loop (ReAct-style, 10 steps max, streams text + tool calls, abort-safe)
- 8 LLM providers (Ollama Cloud, Anthropic, OpenAI, DeepSeek, Gemini, Groq, OpenRouter, plus Mixture-of-Agents virtual provider with default preset `glm-5.2` + `kimi-k2.7-code` → `deepseek-v4-pro`)
- 6 specialists (general, coder, researcher, creative, executive, phone-native) with keyword router + tool-allowlist enforcement
- 5-tab UI (Home greeting, Chat with voice+text, Memory browser, Knowledge Graph, Settings) + 4 secondary routes (History, Hands, Tasks, Proactive history)
- Voice I/O (push-to-talk STT via Android SpeechRecognizer, auto-TTS via Android TextToSpeech)
- Proactive: WorkManager daily morning brief (7am local) + 6h memory decay + 5-min calendar monitor (foreground service)
- Share receiver (`text/plain` + `image/*` from Android share sheet)
- User profile (learned from conversations via regex, injected into system prompt)
- Onboarding wizard (paste API key + verify connectivity)
- Biometric gate for sensitive tools
- 512 unit tests passing across `:aura-core` (380) + `:app` (132)
- 9 connected-device tests passing (7 Room migrations + 2 app smoke tests)

Note: the app uses **cloud providers only** — there is no on-device model.

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
2. Follow the onboarding wizard — paste your API key (Ollama Cloud recommended) and tap verify.
3. Tap the **Chat** tab.
4. Tap the model name in the header → pick a model (or pick a specialist chip).
5. Type or tap the mic icon to start talking.

### Permissions the app will request
- **Internet** — for cloud LLMs and web search
- **Microphone** — for voice input (first time only)
- **Location** — for the `location_now` tool
- **Camera** — for `camera_capture` and image input to vision tools
- **Calendar** — for `calendar_read` / `calendar_write`
- **Contacts** — for `contacts_search`
- **Notifications** — for posting reminders + the morning brief
- **Foreground service** — for the calendar monitor
- **Biometric** — for the biometric gate tool
- **Boot completed** — to reschedule proactive workers after reboot

## Architecture (TL;DR)

```
┌──────────────────────────────────────────┐
│ :app  (Compose UI, 5 tabs + 4 routes)    │  Home · Chat · Memory · Graph · Settings · History · Hands · Tasks · Proactive
│   ViewModels (Hilt @HiltViewModel)        │
└──────────┬───────────────────────────────┘
           │ depends on
┌──────────▼──────────────────────────────────────────┐
│ :aura-core  (logic library, Android-lifecycle-light) │
│   MemoryAugmentedAgenticLoop → Brain → ProviderReg │
│   ToolRegistry (38) · ToolExecutor · Specialist     │
│   Memory (Room + RRF + FadeMem) · KnowledgeGraph   │
│   Hands (Room) · Tasks (Room) · Voice (STT/TTS)     │
│   Proactive (MorningBrief + Decay + Calendar)      │
│   Providers: 7 cloud + MoA virtual                  │
│   SecureDataStore for API keys                      │
└────────────────────────────────────────────────────┘
```

The `:aura-core` module has no Compose dependencies. If you ever port to iOS via KMP, this is the layer you'd reuse.

## Tool catalog (38)

### Web & research
| Tool | What it does | Risk |
|---|---|---|
| `web_search` | DuckDuckGo HTML search | READ_ONLY |
| `brave_search` | Brave Search API (needs key) | READ_ONLY |
| `tavily_search` | Tavily Search API (needs key) | READ_ONLY |
| `fetch_url` | Firecrawl fetch (needs key) | READ_ONLY |
| `deep_research` | Multi-source synthesis with citations | READ_ONLY |
| `vision` | Describe an image | READ_ONLY |
| `image_gen` | Generate an image | WRITE_REMOTE |
| `transcribe` | Audio → text | READ_ONLY |


### Knowledge
| Tool | What it does | Risk |
|---|---|---|
| `remember` | Store a fact in memory | WRITE_LOCAL |
| `recall` | Search memory by semantic query | READ_ONLY |
| `knowledge_graph_extract` | Extract entities/edges from text | WRITE_LOCAL |
| `kg_query` | Query the knowledge graph | READ_ONLY |
| `run_hand` | Execute a named hand (automation macro) | WRITE_LOCAL |

### Phone-native
| Tool | What it does | Risk |
|---|---|---|
| `get_current_time` | Local time helper | READ_ONLY |
| `launch_app` | Open app or URL | WRITE_LOCAL |
| `system_volume` | Get/set audio streams | WRITE_LOCAL |
| `battery_state` | Battery level + charging | READ_ONLY |
| `network_state` | Connection type | READ_ONLY |
| `dnd_mode` | Do-Not-Disturb | WRITE_LOCAL |
| `set_reminder` | Schedule a notification reminder | WRITE_LOCAL |
| `manage_tasks` | Create/list/complete/delete tasks | WRITE_LOCAL |
| `post_notification` | System notification | WRITE_LOCAL |
| `notification_list` | Read active notifications | PRIVACY |
| `location_now` | Last-known GPS | PRIVACY |
| `calendar_read` | Next N days of events | PRIVACY |
| `calendar_write` | Create event | PRIVACY |
| `contacts_search` | Find contact by name | PRIVACY |
| `photo_library` | List recent photos | PRIVACY |
| `camera_capture` | Open camera | WRITE_LOCAL |
| `share` | Android share sheet | WRITE_LOCAL |
| `biometric_prompt` | Face/fingerprint gate | WRITE_LOCAL |

## Providers (8 prefixes)

The model picker routes by `prefix:model` string. Keys are stored locally via DataStore and read on every chat call, so Settings changes take effect immediately.

| Prefix | Display name | Notes |
|---|---|---|
| `ollama` | Ollama Cloud | OpenAI-compatible, free tier available |
| `anthropic` | Anthropic | Native Messages API + tool use |
| `openai` | OpenAI | OpenAI-compatible |
| `deepseek` | DeepSeek | OpenAI-compatible |
| `gemini` | Google Gemini | Native streamGenerateContent |
| `groq` | Groq | OpenAI-compatible, fast inference |
| `openrouter` | OpenRouter | OpenAI-compatible, multi-model gateway |
| `moa` | Mixture of Agents | Virtual — runs 2 references in parallel, synthesizes via aggregator |

`moa:default` uses `glm-5.2` + `kimi-k2.7-code` as references and `deepseek-v4-pro` as aggregator. Presets are loaded from app assets; see `MoaPresetRepository`.

## Specialists (6)

Keyword-routed (see `SpecialistRouter`). Each has a system prompt and allowed tool set. Tool allowlists are enforced in the loop — a specialist cannot call a tool not in its allowlist.

- **general** — fallback, all tools
- **coder** — brave/tavily search + fetch_url
- **researcher** — deep_research + brave/tavily
- **creative** — image generation and visual ideation
- **executive** — calendar + contacts + remember/recall
- **phone_native** — all device-state + camera + location tools

## Proactive workers

Scheduled via WorkManager. Re-scheduled on app start (idempotent, UPDATE policy).

- `MorningBriefWorker` — daily at 7am local. Pulls last 10 memories, asks the configured provider for a 3-5 line brief, posts as a notification.
- `DecayWorker` — every 6h. Runs a memory decay pass via `MemoryStore.runDecayPass`.
- `CalendarMonitorService` — 5-min foreground service. Polls upcoming events, surfaces to `ProactiveEventBus`.
- `ReminderWorker` — fires per `set_reminder` request.

## Build

```bash
./gradlew :app:assembleDebug            # debug APK
./gradlew :app:assembleRelease          # release APK (currently signed with debug key — sideload only)
./gradlew :aura-core:testDebugUnitTest  # unit tests (380)
./gradlew :app:testDebugUnitTest        # unit tests (132)
./gradlew :app:assembleDebug connectedAndroidTest  # androidTests (needs device)
```

Stats: 512 unit tests passing across `:aura-core` (380) + `:app` (132).

CI (`.github/workflows/ci.yml`) runs `assembleDebug` + unit tests on every push and PR.

## Project layout

```
android/
├── app/                  # :app module — Compose UI, ViewModels, NavGraph
│   └── src/main/kotlin/com/aura/
│       ├── ui/           # screens, components, theme, viewmodel, voice, settings
│       ├── di/           # Hilt module (AppModule)
│       ├── MainActivity, AuraApp, ShareReceiverActivity, FirstRunGate, UserPreferences
├── aura-core/            # :aura-core library — all logic
│   └── src/main/kotlin/com/aura/
│       ├── agent/        # Brain, MemoryAugmentedAgenticLoop, Conversation, ToolRegistry, Specialist, SpecialistRouter
│       ├── providers/    # 7 cloud providers + MoA virtual + ProviderKeys + ProviderRegistry
│       ├── memory/       # Room + vector + RRF retrieval + FadeMem + WriteGate + Embedder
│       ├── kg/           # Knowledge graph (Room + extractor + repository)
│       ├── hands/        # Automation macros (Room + repository + worker)
│       ├── tasks/        # Task manager (Room)
│       ├── tools/        # 38 tool implementations + ToolsModule
│       ├── voice/        # SpeechToText + TextToSpeech
│       ├── proactive/    # MorningBrief + Decay + CalendarMonitor + ProactiveEventBus
│       ├── profile/      # UserProfile (learned from conversations)
│       ├── security/     # SecureDataStore, KeyManager, BiometricActivityHolder
│       └── data/         # RoomConfig (centralized DB config)
└── docs/                 # ANDROID_TEST_PLAN.md, architecture.md
```

## Tech stack

- Kotlin 1.9.24, AGP 8.2.2, JVM target 17
- Jetpack Compose (BOM 2024.10.01), Material 3, Navigation Compose
- Hilt 2.51 (DI) + Hilt Work (for WorkManager injection)
- Room 2.6.1 (memory, conversations, knowledge graph, hands, tasks, proactive events)
- WorkManager 2.9.1 (proactive workers)
- OkHttp 4.12.0 + okhttp-sse (streaming LLM responses)
- kotlinx-serialization 1.6.3, kotlinx-coroutines 1.9.0
- DataStore Preferences 1.1.1, Biometric 1.2.0-alpha05
- Testing: JUnit 4, MockK, Turbine, Robolectric, kotlinx-coroutines-test
- minSdk 26 (Android 8.0), targetSdk/compileSdk 35

## Source of truth

The build plan lives at `.hermes/plans/2026-07-05-tier-1-polish.md`. Daily commits document what shipped; `git log --oneline` is the changelog.
