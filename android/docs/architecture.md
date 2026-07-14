# Aura Android — Architecture

## Goal

Native Kotlin/Compose superapp — a full port of the Aura desktop application to Android, targeting a single user with offline-first, privacy-focused design. The Android app mirrors the modular architecture of the backend, adapting UI patterns to Material 3 and Android lifecycle conventions.

## Modules

The project is a 2-module Gradle build:

1. **app** (`:app`) — Main application shell: Hilt graph, navigation graph, top-level UI scaffold, MainActivity, share-target Activity, settings UI, chat/home/memory screens. This is the user-facing module.
2. **aura-core** (`:aura-core`) — Shared library: agentic loop (`Brain`, `MemoryAugmentedAgenticLoop`), provider SDK (Anthropic, DeepSeek, Gemini, Groq, Ollama Cloud, OpenAI-compat, OpenRouter + MoA virtual — 8 providers behind a `Provider` interface), tool registry with 36 tools (web search x3, vision, image gen, deep research, firecrawl fetch, knowledge graph, weather, translate, timer, SMS, email, biometric prompt, and phone-native tools), Room-backed memory + tasks, voice I/O (STT + TTS), proactive layer (morning brief + calendar monitor), DataStore preferences for API keys.

This document is a snapshot of the **actual** project state, not aspirational. The earlier version of this file described a 14-module plan that was never implemented; that description is removed.

## ASCII Architecture Diagram

```
┌────────────────────────────────────────────────────┐
│  :app  (Compose UI + Activities)                   │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐          │
│  │  Home   │  │   Chat   │  │  Memory  │  Settings│
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  ...     │
│       └──────────┬─┴──────────┬─┘                 │
│       ┌──────────▼──────────▼──────────┐          │
│       │   ViewModels (Hilt @HiltVM)   │          │
│       └──────────┬────────────────────┘          │
│                  │                                │
│       ┌──────────▼────────────────────┐          │
│       │  IncomingShareStore           │          │
│       └────────────────────────────────┘          │
└────────────────────┬───────────────────────────────┘
                     │
┌────────────────────▼───────────────────────────────┐
│  :aura-core  (logic library)                       │
│  ┌──────────────────────────────────────────────┐ │
│  │   MemoryAugmentedAgenticLoop  +  Brain      │ │
│  └────┬──────────┬────────────┬─────────────┬───┘ │
│       │          │            │             │     │
│  ┌────▼────┐ ┌───▼────┐ ┌─────▼──────┐ ┌───▼────┐│
│  │Memory  │ │Tool    │ │  Provider  │ │ Voice  ││
│  │Store   │ │Registry│ │   SDK      │ │ I/O    ││
│  │(Room)  │ │(31 tls)│ │ 7 providers│ │STT+TTS ││
│  └────────┘ └────────┘ └────────────┘ └────────┘│
│                                                 │
│  ┌──────────────────────────────────────────────┐│
│  │  Proactive layer (morning brief + monitor)  ││
│  │  ProviderKeys (DataStore)                    ││
│  └──────────────────────────────────────────────┘│
└─────────────────────────────────────────────────┘
```

## Key Design Decisions

- **Single-activity architecture**: One `MainActivity`, all screens are Compose destinations via Navigation Compose.
- **Hilt DI**: Dependency injection throughout, scoped to component lifetimes.
- **Cloud-only LLM providers**: User-supplied API keys (Ollama Cloud is free) read live from DataStore via `ProviderKeys`; no on-device model.
- **Offline-first for memory + tasks**: Room is the source of truth; no network needed for any user data layer.
- **Memory decay (FadeMem)**: 14-day half-life, bumped on recall, recomputed on app start.
- **Permission-gated tools**: `ToolExecutor` checks `ContextCompat.checkSelfPermission` against `Tool.requiredPermissions` at execution time, so a permission that was just granted is honored on the next call without restart.
- **Privacy-centric**: All processing stays on-device; cloud LLMs are an opt-in dependency. No cross-device sync, no analytics, no telemetry.
- **Compose Material 3**: Following Material You design language.
- **Coroutines + Flow**: Async throughout; StateFlow for UI state, SharedFlow for one-shot events.

## What is NOT in this codebase (deliberate non-goals / later versions)

These are features that earlier architecture plans mentioned but were never built. Listing them explicitly so future readers don't go looking:

- No cross-device sync, no Bluetooth, no USB bridge, no nearby share.
- No document ingestion / FTS pipeline.
- No `aura-assistant` / `aura-knowledge` / `aura-automation` / `aura-files` / `aura-media` / `aura-connect` / `aura-location` / `aura-calendar` / `aura-contacts` / `aura-sync` modules. Two-module build.

## Security notes

- API keys live in DataStore and are read at call time; no keys are compiled into the app.
- `SecureDataStore` encrypts sensitive values with AES-256-GCM via Android Keystore (shipped; v1.5+ note in old docs was stale).
- `KeyManager` generates/retrieves the Keystore-backed key lazily and surfaces a clear error on decryption failures rather than silently returning `null`.
- The agent loop's `ToolExecutor` blocks `WRITE_LOCAL` tools when `memoryEnabled=false`, keeping incognito sessions from persisting data.
- Backups intentionally exclude embeddings (model-specific) and API keys (security).
- Backup export files are plaintext JSON containing all conversations, memories, tasks, hands, profile, and preferences. They do NOT include API keys or embeddings, but the exported data is still sensitive (personal conversations, memory contents, profile facts). **Keep backup files private — do not share or commit them.** Store them in app-private storage or transfer directly to a secure location.

## Privacy notes

- All user data (memories, tasks, conversations, profile) stays in local Room/SQLite.
- Cloud LLMs are opt-in and user-keyed; the app does not ship with bundled providers or analytics.

## Version

`BuildConfig.VERSION_NAME` from `app/build.gradle.kts` (currently `0.10.2`, versionCode 3).

Source of truth: `.hermes/plans/2026-07-14-audit-remediation.md` (complete; 15 prior plans archived in git history).
