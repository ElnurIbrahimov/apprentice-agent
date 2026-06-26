# Aura Android — Architecture

## Goal

Native Kotlin/Compose superapp — a full port of the Aura desktop application to Android, targeting a single user with offline-first, privacy-focused design. The Android app mirrors the modular architecture of the backend, adapting UI patterns to Material 3 and Android lifecycle conventions.

## Modules

The project is a 2-module Gradle build:

1. **app** (`:app`) — Main application shell: Hilt graph, navigation graph, top-level UI scaffold, MainActivity, share-target Activity, settings UI, chat/home/memory screens. This is the user-facing module.
2. **aura-core** (`:aura-core`) — Shared library: agentic loop (`Brain`, `MemoryAugmentedAgenticLoop`), provider SDK (Ollama, Anthropic, OpenAI, DeepSeek via `OllamaCloudProvider` + `AnthropicProvider`), tool registry with 23 phone-native tools, Room-backed memory + tasks, voice I/O (STT + TTS), proactive layer (morning brief + calendar monitor), DataStore preferences for API keys.

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
│  │(Room)  │ │(23 tls)│ │ 4 providers│ │STT+TTS ││
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

## What is NOT in this codebase (despite earlier docs)

These are features that the architecture plan mentioned but were never built. Listing them explicitly so future readers don't go looking:

- No Tink encryption — keys are stored in DataStore in plaintext. Encryption is a v1.5+ task.
- No sqlite-vec — vector search is an in-memory `VectorIndex` over a deterministic SHA-256-projection `Embedder` (384-dim, not a real semantic model). Real embedding is v1.5+.
- No CameraX — `ImageInputTool` opens the system camera via `ACTION_IMAGE_CAPTURE`.
- No Coil — image rendering uses Compose `Image` directly.
- No kotlinx-datetime — `java.util.Calendar` and `SimpleDateFormat` throughout.
- No cross-device sync, no Bluetooth, no USB bridge, no nearby share.
- No document ingestion / FTS pipeline.
- No biometric UI — `BiometricPromptTool` is a `NeedsApproval` marker; the real `BiometricPrompt` API requires a `FragmentActivity` reference, which the agent loop doesn't carry.
- No `aura-assistant` / `aura-knowledge` / `aura-automation` / `aura-files` / `aura-media` / `aura-connect` / `aura-location` / `aura-calendar` / `aura-contacts` / `aura-sync` modules. Two-module build.

## Version

`BuildConfig.VERSION_NAME` from `app/build.gradle.kts` (currently `0.1.0`).

Source of truth: `.hermes/plans/2026-06-25_161811-aura-android-superapp.md`
