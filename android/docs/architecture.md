# Aura Android — Architecture

## Goal

Native Kotlin/Compose superapp — a full port of the Aura desktop application to Android, targeting a single user with offline-first, privacy-focused design. The Android app mirrors the modular architecture of the backend, adapting UI patterns to Material 3 and Android lifecycle conventions.

## Modules

1. **app** — Main application shell: DI wiring, navigation graph, top-level UI scaffold, and entry point (AuraApp, MainActivity).
2. **aura-core** — Shared data layer: local persistence (Room), network (OkHttp), serialization (kotlinx), encryption (Tink), and DataStore preferences.
3. **aura-assistant** — Conversational AI module: chat UI, streaming SSE bridge to LLM backends, context management, and prompt history.
4. **aura-knowledge** — Semantic knowledge base: vector-store integration (sqlite-vec), document ingestion, embedding pipeline, and full-text search.
5. **aura-memory** — Persistent memory layer: entity extraction, relationship graphs, spaced-repetition recall, and user-profile embeddings.
6. **aura-automation** — Automation engine: trigger-action rules, WorkManager-based cron scheduling, context-aware action suggestions.
7. **aura-files** — File management: storage access framework integration, file indexing, metadata extraction, and local/cloud file picker.
8. **aura-media** — Media pipeline: CameraX integration, audio recording, image gallery, media metadata, and transcoding.
9. **aura-connect** — Device connectivity: Bluetooth LE scanner, Wi-Fi peer discovery, USB accessory bridge, and nearby share.
10. **aura-location** — Location & context: fused location provider, geofencing, place recognition, and activity detection.
11. **aura-calendar** — Calendar & scheduling: CalendarProvider sync, event management, reminders, and natural-language date parsing.
12. **aura-contacts** — Contacts hub: ContactsProvider sync, contact enrichment, groups, and communication history.
13. **aura-sync** — Cross-device sync: encrypted sync protocol, conflict resolution, delta-based sync engine.
14. **aura-settings** — Settings & onboarding: preference screen, permissions manager, theme config, backup/restore, and first-run wizard.

## ASCII Architecture Diagram

```
┌──────────────────────────────────────────────────────────┐
│                     app (shell)                           │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌─────────────┐ │
│  │ assistant │ │knowledge │ │  memory  │ │ automation  │ │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └──────┬──────┘ │
│  ┌────┴─────┐ ┌────┴─────┐ ┌────┴─────┐ ┌──────┴──────┐ │
│  │  files   │ │  media   │ │ connect  │ │  location   │ │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └──────┬──────┘ │
│  ┌────┴─────┐ ┌────┴─────┐ ┌────┴─────┐ ┌──────┴──────┐ │
│  │ calendar │ │ contacts │ │  sync    │ │  settings   │ │
│  └──────────┘ └──────────┘ └──────────┘ └─────────────┘ │
│                           │                               │
│                    ┌──────┴──────┐                        │
│                    │  aura-core  │                        │
│                    │ (data layer)│                        │
│                    └─────────────┘                        │
└──────────────────────────────────────────────────────────┘
```

## Key Design Decisions

- **Single-activity architecture**: One MainActivity, all screens are Compose destinations via Navigation Compose.
- **Hilt DI**: Dependency injection throughout, scoped to component lifetimes.
- **Offline-first**: Room + DataStore are the source of truth; network is a sync layer.
- **Privacy-centric**: All processing stays on-device unless user explicitly enables sync. Biometric gate for sensitive operations.
- **Modular by domain**: Each module has its own package, can be developed/tested independently, and communicates through repository interfaces defined in aura-core.
- **Compose Material 3**: Following Material You design language with dynamic color support.
- **Kotlin serialization**: Used for all structured data (network DTOs, settings, backup payloads).
- **Coroutines + Flow**: Async throughout; StateFlow for UI state, SharedFlow for one-shot events.

## Version

0.1.0

Source of truth: `.hermes/plans/2026-06-25_161811-aura-android-superapp.md`
