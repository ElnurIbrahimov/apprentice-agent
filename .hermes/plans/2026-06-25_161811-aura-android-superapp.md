# Aura Android — Native Superapp Port

**Goal:** A native Android app (Kotlin/Compose) that runs Aura's full capability surface on a phone, for personal daily use. Sideload v1, Play Store v3.

**Plan file:** `.hermes/plans/2026-06-25_161811-aura-android-superapp.md`
**Repo location:** `D:\Aura\android\` (new Gradle project, sibling to `aura/`)
**Persona:** Single user (Elnur). No auth, no billing, no Play Store constraints in v1.
**Total Kotlin estimate:** ~38K LOC across 13 modules.
**Build cadence:** AI-delegated, parallel subagents. v1 target: 10-15 working days of focused sessions. Polish + safety: 30-45 days.

---

## 0. Prior plan check

Existing plans in `.hermes/plans/`:
- `2026-05-18_block_output_agents_md.md` — not relevant
- `2026-06-21-hermes-feature-parity.md` — feature-parity CLI work, not relevant
- `2026-06-22-cli-polish-round-2.md` — CLI polish, not relevant

No prior Android plan. This is the first.

---

## 1. The non-decisions (locked at the top)

- **No server.** Everything on-device. Cloud APIs called directly from the app via OkHttp. No backend infra, no Tailscale, no laptop dependency.
- **Full capability port.** The phone is a *superset* of Aura's laptop surface, not a subset. Sensors, voice, presence, push, contacts, calendar, location, share, accessibility are all in. Shell, git, file system are routed to a cloud sandbox on demand.
- **All 17 providers target, 4 wired in v1.** Ollama Cloud, DeepSeek, Anthropic, OpenAI. The rest slot into the same `Provider` interface.
- **No auth in v1.** No accounts, no billing, no Play Store. Sideload the APK.
- **On-device model as default tier.** User downloads on first launch. Optional. Deletable. ~3GB.

---

## 2. Architecture overview

```
┌────────────────────────────────────────────────────────────┐
│  UI (Jetpack Compose)                                       │
│    Chat screen · Memory browser · Tools · Automations       │
│    Specialists chip row · Status strip · Voice I/O · Camera  │
└──────────────────┬─────────────────────────────────────────┘
                   │
┌──────────────────▼─────────────────────────────────────────┐
│  Agent Runtime (Kotlin)                                     │
│    ReAct loop · Plan mode · Sub-agent spawning              │
│    Tool dispatch (with permission + signing + taint)        │
│    Conversation forking · Context compression               │
│    Event log · Verification stage                           │
└────┬───────────┬───────────┬──────────────┬────────────────┘
     │           │           │              │
     ▼           ▼           ▼              ▼
┌────────┐  ┌────────┐  ┌─────────┐  ┌──────────────┐
│Provider│  │Memory  │  │ Tools   │  │ Multi-agent  │
│ SDK    │  │ Stack  │  │ Registry│  │ Orchestrator │
│        │  │        │  │         │  │              │
│Local   │  │Room +  │  │80+ tools│  │ Specialists  │
│llama   │  │sqlite- │  │phone +  │  │ + protocol   │
│Ollama  │  │vec     │  │cloud    │  │              │
│Cloud   │  │FTS5    │  │sandbox  │  │              │
│Anthrop │  │FadeMem │  │         │  │              │
│OpenAI  │  │Profile │  │         │  │              │
└────────┘  └────────┘  └─────────┘  └──────────────┘
     │           │           │              │
     └───────────┴───────────┴──────────────┘
                   │
┌──────────────────▼─────────────────────────────────────────┐
│  Proactive Layer (Android primitives)                       │
│    ForegroundService · WorkManager · NotificationManager    │
│    Quick Settings tile · App shortcut · Home widget         │
│    AccessibilityService · NotificationListenerService       │
└────────────────────────────────────────────────────────────┘
```

Three laws:
1. Everything async, no blocking the main thread.
2. Every tool call is permission-checked, signed, taint-tracked, audited.
3. Every UI screen has a one-tap path back to chat.

---

## 3. Module breakdown (file-level)

### Module 0: Project skeleton — `android/`

| Path | Purpose | LOC est |
|---|---|---|
| `android/build.gradle.kts` | Gradle root | 60 |
| `android/settings.gradle.kts` | Settings + module list | 40 |
| `android/gradle/libs.versions.toml` | Version catalog | 80 |
| `android/app/build.gradle.kts` | App module config | 200 |
| `android/app/src/main/AndroidManifest.xml` | Permissions + services | 250 |
| `android/app/proguard-rules.pro` | R8 rules | 80 |
| `android/app/src/main/res/...` | Icons, theme, strings | 200 |
| `docs/architecture.md` | Living architecture doc | 400 |

**AndroidX deps:** Compose BOM, Material 3, Room, sqlite-vec, WorkManager, DataStore, Hilt, OkHttp, kotlinx-serialization, kotlinx-coroutines, kotlinx-datetime, Coil, CameraX, MediaPipe (for on-device multimodal fallback), Whisper.cpp Android binding, llama.cpp Android binding, Porcupine wake-word, Silero VAD ONNX, Piper TTS, Navigation Compose, Accompanist permissions, Biometric, SecurityCrypto, Tink.

**Minimum SDK:** 26 (Android 8.0). **Target SDK:** 35. **Compile SDK:** 35.

**Permissions (requested lazily, justified in UI):**
- `INTERNET`, `ACCESS_NETWORK_STATE`
- `RECORD_AUDIO` (voice input)
- `CAMERA` (vision)
- `ACCESS_FINE_LOCATION`, `ACCESS_COARSE_LOCATION` (proactive context)
- `ACCESS_BACKGROUND_LOCATION` (only if user opts into "ambient mode")
- `READ_CALENDAR`, `WRITE_CALENDAR`
- `READ_CONTACTS`
- `READ_SMS` (opt-in, rare)
- `READ_CALL_LOG` (opt-in, rare)
- `POST_NOTIFICATIONS` (Android 13+)
- `FOREGROUND_SERVICE`, `FOREGROUND_SERVICE_DATA_SYNC`, `FOREGROUND_SERVICE_MICROPHONE`
- `RECEIVE_BOOT_COMPLETED` (resume proactive service)
- `SYSTEM_ALERT_WINDOW` (overlay quick-reply, opt-in)
- `BIND_ACCESSIBILITY_SERVICE` (only if user opts in for screen reading)
- `BIND_NOTIFICATION_LISTENER_SERVICE` (only if user opts in)

**Verification:**
- `./gradlew :app:assembleDebug` builds clean APK
- App installs and launches on emulator + a real device (Elnur's phone)

---

### Module 1: Provider SDK — `android/aura-core/src/main/kotlin/com/aura/providers/`

Port of `aura/providers/` (1.5K LOC Python) → ~3K Kotlin.

| File | Purpose | LOC |
|---|---|---|
| `Provider.kt` | `interface Provider` — mirrors `aura/providers/base.py`. `chat(model, messages, stream, tools, options)`, `listModels()`, `isConfigured()`, `displayName`, `prefix`. | 80 |
| `ProviderMessage.kt` | Message types (system/user/assistant/tool) with kotlinx-serialization. | 120 |
| `ProviderChunk.kt` | Streaming chunk type (text delta, tool_call delta, done, error). | 80 |
| `ToolCall.kt` | OpenAI-compatible tool call schema (works across providers). | 150 |
| `ProviderRegistry.kt` | Holds all providers, routes by `provider:model` prefix. Mirrors `aura/providers/registry.py`. | 250 |
| `CredentialPool.kt` | Encrypted credential storage (DataStore + Android Keystore). Per-provider key. | 280 |
| `LocalProvider.kt` | llama.cpp Android binding. Loads GGUF model from internal storage. Implements `Provider` with tool-call parsing from text. | 600 |
| `OllamaCloudProvider.kt` | Calls `https://ollama.com/v1/chat/completions` (OpenAI-compatible). | 250 |
| `AnthropicProvider.kt` | Anthropic Messages API with streaming and tool_use blocks. | 350 |
| `OpenAIProvider.kt` | OpenAI Chat Completions with tool_calls. | 280 |
| `DeepSeekProvider.kt` | OpenAI-compatible endpoint at `api.deepseek.com`. | 200 |
| `GeminiProvider.kt` | Gemini via `generativelanguage.googleapis.com` (for multimodal). | 280 |
| `ProviderRouter.kt` | Task-aware routing (mirrors `aura/core/router.py` 658 LOC + `aura/routing/` 1054 LOC). Local for simple, cloud for hard, cost-aware. | 600 |
| `ProviderFactory.kt` | DI factory. | 100 |

**Interfaces with:** agent runtime, chat UI, settings.

**Verification:**
- Unit tests: each provider against its real API with recorded fixtures.
- Integration test: streaming chunk assembly for each provider.
- Local model: loads a 1.5B GGUF, runs inference, asserts token output.

---

### Module 2: Brain + Agent Runtime — `android/aura-core/src/main/kotlin/com/aura/agent/`

Port of `aura/brain.py` (2014 LOC) + `aura/agent.py` (1498 LOC) + `aura/core/agentic_loop*.py` (4289 LOC across 6 files). Target: ~6K Kotlin.

| File | Purpose | LOC |
|---|---|---|
| `Brain.kt` | Brain singleton. Wraps `Provider`. Handles system prompt assembly, context budget, conversation state. Mirrors `aura/brain.py` minus the desktop-specific prompt fragments. | 800 |
| `Agent.kt` | Top-level agent orchestrator. Mirrors `aura/agent.py` ApprenticeAgent — ReAct loop, adaptive planning, session persistence, mode dispatch (chat, plan, debate, research, chain, code). | 700 |
| `AgenticLoop.kt` | Core loop. Mirrors `aura/core/agentic_loop.py` (the main 1603-LOC file). ReAct step, tool dispatch, verification stage, error recovery, abort/cancel. | 900 |
| `AgenticLoopEvents.kt` | Event types (model_start, model_chunk, tool_call_start, tool_call_result, model_resume, finished, error, aborted). Maps to `aura/core/agentic_loop_events.py`. | 150 |
| `AgenticLoopToolCalls.kt` | Tool call parsing, JSON schema validation, parallel tool call dispatch. Maps to `aura/core/agentic_loop_tool_calls.py` (290 LOC). | 400 |
| `AgenticLoopOutcomes.kt` | Success/failure/retry decision logic. Maps to `aura/core/agentic_loop_outcomes.py`. | 250 |
| `AgenticLoopModelStep.kt` | The "call the model, get a response" step including streaming assembly. Maps to `aura/core/agentic_loop_model_step.py`. | 350 |
| `ConversationManager.kt` | Multi-conversation persistence, fork, summary, search. Maps to `aura/core/conversation_manager.py` + `conversation_mixin.py` + `conversation_fork.py`. | 700 |
| `ContextCompressor.kt` | Sliding window, summarization, token budget enforcement. Maps to `aura/memory/context_compressor.py` (278 LOC) + `aura/core/token_manager.py`. | 400 |
| `PlanMode.kt` | Plan generation, plan approval, plan execution. Maps to `aura/core/planner.py` + `aura/core/adaptive_planner.py`. | 400 |
| `SubAgent.kt` | Sub-agent spawning and lifecycle. Maps to `aura/core/sub_agent.py`. | 300 |
| `ToolExecutor.kt` | Dispatches tool calls. Wraps ToolRegistry. Permission check, signing, taint, audit. | 350 |
| `JsonUtils.kt` | Robust JSON extraction from model output (handles ```json fences, partial JSON, etc.). | 200 |
| `EventLog.kt` | Persistent event log for debugging and replay. Maps to `aura/core/event_log.py`. | 250 |
| `VerificationStage.kt` | Post-tool self-check (did the tool call actually solve the user's intent?). Maps to `aura/core/verification_stage.py`. | 200 |

**Verification:**
- Unit tests: agent loop with mock provider, plan mode happy path, tool call error recovery, abort.
- Integration: end-to-end "user says X → 3 tool calls → response" with a stub provider.
- "Rover test" (Aura's naming): agent handles a multi-step task with at least 2 sub-agents and 4 tool calls without losing state.

---

### Module 3: Memory Stack — `android/aura-core/src/main/kotlin/com/aura/memory/`

Port of `aura/memory/` (4417 LOC Python) → ~3K Kotlin.

| File | Purpose | LOC |
|---|---|---|
| `MemoryStore.kt` | Room database. Tables: memories, embeddings, user_profile, observations, kg_edges. Maps to `aura/memory/store.py` (1162 LOC). | 700 |
| `UnifiedMemory.kt` | Single query interface. BM25 + vector + recency + importance + emotion. RRF fusion. Maps to `aura/memory/unified_memory.py` (658 LOC). | 500 |
| `Embedder.kt` | Embedding generation. Local: nomic-embed via ONNX Runtime for Android (~80MB) OR remote via Ollama Cloud. Maps to `aura/memory/embedding.py`. | 300 |
| `VectorIndex.kt` | sqlite-vec backed vector search. | 250 |
| `FadeMem.kt` | Decay computation (2-week half-life). Maps to `aura/memory/fade_mem.py` (145 LOC). | 200 |
| `WriteGate.kt` | Decides what gets stored. Maps to `aura/memory/write_gate.py` (518 LOC). | 300 |
| `ObservationMasker.kt` | Strips sensitive observations before write. Maps to `aura/memory/observation_masker.py` (340 LOC). | 200 |
| `UserProfile.kt` | User identity, preferences, history. Maps to `aura/memory/user_profile.py` (246 LOC). | 200 |
| `KnowledgeGraph.kt` | Lightweight KG using ObjectBox or a custom SQLite-backed impl (Kuzu has no Android build). Maps to `aura/core/kg_brain.py` + `kg_sync.py` + `aura/memory/kg_contradiction.py`. | 350 |
| `Retrieval.kt` | Retrieval pipeline: query → embed → BM25 + vector + KG + RRF → rerank → fade weighting → top-k. Maps to `aura/memory/retrieval.py` (399 LOC). | 400 |
| `ContextBudget.kt` | Token budget for memory chunks. | 100 |

**Verification:**
- Unit tests: store 1000 memories, query, assert ranking, assert decay over simulated time.
- Integration: write gate, observation masker, profile updates.
- Fuzz: malformed memory text, dedup, large queries.

---

### Module 4: Tool System — `android/aura-core/src/main/kotlin/com/aura/tools/`

Port of `aura/tools/` (84 files) + phone-native additions → ~6K Kotlin.

Port these from `aura/tools/` (mapping desktop → phone/cloud):
- `web_search.kt` (port from `brave_search.py` / `web_search.py` / `tavily_tool.py`) — 300
- `deep_research.kt` (port from `deep_research.py` + `search_planner.py`) — 400
- `pdf_reader.kt` (port from `pdf_reader.py`) — 250
- `image_gen.kt` (port from `image_gen.py`) — 250
- `audio_transcriber.kt` (port from `audio_transcriber.py`) — uses whisper.cpp Android — 300
- `vision.kt` (port from `vision.py`) — uses CameraX + multimodal model — 400
- `knowledge_graph.kt` (port from `knowledge_graph.py`) — 200
- `task_manager.kt` (port from `task_manager.py`) — uses Room + WorkManager — 350
- `task_scheduler.kt` (port from `task_scheduler.py`) — uses WorkManager + AlarmManager — 300
- `notifications.kt` (port from `notifications.py`) — uses NotificationManager — 200
- `spaced_repetition.kt` (port from `spaced_repetition.py`) — 250
- `code_executor_cloud.kt` (REPLACES desktop `code_executor.py`) — runs in cloud sandbox via API call — 250
- `shell_cloud.kt` (REPLACES `shell_executor.py`) — cloud sandbox — 200
- `web_fetch.kt` (port from `firecrawl_tool.py`) — 250
- `arxiv_search.kt` (port from `arxiv_search.py`) — 200
- `crypto_price.kt` (port from `crypto_price.py`) — 150

Phone-native (NEW, not in Aura):
- `voice_input.kt` — mic + Silero VAD + whisper.cpp STT — 400
- `voice_output.kt` — Piper TTS or system TTS — 200
- `camera_capture.kt` — CameraX + image return — 200
- `location_now.kt` — FusedLocationProvider — 150
- `location_history.kt` — last-N locations — 150
- `motion_state.kt` — ActivityRecognitionClient — 150
- `battery_state.kt` — BatteryManager — 100
- `network_state.kt` — ConnectivityManager — 100
- `screen_state.kt` — Display + power — 100
- `dnd_mode.kt` — NotificationManager + policy access — 150
- `calendar_read.kt` / `calendar_write.kt` — CalendarContract — 300
- `contacts_search.kt` / `contacts_get.kt` — ContactsContract — 300
- `notification_post.kt` — NotificationManager + channel — 150
- `share_intent.kt` — Intent.ACTION_SEND + chooser — 150
- `app_launcher.kt` — startActivity by package — 150
- `app_shortcuts.kt` — ShortcutManager — 150
- `media_control.kt` — MediaSession + transport controls — 250
- `photo_library.kt` — MediaStore — 200
- `file_pick.kt` — Storage Access Framework — 150
- `biometric_prompt.kt` — BiometricPrompt — 150
- `nfc_read.kt` — NfcAdapter — 200
- `bluetooth_scan.kt` — BluetoothLeScanner — 250
- `system_volume.kt` / `system_brightness.kt` — AudioManager + system settings — 150
- `clipboard_read.kt` / `clipboard_write.kt` — ClipboardManager — 100
- `foreground_service.kt` / `workmanager_schedule.kt` — service/job control — 200
- `accessibility_tree.kt` — AccessibilityService — 350 (opt-in)
- `screenshot.kt` — MediaProjection (opt-in) — 350
- `app_usage_stats.kt` — UsageStatsManager — 200
- `quick_settings_tile.kt` — TileService — 200
- `widget_update.kt` — AppWidgetProvider — 200

Tool infrastructure:
- `Tool.kt` — interface, with name, description, schema, risk level, required permissions
- `ToolRegistry.kt` — registration + lookup
- `ToolContext.kt` — execution context (caller, permissions, taint, audit chain)
- `ToolResult.kt` — sealed result type
- `ToolPermission.kt` — gating (with biometric for high-risk)
- `ToolSignature.kt` — HMAC signing of tool calls
- `ToolTaint.kt` — taint tracking (which tools can call which other tools)
- `ToolAudit.kt` — append-only audit log

**Verification:**
- Each tool: unit test with mock Android service + integration test on device.
- Permission tests: tool refuses without permission, succeeds with.
- Taint tests: taint propagates, blocks unsafe chains.
- Audit: every call logged, log survives app restart.

---

### Module 5: Multi-agent Orchestrator + Specialists — `android/aura-core/src/main/kotlin/com/aura/multi_agent/`

Port of `aura/multi_agent/` (5 files) + `aura/multi_agent/specialists/` (5 files) → ~2K Kotlin.

| File | Purpose | LOC |
|---|---|---|
| `Orchestrator.kt` | Top-level multi-agent orchestrator. Picks specialist, routes messages, manages sub-agents, merges results. Maps to `aura/multi_agent/orchestrator.py` + `router.py`. | 500 |
| `BaseAgent.kt` | Base class for all agents. Maps to `aura/multi_agent/base_agent.py`. | 250 |
| `Protocol.kt` | `AgentMessage`, `AgentResult`, `CollaborationMode` (SINGLE/SEQUENTIAL/PARALLEL/DEBATE). Direct port of `aura/multi_agent/protocol.py` (141 LOC). | 200 |
| `Router.kt` | Specialist selection by message + context. Maps to `aura/multi_agent/router.py` + `aura/routing/`. | 200 |
| `Specialist.kt` | `data class Specialist(name, description, systemPrompt, tools, model, triggers, maxToolCalls)`. | 150 |
| `specialists/General.kt` | Default chat specialist. | 100 |
| `specialists/Researcher.kt` | Port of `aura/multi_agent/specialists/research.py`. | 200 |
| `specialists/Coder.kt` | Port of `aura/multi_agent/specialists/coder.py`. | 200 |
| `specialists/Analyst.kt` | Port of `aura/multi_agent/specialists/analyst.py`. | 150 |
| `specialists/Creative.kt` | Port of `aura/multi_agent/specialists/creative.py`. | 150 |
| `specialists/Searcher.kt` | Port of `aura/multi_agent/specialists/searcher.py`. | 150 |
| `specialists/DailyBriefing.kt` | NEW — generates morning brief from calendar + memory + weather. | 200 |
| `specialists/OnTheGo.kt` | NEW — voice-first, location-aware, fast model. | 200 |
| `specialists/PhoneNative.kt` | NEW — full phone-primitive access (camera, contacts, calendar, etc.). | 200 |
| `specialists/CodeDelegator.kt` | NEW — runs code in cloud sandbox. | 200 |

**Verification:**
- Each specialist: unit test with mock provider + tool registry.
- Orchestrator: SINGLE, SEQUENTIAL, PARALLEL, DEBATE modes verified.
- Specialist switching: state isolated, memory preserved.

---

### Module 6: Hands System — `android/aura-core/src/main/kotlin/com/aura/hands/`

Port of `aura/hands/` (10 files, 2632 LOC) → ~2K Kotlin.

| File | Purpose | LOC |
|---|---|---|
| `HandManager.kt` | Lifecycle for all hands. Maps to `aura/hands/manager.py` (808 LOC). | 400 |
| `Hand.kt` | Base class. Maps to `aura/hands/base.py` (220 LOC). | 150 |
| `HandCollector.kt` | Collects data for hands. Maps to `aura/hands/collector.py` (303 LOC). | 250 |
| `HandGuardian.kt` | Safety / approval gating. Maps to `aura/hands/guardian.py` (188 LOC). | 200 |
| `MemoryHand.kt` | Auto-extracts memories from conversation. Maps to `aura/hands/memory_hand.py`. | 200 |
| `ResearcherHand.kt` | Long-running research tasks. Maps to `aura/hands/researcher.py` (276 LOC). | 250 |
| `MorningBriefingHand.kt` | Daily summary at 7am. Maps to `aura/hands/morning_briefing.py` (123 LOC). | 200 |
| `DynamicHand.kt` | User-defined hands. Maps to `aura/hands/dynamic_hand.py` (231 LOC). | 200 |
| `CustomStore.kt` | Storage for user hands. Maps to `aura/hands/custom_store.py` (121 LOC). | 100 |
| `Templates.kt` | Hand templates. Maps to `aura/hands/templates.py` (159 LOC). | 150 |

**Verification:**
- Each hand: integration test that runs end-to-end.
- Guardian: blocks unsafe hands, allows with approval.
- Morning briefing: fires at 7am, generates valid summary, posts notification.

---

### Module 7: Proactive / Consciousness — `android/aura-core/src/main/kotlin/com/aura/proactive/`

Port of `aura/consciousness/` selectively. Skip `intrinsic_motivation`, `self_improvement`, `strategy_bandit`, `reasoning_templates` (research code). Keep and port: `proactive_awareness`, `idle_presence`, `world_model` (slimmed), `reward_signals`, `state_extractor`. Add `aura/proactive/` (monitors, gateway, event bus, salience filter) → ~3K Kotlin.

| File | Purpose | LOC |
|---|---|---|
| `ProactiveEngine.kt` | Orchestrates all proactive monitors. Maps to `aura/consciousness/proactive_awareness.py` (845 LOC). | 500 |
| `Monitor.kt` | `interface Monitor` — emits signals when something is worth surfacing. | 100 |
| `Monitors/CalendarMonitor.kt` | Upcoming events worth mentioning. | 200 |
| `Monitors/LocationMonitor.kt` | Arrived at a place → recall related memory. | 200 |
| `Monitors/MotionMonitor.kt` | Started driving / walking → switch mode. | 150 |
| `Monitors/NotificationMonitor.kt` | NotificationListenerService → "interesting" notifications. | 300 |
| `Monitors/BatteryMonitor.kt` | Low battery + plugged in → suggest charging routine. | 100 |
| `Monitors/TimeOfDayMonitor.kt` | Morning, evening transitions. | 150 |
| `Monitors/ContactMonitor.kt` | Called someone recently → log interaction. | 150 |
| `Monitors/AppUsageMonitor.kt` | Long session in one app → break suggestion. | 150 |
| `EventBus.kt` | Cross-module pub/sub. Maps to `aura/proactive/event_bus.py`. | 200 |
| `SalienceFilter.kt` | Decides what to surface. Maps to `aura/proactive/salience_filter.py`. | 300 |
| `ProactiveMessage.kt` | A surfaced signal with payload. Maps to `aura/proactive/proactive_messages.py`. | 100 |
| `WorldModel.kt` | Slimmed world model (location, motion, time, calendar, contacts, do-not-disturb, charging, network). Maps to `aura/consciousness/world_model.py` (2358 LOC) — take 30%, leave 70% on the cutting room floor. | 500 |
| `IdlePresence.kt` | "Assistant is in your pocket" — passive state tracking. Maps to `aura/consciousness/idle_presence.py` (799 LOC). | 250 |
| `StateExtractor.kt` | Extract structured state from conversation. Maps to `aura/consciousness/state_extractor.py` (294 LOC). | 200 |
| `RewardSignals.kt` | User reactions (thumbs up/down, "wrong", "good") → reward signal. Maps to `aura/consciousness/reward_signals.py` (322 LOC). | 200 |
| `GatewayDaemon.kt` | Android foreground service that holds proactive state alive. Maps to `aura/proactive/gateway_daemon.py`. | 200 |
| `Persistence.kt` | Proactive state survives restart. Maps to `aura/proactive/persistence.py`. | 100 |
| `TheoryOfMind.kt` | Lightweight user model. Maps to `aura/proactive/theory_of_mind.py`. | 200 |
| `CuriosityScanner.kt` | "What would the user want to know right now?" Maps to `aura/proactive/curiosity_scanner.py`. | 200 |
| `ActiveInference.kt` | Decision-theoretic selection of what to do. Maps to `aura/proactive/active_inference.py`. | 200 |
| `MotivationAccumulator.kt` | Drives proactive behavior. Maps to `aura/proactive/motivation_accumulator.py`. | 150 |
| `SkillHealthMonitor.kt` | Detect when a tool starts failing. Maps to `aura/proactive/skill_health_monitor.py`. | 200 |

**Verification:**
- Each monitor: unit test with mock signals.
- End-to-end: simulated day with calendar events, location changes, notifications → proactive messages surface at right times.

---

### Module 8: Emotion / ALMA — `android/aura-core/src/main/kotlin/com/aura/emotion/`

Port of `aura/emotion/` (5 files, 2779 LOC) → ~1.5K Kotlin. Keep the PAD model, drop the neuromodulator math.

| File | Purpose | LOC |
|---|---|---|
| `AlmaEngine.kt` | PAD (Pleasure-Arousal-Dominance) state. Computed from signals + user input. Maps to `aura/emotion/alma_engine.py` (1519 LOC). | 500 |
| `ActionBridge.kt` | PAD state influences response tone. Maps to `aura/emotion/action_bridge.py`. | 200 |
| `Integration.kt` | Wires emotion into brain, memory, proactive. Maps to `aura/emotion/integration.py` (658 LOC). | 400 |
| `MemoryTagging.kt` | Tag memories with PAD at write time. Maps to `aura/emotion/memory_tagging.py`. | 200 |
| `TemporalGrounding.kt` | Emotion has a decay timeline. Maps to `aura/emotion/temporal_grounding.py`. | 200 |

**Verification:**
- PAD state updates correctly from signals.
- Memory retrieval respects emotional congruence.
- Tone shifts based on PAD are visible in chat output.

---

### Module 9: Security — `android/aura-core/src/main/kotlin/com/aura/security/`

Port of `aura/security/` + harden for Android threat model. ~2K Kotlin.

| File | Purpose | LOC |
|---|---|---|
| `ToolSigning.kt` | HMAC sign every tool call. Verify before execution. Maps to `aura/security/tool_signing.py`. | 200 |
| `TaintTracker.kt` | Track tainted data through tool calls. Maps to `aura/security/taint_tracker.py`. | 300 |
| `SsrfGuard.kt` | Block SSRF on outbound HTTP. Maps to `aura/security/ssrf_guard.py`. | 250 |
| `ToolValidator.kt` | Schema validation, range checks. Maps to `aura/security/tool_validator.py`. | 200 |
| `AuditChain.kt` | Append-only hash-chained audit log. Maps to `aura/security/audit_chain.py`. | 300 |
| `SecretStore.kt` | Encrypted credential storage. Tink + Android Keystore. | 300 |
| `PermissionGate.kt` | Wraps Android permissions with Aura's risk levels. | 200 |
| `BiometricUnlock.kt` | Biometric gate for high-risk operations. | 200 |
| `ApprovalDialog.kt` | UI for tool approval (Compose). | 250 |
| `Redaction.kt` | Redact secrets in logs. | 150 |

**Verification:**
- Audit chain detects tampering.
- Taint blocks tool calls that touch tainted data.
- Biometric prompt blocks without unlock.
- SSRF guard blocks known-bad hosts.

---

### Module 10: Channels (bridge to other surfaces) — `android/aura-core/src/main/kotlin/com/aura/channels/`

Port of `aura/channels/bridge.py` + `channel_bridge.py` → ~1K Kotlin. The phone IS a channel. The bridge becomes a way to delegate to external channels (Telegram, Slack).

| File | Purpose | LOC |
|---|---|---|
| `ChannelBridge.kt` | Routing messages to/from external channels. Maps to `aura/channels/bridge.py` + `channel_bridge.py`. | 300 |
| `PhoneChannel.kt` | The phone's own surface — chat UI, notifications, voice. | 200 |
| `TelegramChannel.kt` | Optional: route to Telegram bot. Port of `aura/channels/telegram_channel.py`. | 250 |
| `ExtensionChannel.kt` | Optional: route to browser extension. Port of `aura/channels/extension_channel.py`. | 200 |
| `Display.kt` | Themed output (icons, colors, blocks). Port of `aura/channels/display.py`. | 200 |

---

### Module 11: UI Surface (Compose) — `android/app/src/main/kotlin/com/aura/ui/`

Compose UI for all surfaces. ~6K Kotlin.

| File | Purpose | LOC |
|---|---|---|
| `MainActivity.kt` | Single-activity host. Edge-to-edge. | 150 |
| `AuraApp.kt` | App-level composable. Theme, navigation, system bars. | 200 |
| `nav/NavGraph.kt` | Navigation Compose setup. | 200 |
| `screens/ChatScreen.kt` | The chat surface. Streaming, tool cards, voice, image, screenshot. | 800 |
| `screens/ChatHeader.kt` | Model picker, specialist chips, status. | 300 |
| `screens/MessageList.kt` | Streaming message renderer with blocks. | 500 |
| `screens/MessageBubble.kt` | Single message rendering (text, code, image, tool call, tool result). | 500 |
| `screens/ToolCallCard.kt` | Expandable card for tool calls + results. | 300 |
| `screens/InputBar.kt` | Text + voice + attachment input. | 400 |
| `screens/VoiceOverlay.kt` | Full-screen voice input with VAD waveform. | 300 |
| `screens/MemoryScreen.kt` | Memory browser. List, search, filter, view, edit, forget. | 600 |
| `screens/MemoryDetail.kt` | Single memory view. | 300 |
| `screens/ToolsScreen.kt` | Tool registry browser. Toggle tools, see schema, test. | 400 |
| `screens/AutomationsScreen.kt` | Hands, monitors, scheduled tasks. | 500 |
| `screens/AutomationsDetail.kt` | Single automation config. | 300 |
| `screens/SpecialistsScreen.kt` | Specialist picker + config. | 300 |
| `screens/SettingsScreen.kt` | All settings: providers, models, permissions, voice, theme, security, account, about. | 800 |
| `screens/ProviderConfig.kt` | Per-provider setup (API key, model preference, fallback). | 350 |
| `screens/ModelPicker.kt` | Model picker with model info, context, pricing. | 300 |
| `screens/CommandPalette.kt` | ⌘K-style launcher. | 300 |
| `screens/ApprovalDialog.kt` | Tool approval UI (Compose). | 250 |
| `screens/StatusStrip.kt` | Top status (active model, context %, mood, online/offline). | 200 |
| `screens/NotificationCenter.kt` | Proactive message inbox. | 300 |
| `screens/MoodIndicator.kt` | PAD visualization. | 200 |
| `components/StreamingText.kt` | Animated text streaming. | 200 |
| `components/CodeBlock.kt` | Syntax-highlighted code. | 250 |
| `components/ImagePreview.kt` | Image with viewer. | 200 |
| `components/ModelBadge.kt` | Provider:model badge. | 100 |
| `components/ProviderIcon.kt` | Provider icon set. | 100 |
| `theme/Theme.kt` | Material 3 theme. Light/dark. | 200 |
| `theme/Color.kt` | Color palette (port of Aura's `themes.py` 315 LOC). | 200 |
| `theme/Type.kt` | Typography. | 150 |
| `theme/Shapes.kt` | Shapes. | 100 |

**Verification:**
- Compose preview tests for every screen.
- UI tests: navigate, send message, see response, switch specialist, change model.
- Accessibility: TalkBack labels, content descriptions, large text support.
- Visual regression: golden screenshots on a few key screens.

---

### Module 12: Voice I/O — `android/app/src/main/kotlin/com/aura/voice/`

| File | Purpose | LOC |
|---|---|---|
| `VoiceService.kt` | Foreground service for always-listening mode (opt-in). | 300 |
| `WakeWordDetector.kt` | Porcupine integration. | 250 |
| `VadDetector.kt` | Silero VAD ONNX. | 250 |
| `SttEngine.kt` | whisper.cpp Android binding. | 350 |
| `TtsEngine.kt` | Piper ONNX OR Android TextToSpeech. | 250 |
| `AudioPipeline.kt` | Mic → VAD → STT → result. | 300 |
| `VoiceActivity.kt` | Voice session state (listening, thinking, speaking). | 200 |
| `PushToTalk.kt` | Button-held mode. | 150 |

**Verification:**
- VAD: 100 sample audios, correct segment/non-segment classification >95%.
- STT: 50 recorded utterances transcribed correctly.
- Wake word: false-positive rate <1/hour.
- TTS: synthesized output intelligible.

---

### Module 13: Proactive Android Integration — `android/app/src/main/kotlin/com/aura/integration/`

| File | Purpose | LOC |
|---|---|---|
| `AuraForegroundService.kt` | Holds proactive state alive. | 300 |
| `AuraAccessibilityService.kt` | Screen reading (opt-in). | 400 |
| `AuraNotificationListener.kt` | Reads other apps' notifications (opt-in). | 350 |
| `AuraQuickTile.kt` | Quick Settings tile. | 200 |
| `AuraAppShortcut.kt` | App shortcut (Cmd+K-style launcher). | 200 |
| `AuraWidget.kt` | Home screen widget (briefing, status). | 350 |
| `ShareReceiverActivity.kt` | Handle shared content → Aura. | 200 |
| `ShareSenderActivity.kt` | Send content to other apps from Aura. | 150 |
| `WorkScheduler.kt` | WorkManager jobs (morning brief, memory decay, model refresh). | 250 |
| `BootReceiver.kt` | Resume on boot. | 100 |
| `DeepLinkHandler.kt` | Handle `aura://` deep links. | 150 |

**Verification:**
- Each service: starts, stops, resumes correctly.
- Battery impact: <3% per 24h with proactive enabled.
- Memory footprint: <300MB background.

---

## 4. v1 / v2 / v3 split

### v1 — Daily-usable chat (target: 10-15 days)

- Module 0 (skeleton) — full
- Module 1 (Provider SDK) — Local + OllamaCloud + Anthropic + OpenAI. Other providers stubbed.
- Module 2 (Brain + Agent) — AgenticLoop, ConversationManager, ContextCompressor, PlanMode. Other agentic_loop_* files minimal.
- Module 3 (Memory) — MemoryStore, UnifiedMemory, Embedder, VectorIndex, FadeMem, WriteGate, UserProfile. KG deferred.
- Module 4 (Tools) — Core 20 tools only: web_search, web_fetch, deep_research, voice_input, voice_output, notifications, calendar_read, calendar_write, contacts_search, location_now, photo_library, share_intent, file_pick, app_launcher, system_volume, system_brightness, code_executor_cloud, shell_cloud, clipboard_write, image_gen.
- Module 5 (Multi-agent) — Orchestrator + 3 specialists (General, Researcher, PhoneNative). Others stubbed.
- Module 11 (UI) — ChatScreen, MemoryScreen, SettingsScreen, ModelPicker, CommandPalette, StatusStrip. Other screens minimal/stubbed.
- Module 12 (Voice) — PushToTalk + STT + TTS. No wake word yet.

**v1 acceptance:** Elnur can install the APK, send a message, get a response from a local or cloud model, use 5-10 tools (web search, calendar, contacts, location, share), and see memory grow over a day.

### v2 — Proactive + hands (target: days 15-25)

- Module 1 — DeepSeek, Gemini added. All 17 target providers wired (stubs OK for unused).
- Module 2 — SubAgent, VerificationStage, EventLog full. Plan mode full.
- Module 3 — KnowledgeGraph added. Reranker.
- Module 4 — All 80+ tools wired.
- Module 5 — All 9 specialists wired.
- Module 6 (Hands) — full
- Module 7 (Proactive) — full proactive engine + all monitors
- Module 8 (Emotion) — ALMA integrated
- Module 9 (Security) — full
- Module 11 — ToolsScreen, AutomationsScreen, SpecialistsScreen, NotificationCenter, all UI complete
- Module 12 — WakeWord, VAD, VoiceService
- Module 13 — Full proactive integration (foreground service, accessibility, notification listener, widget, tile, shortcut, share receiver)

**v2 acceptance:** Morning brief fires at 7am, proactive monitors surface relevant signals, hands run scheduled tasks, full voice mode works with wake word, all specialists available.

### v3 — Polish + safety + ship (target: days 25-45)

- Performance: 60fps UI, <200ms input latency, <2s first response.
- Battery: <5% per 24h with proactive.
- Memory: <300MB background, <500MB with one model loaded.
- Accessibility: full TalkBack, large text, color-blind safe.
- Security audit: complete threat model review.
- Settings: every config exposed, every toggle documented.
- Onboarding: first-launch flow with permissions, model download, provider setup.
- App icon, store listing (if going to Play Store), screenshots.
- Crash reporting (Sentry or self-hosted).
- Beta-test with 5-10 trusted users (if going to Play Store).

**v3 acceptance:** Daily-use for 2 weeks with no crashes, no data loss, no privacy incidents. Play Store ready.

---

## 5. Build order (the actual execution sequence)

This is the day-by-day plan assuming parallel subagents where possible.

### Day 1 — Project skeleton + provider SDK
- Module 0: Gradle, manifest, base activity, Hilt, theme
- Module 1: All provider interfaces, LocalProvider stub (loads model but doesn't run), 3 cloud providers functional
- Verification: app launches, calls Ollama Cloud, gets a streaming response

### Day 2 — Agent runtime
- Module 2: Brain, Agent, AgenticLoop, ConversationManager, ContextCompressor
- Verification: end-to-end "user sends message → model streams response → text appears"

### Day 3 — Memory stack
- Module 3: MemoryStore, UnifiedMemory, Embedder, VectorIndex, FadeMem, WriteGate, UserProfile
- Verification: store 100 memories, query, see correct ranking, see decay over time

### Day 4 — Tool system foundation
- Module 4: Tool interface, registry, executor, permission gate. First 5 tools: web_search, notifications, share_intent, location_now, calendar_read
- Verification: "user says remind me at 3pm" → calendar tool called → reminder set

### Day 5 — Chat UI
- Module 11: ChatScreen, MessageList, InputBar, StatusStrip, ModelPicker, CommandPalette
- Verification: full chat UX, streaming visible, model switcher works

### Day 6 — Voice I/O
- Module 12: STT, TTS, PushToTalk, VAD (basic)
- Verification: press mic, speak, see transcription, get voice response

### Day 7 — More tools
- Module 4: contacts_search, contacts_get, photo_library, file_pick, app_launcher, system_volume, clipboard_write, image_gen, pdf_reader
- Verification: "send my latest photo to Telegram" works

### Day 8 — Multi-agent
- Module 5: Orchestrator, BaseAgent, Protocol, Specialist. 3 specialists: General, Researcher, PhoneNative
- Verification: switch specialist, see different behavior, see different tool set

### Day 9 — Memory UI + more screens
- Module 11: MemoryScreen, MemoryDetail, ToolsScreen
- Module 3: KG added
- Verification: browse memory, search, edit, forget

### Day 10 — Hands + morning brief
- Module 6: HandManager, Hand, MemoryHand, MorningBriefingHand
- Module 7: TimeOfDayMonitor, CalendarMonitor, basic proactive engine
- Verification: morning brief fires on schedule

### Day 11 — Proactive layer foundation
- Module 7: LocationMonitor, MotionMonitor, BatteryMonitor, SalienceFilter
- Module 13: AuraForegroundService, WorkScheduler
- Verification: arrive at known location → relevant memory surfaces

### Day 12 — Remaining tools + specialists
- Module 4: all 80+ tools
- Module 5: all specialists
- Verification: "do deep research on X" works end-to-end

### Day 13 — Full proactive integration
- Module 7: NotificationMonitor, ContactMonitor, AppUsageMonitor
- Module 13: Widget, QuickTile, AppShortcut, ShareReceiver
- Verification: home screen widget shows brief, quick tile toggles voice

### Day 14 — Emotion + security
- Module 8: ALMA full
- Module 9: All security modules
- Verification: PAD state updates from signals, audit chain detects tampering, biometric blocks high-risk

### Day 15 — Voice full + accessibility
- Module 12: WakeWord, VoiceService
- Module 11: AccessibilityService support, full TalkBack
- Verification: wake word works, screen reading works (with permission)

### Day 16-30 — Polish (v2 + v3 work, as time allows)

---

## 6. Verification gates (per module)

| Module | Gate |
|---|---|
| 0 Skeleton | `./gradlew :app:assembleDebug` succeeds, app launches on emulator |
| 1 Providers | Each provider: 1 unit test (mock), 1 integration test (real API with recorded fixture) |
| 2 Agent | "Rover test": multi-step task, 2+ sub-agents, 4+ tool calls, no state loss |
| 3 Memory | Store 1000 memories, query, assert top-5 ranking correct, decay over simulated 30 days |
| 4 Tools | Each tool: unit test (mocked Android service) + on-device integration test |
| 5 Multi-agent | All 4 collaboration modes verified, specialist switching isolated |
| 6 Hands | Each hand runs end-to-end, guardian blocks unsafe |
| 7 Proactive | Simulated 24h, monitors fire at right times, salience filter prioritizes |
| 8 Emotion | PAD state updates from 20+ signals correctly |
| 9 Security | Tampering detected, taint blocks chains, SSRF blocks hosts, biometric blocks |
| 10 Channels | Phone channel + at least one external channel (Telegram) |
| 11 UI | Compose preview + UI test for every screen, accessibility audit |
| 12 Voice | VAD >95% accurate, STT >90% accurate, TTS intelligible, wake word <1 FP/hr |
| 13 Integration | Battery <5%/24h, foreground service survives memory pressure |

**Pipeline:** tsc-equivalent (Kotlin compiler) → lint → unit tests → on-device integration tests → manual dogfooding.

---

## 7. Risks and tradeoffs

### R1: On-device model size hurts adoption
3GB on first download is a Play Store filter. Mitigation: download on first use, not in APK. Offer 1.5B / 4B / 7B options. Allow deletion. Local model is optional — cloud works without it.

### R2: Battery drain from proactive
Foreground service with always-on sensors will drain. Mitigation: aggressive Doze, JobScheduler deferral, user-toggle per monitor, "ambient mode" opt-in.

### R3: Permission fatigue
20+ Android permissions scare users. Mitigation: lazy request, just-in-time rationale, graceful degradation (tool returns "permission needed" instead of crashing), settings screen shows what each permission enables.

### R4: Memory growth
10K+ memories with vector search = slow. Mitigation: cap at 5K, prune by FadeMem, sqlite-vec is fast up to ~100K vectors.

### R5: 17 provider API drift
Providers change their APIs. Mitigation: each provider isolated behind `Provider` interface, versioned schema, integration tests catch regressions.

### R6: Tool surface is a security surface
80+ tools = many attack vectors. Mitigation: tool signing, taint tracking, audit chain, permission gating, biometric for high-risk, every tool has explicit risk level.

### R7: Agent loop bugs are subtle
Aura's `agentic_loop.py` is 1.6K LOC and has been hardened over 9 review cycles. Reimplementing in Kotlin will introduce new bugs. Mitigation: extensive unit tests covering edge cases (cancellation, partial JSON, parallel tool calls, tool errors mid-loop, network failures), port test cases from Python, run "Rover test" daily.

### R8: Kuzu has no Android build
Knowledge graph needs a replacement. Mitigation: ObjectBox (commercial) or custom SQLite-backed KG. Defer to v2.

### R9: Compose performance on low-end devices
Some Compose patterns cause jank. Mitigation: baseline profile, baseline metrics, measure on a low-end target (Pixel 4a, Galaxy A-series).

### R10: AI delegation may produce inconsistent code style
Multiple subagents writing Kotlin in parallel = drift. Mitigation: ktlint + detekt, code review at module boundaries, shared style guide at top of repo, pre-commit hook.

---

## 8. Open questions (need decisions before Day 1)

### Q1: Min SDK
Aura runs on Python 3.11+. For Android, min SDK 26 (Android 8.0) gives ~95% device coverage. Going higher (28, 29) buys better APIs (BiometricPrompt, foreground service types) but cuts users. **Recommend: 26.**

### Q2: On-device model
- llama.cpp Android binding — official, large community, MLC-LLM is faster but harder to set up
- Or skip on-device model in v1, ship cloud-only, add local in v1.5
**Recommend: llama.cpp, with 1.5B/4B/7B options. v1 ships 1.5B and 4B, v2 adds 7B.**

### Q3: Voice I/O
- whisper.cpp Android — official, well-supported
- Silero VAD — small ONNX, runs on any device
- Piper TTS — small ONNX, runs on any device
- Porcupine wake word — Picovoice SDK, free tier
**Recommend: all four. Total on-device voice bundle ~200MB.**

### Q4: Knowledge graph
- ObjectBox — fast, commercial license
- Custom SQLite-backed — slower, free
- Drop KG entirely, use only vector + BM25 + recency
**Recommend: drop in v1, add custom SQLite-backed in v2 if user wants it.**

### Q5: Reranking
- bge-reranker-base int8 — ~100MB on-device
- Skip reranking, use RRF only
**Recommend: skip in v1, add in v2.**

### Q6: Cloud sandbox for code/shell
- E2B — well-known, free tier
- Code Interpreter API (OpenAI)
- Custom Docker-based
**Recommend: E2B in v2. v1 has code_executor_cloud and shell_cloud as stubs that error gracefully.**

### Q7: Distribution
- Sideload APK only (v1)
- Play Store internal testing (v2)
- Play Store public (v3)
**Recommend: sideload v1, internal testing v2, public v3.**

### Q8: Backup / sync
- All local, no sync (v1)
- Optional encrypted backup to user's own cloud (Drive/Dropbox)
- Optional sync to a self-hosted Aura instance
**Recommend: local only v1, encrypted backup v2.**

### Q9: Account / multi-device
- Single user, single device (v1)
- Optional account for multi-device sync
**Recommend: single user, single device v1. Account later if ever.**

### Q10: Theme / visual identity
- Port Aura's CLI theme to Compose Material 3
- Design fresh superapp identity
**Recommend: start with Material 3 default + Aura's color palette, refine in v3.**

---

## 9. The "begin" trigger

When Elnur says "begin" (or "go", "ship it", "do it"):
- Start Day 1 immediately.
- Subagent 1: project skeleton (Module 0).
- Subagent 2: provider SDK (Module 1).
- Subagent 3: agent runtime (Module 2).
- Commit at module boundaries.
- Report at end of each day.
- Pause only when blocked or at end of v1.

---

## 10. Files to be created (top-level)

```
D:\Aura\
├── android/                          # NEW Gradle project
│   ├── .gitignore
│   ├── build.gradle.kts
│   ├── settings.gradle.kts
│   ├── gradle/
│   │   └── libs.versions.toml
│   ├── docs/
│   │   ├── architecture.md
│   │   ├── threat-model.md
│   │   └── contributing.md
│   └── app/
│       ├── build.gradle.kts
│       ├── proguard-rules.pro
│       └── src/
│           ├── main/
│           │   ├── AndroidManifest.xml
│           │   ├── kotlin/com/aura/
│           │   │   ├── AuraApp.kt
│           │   │   ├── MainActivity.kt
│           │   │   ├── ui/                # Module 11
│           │   │   ├── voice/             # Module 12
│           │   │   ├── integration/       # Module 13
│           │   │   └── di/                # Hilt modules
│           │   └── res/
│           └── test/
├── aura-core/                        # NEW Kotlin library module
│   ├── build.gradle.kts
│   └── src/
│       ├── main/kotlin/com/aura/
│       │   ├── providers/            # Module 1
│       │   ├── agent/                # Module 2
│       │   ├── memory/               # Module 3
│       │   ├── tools/                # Module 4
│       │   ├── multi_agent/          # Module 5
│       │   ├── hands/                # Module 6
│       │   ├── proactive/            # Module 7
│       │   ├── emotion/              # Module 8
│       │   ├── security/             # Module 9
│       │   └── channels/             # Module 10
│       └── test/
```

---

## 11. Source-of-truth mapping (Aura Python → Android Kotlin)

This table makes the port auditable. Every Android module names the Aura files it ports.

| Android module | Ports from Aura |
|---|---|
| 0 Skeleton | (new) |
| 1 Providers | `aura/providers/` (1.5K), `aura/core/router.py` (658), `aura/routing/` (1054), `aura/core/model_router_mixin.py` (567) |
| 2 Agent | `aura/brain.py` (2014), `aura/brain_support.py` (283), `aura/agent.py` (1498), `aura/core/agentic_loop.py` (1603), `aura/core/agentic_loop_*.py` (1083), `aura/core/conversation_*.py`, `aura/core/token_manager.py`, `aura/core/planner.py`, `aura/core/adaptive_planner.py`, `aura/core/sub_agent.py`, `aura/core/tool_executor.py`, `aura/core/verification_stage.py`, `aura/core/event_log.py`, `aura/core/json_utils.py` |
| 3 Memory | `aura/memory/` (4417) minus Kuzu, plus `aura/core/kg_brain.py`, `kg_sync.py` |
| 4 Tools | `aura/tools/` (84 files) → ~20 ported, ~50 phone-native (new), ~10 dropped |
| 5 Multi-agent | `aura/multi_agent/` (5 files), `aura/multi_agent/specialists/` (5 files), `agents/ceo/`, `agents/founding-engineer/` |
| 6 Hands | `aura/hands/` (10 files, 2632) |
| 7 Proactive | `aura/consciousness/proactive_awareness.py` (845), `idle_presence.py` (799), `world_model.py` (slimmed from 2358), `reward_signals.py` (322), `state_extractor.py` (294), `aura/proactive/` (13 files) |
| 8 Emotion | `aura/emotion/` (5 files, 2779) |
| 9 Security | `aura/security/` (5 files) + Android-specific additions |
| 10 Channels | `aura/channels/bridge.py`, `channel_bridge.py`, `display.py`, `telegram_channel.py`, `extension_channel.py` |
| 11 UI | (new) — inspired by `aura/cli/` (~10K) UX patterns |
| 12 Voice | `aura/tools/voice.py`, `voice_synth.py`, `voice_manager.py` (partial), mostly new |
| 13 Integration | (new) — Android-specific |

**Total Aura LOC ported: ~25K Python → ~30K Kotlin** (the 1.2x is because Kotlin has more type boilerplate, but we're dropping 60% of the desktop-specific logic and adding 30% Android-specific).

**Total new Kotlin (no Aura equivalent): ~8K** — phone-native tools, UI, integration.

---

## 12. What's NOT ported (and why)

| Aura subsystem | LOC | Why dropped |
|---|---|---|
| `aura/consciousness/intrinsic_motivation.py` | 882 | Research code, no user-visible value |
| `aura/consciousness/metacognition.py` | 1041 | Research code, replaced with simpler signals |
| `aura/consciousness/self_improvement.py` | 1205 | GEPA evolution needs server-scale compute |
| `aura/consciousness/strategy_bandit.py` | 1031 | Research code, no user-visible value |
| `aura/consciousness/reasoning_templates.py` | 940 | Research code, prompts inlined into brain instead |
| `aura/evolution/` | (whole dir) | Prompt evolution, needs server-scale compute |
| `aura/code_executor.py` (desktop) | (in tools) | Replaced with `code_executor_cloud.kt` |
| `aura/shell_executor.py` | (in tools) | Replaced with `shell_cloud.kt` |
| `aura/system_control.py`, `windows_control.py` | (in tools) | Not applicable on phone |
| `aura/git_tool.py` | (in tools) | Not applicable on phone |
| `aura/filesystem.py` (large parts) | (in tools) | Replaced with `file_pick.kt` (SAF) |
| `aura/codebase_index.py` (large parts) | (in tools) | Not applicable on phone |
| `aura/browser/` (Playwright) | (whole dir) | Replaced with `web_fetch.kt` + `Chrome Custom Tabs` |
| `aura/multi_user/` | (whole dir) | Single-user app |
| `aura/coding_agent.py` (full) | (in tools) | Coder specialist handles code via cloud sandbox |
| `aura/agentic_loop.py` parts (MCTS, debate) | (parts) | Simplified in v1, full debate/mcts in v2 |

---

## 13. Anti-features (deliberate non-goals)

These are *not* in the app. Documented so future contributors don't add them.

- **No social / sharing features.** The app is for Elnur, not a social network.
- **No in-app purchases / billing.** No Play Store IAP in v1-v3.
- **No ads.** Never.
- **No analytics to third parties.** Sentry is OK if self-hosted. No Google Analytics, no Mixpanel, no Amplitude.
- **No cross-device sync in v1.** Local only. Encrypted backup in v2 if at all.
- **No account creation in v1.** Single device, single user.
- **No real-time collaboration.** No shared chats, no live cursors.
- **No marketplace / skill store.** Tools are bundled. No third-party tools in v1.
- **No cloud sync of conversations.** Conversations stay on device.
- **No telemetry that leaks content.** Opt-in error reporting only, redacted.
- **No background model training.** No using user data to improve models.
- **No ads / sponsored content in tool results.**
- **No dark patterns.** No "are you sure you want to leave" modals. No forced onboarding loops.
- **No Pro/Plus/Free tier UI.** The app is the app. If a feature is gated, it's because it's not done.

---

## 14. The daily ritual (for Elnur)

Once v1 ships, the day looks like:

1. Wake up. Phone already has a notification: "Morning brief: 9am standup, weather, 2 new memories from yesterday." Tap to expand.
2. Commute. Hold the home button, say "play my morning playlist" → MediaSession tool fires. Say "any urgent messages?" → Notification tool reads the important ones.
3. At desk. Open Aura. Chat: "summarize the three unread emails from yesterday" → email tool, cloud model summarizes. Memory: "remember I prefer 15-min standups" → WriteGate stores.
4. Midday. Quick Settings tile → push to talk → "set a reminder for 4pm to call mom" → calendar tool, WorkManager scheduled.
5. Walking. Phone detects motion + arrived at café → "last time you were here Tuesday, you ordered a flat white" surfaces.
6. Evening. Open chat. Say "plan my day tomorrow" → plan mode, generates, persists. "Send the plan to my calendar" → calendar tool.
7. Bed. Say "goodnight" → idle presence state, voice output wishes goodnight, proactive monitors go quiet.

That's the app. Every part of that is in the plan above.
