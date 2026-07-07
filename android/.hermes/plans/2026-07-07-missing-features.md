# Missing Features — Aura Android

> **For Hermes:** Plan+execute pattern. No "should I continue?" between commits.
> Write plan, then implement all items in one session without asking "continue?" between commits.

**Goal:** Implement 26 missing features identified in the 10th-pass engineering review feature inventory.

**Architecture:** Each item is a self-contained addition that extends existing infrastructure. No architectural changes. Items are grouped into 8 phases by dependency order and impact. Each phase is independently shippable.

**Tech Stack:** Kotlin 1.9.24, Compose BOM 2024.10.01, Hilt 2.51, Room 2.6.1, WorkManager, DataStore, AGP 8.2.2, compileSdk 35, minSdk 26.

**Prior plans:**
- `2026-07-07-deep-polish.md` — 10 missing capabilities (context, markdown, failover, undo, back-press, data usage, export with images). EXECUTED in commit ed5dd65.
- `2026-07-07-deep-structural-fixes.md` — 8 deep structural fixes (recall, profile, KG, tools, widget). EXECUTED in commit 2e13770.
- `2026-07-05-tier-1-polish.md` — 7 Tier 1 polish items. EXECUTED in commits 04f6c42, a1a715b.

**Verification gate:** `./gradlew :aura-core:testDebugUnitTest :app:testDebugUnitTest :app:assembleDebug`

**Claims verified false (pre-plan verification):**
- None. All 26 items confirmed missing via grep on 2026-07-07.

**Already partially implemented:**
- #20 Provider health check: `verifyKey()` exists in OnboardingScreen (line 99) but NOT in SettingsScreen. This plan adds it to SettingsScreen.

---

## Phase 1: Quick Wins (single-commit, high daily-use impact)

### Item 1: Reminder creation UI in Tasks screen

**Objective:** Let the user create reminders from the Tasks screen without talking to the agent.

**Files:**
- Create: `app/src/main/kotlin/com/aura/ui/components/AddReminderDialog.kt`
- Modify: `app/src/main/kotlin/com/aura/ui/screens/TasksScreen.kt` — add FAB or "Add reminder" button, render dialog
- Modify: `app/src/main/kotlin/com/aura/ui/viewmodel/TasksViewModel.kt` — add `createReminder(message, triggerAt)` that calls `reminderDao.insert()`
- Test: `app/src/test/kotlin/com/aura/ui/viewmodel/TasksViewModelReminderTest.kt`

**Approach:**
- AddReminderDialog composable: message text field, time picker (uses TimeParser.parse for "HH:mm" input or a Material3 DatePicker + TimePicker), "Add" button
- TasksViewModel.createReminder(message: String, triggerAt: Long) — inserts ReminderEntity, schedules via WorkManager (existing ReminderWorker pattern)
- TasksScreen: add a small "+" FAB or a "Add reminder" button in the Reminders section header, show dialog on tap
- Reuse TimeParser from aura-core for time parsing

**Test:**
- TasksViewModelReminderTest: verify createReminder inserts to dao and schedules work
- Mock ReminderDao, verify insert called with correct entity

**Commit:** `feat(android): reminder creation UI in Tasks screen`

---

### Item 2: Provider health check in Settings

**Objective:** Let the user test if an API key is valid from Settings without sending a real message.

**Files:**
- Modify: `app/src/main/kotlin/com/aura/ui/settings/SettingsViewModel.kt` — add `verifyKey(prefix: String)` (same pattern as OnboardingScreen)
- Modify: `app/src/main/kotlin/com/aura/ui/screens/SettingsScreen.kt` — add "Test" button next to each provider key field, show result inline
- Test: `app/src/test/kotlin/com/aura/ui/settings/SettingsViewModelVerifyKeyTest.kt`

**Approach:**
- Copy the verifyKey pattern from OnboardingScreen.OnboardingViewModel (line 99)
- SettingsViewModel: add `verifyResult: Map<String, String>` to UI state, `verifyKey(prefix)` calls `provider.listModels()` and stores result
- SettingsScreen: per-provider "Test" button that calls viewModel.verifyKey(prefix), shows ✓/✗ result below the key field

**Test:**
- Mock ProviderRegistry, verify verifyKey calls listModels and stores result

**Commit:** `feat(android): provider health check in Settings`

---

### Item 3: Timer tool

**Objective:** Add a simple timer/stopwatch tool the agent can use.

**Files:**
- Create: `aura-core/src/main/kotlin/com/aura/tools/TimerTool.kt`
- Modify: `aura-core/src/main/kotlin/com/aura/tools/ToolsModule.kt` — register timerTool
- Test: `aura-core/src/test/kotlin/com/aura/tools/TimerToolTest.kt`

**Approach:**
- TimerTool: two actions — "start" (returns a timer ID, records start time in an in-memory map) and "check" (returns elapsed seconds). "stop" cancels and returns final elapsed.
- ToolRisk: READ_ONLY (no phone permissions, just in-memory state)
- In-memory timer map: `Map<String, Long>` (timerId → startEpochMillis), synchronized
- No notification — the agent asks "how much time has passed?" and the tool returns the number

**Test:**
- Start timer, check elapsed > 0, stop returns final
- Non-existent timer ID returns error

**Commit:** `feat(android): timer tool for agent time tracking`

---

### Item 4: Weather tool (free, no API key)

**Objective:** Add a weather tool using Open-Meteo (free, no API key required).

**Files:**
- Create: `aura-core/src/main/kotlin/com/aura/tools/WeatherTool.kt`
- Modify: `aura-core/src/main/kotlin/com/aura/tools/ToolsModule.kt` — register weatherTool
- Test: `aura-core/src/test/kotlin/com/aura/tools/WeatherToolTest.kt`

**Approach:**
- WeatherTool: takes optional lat/lon (defaults to device location if LocationNowTool is available, or asks user), calls `https://api.open-meteo.com/v1/forecast?latitude=X&longitude=Y&current=temperature_2m,wind_speed_10m,weather_code`
- Parse JSON response, return formatted string: "18°C, partly cloudy, wind 12 km/h"
- ToolRisk: READ_ONLY (network egress only)
- No API key needed — Open-Meteo is free for non-commercial use
- If no lat/lon provided and no location available, return error asking the agent to get location first

**Test:**
- Mock OkHttpClient with a sample Open-Meteo response, verify parsed output
- Test missing location handling

**Commit:** `feat(android): weather tool via Open-Meteo (free, no API key)`

---

## Phase 2: Memory System Enhancements

### Item 5: Semantic memory dedup

**Objective:** Prevent semantically similar memories from being stored as duplicates.

**Files:**
- Modify: `aura-core/src/main/kotlin/com/aura/memory/MemoryStore.kt` — add semantic dedup check in `maybeStore()` and `store()`
- Modify: `aura-core/src/main/kotlin/com/aura/memory/MemoryDao.kt` — add `allWithEmbeddings()` query (returns all memories that have non-null embeddings)
- Test: `aura-core/src/test/kotlin/com/aura/memory/MemoryStoreSemanticDedupTest.kt`

**Approach:**
- After the write gate says "store" and before embedding, check existing memories:
  1. Exact match (existsByContent) — already exists, skip
  2. Semantic match — embed the new content, scan existing embeddings, if cosine similarity > 0.92 with any existing memory, skip (or merge — see item 6)
- The 0.92 threshold is conservative — "I prefer dark mode" vs "I like dark mode" would have cosine ~0.95, while "I prefer dark mode" vs "I prefer light mode" would be ~0.85
- Performance: the scan is O(n) over memories with embeddings. For a personal app (hundreds of memories), this is fast. For larger installs, limit the scan to the same category.
- Only run in `maybeStore()` (the auto-store path). `store()` (explicit agent call) should also check but log when it skips.

**Test:**
- Store "I prefer dark mode", then try to store "I like dark mode" — should be skipped
- Store "I prefer dark mode", then try to store "I live in Baku" — should be stored (different topic)

**Commit:** `feat(android): semantic memory dedup via cosine similarity threshold`

---

### Item 6: Memory merge for similar memories

**Objective:** When a new memory is semantically similar to an existing one, merge instead of skip — keeping the richer version.

**Files:**
- Modify: `aura-core/src/main/kotlin/com/aura/memory/MemoryStore.kt` — add `mergeSimilar()` method, call from `maybeStore()` when semantic match found
- Test: `aura-core/src/test/kotlin/com/aura/memory/MemoryStoreMergeTest.kt`

**Approach:**
- When semantic match found (cosine > 0.92):
  - If the new content is longer/richer than the existing → replace existing with new (keep old category + importance if higher)
  - If the existing is longer → keep existing, skip new
  - If same length → keep existing (first writer wins)
- Merge = update existing memory's content, re-null the embedding (will be re-embedded on next recall), bump accessedAt
- This is a refinement of Item 5 — instead of just skipping, we merge

**Test:**
- Store "I like dark mode", then store "I prefer dark mode for my IDE" — existing should be replaced with the richer version
- Store "I like dark mode", then store "dark" — existing should be kept (shorter)

**Commit:** `feat(android): memory merge for semantically similar entries`

---

### Item 7: Manual note/pin creation

**Objective:** Let the user manually create a memory without going through the write gate.

**Files:**
- Modify: `app/src/main/kotlin/com/aura/ui/screens/MemoryScreen.kt` — add "Add note" button, show dialog
- Modify: `app/src/main/kotlin/com/aura/ui/viewmodel/MemoryViewModel.kt` — add `createNote(content, category, importance)` that calls `memoryStore.store()` directly (bypasses write gate)
- Test: `app/src/test/kotlin/com/aura/ui/viewmodel/MemoryViewModelNoteTest.kt`

**Approach:**
- "Add note" FAB or button in MemoryScreen header
- Dialog: content text field, category dropdown (the 6 existing categories + "note" as a new manual category), importance slider
- Calls `memoryStore.store()` directly — no write gate, no LLM call, no dedup check. The user explicitly wants this stored.
- The note is tagged with source="manual" so the UI can show it differently if desired

**Test:**
- Verify createNote calls store() with the right parameters
- Verify the note appears in the memory list

**Commit:** `feat(android): manual note creation — bypass write gate for explicit user input`

---

### Item 8: Memory edit history

**Objective:** Track when memories are edited and what changed.

**Files:**
- Create: `aura-core/src/main/kotlin/com/aura/memory/MemoryEditEntity.kt` — Room entity: id, memoryId, oldContent, newContent, editedAt, editedBy ("user" or "agent")
- Create: `aura-core/src/main/kotlin/com/aura/memory/MemoryEditDao.kt` — insert, getForMemory(memoryId)
- Modify: `aura-core/src/main/kotlin/com/aura/memory/MemoryDatabase.kt` — add MemoryEditEntity to the database, bump version, add migration
- Modify: `aura-core/src/main/kotlin/com/aura/memory/MemoryStore.kt` — in `update()`, record the old content before updating
- Modify: `app/src/main/kotlin/com/aura/ui/screens/MemoryScreen.kt` — in the edit dialog, show "Edit history" section with timestamps
- Test: `aura-core/src/test/kotlin/com/aura/memory/MemoryEditHistoryTest.kt`

**Approach:**
- MemoryEditEntity: `@Entity(tableName = "memory_edits")` with foreignKey to MemoryEntity
- MemoryStore.update(): before calling dao.update(), read the existing memory, insert a MemoryEditEntity with old content, then update
- MemoryScreen: in the edit dialog, add a collapsible "Edit history" section that lists prior versions with timestamp + who edited
- Migration: MIGRATION_N_N+1 adds the memory_edits table. Export schema.

**Test:**
- Update a memory, verify edit history entity is created with old content
- Load edit history for a memory, verify correct entries

**Commit:** `feat(android): memory edit history with audit trail`

---

## Phase 3: Conversation & Export

### Item 9: Conversation statistics

**Objective:** Show per-conversation and aggregate stats.

**Files:**
- Modify: `app/src/main/kotlin/com/aura/ui/viewmodel/HistoryViewModel.kt` — add `getStats(conversation)` returning turn count, tool call count, duration
- Modify: `app/src/main/kotlin/com/aura/ui/screens/HistoryScreen.kt` — show stats per row (turns, tools, date range)
- Modify: `app/src/main/kotlin/com/aura/ui/screens/ChatScreen.kt` — show conversation stats in the chat header (turns, tools, tokens)
- Test: `app/src/test/kotlin/com/aura/ui/viewmodel/HistoryViewModelStatsTest.kt`

**Approach:**
- HistoryViewModel.getStats(conv): count turns, count toolTurns across all turns, compute duration (updatedAt - createdAt)
- HistoryScreen: each row shows "12 turns · 3 tools · 2d ago"
- ChatScreen: header shows "Turn 5 · 2 tools · 1.2K tokens" (tokens from UsageTracker)
- No new entity — all computed from existing Conversation data

**Test:**
- Verify getStats returns correct counts for a conversation with known turns

**Commit:** `feat(android): conversation statistics — turns, tools, duration`

---

### Item 10: Bulk conversation export

**Objective:** Export all conversations as a single Markdown or JSON file.

**Files:**
- Modify: `app/src/main/kotlin/com/aura/ui/viewmodel/HistoryViewModel.kt` — add `exportAllMarkdown()` and `exportAllJson()`
- Modify: `app/src/main/kotlin/com/aura/ui/screens/HistoryScreen.kt` — add "Export all" button in the top bar
- Test: `app/src/test/kotlin/com/aura/ui/viewmodel/HistoryViewModelExportAllTest.kt`

**Approach:**
- exportAllMarkdown(): joins all conversations with `---` separators, returns one big string
- exportAllJson(): serializes all conversations to a JSON array via kotlinx.serialization
- HistoryScreen: "Export all" button → share intent with the exported file
- Reuse the shareMarkdown helper pattern from the existing per-conversation export

**Test:**
- Verify exportAllMarkdown contains all conversation titles
- Verify exportAllJson is valid JSON

**Commit:** `feat(android): bulk conversation export — all conversations as one file`

---

### Item 11: Semantic conversation search

**Objective:** Search conversations by meaning, not just keyword match.

**Files:**
- Modify: `aura-core/src/main/kotlin/com/aura/agent/ConversationStore.kt` — add `semanticSearch(query, limit)` that embeds the query and compares against conversation turn embeddings
- Modify: `aura-core/src/main/kotlin/com/aura/agent/ConversationEntity.kt` — add `embedding: ByteArray?` field (nullable, lazy-populated)
- Modify: `aura-core/src/main/kotlin/com/aura/agent/ConversationDao.kt` — add `allWithEmbeddings()`, `updateEmbedding()`
- Modify: `aura-core/src/main/kotlin/com/aura/agent/ConversationDatabase.kt` — bump version, add migration
- Modify: `app/src/main/kotlin/com/aura/ui/viewmodel/HistoryViewModel.kt` — use semanticSearch when query is non-trivial (>3 chars)
- Test: `aura-core/src/test/kotlin/com/aura/agent/ConversationStoreSemanticSearchTest.kt`

**Approach:**
- ConversationEntity gets an `embedding: ByteArray?` column
- When a conversation is saved, embed its last user message (lazy — only embed on first search hit, not on every save)
- semanticSearch: embed query, scan all conversations with embeddings, rank by cosine similarity, return top N
- Fall back to SQL LIKE if no conversations have embeddings yet
- Migration: MIGRATION_N_N+1 adds the embedding column

**Test:**
- Save conversation, embed it, search semantically, verify hit
- Empty embeddings → falls back to LIKE search

**Commit:** `feat(android): semantic conversation search via embeddings`

---

## Phase 4: Voice & Widget

### Item 12: Continuous voice mode

**Objective:** Hands-free voice conversation — talk → response → listen again, without tapping.

**Files:**
- Create: `app/src/main/kotlin/com/aura/ui/voice/ContinuousVoiceMode.kt` — state machine: LISTENING → THINKING → SPEAKING → LISTENING
- Modify: `app/src/main/kotlin/com/aura/ui/screens/ChatScreen.kt` — add "Voice mode" toggle in the header, when active shows a full-screen voice overlay
- Modify: `app/src/main/kotlin/com/aura/ui/viewmodel/ChatViewModel.kt` — add `voiceModeActive` state, `startVoiceMode()` / `stopVoiceMode()`
- Modify: `app/src/main/kotlin/com/aura/ui/voice/VoiceOverlay.kt` — full-screen voice UI with animated waveform
- Test: `app/src/test/kotlin/com/aura/ui/viewmodel/ChatViewModelVoiceModeTest.kt`

**Approach:**
- State machine: LISTENING (SpeechToText active) → on result, send() → THINKING (streaming) → on Done, TTS speaks → SPEAKING → on TTS done, back to LISTENING
- Exit conditions: user taps "Stop", user says "stop listening", 10 seconds of silence in LISTENING state
- VoiceOverlay: full-screen translucent overlay with animated waveform, current state label, "Stop" button
- ChatViewModel: `voiceModeActive: Boolean` in UI state, `startVoiceMode()` sets it true and triggers first listening cycle, `stopVoiceMode()` sets false and cancels
- The voice mode reuses existing SpeechToText + TextToSpeech + the agent loop — no new LLM calls, just a loop wrapper

**Test:**
- Verify startVoiceMode sets state, stopVoiceMode clears state
- Verify the state machine transitions (mocked SpeechToText + TextToSpeech)

**Commit:** `feat(android): continuous voice mode — hands-free conversation loop`

---

### Item 13: Widget configuration

**Objective:** Let the user configure the AskAuraWidget — pick model, set prompt prefix.

**Files:**
- Create: `app/src/main/kotlin/com/aura/widget/WidgetConfigActivity.kt` — configuration activity launched on widget placement
- Modify: `app/src/main/kotlin/com/aura/widget/AskAuraWidget.kt` — read config from SharedPreferences, use configured model
- Modify: `app/src/main/kotlin/com/aura/widget/QuickAskActivity.kt` — use configured model instead of default
- Modify: `app/src/main/AndroidManifest.xml` — add WidgetConfigActivity with APPWIDGET_CONFIGURE intent filter
- Modify: `app/src/main/res/xml/widget_ask_aura_info.xml` — add `android:configure` attribute
- Test: `app/src/test/kotlin/com/aura/widget/WidgetConfigTest.kt`

**Approach:**
- WidgetConfigActivity: model picker (reuse ModelPickerSheet filtered to configured providers), optional prompt prefix text field, "Save" button
- Config stored in SharedPreferences keyed by widget ID (AppWidgetId), so each widget instance can have different settings
- AskAuraWidget.onUpdate reads config, uses configured model for the quick-ask action
- QuickAskActivity reads config, uses configured model
- If no config (existing widgets), falls back to default model

**Test:**
- Verify config is saved and read correctly
- Verify fallback to default when no config exists

**Commit:** `feat(android): widget configuration — model picker + prompt prefix`

---

## Phase 5: Hands & Specialists

### Item 14: Hands step builder UI

**Objective:** Visual step builder for creating hands — tool dropdown + per-tool argument form.

**Files:**
- Create: `app/src/main/kotlin/com/aura/ui/components/ToolArgForm.kt` — composable that adapts to ToolDefinition.parameters, renders a form field per property
- Create: `app/src/main/kotlin/com/aura/ui/components/HandStepBuilder.kt` — composable with tool dropdown, step list, add/remove/reorder steps
- Modify: `app/src/main/kotlin/com/aura/ui/screens/HandsScreen.kt` — replace raw JSON text field with HandStepBuilder in AddHandDialog and EditHandDialog
- Modify: `app/src/main/kotlin/com/aura/ui/viewmodel/HandsViewModel.kt` — add `availableTools: List<ToolDefinition>` to state, populate from ToolRegistry
- Test: `app/src/test/kotlin/com/aura/ui/components/ToolArgFormTest.kt`

**Approach:**
- ToolArgForm: takes a ToolDefinition, renders a text field per property (type=string → TextField, type=integer → number-only TextField, type=boolean → Switch)
- HandStepBuilder: dropdown of available tools (from ToolRegistry.definitions()), when a tool is selected show ToolArgForm for its parameters, "Add step" button appends to step list, steps can be reordered via drag or up/down buttons
- HandsScreen: AddHandDialog and EditHandDialog use HandStepBuilder instead of raw JSON
- HandsViewModel: exposes availableTools from ToolRegistry

**Test:**
- ToolArgForm renders correct field types for string/int/bool properties
- HandStepBuilder produces correct JSON from the form inputs

**Commit:** `feat(android): hands step builder UI — visual tool picker + argument forms`

---

### Item 15: Specialist customization — tools + custom specialists

**Objective:** Let the user customize which tools each specialist has, and create custom specialists.

**Files:**
- Modify: `aura-core/src/main/kotlin/com/aura/agent/Specialist.kt` — add `applyToolOverrides(overrides: Map<String, Set<String>>)` 
- Modify: `aura-core/src/main/kotlin/com/aura/data/UserPreferences.kt` — add `specialistToolOverrides: Flow<String>` (JSON map of specialist name → tool set)
- Modify: `app/src/main/kotlin/com/aura/ui/screens/SettingsScreen.kt` — add per-specialist tool selection UI (checkbox list of all tools)
- Modify: `app/src/main/kotlin/com/aura/ui/viewmodel/ChatViewModel.kt` — apply tool overrides when resolving specialist
- Test: `aura-core/src/test/kotlin/com/aura/agent/SpecialistToolOverrideTest.kt`

**Approach:**
- UserPreferences: new key `specialist_tool_overrides` (JSON: `{"coder": ["brave_search", "fetch_url", "deep_research"]}`)
- Specialist.applyToolOverrides: replaces toolsAllowed with the user's set
- SettingsScreen: per-specialist section with a checkbox list of all 31 tools, checked = allowed
- ChatViewModel: when resolving specialist, read tool overrides from UserPreferences and apply
- Custom specialists: deferred to a future session — the 6 built-in ones cover the main use cases, and custom specialists need a creation UI + storage that's a bigger lift

**Test:**
- Verify applyToolOverrides replaces the toolsAllowed set
- Verify empty override map keeps the default tools

**Commit:** `feat(android): specialist tool customization — per-specialist tool selection in Settings`

---

## Phase 6: Proactive & Notifications

### Item 16: Morning brief time customization

**Objective:** Let the user set when the morning brief runs.

**Files:**
- Modify: `aura-core/src/main/kotlin/com/aura/data/UserPreferences.kt` — add `morningBriefHour: Flow<Int>` (0-23), `morningBriefMinute: Flow<Int>` (0-59)
- Modify: `aura-core/src/main/kotlin/com/aura/proactive/ProactiveScheduler.kt` — use configured time instead of hardcoded 7am
- Modify: `app/src/main/kotlin/com/aura/ui/screens/SettingsScreen.kt` — add time picker for morning brief
- Test: `aura-core/src/test/kotlin/com/aura/proactive/ProactiveSchedulerTimeTest.kt`

**Approach:**
- UserPreferences: new keys `morning_brief_hour` (int, default 7), `morning_brief_minute` (int, default 0)
- ProactiveScheduler.scheduleMorningBrief(): read the configured time, compute the next trigger, schedule with WorkManager PeriodicWorkRequest (initial delay = next occurrence of HH:mm)
- SettingsScreen: Material3 TimePicker dialog, "Morning brief at" → shows current time, tap to change
- When the user changes the time, cancel the existing work and reschedule

**Test:**
- Verify scheduleMorningBrief uses the configured time
- Verify changing the time cancels and reschedules

**Commit:** `feat(android): morning brief time customization — user-configurable time in Settings`

---

### Item 17: Reminders UI — edit + list refinement

**Objective:** Let the user edit existing reminders (change time or message).

**Files:**
- Modify: `app/src/main/kotlin/com/aura/ui/screens/TasksScreen.kt` — add edit button to ReminderRow, show EditReminderDialog
- Create: `app/src/main/kotlin/com/aura/ui/components/EditReminderDialog.kt`
- Modify: `app/src/main/kotlin/com/aura/ui/viewmodel/TasksViewModel.kt` — add `updateReminder(id, message, triggerAt)` that cancels old WorkManager job and schedules new
- Test: `app/src/test/kotlin/com/aura/ui/viewmodel/TasksViewModelEditReminderTest.kt`

**Approach:**
- EditReminderDialog: pre-fills from existing reminder, same fields as AddReminderDialog
- TasksViewModel.updateReminder: cancel existing WorkManager work by tag, insert updated ReminderEntity, schedule new work
- ReminderRow: add edit icon next to cancel icon

**Test:**
- Verify updateReminder cancels old work and schedules new

**Commit:** `feat(android): reminder edit — change time or message for existing reminders`

---

## Phase 7: Communication Tools

### Item 18: Email send tool

**Objective:** Let the agent send emails via Android intents.

**Files:**
- Create: `aura-core/src/main/kotlin/com/aura/tools/EmailSendTool.kt`
- Modify: `aura-core/src/main/kotlin/com/aura/tools/ToolsModule.kt` — register emailSend
- Test: `aura-core/src/test/kotlin/com/aura/tools/EmailSendToolTest.kt`

**Approach:**
- EmailSendTool: takes to, subject, body. Opens an ACTION_SENDTO intent with `mailto:` URI + subject/body extras
- Does NOT send directly — opens the email app for the user to confirm. This is the safe pattern for Android.
- ToolRisk: WRITE_REMOTE (sends data externally)
- If no email app is installed, returns error
- Requires no permissions — uses Intent

**Test:**
- Verify intent is constructed correctly (mock Context)
- Verify error when no email app available

**Commit:** `feat(android): email send tool via Android intent`

---

### Item 19: SMS send tool

**Objective:** Let the agent send SMS via Android intents.

**Files:**
- Create: `aura-core/src/main/kotlin/com/aura/tools/SmsSendTool.kt`
- Modify: `aura-core/src/main/kotlin/com/aura/tools/ToolsModule.kt` — register smsSend
- Test: `aura-core/src/test/kotlin/com/aura/tools/SmsSendToolTest.kt`

**Approach:**
- SmsSendTool: takes to (phone number), body. Opens ACTION_SENDTO with `smsto:` URI
- Does NOT send directly — opens the SMS app for the user to confirm
- ToolRisk: WRITE_REMOTE
- No SEND_SMS permission needed — using intent, not direct SMS Manager

**Test:**
- Verify intent is constructed correctly

**Commit:** `feat(android): SMS send tool via Android intent`

---

### Item 20: Translate tool

**Objective:** Add a translation tool using the LLM itself (no external API).

**Files:**
- Create: `aura-core/src/main/kotlin/com/aura/tools/TranslateTool.kt`
- Modify: `aura-core/src/main/kotlin/com/aura/tools/ToolsModule.kt` — register translate
- Test: `aura-core/src/test/kotlin/com/aura/tools/TranslateToolTest.kt`

**Approach:**
- TranslateTool: takes text, target_language. Calls the configured LLM with a system prompt "Translate the following text to {target_language}. Return only the translation, nothing else."
- Uses the first configured provider's first model (same pattern as DeepResearchTool fix)
- ToolRisk: READ_ONLY (LLM call only, no phone permissions)
- Returns the translated text

**Test:**
- Mock ProviderRegistry, verify the LLM is called with the right system prompt
- Verify the translated text is returned

**Commit:** `feat(android): translate tool via LLM`

---

## Phase 8: Security, Polish & Advanced

### Item 21: Picture-in-Picture mode

**Objective:** Keep the chat visible while using other apps.

**Files:**
- Modify: `app/src/main/kotlin/com/aura/MainActivity.kt` — add `enterPictureInPictureMode` on user action or onActivityPaused
- Modify: `app/src/main/AndroidManifest.xml` — add `android:supportsPictureInPicture="true"` to MainActivity
- Modify: `app/src/main/kotlin/com/aura/ui/screens/ChatScreen.kt` — add PiP button in header
- Test: manual (PiP requires a real device, can't be unit-tested)

**Approach:**
- MainActivity: override onUserLeaveHint() → if chat is active and streaming, enter PiP
- PiP params: aspect ratio 16:9 (or 1:1 for chat), seamless resize
- ChatScreen: add a PiP icon button that calls activity.enterPictureInPictureMode()
- In PiP mode, the chat shows the streaming text in a compact view
- minSdk 26 supports PiP (targetSdk 35)

**Commit:** `feat(android): picture-in-picture mode for chat`

---

### Item 22: Database encryption (SQLCipher)

**Objective:** Encrypt all Room databases at rest.

**Files:**
- Modify: `aura-core/src/main/kotlin/com/aura/data/RoomConfig.kt` — use SQLCipherSupportFactory with a key from KeyManager
- Modify: `aura-core/build.gradle.kts` — add dependency `net.zetetic:android-sqlcipher` (or `androidx.sqlite:sqlite-ktx` with encryption support)
- Modify: `app/build.gradle.kts` — same dependency
- Test: `aura-core/src/test/kotlin/com/aura/data/RoomConfigEncryptionTest.kt`

**Approach:**
- Generate a database encryption key via KeyManager (stored in Android Keystore)
- Pass the key to Room via SupportFactory(SQLCipher)
- All 6+ databases get encrypted: Memory, Conversation, KG, Hand, Task, UserProfile, ProactiveEvent, Reminder
- Existing plaintext databases need a migration: on first launch after update, read all data from the plaintext DB, create the encrypted DB, write all data, delete the plaintext DB
- This is the most complex item — the migration path needs careful testing
- If the migration is too risky for one session, ship the encrypted DB for new installs only and document the migration as a follow-up

**Test:**
- Verify encrypted DB can be opened with the key
- Verify encrypted DB cannot be opened without the key
- Verify migration from plaintext to encrypted preserves all data

**Commit:** `feat(android): database encryption via SQLCipher + Keystore`

---

### Item 23: Crash log persistence

**Objective:** Persist errors to a local log file so they survive the 5-second auto-dismiss.

**Files:**
- Create: `aura-core/src/main/kotlin/com/aura/core/error/CrashLogger.kt` — writes error to a rolling file in app cache dir
- Modify: `app/src/main/kotlin/com/aura/ui/viewmodel/ChatViewModel.kt` — log errors to CrashLogger in addition to UI display
- Modify: `app/src/main/kotlin/com/aura/ui/screens/SettingsScreen.kt` — add "View error log" button that shows the log
- Test: `aura-core/src/test/kotlin/com/aura/core/error/CrashLoggerTest.kt`

**Approach:**
- CrashLogger: singleton, writes to `cacheDir/aura-error-log.txt`, rolling at 100KB (keeps last 100KB, drops older)
- Each entry: timestamp, error code, message, stack trace (if available)
- ChatViewModel.setErrorWithAutoDismiss: also calls CrashLogger.log()
- SettingsScreen: "Diagnostics" section with "View error log" → shows the log in a scrollable text view, "Clear log" button
- No telemetry — the log is local only, never sent anywhere

**Test:**
- Verify log writes to file
- Verify rolling at 100KB
- Verify clear empties the file

**Commit:** `feat(android): persistent crash log — local error trail for debugging`

---

### Item 24: Conversation fork (branch from a specific turn)

**Objective:** Let the user fork a conversation from a specific turn to explore a different path.

**Files:**
- Modify: `aura-core/src/main/kotlin/com/aura/agent/ConversationStore.kt` — add `fork(id, fromTurnIndex): Conversation` that creates a new conversation with turns up to fromTurnIndex
- Modify: `app/src/main/kotlin/com/aura/ui/screens/ChatScreen.kt` — add "Fork from here" action on long-press of a turn
- Modify: `app/src/main/kotlin/com/aura/ui/viewmodel/ChatViewModel.kt` — add `forkConversation(fromTurnIndex)`
- Test: `aura-core/src/test/kotlin/com/aura/agent/ConversationStoreForkTest.kt`

**Approach:**
- ConversationStore.fork(id, fromTurnIndex): load conversation, copy turns[0..fromTurnIndex], create new Conversation with new ID, same systemPrompt + model, title = "{original title} (fork)", save
- ChatScreen: long-press a turn → context menu → "Fork from here" → calls viewModel.forkConversation(turnIndex) → loads the new conversation
- The original conversation is untouched — the fork is a new entry in History

**Test:**
- Fork a conversation with 5 turns from index 2, verify new conversation has 3 turns
- Verify original conversation is unchanged

**Commit:** `feat(android): conversation fork — branch from a specific turn`

---

### Item 25: KG visualization refinement

**Objective:** Improve the graph screen with type clustering, filtering, and edge weight visualization.

**Files:**
- Modify: `app/src/main/kotlin/com/aura/ui/screens/GraphScreen.kt` — add type filter chips, node clustering by type, edge thickness by weight
- Modify: `app/src/main/kotlin/com/aura/ui/viewmodel/GraphViewModel.kt` — add filter state, type-based node grouping
- Test: `app/src/test/kotlin/com/aura/ui/viewmodel/GraphViewModelFilterTest.kt`

**Approach:**
- GraphViewModel: add `nodeTypeFilter: Set<String>` to state, `toggleTypeFilter(type)` method
- GraphScreen: 
  - Filter chips at top (All, Person, Place, Organization, Event, Concept — whatever types exist in the KG)
  - Nodes colored by type (Material3 color scheme)
  - Edge thickness proportional to weight (0.0-1.0 → 1dp-4dp stroke)
  - Node size proportional to accessCount (more accessed = bigger)
- No new Room queries — uses existing dao.allNodes() + dao.allEdges()

**Test:**
- Verify filter hides nodes of unselected types
- Verify toggleTypeFilter adds/removes from filter set

**Commit:** `feat(android): KG visualization — type filtering, clustering, edge weight`

---

### Item 26: Memory categories management

**Objective:** Let the user create, rename, and merge memory categories.

**Files:**
- Modify: `aura-core/src/main/kotlin/com/aura/data/UserPreferences.kt` — add `customCategories: Flow<String>` (JSON list of custom category names)
- Modify: `app/src/main/kotlin/com/aura/ui/screens/MemoryScreen.kt` — add category management UI (create, rename, merge)
- Modify: `app/src/main/kotlin/com/aura/ui/viewmodel/MemoryViewModel.kt` — add `createCategory(name)`, `renameCategory(old, new)`, `mergeCategories(source, target)`
- Modify: `aura-core/src/main/kotlin/com/aura/memory/MemoryDao.kt` — add `updateCategory(oldCategory, newCategory)` for rename/merge
- Test: `aura-core/src/test/kotlin/com/aura/memory/MemoryCategoryManagementTest.kt`

**Approach:**
- UserPreferences: new key `custom_categories` (JSON array string)
- MemoryViewModel.createCategory(name): adds to the custom categories list
- MemoryViewModel.renameCategory(old, new): calls dao.updateCategory(old, new) to rename all memories in that category
- MemoryViewModel.mergeCategories(source, target): calls dao.updateCategory(source, target) to move all memories from source to target
- MemoryScreen: category management dialog accessible from the category filter section — create new, rename existing, merge two

**Test:**
- Verify renameCategory updates all memories in the old category
- Verify mergeCategories moves all memories from source to target

**Commit:** `feat(android): memory category management — create, rename, merge`

---

## Execution Order

1. **Phase 1 (Items 1-4):** Quick wins — 4 single-commit features, no dependencies
2. **Phase 2 (Items 5-8):** Memory system — 4 items, items 5+6 are paired (dedup then merge)
3. **Phase 3 (Items 9-11):** Conversation & export — 3 items, independent of each other
4. **Phase 4 (Items 12-13):** Voice & widget — 2 items, independent
5. **Phase 5 (Items 14-15):** Hands & specialists — 2 items, independent
6. **Phase 6 (Items 16-17):** Proactive & reminders — 2 items, item 17 depends on item 1
7. **Phase 7 (Items 18-20):** Communication tools — 3 independent tools
8. **Phase 8 (Items 21-26):** Security, polish & advanced — 6 items, item 22 (SQLCipher) is the highest-risk

**Total: 26 items, 26 commits, ~8 phases.**

**Estimated wall time:** 6-10 hours with AI delegation (3-4 subagents in parallel per phase where items are independent).

**Risk items:**
- Item 22 (SQLCipher) — database migration is the highest-risk item. If migration is too complex, ship encrypted DB for new installs only, document the migration as a follow-up.
- Item 12 (continuous voice) — requires real-device testing for the voice loop. Unit tests can verify the state machine but not the audio pipeline.
- Item 11 (semantic conversation search) — adds a column to ConversationEntity, needs migration. Lower risk than SQLCipher but still a schema change.
- Item 8 (memory edit history) — adds a new table, needs migration. Low risk.

**Anti-items (not in this plan, listed for completeness):**
- Offline on-device LLM (#26 from the missing list) — this is a massive lift (llama.cpp JNI bindings or MediaPipe LLM integration). Not in this plan. Separate session.
- Wear OS companion — completely separate app. Not in this plan.
- Tablet-optimized layout — Compose UI works on tablets but isn't optimized. Not in this plan.