# Aura Android Runtime Integrity Hardening Plan

> **For Hermes:** Use `software-development:subagent-driven-development` to implement this plan task-by-task. Re-read every target file and grep for existing wiring before changing it.

**Goal:** Make upgrades safe, prevent duplicate or silent user actions, make provider/security state truthful, and remove the remaining runtime paths that can freeze or silently lose user work.

**Architecture:** This is a vertical hardening pass, not a feature pass. The plan fixes database upgrades first because they can block launch; it then repairs lifecycle ownership (voice, app lock, share delivery), provider truthfulness (MoA and Gemini), and finally persistence/error observability. Existing public APIs and database names remain stable. Room upgrades use explicit additive SQL migrations and device migration tests; Compose fixes use StateFlow collection rather than local one-time snapshots.

**Tech stack:** Kotlin 1.9.24, Compose, Hilt, Room 2.6.1, DataStore, WorkManager, OkHttp, kotlinx-coroutines, Robolectric, Android instrumentation tests.

**Prior plans:** `.hermes/plans/2026-07-09-bug-sweep.md` was executed through v0.10.2. It fixed stale cloud-model identifiers, the Gemini live catalog, and prior model-picker issues. This plan deliberately does not repeat those fixes; it hardens their remaining runtime behavior.

---

## Verified facts and scope

### Confirmed defects to fix

1. **Task DB upgrade crash:** `TaskDatabase` changed v1 to v2 when `ReminderEntity` was added, but `TasksModule` passes `emptyArray()` migrations.
2. **Proactive DB upgrade crash:** `ProactiveEventDatabase` changed v1 to v2 when `payload` was added, but `ProactiveEventModule` passes `emptyArray()` migrations.
3. **Continuous voice collector leak:** Every listen/retry/speak cycle starts a new indefinite StateFlow collector without cancelling the old collector.
4. **Live App Lock gap:** `AuraRoot` snapshots `appLockEnabled` once instead of observing it.
5. **MoA false availability:** `MoaProvider.isConfigured()` checks prefix parseability, not the actual aggregator/reference credentials.
6. **Repeated share intent ignored:** `ConsumeIncomingShare` runs only once, so sharing while Chat is already visible is ignored.
7. **Gemini catalog key appears in URL:** `GeminiProvider.listModels()` uses `?key=` while chat correctly uses `X-Goog-Api-Key`.
8. **Main-thread waits:** `AuraApp` blocks `Application.onCreate`; widget configuration blocks the activity while fetching live catalogs.
9. **Silent persistence failures:** conversation saves are swallowed; retryable errors disappear after five seconds; backup export rethrows after reporting failure.
10. **Migration test coverage:** existing instrumentation coverage only exercises memory v1→v2, not the newest memory or conversation migration.

### Deliberately not included

- **PiP API guard:** minSdk is 26, so the N-vs-O check is redundant but not a runtime API-24/25 bug for this app. Clean it up only opportunistically.
- **Debug release signing / blanket ProGuard keep rules:** personal sideload policy; no security theater in this plan.
- **Full FTS / cross-database transaction redesign / provider circuit breaker:** valid future scale work, but not necessary to make the existing single-user product correct.
- **DNS-rebinding-resistant SSRF transport:** defense-in-depth; retain as a separate security-hardening session because it requires changing the OkHttp resolver/connection boundary and deserves focused tests.

---

## Phase 0 — Lock the migration contracts before changing production code

### Task 0.1: Recover and record exact v1 schemas

**Objective:** Build migrations from the actual historic schemas, not guesses.

**Files:**
- Inspect: `git show 14e30f49^:android/aura-core/src/main/kotlin/com/aura/tasks/TaskEntity.kt`
- Inspect: `git show eb877446^:android/aura-core/src/main/kotlin/com/aura/proactive/ProactiveEventEntity.kt`
- Inspect: `aura-core/schemas/`
- Create: `aura-core/src/androidTest/kotlin/com/aura/tasks/TaskDatabaseMigrationTest.kt`
- Create: `aura-core/src/androidTest/kotlin/com/aura/proactive/ProactiveEventDatabaseMigrationTest.kt`

**Steps:**
1. Confirm that v1 Tasks has only the `tasks` table and that v2 adds the `reminders` table plus Room’s expected identity hash.
2. Confirm that v1 Proactive has `proactive_events` without `payload` and v2 adds `payload TEXT NOT NULL DEFAULT ''`.
3. Do not hand-author schema JSON. Use the historical definitions and let Room’s exported schemas validate final identity.
4. Create failing instrumentation tests with `MigrationTestHelper` that create a v1 DB, run migration 1→2, and validate the v2 schema.

**Expected failure:** tests cannot compile until migrations are defined.

**Commit:** none; fold these tests into Phase 1 and Phase 2 commits.

---

## Phase 1 — Eliminate production upgrade crashes

### Task 1.1: Add TaskDatabase v1→v2 migration

**Objective:** Preserve existing tasks and create the reminders table for every existing install.

**Files:**
- Modify: `aura-core/src/main/kotlin/com/aura/tasks/TasksModule.kt`
- Create: `aura-core/src/androidTest/kotlin/com/aura/tasks/TaskDatabaseMigrationTest.kt`
- Verify generated: `aura-core/schemas/com.aura.tasks.TaskDatabase/2.json`

**Step 1: Write failing migration test**

Create a test with this behavioral contract:
```kotlin
@Test
fun migrate1To2_preservesExistingTasks_andCreatesReminders() {
    val db = helper.createDatabase("tasks-v1.db", 1)
    db.execSQL("INSERT INTO tasks (id, title, description, createdAt, dueAt, completedAt, status, priority, tags) VALUES (...)" )
    db.close()

    helper.runMigrationsAndValidate(
        "tasks-v1.db", 2, true, TasksModule.MIGRATION_1_2,
    )
}
```

Use the *actual v1 table column set* recovered in Phase 0; do not insert v2-only columns into a v1 fixture.

**Step 2: Implement the minimal migration**

In `TasksModule`, add:
```kotlin
val MIGRATION_1_2 = object : Migration(1, 2) {
    override fun migrate(db: SupportSQLiteDatabase) {
        db.execSQL(
            """
            CREATE TABLE IF NOT EXISTS reminders (
                id TEXT NOT NULL PRIMARY KEY,
                message TEXT NOT NULL,
                triggerAt INTEGER NOT NULL,
                createdAt INTEGER NOT NULL,
                taskId TEXT NOT NULL
            )
            """.trimIndent(),
        )
    }
}
```

Verify the SQL against Room’s generated v2 schema before committing. Add `MIGRATION_1_2` to `RoomConfig.builder(... migrations = arrayOf(...))`.

**Step 3: Verify**

Run:
```bash
./gradlew :aura-core:connectedDebugAndroidTest --tests '*TaskDatabaseMigrationTest*'
```
Expected: PASS on a device/emulator.

If no emulator is available locally, build the test APK, record that limitation in the commit, and require this test in CI before declaring the phase shipped.

**Commit:**
```text
fix(tasks): migrate reminders schema without breaking upgrades
```

### Task 1.2: Add ProactiveEventDatabase v1→v2 migration

**Objective:** Preserve proactive event history while adding routing payload support.

**Files:**
- Modify: `aura-core/src/main/kotlin/com/aura/proactive/ProactiveEventModule.kt`
- Create: `aura-core/src/androidTest/kotlin/com/aura/proactive/ProactiveEventDatabaseMigrationTest.kt`
- Verify generated: `aura-core/schemas/com.aura.proactive.ProactiveEventDatabase/2.json`

**Step 1: Write failing migration test**

Create v1 database, insert an event row with the pre-payload columns, run migration, then query the migrated row and assert:
- original id/type/title/body/timestamp survive;
- `payload == ""`;
- the v2 schema validates.

**Step 2: Implement migration**

Add:
```kotlin
val MIGRATION_1_2 = object : Migration(1, 2) {
    override fun migrate(db: SupportSQLiteDatabase) {
        db.execSQL(
            "ALTER TABLE proactive_events ADD COLUMN payload TEXT NOT NULL DEFAULT ''",
        )
    }
}
```

Pass it through `RoomConfig.builder` rather than using `emptyArray()`.

**Step 3: Verify and commit**

Run the targeted connected test, then:
```bash
./gradlew :aura-core:testDebugUnitTest :aura-core:assembleDebug
```

**Commit:**
```text
fix(proactive): migrate event payload schema safely
```

### Task 1.3: Complete existing migration coverage

**Objective:** Prove all persisted schema transitions that have already shipped.

**Files:**
- Modify: `aura-core/src/androidTest/kotlin/com/aura/memory/MemoryDatabaseMigrationTest.kt`
- Create: `aura-core/src/androidTest/kotlin/com/aura/agent/ConversationDatabaseMigrationTest.kt`
- Inspect: `aura-core/src/main/kotlin/com/aura/memory/MemoryModule.kt`
- Inspect: `aura-core/src/main/kotlin/com/aura/agent/ConversationModule.kt`

**Steps:**
1. Extend memory tests with direct v2→v3 validation for `memory_edits` and its index/FK contract.
2. Add conversation v1→v2 validation for nullable `embedding BLOB` and a preserved existing conversation row.
3. Add a chained memory v1→v3 test using both migrations in order.
4. Keep fixtures minimal and data-bearing: schema-only migration tests miss bad defaults and lost rows.

**Commit:**
```text
 test(room): cover every shipped database upgrade path
```

---

## Phase 2 — Repair lifecycle ownership and state truthfulness

### Task 2.1: Make continuous voice a single-owner state machine

**Objective:** Ensure one STT collector, one TTS collector, and one wait-for-response job can exist at a time.

**Files:**
- Modify: `app/src/main/kotlin/com/aura/ui/voice/ContinuousVoiceViewModel.kt`
- Create/modify: `app/src/test/kotlin/com/aura/ui/voice/ContinuousVoiceViewModelTest.kt`

**Design:**
- Add `sttCollectionJob`, `ttsCollectionJob`, and `responseWaitJob` properties.
- At the beginning of `startListening`, cancel and replace the previous STT job; do not recursively stack `collect` jobs.
- At the beginning of `speakResponse`, cancel and replace the previous TTS job.
- In `stopLoop()` and `onCleared()`, cancel all three jobs, cancel STT, and stop TTS.
- Guard event callbacks by both `state.active` and the expected phase. A stale collector must not send a message after the phase changed.

**Tests:**
1. Repeated `startListening()` leaves only one active final-result observer.
2. An STT error followed by retry does not duplicate `onSend` for one final result.
3. Stop before a pending response completes does not invoke `speakResponse`.
4. Repeated TTS-ready events only restart listening once.

Use test-owned `MutableStateFlow`s and `runTest`; do not depend on real recognizer/TTS frameworks.

**Commit:**
```text
fix(voice): cancel stale continuous-mode collectors
```

### Task 2.2: Make App Lock reactive and immediately enforceable

**Objective:** Enabling lock should lock the active app session; disabling it should release the gate only after its preference update is observed.

**Files:**
- Modify: `app/src/main/kotlin/com/aura/MainActivity.kt`
- Create/modify: `app/src/test/kotlin/com/aura/ui/settings/SettingsViewModelAppLockTest.kt`
- Create: focused test for extracted lock reducer/state helper if needed.

**Design:**
1. Replace `appLockEnabled = userPreferences.appLockEnabled.first()` with collection of `appLockEnabled` via `collectAsState` / lifecycle-aware state flow in the root.
2. When it transitions false→true, set `unlocked = false` immediately.
3. When lock remains enabled, retain existing `ON_RESUME` relock behavior.
4. Do not fake `FragmentActivity` authentication in a unit test. Extract and test only the state transition rule; leave BiometricPrompt framework invocation to device verification.

**Tests:**
- preference false→true produces locked state;
- true→false does not leave the UI permanently blocked;
- resume while lock enabled produces locked state.

**Commit:**
```text
fix(lock): enforce app lock changes in the active session
```

### Task 2.3: Deliver all incoming shares, not just the first one

**Objective:** Text/image shares received while the chat is already composed should always update Chat.

**Files:**
- Modify: `app/src/main/kotlin/com/aura/IncomingShareStore.kt`
- Modify: `app/src/main/kotlin/com/aura/ui/screens/ChatScreen.kt`
- Create: `app/src/test/kotlin/com/aura/IncomingShareStoreTest.kt`
- Extend: relevant Chat/Robolectric test file.

**Design:**
1. Replace read-and-clear convenience calls with an event/value stream that represents each new share distinctly. Do not use a simple nullable StateFlow alone, because sending the same text twice must still produce two events.
2. Use a small `SharedFlow<IncomingShare>` (buffered) or monotonically identified `StateFlow` event. Model text/image as a sealed payload so one collector processes both paths consistently.
3. In `ConsumeIncomingShare`, acquire the store once with `remember`, then `LaunchedEffect(store)` collects events for the composable lifetime.
4. Decode images on `Dispatchers.IO`; keep the 1024px cap; surface a lightweight error if decode fails rather than silently doing nothing.

**Tests:**
- two identical text shares both reach the consumer;
- a text share after the initial composition updates the draft;
- image event produces one vision staging call;
- no stale event is replayed after collection restarts.

**Commit:**
```text
fix(share): handle new share intents while chat is open
```

---

## Phase 3 — Make provider behavior truthful and credentials private

### Task 3.1: Validate the complete MoA preset before showing it

**Objective:** Do not expose `moa:default` unless its aggregator and every enabled reference provider are usable.

**Files:**
- Modify: `aura-core/src/main/kotlin/com/aura/providers/MoaProvider.kt`
- Extend: `aura-core/src/test/kotlin/com/aura/providers/MoaProviderTest.kt`
- Possibly modify: `app/src/main/kotlin/com/aura/ui/viewmodel/ChatViewModel.kt`

**Design:**
1. Add a single private preset-validation helper in `MoaProvider`.
2. Resolve each `ModelRef` through `ProviderRegistry`; require `Provider.isConfigured()` for aggregator and all required references.
3. If a configured provider exposes live catalogs cheaply/cached, validate model membership there; otherwise preserve current catalog validation as a send-time error to avoid network I/O in composable/model-picker paths.
4. `isConfigured()` returns false when credentials are missing. `listModels()` returns enabled presets only when valid, or an empty list.
5. Optionally expose a structured unavailability reason to Settings/ModelPicker rather than silently hiding MoA.

**Tests:**
- only Ollama key, no DeepSeek aggregator key → `isConfigured() == false`;
- aggregator key but absent reference key → false;
- all provider credentials present → true;
- an invalid/disabled preset is never listed.

**Commit:**
```text
fix(moa): expose presets only when every required provider is ready
```

### Task 3.2: Remove Gemini API keys from catalog URLs

**Objective:** Use the same header authentication path for chat and model discovery.

**Files:**
- Modify: `aura-core/src/main/kotlin/com/aura/providers/GeminiProvider.kt`
- Extend: `aura-core/src/test/kotlin/com/aura/providers/GeminiProviderTest.kt`

**Steps:**
1. Write a MockWebServer test asserting `listModels()` sends `X-Goog-Api-Key` and request path is `/v1beta/models` without `key=`.
2. Change the request builder accordingly.
3. Preserve fallback behavior for unreachable/non-2xx calls.

**Commit:**
```text
fix(gemini): authenticate model discovery with request headers
```

---

## Phase 4 — Remove blocking UI paths and silent data loss

### Task 4.1: Stop blocking Application startup on DataStore

**Objective:** Keep `Application.onCreate()` non-blocking while preserving a correct first-chat/provider state.

**Files:**
- Modify: `app/src/main/kotlin/com/aura/AuraApp.kt`
- Inspect/modify only if needed: `app/src/main/kotlin/com/aura/MainActivity.kt`, `app/src/main/kotlin/com/aura/ui/viewmodel/ChatViewModel.kt`
- Extend: provider/startup state tests.

**Design:**
1. Remove `runBlocking { providerKeys.awaitLoaded() }`.
2. Start proactive bootstrap without assuming the provider-key cache is immediately ready, or make bootstrap await it in its existing IO scope.
3. Keep first UI render responsive. Existing `ChatViewModel.refreshModels()` already waits on `providerKeys.loaded`; preserve this contract.
4. No arbitrary timeout. If decryption/load fails, expose a recoverable configuration error rather than freezing startup.

**Verification:** cold-start test with a delayed fake ProviderKeys; UI must render loading/onboarding without blocking the test main thread.

**Commit:**
```text
fix(startup): load provider credentials without blocking app creation
```

### Task 4.2: Make widget configuration asynchronous and failure-visible

**Objective:** Never fetch live provider catalogs in `WidgetConfigActivity.onCreate()` on the main thread.

**Files:**
- Modify: `app/src/main/kotlin/com/aura/widget/WidgetConfigActivity.kt`
- Create/modify: widget config ViewModel/state test or extract a pure catalog-loader helper.

**Design:**
1. Set Compose content immediately with `loading = true`.
2. Launch catalog/default-model loading in `lifecycleScope` on IO.
3. Populate the form after loading; show exact provider catalog failures and allow retry.
4. If no model resolves, disable Save and direct the user to Settings rather than storing an invalid model string.
5. Preserve per-widget SharedPreferences behavior.

**Commit:**
```text
fix(widget): load provider models off the main thread
```

### Task 4.3: Make failures visible and actionable

**Objective:** A user must never silently lose a save or lose a retry affordance.

**Files:**
- Modify: `app/src/main/kotlin/com/aura/ui/viewmodel/ChatViewModel.kt`
- Modify: `app/src/main/kotlin/com/aura/ui/settings/BackupViewModel.kt`
- Modify: `app/src/main/kotlin/com/aura/ui/screens/ChatScreen.kt` only if new save-warning UI is required
- Extend: `app/src/test/kotlin/com/aura/ui/viewmodel/ChatViewModelTest.kt`
- Extend: `app/src/test/kotlin/com/aura/ui/settings/BackupViewModelTest.kt`

**Design:**
1. Replace `saveConversation()`’s silent `runCatching` with error handling that records a non-blocking persistence warning in state. De-duplicate warnings so every autosave failure does not spam the UI.
2. Keep retryable provider errors visible until user retries, sends a new message, or explicitly dismisses them. Only transient non-retryable status errors may auto-dismiss.
3. Change `prepareExportFile()` to return `Result<File>` or nullable `File` after it updates state. Do not throw from a UI-launched coroutine after setting an error state.
4. Do not convert failures into false success messages.

**Tests:**
- failed conversation save results in a visible warning state;
- retryable error survives the five-second delay;
- non-retryable transient error can still auto-dismiss;
- failed export produces error state and does not throw from caller coroutine.

**Commit:**
```text
fix(data): surface save failures and preserve recovery actions
```

---

## Phase 5 — Preserve agent-context consistency for multimodal turns

### Task 5.1: Give image/audio actions a durable conversation contract

**Objective:** Vision and audio actions should not become invisible dead-end assistant output.

**Files:**
- Inspect/modify: `app/src/main/kotlin/com/aura/ui/viewmodel/ChatViewModel.kt` (`runVisionPrompt`, audio path)
- Inspect/modify: `app/src/main/kotlin/com/aura/ui/viewmodel/ChatSendController.kt`
- Inspect/modify: `aura-core/src/main/kotlin/com/aura/agent/MemoryAugmentedAgenticLoop.kt`
- Extend relevant ChatViewModel / agent loop tests.

**Design decision to make before implementation:**

Use one consistent conversation representation:
- Add a synthetic user turn such as `"[Image analysis request] Describe this image"` / `"[Audio transcription request]"` before the tool output.
- Persist the tool result as a normal `ToolTurn`.
- Route the next language synthesis through the agent loop so regular memory, profile, KG, citation, cancellation, and error contracts apply.
- Keep raw image bytes transient; do not put base64 into Room or memory.

**Acceptance criteria:**
- vision/audio requests appear coherently in history;
- incognito still prevents all persistent learning;
- a failed multimodal request shows the same retry/error contract as chat;
- no image/audio bytes are persisted accidentally.

**Commit:**
```text
fix(multimodal): preserve context for vision and audio turns
```

---

## Phase 6 — Cleanup, test gates, documentation, and release verification

### Task 6.1: Delete or consolidate orphaned Hands builder code

**Objective:** Eliminate two divergent hand JSON parsers.

**Files:**
- Inspect: `app/src/main/kotlin/com/aura/ui/components/HandStepBuilder.kt`
- Inspect: `app/src/main/kotlin/com/aura/ui/screens/HandsScreen.kt`
- Test: hand JSON round-trip unit test.

**Steps:**
1. Confirm `HandStepBuilder` remains unimported.
2. Either delete it or extract its JSON codec into one shared pure `HandStepCodec` used by the live add/edit dialogs.
3. Add round-trip tests covering quote/backslash values, empty args, and malformed stored JSON.
4. Do not change hand schema or add a Room migration.

**Commit:**
```text
refactor(hands): use one tested hand-step codec
```

### Task 6.2: Update release facts and validation instructions

**Files:**
- Modify: `README.md`
- Modify: `.github/workflows/ci.yml` if emulator CI is feasible in the existing environment
- Modify only if necessary: `docs/ANDROID_TEST_PLAN.md`

**Steps:**
1. Update v0.10.2/versionCode 3 and actual test count after final gate.
2. Document migration verification as a release blocker.
3. Add an Android emulator CI job for migration tests if practical. If CI cost/runner setup is unacceptable, retain targeted local connected-test instructions and explicitly flag this as a release checklist item—not a fake CI claim.
4. Do not describe a visual test as completed unless a device/emulator screenshot exists.

**Commit:**
```text
 docs(android): document verified release and migration gates
```

---

## Final verification protocol

### Per-commit

Run targeted tests for the changed subsystem. Examples:
```bash
./gradlew :aura-core:testDebugUnitTest --tests '*MoaProviderTest*'
./gradlew :app:testDebugUnitTest --tests '*ContinuousVoiceViewModelTest*'
./gradlew :app:testDebugUnitTest --tests '*ChatViewModelTest*'
```

### Before the final summary

Run the project-native full gates:
```bash
./gradlew :aura-core:testDebugUnitTest :app:testDebugUnitTest :app:assembleDebug :app:lintDebug --rerun-tasks
```

Then, with an Android emulator/device:
```bash
./gradlew :aura-core:connectedDebugAndroidTest
adb install -r app/build/outputs/apk/debug/app-debug.apk
```

Manual device script:
1. Upgrade an installed v1 fixture/build to the new APK and verify Tasks and Proactive History open with old rows intact.
2. Enable App Lock while Aura is running; background and resume; confirm biometric gate appears.
3. Configure only Ollama; confirm MoA is unavailable. Add every required key; confirm MoA becomes available.
4. Share text twice while Chat is already visible; confirm both arrive. Repeat with an image.
5. Induce two STT retries; speak one final phrase; confirm exactly one chat send occurs.
6. Open widget configuration on slow/offline network; confirm immediate loading UI, not a frozen activity.
7. Force a provider error; confirm retry stays visible until dismissed/retried.
8. Confirm Gemini model discovery request carries no `key=` URL parameter through a MockWebServer test.

### Ship criteria

- All fresh unit tests pass.
- All Room migration instrumentation tests pass.
- `assembleDebug` and `lintDebug` pass; warnings are reviewed rather than hidden.
- No `git diff --check` violations.
- Working tree contains only intentional plan/docs changes before implementation begins.
- Before calling UI work complete, capture an emulator/device screenshot for the affected visible flows.

## Commit sequence

| # | Commit | Scope |
|---|---|---|
| 1 | `fix(tasks): migrate reminders schema without breaking upgrades` | P0 Tasks upgrade path |
| 2 | `fix(proactive): migrate event payload schema safely` | P0 Proactive upgrade path |
| 3 | `test(room): cover every shipped database upgrade path` | Memory + conversation migration evidence |
| 4 | `fix(voice): cancel stale continuous-mode collectors` | duplicate sends / loop collapse |
| 5 | `fix(lock): enforce app lock changes in the active session` | live privacy state |
| 6 | `fix(share): handle new share intents while chat is open` | active-chat share path |
| 7 | `fix(moa): expose presets only when every required provider is ready` | model-picker truthfulness |
| 8 | `fix(gemini): authenticate model discovery with request headers` | credential privacy |
| 9 | `fix(startup): load provider credentials without blocking app creation` | launch resilience |
| 10 | `fix(widget): load provider models off the main thread` | widget resilience |
| 11 | `fix(data): surface save failures and preserve recovery actions` | persistence / retry honesty |
| 12 | `fix(multimodal): preserve context for vision and audio turns` | memory/history consistency |
| 13 | `refactor(hands): use one tested hand-step codec` | orphan-code removal |
| 14 | `docs(android): document verified release and migration gates` | accurate release contract |

## Risks and controls

- **Room SQL schema mismatch:** avoid guessing; recover v1 schemas first, validate with `MigrationTestHelper`, and compare exported v2 JSON.
- **Voice regression from cancellation:** add deterministic StateFlow tests before implementation; verify on real microphone hardware afterward.
- **Live App Lock deadlock:** do not lock from a one-shot callback without state observation; test state reducer separately from BiometricPrompt.
- **MoA catalog network cost:** credential validation is local; defer remote catalog membership checks to existing catalog refresh/send behavior unless cached.
- **Multimodal scope creep:** do not store bitmap/audio payloads. Persist request/tool-result text only.
- **No emulator on this host:** do not fake connected-test or visual verification. Provision an emulator or use a physical Android device before release.
