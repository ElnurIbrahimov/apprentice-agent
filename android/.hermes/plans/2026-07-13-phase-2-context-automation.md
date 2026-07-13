# Aura Android Phase 2 — Context and Automation Completion Plan

> **For Hermes:** Execute immediately using strict TDD. Verify each capability against current source before adding it; existing step-building and trigger-phrase wiring must be extended, not rebuilt.

**Goal:** Turn Aura's existing knowledge graph, local crash log, conversation history, and Hands macros into complete daily-use systems: manageable, inspectable, durable, and automatable.

**Architecture:** Keep each feature behind one domain boundary. Knowledge graph management extends `KnowledgeGraphRepository`; diagnostics extend `CrashLogger`; rolling compaction is a new `ConversationCompactor` called by the agentic loop; Hands runtime extends the existing Room database/repository/WorkManager worker. UI surfaces consume ViewModels and never call DAOs directly.

**Tech stack:** Kotlin 1.9.24, Coroutines/Flow, Hilt, Room 2.6.1, WorkManager, Jetpack Compose Material 3, MockK, Android instrumentation migration tests.

**Constraints:**
- Personal-use sideload app; no Play Store/distribution work.
- Never hardcode provider/model IDs. Compaction uses the active configured model and degrades safely when unavailable.
- Preserve complete conversation turns for history/export; compaction changes only model context.
- API keys and embeddings remain excluded from backups.
- Existing Graph tab removal remains correct: management is a secondary Memory surface, not a fifth top-level tab.
- Existing Hands step builder and trigger-phrase loop are already shipped; do not duplicate them.
- Every behavior change follows RED → GREEN → REFACTOR and receives an atomic commit.

---

## Phase 2.0 — Baseline and verified scope

### Task 1: Establish a clean baseline

**Files:** none

1. Verify branch and tracked working tree.
2. Run `./gradlew :aura-core:testDebugUnitTest :app:testDebugUnitTest`.
3. Confirm existing Phase 1 commits and preserve user-owned untracked files.
4. Verify each Phase 2 claim by reading current code.

**Expected:** Existing tests pass. Existing plumbing confirmed:
- KG extraction/repository/query exist; management UI does not.
- `CrashLogger` writes a local rolling text file; diagnostics UI and global crash capture do not exist.
- `Conversation.toMessages()` drops all turns older than 40; no durable summary exists.
- Hands support CRUD, inline step builder, trigger phrases, manual execution, and a worker; history, schedules, variables, and conditions do not exist.

---

## Phase 2.1 — Knowledge-graph management surface

### Task 2: Add safe KG mutation contracts

**Files:**
- Modify: `aura-core/src/test/kotlin/com/aura/kg/KnowledgeGraphRepositoryTest.kt`
- Modify: `aura-core/src/main/kotlin/com/aura/kg/KnowledgeGraphDao.kt`
- Modify: `aura-core/src/main/kotlin/com/aura/kg/KnowledgeGraphRepository.kt`

**RED:** Add tests for:
- Updating label/type/properties while preserving stable node ID and provenance.
- Merging source node into target: merged properties, rewritten incoming/outgoing edges, no self-edge, source removed.
- Rejecting source==target and missing nodes.

**GREEN:** Add DAO update/delete-edge methods and repository `updateNode`/`mergeNodes`. Keep operations behind the existing repository mutex. Generate rewritten edge IDs through `KgId.edge`.

**Verify:** `./gradlew :aura-core:testDebugUnitTest --tests 'com.aura.kg.KnowledgeGraphRepositoryTest'`

**Commit:** `feat(knowledge): add safe node editing and duplicate merging`

### Task 3: Add KG management ViewModel

**Files:**
- Create: `app/src/main/kotlin/com/aura/ui/viewmodel/KnowledgeGraphViewModel.kt`
- Create: `app/src/test/kotlin/com/aura/ui/viewmodel/KnowledgeGraphViewModelTest.kt`

**RED:** Pin initial load, text/type filters, node selection with neighbors, edit refresh, merge refresh, delete confirmation state, and visible errors.

**GREEN:** Build immutable `KnowledgeGraphUiState`; load max 500 personal-use nodes, stats, and neighbor details through the repository. Debounce search by 250 ms.

**Verify:** targeted app unit test.

### Task 4: Ship native KG management UI and navigation

**Files:**
- Create: `app/src/main/kotlin/com/aura/ui/screens/KnowledgeGraphScreen.kt`
- Modify: `app/src/main/kotlin/com/aura/ui/screens/MemoryScreen.kt`
- Modify: `app/src/main/kotlin/com/aura/ui/nav/NavGraph.kt`

**Acceptance:**
- Memory screen exposes a visible “Knowledge graph” action with node/edge intent.
- Screen shows node/edge counts, search, type chips, empty/error/loading states.
- Node detail shows properties plus resolved incoming/outgoing relation labels.
- Edit supports label, type, and JSON properties validation.
- Delete is confirmed and removes incident edges.
- Merge selects a target and previews the destructive action.
- Bottom navigation remains visible; system back works.

**Verify:** compile, targeted tests, and emulator screenshots for empty and populated/detail states.

**Commit:** `feat(knowledge): add complete graph management surface`

---

## Phase 2.2 — Crash diagnostics and export

### Task 5: Make crash logs structured and inspectable

**Files:**
- Create: `aura-core/src/test/kotlin/com/aura/core/error/CrashLoggerTest.kt`
- Modify: `aura-core/src/main/kotlin/com/aura/core/error/CrashLogger.kt`

**RED:** Pin:
- Parsing legacy multiline entries.
- New structured entries round-trip code/message/stack/time/thread.
- Newest-first history.
- Clear behavior.
- Rolling truncation preserves entry boundaries.
- Export copy returns the exact current log.

**GREEN:** Add `CrashLogEntry`, `entries()`, `exportTo(cacheDir)`, and JSON-lines writes while retaining legacy parser support. Continue best-effort no-throw semantics.

**Commit:** `feat(diagnostics): expose structured local crash history`

### Task 6: Capture process crashes without swallowing them

**Files:**
- Create: `aura-core/src/main/kotlin/com/aura/core/error/CrashHandler.kt`
- Create: `aura-core/src/test/kotlin/com/aura/core/error/CrashHandlerTest.kt`
- Modify: `app/src/main/kotlin/com/aura/AuraApp.kt`

**RED:** Verify uncaught exception is logged once and delegated to the previous handler. Avoid self-delegation.

**GREEN:** Install handler during `AuraApp.onCreate()` via injected `CrashLogger`; preserve Android's prior handler so crash semantics remain intact.

### Task 7: Add Diagnostics screen, sharing, and clear controls

**Files:**
- Create: `app/src/main/kotlin/com/aura/ui/viewmodel/DiagnosticsViewModel.kt`
- Create: `app/src/test/kotlin/com/aura/ui/viewmodel/DiagnosticsViewModelTest.kt`
- Create: `app/src/main/kotlin/com/aura/ui/screens/DiagnosticsScreen.kt`
- Modify: `app/src/main/kotlin/com/aura/ui/screens/SettingsScreen.kt`
- Modify: `app/src/main/kotlin/com/aura/ui/nav/NavGraph.kt`

**Acceptance:**
- Settings includes Diagnostics with entry count and “Open”.
- Screen renders newest first with code, timestamp, message, thread, expandable stack.
- Share exports a local text/JSONL file through existing `FileProvider`.
- Clear-all requires confirmation and updates immediately.
- Empty state explains logs never leave the device unless explicitly shared.

**Verify:** targeted unit tests, compile, and emulator screenshot.

**Commit:** `feat(diagnostics): add local crash viewer export and clear controls`

---

## Phase 2.3 — Rolling conversation compaction

### Task 8: Add durable summary state to the conversation model

**Files:**
- Modify: `aura-core/src/test/kotlin/com/aura/agent/ConversationStoreTest.kt`
- Create/modify: conversation model tests near `ConversationStoreTest.kt`
- Modify: `aura-core/src/main/kotlin/com/aura/agent/Conversation.kt`
- Modify: `aura-core/src/main/kotlin/com/aura/agent/ConversationEntity.kt`
- Modify: `aura-core/src/main/kotlin/com/aura/agent/ConversationStore.kt`

**RED:** Verify:
- Summary is inserted before unsummarized recent messages.
- Turns already covered by `summaryThroughTurn` never re-enter the model context.
- Full `turns` remain unchanged for history.
- Save/load round-trips summary fields.
- Fork retains only valid summary coverage; forking before the boundary clears summary.

**GREEN:** Add `contextSummary: String` and `summaryThroughTurn: Int` to model/entity. Update `toMessages()` to add a labelled system summary and only unsummarized recent turns.

### Task 9: Add Conversation DB v3→v4 migration

**Files:**
- Modify: `aura-core/src/main/kotlin/com/aura/agent/ConversationDatabase.kt`
- Modify: `aura-core/src/main/kotlin/com/aura/agent/ConversationModule.kt`
- Modify: `aura-core/src/androidTest/kotlin/com/aura/agent/ConversationDatabaseMigrationTest.kt`
- Generate: `aura-core/schemas/com.aura.agent.ConversationDatabase/4.json`

**RED:** Migration test inserts v3 row and verifies both summary columns/defaults plus prior index.

**GREEN:** Add non-null text/int columns with safe defaults and register migration.

**Commit:** `feat(conversations): persist rolling context summary state`

### Task 10: Build the compactor with active-model routing

**Files:**
- Create: `aura-core/src/main/kotlin/com/aura/agent/ConversationCompactor.kt`
- Create: `aura-core/src/test/kotlin/com/aura/agent/ConversationCompactorTest.kt`

**Behavior:**
- Trigger only when unsummarized history exceeds 40 turns.
- Summarize the oldest batch while retaining at least 20 verbatim recent turns.
- Fold the previous summary into the next one.
- Use the active model passed to the loop; never hardcode an ID.
- No tool definitions; low temperature; bounded output.
- On provider failure return the original conversation and let the main chat continue.
- Re-throw `CancellationException`.
- Reject blank summary output.

**RED/GREEN:** Use a mocked provider stream and assert exact boundary advancement, prior-summary inclusion, no-op threshold, failure fallback, and cancellation propagation.

### Task 11: Wire compaction into the agent loop and backup

**Files:**
- Modify: `aura-core/src/main/kotlin/com/aura/agent/MemoryAugmentedAgenticLoop.kt`
- Modify: loop tests/constructor call sites
- Modify: `aura-core/src/main/kotlin/com/aura/backup/AuraBackup.kt`
- Modify: `aura-core/src/main/kotlin/com/aura/backup/BackupManager.kt`
- Modify: `aura-core/src/test/kotlin/com/aura/backup/BackupManagerTest.kt`

**Acceptance:** Compaction occurs once before the first model step, its state reaches `AgentEvent.Result`, normal save persists it, and backup/restore preserves it. Increment backup schema once for all Phase 2 additions.

**Commit:** `feat(conversations): compact old context without losing history`

---

## Phase 2.4 — Hands runtime completion

### Task 12: Define Hand schedule/variable/condition/run-history schema

**Files:**
- Modify: `aura-core/src/main/kotlin/com/aura/hands/Hand.kt`
- Create: `aura-core/src/main/kotlin/com/aura/hands/HandRun.kt`
- Modify: `aura-core/src/main/kotlin/com/aura/hands/HandDao.kt`
- Modify: `aura-core/src/main/kotlin/com/aura/hands/HandDatabase.kt`
- Modify: `aura-core/src/main/kotlin/com/aura/hands/HandsModule.kt`
- Create: `aura-core/src/androidTest/kotlin/com/aura/hands/HandDatabaseMigrationTest.kt`
- Generate: `aura-core/schemas/com.aura.hands.HandDatabase/2.json`

**Fields:**
- `variablesJson` map of variable name → default.
- `conditionsJson` array of typed conditions.
- `scheduleType`: none/daily/weekdays/weekly.
- `scheduleHour`, `scheduleMinute`, `scheduleDayOfWeek`.
- `updatedAt`.
- `HandRun`: run ID, hand ID/name, trigger, status, timestamps, output/error, failed step, resolved variables.

**RED:** v1→v2 migration preserves Hands, adds safe defaults, creates indexed run table.

**Commit:** `feat(hands): persist automation configuration and run history`

### Task 13: Implement variables and conditions as pure contracts

**Files:**
- Create: `aura-core/src/main/kotlin/com/aura/hands/HandRuntime.kt`
- Create: `aura-core/src/test/kotlin/com/aura/hands/HandRuntimeTest.kt`

**RED:** Test:
- Default + runtime variable merge.
- `{{variable}}` substitution across all step arguments.
- Unknown-variable error.
- equals/not-equals/contains/not-contains/not-empty/empty conditions.
- Numeric greater/less comparisons with invalid-number failure.
- All conditions must pass; blank list passes.

**GREEN:** Pure serializable models and deterministic evaluator—no Android dependencies or LLM calls.

### Task 14: Record every Hand execution

**Files:**
- Modify: `aura-core/src/test/kotlin/com/aura/hands/HandRepositoryTest.kt`
- Modify: `aura-core/src/main/kotlin/com/aura/hands/HandRepository.kt`
- Modify: `aura-core/src/main/kotlin/com/aura/tools/RunHandTool.kt`
- Modify: `aura-core/src/main/kotlin/com/aura/hands/RunHandWorker.kt`

**RED:** Verify running→success/failure transitions, failed-step index, condition-blocked status, resolved variables, manual/agent/phrase/schedule trigger source, and no secret/unbounded tool output beyond the existing result cap.

**GREEN:** Make repository the sole execution boundary. UI/tool/worker pass trigger and variables; repository owns history writes.

**Commit:** `feat(hands): add variables conditions and durable run history`

### Task 15: Implement reliable schedules

**Files:**
- Create: `aura-core/src/main/kotlin/com/aura/hands/HandSchedule.kt`
- Create: `aura-core/src/main/kotlin/com/aura/hands/HandScheduler.kt`
- Create: `aura-core/src/test/kotlin/com/aura/hands/HandScheduleTest.kt`
- Modify: `aura-core/src/main/kotlin/com/aura/hands/RunHandWorker.kt`
- Modify: relevant Hilt modules

**Behavior:**
- Compute next local run for daily, weekdays, or selected weekly day.
- Enqueue one-time unique work `hand-schedule-<id>` and rotate request identity.
- Editing/toggling/delete immediately reschedules/cancels.
- Worker looks up by stable hand ID, records scheduled trigger, and schedules the next occurrence after terminal completion.
- Disabled/deleted hands do not run.

**Commit:** `feat(hands): schedule recurring automations reliably`

### Task 16: Complete Hands UI

**Files:**
- Create: `app/src/main/kotlin/com/aura/ui/screens/HandEditorDialog.kt`
- Create: `app/src/main/kotlin/com/aura/ui/screens/HandRunHistoryScreen.kt`
- Modify: `app/src/main/kotlin/com/aura/ui/screens/HandsScreen.kt`
- Modify: `app/src/main/kotlin/com/aura/ui/viewmodel/HandsViewModel.kt`
- Modify/create ViewModel tests
- Modify: `app/src/main/kotlin/com/aura/ui/nav/NavGraph.kt`

**Acceptance:**
- Extract duplicated Add/Edit dialog into one editor.
- Editor supports step builder, variable defaults, conditions, schedule type/time/day.
- Run action prompts for unresolved runtime variables.
- Hand cards show enabled/schedule/last-run status.
- History supports All/Success/Failed filters, run detail, and clear history.
- Destructive delete remains confirmed.

**Verify:** targeted tests, compile, emulator screenshots of list/editor/history.

**Commit:** `feat(hands): add scheduling variables conditions and run history UI`

### Task 17: Extend backup schema for Phase 2 Hands

**Files:**
- Modify: `aura-core/src/main/kotlin/com/aura/backup/AuraBackup.kt`
- Modify: `aura-core/src/main/kotlin/com/aura/backup/BackupManager.kt`
- Modify: backup tests

**Acceptance:** Backups include new Hand configuration and run history. Restore re-creates schedules with fresh WorkManager request IDs; historical runs remain historical. Older schema-v2 backups decode with defaults.

**Commit:** `feat(backup): preserve Phase 2 context and automation state`

---

## Phase 2.5 — Verification and delivery

### Task 18: Full verification gate

1. Run fresh unit suites:
   `./gradlew :aura-core:testDebugUnitTest :app:testDebugUnitTest --rerun-tasks`
2. Run device migrations on emulator:
   `./gradlew :aura-core:connectedDebugAndroidTest :app:connectedDebugAndroidTest --rerun-tasks`
3. Run `./gradlew :app:lintDebug :app:assembleDebug`.
4. Parse XML results for exact tests/failures.
5. Install APK on emulator and visually exercise:
   - Memory → Knowledge graph → detail/edit/merge/delete.
   - Settings → Diagnostics → expand/share/clear.
   - Long-conversation summary indication/history preservation.
   - Hands editor, schedule, variables, conditions, manual run, history.
6. Check logcat for runtime crashes.
7. Run `git diff --check`; preserve user-owned untracked files.
8. Push `feat/tier-1-friction`.
9. Watch GitHub CI to final job-level success; fix and repeat if red.

**Final evidence:** commit table, test totals, migration results, lint/build status, visual screenshots, CI URL, remote SHA, APK path/hash, and explicit list of anything not shipped.
