# Aura Android — Round 3 Real Bugs

> **For Hermes:** Execute task-by-task, verify per commit, no "should I continue?" between items.

**Goal:** Close 10 distinct bugs that affect daily use of the app. Each task ships a single atomic commit.

---

### Task 1: Test coverage for what we shipped

**Files:**
- `aura-core/src/test/kotlin/com/aura/agent/ConversationStoreTest.kt` — round-trip save/load, mostRecent
- `aura-core/src/test/kotlin/com/aura/memory/LikeEscapingTest.kt` — escapeLikeWildcards + ESCAPE clause alignment
- `app/src/test/kotlin/com/aura/ui/viewmodel/HistoryViewModelTest.kt` — load, delete, mostRecent flows
- `app/src/test/kotlin/com/aura/ui/viewmodel/TasksViewModelTest.kt` — load + delete

**Key cases:**
- ConversationStore round-trip: save then load returns identical Conversation
- escapeLikeWildcards: `100%` → `100\%`; `_` → `\_`; `\` → `\\`
- LIKE injection: `100% off` doesn't match memory containing `100 off`
- HistoryViewModel.delete removes from state list
- TasksViewModel propagates to DAO

**Verify:** `./gradlew :aura-core:testDebugUnitTest :app:testDebugUnitTest` shows +20 tests, all green.

---

### Task 2: PermissionGranted event for real tool retry

**Problem:** The `retryAfterPermission` hack injects a fake user turn. Model has to re-interpret it, may re-do everything from scratch.

**Fix:**
- Add `PermissionGranted(val permission: String, val toolName: String, val toolArgs: String)` to AgentEvent
- MemoryAugmentedAgenticLoop: on PermissionGranted, re-execute the last failed tool call with the same arguments. Don't re-run the whole model step.
- ChatViewModel.retryAfterPermission: emit PermissionGranted with the last failed tool call info, not a user turn
- ChatScreen: dialog now emits structured retry, model doesn't see the system message

**Verify:** Manual test in app: trigger a tool that returns NeedsPermission, grant, observe the same tool called again with same args, model response is the actual answer not a meta-retry.

---

### Task 3: HandsScreen — "Run" + "Add" + "Edit" affordances

**Files:**
- `app/src/main/kotlin/com/aura/ui/screens/HandsScreen.kt` — add per-row Run button + FAB for Add
- `app/src/main/kotlin/com/aura/ui/viewmodel/HandsViewModel.kt` — add `add(Hand)`, `update(Hand)`, `runHand(name)`
- `app/src/main/kotlin/com/aura/ui/components/AddHandDialog.kt` — name + trigger phrase + steps JSON editor

**Key cases:**
- Run button calls `HandRepository.run()` via ToolExecutor
- Add dialog validates JSON steps, persists
- Edit reuses dialog with pre-filled values

**Verify:** Manual — add a hand, run it, see the steps execute.

---

### Task 4: TasksScreen — "Mark done" + "Add" affordances

**Files:**
- `app/src/main/kotlin/com/aura/ui/screens/TasksScreen.kt` — add Mark Done + Add buttons
- `app/src/main/kotlin/com/aura/ui/viewmodel/TasksViewModel.kt` — add `markDone(id)`, `addTask(title)`

**Key cases:**
- Mark Done calls `taskDao.markComplete(id)`
- Add dialog just title field (description optional, default = title)
- Marked tasks visually distinguished (line-through + faded)

**Verify:** Manual — mark a task done, observe visual change + DAO update.

---

### Task 5: Persist ProactiveHistory to Room

**Files:**
- `aura-core/src/main/kotlin/com/aura/proactive/ProactiveEventEntity.kt` — Room entity
- `aura-core/src/main/kotlin/com/aura/proactive/ProactiveEventDao.kt` — insert, recent(limit)
- `aura-core/src/main/kotlin/com/aura/proactive/ProactiveEvents.kt` — on each event, insert into DAO before updating in-memory list
- New Room database `aura-proactive.db` (or extend existing)

**Key cases:**
- DAO `recent(100)` returns events sorted by timestamp DESC
- ProactiveEvents: in-memory cache (last 100) + persistent store
- On startup, ProactiveEvents loads from DAO into the in-memory list

**Verify:** Manual — fire a few events, kill app, restart, see them in ProactiveHistoryScreen.

---

### Task 6: error banner needs auto-dismiss + retry

**Files:**
- `app/src/main/kotlin/com/aura/ui/viewmodel/ChatViewModel.kt` — `error: String?` should auto-clear after 5 seconds; new `dismissError()` action
- `app/src/main/kotlin/com/aura/ui/screens/ChatScreen.kt` — error surface shows retry button when error.retryable

**Key cases:**
- Error appears, stays 5s, then auto-dismisses
- If retryable, retry button re-runs last message
- Manual dismiss via close button

**Verify:** Manual — trigger 503, observe 5s auto-dismiss, trigger retryable error, observe retry button.

---

### Task 7: TTS filters out tool noise

**Files:**
- `app/src/main/kotlin/com/aura/ui/viewmodel/ChatViewModel.kt` — `textToSpeech.speak()` should only fire on the `is AgentEvent.Done` path, which already uses `responseBuffer` (text delta accumulator). Confirm no other `speak()` calls.

**The bug:** Looking at code, speak() only fires in the Done handler with responseBuffer. The tool result text doesn't enter responseBuffer. So actually this is already correct. But I should add a test that confirms it.

**Files:**
- `aura-core/src/test/kotlin/com/aura/agent/AgentEventTextAccumulatorTest.kt` — extract the text accumulation logic into a testable function

**Verify:** Unit test: when AgentEvent.TextDelta fires, buffer accumulates; when AgentEvent.ToolResult fires, buffer is unchanged.

---

### Task 8: KG extraction throttled + background

**Files:**
- `aura-core/src/main/kotlin/com/aura/kg/ConversationKgExtractor.kt` — debounce 5s; only one extraction runs at a time
- Track in-flight job; if new extract() call comes during running, ignore (next turn will trigger again)

**Key cases:**
- 10 rapid assistant turns → 1 KG extraction (or 2 max), not 10
- Background coroutine doesn't block chat

**Verify:** Manual — chat for 30s, check the model latency stays normal; check Room table for fewer extractions.

---

### Task 9: Dead state in OnboardingViewModel

**Files:**
- `app/src/main/kotlin/com/aura/ui/screens/OnboardingScreen.kt` — remove `testing: Boolean, testError: String?` if not used; or wire them up to an actual test call

**Decision:** The fields were placeholder. If we want to test the API key, we need a real check (e.g., `providerRegistry.configured().any { it.prefix == "ollama" }` after save). Let me actually wire that up — it's a real value-add.

**Files:**
- `app/src/main/kotlin/com/aura/ui/viewmodel/OnboardingViewModel.kt` — add `testKey()` that saves + checks via a cheap provider list call
- `app/src/main/kotlin/com/aura/ui/screens/OnboardingScreen.kt` — show a "Test" button after key entry; if green check, "✓ verified" displays

**Verify:** Manual — paste Ollama key, see green check.

---

### Task 10: `escapeLikeWildcards` test for regression

**Files:**
- `aura-core/src/test/kotlin/com/aura/memory/LikeEscapingTest.kt` — already in task 1

This is part of task 1.

---

## Execution Order

```
Commit 1: Task 1  — Tests for new features (ConversationStore, LIKE escaping, History, Tasks VM)
Commit 2: Task 2  — PermissionGranted event (real tool retry, not fake user turn)
Commit 3: Task 3  — HandsScreen Run + Add affordances
Commit 4: Task 4  — TasksScreen Mark Done + Add
Commit 5: Task 5  — Persist ProactiveHistory to Room
Commit 6: Task 6  — Error banner auto-dismiss + retry button
Commit 7: Task 7  — TTS accumulator test (locks the no-tool-noise contract)
Commit 8: Task 8  — KG extractor debounce
Commit 9: Task 9  — Onboarding "verify key" affordance
```

10 commits, 10 bugs closed. Verification: full test suite green at end.
