# Aura Android — Truth Fixes (Round 2)

> **For Hermes:** Plan+execute pattern. No "should I continue?" between commits.
> Continue the prior v0.4 / v0.5 polish work by closing the 8 highest-leverage
> gaps between what the app CLAIMS and what it actually DOES. Each commit is
> independently shippable, atomic, and ends with a green gate.

**Architecture:** Surgical fixes to existing files. No new modules. All work fits
the existing 2-module build (`app` + `aura-core`).

**Tech Stack:** Kotlin 1.9.24, Compose BOM 2024.10.01, Room 2.6.1, Hilt 2.51,
coroutines, MockK, Turbine, OkHttp 4.x, WorkManager 2.9.x, Hilt Navigation
Compose.

---

## Pre-execution verification (run once)

```bash
cd D:/Aura/android
./gradlew :aura-core:testDebugUnitTest :app:testDebugUnitTest :app:lintDebug \
  :app:assembleDebug --rerun-tasks
```

Establish the green baseline (379 tests pass, lint clean, APK builds). If any of
the 8 commits below introduces a regression, bisect on the commit hash.

---

## Commit 1: Home sub-status → BriefSummary (no more "I remember 47 things")

**Problem:** `HomeScreen.kt` line 100-111 builds the sub-status from raw counts
("You have 3 open tasks"). The `BriefSummary` util + `BriefContext` data
class + `toSummary()` extension already exist in `app/src/main/kotlin/com/aura/ui/util/BriefSummary.kt`
and `aura-core/src/main/kotlin/com/aura/proactive/BriefContext.kt`. The home
screen is the only place in the app that still shows counts.

**Files:**
- Modify: `app/src/main/kotlin/com/aura/ui/screens/HomeScreen.kt` — `subStatus`
  derivation (lines 100-111)
- Modify: `app/src/main/kotlin/com/aura/ui/viewmodel/HomeViewModel.kt` — build
  `BriefContext` from existing task/memory/calendar data and expose as a state
  field

**Fix:**
1. `HomeViewModel` builds a `BriefContext` from the same data it already
   collects (pending tasks, today's calendar events, recent memories).
2. Home screen renders `state.briefContext.toSummary()` for the sub-status
   line instead of the count-based string.
3. "What I remember" card (line 264-270) shows the first sentence of each
   memory, truncated to 60 chars, with `…` ellipsis — not the whole memory.

**Verification:** `./gradlew :app:assembleDebug` + manual: open app cold,
verify the sub-status reads like a sentence ("You have a meeting at 3 and I
remember your dog is named Rex") not a stat.

---

## Commit 2: Proactive event tap → real destinations

**Problem:** `HomeScreen.kt` line 151-163 wires the proactive card tap
handlers. The MorningBrief tap goes to `onOpenChat("")` which opens a NEW
empty chat (the brief is lost). The CalendarEventSoon tap goes to
`onOpenCalendar()` which is the in-app calendar tool surface, not the system
Calendar app the user expects.

**Files:**
- Modify: `app/src/main/kotlin/com/aura/ui/screens/HomeScreen.kt` — `onTap`
  wiring
- Modify: `app/src/main/kotlin/com/aura/ui/nav/NavGraph.kt` — add new route
  for "morning brief in chat" (passes the brief body as the initial draft)
- Modify: `app/src/main/kotlin/com/aura/ui/screens/ChatScreen.kt` — accept a
  `morningBriefBody` parameter and use it as the initial user message via
  `viewModel.onUserMessage(briefBody)` (already supports this for
  `morningBriefSummary`)

**Fix:**
1. MorningBriefReady tap → navigate to Chat with the brief text passed as
   `initialDraft` so the user sees "I have the morning brief ready, what
   would you like to discuss?" pre-filled or the brief context included.
2. CalendarEventSoon tap → fire `Intent(ACTION_VIEW)` on the event's
   `content://com.android.calendar/time/<eventId>` URI to open the system
   Calendar app. The `ProactiveEventBus.Event.CalendarEventSoon` payload
   already carries the event ID (check CalendarMonitor.kt for the exact
   field).
3. MemoryDecayWarning tap → navigate to Memory screen with the fading
   memory's id as a query param so the screen scrolls to it.

**Verification:** `./gradlew :app:assembleDebug` + instrumented test on
emulator: tap each proactive card type, confirm destination.

---

## Commit 3: Onboarding MoA card → conditional

**Problem:** `OnboardingScreen.kt` line 430-453 unconditionally shows the "MoA
— Mixture of Agents" card on the done page. The README says MoA needs the
default preset `glm-5.2` + `kimi-k2.7-code` + `deepseek-v4-pro` aggregator,
i.e. requires 3+ configured providers. The user might finish onboarding
with only Ollama configured.

**Files:**
- Modify: `app/src/main/kotlin/com/aura/ui/screens/OnboardingScreen.kt` —
  `PageDone` composable (line 389-462)
- Modify: `app/src/main/kotlin/com/aura/ui/screens/OnboardingScreen.kt` —
  `OnboardingViewModel` — compute `moaAvailable: Boolean` based on
  `configuredCount >= 3` (or check the MoaPresetRepository's loaded presets
  for the current `defaultPreset.referenceModels.size + 1`)

**Fix:**
1. `OnboardingViewModel.finish()` is preceded by a check: if
   `moaAvailable == false`, replace the MoA card with text: "Tip: add more
   providers in Settings to unlock Mixture of Agents — it runs 3 models
   together for harder questions."
2. The MoA card only shows when `state.moaAvailable == true`.

**Verification:** `./gradlew :app:assembleDebug` + manual: finish onboarding
with 1 provider, no MoA card; finish with 3, MoA card shows.

---

## Commit 4: ProviderKeys sync initial load + loading gate

**Problem:** `ProviderKeys.kt` line 73-82 launches the DataStore load
asynchronously in `init`. For ~200ms after app start, `keyFor(prefix)` returns
null, the Settings screen shows "0 providers configured" then jumps, and
any chat request in that window gets a 401. The `init` is fire-and-forget;
there's no signal to the UI that loading is in progress.

**Files:**
- Modify: `aura-core/src/main/kotlin/com/aura/providers/ProviderKeys.kt` —
  add `val loaded: StateFlow<Boolean>` exposed from the init block
- Modify: `app/src/main/kotlin/com/aura/AuraApp.kt` (or MainActivity) —
  perform a blocking initial read in `onCreate` via `runBlocking` on the
  existing scope (DataStore is fast on warm start, ~5-20ms on cold start
  in our perf testing)
- Modify: `app/src/main/kotlin/com/aura/ui/settings/SettingsViewModel.kt` —
  `reload()` only shows the "configured providers" list when
  `providerKeys.loaded.value == true`, otherwise shows "Loading…"
- Modify: `app/src/main/kotlin/com/aura/ui/screens/OnboardingScreen.kt` —
  `OnboardingViewModel.refreshConfigured()` uses the loaded signal

**Fix:**
1. `ProviderKeys.loaded: StateFlow<Boolean>` flips to true after
   `loadAllKeys()` completes.
2. The Settings screen shows "Loading API keys…" chip until
   `loaded == true`.
3. `AuraApp.onCreate()` blocks on `providerKeys.loaded.first { it }` via
   `runBlocking` to ensure the first chat request sees a populated key
   cache. (runBlocking on app start is acceptable because the DataStore
   read is bounded and we do this on every app launch — it's not on a
   hot path.)

**Verification:** `./gradlew :aura-core:testDebugUnitTest :app:assembleDebug`
+ manual cold start: Settings screen never flickers "0 providers" on first
show.

---

## Commit 5: MarkdownText — real markdown library

**Problem:** `MarkdownText.kt` line 56-77 + 79-100 is a regex-based parser
that misparses `**bold *italic* bold**` (greedy `*` match grabs the
internal italic asterisk, leaves a trailing `*` orphan). It also drops
tables, ordered lists, and links — formats LLMs use heavily.

**Files:**
- Modify: `app/build.gradle.kts` — add `androidx.compose.material3:material3`
  (already present) and the markdown lib dependency
- Create: `app/src/main/kotlin/com/aura/ui/components/MarkdownText.kt` —
  replace the regex parser with a real markdown renderer

**Fix:**
Add `com.mikepenz:multiplatform-markdown-renderer:0.18.0` (or similar
well-maintained Compose markdown lib — verify current version on
mavenCentral before pinning). Replace the regex-based `parseMarkdown()`
with a call into the library. Keep the `parseMarkdown(text: String)` and
`buildStreamingAnnotatedString(text, cursorColor, isStreaming)` signatures
so the rest of the app is untouched. Add a small adapter composable for
streaming-aware rendering that keeps the `▍` cursor block.

**Verification:** `./gradlew :app:testDebugUnitTest :app:assembleDebug` +
visual smoke test on emulator: send "give me a list with **bold** items
and a `code` reference" — confirm bold + code render properly and the
list shows as a real bullet list.

---

## Commit 6: StreamingText — binary blink cursor, not alpha fade

**Problem:** `StreamingText.kt` line 53-62 animates cursor alpha 0.3 → 1.0
over 900ms (a "throb," not a blink). Real terminal cursors are binary
visible/hidden or hard left-right motion.

**Files:**
- Modify: `app/src/main/kotlin/com/aura/ui/components/StreamingText.kt` —
  cursor animation spec

**Fix:**
Change `tween(durationMillis = 900, easing = LinearEasing)` +
`RepeatMode.Reverse` (fade) to `keyframes { 0% visible; 50% hidden;
100% visible }` with `RepeatMode.Restart` and a `tween(1100)` between key
frames, OR animate `alpha` discretely between 1.0 and 0.0 using a step
function. Keep the `▍` glyph, keep the `effectiveCursor` color, keep the
position (end of text).

**Verification:** `./gradlew :app:assembleDebug` + visual: send any chat
message, confirm the cursor blinks in a binary on/off pattern during
streaming, disappears on completion.

---

## Commit 7: ToolExecutor — actually request the permission

**Problem:** `ToolExecutor.kt` line 47-51 returns `NeedsPermission` when
the tool's required Android permission is not granted. The ChatViewModel
sets `pendingPermission` state, the ChatScreen renders a
`PermissionDialog` (line 369-374), and the dialog's "Grant" button calls
`viewModel::retryAfterPermission` — but the dialog does NOT call the
Android system permission prompt. The user taps Grant, the dialog
closes, the tool re-runs, the permission check fails again, infinite
loop.

**Files:**
- Modify: `app/src/main/kotlin/com/aura/ui/screens/ChatScreen.kt` —
  `PermissionDialog` invocation
- Modify: `app/src/main/kotlin/com/aura/ui/viewmodel/ChatViewModel.kt` —
  add a `requestPermission(perm: String, onResult: (Boolean) -> Unit)`
  method that uses `ActivityResultContracts.RequestPermission()` via a
  `MutableStateFlow<String?>` for the requested permission

**Fix:**
1. ChatScreen registers a `rememberLauncherForActivityResult(RequestPermission())`
   that resolves a `Channel<Boolean>` per request.
2. When the user taps Grant in the PermissionDialog, the VM sets
   `requestingPermission: String?` and the launcher fires
   `launch(permission)`.
3. The launcher's callback updates `pendingPermission = null` and
   triggers `retryLast()`.
4. The PermissionDialog shows a loading state during the system
   prompt: title becomes "Requesting permission..." and the Grant button
   is disabled.

**Verification:** `./gradlew :app:testDebugUnitTest :app:assembleDebug` +
emulator manual: trigger a tool that needs `ACCESS_FINE_LOCATION` (e.g.
`location_now` with the tool not yet granted). Confirm the system
permission dialog actually appears, granting it makes the tool succeed.

---

## Commit 8: ChatViewModel — extract send/cancel/permission into a controller

**Problem:** `ChatViewModel.kt` is 774 lines, owns 30+ responsibilities
(per the round-1 review). The blast radius of any change to "send a
message" is the entire file. This commit extracts the streaming send
+ cancel + permission-request flow into a dedicated controller class
`ChatSendController` that the VM delegates to.

**Files:**
- Create: `app/src/main/kotlin/com/aura/ui/viewmodel/ChatSendController.kt`
- Modify: `app/src/main/kotlin/com/aura/ui/viewmodel/ChatViewModel.kt` —
  delegate send/cancel/retry/retryAfterPermission/setModel/setSpecialist
  to the controller
- Modify: `app/src/main/kotlin/com/aura/ui/screens/ChatScreen.kt` — no
  change to call sites; verify the VM surface is unchanged

**Fix:**
1. `ChatSendController` is a `@HiltViewModel` (or non-Hilt class owned
   by ChatViewModel) that takes the loop, the tool executor, the
   tool registry, and the conversation store. It owns the `runJob`,
   the streaming state, the permission request state.
2. `ChatViewModel` becomes the UI state holder: `state: StateFlow<ChatUiState>`,
   `setDraft`, `lastAssistantText`, `setModel`, `setSpecialist`,
   `newConversation`, `deleteCurrentConversation`, `toggleTts`,
   `toggleDeepMode`, `toggleIncognito`, `dismissError`, `dismissPermission`.
3. Controller is testable in isolation: mock the loop + tool executor,
   verify `send()` triggers the loop, `cancel()` cancels the runJob,
   `requestPermission()` opens the system prompt.
4. The VM drops from 774 → ~450 lines. The controller is ~250 lines
   focused on streaming.

**Verification:** `./gradlew :app:testDebugUnitTest :app:assembleDebug` +
existing ChatViewModelTest + new ChatSendControllerTest. All existing
tests pass, new tests cover: send happy path, cancel mid-stream,
retry after permission grant, retry after model switch.

---

## Out of scope (deferred to a later round)

These are real but lower-leverage than the 8 above:

- **Brain.kt identity baked into APK** — the user's "personal use only"
  constraint makes this fine. Removing it would require a full persona
  config flow that doesn't add value for one user.
- **RRF / vector fallback scaling** — `dao.allForExport()` loads all
  memories. At 500 memories this is fine; at 10K it's slow. Could add
  a vector index in HNSW but the SQLite brute-force scan is acceptable
  for personal use. Track if memory count crosses 1K.
- **ToolExecutor date/time parser** — TimeParser is regex, edge cases
  will fail. The user can correct the model when it gets the date wrong.
- **AGP 8.2.2 → 8.5+ upgrade** — 30-50% faster builds but not a
  correctness issue. Defer to a dedicated upgrade session.
- **Specialist `toolsAllowed` set not compile-time checked against
  ToolRegistry** — would need a small unit test harness that walks all
  specialists and asserts each named tool exists. Easy add if needed.

---

## Summary

| # | Item | Severity | Files | Lines |
|---|------|----------|-------|-------|
| 1 | Home sub-status → BriefSummary | Medium | 2 | ~60 |
| 2 | Proactive event tap → real destinations | High | 3 | ~80 |
| 3 | Onboarding MoA card → conditional | Medium | 1 | ~30 |
| 4 | ProviderKeys sync initial load | High | 4 | ~50 |
| 5 | MarkdownText — real lib | Critical | 2 | ~40 (replace) |
| 6 | StreamingText — binary blink | Low | 1 | ~10 |
| 7 | ToolExecutor — real permission prompt | High | 2 | ~70 |
| 8 | ChatViewModel — extract send controller | Medium | 3 | -350 / +250 |

**8 commits, ~290 net new lines, 0 new modules, 1 new dependency
(Compose markdown lib).**

## Verification gates (after all commits)

```bash
cd D:/Aura/android
./gradlew :aura-core:testDebugUnitTest :app:testDebugUnitTest \
  :app:lintDebug :app:assembleDebug --rerun-tasks
```

All gates green. Tag `v0.6.0` with the message:

> v0.6.0 — Truth fixes: home uses BriefSummary, proactive taps go to
> real destinations, onboarding MoA card is conditional, ProviderKeys
> loads synchronously, MarkdownText is a real renderer, streaming cursor
> blinks, ToolExecutor prompts the system, ChatViewModel extracts a
> send controller.

Push, watch CI, report back.

---

## Execution protocol

1. Per Elnur's standing pattern: enter execution immediately after this
   plan is on the page. No "should I continue?" between commits.
2. Each commit is atomic. After each, run
   `./gradlew :aura-core:testDebugUnitTest :app:testDebugUnitTest` as
   the per-commit gate (2-5s).
3. After all 8 commits, run the full gate including `assembleDebug`.
4. Tag v0.6.0 at the end. Don't push without explicit go-ahead.
