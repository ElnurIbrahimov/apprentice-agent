# Aura Android Audit Remediation Plan

> **For Hermes:** Execute every task in order using strict TDD, preserve current user-visible behavior unless the finding explicitly requires change, and commit each independently.

**Goal:** Close the verified runtime, integration, data-quality, configuration, security, and test-gate defects found in the 2026-07-14 current-source audit.

**Architecture:** Fix authority boundaries rather than symptoms: provider/network dispatch at the transport boundary; embeddings at the conversation store; persona at Brain identity resolution; permissions in observable Compose state; tool-call identity as one stable ID; SSRF at every redirect/DNS hop; integration credentials through one Settings model. Keep current Room schemas unless persistence truly changes.

**Tech Stack:** Kotlin 1.9.24, Compose Material 3, coroutines/Flow, OkHttp, Hilt, Room, DataStore, MockK, JUnit, AndroidX Compose UI tests.

---

## Execution order

### 0. Preserve active WIP
- Commit the existing ChatComposer/ChatRoute extraction and its unit/instrumented tests without including `.hermes` screenshots or scratch files.
- Gate: isolated `:app:testDebugUnitTest :app:assembleDebug :app:assembleDebugAndroidTest` already verified green.

### 1. Restore clean Android gates
**Files:**
- `app/src/main/res/values/themes.xml`
- `app/src/main/res/values-night/themes.xml`
- `app/src/androidTest/kotlin/com/aura/SmokeTest.kt`
- `app/src/androidTest/kotlin/com/aura/ModelSelectionFlowTest.kt`

**TDD/gates:**
- Move API-27 navigation-bar resources to `values-v27` / `values-night-v27` or remove redundant declarations.
- Add proper Hilt test setup to SmokeTest.
- Replace brittle placeholder-text lookup with the stable composer test tag.
- Run lint and connected tests; expected: both previously observed failures disappear.

### 2. Make provider and network-tool I/O cancellable/off-main
**Files:**
- `aura-core/src/main/kotlin/com/aura/providers/ProviderRegistry.kt`
- `aura-core/src/main/kotlin/com/aura/providers/AnthropicProvider.kt`
- `aura-core/src/main/kotlin/com/aura/providers/GeminiProvider.kt`
- network tools using synchronous `Call.execute()`
- focused provider/tool executor tests

**TDD/gates:**
- Tests assert upstream provider collection is not executed on Main and cancellation cancels active calls.
- Use `flowOn(Dispatchers.IO)` or `runInterruptible(Dispatchers.IO)` at the owning boundary; do not scatter arbitrary dispatchers through UI code.
- Ensure blocking tools run on IO and timeout/cancellation propagates.

### 3. Make conversation semantic search real
**Files:**
- `aura-core/src/main/kotlin/com/aura/agent/ConversationStore.kt`
- `aura-core/src/test/kotlin/com/aura/agent/ConversationStoreTest.kt`

**TDD/gates:**
- Failing test proves conversations without embeddings are backfilled from their latest user text.
- Persist the generated embedding through `ConversationDao.updateEmbedding`.
- Preserve SQL-LIKE fallback when embedding generation fails.

### 4. Unify persona customization
**Files:**
- `aura-core/src/main/kotlin/com/aura/agent/Brain.kt`
- `aura-core/src/test/kotlin/com/aura/agent/BrainTest.kt`
- `app/src/main/kotlin/com/aura/ui/screens/SettingsScreen.kt`
- `app/src/main/kotlin/com/aura/ui/nav/NavGraph.kt`
- settings/navigation tests

**TDD/gates:**
- One authoritative identity path: effective identity file/asset/fallback.
- Remove or wire the inert `customIdentity` DataStore field; do not retain two competing persona systems.
- Expose the working identity editor from Settings.

### 5. Repair bottom navigation/insets
**Files:**
- `app/src/main/kotlin/com/aura/ui/nav/NavGraph.kt`
- `app/src/main/kotlin/com/aura/ui/nav/AuraBottomNavigation.kt`
- Compose instrumentation test

**TDD/gates:**
- Test reports at least 48dp tab bounds and content remains above the bar on API 30.
- Include bottom safe drawing inset without double-applying navigation bar padding.
- Verify live hierarchy on emulator.

### 6. Repair first-grant microphone state
**Files:**
- `app/src/main/kotlin/com/aura/ui/screens/chat/ChatRoute.kt`
- extracted permission-state helper where needed
- unit/Compose test

**TDD/gates:**
- Failing test proves a false→granted transition opens voice immediately.
- Permission state must be mutable/observable, not a one-time `remember` snapshot.

### 7. Make Retry resend the actual failed user turn
**Files:**
- `app/src/main/kotlin/com/aura/ui/viewmodel/ChatViewModel.kt`
- `app/src/test/kotlin/com/aura/ui/viewmodel/ChatViewModelTest.kt`

**TDD/gates:**
- Test proves retry uses the last user text even when draft is empty.
- Avoid duplicating the user turn in conversation persistence.

### 8. Preserve media/tool-call integrity
**Files:**
- `app/src/main/kotlin/com/aura/ui/viewmodel/ChatViewModel.kt`
- `app/src/main/kotlin/com/aura/ui/screens/chat/ChatRoute.kt`
- relevant tests

**TDD/gates:**
- One stable tool-call UUID is used by Vision ToolCall and ToolResult.
- Check audio descriptor/file size before loading and Base64 expansion where possible.
- Close API 26–27 image streams with `use`.

### 9. Expose all configurable providers and tool credentials
**Files:**
- `app/src/main/kotlin/com/aura/ui/screens/SettingsScreen.kt`
- `app/src/main/kotlin/com/aura/ui/settings/SettingsViewModel.kt`
- `aura-core/src/main/kotlin/com/aura/providers/ProviderKeys.kt`
- tests

**TDD/gates:**
- Gemini appears in Settings and follows the same Save & Test flow.
- Brave, Tavily, and Firecrawl keys have honest settings controls or tool errors stop claiming they do.
- Key drafts use a data-driven map where practical to prevent future parity drift.

### 10. Harden direct URL fetches against redirects and DNS rebinding
**Files:**
- `aura-core/src/main/kotlin/com/aura/core/url/SsrfGuard.kt`
- direct-fetch tools / dedicated safe fetch client
- `SsrfGuardTest.kt` plus redirect integration test

**TDD/gates:**
- Reject unresolved hosts for direct fetches or pin the validated resolved address.
- Disable automatic redirects and manually validate every Location hop.
- Keep public HTTPS redirect behavior working.

### 11. Correct remote-cost tool policy
**Files:**
- tool risk definitions for image generation, transcription, paid search/research
- `ToolExecutor.kt` / approval result path if needed
- tests

**TDD/gates:**
- Cost-bearing remote operations are not classified as pure READ_ONLY.
- Existing confirmation flow is reused; no separate UI policy stack.

### 12. Prevent first-conversation memory duplication
**Files:**
- `ChatViewModel.kt`
- `ChatViewModelTest.kt`

**TDD/gates:**
- Marker is stored once, not after every assistant response in the first conversation.
- Prefer durable deduplication (`existsByContent` or a persisted preference) over conversation-count timing.

### 13. Final verification
- `:aura-core:testDebugUnitTest`
- `:app:testDebugUnitTest`
- `:app:assembleDebug`
- `:app:assembleDebugAndroidTest`
- `:app:lintDebug`
- `:app:connectedDebugAndroidTest`
- Install current APK on the API-30 emulator and re-check onboarding, chat, Settings, microphone grant, retry, model/provider configuration, and bottom-navigation bounds.
- Run `git diff --check` and report exact test counts and any remaining warnings/deferred product decisions.
