# Aura Android Model Reliability + Complete UI/UX Recovery Plan

> **For Hermes:** Execute this plan task-by-task using the `software-development:subagent-driven-development`, `software-development:test-driven-development`, and `software-development:android-kotlin-review` skills. “Begin” means make the first real tool call immediately. Do not pause between tasks once execution starts.

**Goal:** Make Aura’s model-selection flow deterministic and trustworthy, then rebuild every Android surface around one coherent, theme-aware, proportionally correct mobile design system verified on a real emulator/device.

**Architecture:** Replace duplicated ViewModel-level provider discovery with one shared `ModelCatalogRepository`, make credential writes explicit and serialized, remove every concrete production model ID, and represent “no usable model” honestly. Replace the current mixture of `MaterialTheme` and hardcoded `AuraTokens.Dark` with semantic theme-aware tokens, assign system-inset ownership to the root shell, and migrate each route to pure state-driven content that can be rendered and tested independently.

**Tech stack:** Kotlin 1.9.24, Jetpack Compose + Material 3, Hilt, coroutines/Flow, DataStore, OkHttp, MockWebServer, Compose UI tests, Robolectric, Gradle, Android emulator/ADB.

**Path convention:** Unless a path starts with `app/`, `aura-core/`, `gradle/`, `docs/`, or `scripts/`, paths beginning with `ui/` or `widget/` are relative to `app/src/main/kotlin/com/aura/`.

**Expected shape:** 9 phases, 39 atomic commits, approximately 20–30 focused AI-assisted implementation hours plus real-device visual review. No Room migration is expected. No feature reduction is allowed.

---

## 1. Why this plan supersedes the previous plans

Prior plans:

- `.hermes/plans/2026-07-11-extreme-ux-polish.md`
- `.hermes/plans/2026-07-11-shared-screen-shell.md`

Those plans were only 15–17 lines. They did not specify:

- a credential/model state machine;
- real provider failure behavior;
- removal of hardcoded model IDs;
- light/dark support;
- single-owner inset rules;
- numeric mobile proportion targets;
- screen-by-screen acceptance criteria;
- real model-selection integration tests;
- a foreground-package check before screenshots;
- a full route/state visual matrix.

Parts were implemented in commits such as `f457d1a`, `ff8fd7e`, and subsequent UI commits, but the current code proves the work was partial. `AuraScreenHeader` exists, while the promised unified scaffold does not. Model refresh exists, while the actual key → catalog → selection flow remains untested and failure-prone.

This is the replacement plan.

---

## 2. Verified current-state findings

These are verified against the current branch `feat/tier-1-friction`, not recalled from an old review.

### 2.1 Model and credential failures

1. `app/src/main/kotlin/com/aura/ui/settings/ProviderKeyField.kt:34-89`
   - Calls the persistence callback from every `OutlinedTextField.onValueChange` event.
   - Typing one long key can launch dozens of writes.

2. `app/src/main/kotlin/com/aura/ui/settings/SettingsViewModel.kt:267-278`
   - Launches a new coroutine for every key update.
   - There is no per-provider cancellation, debounce, explicit save boundary, or write ordering.

3. `aura-core/src/main/kotlin/com/aura/providers/ProviderKeys.kt:126-146`
   - Every write re-reads every encrypted key.
   - A failed decrypt can prevent `loaded` from reaching a terminal state.
   - The current tests do not exercise `set()`.

4. `app/src/main/kotlin/com/aura/ui/viewmodel/ChatViewModel.kt:301-350` and `SettingsViewModel.kt:280-328`
   - Duplicate model-loading logic.
   - Query providers sequentially.
   - One slow provider can block all successful providers.
   - A refresh already marked loading cannot be superseded.

5. `aura-core/src/main/kotlin/com/aura/providers/AnthropicProvider.kt:137-159` and `GeminiProvider.kt:132-165`
   - Return hardcoded fallback catalogs after unauthorized/network/malformed responses.
   - `SettingsViewModel.verifyKey()` treats any returned list as successful verification.
   - An invalid credential can therefore be reported as “✓ Verified.”

6. `aura-core/src/main/kotlin/com/aura/data/UserPreferences.kt:52`
   - Hardcodes a default chat model.
   - Fresh installs display a model that is not actually configured.

7. Additional hardcoded production model choices remain in:
   - `app/src/main/kotlin/com/aura/ui/settings/SettingsViewModel.kt:32`
   - `app/src/main/kotlin/com/aura/ui/viewmodel/ChatSendController.kt:99`
   - `aura-core/src/main/kotlin/com/aura/providers/ProviderKeys.kt:169`
   - `aura-core/src/main/kotlin/com/aura/providers/AnthropicProvider.kt:159`
   - `aura-core/src/main/kotlin/com/aura/providers/GeminiProvider.kt:139-142`
   - `aura-core/src/main/assets/moa_presets.json`
   - `aura-core/src/main/kotlin/com/aura/proactive/MorningBriefBuilder.kt:150`
   - `aura-core/src/main/kotlin/com/aura/tools/VisionTool.kt:177`

8. `aura-core/src/test/kotlin/com/aura/providers/ProviderKeysTest.kt`
   - Explicitly states that it does not test `set()`.

9. `app/src/test/kotlin/com/aura/ui/viewmodel/ChatViewModelTest.kt:109-126`
   - Only tests a mocked provider returning two successful models.

10. `app/src/androidTest/kotlin/com/aura/SmokeTest.kt`
    - Only checks package identity and that `MainActivity` launches.
    - No Android test touches API-key entry, provider verification, model selection, or sending.

### 2.2 UI/UX and proportion failures

1. `AuraTheme` defaults to system mode, but these files hardcode `AuraTokens.Dark`:
   - `ui/screens/ChatScreen.kt`
   - `ui/components/MessageBubble.kt`
   - `ui/components/EmptyChatState.kt`
   - `ui/components/StreamingText.kt`
   - `ui/nav/NavGraph.kt`

   Result: light-mode Chat displays dark controls and near-white text on a white background.

2. `ui/nav/NavGraph.kt:116`
   - Root `Scaffold` owns safe-area padding.

3. `ui/screens/ChatScreen.kt:598+`
   - Adds `statusBarsPadding()` again.
   - Composer adds `navigationBarsPadding()` again.

4. Nested child `Scaffold`s occur in:
   - `HandsScreen.kt`
   - `TasksScreen.kt`
   - `ProactiveHistoryScreen.kt`
   - `ProfileScreen.kt`
   - `RemindersScreen.kt`
   - `IdentityEditorScreen.kt`

   They currently inherit another safe-area policy and create inconsistent vertical offsets.

5. `ui/nav/NavGraph.kt:243+`
   - Bottom navigation combines external vertical padding, internal padding, a custom container, and navigation-bar padding.
   - It consumes roughly 100dp including the system navigation area on the reference emulator.

6. Current source sizes indicate testability and consistency problems:
   - `ChatScreen.kt`: 1,464 lines
   - `SettingsScreen.kt`: 961 lines
   - `TasksScreen.kt`: 822 lines
   - `MemoryScreen.kt`: 817 lines
   - `ChatViewModel.kt`: 803 lines
   - `HomeScreen.kt`: 600 lines
   - `KnowledgeGraphScreen.kt`: 594 lines
   - `ProactiveHistoryScreen.kt`: 570 lines

7. Only ViewModel/pure-logic tests exist for most screens. There are no real Compose route tests.

8. Three files previously named as model/chat screenshots were actually screenshots of the Android launcher after connected tests uninstalled the app. Visual verification accepted filenames instead of checking the foreground package and pixels.

---

## 3. Non-negotiable product decisions

1. **No concrete model IDs in production source.** Chat, embedding, vision, background/proactive, and MoA roles must resolve from live configured catalogs and explicit user choices.
2. **No fake active model.** If no valid model exists, `activeModel` is absent and Chat says “Connect a provider” or “Choose a model.”
3. **Invalid credentials never become verified through fallback data.** Cached models are display continuity, not credential verification.
4. **System theme remains supported.** Do not hide light-theme bugs by forcing dark mode.
5. **One inset owner.** The root shell owns system bars; child routes own only their internal layout and IME behavior.
6. **Bottom navigation remains available as previously requested, but becomes compact and theme-aware.** Secondary screens retain clear back affordances.
7. **Personal-use scope.** No Play Store, multi-user, analytics, marketing, distribution compliance, or public onboarding work.
8. **No feature reduction.** Existing memory, graph, tasks, reminders, hands, tools, proactive, voice, widgets, diagnostics, and provider functionality remains.
9. **Mobile-native composition, not a desktop-web layout copy.** Brand language may match Aura Web; proportions and navigation must be phone-native.
10. **No “SOTA” claim without live screenshots of Aura itself.**

---

## 4. Target architecture

### 4.1 Credential state

Create a terminal state for every provider credential:

```kotlin
sealed interface ProviderCredentialState {
    data object NotConfigured : ProviderCredentialState
    data object Loading : ProviderCredentialState
    data class Saved(val updatedAt: Long) : ProviderCredentialState
    data class Valid(val modelCount: Int, val checkedAt: Long) : ProviderCredentialState
    data class Invalid(val reason: String, val checkedAt: Long) : ProviderCredentialState
    data class StorageError(val reason: String) : ProviderCredentialState
}
```

Rules:

- Text editing is local draft state.
- Persistence happens only on explicit **Save & test** or **Clear**.
- One provider has at most one active save/test job.
- New save cancels/supersedes the prior job.
- Stored value is trimmed once, encrypted once, and never logged.
- Decryption failure reaches `StorageError`; it never leaves loading suspended forever.

### 4.2 Shared model catalog

Create one singleton `ModelCatalogRepository` used by Chat, Settings, Onboarding, role selectors, and MoA.

```kotlin
data class ModelCatalogSnapshot(
    val models: List<ModelDescriptor>,
    val providers: Map<String, ProviderCatalogStatus>,
    val fetchedAt: Long,
    val isStale: Boolean,
)

sealed interface ModelCatalogState {
    data object Uninitialized : ModelCatalogState
    data class Loading(val cached: ModelCatalogSnapshot?) : ModelCatalogState
    data class Ready(val snapshot: ModelCatalogSnapshot) : ModelCatalogState
    data class Partial(val snapshot: ModelCatalogSnapshot) : ModelCatalogState
    data class Empty(val providers: Map<String, ProviderCatalogStatus>) : ModelCatalogState
    data class Failed(
        val providers: Map<String, ProviderCatalogStatus>,
        val cached: ModelCatalogSnapshot?,
    ) : ModelCatalogState
}
```

Rules:

- Query configured providers concurrently with `supervisorScope + async`.
- Apply a per-provider timeout of 10 seconds.
- Preserve successful provider results if another provider fails.
- A forced refresh cancels/supersedes the previous refresh.
- Persist the last successful provider model list and timestamp in DataStore.
- Cached entries are labeled stale and never used to validate a key.
- Non-2xx, malformed, empty, timeout, and network errors remain typed and provider-specific.
- Provider adapters return live results or throw typed errors; no fallback catalogs.

### 4.3 Model roles

Persist nullable user selections for:

- default chat model;
- embedding model;
- vision model;
- background/proactive model;
- optional MoA preset/model-role mapping.

Rules:

- Existing persisted selections remain only if present in the live/cached catalog.
- Missing/unavailable choices are shown as unavailable and require user action; they are not silently replaced.
- On fresh install, successful provider connection leads directly to model selection.
- If onboarding is skipped, the app stays usable for local surfaces but Chat send is disabled until a model is chosen.
- Virtual identifiers such as a user-defined MoA preset are allowed; concrete provider model IDs inside production source/assets are not.

### 4.4 Semantic visual system

Create:

- `AuraPalette` — semantic light/dark colors;
- `LocalAuraPalette` / `MaterialTheme.auraPalette`;
- `AuraSpacing` — spacing and density contract;
- `AuraDimensions` — toolbar, nav, card, icon, and composer dimensions;
- shared screen/state/editor components.

Raw dark/light palette values become implementation details of `AuraTheme`; screen code cannot reference `AuraTokens.Dark` or `.Light` directly.

### 4.5 System-inset ownership

- `NavGraph` root `Scaffold` owns status, horizontal safe drawing, bottom bar, and navigation-bar policy.
- Child screen scaffolds use `contentWindowInsets = WindowInsets(0)`.
- Child screens consume root padding exactly once.
- Chat owns only IME padding for the composer.
- Modal sheets/dialogs own their own IME/navigation behavior.

### 4.6 Testable route/content split

Each complex route becomes:

```kotlin
@Composable
fun MemoryRoute(viewModel: MemoryViewModel = hiltViewModel(), ...) { ... }

@Composable
internal fun MemoryContent(state: MemoryUiState, actions: MemoryActions, ...) { ... }
```

This permits deterministic light/dark/empty/loading/error/populated rendering without Hilt or real storage.

---

## 5. Numeric mobile proportion contract

Reference compact viewport: **393 × 851dp** (the current 1080 × 2340 emulator at 440dpi).

| Element | Contract |
|---|---|
| Screen horizontal gutter | 16dp compact; 20–24dp medium/expanded |
| Top app bar | 56dp total content height |
| Bottom navigation | 60–64dp visual container, excluding system nav inset |
| Bottom-nav icon / label | 20–22dp / 11–12sp |
| Primary title | 26–30sp; never 40sp on compact screens |
| Section title | 18–20sp |
| Body | 15–16sp |
| Secondary/metadata | 12–13sp, minimum readable contrast |
| Standard card padding | 14–16dp |
| Dense list row | 64–80dp depending on metadata |
| Tab row | 48dp; no vertically stacked icon + label unless necessary |
| Visual icon button | 36–40dp visual with at least 48dp touch target |
| Model pill | 36–40dp high; maximum 55% of compact width |
| Composer | 52dp minimum; 144dp maximum before internal scroll |
| Modal sheet | Content-adaptive; maximum 90% height; no half-screen dead void |
| Empty-state content | Max width 280dp; positioned intentionally, never through arbitrary weighted spacers |
| Motion | 150–220ms standard; spring only for direct manipulation; no idle infinite animation offscreen |
| Normal-text contrast | At least 4.5:1 in both themes |

Additional rules:

- No route may reserve status/navigation insets twice.
- No core screen may leave more than ~30% of the usable height blank unless the blank space is intentional content focus (for example, an active chat timeline before first message).
- Primary action hierarchy: one dominant action, at most two visible secondary actions; overflow lower-frequency actions.
- Emoji are not primary UI icons. Use consistent vector assets.

---

# PHASE 0 — Evidence, guardrails, and test infrastructure

## Task 1: Commit the UX and reliability contract

**Objective:** Put the decisions above into a durable in-repo contract before implementation begins.

**Files:**

- Create: `docs/design/AURA_ANDROID_UX_CONTRACT.md`
- Create: `docs/design/AURA_ANDROID_VISUAL_QA.md`
- Create: `docs/design/AURA_ANDROID_REFERENCE_AUDIT.md`
- Reference: `.hermes/ui-real-audit-sheet.png` and verified live screenshots; do not commit personal runtime data.

**Steps:**

1. Inspect the actually rendered Aura Web UI and current official Android surfaces from ChatGPT, Claude, Gemini, and Perplexity using primary sources available at execution time.
2. Record only transferable mobile patterns: toolbar density, composer geometry, picker recovery, list hierarchy, motion restraint, and light/dark behavior. Do not copy branding or desktop layout.
3. Document semantic tokens, dimensions, inset ownership, screen state requirements, and foreground-package verification.
4. Document the no-hardcoded-model rule and model-state acceptance criteria.
5. List personal-use non-goals.
6. Commit.

**Verification:** Review the document against Sections 2–5 of this plan.

**Commit:** `docs(android): define model reliability and mobile UX contract`

---

## Task 2: Add deterministic provider and Compose test dependencies

**Objective:** Make real HTTP-contract and Compose-state testing possible.

**Files:**

- Modify: `gradle/libs.versions.toml`
- Modify: `aura-core/build.gradle.kts`
- Modify: `app/build.gradle.kts`
- Create: `app/src/androidTest/kotlin/com/aura/testing/TestTags.kt`
- Create: `app/src/androidTest/kotlin/com/aura/testing/TestData.kt`

**Steps:**

1. Add OkHttp MockWebServer using the existing OkHttp version.
2. Add Hilt Android testing/compiler dependencies using the existing Hilt version; Task 11 requires full-route fake-provider injection.
3. Add shared deterministic provider/model fixtures.
4. Run a no-op test to verify the harness compiles.

**Tests:**

```bash
./gradlew :aura-core:testDebugUnitTest :app:testDebugUnitTest :app:compileDebugAndroidTestKotlin
```

**Commit:** `test(android): add provider and Compose integration harness`

---

# PHASE 1 — P0 credential and model-selection recovery

## Task 3: Serialize credential storage and expose terminal load errors

**Objective:** Make key persistence exact, ordered, restart-safe, and observable.

**Files:**

- Create: `aura-core/src/main/kotlin/com/aura/providers/ProviderCredentialState.kt`
- Modify: `aura-core/src/main/kotlin/com/aura/providers/ProviderKeys.kt`
- Modify: `aura-core/src/main/kotlin/com/aura/security/SecureDataStore.kt`
- Rewrite tests: `aura-core/src/test/kotlin/com/aura/providers/ProviderKeysTest.kt`

**Failing tests first:**

- `set persists the exact key and a new ProviderKeys instance reloads it`
- `concurrent updates cannot leave an older partial key persisted`
- `clear removes the key and updates state`
- `decryption failure reaches StorageError instead of hanging loaded`
- `blank key is normalized to clear`

**Implementation:**

- Guard each provider write with a keyed mutex or single ordered actor.
- Do not re-read every credential after every write.
- Expose a terminal load state instead of a Boolean that can remain false forever.

**Targeted test:**

```bash
./gradlew :aura-core:testDebugUnitTest --tests 'com.aura.providers.ProviderKeysTest'
```

**Commit:** `fix(models): make provider credential persistence deterministic`

---

## Task 4: Make provider model APIs truthful and testable

**Objective:** Ensure provider model discovery returns live data or an explicit typed failure—never stale hardcoded success.

**Files:**

- Create: `aura-core/src/main/kotlin/com/aura/providers/ProviderCatalogException.kt`
- Modify: `aura-core/src/main/kotlin/com/aura/providers/Provider.kt`
- Modify: `aura-core/src/main/kotlin/com/aura/providers/OpenAiCompatProvider.kt`
- Modify: `aura-core/src/main/kotlin/com/aura/providers/AnthropicProvider.kt`
- Modify: `aura-core/src/main/kotlin/com/aura/providers/GeminiProvider.kt`
- Modify: `aura-core/src/main/kotlin/com/aura/providers/GroqProvider.kt`
- Modify: `aura-core/src/main/kotlin/com/aura/providers/OpenRouterProvider.kt`
- Modify: `aura-core/src/main/kotlin/com/aura/providers/ProviderModule.kt`
- Add/replace provider tests under `aura-core/src/test/kotlin/com/aura/providers/`

**Failing tests first:**

- 401 produces `Unauthorized`, not a model list.
- 429 produces typed rate-limit failure.
- malformed JSON produces `MalformedResponse`.
- empty list produces `EmptyCatalog`.
- success preserves exact live IDs.
- cancellation is rethrown.
- no test reaches the public network.

**Implementation:**

- Inject provider catalog base URLs for tests.
- Remove Anthropic/Gemini fallback model lists.
- Sanitize error bodies before exposing them to UI.

**Targeted test:**

```bash
./gradlew :aura-core:testDebugUnitTest --tests 'com.aura.providers.*ProviderTest'
```

**Commit:** `fix(models): replace fallback catalogs with typed live discovery`

---

## Task 5: Introduce the shared concurrent model catalog

**Objective:** Replace duplicated sequential refresh logic with one resilient shared source of truth.

**Files:**

- Create: `aura-core/src/main/kotlin/com/aura/providers/ModelDescriptor.kt`
- Create: `aura-core/src/main/kotlin/com/aura/providers/ModelCatalogState.kt`
- Create: `aura-core/src/main/kotlin/com/aura/providers/ModelCatalogCache.kt`
- Create: `aura-core/src/main/kotlin/com/aura/providers/ModelCatalogRepository.kt`
- Modify: `aura-core/src/main/kotlin/com/aura/providers/ProviderModule.kt`
- Create: `aura-core/src/test/kotlin/com/aura/providers/ModelCatalogRepositoryTest.kt`

**Failing tests first:**

- providers refresh in parallel;
- one hung provider times out without hiding a successful provider;
- one failed provider produces `Partial`;
- forced refresh supersedes an older refresh;
- cache survives repository recreation;
- stale cache is labeled stale;
- cache never changes invalid credential verification into success;
- MoA appears only when its real dependencies are configured.

**Implementation:**

- `supervisorScope + async` per provider.
- 10-second timeout per provider.
- one shared `StateFlow<ModelCatalogState>`.
- provider-specific last-success cache in DataStore.

**Targeted test:**

```bash
./gradlew :aura-core:testDebugUnitTest --tests 'com.aura.providers.ModelCatalogRepositoryTest'
```

**Commit:** `feat(models): add concurrent cached model catalog`

---

## Task 6: Remove every concrete production model ID

**Objective:** Replace stale model literals with explicit role selections backed by the catalog.

**Files:**

- Modify: `aura-core/src/main/kotlin/com/aura/data/UserPreferences.kt`
- Modify: `aura-core/src/main/kotlin/com/aura/providers/ProviderKeys.kt`
- Modify: `aura-core/src/main/kotlin/com/aura/providers/ProviderRegistry.kt`
- Modify: `aura-core/src/main/assets/moa_presets.json`
- Modify: `aura-core/src/main/kotlin/com/aura/proactive/MorningBriefBuilder.kt`
- Modify: `aura-core/src/main/kotlin/com/aura/tools/VisionTool.kt`
- Modify: `app/src/main/kotlin/com/aura/ui/viewmodel/ChatSendController.kt`
- Modify: `app/src/main/kotlin/com/aura/ui/settings/SettingsViewModel.kt`
- Create: `aura-core/src/main/kotlin/com/aura/providers/ModelRolePreferences.kt`
- Add tests for each affected role.

**Decisions:**

- `defaultChatModel`, `embeddingModel`, `visionModel`, and `backgroundModel` become nullable selections.
- Remove the `parse("default") → firstConfiguredModelId() → sequential listModels()` send path. Sending must consume an already-resolved catalog selection and must never perform model discovery on the hot path.
- `ProviderRegistry.firstConfiguredModelId()` is deleted or converted to a cache-only helper with no network access; extend `ProviderRegistryTest.kt` to prove a send does not call `listModels()`.
- Missing embedding model uses the existing local embedder path or an actionable “choose embedding model” state; it does not invent a remote model.
- Vision receives a selected model/provider through context/config; it never hardcodes an OpenAI model.
- MoA presets use user-selected role references, not concrete provider model IDs.

**Source gate:**

```bash
python -c "from pathlib import Path; import re; roots=[Path('app/src/main'),Path('aura-core/src/main')]; pat=re.compile(r'\"(?:gpt|claude|gemini|deepseek|llama|qwen|mistral|nomic)[-:][^\"]+\"',re.I); hits=[f'{p}:{i}:{line.strip()}' for root in roots for p in root.rglob('*') if p.suffix in {'.kt','.json'} for i,line in enumerate(p.read_text(encoding='utf-8',errors='ignore').splitlines(),1) if pat.search(line)]; print('\n'.join(hits)); raise SystemExit(bool(hits))"
```

Allow examples only in docs/tests, not production behavior.

**Commit:** `fix(models): replace hardcoded model IDs with role selections`

---

## Task 7: Replace per-keystroke key writes with explicit Save & test

**Objective:** Make provider setup understandable and impossible to race.

**Files:**

- Modify: `app/src/main/kotlin/com/aura/ui/settings/ProviderKeyField.kt`
- Modify: `app/src/main/kotlin/com/aura/ui/settings/SettingsViewModel.kt`
- Create: `app/src/test/kotlin/com/aura/ui/settings/ProviderCredentialViewModelTest.kt`

**UI contract:**

- Local masked draft.
- **Save & test** primary action.
- **Clear** destructive action.
- Inline states: Unsaved, Saving, Testing, Connected with model count, Invalid, Storage error.
- Never display “✓ set” merely because a value is non-empty.
- Never echo the key into logs, error text, semantics, or screenshots.

**Failing tests first:**

- 40 rapid draft updates produce zero persistence calls.
- Save produces exactly one persistence call with the final value.
- save/test jobs are superseded per provider.
- invalid key displays invalid.
- successful key refreshes the shared catalog.
- clear removes credential and related catalog entry.

**Commit:** `fix(settings): make provider credentials explicit and verifiable`

---

## Task 8: Represent active-model availability honestly

**Objective:** Remove fake defaults and prevent sending through an unavailable model.

**Files:**

- Modify: `app/src/main/kotlin/com/aura/ui/viewmodel/ChatViewModel.kt`
- Modify: `app/src/main/kotlin/com/aura/ui/viewmodel/ChatSendController.kt`
- Modify: `app/src/main/kotlin/com/aura/ui/settings/SettingsViewModel.kt`
- Modify: `aura-core/src/main/kotlin/com/aura/data/UserPreferences.kt`
- Create: `app/src/main/kotlin/com/aura/ui/model/ActiveModelState.kt`
- Create: `app/src/test/kotlin/com/aura/ui/viewmodel/ChatModelSelectionTest.kt`

**State contract:**

- `Unconfigured`
- `NeedsSelection(catalog)`
- `Ready(model, scope)`
- `Unavailable(previousModel, reason)`

**Rules:**

- Chat header says **Connect provider** or **Choose model** when appropriate.
- Composer send button is disabled until `Ready`.
- Stored unavailable model is surfaced, not silently replaced.
- Session override and global default remain distinct.
- Changing the default updates selection state immediately without mutating or staling the shared catalog; add tests for `setDefaultModel()` and `makeActiveModelDefault()`.
- No send path may call `listModels()` or resolve a `"default"` sentinel over the network.
- Specialist selection may suggest a role, but cannot silently hard-switch to a concrete hardcoded model.

**Commit:** `fix(chat): gate sending on a real available model`

---

## Task 9: Rebuild the model picker state and recovery actions

**Objective:** Make model selection useful in loading, partial, empty, stale, error, and success states.

**Files:**

- Modify: `app/src/main/kotlin/com/aura/ui/components/ModelPickerSheet.kt`
- Create: `app/src/main/kotlin/com/aura/ui/model/ModelPickerUiState.kt`
- Modify: `app/src/main/kotlin/com/aura/ui/screens/ChatScreen.kt`
- Modify: `app/src/main/kotlin/com/aura/ui/nav/NavGraph.kt`
- Create: `app/src/androidTest/kotlin/com/aura/ui/ModelPickerStateTest.kt`

**Required states:**

- no provider → direct **Open provider settings** action;
- loading with cached models → models remain usable with refresh indicator;
- partial provider failure → successful groups remain selectable;
- stale cache → timestamp/stale label;
- zero live models → provider-specific reason and retry;
- success → grouped models, search, current/default markers;
- explicit scope selector: **This chat** / **Default**.

**Proportion contract:**

- Empty/error sheet shrinks to content instead of leaving half-screen void.
- Search is hidden or disabled when there are zero entries.
- Header actions align on one 56dp row.

**Commit:** `feat(picker): add honest model states and direct recovery`

---

## Task 10: Make onboarding complete the provider → model loop

**Objective:** Ensure first-run success means the user can actually send a message.

**Files:**

- Modify: `app/src/main/kotlin/com/aura/ui/screens/OnboardingScreen.kt`
- Modify: `app/src/main/kotlin/com/aura/FirstRunGate.kt`
- Modify: `app/src/main/kotlin/com/aura/MainActivity.kt`
- Create: `app/src/test/kotlin/com/aura/ui/viewmodel/OnboardingModelFlowTest.kt`

**Flow:**

1. Intro.
2. Provider credential.
3. Live verification.
4. Select default chat model from returned catalog.
5. Optional role selections or defer them to Settings.
6. Complete.

**Skip behavior:**

- Skip remains available for personal use.
- The app enters local-only mode with Chat clearly disabled until setup.
- It never pretends a default model exists.

**Commit:** `fix(onboarding): close the provider-to-first-message loop`

---

## Task 11: Add a real end-to-end model-selection test

**Objective:** Lock the exact flow that repeatedly broke.

**Files:**

- Create: `app/src/androidTest/kotlin/com/aura/ModelSelectionFlowTest.kt`
- Create: `app/src/androidTest/kotlin/com/aura/testing/FakeProviderModule.kt`
- Modify: `app/src/androidTest/kotlin/com/aura/SmokeTest.kt`

**Test sequence:**

1. Launch fresh app.
2. Enter/paste a full test key.
3. Tap Save & test.
4. Mock `/models` returns two known IDs.
5. Select one as default.
6. Kill/recreate Activity and process-equivalent state.
7. Open Chat.
8. Assert selected model appears and send is enabled.
9. Open picker and assert both models appear.
10. Change to chat-only model and verify global default is unchanged.

**Failure variants:** invalid key, one provider timeout plus one success, cached stale catalog, no provider.

**Test:**

```bash
./gradlew :app:connectedDebugAndroidTest
```

**Commit:** `test(models): cover credential-to-chat selection end to end`

---

# PHASE 2 — One coherent mobile design foundation

## Task 12: Introduce theme-aware semantic Aura palettes

**Objective:** Make every custom component correct in both light and dark mode.

**Files:**

- Create: `app/src/main/kotlin/com/aura/ui/theme/AuraPalette.kt`
- Modify: `app/src/main/kotlin/com/aura/ui/theme/AuraTokens.kt`
- Modify: `app/src/main/kotlin/com/aura/ui/theme/Theme.kt`
- Modify: `ui/screens/ChatScreen.kt`
- Modify: `ui/components/MessageBubble.kt`
- Modify: `ui/components/EmptyChatState.kt`
- Modify: `ui/components/StreamingText.kt`
- Modify: `ui/nav/NavGraph.kt`
- Create: `app/src/test/kotlin/com/aura/ui/theme/AuraPaletteTest.kt`

**Implementation:**

- `LocalAuraPalette` chooses light/dark alongside `MaterialTheme`.
- Migrate all 87 direct `AuraTokens.Dark` uses.
- Make raw dark/light palettes inaccessible to screen code after migration.
- Gradients derive from the current semantic palette.

**Acceptance:** zero production references to `AuraTokens.Dark` or `AuraTokens.Light` outside the theme implementation.

**Commit:** `refactor(ui): make Aura components theme-aware`

---

## Task 13: Define spacing, dimensions, typography, and component primitives

**Objective:** Replace one-off dimensions with the numeric contract in Section 5.

**Files:**

- Create: `app/src/main/kotlin/com/aura/ui/theme/AuraSpacing.kt`
- Create: `app/src/main/kotlin/com/aura/ui/theme/AuraDimensions.kt`
- Modify: `app/src/main/kotlin/com/aura/ui/theme/Type.kt`
- Modify: `app/src/main/kotlin/com/aura/ui/theme/Shapes.kt`
- Create: `app/src/main/kotlin/com/aura/ui/components/AuraButtons.kt`
- Create: `app/src/main/kotlin/com/aura/ui/components/AuraCards.kt`
- Create: `app/src/main/kotlin/com/aura/ui/components/AuraChips.kt`
- Create: `app/src/main/kotlin/com/aura/ui/components/AuraIconButton.kt`
- Create: `app/src/main/kotlin/com/aura/ui/components/ResponsiveContainer.kt`

**Acceptance:**

- Components expose 48dp touch targets with compact visual bounds.
- No screen invents a new primary radius, toolbar height, or icon-container size.
- `ResponsiveContainer` fills compact width and caps content at 600dp, centered, on medium/landscape viewports; graph/canvas surfaces may explicitly opt out.
- Fraunces is reserved for deliberate display moments; Inter remains the UI workhorse.

**Commit:** `feat(ui): add compact mobile dimensions and primitives`

---

## Task 14: Enforce one-owner system insets

**Objective:** Remove double status/navigation spacing across the app.

**Files:**

- Modify: `app/src/main/kotlin/com/aura/ui/nav/NavGraph.kt`
- Rewrite: `app/src/main/kotlin/com/aura/ui/components/AuraScreenShell.kt`
- Modify: `ChatScreen.kt`
- Modify nested scaffolds in:
  - `HandsScreen.kt`
  - `TasksScreen.kt`
  - `ProactiveHistoryScreen.kt`
  - `ProfileScreen.kt`
  - `RemindersScreen.kt`
  - `IdentityEditorScreen.kt`
- Create: `app/src/androidTest/kotlin/com/aura/ui/InsetLayoutTest.kt`

**Implementation:**

- Root shell owns safe drawing.
- Child `Scaffold`s use `WindowInsets(0)`.
- Remove duplicate Chat status/navigation padding.
- Apply `consumeWindowInsets` where padding crosses composition boundaries.
- Keep IME handling local to composer/editor surfaces.

**Acceptance:**

- Header baselines match across Home, Chat, Hands, Tasks, and Diagnostics within the same design token.
- No route jumps vertically when bottom navigation visibility/state changes.

**Commit:** `fix(ui): centralize system inset ownership`

---

## Task 15: Rebuild the bottom navigation at compact mobile proportions

**Objective:** Stop navigation from dominating every screen.

**Files:**

- Modify: `app/src/main/kotlin/com/aura/ui/nav/NavGraph.kt`
- Create: `app/src/main/kotlin/com/aura/ui/nav/AuraBottomNavigation.kt`
- Create: `app/src/androidTest/kotlin/com/aura/ui/BottomNavigationTest.kt`

**Contract:**

- Four top-level items remain: Home, Chat, Memory, Settings.
- 60–64dp visual container, excluding the system nav inset.
- Theme-aware surface and subtle selected state.
- No heavy black floating pill in light mode.
- Route state remains correct with query parameters.
- Secondary screens retain navigation plus explicit back affordance where needed.

**Commit:** `feat(nav): replace oversized floating bar with compact navigation`

---

## Task 16: Add shared loading, empty, error, and inline-status components

**Objective:** Remove generic spinners and inconsistent dead-end states.

**Files:**

- Create: `ui/components/AuraLoadingState.kt`
- Create: `ui/components/AuraEmptyState.kt`
- Create: `ui/components/AuraErrorState.kt`
- Create: `ui/components/AuraInlineStatus.kt`
- Create: `ui/components/AuraSkeleton.kt`
- Create tests/previews under `app/src/test` and `app/src/debug`.

**Contract:**

- Errors always explain what happened and provide a relevant recovery action.
- Empty states have one useful next action.
- Skeletons match final geometry.
- Full-screen spinner is reserved for true startup only.

**Commit:** `feat(ui): standardize loading empty and error states`

---

## Task 17: Create a deterministic UI catalog and screenshot harness

**Objective:** Make every important screen state renderable without live data and prevent launcher screenshots from being accepted.

**Files:**

- Create: `app/src/debug/kotlin/com/aura/ui/catalog/UiCatalogActivity.kt`
- Create: `app/src/debug/kotlin/com/aura/ui/catalog/UiCatalogStates.kt`
- Create: `app/src/debug/AndroidManifest.xml`
- Create: `scripts/capture_android_ui.py`
- Create: `docs/design/AURA_ANDROID_SCREENSHOT_MATRIX.md`

**Harness requirements:**

1. Install the exact APK under test.
2. Launch with `am start -W`.
3. Verify `mResumedActivity` contains `com.aura.debug/com.aura.MainActivity` or the debug catalog activity.
4. Verify the UIAutomator root package is `com.aura.debug`.
5. Abort if the launcher is foreground.
6. Capture light/dark and named state.
7. Record viewport, density, font scale, app version, package, and commit SHA.
8. Reinstall after connected tests before any manual screenshot.

**Commit:** `test(ui): add foreground-verified visual catalog harness`

---

# PHASE 3 — First-impression and core interaction redesign

## Task 18: Fix launch, startup gate, and app-lock proportions

**Objective:** Eliminate the tiny spinner/blank canvas and make startup visually continuous with the active theme.

**Files:**

- Modify: `app/src/main/res/values/themes.xml`
- Modify: `app/src/main/res/values-night/themes.xml`
- Modify: `app/src/main/kotlin/com/aura/MainActivity.kt`
- Create: `app/src/main/kotlin/com/aura/ui/components/AuraStartupState.kt`
- Modify: app-lock content in `MainActivity.kt`

**Acceptance:**

- Light system launches light; dark launches dark—no black→white flash.
- Startup gate shows a branded, correctly sized static state rather than a lonely indeterminate dot.
- App-lock content is vertically balanced without a fixed 120dp spacer.

**Commit:** `fix(startup): replace blank launch gate with themed startup state`

---

## Task 19: Redesign onboarding for compact completion

**Objective:** Apply the model flow from Task 10 using the final visual foundation.

**Files:**

- Split `OnboardingScreen.kt` into:
  - `ui/screens/onboarding/OnboardingRoute.kt`
  - `OnboardingContent.kt`
  - `ProviderSetupStep.kt`
  - `ModelSelectionStep.kt`
- Add catalog states/previews/tests.

**Acceptance:**

- No oversized dead hero area.
- Progress and back/skip are obvious.
- Provider status and selected default are visible before completion.
- Keyboard never covers actions.

**Commit:** `feat(onboarding): deliver compact provider-first onboarding`

---

## Task 20: Redesign Home around daily priority, not a launcher grid

**Objective:** Make Home feel like an assistant’s useful briefing rather than six equal app tiles.

**Files:**

- Split `HomeScreen.kt` into:
  - `ui/screens/home/HomeRoute.kt`
  - `HomeContent.kt`
  - `HomeBriefCard.kt`
  - `HomePrimaryAction.kt`
  - `HomeSecondaryActions.kt`
- Modify: `ui/viewmodel/HomeViewModel.kt` only where state mapping must support the new hierarchy.
- Update `HomeViewModelTest.kt` wherever state mapping changes.
- Add Compose state tests.

**Hierarchy:**

1. Compact greeting/date.
2. One primary “Ask Aura” action/composer.
3. Current priority: brief, next event, overdue task, or memory insight.
4. Recent/secondary destinations in a compact row/list—not two giant 3-column grids.

**Acceptance:**

- Home has explicit loading, content, empty, and data-source error states; initial loads never flash the empty state before data resolves.
- Empty and populated Home both use the full viewport intentionally.
- Zero-count destinations become useful actions such as **Add memory** or **Create task** rather than inert `0` tiles or disabled dead ends.
- Counts are metadata, not awkward second-line labels.
- No content cluster ending halfway down the screen with 250dp of accidental void.

**Commit:** `feat(home): replace launcher grid with daily assistant hierarchy`

---

## Task 21: Split and rebuild the Chat shell/header

**Objective:** Reduce `ChatScreen.kt` complexity and correct the top-level hierarchy.

**Files:**

- Split `ChatScreen.kt` into:
  - `ui/screens/chat/ChatRoute.kt`
  - `ChatContent.kt`
  - `ChatHeader.kt`
  - `ChatTimeline.kt`
  - `ChatDialogs.kt`
- Retain `ChatViewModel` behavior except model-state changes already planned.

**Header contract:**

- 56dp.
- Model pill max 55% width.
- Keep model selector and one high-frequency action.
- Move TTS/history/delete/secondary actions into a compact overflow menu.
- Model state never truncates every action offscreen.

**Tests:** compact width, long model label, no model, session override, streaming.

**Commit:** `refactor(chat): rebuild compact testable chat shell`

---

## Task 22: Rebuild Chat empty state, message timeline, and composer

**Objective:** Make the most-used screen readable, dense, and coherent in both themes.

**Files:**

- Modify: `ui/components/EmptyChatState.kt`
- Modify: `ui/components/MessageBubble.kt`
- Modify: `ui/components/MarkdownText.kt`
- Modify: `ui/components/StreamingText.kt`
- Modify: `ui/components/StreamingMarkdownState.kt`
- Modify: `ui/components/ToolCallBadge.kt`
- Modify: `ui/components/MemoryRecallChip.kt`
- Modify: `ui/components/MoaThinkingIndicator.kt`
- Modify: `ui/components/SpecialistChips.kt`
- Modify: `ui/components/FollowUpSuggestionChips.kt`
- Modify: `ui/components/FollowUpSuggestions.kt`
- Modify: `ui/components/VisionPromptChips.kt`
- Create: `ui/screens/chat/ChatComposer.kt`

**Contract:**

- Empty-state hero uses explicit alignment rather than arbitrary weighted spacers.
- Conversation resume has a shaped loading state instead of briefly showing the first-use empty state.
- Starter actions are vector-icon chips and never clip without indicating horizontal scroll.
- Composer is one visual unit, 52dp minimum, internally scrollable to 144dp.
- Camera/voice/send have compact visuals and 48dp touch targets.
- Tap-to-speak, hold-to-talk, and continuous voice are explicitly labeled and discoverable; none relies on trial-and-error discovery.
- Auto-scroll follows tokens only while the user is already at the live edge. Once the user scrolls upward, streaming never yanks them down; **Jump to latest** becomes the only return action.
- Stop-streaming confirmation uses unambiguous **Stop** / **Keep streaming** copy rather than “Keep listening.”
- User/assistant/tool/memory/citation hierarchy is obvious without excessive containers.
- Keyboard, bottom nav, and composer never overlap.

**Visual cases:** empty, one exchange, long Markdown, streaming, tool call, citation, image, error, keyboard open, light/dark.

**Commit:** `feat(chat): polish messages composer and first-use state`

---

## Task 23: Finish the model picker visual design

**Objective:** Apply final dimensions and semantic components to Task 9’s reliable state model.

**Files:**

- Modify: `ui/components/ModelPickerSheet.kt`
- Modify: `ui/util/ModelLabels.kt`
- Add state previews and Compose tests.

**Acceptance:**

- Provider groups, counts, status, current/default markers, and scope fit without tiny text.
- Empty/error state has a direct CTA.
- Partial results remain selectable.
- Sheet height follows content until results require expansion.
- Refresh, close, and scope controls align on the grid.

**Commit:** `polish(picker): compact provider groups and recovery states`

---

# PHASE 4 — Settings, identity, and diagnostics

## Task 24: Replace the 961-line Settings page with navigable sections

**Objective:** Make Settings scannable and stop one screen from mixing every form and state.

**Files:**

- Replace `ui/screens/SettingsScreen.kt` with `ui/screens/settings/SettingsHomeScreen.kt`
- Create:
  - `ProvidersSettingsScreen.kt`
  - `ModelsSettingsScreen.kt`
  - `AppearanceSettingsScreen.kt`
  - `PersonaSettingsScreen.kt`
  - `PrivacySettingsScreen.kt`
  - `ProactiveSettingsScreen.kt`
  - `DataSettingsScreen.kt`
- Modify: `ui/nav/NavGraph.kt`
- Reuse: `ui/settings/SettingsViewModel.kt`

**Acceptance:**

- Settings home is a compact list of semantic rows.
- API keys do not share one enormous expanded form.
- Default/embedding/vision/background model selectors use the shared catalog.
- Every detail screen uses the shared shell and state components.

**Commit:** `refactor(settings): split settings into focused mobile routes`

---

## Task 25: Polish Profile, Identity, Appearance, and App Lock

**Objective:** Make personal configuration feel deliberate and safe.

**Files:**

- Modify: `ProfileScreen.kt`, `ProfileViewModel.kt`
- Modify: `IdentityEditorScreen.kt`
- Modify: appearance/privacy screens created in Task 24
- Modify app-lock screen in `MainActivity.kt` if still required.

**Acceptance:**

- Forms expose dirty/saving/saved/error state; failed name/trait/fact/identity saves never disappear silently.
- Destructive reset is separated and confirmed.
- Theme preview changes live without mixed tokens.
- Identity editing is reachable from the final Settings/Persona information architecture and uses a keyboard-safe full-height editor with sticky save action.

**Commit:** `polish(settings): unify profile persona theme and lock flows`

---

## Task 26: Redesign Diagnostics for fast personal debugging

**Objective:** Keep the useful technical density while making severity and actions clearer.

**Files:**

- Split `DiagnosticsScreen.kt` into route/content/row components.
- Modify: `DiagnosticsViewModel.kt` only for filter/state mapping.
- Add empty, populated, expanded, and error tests.

**Acceptance:**

- Compact severity filter/count summary.
- Share and Clear do not compete equally when there are no logs.
- Rows show severity, message, time, source, and disclosure hierarchy.
- Empty state explains that no local crashes were captured.

**Commit:** `polish(diagnostics): clarify local failures and actions`

---

# PHASE 5 — Daily-use surface redesign

## Task 27: Redesign Memory

**Files:**

- Split `MemoryScreen.kt` into route/content/list/card/filter/editor pieces.
- Modify `MemoryViewModel.kt` only where UI state composition improves.
- Add Compose state tests.

**Acceptance:**

- Search/filter controls remain reachable without dominating the screen.
- Header, search, filters, and always-visible actions consume no more than 40% of the compact viewport; destructive rebuild/clear operations move into a secondary overflow or contextual action area.
- The memory list receives the remaining viewport and can scroll as one coherent surface—no fixed ~600dp control stack before item one.
- Memory content, category, importance, tags, source, and age have a clear hierarchy.
- Add/edit/delete/undo/bulk actions remain intact.
- Loading, empty, no-search-results, save/delete/rebuild error, and populated states are distinct.

**Commit:** `polish(memory): rebuild dense searchable memory surface`

---

## Task 28: Redesign History

**Files:**

- Split `HistoryScreen.kt` into route/content/row/action components.
- Preserve pin, rename, delete, fork, search, model label, and export.

**Acceptance:**

- Rows are 72–88dp based on metadata.
- Pinned and recent sections are obvious without giant cards.
- Long-press is not the only way to discover rename/delete.
- Failed delete, rename, share, fork, and export actions surface an error with retry where safe.
- Empty/search-empty/loading/error states have recovery actions.

**Commit:** `polish(history): compact conversation browsing and actions`

---

## Task 29: Unify Tasks and Reminders

**Files:**

- Split `TasksScreen.kt` into route/content/list/filter/editor components.
- Modify: `ui/viewmodel/TasksViewModel.kt` only for normalized task/reminder presentation state.
- Modify `RemindersScreen.kt`
- Modify: `ui/viewmodel/RemindersViewModel.kt` only for normalized presentation state.
- Modify `ReminderEditorDialog.kt`
- Create shared `AuraEditorSheet.kt` if not already created.

**Acceptance:**

- Pending/done/overdue hierarchy is immediately readable.
- `TasksViewModel` and `RemindersViewModel` consume one observable reminder source; an edit from either route appears immediately in the other without reopening or manual refresh.
- Priority, due date, description, and tags are visible without opening every row.
- Reminder create/edit/cancel is keyboard-safe and has explicit loading and failure states.
- Tabs/filters are 48dp, not oversized icon stacks.
- Task totals and reminder totals are labeled separately; no subtitle silently combines unlike counts.
- Clear completed remains confirmed and secondary.

**Commit:** `polish(tasks): unify task and reminder workflows`

---

## Task 30: Rebuild Hands and the Hand editor

**Files:**

- Split `HandsScreen.kt` into route/content/cards/history.
- Modify: `ui/viewmodel/HandsViewModel.kt` only for normalized editor/run presentation state.
- Split `HandEditorDialog.kt` into stateful editor components.
- Reuse `ToolArgForm.kt`.

**Acceptance:**

- Automations/History use a compact 48dp text tab row with inline counts.
- Creation remains discoverable without a jarring disappearing FAB: use an inline/header action or preserve a stable action position across tabs.
- Hand cards prioritize name, enabled state, trigger, schedule, last result, and primary Run action.
- Run history supports status filtering and complete empty/loading/error states.
- Editor uses progressive disclosure: basics → trigger/schedule → steps → gates/variables → review.
- Run state, skipped/failure reasons, edit, duplicate, and delete remain visible.
- No giant blank viewport around one run record.

**Commit:** `polish(hands): rebuild automation cards editor and history`

---

## Task 31: Redesign Tools

**Files:**

- Split `ToolsScreen.kt` into route/content/search/category/row/detail.
- Modify `ToolsViewModel.kt` only for filters/state.

**Acceptance:**

- Search and category filters.
- Risk/permission badges use consistent semantic colors.
- Tool descriptions and parameters are readable without opening raw JSON.
- Run/permission/approval outcomes use shared status components.

**Commit:** `polish(tools): add compact searchable tool catalog`

---

## Task 32: Redesign Proactive history

**Files:**

- Split `ProactiveHistoryScreen.kt` into route/content/timeline/card/filter.
- Modify `ProactiveHistoryViewModel.kt` as needed.

**Acceptance:**

- Chronological timeline with event type, age, context, status, and action.
- Morning brief, calendar, memory decay, and location events have meaningful tap actions.
- Read/unread/filter state does not require oversized cards.
- Empty and permission-disabled states point to the exact setting.

**Commit:** `polish(proactive): turn events into an actionable timeline`

---

## Task 33: Redesign Knowledge Graph

**Files:**

- Split `KnowledgeGraphScreen.kt` into route/content/canvas/controls/list fallback.
- Modify `KnowledgeGraphViewModel.kt` only for display state.

**Acceptance:**

- Graph canvas gets maximum usable area.
- Search/type filters collapse when not needed.
- Node details use a sheet rather than covering the graph.
- Empty graph teaches how facts create nodes.
- List fallback remains useful when the graph is sparse or large.

**Commit:** `polish(graph): prioritize graph canvas and node exploration`

---

# PHASE 6 — Voice and secondary entry points

## Task 34: Unify voice overlays with the main composer

**Files:**

- Modify: `ui/voice/VoiceOverlay.kt`
- Modify: `ui/voice/ContinuousVoiceOverlay.kt`
- Modify: `ui/voice/VoiceViewModel.kt` if display-state mapping needs normalization
- Modify: `ui/voice/ContinuousVoiceViewModel.kt` if display-state mapping needs normalization
- Modify relevant Chat composer integration.
- Extend: `app/src/test/kotlin/com/aura/ui/voice/HoldToTalkTranscriptMirrorTest.kt`
- Add focused voice-state tests for any ViewModel state changes.

**Acceptance:**

- Listening/thinking/speaking/error/cancel states share the same visual language.
- Tap-to-speak, hold-to-talk, and continuous voice each have an explicit labeled entry point; continuous voice is no longer wired-but-invisible.
- Orb/motion is restrained and stops when invisible.
- Transcript is readable and does not jump layout.
- Push-to-talk and continuous modes are unmistakably different.

**Commit:** `polish(voice): unify listening thinking and speaking states`

---

## Task 35: Polish Quick Ask and widget configuration

**Files:**

- Modify: `widget/QuickAskActivity.kt`
- Modify: `widget/WidgetConfigActivity.kt`
- Modify: `widget/AskAuraWidget.kt`
- Modify widget XML/colors as required.

**Acceptance:**

- Widget and overlay use semantic theme colors.
- Quick Ask clearly handles no provider/no model/error/success.
- Configuration uses the live catalog and never hardcodes a model.
- Overlay proportions work on compact portrait and landscape.

**Commit:** `polish(widget): align quick ask and configuration with Aura`

---

# PHASE 7 — Cross-screen responsive and interaction audit

## Task 36: Run the loading/error/empty/populated state audit

**Objective:** Ensure every route has complete states and recovery.

**Files:** all route/content files created above.

**Required state matrix:**

| Surface | Loading | Empty | Populated | Error | Recovery |
|---|---:|---:|---:|---:|---:|
| Onboarding/provider | ✓ | ✓ | ✓ | ✓ | Save/test/retry |
| Home | ✓ if needed | ✓ | ✓ | ✓ | Refresh/open target |
| Chat | ✓ | ✓ | ✓ | ✓ | Retry/change model/settings |
| Model picker | ✓ | ✓ | ✓ | ✓ | Refresh/settings |
| Memory | ✓ | ✓ | ✓ | ✓ | Retry/add/import |
| History | ✓ | ✓ | ✓ | ✓ | Retry/new chat |
| Tasks/reminders | ✓ | ✓ | ✓ | ✓ | Retry/add |
| Hands/history | ✓ | ✓ | ✓ | ✓ | Retry/add |
| Tools | ✓ | ✓ | ✓ | ✓ | Retry/settings |
| Proactive | ✓ | ✓ | ✓ | ✓ | Permission/settings |
| Graph | ✓ | ✓ | ✓ | ✓ | Retry/open memory |
| Settings/providers | ✓ | ✓ | ✓ | ✓ | Save/test/clear |
| Diagnostics | ✓ | ✓ | ✓ | ✓ | Refresh/share/clear |
| Voice/Quick Ask | ✓ | ✓ | ✓ | ✓ | Retry/change model |

**Commit:** `polish(ui): complete cross-screen state recovery`

---

## Task 37: Run the proportion, IME, font-scale, and motion audit

**Objective:** Prove the app behaves outside one happy screenshot.

**Viewports:**

- 360 × 800dp compact
- 393 × 851dp reference
- 600 × 960dp medium/tablet
- compact landscape

**Modes:** light, dark, font scale 1.0, font scale 1.3, keyboard open.

**Checks:**

- no clipped text/actions;
- no double inset;
- no composer/editor hidden by IME;
- no unintended >30% dead region;
- minimum readable contrast;
- 48dp touch targets;
- infinite transitions stop when inactive;
- animations respect system animator scale where practical.

**Commit:** `fix(ui): harden responsive proportions keyboard and motion`

---

# PHASE 8 — Visual proof, full verification, and release

## Task 38: Capture and review the complete screenshot matrix

**Objective:** Visually inspect the actual app, not filenames or test teardown artifacts.

**Matrix minimum:**

- every primary/secondary route in light and dark;
- empty and populated state for every data screen;
- loading/error/no-provider/partial-provider/success picker;
- Chat empty, populated, streaming, tool, citation, image, error, keyboard;
- onboarding, app lock, voice modes, Quick Ask, widget config;
- reference compact viewport plus at least one smaller/medium viewport.

**Process:**

1. Run unit/build gates.
2. Run connected tests.
3. Reinstall exact APK because connected tests may uninstall it.
4. Start Aura.
5. Assert foreground package/activity.
6. Capture matrix.
7. Inspect pixels and XML semantics.
8. Fix every P0/P1 visual defect found.
9. Recapture changed surfaces.

**No commit message may say “SOTA” or “visually verified” until this task passes.**

**Commit:** `test(ui): verify complete Aura visual state matrix`

---

## Task 39: Run the complete release gate and ship the APK

**Objective:** Finish with a reproducible, installable artifact and green CI.

**Files:**

- Modify: `app/build.gradle.kts` (`versionCode` / `versionName`)
- Modify: release notes/documentation required by the repository
- Produce: `app/build/outputs/apk/debug/app-debug.apk`
- Produce: `app/build/outputs/apk/release/app-release.apk`
- Copy debug artifact to: `releases/aura-debug-v<version>.apk`

**Local gates:**

```bash
./gradlew \
  :aura-core:testDebugUnitTest \
  :app:testDebugUnitTest \
  :aura-core:connectedDebugAndroidTest \
  :app:connectedDebugAndroidTest \
  :app:lintDebug \
  :app:assembleDebug \
  :app:assembleRelease \
  --rerun-tasks
```

**Real-provider manual gates:**

1. Install on the user’s device/emulator without exposing the key.
2. Save and test the actual Ollama Cloud key.
3. Verify the live model count is non-zero.
4. Choose a default.
5. Kill/relaunch.
6. Confirm the model remains selected.
7. Send one real message successfully.
8. Open picker and force refresh.
9. Confirm partial/network failure recovery by temporarily disabling network.

**Artifact gates:**

- `aapt dump badging` confirms intended version/package.
- Copy exact APK to `releases/aura-debug-v<version>.apk`.
- Verify SHA-256.
- Push source commits.
- Watch GitHub CI to completion; fix/re-push until every job is green.
- Publish APK through GitHub Releases, not git history.

**Recommended release:** next minor version after `0.10.2`, because this changes the app’s model contract and entire visual shell.

**Commit:** `release(android): ship model reliability and UI recovery`

---

## 6. Screen-by-screen definition of done

| Surface | Definition of done |
|---|---|
| Startup | Correct light/dark launch canvas, branded gate, no tiny isolated spinner or flash |
| Onboarding | Valid provider + selected model before normal completion; honest local-only skip |
| Home | One primary assistant action, contextual priority, compact secondary destinations, no accidental lower-half void |
| Chat | Readable both themes, compact header, no fake model, keyboard-safe composer, coherent messages/tool/citations |
| Model picker | Live/cached/partial/error states, provider groups, direct Settings CTA, explicit selection scope |
| Settings | Focused subroutes, explicit key save/test, role model selectors, no 961-line mega-page |
| Memory | Dense search/filter/list, complete CRUD/bulk/undo states |
| History | Compact rows, model metadata, pin/rename/delete/fork/export discoverable |
| Tasks | Priority/due/status visible, compact filters, keyboard-safe editor |
| Reminders | Upcoming/past clarity, edit/cancel, actionable empty state |
| Hands | Compact tabs/cards, progressive editor, useful run history/results |
| Tools | Search/group/risk clarity, approval and permission outcomes |
| Proactive | Actionable chronological timeline, setting/permission recovery |
| Knowledge graph | Canvas-first composition, filters/details that do not crush the graph |
| Profile/Identity | Dirty/saving/error state, safe reset, keyboard-safe editing |
| Diagnostics | Dense severity-first logs, clear share/clear hierarchy |
| Voice | Coherent listening/thinking/speaking states, controlled motion |
| Quick Ask/widget | Theme-aware, model-aware, usable no-model/error recovery |
| Navigation | Compact, theme-aware, correct route state, no duplicate system inset |

---

## 7. Test inventory to create or strengthen

### Core unit tests

- `ProviderKeysTest`
- `ModelCatalogRepositoryTest`
- per-provider MockWebServer catalog tests
- `ModelRolePreferencesTest`
- `UserPreferencesModelMigrationTest`

### App ViewModel tests

- `ProviderCredentialViewModelTest`
- `ChatModelSelectionTest`
- `OnboardingModelFlowTest`
- state mapping tests for each redesigned screen

### Compose UI tests

Minimum target: **50 focused Compose state/interaction tests** across the matrix below. Prefer one behavior per test over monolithic screenshots.

- model picker state matrix;
- compact header with long/no model;
- light/dark token correctness;
- bottom navigation selection;
- inset position assertions;
- keyboard-safe composer/editor;
- empty/error recovery CTAs;
- core route content tests using pure state.

### Instrumentation flow tests

- provider key → live model list → default selection → process recreation → Chat send enabled;
- invalid key stays invalid;
- successful provider remains visible when another times out;
- fresh no-provider app never displays a fake active model;
- screenshot harness rejects launcher foreground.

---

## 8. Execution discipline

For every task:

1. Re-read target files; verify the issue still exists.
2. Write the failing targeted test first.
3. Run it and record the expected failure.
4. Implement the smallest complete fix.
5. Run the targeted test.
6. Compile the affected module.
7. For visible tasks, install and inspect the changed surface before reporting it.
8. Commit atomically with the exact task intent.
9. Continue immediately to the next task; no “should I continue?” pauses.

Checkpoint gates:

- after Phase 1: all model tests + real picker flow;
- after Phase 2: full unit suite + debug build + light/dark shell screenshots;
- after Phase 3: Chat/Home/Onboarding screenshot matrix;
- after each Phase 5 pair: full app unit suite + build;
- final: complete gates in Task 39.

Do not kill delegated agents without the user’s permission. Do not claim completion without tool output from the same execution turn.

---

## 9. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Nullable active model ripples through the agent loop | Introduce typed state first; compile-fix every boundary; never use empty-string sentinel |
| Provider APIs paginate or vary response shape | Keep parsing in provider adapters; MockWebServer fixtures per provider; surface partial state |
| Cached model data is mistaken for credential validity | Separate `LiveVerificationResult` from `ModelCatalogSnapshot`; UI labels stale cache |
| Theme migration breaks many files at once | Add semantic palette, migrate six direct-dark files, then make raw palettes inaccessible |
| Inset changes cause new overlaps | Root ownership contract + automated position tests + keyboard screenshots |
| Massive screen edits become unreviewable | Route/content split and one screen per atomic commit |
| Visual screenshots capture launcher again | Foreground package/activity assertion is mandatory and automated |
| Connected tests uninstall the app | Screenshot script always reinstalls exact APK after connected tests |
| Existing user model becomes unavailable | Preserve as unavailable state and ask for explicit replacement; no silent model switch |
| UI polish accidentally removes functionality | Per-screen acceptance inventory explicitly lists all actions to preserve |
| Light theme remains lower quality | Every visual task requires both light and dark captures before completion |

---

## 10. Explicit non-goals

- No Play Store publishing work.
- No analytics or telemetry.
- No multi-user/team features.
- No backend feature rewrite unrelated to model selection or visible UX.
- No removal of power-user capabilities to make screens simpler.
- No forced-dark workaround.
- No 1:1 port of Aura Web’s desktop layout.
- No invented hardcoded model fallback.
- No claim that unit-test count alone proves user-facing quality.

---

## 11. Final acceptance checklist

### Model reliability

- [ ] Rapid key typing never persists a partial key.
- [ ] Save & test is explicit and survives restart.
- [ ] Invalid credentials are never marked verified.
- [ ] Provider refresh runs concurrently with a per-provider timeout.
- [ ] One provider failure cannot hide another provider’s models.
- [ ] No concrete production model IDs remain.
- [ ] Fresh install has no fake active model.
- [ ] Chat cannot send without a real selected model.
- [ ] Actual Ollama Cloud catalog loads and one real message sends.

### UI system

- [ ] No direct dark/light raw token use in screen/component code.
- [ ] Every route is readable in system light and dark mode.
- [ ] Root shell is the sole system-inset owner.
- [ ] Bottom navigation meets the 60–64dp contract.
- [ ] Header and composer dimensions meet the compact contract.
- [ ] No core screen has accidental giant dead regions.
- [ ] Every screen has loading/empty/error/populated behavior and recovery.
- [ ] All interactive visuals have at least 48dp touch targets.
- [ ] Keyboard and font scale 1.3 do not clip critical content.

### Verification

- [ ] Unit, connected, lint, debug build, and release build gates pass.
- [ ] Screenshot harness verifies Aura is foreground.
- [ ] Full light/dark state matrix has been inspected pixel-by-pixel.
- [ ] Exact APK metadata and checksum recorded.
- [ ] GitHub CI is green.
- [ ] Release APK is attached to GitHub Release and available at the documented local path.
