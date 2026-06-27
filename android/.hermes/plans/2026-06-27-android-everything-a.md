# Plan: bring Aura Android to A across the board

Current baseline: B+/A- with weak spots in UI tests, Room migrations, error types, ViewModel size, and module boundaries.

## Goal
Every grade category becomes A by the end of this campaign. Work happens in the order below — each item is an atomic or tightly-scoped commit.

## Phases

### Phase 0 — Baseline (DONE)
- [x] Fix broken `EndToEndTest` (`UserProfileStore` constructor drift)
- [x] Fix `ImageDecoder` API-level lint crash for minSdk 26
- [x] Remove `WorkManagerInitializer` from manifest (on-demand init)
- [x] Run `:aura-core:testDebugUnitTest`, `:app:testDebugUnitTest`, `:app:lintDebug` green

### Phase 1 — Safety first: Room migrations
- [ ] Identify all Room databases and versions
  - ConversationDatabase v1
  - MemoryDatabase v2 (has migration 1→2 already)
  - TaskDatabase v1
  - HandDatabase v1
  - ProactiveEventDatabase v1
  - UserProfileDatabase v1
- [ ] Adopt `exportSchema = true` so migrations are validated
- [ ] Add destructive fallback only for debug builds; production requires explicit migrations
- [ ] Add migration smoke tests using `MigrationTestHelper` for every v2+ database
- [ ] Commit: `feat(android): Room migration hygiene + export schemas`

### Phase 2 — Sealed error hierarchy
- [ ] Add `AuraError` sealed class in `aura-core` (domain error contract)
  - Network, ApiKeyMissing, ModelUnavailable, ToolDenied, ToolFailed, SafetyBlocked, Storage, Unknown
- [ ] Replace `ChatUiState.error: String?` with `ChatUiState.error: AuraError?`
- [ ] Replace `ToolResult.Error.message: String` with `ToolResult.Error(error: AuraError, displayMessage: String? = null)`
- [ ] Replace `AgentEvent.Error(message: String)` with typed error
- [ ] Add `ChatScreen` localized error banner with retry/dismiss and rationale
- [ ] Add unit tests for each error type mapping from providers/tools
- [ ] Commit: `feat(android): sealed error hierarchy + typed UI error states`

### Phase 3 — ChatViewModel decomposition
- [ ] Extract domain services
  - `ChatStateManager` (holds StateFlow, applies reducer-style updates, thread-confined)
  - `ChatStreamProcessor` (collects provider stream, handles tool calls, streaming text)
  - `ChatTurnCoordinator` (load/save conversation, retry, cancellation)
  - `ChatInputCoordinator` (attachments, specialist selection, draft)
- [ ] Remove `runBlocking` usage entirely; use `viewModelScope.launch` + `Dispatchers.Default`/`IO`
- [ ] Move all coroutine helpers to `viewModelScope` with supervisor jobs
- [ ] Add `Main.Immediate` tests for ViewModel state emission order
- [ ] Commit: `refactor(android): decompose ChatViewModel into state + stream + turn coordinators`

### Phase 4 — DI hygiene: reduce EntryPoints
- [ ] Audit every `EntryPointAccessors.fromApplication` call
  - `IncomingShareStore` in ChatScreen
  - Any in Workers / Services
- [ ] Provide app-level `IncomingShareStore` through constructor / `ViewModel` SavedStateHandle or a small app-scoped coordinator
- [ ] Move workers to `@HiltWorker` constructor injection instead of EntryPoints
- [ ] Add compile-time lint rule (detekt / custom) that flags new EntryPoint usage
- [ ] Commit: `refactor(android): eliminate EntryPointAccessors from UI and workers`

### Phase 5 — MoA presets configurable
- [ ] Add `MoaPresetRepository` backed by DataStore/proto or JSON asset + user overrides
- [ ] Define preset data model: name, reference models, aggregator, temps, enabled
- [ ] Load default preset from `assets/moa_presets.json` at startup; merge user edits
- [ ] Add settings UI to enable/disable reference models and pick aggregator
- [ ] Add unit tests for preset merge/validation and fallback when provider missing
- [ ] Commit: `feat(android): configurable MoA presets with user overrides`

### Phase 6 — Compose UI tests
- [ ] Add `compose-ui-test` dependencies to app module
- [ ] Add `ChatScreenTest` covering:
  - send message, typing indicator, message rendering
  - error banner retry/dismiss
  - specialist chips selection
  - model picker sheet
  - deep-mode toggle
  - sources sheet with citations
- [ ] Add `HistoryScreenTest` if exists
- [ ] Add test-only Hilt injection or use manual ViewModel factories in tests
- [ ] Commit: `test(android): Compose UI tests for chat screen`

### Phase 7 — Provider / tool polish
- [ ] Add structured `FinishReason` handling per provider (stop, length, content_filter, tool_calls)
- [ ] Add provider timeout / retry policy tests
- [ ] Ensure all 32 tools return typed results; no raw string exceptions
- [ ] Add test for `DeepResearchTool` happy path with mock search/fetch
- [ ] Commit: `feat(android): provider finish-reason discipline + tool result types`

### Phase 8 — Module split (architectural A)
- [ ] Create new `:aura-platform` module
  - Move `Context`, `Location`, `Contacts`, `Calendar`, `Biometric`, `Notification`, `FileProvider`, `WorkManager`, `TTS` helpers
- [ ] Keep `:aura-core` pure domain: providers, agent loop, memory, tools, models
- [ ] Re-wire Hilt modules; `:app` depends on both, `:aura-core` depends on nothing Android
- [ ] Update Gradle dependency graph; `:aura-core` only uses Kotlin stdlib + Ktor/OkHttp
- [ ] Add `:aura-core` JVM-only unit tests that run faster than Android tests
- [ ] Commit: `refactor(android): split aura-platform from aura-core`

### Phase 9 — Final verification
- [ ] Full `./gradlew clean build connectedCheck` (if emulator available) or `test + lint`
- [ ] Detekt baseline updated
- [ ] Test coverage report: target 70% unit + UI combined
- [ ] README architecture section updated

## Stop criteria
The campaign is complete when every Phase 1–9 item above is checked, `./gradlew test lint` is green, and the self-assessment matches A in each category.

## Deferred (post-A)
- Benchmark tests
- Macrobenchmark for startup
- Figma/UX polish (out of scope for code-grade A)
