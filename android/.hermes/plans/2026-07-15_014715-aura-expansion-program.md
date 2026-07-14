# Aura Android Expansion Program — Implementation Plan

> **For Hermes:** Execute this plan task-by-task using strict TDD, atomic commits, phase checkpoints, and emulator visual verification. Do not pause for permission between tasks.

**Goal:** Close Aura Android’s remaining trust and automation gaps, add a persisted Creative Writing and World Simulation Studio, expand personality controls, register current API providers through live catalogs, add capability APIs, and deliver a visually verified mobile experience.

**Architecture:** Existing infrastructure is reused whenever possible: `MemoryAugmentedAgenticLoop` remains the single agent runtime, `OpenAiCompatProvider` remains the shared OpenAI-compatible transport, and the existing Room/backup patterns remain canonical. New creative data gets its own Room database because it has an independent lifecycle. Personality remains a layered prompt concern stored in DataStore, not a second identity source. No model identifiers are hardcoded; every newly registered provider must discover models from its live `/models` endpoint.

**Tech Stack:** Kotlin 1.9, Jetpack Compose Material 3, Hilt, Room, WorkManager, kotlinx.serialization, OkHttp/SSE, MockK, kotlinx-coroutines-test, Android emulator.

**Baseline:** Branch `feat/tier-1-friction`, HEAD `991af91`, `:aura-core:testDebugUnitTest` + `:app:testDebugUnitTest` green, 754 XML-counted tests, 36 registered tools, 8 LLM providers.

**Prior artifacts:** `.hermes/plans/` contains only `2026-07-14-audit-remediation-final-verification.md`; this is a new program, not a superseding plan.

---

## Verified false or intentionally excluded claims

- `ContinuousVoiceViewModel` does surface STT errors through `VoiceModeUiState.error`; do not rewrite voice DI as a “fix.”
- `VoiceModule` being empty is not broken: `SpeechToText` and `TextToSpeech` use constructor injection.
- Email/SMS tools intentionally open local composer intents and remain `WRITE_LOCAL`; they do not send autonomously.
- Message reactions, profile save feedback, clickable links, and history multi-select are already shipped.
- No Play Store, multi-user, billing, enterprise RBAC, or team features: Aura Android is a personal sideloaded app.

---

# Phase 1 — Trust, provenance, and automation closure

## Task 1: Hands approval and permission recovery

**Objective:** A Hand run that stops for approval or permission can be completed from run history without restarting from step one.

**Files:**
- Modify: `aura-core/src/main/kotlin/com/aura/agent/ToolRegistry.kt` (`ToolContext` explicit approval)
- Modify: `aura-core/src/main/kotlin/com/aura/agent/ToolExecutor.kt`
- Modify: `aura-core/src/main/kotlin/com/aura/hands/HandRepository.kt`
- Modify: `app/src/main/kotlin/com/aura/ui/viewmodel/HandsViewModel.kt`
- Modify: `app/src/main/kotlin/com/aura/ui/screens/HandsScreen.kt`
- Test: `aura-core/src/test/kotlin/com/aura/agent/ToolExecutorRemoteCostApprovalTest.kt`
- Test: `aura-core/src/test/kotlin/com/aura/hands/HandRepositoryTest.kt`
- Test: `app/src/test/kotlin/com/aura/ui/viewmodel/HandsViewModelTest.kt`

**TDD contract:**
1. Explicit UI approval in `ToolContext` authorizes exactly the named tool call; unrelated remote-cost tools remain blocked.
2. Repository can resume at `failedStep - 1` using persisted variables.
3. Permission status exposes the exact permission and retries only after grant.
4. History card renders `Approve & resume` or `Grant & resume` actions.

**Verification:** targeted three test classes, then module tests.

**Commit:** `feat(android): resume Hands after approval or permission`

## Task 2: Remove the phantom LocationArrived feature

**Objective:** Stop advertising a location automation that has no producer. Keep `location_now` as the honest on-demand capability.

**Files:**
- Modify: `aura-core/src/main/kotlin/com/aura/proactive/ProactiveEventBus.kt`
- Modify: `aura-core/src/main/kotlin/com/aura/proactive/ProactiveEvents.kt`
- Modify: `app/src/main/kotlin/com/aura/ui/screens/ProactiveHistoryScreen.kt`
- Modify: `app/src/main/kotlin/com/aura/ui/screens/home/HomeBriefCard.kt`
- Modify: `app/src/main/kotlin/com/aura/ui/screens/home/HomeContent.kt`
- Test: `aura-core/src/test/kotlin/com/aura/proactive/ProactiveEventsTest.kt`

**TDD contract:** no serialization/deserialization branch or UI copy claims passive place arrival; legacy stored rows decode safely as unknown/ignored events rather than crashing.

**Commit:** `fix(android): remove phantom location-arrival automation`

## Task 3: Real memory and knowledge provenance

**Objective:** Every newly learned memory/KG fact records the originating conversation and turn timestamp, and the UI can navigate back to that conversation.

**Files:**
- Modify: `aura-core/src/main/kotlin/com/aura/memory/MemoryEntity.kt`
- Modify: `aura-core/src/main/kotlin/com/aura/kg/KgEntities.kt`
- Modify: `aura-core/src/main/kotlin/com/aura/memory/MemoryDatabase.kt`
- Modify: `aura-core/src/main/kotlin/com/aura/memory/MemoryModule.kt` (v3→v4 migration)
- Modify: `aura-core/src/main/kotlin/com/aura/kg/ConversationKgExtractor.kt`
- Modify: `aura-core/src/main/kotlin/com/aura/agent/MemoryAugmentedAgenticLoop.kt`
- Modify: `aura-core/src/main/kotlin/com/aura/memory/MemoryStore.kt`
- Modify: `aura-core/src/main/kotlin/com/aura/backup/AuraBackup.kt`
- Modify: `aura-core/src/main/kotlin/com/aura/backup/BackupManager.kt` (schema v4)
- Modify: `app/src/main/kotlin/com/aura/ui/screens/MemoryScreen.kt`
- Modify: `app/src/main/kotlin/com/aura/ui/screens/KnowledgeGraphScreen.kt`
- Modify: `app/src/main/kotlin/com/aura/ui/nav/NavGraph.kt`
- Tests: migration, backup roundtrip, extractor/store provenance, screen callback contracts.

**Schema:** `memories.sourceConversationId TEXT NOT NULL DEFAULT ''`, `memories.sourceTurnTimestamp INTEGER NOT NULL DEFAULT 0`, matching columns on `kg_nodes` and `kg_edges`; new indices on conversation IDs.

**TDD contract:** legacy rows migrate with empty provenance; new extraction writes real conversation ID; missing/deleted conversations render “Source unavailable”; valid source opens `chat?convId=`.

**Commit:** `feat(android): trace memories and knowledge back to source chats`

## Task 4: Full-agent Quick Ask

**Objective:** Quick Ask uses the same agent runtime, identity, profile, memory, tools, failover, and persistence as Chat while keeping the compact overlay.

**Files:**
- Modify: `app/src/main/kotlin/com/aura/widget/QuickAskActivity.kt`
- Reuse: `app/src/main/kotlin/com/aura/ui/viewmodel/ChatViewModel.kt`
- Create if needed: `app/src/main/kotlin/com/aura/widget/QuickAskContent.kt`
- Test: `app/src/test/kotlin/com/aura/widget/QuickAskAgentContractTest.kt`

**TDD contract:** sending through Quick Ask creates a persisted conversation, uses resolved identity/profile/memory, receives tool states, and cancels cleanly when Activity closes. The overlay labels itself “Full Aura” rather than a context-free provider call.

**Commit:** `feat(android): run Quick Ask through the full Aura agent`

## Task 5: Visible provider failover

**Objective:** Aura tells the user when and why it switched providers and records the model that actually answered.

**Files:**
- Modify: `aura-core/src/main/kotlin/com/aura/agent/MemoryAugmentedAgenticLoop.kt`
- Modify: `app/src/main/kotlin/com/aura/ui/viewmodel/ChatSendController.kt`
- Modify: `app/src/main/kotlin/com/aura/ui/viewmodel/ChatViewModel.kt`
- Modify: `app/src/main/kotlin/com/aura/ui/screens/chat/ChatContent.kt`
- Create: `app/src/main/kotlin/com/aura/ui/components/ProviderFailoverBanner.kt`
- Tests: loop failover metadata + ViewModel warning state.

**TDD contract:** `AgentEvent.Warning` reaches UI; banner names old/new provider without secrets; final `Conversation.model` is the actual model; warning is dismissible and survives until the turn completes.

**Commit:** `feat(android): surface provider failover and actual model`

## Task 6: Inline citation markers

**Objective:** `[1]`, `[2]` markers in assistant prose become interactive citation annotations instead of duplicate raw text plus chips.

**Files:**
- Modify: `app/src/main/kotlin/com/aura/ui/components/MarkdownText.kt`
- Modify: `app/src/main/kotlin/com/aura/ui/components/StreamingText.kt`
- Modify: `app/src/main/kotlin/com/aura/ui/components/MessageBubble.kt`
- Test: `app/src/test/kotlin/com/aura/ui/components/MarkdownCitationTest.kt`

**TDD contract:** only valid citation indices are annotated; code spans/blocks are untouched; incomplete streaming marker is rendered safely; tapping an inline marker opens the same source preview as the chip.

**Commit:** `fix(android): render inline citation markers as source links`

## Task 7: Extraction feedback and memory edit history

**Objective:** Best-effort learning is observable and memory edits are reversible from the UI.

**Files:**
- Modify: `aura-core/src/main/kotlin/com/aura/kg/ConversationKgExtractor.kt` (status flow, failure detail without chat interruption)
- Modify: `app/src/main/kotlin/com/aura/ui/viewmodel/MemoryViewModel.kt`
- Modify: `app/src/main/kotlin/com/aura/ui/screens/MemoryScreen.kt`
- Modify: `app/src/main/kotlin/com/aura/ui/viewmodel/ChatViewModel.kt`
- Tests: extractor state, edit-history restore, undo snackbar.

**TDD contract:** chat can show “Knowledge updated” or a quiet warning; Memory exposes edit timeline, diff, restore, and delete Undo.

**Commit:** `feat(android): expose learning status and memory history`

---

# Phase 2 — Personality Studio

## Task 8: Structured personality profile

**Objective:** Layer editable behavior controls over the canonical SOUL without creating a second identity source.

**Files:**
- Create: `aura-core/src/main/kotlin/com/aura/agent/PersonalityProfile.kt`
- Create: `aura-core/src/main/kotlin/com/aura/agent/PersonalityPromptComposer.kt`
- Modify: `aura-core/src/main/kotlin/com/aura/data/UserPreferences.kt`
- Modify: `aura-core/src/main/kotlin/com/aura/agent/Brain.kt`
- Modify: `aura-core/src/main/kotlin/com/aura/backup/AuraBackup.kt`
- Modify: `aura-core/src/main/kotlin/com/aura/backup/BackupManager.kt`
- Tests: pure composer, serialization, backup roundtrip.

**Profile fields:** warmth, directness, playfulness, detail level, creativity, emotional expressiveness (0–100); response principles; boundaries; likes/dislikes; relationship notes; selected preset.

**TDD contract:** base SOUL is always first; structured profile is clearly delimited as user customization; blank/default profile adds no noise; values clamp safely; prompt injection-like profile text remains lower-priority user customization.

**Commit:** `feat(android): add layered personality profiles`

## Task 9: Personality Studio UI

**Objective:** Replace raw-prompt-only personality customization with a premium structured editor while retaining the advanced SOUL editor.

**Files:**
- Create: `app/src/main/kotlin/com/aura/ui/screens/PersonalityStudioScreen.kt`
- Create: `app/src/main/kotlin/com/aura/ui/components/PersonalityTraitSlider.kt`
- Modify: `app/src/main/kotlin/com/aura/ui/settings/SettingsViewModel.kt`
- Modify: `app/src/main/kotlin/com/aura/ui/screens/SettingsScreen.kt`
- Modify: `app/src/main/kotlin/com/aura/ui/nav/NavGraph.kt`
- Tests: preset application and ViewModel persistence.

**Presets:** Balanced, Precise Analyst, Warm Companion, Creative Muse, Worldmaster. Presets never select a model.

**Commit:** `feat(android): build Personality Studio`

## Task 10: Specialist expansion and controls

**Objective:** Add Writer, Worldbuilder, and Simulator specialists and finally expose per-specialist tool allowlists.

**Files:**
- Modify: `aura-core/src/main/kotlin/com/aura/agent/Specialist.kt`
- Modify: `aura-core/src/main/kotlin/com/aura/agent/SpecialistRouter.kt`
- Modify: `app/src/main/kotlin/com/aura/ui/screens/SettingsScreen.kt`
- Modify: `app/src/main/kotlin/com/aura/ui/settings/SettingsViewModel.kt`
- Tests: routing, prompt overrides, tool allowlist persistence.

**Commit:** `feat(android): add writing, worldbuilding, and simulation specialists`

---

# Phase 3 — Creative Writing and World Simulation Studio

## Task 11: Creative project database

**Objective:** Persist creative projects independently from transient chats.

**Files:**
- Create: `aura-core/src/main/kotlin/com/aura/creative/CreativeProject.kt`
- Create: `aura-core/src/main/kotlin/com/aura/creative/CreativeProjectEntity.kt`
- Create: `aura-core/src/main/kotlin/com/aura/creative/CreativeProjectDao.kt`
- Create: `aura-core/src/main/kotlin/com/aura/creative/CreativeDatabase.kt`
- Create: `aura-core/src/main/kotlin/com/aura/creative/CreativeModule.kt`
- Create: `aura-core/src/main/kotlin/com/aura/creative/CreativeRepository.kt`
- Tests: DAO/repository serialization and CRUD.

**Project structure:** id, title, kind (`story`, `world`, `simulation`, `game`, `other`), premise, genre, tone, world rules, characters, locations, factions, timeline, lore, open threads, style guide, simulation state, notes, linkedConversationIds, createdAt, updatedAt.

**Commit:** `feat(android): persist creative worlds and stories`

## Task 12: Worldbuilding and writing prompt engine

**Objective:** Turn project state into disciplined prompts for ideation, drafting, editing, continuity checking, and simulation.

**Files:**
- Create: `aura-core/src/main/kotlin/com/aura/creative/CreativeMode.kt`
- Create: `aura-core/src/main/kotlin/com/aura/creative/CreativePromptBuilder.kt`
- Create: `aura-core/src/main/kotlin/com/aura/creative/SimulationScenario.kt`
- Test: `aura-core/src/test/kotlin/com/aura/creative/CreativePromptBuilderTest.kt`

**Modes:** brainstorm, outline, draft scene, rewrite, dialogue pass, style pass, character interview, expand lore, continuity audit, timeline audit, consequence simulation, faction simulation, alternate history, emergent-event simulation.

**TDD contract:** prompts include only relevant project sections, label project data as context not system instruction, preserve user constraints, and ask simulations to return state deltas plus narrative—not merely prose.

**Commit:** `feat(android): add creative and simulation prompt engine`

## Task 13: Agent-accessible creative tools

**Objective:** Let the agent read and safely update creative projects during chat.

**Files:**
- Create: `aura-core/src/main/kotlin/com/aura/tools/CreativeProjectTool.kt`
- Modify: `aura-core/src/main/kotlin/com/aura/tools/ToolsModule.kt`
- Modify: `aura-core/src/main/kotlin/com/aura/agent/Specialist.kt`
- Tests: read/list/update operations, malformed patch rejection, incognito blocking of writes.

**Tools:** `creative_project_list`, `creative_project_read`, `creative_project_update`. Updates use typed field patches, never arbitrary SQL/JSON replacement.

**Commit:** `feat(android): let Aura read and evolve creative projects`

## Task 14: Creative Studio UI and navigation

**Objective:** Add a premium fifth top-level `Create` surface with projects, templates, editor, and simulation launcher.

**Files:**
- Create: `app/src/main/kotlin/com/aura/ui/screens/creative/CreativeStudioScreen.kt`
- Create: `app/src/main/kotlin/com/aura/ui/screens/creative/CreativeProjectEditor.kt`
- Create: `app/src/main/kotlin/com/aura/ui/screens/creative/CreativeLaunchSheet.kt`
- Create: `app/src/main/kotlin/com/aura/ui/viewmodel/CreativeStudioViewModel.kt`
- Modify: `app/src/main/kotlin/com/aura/ui/nav/AuraBottomNavigation.kt`
- Modify: `app/src/main/kotlin/com/aura/ui/nav/NavGraph.kt`
- Tests: ViewModel CRUD, template prompt routing, nav route contract.

**Templates:** Novel, Short Story, Screenplay, RPG Campaign, Fantasy World, Sci-Fi Civilization, Alternate History, Political Simulation, Character Study.

**Commit:** `feat(android): launch Creative and World Studio`

## Task 15: Creative backup and continuity report

**Objective:** Include projects in backups and give each project a deterministic continuity-health report.

**Files:**
- Modify: `aura-core/src/main/kotlin/com/aura/backup/AuraBackup.kt` (schema v5)
- Modify: `aura-core/src/main/kotlin/com/aura/backup/BackupManager.kt`
- Create: `aura-core/src/main/kotlin/com/aura/creative/ContinuityAnalyzer.kt`
- Modify: Creative Studio UI.
- Tests: backup roundtrip, duplicate names, contradictory timeline dates, unresolved references.

**Commit:** `feat(android): back up worlds and audit continuity`

---

# Phase 4 — Provider and capability expansion

## Task 16: Register five live-catalog LLM providers

**Objective:** Add Mistral, xAI, Together AI, Cerebras, and NVIDIA NIM with zero hardcoded model IDs.

**Files:**
- Modify: `aura-core/src/main/kotlin/com/aura/providers/ProviderModule.kt`
- Modify: `aura-core/src/main/kotlin/com/aura/providers/ProviderKeys.kt`
- Modify: onboarding/provider settings lists if they duplicate prefixes.
- Tests: provider registration, live-catalog URL/auth request contracts, key lifecycle.

**Endpoints (must be re-verified against official docs before code):**
- Mistral `https://api.mistral.ai/v1`
- xAI `https://api.x.ai/v1`
- Together `https://api.together.xyz/v1`
- Cerebras `https://api.cerebras.ai/v1`
- NVIDIA `https://integrate.api.nvidia.com/v1`

**TDD contract:** each calls `/models`, uses bearer auth, appears only when configured, and reuses `OpenAiCompatProvider`.

**Commit:** `feat(android): add five live-catalog LLM providers`

## Task 17: Custom OpenAI-compatible provider

**Objective:** Let the user configure one arbitrary OpenAI-compatible endpoint without recompiling Aura.

**Files:**
- Create: `aura-core/src/main/kotlin/com/aura/providers/CustomOpenAiProvider.kt`
- Modify: `ProviderKeys.kt` or create typed `CustomProviderSettings`
- Modify: `ProviderModule.kt`
- Modify: Settings UI/ViewModel and backup preferences.
- Tests: URL normalization, HTTPS-only except explicit localhost, `/models` discovery, key updates without restart.

**Commit:** `feat(android): support custom OpenAI-compatible endpoints`

## Task 18: Exa search and Jina Reader capabilities

**Objective:** Add semantic web search and high-quality page extraction while retaining free DuckDuckGo fallback.

**Files:**
- Create: `aura-core/src/main/kotlin/com/aura/tools/ExaSearchTool.kt`
- Create: `aura-core/src/main/kotlin/com/aura/tools/JinaReaderTool.kt`
- Modify: `ToolsModule.kt`, `ProviderKeys.kt`, Settings provider/API section.
- Tests: auth headers, URL safety, response parsing, truncation, timeout/cancellation.

**Rules:** Exa uses its official key header; Jina Reader never receives private/local URLs and passes through the shared SSRF guard before remote extraction.

**Commit:** `feat(android): add semantic search and reader APIs`

## Task 19: Pluggable media generation providers

**Objective:** Remove the single-provider image-generation bottleneck and give the user OpenAI, Stability, and Pollinations choices without hardcoded model names where catalogs exist.

**Files:**
- Refactor: `aura-core/src/main/kotlin/com/aura/tools/ImageGenTool.kt`
- Create: `aura-core/src/main/kotlin/com/aura/media/ImageGenerationProvider.kt`
- Create provider adapters under `aura-core/src/main/kotlin/com/aura/media/`
- Modify: Settings media/API section and `ProviderKeys.PREFIXES`.
- Tests: provider routing, binary/base64/URL handling, safe filenames, cancellation, no secret in URL.

**Commit:** `feat(android): add pluggable image generation providers`

---

# Phase 5 — Product-surface and UI/UX polish

## Task 20: Generated-image lightbox and gallery

**Objective:** Generated images are first-class artifacts, not naked markdown links.

**Files:**
- Create: `app/src/main/kotlin/com/aura/ui/components/ImageLightbox.kt`
- Create: `app/src/main/kotlin/com/aura/ui/screens/GeneratedMediaScreen.kt`
- Modify: `MessageBubble.kt`, Chat route, Creative Studio.
- Tests: URL recognition and gallery grouping pure functions.

**Commit:** `feat(android): add generated media gallery and lightbox`

## Task 21: Pinned Home, tool controls, and status affordances

**Objective:** Surface high-value state on Home and give the user real control over tool exposure.

**Files:**
- Modify: Home ViewModel/content to show pinned conversations and pending Hand approvals.
- Modify: `ToolsScreen.kt` / `ToolsViewModel.kt` to show risk, permission state, enabled toggle.
- Modify: `UserPreferences.kt` + `ToolExecutor.kt` for disabled tools.
- Modify: bottom navigation to show approval/proactive badges.
- Tests: disabled-tool boundary, pinned ordering, badge counts.

**Commit:** `feat(android): surface pinned work and tool controls`

## Task 22: Extreme visual polish and emulator verification

**Objective:** Harmonize the new and existing surfaces into one premium design and verify actual rendering.

**Files/surfaces:** Creative Studio, Personality Studio, Chat failover/citations/media, Home pinned section, Tools controls, Hands approval cards, bottom navigation.

**Polish contracts:**
- One shared screen shell/header rhythm.
- No fixed-height control blocks that starve lists.
- 48dp interaction targets.
- Empty/loading/error/populated states for every new screen.
- Consistent Aura radii, typography, tonal hierarchy, animation timing.
- Light and dark mode sanity.
- No “settings form dumped into cards” aesthetic.

**Verification:** build/install debug APK on emulator, capture screenshots for Home, Chat with failover/citations, Creative project list/editor/simulation, Personality Studio, Tools, Hands approval, generated-media lightbox. Fix visual defects before completion.

**Commit:** `style(android): polish expansion surfaces on device`

---

# Final verification and delivery

1. `./gradlew :aura-core:testDebugUnitTest :app:testDebugUnitTest`
2. Count XML test results and report delta from 754.
3. `./gradlew :app:lintDebug :app:assembleDebug`
4. Install APK on emulator and run the screenshot matrix from Task 22.
5. `git status --short`; do not stage pre-existing `.hermes/*.png|xml`, controller-build files, or `app/src/main/baseline-prof.txt` unless this program intentionally modifies them.
6. Commit each task atomically; push branch after all gates pass.
7. Check CI with `gh run list` / `gh run watch`; fix until green.
8. Build fresh APK and place it at `releases/aura-debug-v<next-version>.apk`; create a GitHub Release only if requested in this execution thread.

# Phase checkpoints

- After Phase 1: trust flows complete, targeted tests + full unit gate.
- After Phase 2: personality prompt snapshot tests + app tests.
- After Phase 3: creative DB/repository/prompt/UI tests + assembleDebug.
- After Phase 4: provider/tool contract tests + live-catalog mock tests.
- After Phase 5: full tests, lint, assemble, emulator screenshots.

# Definition of done

- Every new behavior was driven by a failing test first.
- No provider model ID is hardcoded.
- Every persisted field is included in backup/restore or explicitly documented as derived/cache data.
- Every new automation has a user-visible control and recovery path.
- Every new UI surface has loading, empty, error, and populated states.
- Full Gradle gates and emulator verification are green.
- Final report lists commits, test delta, APK path, CI status, and any intentionally unshipped item with an exact reason.
