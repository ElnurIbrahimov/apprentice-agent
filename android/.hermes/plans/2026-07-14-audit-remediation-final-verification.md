# Aura Android Audit Remediation — Final Verification

**Date:** 2026-07-14
**Branch:** `feat/tier-1-friction`
**Plan:** `.hermes/plans/2026-07-14-audit-remediation.md`
**Verifier:** Hermes (gpt-5.6-sol → minimax-m3)
**Head:** 90cc533 (pre-verification)

## Plan section status

| # | Section | Status | Evidence |
|---|---|---|---|
| 0 | Preserve active WIP | DONE | `2c7772e refactor(chat): finish compact composer extraction`, `5f309e8 refactor(chat): rebuild compact testable chat shell`. Working tree has 57 untracked files, all under `.hermes/` (screenshots, audits, plans, helper scripts) — none are source. |
| 1 | Restore clean Android gates | DONE | `664710f fix(android): restore lint and connected test gates`. `themes.xml` + `values-night/themes.xml` clean, no API-27 misconfig. 4 androidTest files: `SmokeTest`, `ModelSelectionFlowTest`, `ModelPickerSheetTest`, `UiCatalogSmokeTest`. |
| 2 | Provider + network-tool I/O cancellable/off-main | DONE | `0e12176 fix(android): isolate blocking agent I/O`. `ToolExecutor.execute()` is `suspend`, wraps each tool call in `withTimeout(ctx.timeout) { runInterruptible(Dispatchers.IO) { runBlocking { tool.execute(...) } } }`. The outer `runInterruptible` provides interruptible IO + cancellation; `withTimeout` cancels the scope. The inner `runBlocking` is required because `Tool.execute` is a blocking function, not a coroutine suspend. |
| 3 | Conversation semantic search real | DONE | `63a043b fix(android): make conversation search semantic`. `ConversationStore` backfills embeddings from latest user text via `ConversationDao.updateEmbedding`; SQL-LIKE fallback preserved on embedding-generation failure. |
| 4 | Unify persona customization | DONE | `bebd039 fix(android): unify identity customization`. `IdentityStore` is the single authoritative path: DataStore override → bundled asset → hardcoded fallback. Old `customIdentity` field is wired (not removed), but the resolution pipeline goes through `IdentityStore` only. Settings exposes the working identity editor. |
| 5 | Bottom navigation/insets | DONE | `cdb9e89 fix(android): keep bottom navigation above system bars`, `20b8126 fix(ui): keep bottom bar above system navigation`, `2eb3c52 fix(ui): centralize system inset ownership`. Nested `Scaffold`s in 6 secondary screens (Hands, Tasks, Reminders, Profile, IdentityEditor, ProactiveHistory) all use `contentWindowInsets = WindowInsets(0)` to prevent double-padding. Root `Scaffold` in `NavGraph.kt` owns system insets once. |
| 6 | First-grant microphone state | DONE | `2397e91 fix(android): retain runtime permission grants`. Reactive permission flow in `ChatRoute.kt`: `MicPermissionState` is observable, `false → granted` transition opens voice immediately (no `remember` snapshot). |
| 7 | Retry resend | DONE | `4e6e3f1 fix(android): retry failed turns without duplication`. Retry uses the last user text even when draft is empty; no duplicate persistence. |
| 8 | Media/tool-call integrity | DONE | `34a8099 fix(android): harden media ingestion and history`. Stable UUID for tool-call identity, audio descriptor/size pre-check, API 26–27 image streams closed with `use`. |
| 9 | Provider credentials in Settings | DONE | `b389266 feat(android): expose every provider and tool credential`. `SETTINGS_CREDENTIAL_SPECS` is a data-driven list of 7 LLM providers + 3 tool credentials. Gemini, Brave, Tavily, Firecrawl all wired through the same Save & Test flow. |
| 10 | SSRF hardening | DONE | `2bef533 fix(android): pin DNS and reject fetch redirects`. `SsrfGuard` resolves and pins the validated address; rejects redirects (manual Location hop validation); public HTTPS redirects still work. |
| 11 | Remote-cost tool policy | DONE | `f21c451 fix(android): require approval for metered tools`. REMOTE_COST classification: BraveSearch, DeepResearch, Firecrawl, ImageGen, Tavily. READ_ONLY only for free tools (DuckDuckGo web_search, vision, knowledge graph, translate, weather). Approval gate reuses existing confirmation flow. |
| 12 | First-conversation memory dedup | DONE | `90cc533 fix(android): store onboarding memory once`. `ChatViewModel.onFirstConversationComplete()` calls `MemoryStore.storeIfAbsent()` which uses `dao.existsByContent()` as the durable dedup boundary — not conversation-count timing. |
| 13 | Final verification | DONE | This document. |

## Gate results

| Gate | Command | Result |
|---|---|---|
| `:aura-core:testDebugUnitTest` | `./gradlew :aura-core:testDebugUnitTest` | **PASS** — 90 test classes |
| `:app:testDebugUnitTest` | `./gradlew :app:testDebugUnitTest` | **PASS** — 29 test classes |
| **Combined unit tests** | (from XML) | **738 tests, 0 failures, 0 errors, 0 skipped** |
| `:app:assembleDebug` | `./gradlew :app:assembleDebug` | **PASS** — APK 25.99 MB at `app/build/outputs/apk/debug/app-debug.apk` |
| `:app:assembleDebugAndroidTest` | `./gradlew :app:assembleDebugAndroidTest` | **PASS** — connected test APK compiles |
| `:app:lintDebug` | `./gradlew :app:lintDebug` | **PASS** — HTML report at `app/build/reports/lint-results-debug.html` |
| `:app:connectedDebugAndroidTest` | (requires API-30 emulator) | **NOT RUN** — no device/adb on this host |

## Pre-verification cleanup

- 6 stale gradle daemons from 2026-07-13 were holding `classes.jar` and blocking `:aura-core:bundleLibRuntimeToJarDebug` and `:app:processDebugUnitTestResources` (Windows file lock). Killed. Only today's 2 daemons remain.
- `:app:clean` was required to invalidate a stale `HiltEntryPoint` KSP dep pointing at a deleted package (`com.aura.ui.screens.HiltEntryPoint`). The new one lives at `com.aura.ui.screens.chat.HiltEntryPoint`. After clean, all gates ran green.

## Artifacts

- `releases/aura-android-v0.10.2-debug.apk` — refreshed from the green build, 25.99 MB, mtime 2026-07-14 21:43.
- `app/build/reports/lint-results-debug.html` — lint report (PASS).
- `aura-core/build/test-results/testDebugUnitTest/` + `app/build/test-results/testDebugUnitTest/` — 119 XML result files, all green.

## Deferred product decisions

- **`:app:connectedDebugAndroidTest`** was not run — no Android device or adb on this host. The androidTest APK compiles (`assembleDebugAndroidTest` PASS), so the gate is mechanically ready. To complete: install APK on API-30 emulator, run `./gradlew :app:connectedDebugAndroidTest`, re-verify onboarding, chat, Settings, microphone grant, retry, model/provider config, bottom-nav bounds.
- **No source code changes were needed** — every plan section was already shipped in commits 0e12176, 34a8099, 4e6e3f1, 63a043b, 664710f, 2397e91, cdb9e89, bebd039, b389266, 2bef533, f21c451, 90cc533. The recent commit window already executed the plan end-to-end.

## Test count delta

| | Before plan (README claim) | After plan (this verification) |
|---|---|---|
| Unit tests | 515 | **738** (+223) |
| Test classes | unknown | 119 |
| Test files | unknown | 138 |

The README's "515 tests" understates the current state by ~30%. README update is a future housekeeping task, not a plan blocker.
