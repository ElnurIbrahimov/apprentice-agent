# Aura Android — Real Structural Gaps

> **For Hermes:** Execute in order. Verify before each commit that the feature doesn't already exist.

**Goal:** Close the 7 genuine gaps between what's built and what a user needs to actually use the app.

**Current state:** 5-tab bottom nav (Home/Chat/Memory/Settings/Graph). All 5 screens exist with working ViewModels. 213 tests pass. APK builds.

**What already exists and should NOT be rebuilt:**
- SettingsScreen: 6 provider key fields + model picker + configured-provider chips + firstRunComplete flag (SettingsViewModel already persists)
- MemoryScreen: search, category chips, memory cards with decay display, forget button
- HomeScreen: greeting, proactive event cards, recent memories, pending tasks, calendar events
- GraphScreen: node search, neighbor detail, path finding
- ChatScreen: voice overlay, camera/gallery/audio pickers, TTS toggle, model picker, specialist chips

---

## GAP 1 (P0): Onboarding

**Problem:** App opens to HomeScreen with no providers configured. User must discover Settings tab, find provider key fields, fill them in. `firstRunComplete` flag exists in SettingsViewModel but nobody reads it.

**Fix (3 files):**
- `MainActivity.kt`: read `firstRunComplete` flag before setting content. If false, show OnboardingScreen.
- Create `OnboardingScreen.kt`: 3-page HorizontalPager — (1) Welcome "Aura needs an LLM to think" with provider icons, (2) Paste API key with mask toggle + provider selector chips, (3) "All set! Tap begin" with test-chat button that verifies the provider is configured.
- `SettingsScreen.kt`: remove "Restart the app after adding a key" text — ProviderKeys is now live, no restart needed. Update model list to actual configured models.

**Commit:** `feat(android): first-run onboarding flow`

---

## GAP 2 (P0): Permission Request Flow for Tools

**Problem:** When a tool returns `ToolResult.NeedsPermission`, ChatScreen renders it as error text. User sees "Error: Permission needed: android.permission.READ_MEDIA_IMAGES" but can't grant it.

**Fix (3 files):**
- `ChatViewModel.kt`: add `pendingPermissionRequest: ToolResult.NeedsPermission?` to ChatUiState. On NeedsPermission event, store it instead of treating as error.
- `ChatScreen.kt`: when `pendingPermissionRequest` is set, show a dialog with rationale text and "Grant" button. Launch `rememberLauncherForActivityResult(RequestPermission)`. On grant, retry the last tool call (or surface a message to the agent).
- Create `PermissionRequestDialog.kt` composable: permission name, rationale, Grant/Deny buttons.

**Commit:** `feat(android): permission grant flow from tool NeedsPermission`

---

## GAP 3 (P1): ConversationStore — persist conversations across restarts

**Problem:** Conversations live in ChatViewModel's StateFlow only. App restart = conversation lost. Comment in Conversation.kt says "Room-backed ConversationStore in module 3" but it doesn't exist.

**Fix (4 files):**
- Create `ConversationEntity.kt`: Room entity mirroring Conversation data class fields (id, title, createdAt, updatedAt, systemPrompt, model, metadata). Turns stored as JSON blob (serialized via kotlinx.serialization).
- Create `ConversationDao.kt`: insert, update, getById, recent(list), delete.
- Create `ConversationModule.kt`: Hilt module providing ConversationDatabase + ConversationDao. Database at version 1.
- `ChatViewModel.kt`: on init, load most recent conversation. On assistant turn complete, save conversation. Add "New chat" action that creates a fresh conversation.

**Commit:** `feat(android): persist conversations across restarts`

---

## GAP 4 (P1): Conversation History Screen

**Problem:** With ConversationStore in place, user needs a way to see past threads and resume them. Currently no history screen — Chat tab always shows current conversation.

**Fix (2 files):**
- Create `HistoryScreen.kt`: LazyColumn of past conversations sorted by updatedAt DESC. Each row: title, last message preview, timestamp. Tap → navigate to ChatScreen with that conversation loaded. Swipe to delete. Empty state: "No conversations yet."
- `NavGraph.kt`: add History as a destination reachable from ChatScreen (e.g., a clock icon in ChatScreen header, or a sub-route).

**Commit:** `feat(android): conversation history screen`

---

## GAP 5 (P1): Hands Dashboard

**Problem:** HandDatabase/HandDao/HandsModule exist. SpecialistRouter delegates to specialists. But there's NO screen to see active hands, their task goals, or their status. Automation is invisible.

**Fix (3 files):**
- Create `HandsScreen.kt`: Active hands section (card per hand: specialist type icon, task goal, status badge, progress if available). Completed hands section (collapsed, expandable). Empty state: "No hands running. Hands are created when Aura delegates a complex task to a specialist."
- Create `HandsViewModel.kt`: inject HandDao, list active + recent completed.
- `NavGraph.kt`: add Hands tab to bottom nav OR add it as a sub-route from HomeScreen.

**Design decision:** Bottom nav already has 5 tabs. Adding a 6th is crowded. Put Hands as a tile on HomeScreen ("Active hands: 2") that navigates to a full screen.

**Commit:** `feat(android): hands dashboard screen`

---

## GAP 6 (P2): Tasks / Reminders Management

**Problem:** TaskDao/TaskDatabase exist. HomeScreen shows pending task count. But user can't see all tasks, cancel reminders, or view completed ones. SetReminderTool schedules WorkManager jobs but there's no UI to cancel them.

**Fix (2 files):**
- Create `TasksScreen.kt`: Pending section (reminder cards: title, time, message, cancel button that calls WorkManager.cancelWorkById). Past section (completed/expired). Empty state: "No pending tasks. Ask Aura to set a reminder."
- Create `TasksViewModel.kt`: inject TaskDao + WorkManager, list pending + completed, cancel by work name.

**Commit:** `feat(android): tasks and reminders management screen`

---

## GAP 7 (P2): Proactive Event History

**Problem:** ProactiveEvents holds only the latest event in a StateFlow. HomeScreen shows it as a dismissible card. But once dismissed, it's gone. No history of what fired. CalendarMonitor fires every 5 min; MorningBrief runs at 7am — no log.

**Fix (3 files):**
- Create `ProactiveEventEntity.kt` + add to an existing Room DB (or new one). Fields: id, type, title, body, timestamp.
- Update `ProactiveEvents.kt`: on each new event, insert into DAO.
- Create `ProactiveHistoryScreen.kt`: timeline grouped by date. Each entry: type icon, title, body, time. Auto-scroll to latest.

**Commit:** `feat(android): proactive event history log`

---

## NOT IN THIS PLAN (already exists or deliberately skipped)

| Item | Status |
|---|---|
| Settings screen (provider keys) | Already exists and works |
| Memory browser (search/filter/forget) | Already exists and works |
| Knowledge graph viewer | Already exists and works |
| Chat with voice/camera/gallery | Already exists and works |
| Model picker | Already exists in ChatScreen |
| TTS toggle | Already exists in ChatScreen |
| TTS voice/pitch/speed settings | Deferred — needs TextToSpeech engine introspection API, non-trivial |
| Notification history (Android system) | Deliberately skipped — Android Settings app does this already |
| Dark/light theme | Deferred — not a structural gap |
| Multi-device sync | Anti-feature by design |

---

## Execution Order

```
Commit 1: GAP 1 — Onboarding (P0, makes app usable on first open)
Commit 2: GAP 2 — Permission request flow (P0, turns errors into grantable dialogs)
Commit 3: GAP 3 — ConversationStore (P1, persistence backbone)
Commit 4: GAP 4 — Conversation history screen (P1, depends on GAP 3)
Commit 5: GAP 5 — Hands dashboard (P1)
Commit 6: GAP 6 — Tasks management (P2)
Commit 7: GAP 7 — Proactive history (P2)
```

**Verification per commit:** `./gradlew :app:assembleDebug` passes.
**Final:** `./gradlew :aura-core:testDebugUnitTest :app:assembleDebug` all green.
