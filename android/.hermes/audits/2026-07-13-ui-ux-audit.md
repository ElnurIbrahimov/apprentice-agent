# Aura Android — Full UI/UX Audit Report

**Branch:** `feat/tier-1-friction`
**Date:** 2026-07-13
**Mode:** Read-only inventory, proportion/hierarchy/state analysis, redesign wave grouping, acceptance criteria, screenshot/test matrix

---

## 1. Complete Screen/Route/Component Inventory

### 1.1 Navigation Graph (NavGraph.kt)

| Route | Screen | Type | Bottom Bar Visible | Entry Points |
|-------|--------|------|--------------------|--------------|
| `home` | HomeScreen | Top-level tab | Yes | Start destination |
| `chat?convId={}&draft={}&brief={}` | ChatScreen | Top-level tab | Yes | Home tap, deep link, share |
| `memory` | MemoryScreen | Top-level tab | Yes | Bottom bar |
| `settings` | SettingsScreen | Top-level tab | Yes | Bottom bar |
| `diagnostics` | DiagnosticsScreen | Push | Yes | Settings > Diagnostics |
| `identity_editor` | IdentityEditorScreen | Push | Yes | (unused in NavGraph currently) |
| `knowledge_graph` | KnowledgeGraphScreen | Push | Yes | Memory > Knowledge graph |
| `history` | HistoryScreen | Push | Yes | Chat header history icon |
| `hands` | HandsScreen | Push | Yes | Home card tap |
| `tasks` | TasksScreen | Push | Yes | Home card tap |
| `tools` | ToolsScreen | Push | Yes | Home card tap |
| `proactive` | ProactiveHistoryScreen | Push | Yes | Home proactive link |
| `reminders` | RemindersScreen | Push | Yes | Home card tap |
| `profile` | ProfileScreen | Push | Yes | Settings > persona > Edit |

**Note:** Bottom bar is now always visible (changed from hiding on secondary screens). This matches the web pattern.

### 1.2 Screens Detail

#### 1.2.1 OnboardingScreen (shown on first launch)
- **Pages:** PageWelcome, PageKeys, PageDone
- **Components:** FeaturePill, ProviderKeyField, AnimatedContent transitions
- **Dialog states:** None

#### 1.2.2 HomeScreen
- **Composables:** ExamplePromptsGrid, ExamplePromptChip, QuickActionCard (×6), ProactiveEventCard, ProactiveUnreadLink, BriefCard (×4)
- **Dynamic sections:** Hero + greeting + subStatus + date, proactive card, example prompts (empty state only), 2×3 quick action grid, detail cards (tasks, reminders, memories, calendar)

#### 1.2.3 ChatScreen (largest composable, ~1464 lines)
- **Composables:** ChatHeader, ChatMessageList, ChatInputBar, EmptyChatState, ErrorBanner, SaveWarningBanner, IncognitoBanner, SpecialistChips, VisionPromptChips, QuickChipRow, MoaThinkingIndicator, ModelPickerSheet, SourcesSheet, VoiceOverlay, ContinuousVoiceOverlay
- **Dialogs:** Stop streaming confirm, delete confirm, model picker modal
- **Launchers:** Camera, gallery, audio, mic permission

#### 1.2.4 MemoryScreen
- **Composables:** AuraScreenHeader, Search bar, category FlowRow chips, Add note button, Knowledge graph button, Rebuild embeddings, Clear category/all buttons, MemorySkeletonLoading, MemoryRow, EditMemoryDialog, AddNoteDialog

#### 1.2.5 SettingsScreen (largest scrolling screen, ~961 lines)
- **Sections:** AI & Models (expanded by default), Usage, Appearance, Persona, Privacy, Data & Backup
- **Composables:** SettingsSection (collapsible card), ProviderKeyField, ModelPickerSheet, AssistChips as toggle/selection

#### 1.2.6 HandsScreen
- **Tabs:** Automations (tab 0), Run history (tab1)
- **Composables:** AuraScreenHeader, TabRow, HandCard, MetadataPill, RunHistory, RunHistoryCard, HandsSkeletonLoading, HandsEmptyState, RunHandDialog
- **Dialogs:** HandEditorDialog, Delete hand, Clear history, Hand finished result

#### 1.2.7 HistoryScreen
- **Composables:** AuraScreenHeader (with "Export all" action), Search bar, LinearProgressIndicator (searching), HistorySkeletonLoading, HistoryRow
- **Dialogs:** Rename conversation

#### 1.2.8 ToolsScreen
- **Composables:** Header, Search bar, CategoryHeader (grouped), ToolRow

#### 1.2.9 TasksScreen
- **Composables:** AuraScreenHeader, Status filter chips (All/Pending/Done), Reminders section, TasksSkeletonLoading, TaskRow, ReminderRow, RemindersHeader
- **Dialogs:** AddTaskDialog, EditTaskDialog, Clear completed confirm, AddReminderDialog, EditReminderDialog

#### 1.2.10 RemindersScreen
- **Composables:** AuraScreenHeader, Upcoming/History chip toggle, ReminderLifecycleRow
- **Dialogs:** ReminderEditorDialog, Cancel reminder confirm, Clear history confirm

#### 1.2.11 ProactiveHistoryScreen
- **Composables:** TopAppBar, HistoryCard (×4 kinds), DebugSection, DebugActionRow
- **States:** Empty state

#### 1.2.12 KnowledgeGraphScreen
- **Composables:** Header with back + refresh, GraphStatCard (×2), Search bar, TypeChip filter scroll row, GraphNodeCard, GraphEmptyState, ModalBottomSheet (detail view), EditGraphNodeDialog, MergeGraphNodeDialog

#### 1.2.13 DiagnosticsScreen
- **Composables:** Header with back + refresh, Info banner, Share/Clear buttons, DiagnosticCard (expandable), DiagnosticsEmptyState

#### 1.2.14 ProfileScreen
- **Composables:** TopAppBar, Name field + Save, Traits FlowRow, Facts list, Add trait/fact fields

#### 1.2.15 IdentityEditorScreen
- **Composables:** TopAppBar with Save/Reset, full-screen monospace OutlinedTextField

#### 1.2.16 HandEditorDialog (used by HandsScreen)
- **Composables:** AlertDialog-based editor with sections: Name, Trigger, Steps, Variables, Conditions, Schedule

### 1.3 Shared Components

| Component | Used In | Notes |
|-----------|---------|-------|
| AuraScreenHeader | Memory, Hands, History, Tasks, Reminders | Title + subtitle + optional action |
| AuraScreenShell | (defined, underused) | Only 1 screen uses it |
| EmptyChatState | ChatScreen | Logo + "Welcome to Aura" + tagline |
| QuickChipRow | ChatScreen empty state | 5 horizontal chips below input |
| MessageBubble | ChatScreen | UserBubble (pointed) + AssistantMessage (avatar + content, no bubble) |
| MarkdownText | ChatScreen assistant messages | Renders markdown |
| ModelPickerSheet | ChatScreen + Settings | Modal bottom sheet search + list |
| MoaThinkingIndicator | ChatScreen deep mode | |
| SpecialistChips | ChatScreen | Horizontal scrollable filter chips |
| VisionPromptChips | ChatScreen | 3 prompts for image analysis |
| FollowUpSuggestions | ChatScreen | Canned heuristic suggestions |
| MemoryRecallChip | ChatScreen | |
| ToolCallBadge | ChatScreen | |
| ToolArgForm | ChatScreen | |
| StreamingText / StreamingMarkdownState | ChatScreen | |
| VoiceOverlay / ContinuousVoiceOverlay | ChatScreen | |
| AuraTokens | Theme | Custom design tokens |
| AuraAiAvatar | ChatScreen, header | Breathing avatar |

### 1.4 Voice Overlays

| Component | Trigger | Description |
|-----------|---------|-------------|
| VoiceOverlay (tap-to-speak) | Tap mic | Auto-send on first final result |
| VoiceOverlay (hold-to-talk) | Long-press mic | User controls send via stop button |
| ContinuousVoiceOverlay | (wired but no visible trigger) | Hands-free LISTENING→THINKING→SPEAKING loop |

---

## 2. Proportion & Hierarchy Audit

### 2.1 Critical Proportion Issues (Highest Urgency)

| # | Screen | Issue | Severity |
|---|--------|-------|----------|
| P1 | **ChatScreen** | `EmptyChatState` uses `Spacer(weight(0.6f))` above logomark to push content to ~25% from top — this is fragile on different screen sizes. Below a certain viewport the welcome text + input bar collide. | High |
| P2 | **ChatScreen** | Message list padding: `padding(horizontal = 16.dp, vertical = 12.dp)` on `AssistantMessage` but `padding(horizontal = 16.dp, vertical = 8.dp)` on `UserBubble`. The 4dp vertical disparity makes the chat feel uneven when user/assistant messages alternate. | Medium |
| P3 | **HomeScreen** | Greeting uses `displayMedium` (45sp / 52sp) but the quick action grid icons are `36.dp` with `12sp` labels. The visual weight ratio between the hero text and the action cards is ~4:1 — the hero dominates the entire screen before scrolling. | Medium |
| P4 | **SettingsScreen** | Collapsible sections use `surfaceVariant.copy(alpha = 0.40f)` background. API key fields directly inside the collapsible region have no additional separation — no section-3 inset or card elevation. On large screens the key list visually merges. | Low-Medium |
| P5 | **MemoryScreen** | The entire screen is a `Column` with no scroll for the header/search/chips/buttons section, then a `LazyColumn` for memory items. If there are 100+ memories, the top controls (header + search + chips + 4 buttons + rebuild + clear) take up ~600dp of fixed space before the first memory item appears. | High |
| P6 | **ChatScreen** | Input bar: `imePadding()` is applied to the outer `Box`, but buttons inside `ChatInputBar` have no intrinsic lower bound. On devices with gesture nav, the bottom bar's 12dp bottom padding + the input bar area can overlap. | Medium |
| P7 | **HomeScreen** | Two Rows of QuickActionCard (3 each) — `weight(1f)` means equal width. But "Hands" and "Tools" are conceptually different from "Proactive". All 6 get the same visual weight when they shouldn't. | Low |

### 2.2 Visual Hierarchy Inconsistencies

| # | Issue | Affected Screens |
|---|-------|-----------------|
| H1 | **Inconsistent header patterns:** Home uses `displayMedium` (45sp), Chat has custom `ChatHeader` composable, Memory/Hands/History/Tasks/Reminders use `AuraScreenHeader` with `headlineMedium` (34sp), Settings uses `displaySmall` (36sp), Diagnostics uses `headlineMedium`. Three different sizes for "page title" across the app. | Home, Chat, Memory, Hands, History, Tasks, Reminders, Settings, Diagnostics |
| H2 | **Inconsistent action placement:** Home has tasks/reminders/hands links inside cards; Memory has actions as full-width buttons inside scroll; Settings uses collapsible sections; Hands uses FAB + TabRow. No unified action pattern. | All |
| H3 | **Card radius inconsistency:** `RoundedCornerShape(10.dp)` (memories), `RoundedCornerShape(14.dp)` (settings sections), `RoundedCornerShape(16.dp)` (home cards), `RoundedCornerShape(18.dp)` (hand cards), `RoundedCornerShape(20.dp)` (chat CTA). Five different radii for semantically identical "card" surfaces. | Universal |
| H4 | **Empty state inconsistency:** Home shows ExamplePromptsGrid; Chat shows `EmptyChatState` (logomark + welcome); Memory shows card with "No memories yet" + helper text; History shows generic icon + text; Hands shows branded empty state with icon; Tools shows plain text. Different visual languages for "no data." | Home, Chat, Memory, History, Hands, Tools |
| H5 | **Loading state inconsistency:** Memory has `MemorySkeletonLoading` (generic pulse blobs); History has `HistorySkeletonLoading` (shaped placeholders); Hands has `HandsSkeletonLoading` (simple card blobs); Settings has no loading state; KnowledgeGraph has `CircularProgressIndicator`. | Memory, History, Hands, Settings, KG |
| H6 | **AuraScreenShell vs inline composition:** Only ProactiveHistoryScreen? actually uses `AuraScreenShell`. All other screens inline the same header+padding pattern manually. Dead code — the shell was intended for consistency but only used by 1 screen. | Universal |

---

## 3. State Coverage Audit

### 3.1 Empty States

| Screen | Empty State? | Quality |
|--------|-------------|---------|
| Home | ✅ ExamplePromptsGrid (6 prompt chips) | Good — proactive onboarding feel |
| Chat | ✅ EmptyChatState (logomark + welcome + chips below input) | Good |
| Memory | ✅ Two variants (query blank vs filtered) | Good |
| History | ✅ Two variants (no convos vs no search match) | Good — distinct messages |
| Hands | ✅ HandsEmptyState (branded), RunHistory empty | Good |
| Tasks | ✅ Text per filter (all/pending/done) + reminders empty | Adequate — just text |
| Reminders | ✅ Two variants upcoming/history | Adequate |
| Proactive | ✅ EmptyState composable | Good — branded + description |
| Tools | ✅ Text only ("No tools registered" / "No tools match...") | Poor — just text, no icon |
| KnowledgeGraph | ✅ GraphEmptyState (filtered vs quiet) | Good |
| Diagnostics | ✅ DiagnosticsEmptyState ("That is the good kind of empty.") | Good — character |
| Settings | ✅ Text for no providers/no usage | Adequate |
| Profile | ✅ Just no name/traits/facts shown | Adequate |

### 3.2 Loading States

| Screen | Loading State? | Quality |
|--------|---------------|---------|
| Home | ❌ None — data loads fire in ViewModel init, screen renders with empty state until data arrives | **MISSING** |
| Chat | ❌ None — initial load is instant (no remote fetch) | OK (acceptable) |
| Memory | ✅ MemorySkeletonLoading (generic pulse) | Adequate |
| History | ✅ HistorySkeletonLoading (shaped to card layout) | Good |
| Hands | ✅ HandsSkeletonLoading (3 card blobs) | Adequate |
| Tasks | ✅ TasksSkeletonLoading (shaped to card layout) | Good |
| Reminders | ❌ None — `if (!state.loading && rows.isEmpty())` handles post-load empty but no loading animation | **MISSING** |
| Proactive | ❌ None — events load synchronously | OK (acceptable) |
| Tools | ❌ None — tools are cached from boot | OK (acceptable) |
| KnowledgeGraph | ✅ CircularProgressIndicator centered in box | Adequate |
| Diagnostics | ✅ CircularProgressIndicator centered | Adequate |
| Settings | ❌ None — model list loading shown inline in "Choose model" button area | Partial |
| Profile | ❌ None — data loads synchronously | OK (acceptable) |

### 3.3 Error States

| Screen | Error Handling? | Quality |
|--------|---------------|---------|
| Chat | ✅ ErrorBanner (retryable + typed error), SaveWarningBanner, model fetch error | **Good** — best in app |
| Memory | ❌ Rebuild result shown but no general error state | **MISSING** |
| History | ❌ No error state for failed delete/share/rename | **MISSING** |
| Hands | ✅ Snackbar-hosted error display | Good |
| Tasks | ❌ No error state for create/update/delete failures | **MISSING** |
| Reminders | ❌ No error state | **MISSING** |
| KnowledgeGraph | ✅ Error banner with dismiss | Good |
| Diagnostics | ✅ Error banner in-body | Good |
| Settings | ✅ Model refresh error shown inline; verify result displayed | Good |
| Profile | ❌ No error state for save failures | **MISSING** |
| Home | ❌ No error state for data fetch failures | **MISSING** |
| Chat model picker | ✅ Error message when models fail to load | Good |

### 3.4 Edge Cases

| Edge Case | Where | Status |
|-----------|-------|--------|
| Very long conversation (1000+ turns) | ChatScreen | `LazyColumn` handles — but header + toolbar + input are fixed-height, could improve |
| 200+ API keys | SettingsScreen | Vertical scroll handles — but provider chips row doesn't wrap on small widths |
| 500+ memories | MemoryScreen | Top controls consume ~600dp before list items — needs refactor |
| 50+ different tools | ToolsScreen | Search + category grouping handles well |
| Keyboard open while streaming | ChatScreen | `imePadding()` on outer Box handles — but bottom chip row may overlap input |
| Deep mode + streaming + error + keyboard | ChatScreen | Multiple composables stack vertically — fragile |
| 0 providers configured | Settings | Shows "No providers configured yet" — correct |
| Rapid fire send/cancel/stream | ChatScreen | Uses `hapticView` check with `LaunchedEffect(state.streaming)` — robust |
| Privacy: biometric + incognito | Settings + Chat | Both implemented but incognito banner + app lock are separate concerns unclear to user |

---

## 4. Interaction & Usability Audit

### 4.1 Top Interaction Issues

| # | Screen | Issue | Severity |
|---|--------|-------|----------|
| I1 | **Home** | QuickActionCard showing "0" count is not actionable — tapping Memory with 0 memories opens an empty screen. Should be disabled or show different state. | Medium |
| I2 | **ChatScreen** | "Jump to latest" pill appears when `firstVisibleItemIndex > 5`. But if the user has scrolled up reading and a new token arrives, the auto-scroll check (`lastVisible >= target - 1`) fires — yanking them back. | High |
| I3 | **ChatScreen** | Three voice entry points (tap mic, long-press mic, continuous voice) is overwhelming. User must discover the difference through trial. | Medium |
| I4 | **HandsScreen** | Tab 0 shows hands + FAB; Tab 1 shows run history with no FAB. The tab inconsistency is documented but jarring — user reaches for FAB and it's gone. | Low |
| I5 | **MemoryScreen** | Rebuild embeddings + Clear category + Clear all are always visible once memories exist. No confirmation has "and don't ask again" — destructive actions. | Medium |
| I6 | **TasksScreen** | The "Clear N done" action appears as a header button only when filter is not "pending". But the subtitle says "X tasks" including reminders count (which is always shown first) — confusing count semantics. | Low |
| I7 | **ChatScreen** | Stop-streaming confirm dialog appears when back is pressed during streaming. The "Keep listening" button name is confusing — it should say "Cancel" or "Don't stop". | Low |

### 4.2 Navigation & Information Architecture

| Finding | Detail |
|---------|--------|
| **Bottom bar is always visible** | Good — matches web sidebar persistence. But it shows 4 tabs while the app has 15+ routes. Secondary screens (hands, tasks, reminders) are only reachable from Home cards or deep links — no persistent nav for them. |
| **Home is a dashboard** | Home aggregates: greeting, proactive events, 6 quick action cards, and 4 detail sections. It's a single scrolling column mixing glanceable data and CTAs — risks becoming overwhelming. |
| **Chat is the primary interface** | ChatScreen handles: text chat, specialist selection, model switching, image capture/upload, audio upload, voice modes (3), streaming, citations, incognito mode, deep mode. This is ~1464 lines of composable — the most complex screen. |
| **Settings is a long scroll** | 6 collapsible sections on one page. Persona section inside Settings duplicates functionality from ProfileScreen. Identity editor is a separate screen. Three places to edit persona data. |
| **Duplicate reminder editing** | Reminders can be edited from both TasksScreen and RemindersScreen — each with its own ViewModel. If a user edits a reminder in Tasks then opens Reminders, the state is stale. |

---

## 5. Existing Test Coverage Audit

### 5.1 UI/Composable Tests (`app/src/test/kotlin`)

| Test File | What It Tests | Type | Coverage Quality |
|-----------|--------------|------|-----------------|
| `DefaultQuickActionsTest.kt` | Chip count, non-blank, uniqueness | Unit | Light contract lock |
| `FollowUpSuggestionsTest.kt` | Heuristic logic output | Unit | Good logic coverage |
| `MarkdownTest.kt` | Markdown parsing | Unit | Good |
| `StreamingMarkdownStateTest.kt` | Streaming state logic | Unit | Good |
| `StreamingTokensPerSecondTest.kt` | Tokens/s calculation | Unit | Good |
| `VisionPromptChipsTest.kt` | Prompt list contract | Unit | Light contract lock |

### 5.2 ViewModel Tests (`app/src/test/kotlin`)

| Test File | What It Tests | Coverage |
|-----------|--------------|----------|
| `ChatViewModelTest.kt` | Core send/stream/cancel | Good |
| `ChatViewModelScreenTest.kt` | Screen-level ViewModel state | Good |
| `ChatViewModelLastAssistantTest.kt` | Edge case: empty assistant | Adequate |
| `HandsViewModelTest.kt` | Hand CRUD + run lifecycle | Good |
| `HistoryViewModelTest.kt` | Search, delete, export | Good |
| `KnowledgeGraphViewModelTest.kt` | Node CRUD, search, merge | Good |
| `MemoryViewModelTest.kt` | Memory CRUD, search, rebuild | Good |
| `TasksViewModelTest.kt` | Task CRUD, filters | Good |
| `RemindersViewModelTest.kt` | Reminder CRUD, schedule | Good |
| `ProfileViewModelTest.kt` | Name/traits/facts | Good |
| `DiagnosticsViewModelTest.kt` | Entry CRUD, export | Good |
| `SettingsViewModelAppLockTest.kt` | App lock toggle | Good |
| `BackupViewModelTest.kt` | Export/import | Good |
| `TtsTest.kt` | TTS state | Adequate |
| `HoldToTalkTranscriptMirrorTest.kt` | Voice transcript mirror | Adequate |
| `ModelLabelsTest.kt` | Model display name formatting | Good |

### 5.3 Instrumentation Tests (`app/src/androidTest`)

| Test File | What It Tests | Coverage |
|-----------|--------------|----------|
| `SmokeTest.kt` | App context exists + MainActivity launches | Minimal — 2 tests |

### 5.4 Critical Test Gaps

| Gap | Risk |
|-----|------|
| **No Compose UI test anywhere** — no `createComposeRule()` tests for any screen | Any visual regression, layout breakage, or accessibility issue goes undetected |
| **No screenshot tests** — no Paparazzi, Roborazzi, or `takeScreenshot()` tests | Impossible to catch proportion/spacing regressions in CI |
| **No navigation tests** — no test that verifies route transitions or backstack behavior | Deep links and bottom bar state untested |
| **No empty/loading/error state rendering tests** | ViewModels handle data but no test verifies the screen renders the right state |
| **No theme/snapshot tests for light/dark mode** | Color token usage across 15+ screens unchecked |
| **No accessibility tests** (contentDescription, touch target sizes, contrast ratios) | Mandatory for any eventual distribution |
| **SmokeTest is the only androidTest** — only 2 basic assertions | No real instrumentation coverage |

---

## 6. Redesign Wave Grouping

### Wave 1: Foundation Fixes (High Impact, Low Effort)
*Target: Proportion consistency, missing states, critical bugs*

| # | Screen | Item | Effort | Impact |
|---|--------|------|--------|--------|
| 1.1 | Universal | **Unify card radius** — pick ONE radius for cards (recommend: 14.dp as midpoint) and apply everywhere. Currently ranges 10-20dp. | 1h | High |
| 1.2 | Universal | **Unify header typography** — all screens should use either `AuraScreenHeader` (headlineMedium 34sp) or a consistent display. Home/Settings/Diagnostics use different sizes. | 2h | High |
| 1.3 | Home | **Add loading state** — show skeleton grid while ViewModel initializes data. | 2h | High |
| 1.4 | Reminders | **Add loading state** — implement skeleton or progress indicator. | 1h | Medium |
| 1.5 | ProactiveHistory | **Add loading state** — basic spinner if events async. | 1h | Low |
| 1.6 | Home | **Disable zero-count quick actions** — grey out Memory/Tasks/Calendar when count=0, or skip them. | 1h | Medium |
| 1.7 | Chat | **Fix auto-scroll yank** — don't auto-scroll if `firstVisibleItemIndex > 2` (change threshold or remove auto-scroll entirely when user has scrolled up). | 1h | High |
| 1.8 | Chat | **Rename "Keep listening" → "Cancel"** in stop-stream dialog. | 30min | Low |
| 1.9 | Universal | **Adopt AuraScreenShell** — replace inline Column+padding patterns in Memory, Hands, History, Tasks, Reminders with the shared shell composable. Currently only 1 screen uses it. | 3h | Medium |
| 1.10 | Memory | **Fix fixed top controls** — make the full screen scrollable or move action buttons inside the LazyColumn so they don't consume 600dp before data. | 3h | High |

### Wave 2: State Parity (Medium Impact)
*Target: Error handling, empty state consistency, interaction edge cases*

| # | Screen | Item | Effort | Impact |
|---|--------|------|--------|--------|
| 2.1 | Memory | **Add error state** — banner for failed save/delete/rebuild. | 2h | Medium |
| 2.2 | History | **Add error snackbar** — for failed delete/rename/share. | 2h | Medium |
| 2.3 | Tasks | **Add error snackbar** — for failed CRUD operations. | 2h | Medium |
| 2.4 | Reminders | **Add error snackbar** — for failed create/update/cancel. | 2h | Medium |
| 2.5 | Profile | **Add error handling** — for save failures. | 1h | Medium |
| 2.6 | Home | **Add error state** — for failed data fetch. | 2h | Medium |
| 2.7 | Universal | **Design System: Empty States** — create shared empty state composable with props (icon, title, description, action). Apply to all 12 screens. | 4h | High |
| 2.8 | Universal | **Design System: Loading States** — create shared skeleton component with configurable layout. Apply to all screens that need loading. | 3h | High |
| 2.9 | Chat | **Add loading state for conversation load** — show skeleton/placeholder while a saved conversation is being loaded from DB. | 2h | Medium |
| 2.10 | Tools | **Upgrade empty state** — from plain text to branded empty state with icon (matches Hands/Diagnostics pattern). | 1h | Low |

### Wave 3: Layout & Proportion Overhaul (High Impact)
*Target: Spacing, visual weight, responsive layout*

| # | Screen | Item | Effort | Impact |
|---|--------|------|--------|--------|
| 3.1 | Universal | **Define and enforce spacing scale** — currently uses ad-hoc 8/10/12/14/16/20/24/28dp. Standardize to 4dp grid (4/8/12/16/20/24/32/48). | 3h | High |
| 3.2 | Home | **Redesign hero section** — reduce greeting size from `displayMedium` to `headlineLarge`, use more compact metadata row. Quick action cards should use consistent icon sizes. | 4h | High |
| 3.3 | Chat | **User/Assistant message vertical parity** — match padding to `10dp` vertical on both user and assistant rows. | 30min | Medium |
| 3.4 | Chat | **Replace EmptyChatState `weight(0.6f)`** with a constraint-based layout that guarantees the welcome text + input bar never overlap regardless of screen size. | 2h | High |
| 3.5 | Settings | **Add card inset/grouping** to API key fields inside collapsible sections — wrap in a `Surface` with consistent padding. | 2h | Medium |
| 3.6 | Universal | **Bottom nav + gesture nav inset audit** — verify `navigationBarsPadding()` is consistently applied and input bar doesn't overlap system gestures. | 2h | High |

### Wave 4: Testing Infrastructure (Foundation)
*Target: Test gaps that prevent confident redesign*

| # | Item | Effort | Impact |
|---|------|--------|--------|
| 4.1 | **Add Compose UI test dependency** (compose-ui-test + manifest) | 1h | High |
| 4.2 | **Add screenshot test dependency** (Roborazzi or Paparazzi) | 2h | High |
| 4.3 | **Screenshot test for HomeScreen** (all states: empty, with data, proactive event) | 3h | High |
| 4.4 | **Screenshot test for ChatScreen** (empty, with messages, streaming, error) | 4h | High |
| 4.5 | **Screenshot test for MemoryScreen** (empty, filtered, with items) | 2h | Medium |
| 4.6 | **Navigation test** — validate bottom bar route transitions + secondary screen navigation | 3h | High |
| 4.7 | **Accessibility test** — contentDescription scan for all screens | 2h | Medium |
| 4.8 | **Light/dark theme screenshot tests** for top 5 screens | 3h | Medium |
| 4.9 | **Error state rendering tests** for each screen's error handling | 3h | Medium |
| 4.10 | **Loading state rendering tests** for each screen's loading state | 2h | Medium |

---

## 7. Per-Screen Acceptance Criteria

### 7.1 HomeScreen
- [ ] Greeting text does not exceed 50% of viewport height
- [ ] Each QuickActionCard with count=0 is visually disabled or hidden
- [ ] Loading state renders skeleton placeholders (not raw empty state flash)
- [ ] Error state renders inline error message
- [ ] Proactive event card auto-dismisses after 30s or on tap
- [ ] Proactive unread count links to proactive history
- [ ] BriefCard remembers truncate at 60 chars with ellipsis
- [ ] Example prompts grid ONLY shows when user has no data (empty state)

### 7.2 ChatScreen
- [ ] Empty state: logomark + "Welcome to Aura" + tagline centered between header and input
- [ ] Empty state chips (5) are rendered below input bar
- [ ] User messages: right-aligned, pointed 24/24/24/4 radius, white bg
- [ ] Assistant messages: no bubble, avatar + "AURA" label + content
- [ ] Auto-scroll does NOT yank user if they've scrolled up > 2 items
- [ ] Streaming: blinking cursor, model label + timestamp hidden during stream
- [ ] "Jump to latest" pill appears when scrolled up > 5 items
- [ ] Stop-stream confirm dialog on back press during stream
- [ ] ErrorBanner shown for API/network errors (retryable vs typed)
- [ ] Error banner shown when saving fails (SaveWarningBanner)
- [ ] All 3 voice modes reachable and correctly labeled
- [ ] VisionPromptChips shown after image capture, dismissed after selection
- [ ] Loading state for conversation resume (model loading from DB)

### 7.3 MemoryScreen
- [ ] Search bar filters memory list in real-time
- [ ] Category filter chips correctly highlight selected
- [ ] Empty state differentiates "no memories" from "no search/filter matches"
- [ ] Loading state shows skeleton pulse cards
- [ ] Error handling for failed save/delete/rebuild
- [ ] MemoryRow shows: category dot, content, category·source·age, fading flag, importance, recall count, tags
- [ ] Edit dialog pre-fills current memory values
- [ ] Add Note dialog supports content + category + importance
- [ ] Confirm dialogs for: rebuild, clear all, clear category
- [ ] Top controls do not consume more than 40% of viewport

### 7.4 SettingsScreen
- [ ] AI & Models section expanded by default, others collapsed
- [ ] Provider keys show masked value, reveal on focus
- [ ] Verify button shows success/error result inline
- [ ] Default model picker opens ModelPickerSheet
- [ ] Embedding model selection updates immediately
- [ ] Usage section shows per-model token counts
- [ ] Theme mode toggle (System/Light/Dark) persists across restart
- [ ] Identity textarea saves/loads correctly
- [ ] Specialist override editor shows specialists with current override status
- [ ] App lock toggle persists
- [ ] Morning brief hour picker works
- [ ] Export/restore produces valid JSON

### 7.5 HandsScreen
- [ ] Tab 0 (Automations): lists hands with name, trigger, metadata pills
- [ ] Tab 1 (Run history): shows runs with status filter (All/Success/Failed)
- [ ] Empty state for no hands (branded)
- [ ] Empty state for no runs (branded)
- [ ] Loading state shows skeleton pulse cards
- [ ] Error displayed via snackbar
- [ ] New/Edit hand dialog has all sections (name/trigger/steps/variables/conditions/schedule)
- [ ] Run hand dialog shows variable overrides
- [ ] Delete hand shows confirmation
- [ ] Clear history shows confirmation

### 7.6 HistoryScreen
- [ ] Search bar filters by title and message text
- [ ] Loading state shows skeleton cards matching HistoryRow layout
- [ ] Empty state differentiates "no conversations" vs "no match"
- [ ] LinearProgressIndicator during search
- [ ] HistoryRow shows: pin icon, title (semi-bold), preview (80 chars), model, date, actions (pin/share/delete)
- [ ] Long-press opens rename dialog
- [ ] "Export all" button shown when conversations exist and no query active
- [ ] Share creates markdown file in cache, opens system share sheet

### 7.7 ToolsScreen
- [ ] Search bar filters by tool name/description
- [ ] Tools grouped by category with emoji icon + count
- [ ] ToolRow shows: name, arity count, description, up to 4 argument chips
- [ ] Empty state: branded with icon (not just text)
- [ ] No loading state needed (tools boot-cached)

### 7.8 TasksScreen
- [ ] Status filter chips (All/Pending/Done) work correctly
- [ ] Loading state shows skeleton pulse cards
- [ ] Reminders section shown when filter != "Done"
- [ ] "Clear N done" appears when done tasks exist and filter != "Pending"
- [ ] TaskRow shows: title (line-through when done), description, due date, priority chip, tags
- [ ] Error handling for failed CRUD via snackbar
- [ ] Add/Edit task dialog supports: title, description, due date, priority, tags

### 7.9 RemindersScreen
- [ ] Upcoming/History chip toggle works
- [ ] Loading state shown while data loads
- [ ] Empty state differentiates "Nothing scheduled" from "No reminder history"
- [ ] "Clear history" button only in History tab
- [ ] FAB only shown in Upcoming tab
- [ ] ReminderLifecycleRow shows: message, timestamp + recurrence, edit/cancel (upcoming) or delete (history)
- [ ] Cancel reminder confirmation dialog

### 7.10 KnowledgeGraphScreen
- [ ] Stat cards show node + edge counts
- [ ] Search bar filters by label/type/properties
- [ ] Type filter chips scroll horizontally
- [ ] Error banner shown on fetch/update errors
- [ ] Loading state shows centered spinner
- [ ] Empty state differentiates "no data" vs "filtered out"
- [ ] GraphNodeCard shows: type dot, label, type+properties subtitle, confidence %
- [ ] Modal bottom sheet shows: node details, properties JSON, incoming/outgoing relations, Edit/Merge/Delete actions
- [ ] Edit dialog validates JSON property input
- [ ] Merge dialog shows candidate list (excludes self)

### 7.11 ProactiveHistoryScreen
- [ ] Events shown in reverse chronological order (newest first)
- [ ] HistoryCard correctly renders all 4 event kinds with color coding
- [ ] Empty state shown when no events
- [ ] Tap hint shown for actionable events
- [ ] Badge shown for events with counts (Location arrival memories)
- [ ] Debug section at bottom with fire-now buttons

### 7.12 DiagnosticsScreen
- [ ] Entry count shown in subtitle
- [ ] Loading state shows centered spinner
- [ ] Empty state: branded with "That is the good kind of empty."
- [ ] Error banner shown on export/load errors
- [ ] DiagnosticCard shows: FATAL/ERROR badge, code, message, timestamp + thread, expandable stack trace
- [ ] Share creates NDJSON file via FileProvider
- [ ] Clear shows confirmation dialog

### 7.13 ProfileScreen
- [ ] Name field shows current value, Save button enabled only when changed
- [ ] Traits shown as InputChips with delete action
- [ ] Add trait field + button work
- [ ] Facts shown as list with delete action
- [ ] Add fact field (multi-line) + button work
- [ ] Clear profile confirmation dialog

### 7.14 IdentityEditorScreen
- [ ] Monospace text editor with word count
- [ ] Save writes to identity.md
- [ ] Reset restores bundled SOUL.md
- [ ] Shows "Customized" vs "Default persona" status

---

## 8. Screenshot & Test Matrix

### 8.1 Screenshot Test Matrix

Cells marked **✓** = screenshot test needed for that state.

| Screen | Empty | With Data | Loading | Error | Streaming | Deep Mode | Incognito |
|--------|-------|-----------|---------|-------|-----------|-----------|-----------|
| Home | ✓ | ✓ | ✓ | | | | |
| Chat (empty) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Chat (history) | | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Chat (image) | | ✓ | | | | | |
| Memory | ✓ | ✓ | ✓ | | | | |
| Settings | | ✓ (AI expanded) | | | | | |
| Settings (all open) | | ✓ | | | | | |
| Hands | ✓ | ✓ | ✓ | | | | |
| Hands Run History | ✓ | ✓ | | | | | |
| History | ✓ | ✓ | ✓ | | | | |
| Tasks | ✓ | ✓ | ✓ | | | | |
| Reminders | ✓ | ✓ | ✓ | | | | |
| KnowledgeGraph | ✓ | ✓ | ✓ | ✓ | | | |
| ProactiveHistory | ✓ | ✓ | | | | | |
| Diagnostics | ✓ | ✓ | ✓ | ✓ | | | |
| Profile | | ✓ | | | | | |
| Onboarding (3 pages) | ✓ | | | | | | |
| IdentityEditor | | ✓ | | | | | |

**Total: ~50 screenshot tests**

### 8.2 Compose UI Test Priority Matrix

| Priority | Test | Count | Reason |
|----------|------|-------|--------|
| P0 | Navigation: bottom bar switches tabs correctly | 4 | Core smoke test for all routes |
| P0 | Navigation: secondary screen backstack | 10 | Deep link + chat restore |
| P0 | ChatScreen: send message → bubble appears | 1 | Core user flow |
| P0 | ChatScreen: empty state renders correctly | 1 | First impression |
| P1 | All screens: empty state renders | 12+ | State parity |
| P1 | All screens with loading: skeleton renders | 7 | State parity |
| P1 | ChatScreen: streaming token appends to last assistant | 1 | Core UX |
| P1 | HomeScreen: quick action cards with/without count | 2 | Data-driven UI |
| P2 | MemoryScreen: category filter + search interaction | 2 | Filter UX |
| P2 | HandsScreen: tab switching | 2 | Tab UI |
| P2 | TasksScreen: filter chips | 3 | Filter UI |
| P2 | ChatScreen: model picker opens/closes | 2 | Modal UX |
| P3 | Settings: collapsible sections toggle | 6 | Section UX |
| P3 | Profile: add/remove traits and facts | 4 | CRUD interactions |

**Total: ~51 Compose UI tests**

---

## 9. Summary of Findings

### 9.1 What's Working Well
- **Empty state coverage is strong** — 11/14 screens have branded, contextual empty states
- **ViewModel test coverage is excellent** — all major screens have ViewModel unit tests
- **Error state coverage on ChatScreen** is best-in-class (error banner, retry logic, typed errors)
- **Custom design tokens** (AuraTokens) provide a solid foundation for visual consistency
- **Skeleton loading** on History, Tasks, Memory, and Hands is well-implemented with shaped placeholders matching real content layout
- **Navigation** keeps bottom bar always visible — good UX decision matching web pattern

### 9.2 What Needs Immediate Attention (Wave 1)
- **Loading states on Home and Reminders** are missing — user sees blank/empty before data loads
- **Card radius inconsistency** across the app (5 different values) undermines visual cohesion
- **Header typography is inconsistent** — 3 different sizes for page titles
- **AuraScreenShell is underused** — it was designed for consistency but only 1 screen adopts it
- **MemoryScreen fixed top controls** consume too much space with many memories
- **Chat auto-scroll yank** breaks reading context

### 9.3 What Needs Medium-Term Work (Waves 2-3)
- **Error states missing** on Memory, History, Tasks, Reminders, Profile, Home
- **No error handling at all** on several screens for CRUD operations
- **Empty state inconsistency** — Tools, Tasks, and Profile use plain text while others use branded layouts
- **Spacing scale is ad-hoc** — no design-token-driven spacing rhythm
- **Proportion issues** on Home (hero too dominant) and Chat (user/assistant padding mismatch)

### 9.4 Critical Test Gaps (Wave 4)
- **Zero Compose UI tests** — no composable rendering tests exist
- **Zero screenshot tests** — no visual regression safety net
- **Only 2 instrumentation tests** — no integration test coverage
- **No navigation tests** — route transitions are untested
- **No accessibility tests** — contentDescription, touch target sizes, contrast all unchecked

### 9.5 Files Examined

| Category | Files |
|----------|-------|
| Screens (14) | HomeScreen, ChatScreen, MemoryScreen, SettingsScreen, HandsScreen, HistoryScreen, ToolsScreen, TasksScreen, RemindersScreen, ProactiveHistoryScreen, KnowledgeGraphScreen, DiagnosticsScreen, ProfileScreen, OnboardingScreen |
| Sub-screens (2) | IdentityEditorScreen, HandEditorDialog |
| Components (15+) | AuraScreenShell, EmptyChatState, MessageBubble, MarkdownText, ModelPickerSheet, SpecialistChips, VisionPromptChips, QuickChipRow, FollowUpSuggestions, AuraAiAvatar, MoaThinkingIndicator, ToolCallBadge, MemoryRecallChip, ToolArgForm, StreamingText |
| Voice overlays (3) | VoiceOverlay, ContinuousVoiceOverlay, HoldToTalk |
| Theme (4) | AuraTokens, Theme, Type, Shapes |
| Navigation (1) | NavGraph |
| Unit tests (23) | SmokeTest + 22 unit/ViewModel tests |
| Build config (2) | app/build.gradle.kts, aura-core/build.gradle.kts |
