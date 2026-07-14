# Android Compose UI / UX / Daily-Use Friction Audit

**Date:** 2026-07-10  
**Branch:** feat/tier-1-friction  
**Repo:** D:\Aura\android  
**Reference Web:** D:\Aura\web (built React/TypeScript app)

---

## P1 — Must Fix Before Ship

### 1. Mood / Emotion-Aware Theming — Missing on Android Entirely

**Severity:** P1  
**Files:**  
- `D:\Aura\web\src\hooks\useMoodTheme.ts` (web has it)  
- `D:\Aura\web\src\utils\moodTheme.ts` (web implementation)  
- `D:\Aura\android\app\src\main\kotlin\com\aura\ui\theme\Theme.kt` (Android — no mood integration)  

**Finding:** The web app uses `useMoodTheme()` to convert PAD mood data (valence/arousal from the chat store) into dynamic CSS custom properties (`--mood-accent`, `--mood-glow`, `--mood-bg-tint`, `--mood-mesh-1`) that hue-shift the entire UI — purple for neutral, warm for happy, cool blue for sad, red for angry/aroused. The Android theme (`AuraTheme` / `AuraTokens`) has zero awareness of this. The `chatStore.ts` on the web carries `mood: MoodState | null` and updates it from the server; Android's `ChatViewModel`/`ChatUiState` has no `mood` field.

**Fix:** Add `mood: MoodState?` to Android's `ChatUiState`. Map valence/arousal to Compose `Color` overrides (e.g., `MoodColors(valence, arousal): Colors`) and inject into `MaterialTheme` via `AuraTheme` or a wrapper. Mirror the web's hue-rotation math from `getMoodCSSVars`.

---

### 2. Inline Citation Badges — Not Rendered in Assistant Messages

**Severity:** P1  
**Files:**  
- `D:\Aura\android\app\src\main\kotlin\com\aura\ui\components\MessageBubble.kt` lines 422-424, 463-500  
- `D:\Aura\web\src\components\MessageBubble.tsx` lines 54-127 (inline citation badges), lines 201-228 (renderContentWithCitations)  

**Finding:** On the web, `[1]`, `[2]` patterns in assistant text are rendered as interactive tap-to-open/hover-for-tooltip circular badges with citation tooltip overlay. On Android, citations are rendered as a separate `CitationChipRow` component below the message body (lines 422-424 in `MessageBubble.kt`). **The inline `[N]` markers in the raw markdown text are rendered as plain text** — the user sees literal `[1]` characters inside the message, and the citation chips below are duplicates. The `MarkdownText.kt` parser does not have a citation-badge renderer.

**Fix:** Either (a) strip `[N]` patterns from text when citations exist and replace them with clickable in-text badges, or (b) pre-process `[N]` → styled clickable spans in `MarkdownText.kt` that open the citation dialog. Add `Modifier.semantics { onClick { showCitations() } }` for accessibility.

---

### 3. Missing Top-Level Tabs: Create and Insights (Web Parity Gap)

**Severity:** P1  
**Files:**  
- `D:\Aura\android\app\src\main\kotlin\com\aura\ui\nav\NavGraph.kt` lines 72-80  
- `D:\Aura\web\src\components\BottomTabBar.tsx` lines 19-25  
- `D:\Aura\web\src\components\TabRouter.tsx` (full file — 30+ panels)  

**Finding:** The web has 5 bottom tabs: **Chat, Create, Tools, Insights, Settings**. Android has 4: **Home, Chat, Memory, Settings**.  
- **Create tab** (Code/Website/App/Game/Dashboard/Slides/Image sub-tabs) — Android has no equivalent. `AppCreator`, `GameCreator`, `DashboardCreator`, code interpreter invocation exist as separate/shell screens but are not surfaced in the main navigation.  
- **Insights tab** (11 sub-tabs: Mind/Insights/Briefing/Dreams/Evolution/World/Activity/Memory/Graph/Queue/Advanced) — Android has `MemoryScreen`, `GraphScreen` as flat destinations, and `HomeScreen` shows some of this data as cards, but there's no comprehensive "Insights" hub.  
- **Tools tab** on web is a full tool launcher with 20+ sub-panels; Android's `ToolsScreen` is a read-only tool browser.

**Fix:** Add "Create" and "Insights" (or "Tools") to the Android bottom bar, replacing "Home" or adding as a 5th tab. Map the web's sub-tab content to Compose screens.

---

### 4. No Clickable Links in Markdown Output

**Severity:** P1  
**Files:**  
- `D:\Aura\android\app\src\main\kotlin\com\aura\ui\components\MarkdownText.kt` lines 204-222 (link rendering)  

**Finding:** `MarkdownText.kt` renders links as styled text with underline + primary color but explicitly states they are NOT clickable (line 206-208: "no click handler"). The link URL is appended in parentheses after the label text (e.g., `label (url)`). Users cannot tap links to open them in a browser. The web uses `<a href={href} target="_blank">` which opens in a new tab.

**Fix:** Use `ClickableText` with `AnnotatedString` + `pushStringAnnotation("URL", url)` to handle link taps via `UriHandler.openUri()`. Replace the current `Text(... parseMarkdown(...))` with a clickable variant.

---

### 5. No Message Reactions / Feedback (Thumbs Up/Down)

**Severity:** P1  
**Files:**  
- `D:\Aura\android\app\src\main\kotlin\com\aura\ui\components\MessageBubble.kt` lines 451-456 (only Copy action in footer)  
- `D:\Aura\web\src\components\MessageBubble.tsx` lines 422-440 (`handleReaction`)  

**Finding:** The web has thumbs-up/thumbs-down reaction buttons on each assistant message that POST feedback to `/api/chat/messages/feedback`. Android has zero reaction UI — only a Copy button in the bubble footer. The feedback API endpoint and chat store are missing from Android.

**Fix:** Add reaction buttons (`Icons.Filled.ThumbUp` / `ThumbDown`) to the assistant message footer. Wire to `ChatViewModel.feedback(messageId, rating)` → API call.

---

## P2 — Should Fix

### 6. Memory Screen: No Embedding Status / Decay Visualization

**Severity:** P2  
**Files:**  
- `D:\Aura\android\app\src\main\kotlin\com\aura\ui\screens\MemoryScreen.kt` lines 448-455 (shows `decayScore < 0.5f` as text "fading")  
- `D:\Aura\web\src\components\MemoryTimeline.tsx` (web has visual decay timeline)  

**Finding:** Android shows a simple text label "fading" for decaying memories. The web has a visual timeline with decay gradient bars. Android's `MemoryRow` shows importance percentage, recall count, tags — data the web doesn't display. But Android lacks the timeline visualization.

---

### 7. Graph Screen: No Visual Graph Rendering (Text-Only Node List)

**Severity:** P2  
**File:** `D:\Aura\android\app\src\main\kotlin\com\aura\ui\screens\GraphScreen.kt`  

**Finding:** The entire "Knowledge Graph" screen is a **text-only scrolling list of nodes** with no graph visualization. No edges shown. No visual layout. The web has an interactive graph renderer. This is a flat database dump, not a knowledge graph explorer.

---

### 8. Chat History Screen: No Batch Operations

**Severity:** P2  
**Files:**  
- `D:\Aura\android\app\src\main\kotlin\com\aura\ui\screens\HistoryScreen.kt` lines 153-168  

**Finding:** History screen supports individual rename, delete, pin, share. No multi-select for batch delete/export. No swipe-to-delete gesture. Web has conversation search drawer with similar individual operations.

---

### 9. Profile Screen: Save Button Without Visual Feedback

**Severity:** P2  
**Files:**  
- `D:\Aura\android\app\src\main\kotlin\com\aura\ui\screens\ProfileScreen.kt` lines 70-76 (only "Save name" button, no snackbar/toast)  

**Finding:** "Save name" button has no success/error feedback. "Add trait" and "Add fact" icons provide no confirmation. No loading state for save operations.

---

### 10. Bottom Bar: No Badge Indicators for Home/Actions

**Severity:** P2  
**Files:**  
- `D:\Aura\android\app\src\main\kotlin\com\aura\ui\nav\NavGraph.kt` lines 234-308  

**Finding:** The web's bottom tab bar supports badges (red dot indicators) on tabs. Android's floating pill bottom bar has no badge support. Proactive unread count, pending tasks count, etc. are not surfaced on the tab bar.

---

### 11. Chat Header: No Conversation Export/Share

**Severity:** P2  
**Files:**  
- `D:\Aura\android\app\src\main\kotlin\com\aura\ui\screens\ChatScreen.kt` lines 586-700 (ChatHeader)  
- `D:\Aura\web\src\components\ChatToolbar.tsx` (web has export/share menu)  

**Finding:** The web's `ChatToolbar` has export-as-Markdown/JSON/HTML, copy-to-clipboard, and share-link functionality. Android's `ChatHeader` only has TTS toggle, new conversation, history, delete conversation. No export/share. History screen has export-all-markdown but per-conversation share from chat is missing.

---

### 12. Accessibility: Several Icons Missing contentDescription

**Severity:** P2  
**Files (partial list — 21 files with the pattern):**  
- `D:\Aura\android\app\src\main\kotlin\com\aura\ui\screens\HomeScreen.kt` lines 232, 400 — `contentDescription = null`  
- `D:\Aura\android\app\src\main\kotlin\com\aura\ui\components\EmptyChatState.kt` line 154 — `contentDescription = null`  
- `D:\Aura\android\app\src\main\kotlin\com\aura\ui\screens\ToolsScreen.kt` line 82 — `contentDescription = null`  

**Finding:** Search icons, decorative icons, and avatar icons frequently use `contentDescription = null` or are missing semantic labels. TalkBack users cannot navigate effectively.

---

## P3 — Polish / Nice to Have

### 13. No Pull-to-Refresh on Chat or List Screens

Web has pull-to-refresh on ChatContainer (reconnects WebSocket). Android has zero pull-to-refresh anywhere.

### 14. No In-Chat Message Search (Ctrl+F)

Web has in-conversation search (Ctrl+F overlay with match navigation). Android has no equivalent.

### 15. Message Long-Press Context Menu

Web has a long-press context menu on messages with Copy/Regenerate/Share. Android has no context menu — only a Copy button in the footer.

### 16. No Keyboard Shortcuts

Web has extensive keyboard shortcuts (Ctrl+K focus input, Ctrl+F search, etc.). Android has none (expected for mobile, but notable for tablet/keyboard users).

---

## Daily-Use Friction Summary

### Good (No Issue)
- **Chat auto-scroll**: Smart auto-scroll with `isNearBottom` detection ✅
- **Streaming cursor + tok/s badge**: Parity with web ✅
- **Empty states**: Every screen has appropriate empty state text ✅
- **Memory CRUD**: Full create/edit/delete/search/filter ✅
- **Tasks CRUD**: Full create/edit/delete/done/reopen with date pickers ✅
- **Delete confirmations**: All destructive actions have AlertDialog confirmation ✅
- **Back-press during streaming**: Intercept + save partial response ✅
- **Insets**: Proper `imePadding()`, `statusBarsPadding()`, `navigationBarsPadding()` ✅
- **Haptic feedback**: Streaming end haptic implemented ✅
- **Custom fonts**: InterDisplay, Fraunces, JetBrainsMono configured ✅

### Issues
1. **No mood/emotion theming** — entire app has a static color scheme
2. **Inline citations render as raw text `[1]`** — confusing duplicates with CitationChipRow
3. **Cannot tap links in messages** — LLM output with URLs is frustrating
4. **No message feedback** — missing user engagement signal for the ML loop
5. **No Create/Insights tabs** — key web parity gap
6. **Graph screen is just a list** — not a graph at all
7. **Missing export from chat** — export exists in History but not inline
8. **No swipe gestures** — no swipe-to-delete, no pull-to-refresh
9. **Accessibility gaps** — various icons missing contentDescription
10. **Knowledge Graph is purely textual** — no visual rendering

---

## File-by-File Scorecard

| File | Issues |
|------|--------|
| `Theme.kt` | No mood awareness (P1) |
| `NavGraph.kt` | Missing Create/Insights tabs (P1); no badge support (P2) |
| `MessageBubble.kt` | Inline citation badges plain text (P1); no reactions (P1); no context menu (P3) |
| `MarkdownText.kt` | Links not clickable (P1) |
| `ChatScreen.kt` | No export/share (P2); no in-chat search (P3) |
| `HomeScreen.kt` | Some icons missing contentDescription (P2) |
| `MemoryScreen.kt` | No decay timeline visualization (P2) |
| `GraphScreen.kt` | Text-only node list, not a graph (P2) |
| `ProfileScreen.kt` | No save feedback (P2) |
| `HistoryScreen.kt` | No batch operations (P2) |
| `EmptyChatState.kt` | Icon missing contentDescription (P2) |
| `ToolsScreen.kt` | Icon missing contentDescription (P2) |
| `StreamingText.kt` | ✅ Good parity with web |
| `StreamingMarkdownState.kt` | ✅ Good flicker suppression |
| `TasksScreen.kt` | ✅ Full CRUD |
| `RemindersScreen.kt` | ✅ Good empty state |
