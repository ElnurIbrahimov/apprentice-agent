# SOTA: make the agentic loop visible + add multi-modal surfaces

## Goal

Aura is a personal AI with 31 tools, a knowledge graph, memory, vision, and TTS.
The user sees almost none of it — responses just appear. This plan ships
**Dims 1 + 2** of the SOTA assessment:

  - **Dim 1 — Streaming + tool visibility** (commits 1-5): make the agentic
    loop visible. Live tool badges, memory recall chip, inline citations,
    incremental markdown, smart follow-up suggestions.
  - **Dim 2 — Real Jarvis / multi-modal** (commits 6-8): push-to-talk voice
    conversation, live camera + vision, sidebar drawer with conversation
    list.

After this lands, Aura goes from "personal chat app that knows me" to
"personal AI that visibly works in front of me" — the actual SOTA bar.

Dims 3+4 (premium feel + power features) are deferred to a follow-up
plan; they're 80/20 diminishing returns after this batch.

## Constraints

  - `feat/tier-1-friction` branch, no breaking of existing 393 tests.
  - Per-commit gate: `aura-core:testDebugUnitTest :app:testDebugUnitTest
    :app:assembleDebug :app:lintDebug` — all must be green.
  - Atomic commits, one focused user-visible change per commit.
  - No new top-level dependencies unless proven necessary.
  - Plan + execute in same session (no "should I continue?" between
    commits).
  - Hilt-injected types live in `com.aura.*`, NOT `com.aura.providers.*`
    or `com.aura.speech.*`. Always `com.aura.voice.TextToSpeech` etc.
  - `Citation` is `com.aura.tools.Citation`, not `com.aura.ui.util.*`.
  - `Specialist` is `com.aura.agent.Specialist`, not `com.aura.providers.*`.
  - `MutableStateFlow.update {}` is the only legal state mutation; no
    direct `value =` from outside the owning class.
  - Always `rememberMarkdownColors()` before any non-composable
    `parseMarkdown` call from a Compose tree.
  - Hilt doesn't know about helper classes (no `@Inject` on
    ChatSendController pattern) — instantiate via `by lazy` from the
    VM when only one consumer exists.

## Current state (verified)

  - `ChatViewModel`: 628 lines. `ChatSendController`: 263 lines.
  - 393 unit tests pass, lint clean, debug APK builds.
  - `ToolExecutor` returns `ToolResult.NeedsPermission` for permissions
    and `ToolResult.Ok` / `ToolResult.Error` for outcomes.
  - `MemoryAugmentedAgenticLoop.run()` emits `AgentEvent.ToolResult`
    with name + result text + needsPermission + permissionRationale.
    Does NOT emit a "started" event when a tool call begins.
  - `MemoryStore.searchByText()` and `query()` (vector RRF) exist.
  - `ConversationStore` has `load(id)`, `save(conv)`, `recent(n)`,
    `delete(id)`, `fork(id, fromTurnIndex)`. No `search(query)`.
  - `StreamingText` already builds a `MarkdownColors` and renders
    annotated text per chunk.
  - `MarkdownText` parser is private + internal helper.
  - `TtsTool` / `TextToSpeech` are in `com.aura.voice` / `com.aura.tts`.
  - `VisionTool` exists, takes base64 image input.
  - No camera capture / live preview.
  - `NavGraph` has `TopLevelRoute.{Home, Chat, Memory, History, Tasks,
    Hands, Settings, Proactive}.route` — no drawer, no per-chat list.
  - Quick-ask widget (`QuickAskActivity`) exists with `MutableStateFlow`
    bridge, but main ChatScreen has no streaming cursor or tools visible.

## Out of scope (deferred)

  - Dim 3 (premium feel): spring physics, Lottie empty states,
    shared element transitions, Dynamic Island, app icon redesign.
  - Dim 4 (power features): inline message edit + branch, per-conv
    persona, memory import/export, conversation tags/folders, persona
    deep-edit, model picker with filters/search.
  - Sidebar drawer for non-conversation items (memory, tasks, hands).
  - Push-to-talk wake-word ("Hey Aura") — out, just a button.
  - Multi-camera (rear/front toggle is enough for commit 7).
  - Continuous voice mode (LISTENING→THINKING→SPEAKING state machine) —
    the v0.3 widget feature; reuse the same state machine in commit 6.

---

## Commit 1 — live tool call badges in chat

**What the user sees**

Mid-response, when the agent invokes a tool, a small pill appears above
the streaming text:

```
  [ searching the web... ]
  [ reading URL: anthropic.com/news ]
  [ ✓ web_search — 3 results ]
```

The pill animates in (fade + slide up), shows progress, then collapses
to a single-line summary on completion. The next pill animates in
1 frame later. Tap the pill to expand a "details" popover with the
full result preview. Persistent during the turn, dismissed when the
next user message starts.

**Backend / data**

  - `AgenticLoop` needs a new `AgentEvent.ToolStarted(name, args,
    stepId)` event emitted before the tool call is dispatched, paired
    with the existing `ToolResult` event. The loop currently emits
    `ToolResult` AFTER the call returns; we need a "started" event
    BEFORE the call.
  - `AgentEvent` gains `ToolStarted` variant in the sealed class.
  - `ChatSendController` collects `ToolStarted` and:
    1.  Pushes a new `ChatUiState.ToolCallInFlight(stepId, name, args,
        startedAtMs)` onto a list.
    2.  Sets `streaming = true` if not already.
  - When the matching `ToolResult` arrives (same stepId), the in-flight
    entry is replaced by `ChatUiState.ToolCallDone(stepId, name, args,
    resultText, ok, durationMs)`.
  - `ChatUiState` gains:
    ```kotlin
    data class ToolCall(
        val stepId: String,
        val name: String,
        val args: String,
        val resultText: String?,
        val ok: Boolean,
        val startedAtMs: Long,
        val finishedAtMs: Long?,
    )
    val toolCalls: List<ToolCall> = emptyList()
    ```
  - Persist `toolCalls` per turn. `Conversation.addUser/addAssistant`
    does NOT include tools — need a richer `Turn` model. The minimum
    surgery: add `tools: List<TurnToolCall>` field to `Turn` and bump
    DB version. Existing conversations load with empty list.
  - `TurnToolCall` is a Room `@Entity` embedded value (the current
    Turn is Room-managed).
  - Migration: 1→2 (ConversationDatabase) and ConversationStore
    decoder falls back to empty tools list if absent.

**UI / screen**

  - `ChatScreen` renders `state.toolCalls` as a vertical stack of
    `ToolCallBadge` composables BELOW the streaming text and ABOVE
    the chat input. Each badge is a Card with rounded corners,
    subtle elevation, M3 colors.
  - The current streaming text moves up as new badges appear (use
    `Modifier.animateContentSize()` or a `LazyColumn` for the full
    chat scroll).
  - Badge states: `Running` (cyan dot, "running..."), `Done` (green
    check, "X result"), `Failed` (red X, error message).
  - Tap → `AlertDialog` showing the result text, truncated to 2000
    chars with "show more".
  - When the next user message sends, the previous turn's badges
    remain visible in the conversation scroll (it's part of the turn).

**Tests / lock-in**

  - `AgentEventTest` — new test that `MemoryAugmentedAgenticLoop`
    emits `ToolStarted` BEFORE `ToolResult` for a tool call. Run a
    real test with a mock ToolRegistry and capture both events in
    order.
  - `ChatSendControllerTest` (new) — collect events from a fake
    AgenticLoop emitting `ToolStarted` then `ToolResult`, assert
    `state.toolCalls` is updated correctly.
  - `ToolCallBadgeSnapshotTest` (new) — Paparazzi/Compose UI test
    that the badge renders in 3 states (Running, Done, Failed) and
    the tap popover shows.

**Per-commit gate**

  - All existing tests pass.
  - 3 new test files.
  - Build green, lint clean.

**Files**

  - `aura-core/src/main/kotlin/com/aura/agent/AgentEvent.kt` (add
    `ToolStarted` variant)
  - `aura-core/src/main/kotlin/com/aura/agent/MemoryAugmentedAgenticLoop.kt`
    (emit `ToolStarted` before tool call)
  - `app/src/main/kotlin/com/aura/ui/viewmodel/ChatUiState.kt` (add
    `ToolCall`, `toolCalls` field)
  - `app/src/main/kotlin/com/aura/ui/viewmodel/ChatSendController.kt`
    (collect `ToolStarted`, manage `toolCalls` list)
  - `app/src/main/kotlin/com/aura/ui/viewmodel/ChatViewModel.kt`
    (clear `toolCalls` on `newConversation`)
  - `app/src/main/kotlin/com/aura/ui/screens/ChatScreen.kt` (render
    badges)
  - `app/src/main/kotlin/com/aura/ui/components/ToolCallBadge.kt` (new)
  - `aura-core/src/main/kotlin/com/aura/agent/Conversation.kt` (add
    `tools: List<TurnToolCall>` to `Turn`, embed Room @Entity)
  - `aura-core/src/main/kotlin/com/aura/agent/ConversationStore.kt`
    (decoder fallback)
  - `aura-core/.../conversation/ConversationEntity.kt` + migration
  - 3 new test files

---

## Commit 2 — memory recall chip at end of response

**What the user sees**

After the assistant finishes streaming, a small footer chip appears
below the response:

```
  Used 3 memories · 1 fact · 1 hand  [tap to expand]
```

Tap → bottom sheet with the memory titles, fact text, hand name. Each
row is tappable → opens Memory/Hand detail screen. The chip is
subtle (low-elevation, secondary color text) and dismissible.

**Backend / data**

  - `MemoryAugmentedAgenticLoop` already injects memory into the
    system prompt. We need to capture WHICH memories/facts/hands
    were used. The simplest way: the loop already has a `MemoryStore`
    dependency. Add a `lastRecallSummary: RecallSummary?` field on
    `AgenticLoop` that the VM can read.
  - `RecallSummary(memories: List<MemoryId>, facts: List<MemoryId>,
    hands: List<HandId>)` — IDs only, full content loaded on tap.
  - The `MemoryRecallRanker` (or whatever the current recall code is)
    returns the IDs of the rows it picked. Persist that.
  - `AgentEvent.Result` (the final conversation snapshot event) gains
    a `recall: RecallSummary?` field. The loop populates it before
    emitting.
  - Persist `recall` per turn. New `Turn.recall: RecallSummary?`
    field + Room column.

**UI / screen**

  - `ChatScreen` checks the last assistant turn's `recall` field.
    Renders `MemoryRecallChip` below the turn's text.
  - Tap → `ModalBottomSheet` with a list of memories, each
    clickable. Tapping a memory row navigates to MemoryScreen and
    pre-selects that memory (deep link with memory id as query arg).
  - Hide chip if `recall` is null (manual turn, no recall).

**Tests / lock-in**

  - `MemoryRecallChipTest` — given a conversation with recall data,
    chip renders. Tap fires nav callback.
  - `AgenticLoopRecallTest` — given a query that should recall
    memories, the emitted `Result` event has the `recall` field
    populated with the expected IDs.

**Per-commit gate**

  - Existing tests pass + 2 new.
  - Build green, lint clean.

**Files**

  - `aura-core/.../MemoryStore.kt` (add `lastRecall` capture)
  - `aura-core/.../MemoryAugmentedAgenticLoop.kt` (emit `recall` in
    `Result` event)
  - `aura-core/.../AgentEvent.kt` (add `recall: RecallSummary?` to
    `Result`)
  - `aura-core/.../Conversation.kt` (add `recall: RecallSummary?` to
    `Turn`)
  - `aura-core/.../ConversationStore.kt` (decoder)
  - `aura-core/.../conversation/ConversationEntity.kt` + migration
    (v2→v3)
  - `app/.../ChatScreen.kt` (render chip)
  - `app/.../components/MemoryRecallChip.kt` (new)
  - `app/.../nav/NavGraph.kt` (deep-link with memory id)
  - 2 new test files

---

## Commit 3 — inline citation chips [1][2][3] with tap-to-source

**What the user sees**

When the response includes citations (web search, deep research,
Wikipedia tool), inline chips appear in the text:

```
  Anthropic just released Claude 4 [1]. It's a major step
  forward for coding tasks [2][3].
```

The chips are small numbered pills, brand color, tappable. Tap →
popover showing title + URL, with a "Open" button that launches the
browser. ChatGPT-Pattern meets Perplexity.

**Backend / data**

  - The markdown parser already supports `text[1]` for citation
    indexing, but it doesn't currently render as a clickable chip.
  - The `Citation` data class (`com.aura.tools.Citation`) already has
    `index, title, url`. The `extractCitations` function in
    `ChatViewModel` parses them out of the tool result.
  - The chip rendering layer is missing. We need to:
    1.  Parse the streaming text for `[N]` patterns.
    2.  For each, replace with a chip that has the citation index,
        title (from the parsed citations list), and url.
    3.  Render in `MarkdownText` via `AnnotatedString` — but chips
        are typically rendered as `Text` with a `LinkAnnotation` or
        a custom `ClickableText`. Compose's `BasicText` with
        `InlineTextContent` is the right tool.
  - Limit: max 10 citations per response, drop the rest with a
    "... and 5 more" link that opens a list.

**UI / screen**

  - `MarkdownText` gains a `citations: List<Citation>` parameter.
    When the text contains `[N]` and `N <= citations.size`, render
    as a small elevated chip with brand color.
  - Tap chip → small popover with title (bold), URL (regular, blue),
    "Open" button. Outside-tap dismisses.
  - For text without citations, the existing markdown parser is
    unchanged.
  - `ChatScreen` passes `state.citationsForTurn(turnIndex)` to
    `MarkdownText`.

**Tests / lock-in**

  - `MarkdownCitationTest` — given text "Hello [1] world" and
    citations `[Citation(1, "A", "a.com")]`, the rendered output
    contains the citation index in an `AnnotatedString.Range`.
  - `MarkdownTextTest` (existing) — add 3 cases:
    1.  No citations param — text renders as before.
    2.  With citations — `[1]` becomes a chip.
    3.  Out-of-range index (`[99]`) — renders as raw text.

**Per-commit gate**

  - Existing 14 markdown tests pass + 4 new.
  - Build green, lint clean.

**Files**

  - `app/.../components/MarkdownText.kt` (add `citations` param,
    parse `[N]`)
  - `app/.../components/CitationChip.kt` (new)
  - `app/.../screens/ChatScreen.kt` (wire citations to
    `MarkdownText`)
  - `app/.../test/.../MarkdownCitationTest.kt` (new)

---

## Commit 4 — incremental markdown streaming

**What the user sees**

Currently: the response streams as plain text and the markdown
parser runs on the WHOLE string every chunk. So bold/italic/code
sometimes appear with a delay (the parser is waiting for the closing
`**`).

After: the parser runs incrementally. As soon as a `**` is matched,
the text is bold. As soon as `` ` `` is matched, inline code
appears. No more "the assistant typed **bold** and the ** is still
visible".

**Backend / data**

  - The current `parseMarkdown` runs on the full string per chunk
    via `appendInlineMarkdown` → `for (line in lines) { ... }`.
    The issue is that a chunk can end mid-markdown (e.g.
    "**this is bo" before the closing `**`).
  - Fix: instead of re-parsing the whole string every chunk,
    maintain a parser STATE. On each new chunk, advance the parser
    from the previous end state. The parser tracks:
    - Whether we're inside a `**` bold.
    - Whether we're inside a `*` italic.
    - Whether we're inside a `` ` `` inline code.
    - The current line we're on (for headers/lists).
  - This is a state machine, not a regex. Implement as
    `MarkdownStreamParser` that takes chunks and produces
    AnnotatedString ranges.
  - For our purposes, the simplest correct version:
    - Hold the last `parseMarkdown` output.
    - Re-parse on each chunk. The fix is in HOW the parser handles
      unclosed markers.
    - Currently `**(.+?)**` is non-greedy. On "**bold" (no closing
      `**`), the regex doesn't match — so the text appears as raw
      "**bold". We need a "tolerant" mode: if `**` is at the end
      with no closer, treat as opening only and apply bold when
      the closer arrives.
  - Alternative simpler approach: cache the chunk text and
    re-render the entire visible text in the current chunk's
    AnnotatedString, but use a single-pass state machine parser.
  - Measure: parse the last 1000 chars of the streaming buffer
    instead of the full buffer (text already confirmed is stable
    for the last 1000 chars).

**UI / screen**

  - `StreamingText` calls a new `parseMarkdownIncremental(text)`
    instead of `parseMarkdown(text)`. Returns the same
    `AnnotatedString` but with completed markers styled.
  - Visual: the user sees the text appear character-by-character
    as before, but the `**` markers themselves never appear (they
    get consumed and the content is bolded as the closer arrives).

**Tests / lock-in**

  - `MarkdownIncrementalTest`:
    1.  Chunk 1: "**bo" → text shows "**bo" raw (waiting for
        closer).
    2.  Chunk 2: "**bold**" → text shows "**bo" with the `**`
        consumed and "bo" rendered as bold, then "ld**" appended
        and the trailing `**` consumed.
    3.  Chunk 3: "more text" → appended to bold-closed text.
    4.  At end of streaming, "**bold** more text" renders as
        bold "bold" + plain " more text" (no `**` visible).
  - 5 cases covering: bold, italic, code, links, list bullets.

**Per-commit gate**

  - Existing 14 markdown tests pass + 5 new.
  - Build green, lint clean.

**Files**

  - `app/.../components/MarkdownText.kt` (add
    `parseMarkdownIncremental` state machine)
  - `app/.../components/StreamingText.kt` (call incremental
    version)
  - `app/.../test/.../MarkdownIncrementalTest.kt` (new)

---

## Commit 5 — smart follow-up suggestions

**What the user sees**

After each assistant response finishes streaming, 2-4 suggestion
chips appear below the message:

```
  What else can I help with?

  [ What's the weather tomorrow? ]   [ Show me my tasks ]
  [ Set a reminder for 3pm ]         [ Open the calendar ]
```

The chips are derived from the conversation context — what tools
the agent used, what memories were recalled, what the user just
asked. Tapping a chip sends the message directly. The chips are
local, deterministic (no LLM call to generate them), and respect
the persona.

**Backend / data**

  - `FollowUpSuggester` is a pure function:
    `(lastTurn: Turn, toolCalls: List<ToolCall>,
     recall: RecallSummary?) -> List<String>`.
  - Heuristics:
    - If `toolCalls` includes `web_search` and the user is on a
      "search X" turn → suggest related queries ("Search more
      about X", "Compare with Y", "Summarize findings").
    - If `recall.hands` is non-empty → suggest running one
      ("Run this hand again?").
    - If `recall.memories` includes a `category=preference` row →
      suggest applying it ("Use this preference next time?").
    - If the response ended with a question → suggest a yes/no
      follow-up.
    - Default: 3 chips from a hardcoded list of common asks ("What
      can you do?", "Help me plan my day", "Tell me a joke").
  - `ChatSendController` runs `FollowUpSuggester.suggest(...)`
    inside the `AgentEvent.Done` handler. Stores the result in
    `ChatUiState.suggestions: List<String>`.
  - On `newConversation()`, clear suggestions.

**UI / screen**

  - `ChatScreen` renders `state.suggestions` as a horizontal
    scrollable row of `SuggestionChip` below the last turn.
  - `SuggestionChip` is M3's `AssistChip` with a subtle icon
    (lightbulb) and the suggestion text.
  - Tap → `viewModel.onUserMessage(suggestion)` (existing path).

**Tests / lock-in**

  - `FollowUpSuggesterTest`:
    1.  Tool call was `web_search` → first chip is "Search more
        about <topic>".
    2.  Recall has 1 hand → chip is "Run this hand again?".
    3.  No tools, no recall → 3 default chips.
    4.  Last assistant turn ends with "?" → first chip is "Yes"
        and second is "No".
  - 6 cases minimum.

**Per-commit gate**

  - Existing tests pass + 6 new.
  - Build green, lint clean.

**Files**

  - `aura-core/.../agent/FollowUpSuggester.kt` (new, pure function)
  - `app/.../ui/viewmodel/ChatUiState.kt` (add `suggestions` field)
  - `app/.../ui/viewmodel/ChatSendController.kt` (compute
    suggestions in `Done` handler)
  - `app/.../ui/screens/ChatScreen.kt` (render chips)
  - `app/.../ui/components/SuggestionChips.kt` (new)
  - `app/.../test/.../FollowUpSuggesterTest.kt` (new)

---

## Commit 6 — push-to-talk voice conversation mode

**What the user sees**

A mic button next to the chat input. Tap-and-hold (or toggle) → Aura
enters voice mode. The screen transforms:
- Mic input: the chat input becomes a large, animated waveform
  showing live audio levels.
- Transcription: the user's spoken words appear as text in real-time
  (STT) below the waveform.
- The user's turn is automatically sent when speech stops (after
  1.5s of silence) OR a "Send" button.
- Assistant response: still streams text, but ALSO speaks the
  response via TTS as it streams (incremental TTS, not waiting for
  completion).
- A "Tap to interrupt" button mid-response stops the TTS and
  starts listening for the user's next turn.
- The state machine: LISTENING → THINKING → SPEAKING → LISTENING.

**Backend / data**

  - `VoiceConversationController` (new): owns the state machine
    (LISTENING / THINKING / SPEAKING / IDLE), the STT (Android
    `SpeechRecognizer`), the TTS (`TextToSpeech`), the audio level
    polling.
  - State machine transitions:
    - `IDLE → LISTENING` when user taps the mic button.
    - `LISTENING → THINKING` when speech stops (silence detected)
      and the recognized text is non-empty.
    - `THINKING → SPEAKING` when the agent starts emitting
      `TextDelta` events.
    - `SPEAKING → LISTENING` when TTS finishes (utterance complete)
      AND continuous mode is on.
    - Any state → `IDLE` when user taps stop.
  - Continuous mode: a settings toggle (default ON) — Aura keeps
    listening after each response.
  - `ChatSendController` already streams events. Wire
    `VoiceConversationController` to the same stream so the voice
    mode shows the same text as the chat.
  - `SpeechRecognizer` is in `com.aura.voice.SpeechRecognizer`
    (already exists per the round-1 review).

**UI / screen**

  - `VoiceOverlay` (new) is a full-screen Compose layer that
    appears when voice mode is active. It contains:
    - Large animated waveform (Canvas-based, 60Hz update via
      `withFrameNanos`).
    - Live transcript text (AutoSizeText with M3 typography).
    - Mode label (LISTENING / THINKING / SPEAKING) at the top.
    - Stop button at the bottom.
  - `ChatScreen` shows the voice button next to the send button.
  - When voice mode is on, the regular chat input collapses and
    the VoiceOverlay takes over.
  - `MainActivity` may need to request RECORD_AUDIO permission
    (already in manifest per the round-1 review).

**Tests / lock-in**

  - `VoiceConversationControllerTest`:
    1.  `start()` → state IDLE→LISTENING.
    2.  Mock STT delivers a result → state LISTENING→THINKING.
    3.  Mock agent emits text → state THINKING→SPEAKING.
    4.  TTS utterance complete (continuous mode on) → state
        SPEAKING→LISTENING.
    5.  `stop()` from any state → IDLE.
  - 5+ cases.

**Per-commit gate**

  - Existing tests pass + 5 new.
  - Build green, lint clean.
  - RECORD_AUDIO permission already in manifest (verified).

**Files**

  - `app/.../voice/VoiceConversationController.kt` (new)
  - `app/.../voice/VoiceOverlay.kt` (new)
  - `app/.../voice/Waveform.kt` (new)
  - `app/.../ui/screens/ChatScreen.kt` (wire voice overlay)
  - `app/.../ui/viewmodel/ChatViewModel.kt` (expose voice
    state, start/stop)
  - `app/.../test/.../VoiceConversationControllerTest.kt` (new)

---

## Commit 7 — camera preview with live vision

**What the user sees**

A camera button next to the chat input. Tap → a CameraX preview
opens inline in the chat (replacing the input area but staying
in the same screen). The preview shows the live camera feed. A
"capture & ask" button below the preview. The user can also
tap-and-hold a "Quick Ask" floating button to send a single
frame as a vision question.

The captured frame is sent to the cloud vision model (VisionTool
already exists). The user sees the response in the chat as a
normal text response, with a thumbnail of the captured image
attached.

**Backend / data**

  - `CameraController` (new): wraps CameraX `Preview` + `ImageCapture`
    use cases. Manages the camera lifecycle (bind on enter, unbind
    on leave).
  - `VisionTool` already takes `image_base64` + `prompt`. The
    existing flow is: capture bitmap → base64 → tool call. Just
    needs the capture source to be the live preview instead of a
    file picker.
  - `ChatViewModel.onImageCaptured(bitmap, question)` already
    exists. Reuse it from the camera button.

**UI / screen**

  - `CameraPreview` (new) composable: wraps `AndroidView` with
    a `PreviewView`. Full lifecycle handling via `DisposableEffect`.
  - `ChatScreen` shows a small camera icon button next to the
    voice button. Tap → the chat input area is replaced with the
    CameraPreview + a "Capture" button.
  - On capture, the camera panel collapses and the captured
    thumbnail is shown next to the user's message.
  - Permissions: CAMERA is a runtime permission. Use the
    `PermissionDialog` pattern from commit 7 of truth-fixes (the
    permission flow is already in place).

**Tests / lock-in**

  - `CameraControllerTest` (limited — CameraX needs real device):
    1.  `start()` sets up preview use case.
    2.  `capture()` invokes the image capture callback.
    3.  `stop()` unbinds use cases.
  - Skip if CameraX mocking is too painful. Manual verification
    on a real device is the right gate.

**Per-commit gate**

  - Existing tests pass + 2 new (controller unit tests).
  - Build green, lint clean.
  - Manual camera test on emulator/device (document in commit
    message if not feasible in CI).

**Files**

  - `app/.../camera/CameraController.kt` (new)
  - `app/.../camera/CameraPreview.kt` (new)
  - `app/.../ui/screens/ChatScreen.kt` (wire camera button)
  - `app/.../test/.../CameraControllerTest.kt` (new, mockable
    parts only)
  - `app/build.gradle.kts` (verify `androidx.camera:*` deps; add
    if missing)

---

## Commit 8 — sidebar drawer with conversation list + search

**What the user sees**

A hamburger menu button in the ChatScreen top bar. Tap → a
side drawer slides in from the left (Material 3 NavigationDrawer).
Contents:
- Search bar at the top (filter conversations by title).
- "New Chat" button below.
- Conversation list, sorted by `updatedAt` DESC. Each row shows
  the title (1 line), preview (1 line, ellipsized), relative
  timestamp, and a small model tag (the first 2 letters of the
  model used).
- A small star icon to pin conversations to the top.
- A "..." overflow menu per row with Rename, Delete, Fork (the
  existing fork action).
- The current conversation is highlighted with a subtle accent
  background.

This is the ChatGPT-web / Claude-web pattern, made native. It
replaces the History tab for in-chat navigation but keeps the
History tab for browsing the full archive.

**Backend / data**

  - `ConversationStore` gains:
    - `all(): Flow<List<Conversation>>` (already exists as
      `recent(n)` but no full list).
    - `search(query: String): Flow<List<Conversation>>` (new —
      SQL `LIKE` on title + first user message).
    - `pin(id: String, pinned: Boolean)` (new).
    - `rename(id: String, newTitle: String)` (new).
  - Room: `ConversationEntity` gains `pinned: Boolean` column.
    Migration 3→4.
  - `ChatViewModel` exposes:
    - `state.allConversations: StateFlow<List<Conversation>>`
    - `state.searchQuery: String`
    - `setSearchQuery(q)`, `pinConversation(id)`, `renameConv(id, title)`.

**UI / screen**

  - `ChatDrawer` (new) composable: wraps M3 `ModalNavigationDrawer`
    with a `DrawerContent` showing the search bar + list.
  - `ChatScreen` wraps its content in `ModalNavigationDrawer` with
    the drawer content = `ChatDrawer`.
  - The drawer button (hamburger) in the top bar toggles the drawer.
  - The list uses `LazyColumn` for performance with 100+
    conversations.
  - Search updates the list in real-time (debounce 200ms).
  - Tap a conversation → loads it, closes the drawer.
  - Long-press a conversation → action sheet (Pin, Rename, Delete).

**Tests / lock-in**

  - `ChatDrawerViewModelTest`:
    1.  Empty search query → all conversations.
    2.  Search "weather" → conversations whose title or first
        user message contains "weather".
    3.  Pin → conversation appears in pinned section.
    4.  Rename → title updates and persists.
    5.  Delete → conversation removed from list.
  - 5+ cases.

**Per-commit gate**

  - Existing tests pass + 5 new.
  - Build green, lint clean.
  - Manual drawer test on emulator/device (gesture + button).

**Files**

  - `aura-core/.../conversation/ConversationEntity.kt` (add
    `pinned` column)
  - `aura-core/.../ConversationStore.kt` (add `all`, `search`,
    `pin`, `rename`)
  - `aura-core/.../conversation/ConversationDao.kt` (new query
    methods)
  - Migration v3→v4
  - `app/.../ui/viewmodel/ChatViewModel.kt` (expose drawer state)
  - `app/.../ui/screens/ChatScreen.kt` (wrap in drawer)
  - `app/.../ui/components/ChatDrawer.kt` (new)
  - `app/.../ui/components/ConversationRow.kt` (new)
  - `app/.../test/.../ChatDrawerViewModelTest.kt` (new)

---

## What this delivers

After 8 commits, the user gets:

  - **Dim 1 (Streaming + tool visibility)**: 5 commits.
    - Live tool call badges mid-response.
    - Memory recall chip at end of response.
    - Inline citation chips with tap-to-source.
    - Incremental markdown streaming (no flicker on `**`).
    - Smart follow-up suggestion chips.
  - **Dim 2 (Real Jarvis / multi-modal)**: 3 commits.
    - Push-to-talk voice conversation mode.
    - Live camera preview + vision.
    - Sidebar drawer with conversation list + search.

Net result: Aura is visibly working in front of the user, has
multi-modal surfaces (voice + camera), and navigation matches
ChatGPT/Claude web. The "personal AI" promise becomes a visible
one.

## What this does NOT deliver (intentional, for follow-up)

  - Spring physics / 60Hz animations (Dim 3).
  - Shared element transitions (Dim 3).
  - Lottie empty states (Dim 3).
  - Dynamic Island / live activity pings (Dim 3).
  - App icon redesign (Dim 3).
  - Inline message edit + branch (Dim 4).
  - Per-conversation persona (Dim 4).
  - Memory import/export (Dim 4).
  - Conversation tags/folders (Dim 4).
  - Persona deep-edit UI (Dim 4).
  - Model picker with search/filter (Dim 4).
  - Sidebar drawer for non-conversation items (memory, tasks,
    hands) (partial — conversations only).

These are all 80/20 diminishing returns after the 8 commits
above. A separate plan can address them in a follow-up session.

## Risks

  - **Commit 1** (tool badges) requires Room migration + new
    `TurnToolCall` entity. Migration is straightforward but
    bumps DB version; existing users get a clean migration.
  - **Commit 2** (memory recall) requires the agentic loop to
    capture recall IDs. The current recall code is in
    `MemoryAugmentedAgenticLoop` — needs a focused refactor to
    emit IDs alongside the system prompt injection. Risk:
    refactor breaks recall. Mitigation: snapshot test the
    existing recall behavior before the change.
  - **Commit 4** (incremental markdown) is the most subtle. A
    state machine parser is a real piece of code. Risk: the
    state machine handles edge cases (nested bold, unclosed
    code) wrong. Mitigation: extensive test coverage (10+
    cases) and falling back to re-parsing if the state machine
    gets stuck.
  - **Commit 6** (voice) requires RECORD_AUDIO at runtime. The
    permission flow is already in place from truth-fixes
    commit 7. Risk: SpeechRecognizer on different Android
    versions has different behaviors. Mitigation: feature-flag
    voice mode behind a settings toggle.
  - **Commit 7** (camera) requires CAMERA at runtime + CameraX
    deps. Risk: CameraX adds ~500KB to APK and requires
    careful lifecycle handling. Mitigation: feature-flag
    camera mode behind a settings toggle.
  - **Commit 8** (drawer) requires Room migration for
    `pinned` column. Risk: search performance with 1000+
    conversations. Mitigation: SQL `LIKE` with index on
    `title` and `firstUserMessage`.

## Per-commit verification pattern

Each commit runs the same gate before commit:

```bash
cd D:/Aura/android && ./gradlew :aura-core:testDebugUnitTest :app:testDebugUnitTest :app:assembleDebug :app:lintDebug
```

Expected: BUILD SUCCESSFUL, all tests pass, lint clean.

If a commit fails the gate, fix the failure before moving to the
next commit. Do not stack failures.

## Per-commit commit message pattern

```
<type>(<scope>): <one-line summary>

<2-3 sentence problem statement>
<2-3 sentence solution>
<what the user sees — 1-2 sentences>

gate: <test counts> green
```

## End-of-session expectations

  - 8 atomic commits on `feat/tier-1-friction`.
  - All commits unsigned (GPG key not available — environmental).
  - No new top-level dependencies added (CameraX may be added
    in commit 7 if not already present; check first).
  - 3+ Room migrations: ConversationDatabase v1→v2 (commit 1),
    v2→v3 (commit 2), v3→v4 (commit 8). Each migration is
    additive (new column, new entity) and back-compatible.
  - All existing tests pass; ~35 new test cases.
  - 0 behavior regressions on Truth-fixes round 2 surface.
  - No new permissions required (CAMERA + RECORD_AUDIO are
    already in the manifest per the round-1 review).

## Per-commit ETA

  - Commit 1: 4-6 hours (tool badges + Room migration).
  - Commit 2: 3-4 hours (recall capture + chip).
  - Commit 3: 2-3 hours (citation chips in markdown).
  - Commit 4: 4-6 hours (state machine parser).
  - Commit 5: 2-3 hours (heuristic suggester + chips).
  - Commit 6: 6-8 hours (state machine + voice overlay).
  - Commit 7: 3-4 hours (CameraX + preview composable).
  - Commit 8: 3-4 hours (drawer + search + pin/rename).

  Total: 27-38 hours of focused work. 3-5 working days.

  This plan is sized for execution in 1-2 focused sessions if
  parallelized, or 2-3 sessions if serialized with review
  between.
