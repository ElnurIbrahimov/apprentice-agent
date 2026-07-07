# Deep Polish — 10 Missing Capabilities

> **For Hermes:** Plan+execute pattern. No "should I continue?" between commits.

**Goal:** Fix 10 deep structural gaps in real-world usage: context overflow, markdown rendering, failover, tool truncation, images in history, undo, back-press, data usage, auto-lock timeout, export with images.

**Architecture:** Surgical fixes to existing files. No new modules. All changes work within the existing 2-module build (app + aura-core).

**Tech Stack:** Kotlin 1.924, Compose BOM 2024.10.01, Room 2.6.1, Hilt 2.51, coroutines, MockK, Turbine

---

## Pre-execution verification

Before each item, grep the target file to confirm the issue still exists.

---

## Commit 1: Context window management — truncate old turns

**Problem:** `Conversation.toMessages()` sends ALL turns to the LLM. After 30-40 turns with tool results, this overflows context windows.

**Files:**
- Modify: `aura-core/src/main/kotlin/com/aura/agent/Conversation.kt` — `toMessages()`

**Fix:** Add a `maxTurns` parameter to `toMessages()`. When set, keep the system prompt + last N turns. Older turns are dropped from the message list. The full conversation is still stored in Room — this only affects what's sent to the LLM.

**Code:**
```kotlin
fun toMessages(maxTurns: Int = 40): List<ProviderMessage> {
    val out = mutableListOf<ProviderMessage>()
    val sys = listOfNotNull(systemPrompt).filter { it.isNotBlank() }
    if (sys.isNotEmpty()) {
        out += ProviderMessage(role = ProviderMessage.Role.system, content = sys.joinToString("\n\n"))
    }
    val visibleTurns = if (turns.size > maxTurns) turns.takeLast(maxTurns) else turns
    for (turn in visibleTurns) {
        turn.user?.let { out += ProviderMessage(role = ProviderMessage.Role.user, content = it) }
        turn.assistant?.let { out += ProviderMessage(role = ProviderMessage.Role.assistant, content = it) }
        for (toolTurn in turn.toolTurns) {
            out += ProviderMessage(role = ProviderMessage.Role.tool, content = toolTurn.result, toolCallId = toolTurn.id)
        }
    }
    return out
}
```

Also update the loop to call `conversation.toMessages(maxTurns = 40)` instead of `conversation.toMessages()`.

**Verification:** `./gradlew :aura-core:testDebugUnitTest`

---

## Commit 2: Markdown rendering in chat

**Problem:** `MessageBubble` renders LLM responses as plain `Text`. Code blocks, bold, headers, lists all show as raw markdown syntax.

**Files:**
- Modify: `app/src/main/kotlin/com/aura/ui/screens/ChatScreen.kt` — `MessageBubble`

**Fix:** Write a `renderMarkdown` function that parses basic markdown (code blocks, bold, italic, inline code, headers, bullet lists) into `AnnotatedString` and renders it with `Text(text = annotatedString)`. No external library — a lightweight regex-based parser that handles the 80% case.

**Approach:**
- `\`\`\`language\ncode\n\`\`\`` → monospace font + background color
- `**bold**` → bold span
- `*italic*` / `_italic_` → italic span
- `` `inline code` `` → monospace span + background
- `# Header` → bold + larger text
- `- item` / `* item` → bullet (• prefix)

**Verification:** `./gradlew :app:assembleDebug`

---

## Commit 3: Tool result truncation for context

**Problem:** deep_research returns ~6000 chars, firecrawl returns ~8000 chars. These go into `toolTurns[].result` which goes to the model via `toMessages()`. After 3 tool calls, 18K+ chars of tool results in context.

**Files:**
- Modify: `aura-core/src/main/kotlin/com/aura/agent/Conversation.kt` — `toMessages()`

**Fix:** Truncate tool results to 2000 chars when building the message list for the LLM. The full result stays in `toolTurns` for UI display. Only the context sent to the model is truncated.

**Code in toMessages:**
```kotlin
for (toolTurn in turn.toolTurns) {
    val resultForModel = if (toolTurn.result.length > 2000) {
        toolTurn.result.take(2000) + "\n[... truncated]"
    } else {
        toolTurn.result
    }
    out += ProviderMessage(role = ProviderMessage.Role.tool, content = resultForModel, toolCallId = toolTurn.id)
}
```

**Verification:** `./gradlew :aura-core:testDebugUnitTest`

---

## Commit 4: Provider failover on error

**Problem:** If the LLM call fails (401, 429, 500), the error banner shows Retry but uses the same provider. No automatic fallthrough to the next configured provider.

**Files:**
- Modify: `aura-core/src/main/kotlin/com/aura/agent/MemoryAugmentedAgenticLoop.kt` — brain.stream call
- Modify: `aura-core/src/main/kotlin/com/aura/brain/Brain.kt` — stream method

**Fix:** When `brain.stream()` returns an error chunk with `retryable = true`, try the next configured provider before giving up. Keep the current provider as primary — only failover on retryable errors (5xx, 429, network timeout). Don't failover on 401 (bad key — the next provider might work) or 400 (bad request — retrying won't help).

**Code approach:**
In the loop, wrap the `brain.stream()` call in a retry loop:
```kotlin
var streamError: ProviderError? = null
var attemptModel = model
val triedModels = mutableSetOf<String>()
for (attempt in 0 until 2) {
    triedModels.add(attemptModel)
    streamError = null
    brain.stream(attemptModel, conversation.toMessages(maxTurns = 40), ...).collect { chunk ->
        if (chunk.error != null) {
            streamError = chunk.error
            // break out of collect
        } else {
            // normal processing
        }
    }
    if (streamError == null || !streamError!!.retryable) break
    // Try next configured provider
    val next = providerRegistry.configured()
        .firstOrNull { p -> !triedModels.any { it.startsWith("${p.prefix}:") } }
    if (next != null) {
        attemptModel = "${next.prefix}:${next.listModels().firstOrNull() ?: "default"}"
    } else break
}
```

Wait — `providerRegistry.configured()` is not a suspend function but `listModels()` might be. Need to check. Also the brain.stream call is inside a `flow {}` block which is a suspend lambda. The collect is inside the flow. Need to restructure so the failover happens within the flow.

Simpler approach: keep the loop as-is (error → AgentEvent.Error) but add a `retryWithFallback` flag. When the error is retryable and failover is available, emit a warning event and retry with the next provider instead of emitting a terminal error.

Actually simplest: make `Brain.stream()` handle failover internally. Brain already has access to ProviderRegistry. When the primary provider errors with a retryable code, try the next configured provider.

**Verification:** `./gradlew :aura-core:testDebugUnitTest`

---

## Commit 5: Images in conversation history

**Problem:** Images sent to vision are not stored in conversation history. Reopening a conversation from History shows only text — the photo is gone.

**Files:**
- Modify: `aura-core/src/main/kotlin/com/aura/agent/Conversation.kt` — `Turn` data class
- Modify: `aura-core/src/main/kotlin/com/aura/agent/ConversationEntity.kt` — add imageUri field
- Modify: `app/src/main/kotlin/com/aura/ui/viewmodel/ChatViewModel.kt` — store image URI

**Fix:** Add an optional `imageUri: String?` field to `Turn`. When an image is captured/shared, store its URI in the turn. When rendering in chat, show the image inline with the user message. When exporting, include the image reference.

Wait — this requires a Room migration (adding a column to ConversationEntity). The conversation is stored as JSON (`turnsJson`), so adding a field to `Turn` is just a JSON schema change — no migration needed as long as the JSON serializer handles missing fields (which `kotlinx.serialization` does with defaults).

**Code:**
```kotlin
@Serializable
data class Turn(
    val user: String? = null,
    val assistant: String? = null,
    val toolTurns: List<ToolTurn> = emptyList(),
    val imageUri: String? = null,  // NEW: URI of image sent with this turn
)
```

In ChatViewModel, when `onImageCaptured` is called, store the image URI in the conversation:
```kotlin
// In send(), when adding the user turn with an image:
currentConversation = currentConversation.copy(
    turns = currentConversation.turns + Turn(user = text, imageUri = imageUri)
)
```

In ChatScreen, render the image in the message bubble if `turn.imageUri != null`.

**Verification:** `./gradlew :aura-core:testDebugUnitTest :app:assembleDebug`

---

## Commit 6: Undo snackbar for destructive actions

**Problem:** Memory/task/conversation delete is immediate with no undo. Confirm dialog is the only safety net.

**Files:**
- Modify: `app/src/main/kotlin/com/aura/ui/screens/MemoryScreen.kt` — add SnackbarHost
- Modify: `app/src/main/kotlin/com/aura/ui/screens/TasksScreen.kt` — add SnackbarHost
- Modify: `app/src/main/kotlin/com/aura/ui/viewmodel/MemoryViewModel.kt` — soft-delete + restore
- Modify: `app/src/main/kotlin/com/aura/ui/viewmodel/TasksViewModel.kt` — soft-delete + restore

**Fix:** Instead of deleting immediately, implement a "soft delete + undo" pattern:
1. On delete, store the deleted item(s) in a `recentlyDeleted` state variable
2. Show a Snackbar with "Undo" action
3. If user taps Undo within 5 seconds, restore the item
4. If the Snackbar dismisses without Undo, the delete is permanent

For Memory: `MemoryViewModel` keeps a `recentlyDeleted: MemoryEntity?` field. `forget()` moves the item to `recentlyDeleted` then deletes from DAO. `undoDelete()` re-inserts it.

For Tasks: same pattern.

For Conversations: same pattern in HistoryScreen.

**Verification:** `./gradlew :app:testDebugUnitTest`

---

## Commit 7: Back press handling during streaming

**Problem:** Pressing back during streaming navigates away. The stream continues but the partial response is saved to a conversation you're no longer viewing.

**Files:**
- Modify: `app/src/main/kotlin/com/aura/ui/screens/ChatScreen.kt`

**Fix:** Add a `BackHandler` that intercepts back press when `state.isStreaming`. Show a confirm dialog: "Response is still streaming. Stop and save?" with "Stop" and "Cancel" buttons.

**Code:**
```kotlin
if (state.isStreaming) {
    BackHandler(enabled = true) {
        showStopStreamConfirm = true
    }
}

if (showStopStreamConfirm) {
    AlertDialog(
        onDismissRequest = { showStopStreamConfirm = false },
        title = { Text("Stop streaming?") },
        text = { Text("The response will be saved with what's been generated so far.") },
        confirmButton = {
            TextButton(onClick = {
                viewModel.cancel()
                showStopStreamConfirm = false
                // Navigate back
            }) { Text("Stop and save") }
        },
        dismissButton = {
            TextButton(onClick = { showStopStreamConfirm = false }) { Text("Keep listening") }
        },
    )
}
```

**Verification:** `./gradlew :app:assembleDebug`

---

## Commit 8: Data usage indicator

**Problem:** No indicator of cumulative token/data usage. A deep_research call can eat 50+ MB on metered data.

**Files:**
- Create: `aura-core/src/main/kotlin/com/aura/usage/UsageTracker.kt`
- Modify: `app/src/main/kotlin/com/aura/ui/viewmodel/ChatViewModel.kt` — track usage
- Modify: `app/src/main/kotlin/com/aura/ui/screens/ChatScreen.kt` — show usage chip

**Fix:** Create a `UsageTracker` singleton that accumulates approximate character counts for all LLM calls (input + output) and tool fetches. Show a small chip in the chat header: "12K tokens" or "3.4 MB". Reset per conversation or per session.

Simple approach: count characters in/out of `brain.stream()` and tool results. `chars / 4 ≈ tokens`. Display in Settings as cumulative + per-conversation in the chat header.

**Verification:** `./gradlew :aura-core:testDebugUnitTest :app:assembleDebug`

---

## Commit 9: App lock inactivity timeout

**Problem:** App lock only triggers on `ON_RESUME`. No inactivity timeout — if you leave the app open on the chat screen, anyone can read it.

**Simpler problem first:** The `ON_RESUME` gate actually works correctly — it re-locks every time you return to the app from another app or the home screen. The gap is: if the screen turns off while the app is open, `ON_PAUSE` fires, then `ON_STOP` fires. When you wake the phone, `ON_START` + `ON_RESUME` fire → the lock triggers. So the lock DOES work on screen-off.

The real gap: if you're actively using the app and someone picks up your phone, there's no inactivity re-lock. But this is a rare edge case for a personal sideload app — the ON_RESUME gate covers the common case (put phone down → pick up → locked).

**Decision:** The ON_RESUME gate already handles the main case. Skip inactivity timeout — it's edge-case engineering for a personal app.

---

## Commit 10: Export with images

**Problem:** `exportMarkdown` exports text turns and tool calls but not images.

**Files:**
- Modify: `app/src/main/kotlin/com/aura/ui/viewmodel/HistoryViewModel.kt` — `exportMarkdown()`

**Fix:** When a turn has an `imageUri`, include a markdown image reference in the export: `![image](file:///path/to/image.png)`. If the image is a content URI, it may not be accessible after export — include a note: `[Image: <uri>]`.

**Code:**
```kotlin
turn.imageUri?.let { uri ->
    append("![image](").append(uri).append(")\n\n")
}
```

**Verification:** `./gradlew :app:testDebugUnitTest`

---

## Summary

| # | Item | Severity | Commit |
|---|------|----------|--------|
| 1 | Context window truncation | Critical | Yes |
| 2 | Markdown rendering | Critical | Yes |
| 3 | Tool result truncation for context | High | Yes |
| 4 | Provider failover | High | Yes |
| 5 | Images in conversation history | Medium | Yes |
| 6 | Undo snackbar for destructive actions | Medium | Yes |
| 7 | Back press during streaming | Medium | Yes |
| 8 | Data usage indicator | Low | Yes |
| 9 | App lock inactivity timeout | Skip — ON_RESUME covers it | Skip |
| 10 | Export with images | Low | Yes (depends on #5) |

**9 commits, 9 items.** Run full gate after all commits.

---

## Verification gates

After all commits:
1. `./gradlew :aura-core:testDebugUnitTest :app:testDebugUnitTest :app:assembleDebug --rerun-tasks`
2. `gh run watch <latest>` — CI green