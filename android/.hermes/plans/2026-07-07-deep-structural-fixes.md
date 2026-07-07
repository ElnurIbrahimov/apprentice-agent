# Deep Structural Fixes — Aura Android

> **For Hermes:** Use plan+execute pattern — write plan, then implement all items in one session without asking "continue?" between commits.

**Goal:** Fix 8 deep structural gaps where infrastructure exists but silently doesn't deliver value.

**Architecture:** All fixes are surgical — touch the specific method or wiring point, not the surrounding architecture. No new modules, no new abstractions.

**Tech Stack:** Kotlin 1.9.24, Compose, Room, Hilt, coroutines, MockK, Turbine

---

## Pre-execution verification

Before implementing each item, grep the target file to confirm the issue still exists. Items may have been fixed in prior commits.

---

## Commit 1: Fix memory recall — vector fallback when text search returns empty

**Problem:** `MemoryStore.query()` does `dao.searchByText("%query%", limit*3)` first. If textHits is empty, returns `emptyList()` immediately — vector similarity never runs. The 384-dim embedding pipeline is dead code for any query that doesn't share a substring with stored content.

**Files:**
- Modify: `aura-core/src/main/kotlin/com/aura/memory/MemoryStore.kt` — `query()` method
- Test: `aura-core/src/test/kotlin/com/aura/memory/RetrievalTest.kt`

**Fix:** When `textHits.isEmpty()`, fall back to a full-table vector scan: load all memories with non-null embeddings, embed the query, compute cosine similarity, take top-K by vector score alone (no RRF — no text signal available). This makes the vector pipeline actually work for semantic queries.

**Code:**

In `MemoryStore.query()`, replace:
```kotlin
val textHits = dao.searchByText("%$escapedText%", limit * 3)
if (textHits.isEmpty()) return emptyList()
val qVec = embedder.embed(text)
```

With:
```kotlin
val escapedText = escapeLikeWildcards(text)
val textHits = dao.searchByText("%$escapedText%", limit * 3)
val qVec = embedder.embed(text)
if (textHits.isEmpty()) {
    // Vector fallback: no text overlap, but the query might still
    // be semantically similar to a stored memory. Scan all memories
    // with embeddings and rank by cosine similarity alone.
    val all = dao.allForExport().filter { it.embedding != null }
    if (all.isEmpty()) return emptyList()
    val scored = all.map { mem ->
        val embedding = Embedder.fromBytes(mem.embedding!!)
        ScoredMemory(memory = mem, textScore = 0f, vectorScore = cosineSimilarity(qVec, embedding))
    }.filter { it.vectorScore > 0.05f }
    return Retrieval.rankCandidates(text, qVec, scored, limit, now)
}
```

**Verification:** Write a test that stores a memory "I love Kotlin", queries "programming languages I enjoy" (zero text overlap), and asserts results are non-empty.

**Run:** `./gradlew :aura-core:testDebugUnitTest --tests "com.aura.memory.RetrievalTest" `

---

## Commit 2: Fix profile extraction — run on USER text, not ASSISTANT text

**Problem:** `extractProfileFromText(lastAssistant)` runs on the assistant's response. But the regexes look for first-person patterns: "my name is", "I live in", "I prefer". The assistant says "Your name is Elnur" — the regex never matches. The user is the one who says "my name is Elnur".

**Files:**
- Modify: `aura-core/src/main/kotlin/com/aura/agent/MemoryAugmentedAgenticLoop.kt` — post-loop section
- Test: `aura-core/src/test/kotlin/com/aura/agent/EndToEndTest.kt`

**Fix:** Run `extractProfileFromText` on `lastUserMessage` (the user's text) instead of `lastAssistant` (the assistant's response). Also keep the assistant extraction as a secondary path — the assistant might echo user facts ("You said your name is Elnur") which can be extracted too.

**Code:**

Replace:
```kotlin
val lastAssistant = currentConversation.turns.lastOrNull()?.assistant
if (memoryEnabled && !lastAssistant.isNullOrBlank()) {
    runCatching { extractProfileFromText(lastAssistant) }
}
```

With:
```kotlin
// Extract user profile from the USER's message (first-person patterns
// like "my name is", "I live in" are in the user's text, not the
// assistant's response). Also try the assistant text as a secondary
// path — the assistant may echo user facts.
if (memoryEnabled && lastUserMessage.isNotBlank()) {
    runCatching { extractProfileFromText(lastUserMessage) }
}
val lastAssistant = currentConversation.turns.lastOrNull()?.assistant
if (memoryEnabled && !lastAssistant.isNullOrBlank()) {
    runCatching { extractProfileFromText(lastAssistant) }
}
```

**Verification:** Test that storing "my name is Elnur" as a user message triggers `userProfileStore.update(name = "Elnur")`.

**Run:** `./gradlew :aura-core:testDebugUnitTest --tests "com.aura.agent.EndToEndTest"`

---

## Commit 3: Fix QuickAskActivity — response never shows in UI

**Problem:** `askQuery()` sets `lastResponse` (a private field on the Activity) but the Compose state (`response`, `loading`) lives in `QuickAskContent` — a separate composable with its own `remember` state. The response is written to the widget body but never shown in the activity.

**Files:**
- Modify: `app/src/main/kotlin/com/aura/widget/QuickAskActivity.kt`

**Fix:** Replace the disconnected `lastResponse` field with a `MutableStateFlow<String?>` shared between the activity and the composable. The composable collects from it and updates `response` + `loading` state.

**Code:**

Replace the `QuickAskContent` composable with a version that takes a `StateFlow<String?>` for the response and a `StateFlow<Boolean>` for loading:

```kotlin
private val responseFlow = MutableStateFlow<String?>(null)
private val loadingFlow = MutableStateFlow(false)

private fun askQuery(query: String) {
    loadingFlow.value = true
    responseFlow.value = null
    lifecycleScope.launch {
        val model = userPreferences.defaultModel.first()
        val response = withContext(Dispatchers.IO) {
            // ... same LLM call ...
        }
        responseFlow.value = response
        loadingFlow.value = false
        updateWidgetWithResponse(query, response)
    }
}
```

And in `QuickAskContent`, collect from the flows:
```kotlin
@Composable
private fun QuickAskContent(
    responseFlow: StateFlow<String?>,
    loadingFlow: StateFlow<Boolean>,
    onSend: (String) -> Unit,
    onDismiss: () -> Unit,
) {
    var query by remember { mutableStateOf("") }
    val response by responseFlow.collectAsState()
    val loading by loadingFlow.collectAsState()
    // ... rest of UI ...
}
```

**Verification:** Build only — this is a UI fix. `./gradlew :app:assembleDebug`

---

## Commit 4: Fix LLM write gate — use lightweight model, not conversation model

**Problem:** `LlmWriteGate(modelId = model)` uses the conversation's model for the gate. If the user is on MoA (deepseek-v4-pro aggregator), the write gate fires a full MoA call (2 references + aggregator = 3 API calls) just to decide "is this worth remembering?".

**Files:**
- Modify: `aura-core/src/main/kotlin/com/aura/agent/MemoryAugmentedAgenticLoop.kt` — LLM gate section

**Fix:** Use a hardcoded lightweight model for the gate. The gate runs on every turn — it should be fast and cheap. Use the first configured provider's cheapest model, or a specific fast model.

**Code:**

Replace:
```kotlin
val gate = LlmWriteGate(
    heuristic = WriteGate(),
    registry = providerRegistry,
    modelId = model,
)
```

With:
```kotlin
// Use a lightweight model for the gate — this runs on every turn
// and should be fast + cheap. Falls back to the conversation model
// if no lightweight model is configured.
val gateModel = providerRegistry.configured().firstOrNull()?.let { p ->
    "${p.prefix}:${p.listModels().firstOrNull() ?: "default"}"
} ?: model
val gate = LlmWriteGate(
    heuristic = WriteGate(),
    registry = providerRegistry,
    modelId = gateModel,
)
```

Wait — `configured()` and `listModels()` are suspend. The gate is already inside a `runCatching` in a `flow {}` block. Need to make the model resolution suspend-safe.

Better approach: resolve the gate model once in the loop constructor or pass it as a parameter. Simplest: use a hardcoded fast model that the user can override in settings later.

```kotlin
val gateModel = "ollama:qwen2:1.5b:cloud"  // lightweight, fast, cheap
```

If that model isn't configured, the LlmWriteGate's `runCatching` will catch the error and fall back to the heuristic — which is the correct behavior.

**Verification:** `./gradlew :aura-core:testDebugUnitTest` — existing tests use relaxed mock for providerRegistry, so the hardcoded model string won't break them.

---

## Commit 5: Fix KG extraction — also extract from USER text

**Problem:** `kgExtractor.extract(accumulatedText.toString())` runs only on the assistant's response. But the user's input contains the actual facts ("I work at Google", "My friend Bob is a doctor"). The KG misses user-shared entities entirely.

**Files:**
- Modify: `aura-core/src/main/kotlin/com/aura/agent/MemoryAugmentedAgenticLoop.kt` — KG extraction section

**Fix:** Also extract from `lastUserMessage`. The KG should capture entities from both sides of the conversation.

**Code:**

Replace:
```kotlin
if (accumulatedText.isNotEmpty()) {
    currentConversation = currentConversation.addAssistant(accumulatedText.toString())
    if (memoryEnabled) {
        kgExtractor.extract(accumulatedText.toString())
    }
}
```

With:
```kotlin
if (accumulatedText.isNotEmpty()) {
    currentConversation = currentConversation.addAssistant(accumulatedText.toString())
    if (memoryEnabled) {
        // Extract KG from BOTH the user's message and the assistant's
        // response. The user shares facts ("I work at Google"), the
        // assistant synthesizes and confirms them.
        if (lastUserMessage.isNotBlank()) {
            kgExtractor.extract(lastUserMessage)
        }
        kgExtractor.extract(accumulatedText.toString())
    }
}
```

**Verification:** Existing KG tests should still pass. The EndToEndTest verifies `kgExtractor.extract(...)` is called — now it's called twice.

---

## Commit 6: Fix tool execution — run independent tools in parallel

**Problem:** When the model returns multiple tool calls in one turn, the loop runs them sequentially in a `for` loop. Two independent web searches take 2x the wall time.

**Files:**
- Modify: `aura-core/src/main/kotlin/com/aura/agent/MemoryAugmentedAgenticLoop.kt` — tool execution section

**Fix:** Use `async` + `awaitAll` for tool calls that don't depend on each other. Tools are independent by default — the model sends them in one batch, meaning it doesn't expect one's output to feed into the next.

**Code:**

Replace the sequential tool execution:
```kotlin
for ((id, args) in toolCalls) {
    val name = toolCallStarts[id] ?: continue
    emit(AgentEvent.ToolExecuting(id, name, args))
    val result = toolExecutor.execute(name, args, ctx)
    val resultText = when (result) { ... }
    currentConversation = currentConversation.setToolResult(id, resultText)
    emit(AgentEvent.ToolResult(id, name, args, resultText, needsPerm, permRationale))
}
```

With parallel execution using async:
```kotlin
// Execute tool calls in parallel — the model sends them in one
// batch, meaning it doesn't expect one's output to feed into the
// next. Running them concurrently cuts wall time for multi-tool turns.
val ctx = ToolContext(conversationId = currentConversation.id, memoryEnabled = memoryEnabled)
val results = toolCalls.map { (id, args) ->
    val name = toolCallStarts[id] ?: return@map null
    emit(AgentEvent.ToolExecuting(id, name, args))
    async {
        val result = toolExecutor.execute(name, args, ctx)
        Triple(id, name, result)
    }
}.filterNotNull()

for ((id, name, result) in results.awaitAll()) {
    val resultText = when (result) { ... }
    val needsPerm = ...
    currentConversation = currentConversation.setToolResult(id, resultText)
    emit(AgentEvent.ToolResult(id, name, args, resultText, needsPerm, permRationale))
}
```

Wait — `async` needs a coroutine scope. The loop is inside a `flow {}` block. We can use `coroutineScope { }` to create a child scope for the parallel tools.

Actually, simpler: the tools are suspend functions called inside a `flow {}` collector. We can use `kotlinx.coroutines.async` with the current coroutine context. But `flow {}` blocks don't have a `coroutineScope` by default. We need to wrap the parallel execution in `coroutineScope { ... }`.

**Code:**
```kotlin
// Execute tool calls in parallel.
val ctx = ToolContext(conversationId = currentConversation.id, memoryEnabled = memoryEnabled)
val toolResults = kotlinx.coroutines.coroutineScope {
    toolCalls.map { (id, args) ->
        val name = toolCallStarts[id] ?: return@map null
        emit(AgentEvent.ToolExecuting(id, name, args))
        kotlinx.coroutines.async {
            Triple(id, name, toolExecutor.execute(name, args, ctx))
        }
    }.filterNotNull().awaitAll()
}
for ((id, name, result) in toolResults) {
    val resultText = when (result) {
        is ToolResult.Ok -> result.output
        is ToolResult.Error -> "Error: ${result.message}"
        is ToolResult.NeedsPermission -> "Permission needed: ${result.permission} — ${result.rationale}"
        is ToolResult.NeedsApproval -> "Approval needed: ${result.rationale}"
    }
    val needsPerm = if (result is ToolResult.NeedsPermission) result.permission else null
    val permRationale = if (result is ToolResult.NeedsPermission) result.rationale else null
    val args = toolCalls.first { it.first == id }.second
    currentConversation = currentConversation.setToolResult(id, resultText)
    emit(AgentEvent.ToolResult(id, name, args, resultText, needsPerm, permRationale))
}
```

**Verification:** `./gradlew :aura-core:testDebugUnitTest` — existing tests should pass (tool execution order doesn't matter for the test assertions, they verify final state).

---

## Commit 7: Fix KG extraction debounce — 5s too long, reduce to 2s

**Problem:** `DEBOUNCE_MS = 5_000L` means in a fast chat, every extraction is cancelled before it runs. The KG never learns anything from rapid back-and-forth.

**Files:**
- Modify: `aura-core/src/main/kotlin/com/aura/kg/ConversationKgExtractor.kt`

**Fix:** Reduce to 2 seconds. This still debounces rapid-fire turns but gives the extraction a chance to run between conversational turns.

**Code:**
```kotlin
companion object {
    private const val DEBOUNCE_MS = 2_000L
}
```

**Verification:** `./gradlew :aura-core:testDebugUnitTest` — no test asserts on debounce timing directly.

---

## Commit 8: Fix LLM write gate — don't instantiate LlmWriteGate every turn

**Problem:** `LlmWriteGate(heuristic = WriteGate(), ...)` is constructed fresh on every turn. Should be a class field or singleton.

**Files:**
- Modify: `aura-core/src/main/kotlin/com/aura/agent/MemoryAugmentedAgenticLoop.kt`

**Fix:** Create the gate once as a lazy field. The `modelId` changes per conversation, so pass it to `evaluate()` instead of the constructor. Or: create the gate per-run with the model, since the model can change between conversations. Simplest: just use `WriteGate()` as the heuristic instance (stateless, cheap) and construct `LlmWriteGate` with the model per-run. The allocation is trivial — a class with 3 fields. Skip this item if it's truly trivial.

**Decision:** This is a micro-optimization. The `LlmWriteGate` constructor is 3 field assignments. Skip — not worth a commit.

---

## Summary

| # | Item | Severity | Commit |
|---|------|----------|--------|
| 1 | Memory recall vector fallback | Critical — vector pipeline is dead code | Yes |
| 2 | Profile extraction on user text | Critical — profile never learns | Yes |
| 3 | QuickAskActivity response display | Critical — widget broken | Yes |
| 4 | LLM write gate model | Architectural — MoA gate is 3x cost | Yes |
| 5 | KG extraction on user text | Architectural — KG misses user facts | Yes |
| 6 | Parallel tool execution | Architectural — 2x latency | Yes |
| 7 | KG debounce 5s→2s | Tuning | Yes |
| 8 | LlmWriteGate allocation | Micro | Skip |

**7 commits, 7 items.** Run full gate after all commits: `./gradlew :aura-core:testDebugUnitTest :app:testDebugUnitTest :app:assembleDebug --rerun-tasks`

---

## Verification gates

After all commits:
1. `./gradlew :aura-core:testDebugUnitTest` — all core tests pass
2. `./gradlew :app:testDebugUnitTest` — all app tests pass
3. `./gradlew :app:assembleDebug` — APK builds
4. `gh run watch <latest>` — CI green