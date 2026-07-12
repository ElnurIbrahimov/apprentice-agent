package com.aura.agent

import com.aura.kg.ConversationKgExtractor
import com.aura.memory.LlmWriteGate
import com.aura.memory.MemoryStore
import com.aura.memory.WriteGate
import com.aura.providers.ChatOptions
import com.aura.providers.ProviderMessage
import com.aura.providers.ProviderMessage.Role
import com.aura.providers.ToolDefinition
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.async
import kotlinx.coroutines.awaitAll
import kotlinx.coroutines.coroutineScope
import javax.inject.Inject
import javax.inject.Singleton
import kotlin.coroutines.coroutineContext
import kotlinx.coroutines.ensureActive

/**
 * Maximum length of a tool result that gets appended to the
 * conversation history. Tool results that exceed this budget are
 * truncated and marked so the model can ask for the rest if it
 * needs more detail. Without this, a single web search or
 * deep-research tool could blow the context window — 8k tokens
 * at 4 chars/token = 32k chars per result.
 *
 * Per-tool truncation (Firecrawl 8k, DeepResearch 6k) is the
 * first line of defense; this is the safety net for everything
 * else.
 */
private const val MAX_TOOL_RESULT_CHARS = 4_000

private const val TOOL_RESULT_TRUNCATION_MARKER =
    "\n\n[...truncated, ask the assistant to retrieve the full result if needed]"

/**
 * Truncate a tool result string to fit the conversation-history budget.
 * Long results (deep research, web fetches, photo library) are
 * truncated at [MAX_TOOL_RESULT_CHARS] with a marker so the model
 * knows to ask for the rest if it needs more detail.
 */
internal fun truncateToolResult(raw: String): String =
    if (raw.length > MAX_TOOL_RESULT_CHARS) {
        raw.take(MAX_TOOL_RESULT_CHARS) + TOOL_RESULT_TRUNCATION_MARKER
    } else {
        raw
    }

/**
 * The memory-augmented agentic loop. Pre-pends relevant memories to the system
 * prompt before each model call, and auto-stores memorable user facts after.
 * Also extracts a knowledge graph from each assistant turn (best-effort).
 */
@Singleton
class MemoryAugmentedAgenticLoop @Inject constructor(
    private val brain: Brain,
    private val toolRegistry: ToolRegistry,
    private val toolExecutor: ToolExecutor,
    private val memoryStore: MemoryStore,
    private val kgExtractor: ConversationKgExtractor,
    private val userProfileStore: com.aura.profile.UserProfileStore,
    private val handRepository: com.aura.hands.HandRepository,
    private val providerRegistry: com.aura.providers.ProviderRegistry,
) {
    /**
     * Run the agentic loop, optionally overriding the base system prompt
     * with a [Specialist]'s system prompt.
     */
    fun run(
        conversation: Conversation,
        model: String,
        maxSteps: Int = 10,
        options: ChatOptions = ChatOptions(),
        recallLimit: Int = 5,
        specialist: Specialist? = null,
        /**
         * Session-level write flag. When false:
         *   - The user message is NOT auto-stored.
         *   - The assistant turn is NOT fed to the profile extractor.
         *   - The knowledge-graph extractor does NOT run.
         *   - WRITE_LOCAL tools (e.g. `remember`) are refused by ToolExecutor.
         * Read-only tools (recall, web_search, kg_query) and the agent's
         * general reasoning still run. Used by the incognito toggle in
         * ChatScreen so the user can have a private session without any
         * long-term writes.
         */
        memoryEnabled: Boolean = true,
    ): Flow<AgentEvent> = flow {
        val tools = specialist?.let { s ->
            val allowed = s.toolsAllowed
            if (allowed.isEmpty()) toolRegistry.definitions()
            else toolRegistry.definitions().filter { it.name in allowed }
        } ?: toolRegistry.definitions()
        var step = 0
        var finished = false
        var lastUserMessage = ""
        var currentConversation = conversation
        // Tracks the most recent recall across all steps. The
        // agentic loop can run multiple model steps per turn (e.g.
        // one with tools, one without); we capture the recall
        // from the last step that actually performed recall so the
        // chip shows the most relevant memories for the final
        // assistant text. Null when no step recalled anything
        // (or memoryEnabled=false in incognito mode).
        var lastRecall: com.aura.agent.RecallSummary? = null

        while (!finished && step < maxSteps) {
            step += 1
            coroutineContext.ensureActive()

            // 1) Recall relevant memories for the last user message
            lastUserMessage = currentConversation.turns.lastOrNull { it.user != null }?.user ?: ""

            // 1b) Trigger-phrase hand check: if an enabled hand's phrase is a
            // substring of the user's message, prepend a run_hand tool call
            // so the model executes it. This is the wiring that makes a
            // hand's advertised trigger phrase actually fire.
            val handTrigger = findMatchingHand(lastUserMessage)
            val handContext = handTrigger?.let { hand ->
                "\n\n# Triggered automation:\nThe user triggered hand '${hand.name}'. " +
                    "Run it with run_hand(name=\"${hand.name}\")."
            } ?: ""

            // Single recall pass per step. Earlier revisions of this loop
            // called memoryStore.query() twice — once to capture the IDs
            // for the recall chip, and again to build the memoryContext
            // injected into the system prompt. Each call is an embedder
            // hit (cloud round-trip) plus an RRF rank. We now compute the
            // hits once and reuse the list for both the chip and the
            // context block. The semantic dedup check in MemoryStore is
            // still applied at write time; here we only read.
            //
            // Skip in incognito mode — we don't want to surface "Aura
            // used N memories" when the user opted out of memory
            // entirely. The MemoryStore call itself is read-only so this
            // gate is just about the chip and the persisted Turn.recall
            // field.
            val recallHits: List<com.aura.memory.MemoryEntity> =
                if (memoryEnabled && lastUserMessage.isNotBlank()) {
                    memoryStore.query(lastUserMessage, recallLimit)
                } else emptyList()
            val stepHandIds: List<String> = handTrigger?.let { listOf(it.id) } ?: emptyList()
            if (memoryEnabled) {
                // Only persist a non-null recall when memory is on
                // for this turn. The chip is hidden otherwise.
                lastRecall = com.aura.agent.RecallSummary(
                    memoryIds = recallHits.map { it.id },
                    handIds = stepHandIds,
                    noResults = recallHits.isEmpty() && stepHandIds.isEmpty(),
                )
            }

            val memoryContext = if (recallHits.isNotEmpty()) {
                val lines = recallHits.map { m ->
                    "- [${m.category}] ${m.content}"
                }.joinToString("\n")
                "\n\n# Relevant memories:\n$lines"
            } else ""

            // 2) Build messages
            val messages = buildList {
                val sys = listOfNotNull(
                    specialist?.systemPrompt,
                    currentConversation.systemPrompt,
                    brain.resolvedIdentity().ifBlank { null },
                    userProfileStore.getSystemPrompt().ifBlank { null },
                ).joinToString("\n\n") + memoryContext + handContext
                if (sys.isNotBlank()) add(ProviderMessage(role = Role.system, content = sys))
                addAll(currentConversation.toMessages())
            }

            // 3) Stream the model step
            val toolCalls = mutableListOf<Pair<String, String>>()
            val toolCallStarts = mutableMapOf<String, String>()
            val toolCallArgs = mutableMapOf<String, StringBuilder>()
            val accumulatedText = StringBuilder()
            var finishReason: String? = null
            var stepError: String? = null

            // Provider failover: if the primary model returns a retryable
            // error (5xx, 429, network timeout), try the next configured
            // provider before giving up. Don't failover on 401 (bad key)
            // or 400 (bad request) — retrying won't help.
            var currentModel = model
            val triedModels = mutableSetOf<String>()

            stream@ while (true) {
                triedModels.add(currentModel)
                stepError = null
                accumulatedText.clear()
                toolCalls.clear()
                toolCallStarts.clear()
                toolCallArgs.clear()
                finishReason = null

                try {
                    brain.stream(currentModel, messages, tools, options).collect { chunk ->
                        when (chunk) {
                            is BrainChunk.Text -> {
                                accumulatedText.append(chunk.text)
                                emit(AgentEvent.TextDelta(chunk.text))
                            }
                            is BrainChunk.ToolCallStart -> {
                                toolCallStarts[chunk.id] = chunk.name
                                emit(AgentEvent.ToolCallStart(chunk.id, chunk.name))
                            }
                            is BrainChunk.ToolCallDelta -> {
                                val id = chunk.id.ifEmpty { toolCallStarts.keys.lastOrNull() ?: "" }
                                toolCallArgs.getOrPut(id) { StringBuilder() }.append(chunk.argumentsDelta)
                            }
                            is BrainChunk.ToolCallEnd -> {
                                toolCalls += chunk.id to chunk.arguments
                                emit(AgentEvent.ToolCallEnd(chunk.id, chunk.name, chunk.arguments))
                            }
                            is BrainChunk.Finished -> { finishReason = chunk.reason }
                            is BrainChunk.Error -> {
                                stepError = "${chunk.code}: ${chunk.message}"
                                if (chunk.retryable && triedModels.size < 2) {
                                    val triedPrefixes = triedModels.mapTo(mutableSetOf()) {
                                        it.substringBefore(":")
                                    }
                                    val nextModel = providerRegistry.firstConfiguredModelId(triedPrefixes)
                                    if (nextModel != null) {
                                        emit(AgentEvent.Warning("Provider ${currentModel} failed (${chunk.code}), falling back to $nextModel"))
                                        currentModel = nextModel
                                        throw kotlinx.coroutines.CancellationException("failover")
                                    }
                                }
                                emit(AgentEvent.Error(chunk.code, chunk.message, chunk.retryable, chunk.error?.toAuraError()))
                            }
                        }
                    }
                    break@stream
                } catch (e: kotlinx.coroutines.CancellationException) {
                    if (e.message == "failover") continue@stream
                    throw e
                }
            }

            if (stepError != null) { finished = true; break }

            // Resolve in-progress tool calls
            for (id in toolCallStarts.keys) {
                if (toolCalls.none { it.first == id }) {
                    val name = toolCallStarts[id] ?: continue
                    val args = toolCallArgs[id]?.toString() ?: ""
                    toolCalls += id to args
                    emit(AgentEvent.ToolCallEnd(id, name, args))
                }
            }

            if (accumulatedText.isNotEmpty()) {
                currentConversation = currentConversation.addAssistant(accumulatedText.toString())
                // KG extraction is gated by memoryEnabled: in incognito mode
                // we must not learn entities / relations from this turn.
                // Extract from BOTH the user's message and the assistant's
                // response — the user shares facts ("I work at Google"),
                // the assistant synthesizes and confirms them.
                if (memoryEnabled) {
                    if (lastUserMessage.isNotBlank()) {
                        kgExtractor.extract(lastUserMessage)
                    }
                    kgExtractor.extract(accumulatedText.toString())
                }
            }
            for ((id, args) in toolCalls) {
                val name = toolCallStarts[id] ?: ""
                currentConversation = currentConversation.addToolCall(id, name, args)
            }

            if (toolCalls.isEmpty() || finishReason == "stop" || finishReason == "length") {
                finished = true
                break
            }

            // Execute tool calls in parallel. The model sends them in one
            // batch, meaning it doesn't expect one's output to feed into
            // the next. Running them concurrently cuts wall time for
            // multi-tool turns (e.g. two web searches in 1 turn = half
            // the latency). coroutineScope ensures all complete before
            // we process results and continue the loop.
            val ctx = ToolContext(
                conversationId = currentConversation.id,
                userMessage = lastUserMessage,
                memoryEnabled = memoryEnabled,
            )
            val toolResults = coroutineScope {
                toolCalls.map { (id, args) ->
                    val name = toolCallStarts[id] ?: return@map null
                    emit(AgentEvent.ToolExecuting(id, name, args))
                    async {
                        Triple(id, name to args, toolExecutor.execute(name, args, ctx))
                    }
                }.filterNotNull().awaitAll()
            }
            for ((id, nameAndArgs, result) in toolResults) {
                val (name, args) = nameAndArgs
                val rawResultText = when (result) {
                    is ToolResult.Ok -> result.output
                    is ToolResult.Error -> "Error: ${result.message}"
                    is ToolResult.NeedsPermission -> "Permission needed: ${result.permission} — ${result.rationale}"
                    is ToolResult.NeedsApproval -> "Approval needed: ${result.rationale}"
                }
                val resultText = truncateToolResult(rawResultText)
                val needsPerm = if (result is ToolResult.NeedsPermission) result.permission else null
                val permRationale = if (result is ToolResult.NeedsPermission) result.rationale else null
                currentConversation = currentConversation.setToolResult(id, resultText)
                emit(AgentEvent.ToolResult(id, name, args, resultText, needsPerm, permRationale))
            }
        }

        // 4) Auto-store the user's last message via LLM write gate (best-effort, non-blocking)
        //    Skipped when memoryEnabled is false (incognito mode).
        //    The LLM gate wraps the heuristic gate: heuristic runs first
        //    (fast, no network), and if it says "store", the LLM makes the
        //    final decision with better category + importance. If the LLM
        //    call fails, the heuristic decision is used as fallback.
        if (memoryEnabled && lastUserMessage.isNotBlank()) {
            runCatching {
                // Use the user's default model for the gate. If the default
                // is MoA (expensive — 3 API calls for a yes/no), fall back to
                // the first configured non-MoA provider's first model.
                // The heuristic gate is the pre-filter; the LLM gate only
                // runs when the heuristic says "store", so this is one
                // lightweight call per memorable turn — not per message.
                val gateModel = if (model.startsWith("moa:")) {
                    providerRegistry.firstConfiguredModelId(setOf("moa")) ?: model
                } else {
                    model
                }
                val gate = LlmWriteGate(
                    heuristic = WriteGate(),
                    registry = providerRegistry,
                    modelId = gateModel,
                )
                val decision = gate.evaluate(lastUserMessage, "user")
                if (decision.shouldStore) {
                    memoryStore.store(
                        content = lastUserMessage,
                        source = "user",
                        category = decision.category,
                        importance = decision.importance,
                    )
                }
            }
        }

        // 5) Extract user profile from the conversation.
        //    First-person patterns ("my name is", "I live in", "I prefer")
        //    are in the USER's text, not the assistant's response. Run
        //    extraction on the user message first, then on the assistant
        //    text as a secondary path (the assistant may echo user facts).
        if (memoryEnabled && lastUserMessage.isNotBlank()) {
            runCatching { extractProfileFromText(lastUserMessage) }
        }
        val lastAssistant = currentConversation.turns.lastOrNull()?.assistant
        if (memoryEnabled && !lastAssistant.isNullOrBlank()) {
            runCatching { extractProfileFromText(lastAssistant) }
        }

        if (!finished) {
            emit(AgentEvent.Error("max_steps_exceeded", "Hit max steps ($maxSteps) without finishing.", retryable = false))
        }
        // Persist the recall summary on the conversation's last
        // turn so the chat UI can render the chip on history
        // replays and so the loop stays the single source of
        // truth for what was recalled.
        val finalConv = if (lastRecall != null) {
            currentConversation.attachRecallToLastTurn(lastRecall)
        } else currentConversation
        emit(AgentEvent.Result(finalConv, lastRecall))
        emit(AgentEvent.Done)
    }

    /** Lightweight regex-based profile extraction. Updates name, traits, and facts. */
    private suspend fun extractProfileFromText(text: String) {
        val facts = mutableListOf<String>()
        Regex("""(?:my name is|i'm|i am|call me)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)""", RegexOption.IGNORE_CASE)
            .find(text)?.groupValues?.getOrNull(1)?.let { name ->
                userProfileStore.update(name = name.trim())
            }
        Regex("""(?:i (?:live|am|work) (?:in|at|from))\s+([A-Z][a-zA-Z\s,]+?)(?:\.|,|\s+(?:and|but|so|because)|\s*$)""", RegexOption.IGNORE_CASE)
            .find(text)?.groupValues?.getOrNull(1)?.let { loc -> facts.add("Lives in ${loc.trim()}") }
        Regex("""(?:i(?:'m| am) an?\s+|i work as (?:an?\s+)?)([a-z]+(?:\s+[a-z]+){0,3})""", RegexOption.IGNORE_CASE)
            .find(text)?.groupValues?.getOrNull(1)?.let { job -> facts.add("Works as ${job.trim()}") }
        Regex("""(?:i (?:prefer|like|love))\s+(.+?)(?:\.|,|\s+(?:and|but|so|because)|\s*$)""", RegexOption.IGNORE_CASE)
            .find(text)?.groupValues?.getOrNull(1)?.let { pref -> facts.add("Prefers ${pref.trim()}") }
        if (facts.isNotEmpty()) userProfileStore.mergeFacts(facts)
    }

    /**
     * Find the first enabled hand whose trigger phrase is contained in the
     * user's message. Substring match; case-insensitive; blank phrases are
     * ignored. If multiple hands match, the newest one wins (getEnabled
     * orders by createdAt DESC).
     */
    private suspend fun findMatchingHand(userMessage: String): com.aura.hands.Hand? {
        if (userMessage.isBlank()) return null
        val lower = userMessage.lowercase()
        return handRepository.getEnabled().firstOrNull { hand ->
            hand.triggerPhrase.isNotBlank() &&
                lower.contains(hand.triggerPhrase.lowercase())
        }
    }
}

sealed class AgentEvent {
    data class TextDelta(val text: String) : AgentEvent()
    data class ToolCallStart(val id: String, val name: String) : AgentEvent()
    data class ToolCallEnd(val id: String, val name: String, val arguments: String) : AgentEvent()
    data class ToolExecuting(val id: String, val name: String, val arguments: String = "") : AgentEvent()
    data class ToolResult(val id: String, val name: String, val arguments: String, val result: String, val needsPermission: String? = null, val permissionRationale: String? = null) : AgentEvent()
    /** Emitted by the UI after a permission was granted. The loop re-executes the named tool with the given args. */
    data class PermissionGranted(val toolName: String, val arguments: String) : AgentEvent()
    data class Error(val code: String, val message: String, val retryable: Boolean, val typedError: com.aura.core.error.AuraError? = null) : AgentEvent()
    data class Warning(val message: String) : AgentEvent()
    data class Result(
        val conversation: com.aura.agent.Conversation,
        /**
         * The recall summary for the turn that just completed.
         * Null when the loop skipped recall (incognito mode).
         * Stored on the conversation's last turn by the UI layer
         * so the chip renders on history replays.
         */
        val recall: com.aura.agent.RecallSummary? = null,
    ) : AgentEvent()
    data object Done : AgentEvent()
}