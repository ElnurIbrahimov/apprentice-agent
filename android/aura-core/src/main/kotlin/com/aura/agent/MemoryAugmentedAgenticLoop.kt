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

            val memoryContext = if (lastUserMessage.isNotBlank()) {
                val hits = memoryStore.query(lastUserMessage, recallLimit)
                if (hits.isNotEmpty()) {
                    val lines = hits.mapIndexed { i, m ->
                        "- [${m.category}] ${m.content}"
                    }.joinToString("\n")
                    "\n\n# Relevant memories:\n$lines"
                } else ""
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
                                    val nextProvider = providerRegistry.configured()
                                        .firstOrNull { p ->
                                            triedModels.none { it.startsWith("${p.prefix}:") }
                                        }
                                    if (nextProvider != null) {
                                        val nextModel = "${nextProvider.prefix}:${
                                            runCatching { nextProvider.listModels().firstOrNull() }.getOrNull() ?: "default"
                                        }"
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
                val resultText = when (result) {
                    is ToolResult.Ok -> result.output
                    is ToolResult.Error -> "Error: ${result.message}"
                    is ToolResult.NeedsPermission -> "Permission needed: ${result.permission} — ${result.rationale}"
                    is ToolResult.NeedsApproval -> "Approval needed: ${result.rationale}"
                }
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
                    providerRegistry.configured().firstOrNull()?.let { p ->
                        "${p.prefix}:${p.listModels().firstOrNull() ?: "default"}"
                    } ?: model
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
        emit(AgentEvent.Result(currentConversation))
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
        if (facts.isNotEmpty()) userProfileStore.update(facts = facts)
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
    data class Result(val conversation: com.aura.agent.Conversation) : AgentEvent()
    data object Done : AgentEvent()
}