package com.aura.agent

import com.aura.kg.ConversationKgExtractor
import com.aura.memory.MemoryStore
import com.aura.providers.ChatOptions
import com.aura.providers.ProviderMessage
import com.aura.providers.ProviderMessage.Role
import com.aura.providers.ToolDefinition
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import javax.inject.Inject
import javax.inject.Singleton
import kotlin.coroutines.coroutineContext
import kotlinx.coroutines.currentCoroutineContext
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
    ): Flow<AgentEvent> = flow {
        val tools = toolRegistry.definitions()
        var step = 0
        var finished = false
        var lastUserMessage = ""
        var currentConversation = conversation

        while (!finished && step < maxSteps) {
            step += 1
            coroutineContext.ensureActive()

            // 1) Recall relevant memories for the last user message
            lastUserMessage = currentConversation.turns.lastOrNull { it.user != null }?.user ?: ""
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
                    brain.identity.ifBlank { null },
                ).joinToString("\n\n") + memoryContext
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

            brain.stream(model, messages, tools, options).collect { chunk ->
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
                        emit(AgentEvent.Error(chunk.code, chunk.message, chunk.retryable))
                    }
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
                kgExtractor.extract(accumulatedText.toString())
            }
            for ((id, args) in toolCalls) {
                val name = toolCallStarts[id] ?: ""
                currentConversation = currentConversation.addToolCall(id, name, args)
            }

            if (toolCalls.isEmpty() || finishReason == "stop" || finishReason == "length") {
                finished = true
                break
            }

            for ((id, args) in toolCalls) {
                val name = toolCallStarts[id] ?: continue
                emit(AgentEvent.ToolExecuting(id, name))
                val result = toolExecutor.execute(name, args, ToolContext(conversationId = currentConversation.id))
                val resultText = when (result) {
                    is ToolResult.Ok -> result.output
                    is ToolResult.Error -> "Error: ${result.message}"
                    is ToolResult.NeedsPermission -> "Permission needed: ${result.permission} — ${result.rationale}"
                    is ToolResult.NeedsApproval -> "Approval needed: ${result.rationale}"
                }
                val needsPerm = if (result is ToolResult.NeedsPermission) result.permission else null
                val permRationale = if (result is ToolResult.NeedsPermission) result.rationale else null
                currentConversation = currentConversation.setToolResult(id, resultText)
                emit(AgentEvent.ToolResult(id, name, resultText, needsPerm, permRationale))
            }
        }

        // 4) Auto-store the user's last message via WriteGate (best-effort, non-blocking)
        if (lastUserMessage.isNotBlank()) {
            runCatching { memoryStore.maybeStore(lastUserMessage, source = "user") }
        }

        if (!finished) {
            emit(AgentEvent.Error("max_steps_exceeded", "Hit max steps ($maxSteps) without finishing.", retryable = false))
        }
        emit(AgentEvent.Result(currentConversation))
        emit(AgentEvent.Done)
    }
}

sealed class AgentEvent {
    data class TextDelta(val text: String) : AgentEvent()
    data class ToolCallStart(val id: String, val name: String) : AgentEvent()
    data class ToolCallEnd(val id: String, val name: String, val arguments: String) : AgentEvent()
    data class ToolExecuting(val id: String, val name: String) : AgentEvent()
    data class ToolResult(val id: String, val name: String, val result: String, val needsPermission: String? = null, val permissionRationale: String? = null) : AgentEvent()
    data class Error(val code: String, val message: String, val retryable: Boolean) : AgentEvent()
    data class Result(val conversation: com.aura.agent.Conversation) : AgentEvent()
    data object Done : AgentEvent()
}
