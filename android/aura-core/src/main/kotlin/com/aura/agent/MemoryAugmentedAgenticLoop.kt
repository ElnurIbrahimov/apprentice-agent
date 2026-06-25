package com.aura.agent

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
 * Mirrors how aura/core/agentic_loop.py interacts with aura/memory/.
 */
@Singleton
class MemoryAugmentedAgenticLoop @Inject constructor(
    private val brain: Brain,
    private val toolRegistry: ToolRegistry,
    private val toolExecutor: ToolExecutor,
    private val memoryStore: MemoryStore,
) {
    fun run(
        conversation: Conversation,
        model: String,
        maxSteps: Int = 10,
        options: ChatOptions = ChatOptions(),
        recallLimit: Int = 5,
    ): Flow<AgentEvent> = flow {
        val tools = toolRegistry.definitions()
        var step = 0
        var finished = false
        var lastUserMessage = ""

        while (!finished && step < maxSteps) {
            step += 1
            coroutineContext.ensureActive()

            // 1) Recall relevant memories for the last user message
            lastUserMessage = conversation.turns.lastOrNull { it.user != null }?.user ?: ""
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
                    conversation.systemPrompt,
                    brain.identity.ifBlank { null },
                ).joinToString("\n\n") + memoryContext
                if (sys.isNotBlank()) add(ProviderMessage(role = Role.system, content = sys))
                addAll(conversation.toMessages())
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

            if (accumulatedText.isNotEmpty()) conversation.addAssistant(accumulatedText.toString())
            for ((id, args) in toolCalls) {
                val name = toolCallStarts[id] ?: ""
                conversation.addToolCall(id, name, args)
            }

            if (toolCalls.isEmpty() || finishReason == "stop" || finishReason == "length") {
                finished = true
                break
            }

            for ((id, args) in toolCalls) {
                val name = toolCallStarts[id] ?: continue
                emit(AgentEvent.ToolExecuting(id, name))
                val result = toolExecutor.execute(name, args, ToolContext(conversationId = conversation.id))
                val resultText = when (result) {
                    is ToolResult.Ok -> result.output
                    is ToolResult.Error -> "Error: ${result.message}"
                    is ToolResult.NeedsPermission -> "Permission needed: ${result.permission} — ${result.rationale}"
                    is ToolResult.NeedsApproval -> "Approval needed: ${result.rationale}"
                }
                conversation.setToolResult(id, resultText)
                emit(AgentEvent.ToolResult(id, name, resultText))
            }
        }

        // 4) Auto-store the user's last message via WriteGate (best-effort, non-blocking)
        if (lastUserMessage.isNotBlank()) {
            runCatching { memoryStore.maybeStore(lastUserMessage, source = "user") }
        }

        if (!finished) {
            emit(AgentEvent.Error("max_steps_exceeded", "Hit max steps ($maxSteps) without finishing.", retryable = false))
        }
        emit(AgentEvent.Done)
    }
}

sealed class AgentEvent {
    data class TextDelta(val text: String) : AgentEvent()
    data class ToolCallStart(val id: String, val name: String) : AgentEvent()
    data class ToolCallEnd(val id: String, val name: String, val arguments: String) : AgentEvent()
    data class ToolExecuting(val id: String, val name: String) : AgentEvent()
    data class ToolResult(val id: String, val name: String, val result: String) : AgentEvent()
    data class Error(val code: String, val message: String, val retryable: Boolean) : AgentEvent()
    data object Done : AgentEvent()
}
