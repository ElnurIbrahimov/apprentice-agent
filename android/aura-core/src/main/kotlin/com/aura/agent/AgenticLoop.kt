package com.aura.agent

import com.aura.providers.ChatOptions
import com.aura.providers.ProviderMessage
import com.aura.providers.ProviderMessage.Role
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import javax.inject.Inject
import javax.inject.Singleton
import kotlin.coroutines.coroutineContext
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.ensureActive

/**
 * The core ReAct loop. Mirrors aura/core/agentic_loop.py + the 5 split files
 * in aura/core/agentic_loop_*.py. The first version is a single-file port
 * that handles the happy path + tool call dispatch + abort. Later iterations
 * split this into AgenticLoopEvents / AgenticLoopToolCalls / etc. as the
 * edge cases accumulate.
 */
@Singleton
class AgenticLoop @Inject constructor(
    private val brain: Brain,
    private val toolRegistry: ToolRegistry,
    private val toolExecutor: ToolExecutor,
) {
    /**
     * Run the agentic loop over a conversation. Streams events.
     */
    fun run(
        conversation: Conversation,
        model: String,
        maxSteps: Int = 10,
        options: ChatOptions = ChatOptions(),
    ): Flow<AgentEvent> = flow {
        val tools = toolRegistry.definitions()
        var step = 0
        var finished = false

        while (!finished && step < maxSteps) {
            step += 1
            coroutineContext.ensureActive()

            val messages = buildList {
                if (conversation.systemPrompt != null || brain.identity.isNotBlank()) {
                    val sys = listOfNotNull(conversation.systemPrompt, brain.identity.ifBlank { null })
                        .joinToString("\n\n")
                    add(ProviderMessage(role = Role.system, content = sys))
                }
                addAll(conversation.toMessages())
            }

            // Stream the model step
            val toolCalls = mutableListOf<Pair<String, String>>() // (id, full_args)
            val toolCallStarts = mutableMapOf<String, String>() // id -> name
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
                    is BrainChunk.Finished -> {
                        finishReason = chunk.reason
                    }
                    is BrainChunk.Error -> {
                        stepError = "${chunk.code}: ${chunk.message}"
                        emit(AgentEvent.Error(chunk.code, chunk.message, chunk.retryable))
                    }
                }
            }

            if (stepError != null) {
                finished = true
                break
            }

            // Resolve any in-progress tool calls that didn't get an explicit End
            for (id in toolCallStarts.keys) {
                if (toolCalls.none { it.first == id }) {
                    val name = toolCallStarts[id] ?: continue
                    val args = toolCallArgs[id]?.toString() ?: ""
                    toolCalls += id to args
                    emit(AgentEvent.ToolCallEnd(id, name, args))
                }
            }

            // Persist the assistant turn
            if (accumulatedText.isNotEmpty()) {
                conversation.addAssistant(accumulatedText.toString())
            }
            for ((id, args) in toolCalls) {
                val name = toolCallStarts[id] ?: ""
                conversation.addToolCall(id, name, args)
            }

            // Dispatch tool calls
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
