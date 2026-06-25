package com.aura.agent

import com.aura.providers.ChatOptions
import com.aura.providers.ProviderMessage
import com.aura.providers.ProviderRegistry
import com.aura.providers.ToolDefinition
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import javax.inject.Inject
import javax.inject.Singleton

/**
 * The brain. Wraps the provider registry, assembles system prompts,
 * owns the conversation context, hands the work to the agentic loop.
 * Mirrors aura/brain.py (the singleton + system prompt assembly part).
 */
@Singleton
class Brain @Inject constructor(
    private val providerRegistry: ProviderRegistry,
) {
    val identity: String = IDENTITY.trimIndent()

    companion object {
        val IDENTITY = """
            You are Aura, a personal AI assistant running natively on Android.
            You have a memory of past conversations, a tool system that can
            act on the phone (calendar, contacts, location, notifications,
            camera, voice, share, files, web search, deep research), and a
            multi-agent system for delegating complex tasks.
            Be concise. Be direct. Don't sugarcoat. When given a multi-step
            task, work through it without asking for confirmation between
            steps. When you don't know, say so. When the user says "commit"
            or "ship it", execute. When you use a tool, briefly explain why.
        """.trimIndent()
    }

    fun stream(
        model: String,
        messages: List<ProviderMessage>,
        tools: List<ToolDefinition> = emptyList(),
        options: ChatOptions = ChatOptions(),
    ): Flow<BrainChunk> = flow {
        // nameById accumulates tool-call ids to names across the stream so
        // providers that send argument deltas without re-sending the name
        // (e.g. Anthropic input_json_delta) can still be routed to the
        // correct tool. Reset per stream call.
        val nameById = mutableMapOf<String, String>()
        providerRegistry.chat(model, messages, options, tools).collect { providerChunk ->
            emit(BrainChunk.fromProvider(providerChunk, nameById))
        }
    }
}

sealed class BrainChunk {
    data class Text(val text: String) : BrainChunk()
    data class ToolCallStart(val id: String, val name: String) : BrainChunk()
    data class ToolCallDelta(val id: String, val argumentsDelta: String) : BrainChunk()
    data class ToolCallEnd(val id: String, val name: String, val arguments: String) : BrainChunk()
    data class Finished(val reason: String) : BrainChunk()
    data class Error(val code: String, val message: String, val retryable: Boolean) : BrainChunk()

    companion object {
        /**
         * Map a ProviderChunk into the higher-level BrainChunk stream the
         * agentic loop consumes. For tool calls we emit:
         *   - ToolCallStart the first time we see a given id with a name
         *   - ToolCallDelta for every subsequent argument chunk
         *   - ToolCallEnd once (typically on the finish_reason=tool_calls event,
         *     but providers are inconsistent so we also accept ProviderChunks
         *     that carry a full arguments string).
         *
         * The previous version collapsed all three into a single ToolCallDelta
         * and never emitted ToolCallStart, which broke the loop's tool-name
         * lookup for OpenAI/Anthropic (the loop reads the name from
         * `toolCallStarts[id]`).
         */
        fun fromProvider(p: com.aura.providers.ProviderChunk, nameById: MutableMap<String, String> = mutableMapOf()): BrainChunk {
            p.error?.let { return Error(it.code, it.message, it.retryable) }
            p.finishReason?.let { return Finished(it.name) }
            val tc = p.toolCall
            if (tc != null) {
                if (tc.id.isNotEmpty() && tc.name.isNotEmpty()) {
                    if (nameById.put(tc.id, tc.name) == null) {
                        return ToolCallStart(tc.id, tc.name)
                    }
                    if (tc.arguments.isNotEmpty()) {
                        return ToolCallEnd(tc.id, tc.name, tc.arguments)
                    }
                    return ToolCallDelta(tc.id, "")
                }
                // delta-style chunk (Anthropic input_json_delta): no id, no name.
                // Look up the most-recent id we saw and append to its args.
                val id = nameById.keys.lastOrNull() ?: return Text("")
                return ToolCallDelta(id, tc.arguments)
            }
            p.text?.let { return Text(it) }
            return Text("")
        }
    }
}
