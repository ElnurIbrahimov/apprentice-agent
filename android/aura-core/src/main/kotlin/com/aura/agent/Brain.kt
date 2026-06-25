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
        providerRegistry.chat(model, messages, options, tools).collect { providerChunk ->
            emit(BrainChunk.fromProvider(providerChunk))
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
        fun fromProvider(p: com.aura.providers.ProviderChunk) = when {
            p.error != null -> Error(p.error.code, p.error.message, p.error.retryable)
            p.finishReason != null -> Finished(p.finishReason.name)
            p.toolCall != null -> ToolCallDelta(p.toolCall.id, p.toolCall.arguments)
            p.text != null -> Text(p.text)
            else -> Text("")
        }
    }
}
