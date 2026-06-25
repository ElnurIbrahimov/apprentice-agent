package com.aura.providers

import kotlinx.coroutines.flow.Flow

/**
 * Base interface every LLM provider implements.
 * Mirrors aura/providers/base.py BaseProvider.
 */
interface Provider {
    val prefix: String
    val displayName: String
    fun isConfigured(): Boolean

    /**
     * Send a chat request.
     * @return a Flow of ProviderChunks. The terminal chunk has done=true and may include usage.
     */
    fun chat(
        model: String,
        messages: List<ProviderMessage>,
        options: ChatOptions = ChatOptions(),
        tools: List<ToolDefinition> = emptyList(),
    ): Flow<ProviderChunk>

    /**
     * List models this provider exposes. May be a hardcoded list or fetched from /models.
     */
    suspend fun listModels(): List<String>

    /**
     * Cancel an in-flight request. Implementations should propagate to the underlying HTTP/SDK call.
     */
    suspend fun cancel()
}
