package com.aura.providers

import kotlinx.serialization.Serializable

@Serializable
data class ProviderChunk(
    val text: String? = null,
    val toolCall: ToolCall? = null,
    val finishReason: FinishReason? = null,
    val usage: Usage? = null,
    val error: ProviderError? = null,
) {
    val isDone: Boolean get() = finishReason != null || error != null
}

@Serializable
enum class FinishReason { stop, length, tool_calls, error, cancelled }

@Serializable
data class Usage(
    val promptTokens: Int = 0,
    val completionTokens: Int = 0,
    val totalTokens: Int = 0,
)

@Serializable
data class ProviderError(
    val code: String,
    val message: String,
    val retryable: Boolean = false,
    val cause: String? = null,
)
