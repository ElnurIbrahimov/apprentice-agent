package com.aura.providers

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable

@Serializable
data class ProviderMessage(
    val role: Role,
    val content: String,
    val name: String? = null,
    @SerialName("tool_calls") val toolCalls: List<ToolCall>? = null,
    @SerialName("tool_call_id") val toolCallId: String? = null,
) {
    @Serializable
    enum class Role { system, user, assistant, tool }
}
