package com.aura.providers

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable

@Serializable
data class ToolCall(
    val id: String,
    val name: String,
    val arguments: String, // raw JSON string; parsed lazily
)

@Serializable
data class ToolDefinition(
    val name: String,
    val description: String,
    val parameters: ToolParameters,
)

@Serializable
data class ToolParameters(
    val type: String = "object",
    val properties: Map<String, ToolProperty> = emptyMap(),
    val required: List<String> = emptyList(),
)

@Serializable
data class ToolProperty(
    val type: String,
    val description: String? = null,
    val enum: List<String> = emptyList(),
    @SerialName("default") val defaultValue: kotlinx.serialization.json.JsonElement? = null,
)
