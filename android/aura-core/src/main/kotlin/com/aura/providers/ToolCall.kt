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
/**
 * Lightweight, public-safe description of a tool — name, what it
 * does, the arguments it accepts, and the UI category it belongs
 * to. Returned by [com.aura.agent.ToolRegistry.definitions] for
 * the LLM prompt and for the Tools browser screen. Never includes
 * the `execute` lambda or risk metadata — those are private to the
 * executor.
 */
data class ToolDefinition(
    val name: String,
    val description: String,
    val parameters: ToolParameters,
    /**
     * UI category used by the Tools browser screen. See
     * [com.aura.tools.ToolCategories] for the list of valid values.
     * Empty string = "other" (default if a tool hasn't been
     * categorized yet).
     */
    val category: String = "",
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
