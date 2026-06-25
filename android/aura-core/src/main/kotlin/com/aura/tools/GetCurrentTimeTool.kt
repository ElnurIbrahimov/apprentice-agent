package com.aura.tools

import com.aura.agent.Tool
import com.aura.agent.ToolContext
import com.aura.agent.ToolResult
import com.aura.agent.ToolRisk
import com.aura.providers.ToolDefinition
import com.aura.providers.ToolParameters
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import javax.inject.Inject
import javax.inject.Singleton

/**
 * Get the current date/time. Trivial but the model uses it to compute
 * "in 30 minutes", "tomorrow at 3pm", etc.
 */
@Singleton
class GetCurrentTimeTool @Inject constructor() {
    fun definition() = ToolDefinition(
        name = "get_current_time",
        description = "Get the current local date and time in human-readable form.",
        parameters = ToolParameters(properties = emptyMap(), required = emptyList()),
    )

    val tool = Tool(
        name = "get_current_time",
        description = definition().description,
        risk = ToolRisk.READ_ONLY,
        parameters = definition().parameters,
        execute = { call, ctx ->
            val now = System.currentTimeMillis()
            val human = SimpleDateFormat("EEE MMM d yyyy, HH:mm:ss", Locale.US).format(Date(now))
            val iso = SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss", Locale.US).format(Date(now))
            ToolResult.Ok("Now: $human (ISO $iso, epoch $now)")
        },
    )
}
