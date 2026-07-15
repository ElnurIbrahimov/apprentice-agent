package com.aura.tools

import android.content.Context
import android.content.ClipData
import android.content.ClipDescription
import android.content.ClipboardManager
import com.aura.agent.Tool
import com.aura.agent.ToolResult
import com.aura.agent.ToolRisk
import com.aura.providers.ToolDefinition
import com.aura.providers.ToolParameters
import com.aura.providers.ToolProperty
import dagger.hilt.android.qualifiers.ApplicationContext
import javax.inject.Inject
import javax.inject.Singleton

/**
 * Write plain text to the system clipboard.
 * Risk: WRITE_LOCAL.
 */
@Singleton
class ClipboardWriteTool @Inject constructor(
    @ApplicationContext private val context: Context,
) {
    fun definition() = ToolDefinition(
        name = "clipboard_write",
        description = "Write text to the system clipboard.",
        parameters = ToolParameters(
            properties = mapOf(
                "text" to ToolProperty(type = "string", description = "Text to copy"),
                "label" to ToolProperty(type = "string", description = "Optional clipboard label (default: Aura)"),
            ),
            required = listOf("text"),
        ),
    )

    val tool = Tool(
        name = "clipboard_write",
        description = definition().description,
        risk = ToolRisk.WRITE_LOCAL,
        parameters = definition().parameters,
        execute = { call, _ ->
            val text = call.arguments["text"] as? String
                ?: return@Tool ToolResult.Error("missing 'text'", "bad_args")
            val label = call.arguments["label"] as? String ?: "Aura"
            val manager = context.getSystemService(Context.CLIPBOARD_SERVICE) as? ClipboardManager
                ?: return@Tool ToolResult.Error("ClipboardManager unavailable", "system_error")
            manager.setPrimaryClip(ClipData.newPlainText(label, text))
            ToolResult.Ok("Copied to clipboard ($label): ${text.take(80)}")
        },
        category = "device",
    )
}
