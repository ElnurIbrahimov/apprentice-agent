package com.aura.tools

import android.content.Context
import android.content.Intent
import com.aura.agent.Tool
import com.aura.agent.ToolContext
import com.aura.agent.ToolResult
import com.aura.agent.ToolRisk
import com.aura.providers.ToolDefinition
import com.aura.providers.ToolParameters
import com.aura.providers.ToolProperty
import dagger.hilt.android.qualifiers.ApplicationContext
import javax.inject.Inject
import javax.inject.Singleton

/**
 * Send content to another app via Android's share sheet.
 * Mirrors aura/tools/share_intent.py (none — this is phone-native).
 * Risk: WRITE_LOCAL (just opens a chooser).
 */
@Singleton
class ShareIntentTool @Inject constructor(
    @ApplicationContext private val context: Context,
) {
    fun definition() = ToolDefinition(
        name = "share",
        description = "Open the system share sheet with the given text content.",
        parameters = ToolParameters(
            properties = mapOf(
                "text" to ToolProperty(type = "string", description = "Text to share"),
                "title" to ToolProperty(type = "string", description = "Optional share-sheet title"),
            ),
            required = listOf("text"),
        ),
    )

    val tool = Tool(
        name = "share",
        description = definition().description,
        risk = ToolRisk.WRITE_LOCAL,
        parameters = definition().parameters,
        execute = { call, ctx ->
            val text = call.arguments["text"] as? String ?: return@Tool ToolResult.Error("missing 'text'", "bad_args")
            val title = call.arguments["title"] as? String ?: "Share via Aura"
            try {
                val send = Intent(Intent.ACTION_SEND).apply {
                    type = "text/plain"
                    putExtra(Intent.EXTRA_TEXT, text)
                    putExtra(Intent.EXTRA_TITLE, title)
                    flags = Intent.FLAG_ACTIVITY_NEW_TASK
                }
                val chooser = Intent.createChooser(send, title).apply {
                    flags = Intent.FLAG_ACTIVITY_NEW_TASK
                }
                context.startActivity(chooser)
                ToolResult.Ok("Share sheet opened.")
            } catch (e: Exception) {
                ToolResult.Error("share failed: ${e.message}", "exception")
            }
        },
    )
}
