package com.aura.tools

import android.content.Context
import android.content.ClipData
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
 * Read the system clipboard as plain text. Mirrors the obvious missing
 * capability every phone assistant needs.
 * Risk: PRIVACY (reads clipboard).
 */
@Singleton
class ClipboardReadTool @Inject constructor(
    @ApplicationContext private val context: Context,
) {
    fun definition() = ToolDefinition(
        name = "clipboard_read",
        description = "Read the current text from the system clipboard.",
        parameters = ToolParameters(properties = emptyMap(), required = emptyList()),
    )

    val tool = Tool(
        name = "clipboard_read",
        description = definition().description,
        risk = ToolRisk.PRIVACY,
        parameters = definition().parameters,
        execute = { _, _ ->
            val manager = context.getSystemService(Context.CLIPBOARD_SERVICE) as? ClipboardManager
                ?: return@Tool ToolResult.Error("ClipboardManager unavailable", "system_error")
            val text = manager.primaryClip?.takeIf { it.itemCount > 0 }
                ?.getItemAt(0)?.text?.toString()
            if (text == null) {
                ToolResult.Ok("Clipboard is empty or contains non-text content.")
            } else {
                ToolResult.Ok(text)
            }
        },
        category = "device",
    )
}
