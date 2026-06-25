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
 * Opens the system file picker (Storage Access Framework) so the model can
 * access user-selected documents. The picked file URI is broadcast via
 * the app's LocalBroadcastManager (v1) or ActivityResult (v1.5) so the
 * chat loop can ingest its contents.
 *
 * Risk: WRITE_LOCAL (just opens a picker).
 */
@Singleton
class FilePickerTool @Inject constructor(
    @ApplicationContext private val context: Context,
) {
    fun definition() = ToolDefinition(
        name = "file_pick",
        description = "Open the system file picker. Returns the URI of the picked file. The next message can reference it via 'the file I just picked'.",
        parameters = ToolParameters(
            properties = mapOf(
                "mime_type" to ToolProperty(type = "string", description = "Optional MIME filter, e.g. 'application/pdf' or 'text/plain'. Default: any file."),
            ),
            required = emptyList(),
        ),
    )

    val tool = Tool(
        name = "file_pick",
        description = definition().description,
        risk = ToolRisk.WRITE_LOCAL,
        parameters = definition().parameters,
        execute = { call, ctx ->
            val mime = call.arguments["mime_type"] as? String ?: "*/*"
            try {
                val intent = Intent(Intent.ACTION_OPEN_DOCUMENT).apply {
                    addCategory(Intent.CATEGORY_OPENABLE)
                    type = mime
                    addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
                }
                context.startActivity(intent)
                ToolResult.Ok("File picker opened. Pick a file, then continue the conversation.")
            } catch (e: Exception) {
                ToolResult.Error("file_pick failed: ${e.message}", "exception")
            }
        },
    )
}
