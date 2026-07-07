package com.aura.tools

import android.content.Context
import android.content.Intent
import android.net.Uri
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
 * Opens the email app with a pre-filled compose intent (mailto:).
 * The user must tap Send in the email app — this tool does NOT send directly.
 *
 * Risk: WRITE_REMOTE (triggers network egress when the user sends).
 */
@Singleton
class EmailSendTool @Inject constructor(
    @ApplicationContext private val context: Context,
) {
    fun definition() = ToolDefinition(
        name = "email_send",
        description = "Open the email app to compose a new email. The user must tap Send — this tool does not send directly.",
        parameters = ToolParameters(
            properties = mapOf(
                "to" to ToolProperty(
                    type = "string",
                    description = "Recipient email address (e.g. user@example.com)",
                ),
                "subject" to ToolProperty(
                    type = "string",
                    description = "Email subject line",
                ),
                "body" to ToolProperty(
                    type = "string",
                    description = "Email body content",
                ),
            ),
            required = listOf("to"),
        ),
    )

    val tool = Tool(
        name = "email_send",
        description = definition().description,
        risk = ToolRisk.WRITE_REMOTE,
        parameters = definition().parameters,
        execute = { call, _ ->
            val to = call.arguments["to"] as? String
                ?: return@Tool ToolResult.Error("missing 'to' argument", "bad_args")
            val subject = call.arguments["subject"] as? String ?: ""
            val body = call.arguments["body"] as? String ?: ""

            try {
                val uri = Uri.parse("mailto:$to")
                val intent = Intent(Intent.ACTION_SENDTO, uri).apply {
                    putExtra(Intent.EXTRA_SUBJECT, subject)
                    putExtra(Intent.EXTRA_TEXT, body)
                    flags = Intent.FLAG_ACTIVITY_NEW_TASK
                }

                if (intent.resolveActivity(context.packageManager) == null) {
                    return@Tool ToolResult.Error(
                        "No email app found on the device",
                        "no_activity",
                    )
                }

                context.startActivity(intent)
                ToolResult.Ok("Email app opened for user confirmation.")
            } catch (e: Exception) {
                ToolResult.Error("Failed to open email app: ${e.message}", "email_error")
            }
        },
    )
}
