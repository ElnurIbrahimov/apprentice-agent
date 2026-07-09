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
 * Opens the SMS app with a pre-filled compose intent (smsto:).
 * The user must tap Send in the SMS app — this tool does NOT send directly.
 *
 * Risk: WRITE_LOCAL (opens an Activity on the device; no network
 * egress from Aura itself).
 */
@Singleton
class SmsSendTool @Inject constructor(
    @ApplicationContext private val context: Context,
) {
    fun definition() = ToolDefinition(
        name = "sms_send",
        description = "Open the SMS app to compose a new text message. The user must tap Send — this tool does not send directly.",
        parameters = ToolParameters(
            properties = mapOf(
                "to" to ToolProperty(
                    type = "string",
                    description = "Recipient phone number (e.g. +1234567890)",
                ),
                "body" to ToolProperty(
                    type = "string",
                    description = "Message body content",
                ),
            ),
            required = listOf("to"),
        ),
    )

    val tool = Tool(
        name = "sms_send",
        description = definition().description,
        risk = ToolRisk.WRITE_LOCAL,
        parameters = definition().parameters,
        execute = { call, _ ->
            val to = call.arguments["to"] as? String
                ?: return@Tool ToolResult.Error("missing 'to' argument", "bad_args")
            val body = call.arguments["body"] as? String ?: ""

            try {
                val uri = Uri.parse("smsto:$to")
                val intent = Intent(Intent.ACTION_SENDTO, uri).apply {
                    putExtra(Intent.EXTRA_TEXT, body)
                    flags = Intent.FLAG_ACTIVITY_NEW_TASK
                }

                if (intent.resolveActivity(context.packageManager) == null) {
                    return@Tool ToolResult.Error(
                        "No SMS app found on the device",
                        "no_activity",
                    )
                }

                context.startActivity(intent)
                ToolResult.Ok("SMS app opened for user confirmation.")
            } catch (e: Exception) {
                ToolResult.Error("Failed to open SMS app: ${e.message}", "sms_error")
            }
        },
    category = "communication")
}
