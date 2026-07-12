package com.aura.tools

import com.aura.agent.Tool
import com.aura.agent.ToolResult
import com.aura.agent.ToolRisk
import com.aura.notifications.NotificationCaptureStore
import com.aura.providers.ToolDefinition
import com.aura.providers.ToolParameters
import com.aura.providers.ToolProperty
import javax.inject.Inject
import javax.inject.Singleton

/**
 * Reads the process-local snapshot populated by AuraNotificationListenerService.
 * Notification contents remain in memory and are never persisted by this tool.
 */
@Singleton
class NotificationListTool @Inject constructor(
    private val store: NotificationCaptureStore,
) {
    fun definition() = ToolDefinition(
        name = "notification_list",
        description = "List active device notifications captured through Aura's opt-in notification listener. Returns package, title, and preview text, newest first.",
        parameters = ToolParameters(
            properties = mapOf(
                "limit" to ToolProperty(type = "integer", description = "Max notifications to return (default 20, max 50)"),
            ),
            required = emptyList(),
        ),
    )

    val tool = Tool(
        name = "notification_list",
        description = definition().description,
        risk = ToolRisk.PRIVACY,
        parameters = definition().parameters,
        execute = { call, _ ->
            if (!store.connected.value) {
                return@Tool ToolResult.NeedsPermission(
                    permission = NOTIFICATION_LISTENER_PERMISSION,
                    rationale = "Enable Notification access for Aura in Android Settings, then retry.",
                )
            }
            val limit = (call.arguments["limit"] as? Int ?: 20).coerceIn(1, 50)
            val rows = store.snapshot(limit)
            if (rows.isEmpty()) {
                ToolResult.Ok("No active device notifications.")
            } else {
                ToolResult.Ok(
                    rows.mapIndexed { index, row ->
                        val title = row.title.cleanNotificationText().ifBlank { "(no title)" }
                        val text = row.text.cleanNotificationText()
                        "${index + 1}. ${row.packageName}: $title${if (text.isNotEmpty()) " — $text" else ""}"
                    }.joinToString("\n"),
                )
            }
        },
        category = "communication",
    )

    private fun String.cleanNotificationText(): String =
        replace(Regex("[\\r\\n\\t]+"), " ").trim().take(500)

    companion object {
        const val NOTIFICATION_LISTENER_PERMISSION = "android.permission.BIND_NOTIFICATION_LISTENER_SERVICE"
    }
}
