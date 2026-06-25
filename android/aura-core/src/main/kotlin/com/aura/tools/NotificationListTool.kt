package com.aura.tools

import android.content.Context
import android.service.notification.NotificationListenerService
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
 * List currently-visible notifications on the device. v1: reads from the
 * device's notification listener service which the user must opt into.
 * v1.5: integrates with the AccessibilityService for richer state.
 * Risk: PRIVACY (BIND_NOTIFICATION_LISTENER_SERVICE).
 */
@Singleton
class NotificationListTool @Inject constructor(
    @ApplicationContext private val context: Context,
) {
    fun definition() = ToolDefinition(
        name = "notification_list",
        description = "List currently-visible notifications. Returns 'pkg: title' for each. Limited by Android — only notifications the listener service has captured since boot.",
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
        requiredPermissions = listOf("android.permission.BIND_NOTIFICATION_LISTENER_SERVICE"),
        parameters = definition().parameters,
        execute = { call, ctx ->
            val limit = (call.arguments["limit"] as? Int ?: 20).coerceIn(1, 50)
            val mgr = context.getSystemService(Context.NOTIFICATION_SERVICE) as? android.app.NotificationManager
                ?: return@Tool ToolResult.Error("no NotificationManager", "system_error")
            val sb = StringBuilder()
            var count = 0
            try {
                val active = mgr.activeNotifications
                for (sbn in active) {
                    if (count >= limit) break
                    val pkg = sbn.packageName
                    val extras = sbn.notification.extras
                    val title = extras.getCharSequence(android.app.Notification.EXTRA_TITLE)?.toString() ?: "(no title)"
                    val text = extras.getCharSequence(android.app.Notification.EXTRA_TEXT)?.toString() ?: ""
                    sb.appendLine("${count + 1}. $pkg: $title${if (text.isNotEmpty()) " — $text" else ""}")
                    count++
                }
            } catch (e: Exception) {
                return@Tool ToolResult.Error("notification_list failed: ${e.message}", "exception")
            }
            if (count == 0) ToolResult.Ok("No active notifications (or notification listener not granted).")
            else ToolResult.Ok(sb.toString())
        },
    )
}
