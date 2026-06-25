package com.aura.tools

import android.app.NotificationManager
import android.content.Context
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
 * Get or set Do-Not-Disturb interruption filter. Mirrors aura/proactive/theory_of_mind.py (none — new).
 * Risk: WRITE_LOCAL (system settings). Note: the app needs to be a "Notification Policy Access"
 * app in Settings → Notifications → Special access for write access on Android 6+.
 */
@Singleton
class DndModeTool @Inject constructor(
    @ApplicationContext private val context: Context,
) {
    fun definition() = ToolDefinition(
        name = "dnd_mode",
        description = "Get or set the Do-Not-Disturb interruption filter. Levels: 'off' (all), 'priority' (priority only), 'alarms' (alarms only), 'silence' (total silence). Omit 'level' to get current state.",
        parameters = ToolParameters(
            properties = mapOf(
                "level" to ToolProperty(type = "string", description = "off|priority|alarms|silence (omit to query)"),
            ),
            required = emptyList(),
        ),
    )

    val tool = Tool(
        name = "dnd_mode",
        description = definition().description,
        risk = ToolRisk.WRITE_LOCAL,
        parameters = definition().parameters,
        execute = { call, ctx ->
            val mgr = context.getSystemService(Context.NOTIFICATION_SERVICE) as? NotificationManager
                ?: return@Tool ToolResult.Error("no NotificationManager", "system_error")
            val current = mgr.currentInterruptionFilter
            val currentName = when (current) {
                NotificationManager.INTERRUPTION_FILTER_UNKNOWN -> "unknown"
                NotificationManager.INTERRUPTION_FILTER_ALL -> "off"
                NotificationManager.INTERRUPTION_FILTER_PRIORITY -> "priority"
                NotificationManager.INTERRUPTION_FILTER_ALARMS -> "alarms"
                NotificationManager.INTERRUPTION_FILTER_NONE -> "silence"
                else -> "?"
            }
            val level = call.arguments["level"] as? String
            if (level == null) {
                return@Tool ToolResult.Ok("DND currently: $currentName")
            }
            val target = when (level.lowercase()) {
                "off" -> NotificationManager.INTERRUPTION_FILTER_ALL
                "priority" -> NotificationManager.INTERRUPTION_FILTER_PRIORITY
                "alarms" -> NotificationManager.INTERRUPTION_FILTER_ALARMS
                "silence" -> NotificationManager.INTERRUPTION_FILTER_NONE
                else -> return@Tool ToolResult.Error("unknown level: $level", "bad_args")
            }
            return@Tool try {
                mgr.setInterruptionFilter(target)
                ToolResult.Ok("DND set to: ${level.lowercase()}")
            } catch (e: SecurityException) {
                ToolResult.NeedsPermission("notification_policy_access", "Grant Notification Policy Access in Settings → Notifications → Special access → Aura.")
            } catch (e: Exception) {
                ToolResult.Error("dnd_mode failed: ${e.message}", "exception")
            }
        },
    )
}
