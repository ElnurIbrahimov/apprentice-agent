package com.aura.tools

import android.content.Context
import androidx.work.Constraints
import androidx.work.Data
import androidx.work.ExistingWorkPolicy
import androidx.work.OneTimeWorkRequestBuilder
import androidx.work.WorkManager
import com.aura.agent.Tool
import com.aura.agent.ToolContext
import com.aura.agent.ToolResult
import com.aura.agent.ToolRisk
import com.aura.providers.ToolDefinition
import com.aura.providers.ToolParameters
import com.aura.providers.ToolProperty
import dagger.hilt.android.qualifiers.ApplicationContext
import java.text.SimpleDateFormat
import java.util.Locale
import java.util.concurrent.TimeUnit
import javax.inject.Inject
import javax.inject.Singleton

/**
 * Schedule a reminder via WorkManager. The agent calls this with a time
 * (ISO 8601 or "HH:mm" for today) and a message. WorkManager fires the
 * reminder worker at the scheduled time, which posts a notification.
 *
 * Mirrors aura/tools/task_scheduler.py + notifications.py.
 * Risk: WRITE_LOCAL (schedules a system job).
 */
@Singleton
class SetReminderTool @Inject constructor(
    @ApplicationContext private val context: Context,
) {
    fun definition() = ToolDefinition(
        name = "set_reminder",
        description = "Schedule a reminder. 'when' is HH:mm (24h, today or tomorrow) or an ISO 8601 datetime. Message is the reminder body.",
        parameters = ToolParameters(
            properties = mapOf(
                "when" to ToolProperty(type = "string", description = "Time to fire, e.g. '15:00' or '2026-06-26T15:00:00'"),
                "message" to ToolProperty(type = "string", description = "Reminder message"),
            ),
            required = listOf("when", "message"),
        ),
    )

    val tool = Tool(
        name = "set_reminder",
        description = definition().description,
        risk = ToolRisk.WRITE_LOCAL,
        parameters = definition().parameters,
        execute = { call, ctx ->
            val whenStr = call.arguments["when"] as? String ?: return@Tool ToolResult.Error("missing 'when'", "bad_args")
            val message = call.arguments["message"] as? String ?: return@Tool ToolResult.Error("missing 'message'", "bad_args")
            val triggerAt = parseTime(whenStr)
            if (triggerAt == null) {
                return@Tool ToolResult.Error("could not parse 'when': $whenStr (use HH:mm or ISO 8601)", "bad_args")
            }
            val delayMs = (triggerAt - System.currentTimeMillis()).coerceAtLeast(0L)
            val work = OneTimeWorkRequestBuilder<ReminderWorker>()
                .setInitialDelay(delayMs, TimeUnit.MILLISECONDS)
                .setInputData(
                    Data.Builder()
                        .putString("title", "⏰ Reminder")
                        .putString("body", message)
                        .build()
                )
                .setConstraints(Constraints.NONE)
                .addTag("reminder")
                .build()
            WorkManager.getInstance(context)
                .enqueueUniqueWork("reminder-${System.currentTimeMillis()}", ExistingWorkPolicy.REPLACE, work)
            val humanTime = SimpleDateFormat("EEE MMM d, HH:mm", Locale.US).format(triggerAt)
            ToolResult.Ok("Reminder set for $humanTime: $message")
        },
    )

    private fun parseTime(s: String): Long? {
        // Try ISO 8601 first
        return try {
            val iso = java.text.SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss", Locale.US).parse(s)
            iso?.time
        } catch (_: Exception) {
            // Try HH:mm (today or tomorrow)
            try {
                val parts = s.split(":")
                if (parts.size != 2) return null
                val hour = parts[0].toInt()
                val minute = parts[1].toInt()
                val cal = java.util.Calendar.getInstance()
                cal.set(java.util.Calendar.HOUR_OF_DAY, hour)
                cal.set(java.util.Calendar.MINUTE, minute)
                cal.set(java.util.Calendar.SECOND, 0)
                cal.set(java.util.Calendar.MILLISECOND, 0)
                if (cal.timeInMillis <= System.currentTimeMillis()) {
                    cal.add(java.util.Calendar.DAY_OF_YEAR, 1)
                }
                cal.timeInMillis
            } catch (_: Exception) {
                null
            }
        }
    }
}
