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
import com.aura.tasks.ReminderDao
import com.aura.tasks.ReminderEntity
import dagger.hilt.android.qualifiers.ApplicationContext
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicInteger
import javax.inject.Inject
import javax.inject.Singleton

/** Schedule a reminder via WorkManager. The agent calls this with a time
 * (ISO 8601 or "HH:mm" for today) and a message. WorkManager fires the
 * reminder worker at the scheduled time, which posts a notification.
 *
 * A companion row in the reminders table gives the UI a stable list of
 * upcoming reminders and lets the user cancel before they fire.
 *
 * Mirrors aura/tools/task_scheduler.py + notifications.py.
 * Risk: WRITE_LOCAL (schedules a system job).
 */
@Singleton
class SetReminderTool @Inject constructor(
    @ApplicationContext private val context: Context,
    private val reminderDao: ReminderDao,
) {
    /** Monotonically increasing notification IDs for reminders. Starts
     * above the small system-reserved range to avoid collisions with
     * morning-brief / proactive notifications. */
    private val nextNotificationId = AtomicInteger(NOTIFICATION_ID_START)

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
            val triggerAt = TimeParser.parse(whenStr)
            if (triggerAt == null) {
                return@Tool ToolResult.Error("could not parse 'when': $whenStr (use HH:mm or ISO 8601)", "bad_args")
            }
            val delayMs = (triggerAt - System.currentTimeMillis()).coerceAtLeast(0L)
            val notificationId = nextNotificationId.getAndIncrement()
            val work = OneTimeWorkRequestBuilder<ReminderWorker>()
                .setInitialDelay(delayMs, TimeUnit.MILLISECONDS)
                .setInputData(
                    Data.Builder()
                        .putString("title", "⏰ Reminder")
                        .putString("body", message)
                        .putInt("notificationId", notificationId)
                        .build()
                )
                .setConstraints(Constraints.NONE)
                .addTag("reminder")
                .build()
            val workName = "reminder-${System.currentTimeMillis()}"
            WorkManager.getInstance(context)
                .enqueueUniqueWork(workName, ExistingWorkPolicy.REPLACE, work)
            // Persist the metadata so the UI can list/cancel upcoming reminders.
            runCatching {
                reminderDao.insert(
                    ReminderEntity(
                        id = work.id.toString(),
                        message = message,
                        triggerAt = triggerAt,
                    )
                )
            }
            ToolResult.Ok("Reminder set for ${TimeParser.format(triggerAt)}: $message")
        },
    )

    companion object {
        /** Reserved notification ID space: 1000..9999 for reminders. */
        private const val NOTIFICATION_ID_START = 1000
    }
}
