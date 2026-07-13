package com.aura.tools

import com.aura.agent.Tool
import com.aura.agent.ToolResult
import com.aura.agent.ToolRisk
import com.aura.providers.ToolDefinition
import com.aura.providers.ToolParameters
import com.aura.providers.ToolProperty
import com.aura.tasks.ReminderRecurrence
import com.aura.tasks.ReminderScheduler
import javax.inject.Inject
import javax.inject.Singleton

/** Schedule one-time or recurring reminders through the shared lifecycle scheduler. */
@Singleton
class SetReminderTool @Inject constructor(
    private val reminderScheduler: ReminderScheduler,
) {
    fun definition() = ToolDefinition(
        name = "set_reminder",
        description = "Schedule a one-time or recurring reminder. 'when' is HH:mm or ISO 8601. recurrence is none, daily, weekly, or monthly.",
        parameters = ToolParameters(
            properties = mapOf(
                "when" to ToolProperty(type = "string", description = "Time to fire, e.g. '15:00' or '2026-06-26T15:00:00'"),
                "message" to ToolProperty(type = "string", description = "Reminder message"),
                "recurrence" to ToolProperty(type = "string", description = "Optional: none, daily, weekly, or monthly"),
            ),
            required = listOf("when", "message"),
        ),
    )

    val tool = Tool(
        name = "set_reminder",
        description = definition().description,
        risk = ToolRisk.WRITE_LOCAL,
        parameters = definition().parameters,
        execute = { call, _ ->
            val whenStr = call.arguments["when"] as? String
                ?: return@Tool ToolResult.Error("missing 'when'", "bad_args")
            val message = (call.arguments["message"] as? String)?.trim()
                ?: return@Tool ToolResult.Error("missing 'message'", "bad_args")
            if (message.isBlank()) return@Tool ToolResult.Error("message cannot be blank", "bad_args")
            val triggerAt = TimeParser.parse(whenStr)
                ?: return@Tool ToolResult.Error(
                    "could not parse 'when': $whenStr (use HH:mm or ISO 8601)",
                    "bad_args",
                )
            val rawRecurrence = call.arguments["recurrence"] as? String ?: "none"
            val recurrence = ReminderRecurrence.normalize(rawRecurrence)
            if (rawRecurrence.lowercase() !in ReminderRecurrence.supported) {
                return@Tool ToolResult.Error(
                    "unsupported recurrence '$rawRecurrence' (use none, daily, weekly, or monthly)",
                    "bad_args",
                )
            }
            reminderScheduler.create(message, triggerAt, recurrence)
            ToolResult.Ok(
                "${if (recurrence == "none") "Reminder" else recurrence.replaceFirstChar { it.uppercase() } + " reminder"} set for ${TimeParser.format(triggerAt)}: $message",
            )
        },
        category = "productivity",
    )
}
