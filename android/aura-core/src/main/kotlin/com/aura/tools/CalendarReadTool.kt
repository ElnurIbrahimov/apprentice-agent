package com.aura.tools

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.provider.CalendarContract
import androidx.core.content.ContextCompat
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
import java.util.Calendar
import java.util.TimeZone

/**
 * Read calendar events. v1: from now to 7 days ahead. v1.5: configurable range.
 * Mirrors aura/tools/calendar_tool.py.
 * Risk: PRIVACY (READ_CALENDAR).
 */
@Singleton
class CalendarReadTool @Inject constructor(
    @ApplicationContext private val context: Context,
) {
    fun definition() = ToolDefinition(
        name = "calendar_read",
        description = "Read upcoming calendar events. Default: today + next 7 days, max 20 events.",
        parameters = ToolParameters(
            properties = mapOf(
                "days" to ToolProperty(type = "integer", description = "Number of days to look ahead (default 7, max 30)"),
                "max_results" to ToolProperty(type = "integer", description = "Maximum events to return (default 20, max 50)"),
            ),
            required = emptyList(),
        ),
    )

    val tool = Tool(
        name = "calendar_read",
        description = definition().description,
        risk = ToolRisk.PRIVACY,
        requiredPermissions = listOf(Manifest.permission.READ_CALENDAR),
        parameters = definition().parameters,
        execute = { call, ctx ->
            if (ContextCompat.checkSelfPermission(context, Manifest.permission.READ_CALENDAR)
                != PackageManager.PERMISSION_GRANTED
            ) {
                return@Tool ToolResult.NeedsPermission(
                    Manifest.permission.READ_CALENDAR,
                    "Calendar access is needed to read events.",
                )
            }
            val days = (call.arguments["days"] as? Int ?: 7).coerceIn(1, 30)
            val max = (call.arguments["max_results"] as? Int ?: 20).coerceIn(1, 50)
            try {
                val events = readEvents(days, max)
                ToolResult.Ok(formatEvents(events))
            } catch (e: SecurityException) {
                ToolResult.NeedsPermission(Manifest.permission.READ_CALENDAR, "Calendar permission revoked.")
            } catch (e: Exception) {
                ToolResult.Error("calendar read failed: ${e.message}", "exception")
            }
        },
    )

    private data class Event(val title: String, val begin: Long, val end: Long, val location: String, val allDay: Boolean)

    private fun readEvents(days: Int, max: Int): List<Event> {
        val cal = Calendar.getInstance(TimeZone.getDefault())
        val now = cal.timeInMillis
        cal.add(Calendar.DAY_OF_YEAR, days)
        val end = cal.timeInMillis

        val projection = arrayOf(
            CalendarContract.Events.TITLE,
            CalendarContract.Events.DTSTART,
            CalendarContract.Events.DTEND,
            CalendarContract.Events.EVENT_LOCATION,
            CalendarContract.Events.ALL_DAY,
        )
        val selection = "${CalendarContract.Events.DTSTART} >= ? AND ${CalendarContract.Events.DTSTART} <= ?"
        val args = arrayOf(now.toString(), end.toString())
        val sort = "${CalendarContract.Events.DTSTART} ASC LIMIT $max"

        val out = mutableListOf<Event>()
        context.contentResolver.query(
            CalendarContract.Events.CONTENT_URI, projection, selection, args, sort
        )?.use { c ->
            val titleIdx = c.getColumnIndex(CalendarContract.Events.TITLE)
            val beginIdx = c.getColumnIndex(CalendarContract.Events.DTSTART)
            val endIdx = c.getColumnIndex(CalendarContract.Events.DTEND)
            val locIdx = c.getColumnIndex(CalendarContract.Events.EVENT_LOCATION)
            val allDayIdx = c.getColumnIndex(CalendarContract.Events.ALL_DAY)
            while (c.moveToNext()) {
                out += Event(
                    title = c.getString(titleIdx) ?: "(no title)",
                    begin = c.getLong(beginIdx),
                    end = c.getLong(endIdx),
                    location = c.getString(locIdx) ?: "",
                    allDay = c.getInt(allDayIdx) == 1,
                )
            }
        }
        return out
    }

    private fun formatEvents(events: List<Event>): String {
        if (events.isEmpty()) return "No upcoming events in the next 7 days."
        val df = java.text.SimpleDateFormat("EEE MMM d, HH:mm", java.util.Locale.US)
        return events.mapIndexed { i, e ->
            val time = if (e.allDay) "all day ${df.format(java.util.Date(e.begin)).substring(0, 10)}" else "${df.format(java.util.Date(e.begin))} - ${df.format(java.util.Date(e.end)).substringAfter(' ')}"
            val loc = if (e.location.isNotEmpty()) " @ ${e.location}" else ""
            "${i + 1}. ${e.title} ($time)$loc"
        }.joinToString("\n")
    }
}
