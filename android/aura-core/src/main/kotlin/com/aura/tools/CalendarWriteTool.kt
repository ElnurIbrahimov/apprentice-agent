package com.aura.tools

import android.Manifest
import android.content.ContentUris
import android.content.ContentValues
import android.content.Context
import android.content.pm.PackageManager
import android.net.Uri
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
import java.util.TimeZone
import javax.inject.Inject
import javax.inject.Singleton

/**
 * Create a calendar event. Mirrors aura/tools/calendar_tool.py.
 * Risk: PRIVACY (WRITE_CALENDAR).
 */
@Singleton
class CalendarWriteTool @Inject constructor(
    @ApplicationContext private val context: Context,
) {
    fun definition() = ToolDefinition(
        name = "calendar_write",
        description = "Create a new calendar event with a title, start time, and optional end time and location. Times are ISO 8601 (e.g. '2026-06-26T15:00:00') or HH:mm (today/tomorrow). On the first call, a preview is returned for the user to confirm. Set confirmed=true on the second call to actually insert the event.",
        parameters = ToolParameters(
            properties = mapOf(
                "title" to ToolProperty(type = "string", description = "Event title"),
                "start" to ToolProperty(type = "string", description = "Start time (ISO 8601 or HH:mm)"),
                "end" to ToolProperty(type = "string", description = "End time (ISO 8601 or HH:mm). Defaults to 1 hour after start."),
                "location" to ToolProperty(type = "string", description = "Optional location"),
                "description" to ToolProperty(type = "string", description = "Optional description"),
                "confirmed" to ToolProperty(type = "boolean", description = "Set to true to confirm and insert the event after reviewing the preview."),
            ),
            required = listOf("title", "start"),
        ),
    )

    val tool = Tool(
        name = "calendar_write",
        description = definition().description,
        risk = ToolRisk.PRIVACY,
        requiredPermissions = listOf(Manifest.permission.WRITE_CALENDAR, Manifest.permission.READ_CALENDAR),
        parameters = definition().parameters,
        execute = { call, ctx ->
            val title = call.arguments["title"] as? String ?: return@Tool ToolResult.Error("missing 'title'", "bad_args")
            val startStr = call.arguments["start"] as? String ?: return@Tool ToolResult.Error("missing 'start'", "bad_args")
            val endStr = call.arguments["end"] as? String
            val location = call.arguments["location"] as? String
            val description = call.arguments["description"] as? String
            val confirmed = call.arguments["confirmed"] as? Boolean ?: false

            val granted = listOf(Manifest.permission.WRITE_CALENDAR, Manifest.permission.READ_CALENDAR)
                .all { ContextCompat.checkSelfPermission(context, it) == PackageManager.PERMISSION_GRANTED }
            if (!granted) {
                return@Tool ToolResult.NeedsPermission(Manifest.permission.WRITE_CALENDAR, "Calendar write access required.")
            }
            val start = parseTime(startStr) ?: return@Tool ToolResult.Error("bad 'start' time: $startStr", "bad_args")
            val end = if (endStr != null) {
                parseTime(endStr) ?: return@Tool ToolResult.Error("bad 'end' time: $endStr", "bad_args")
            } else start + 60L * 60L * 1000L

            // Confirmation gate: on the first call (confirmed=false), return a
            // preview of the event for the user to review. The model presents
            // this to the user and re-calls with confirmed=true if they agree.
            // This prevents the agent from silently inserting calendar events
            // without the user seeing the details first.
            if (!confirmed) {
                val preview = buildString {
                    appendLine("Please confirm the following calendar event:")
                    appendLine("  Title: $title")
                    appendLine("  Start: ${java.text.SimpleDateFormat("yyyy-MM-dd HH:mm z", java.util.Locale.getDefault()).format(java.util.Date(start))}")
                    appendLine("  End:   ${java.text.SimpleDateFormat("yyyy-MM-dd HH:mm z", java.util.Locale.getDefault()).format(java.util.Date(end))}")
                    if (location != null) appendLine("  Location: $location")
                    if (description != null) appendLine("  Description: $description")
                    appendLine()
                    appendLine("Reply with yes to confirm, or tell me what to change.")
                }
                return@Tool ToolResult.Ok(preview)
            }

            try {
                val eventId = insertEvent(title, start, end, location, description)
                ToolResult.Ok("Event created (id $eventId): $title")
            } catch (e: SecurityException) {
                ToolResult.NeedsPermission(Manifest.permission.WRITE_CALENDAR, "Calendar write permission revoked.")
            } catch (e: Exception) {
                ToolResult.Error("calendar_write failed: ${e.message}", "exception")
            }
        },
    category = "productivity")
    /**
     * Parse the time string. Delegates to [TimeParser] so reminder
     * events and ad-hoc calendar events agree on input formats and
     * validation (HH:mm must have an hour in 0..23 and a minute in
     * 0..59 — the local copy of this routine silently accepted
     * "99:99" as 99:99 and the calendar event landed at midnight + 1
     * day. TimeParser is the canonical parser now).
     */
    private fun parseTime(s: String): Long? = TimeParser.parse(s)

    private fun insertEvent(title: String, start: Long, end: Long, location: String?, description: String?): Long {
        val calId = primaryCalendarId() ?: throw IllegalStateException("No calendar account on device.")
        val values = ContentValues().apply {
            put(CalendarContract.Events.DTSTART, start)
            put(CalendarContract.Events.DTEND, end)
            put(CalendarContract.Events.TITLE, title)
            if (location != null) put(CalendarContract.Events.EVENT_LOCATION, location)
            if (description != null) put(CalendarContract.Events.DESCRIPTION, description)
            put(CalendarContract.Events.CALENDAR_ID, calId)
            put(CalendarContract.Events.EVENT_TIMEZONE, TimeZone.getDefault().id)
        }
        val uri: Uri? = context.contentResolver.insert(CalendarContract.Events.CONTENT_URI, values)
        return uri?.let { ContentUris.parseId(it) } ?: -1L
    }

    private fun primaryCalendarId(): Long? {
        val projection = arrayOf(CalendarContract.Calendars._ID, CalendarContract.Calendars.IS_PRIMARY, CalendarContract.Calendars.VISIBLE)
        context.contentResolver.query(CalendarContract.Calendars.CONTENT_URI, projection, null, null, null)?.use { c ->
            var fallback: Long? = null
            while (c.moveToNext()) {
                val id = c.getLong(0)
                if (c.getInt(1) == 1) return id
                if (fallback == null && c.getInt(2) == 1) fallback = id
            }
            return fallback
        }
        return null
    }
}
