package com.aura.tools

import java.util.Calendar
import java.util.Locale
import java.util.TimeZone

/**
 * Shared time parser for reminder/task scheduling. Used by [SetReminderTool]
 * and [TaskManagerTool] so the two agree on input formats.
 *
 * Accepted forms:
 *   - "HH:mm"           (24h local time, today or tomorrow if already passed)
 *   - "2026-06-26T15:00:00"  (ISO 8601, second precision, local timezone)
 *
 * Returns epoch millis in the local timezone, or null on parse failure.
 */
object TimeParser {
    private val iso = java.text.SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss", Locale.US)
        .apply { timeZone = TimeZone.getDefault() }

    fun parse(s: String): Long? {
        // Try ISO 8601 first; fall back to HH:mm. The exception flow is
        // explicit (no try/catch around iso) so a malformed ISO that
        // happens to look like "HH:mm" doesn't get mis-parsed.
        return tryIso(s) ?: tryHhMm(s)
    }

    private fun tryIso(s: String): Long? = try {
        iso.parse(s)?.time
    } catch (_: Exception) {
        null
    }

    private fun tryHhMm(s: String): Long? {
        val parts = s.split(":")
        if (parts.size != 2) return null
        val hour = parts[0].toIntOrNull() ?: return null
        val minute = parts[1].toIntOrNull() ?: return null
        if (hour !in 0..23 || minute !in 0..59) return null
        val cal = Calendar.getInstance()
        cal.set(Calendar.HOUR_OF_DAY, hour)
        cal.set(Calendar.MINUTE, minute)
        cal.set(Calendar.SECOND, 0)
        cal.set(Calendar.MILLISECOND, 0)
        if (cal.timeInMillis <= System.currentTimeMillis()) {
            cal.add(Calendar.DAY_OF_YEAR, 1)
        }
        return cal.timeInMillis
    }

    fun format(triggerAt: Long): String =
        java.text.SimpleDateFormat("EEE MMM d, HH:mm", Locale.US).format(triggerAt)
}
