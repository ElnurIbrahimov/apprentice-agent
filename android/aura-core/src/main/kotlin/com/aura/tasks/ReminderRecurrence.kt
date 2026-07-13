package com.aura.tasks

import java.time.Instant
import java.time.ZoneId
import java.time.ZonedDateTime

object ReminderRecurrence {
    val supported = setOf("none", "daily", "weekly", "monthly")

    fun normalize(value: String?): String =
        value?.lowercase()?.takeIf { it in supported } ?: "none"

    fun nextTrigger(
        previousTriggerAt: Long,
        recurrence: String,
        now: Long = System.currentTimeMillis(),
        zoneId: ZoneId = ZoneId.systemDefault(),
    ): Long? {
        val normalized = normalize(recurrence)
        if (normalized == "none") return null
        var next = Instant.ofEpochMilli(previousTriggerAt).atZone(zoneId)
        do {
            next = advance(next, normalized)
        } while (next.toInstant().toEpochMilli() <= now)
        return next.toInstant().toEpochMilli()
    }

    private fun advance(value: ZonedDateTime, recurrence: String): ZonedDateTime = when (recurrence) {
        "daily" -> value.plusDays(1)
        "weekly" -> value.plusWeeks(1)
        "monthly" -> value.plusMonths(1)
        else -> value
    }
}
