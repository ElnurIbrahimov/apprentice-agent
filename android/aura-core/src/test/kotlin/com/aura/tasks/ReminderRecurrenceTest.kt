package com.aura.tasks

import java.time.LocalDateTime
import java.time.ZoneId
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNull

class ReminderRecurrenceTest {
    private val utc = ZoneId.of("UTC")

    @Test
    fun `none has no next occurrence`() {
        assertNull(ReminderRecurrence.nextTrigger(1_000L, "none", 2_000L, utc))
    }

    @Test
    fun `daily advances until occurrence is in future`() {
        val original = LocalDateTime.of(2026, 7, 1, 9, 30).atZone(utc).toInstant().toEpochMilli()
        val now = LocalDateTime.of(2026, 7, 3, 10, 0).atZone(utc).toInstant().toEpochMilli()
        val expected = LocalDateTime.of(2026, 7, 4, 9, 30).atZone(utc).toInstant().toEpochMilli()

        assertEquals(expected, ReminderRecurrence.nextTrigger(original, "daily", now, utc))
    }

    @Test
    fun `monthly preserves local wall-clock schedule`() {
        val original = LocalDateTime.of(2026, 1, 15, 8, 0).atZone(utc).toInstant().toEpochMilli()
        val now = LocalDateTime.of(2026, 2, 16, 8, 0).atZone(utc).toInstant().toEpochMilli()
        val expected = LocalDateTime.of(2026, 3, 15, 8, 0).atZone(utc).toInstant().toEpochMilli()

        assertEquals(expected, ReminderRecurrence.nextTrigger(original, "monthly", now, utc))
    }
}
