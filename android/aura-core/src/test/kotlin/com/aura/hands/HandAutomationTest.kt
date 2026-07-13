package com.aura.hands

import java.time.DayOfWeek
import java.time.ZoneId
import java.time.ZonedDateTime
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNull

class HandAutomationTest {

    private val monday = ZonedDateTime.of(2026, 7, 13, 10, 0, 0, 0, ZoneId.of("UTC"))

    @Test
    fun `daily schedule rolls to tomorrow when today's time passed`() {
        val hand = Hand(
            id = "h1",
            name = "Morning",
            scheduleType = HandScheduleType.DAILY.value,
            scheduleHour = 9,
            scheduleMinute = 30,
        )

        val next = HandScheduler.nextRunAt(hand, monday)

        assertEquals(ZonedDateTime.of(2026, 7, 14, 9, 30, 0, 0, ZoneId.of("UTC")), next)
    }

    @Test
    fun `weekday schedule skips weekend`() {
        val friday = ZonedDateTime.of(2026, 7, 17, 18, 0, 0, 0, ZoneId.of("UTC"))
        val hand = Hand(
            id = "h1",
            name = "Workday close",
            scheduleType = HandScheduleType.WEEKDAYS.value,
            scheduleHour = 17,
            scheduleMinute = 0,
        )

        val next = HandScheduler.nextRunAt(hand, friday)

        assertEquals(DayOfWeek.MONDAY, next?.dayOfWeek)
        assertEquals(17, next?.hour)
    }

    @Test
    fun `weekly schedule uses selected local weekday`() {
        val hand = Hand(
            id = "h1",
            name = "Wednesday report",
            scheduleType = HandScheduleType.WEEKLY.value,
            scheduleHour = 8,
            scheduleMinute = 15,
            scheduleDayOfWeek = DayOfWeek.WEDNESDAY.value,
        )

        val next = HandScheduler.nextRunAt(hand, monday)

        assertEquals(DayOfWeek.WEDNESDAY, next?.dayOfWeek)
        assertEquals(8, next?.hour)
        assertEquals(15, next?.minute)
    }

    @Test
    fun `disabled or unscheduled hand has no next run`() {
        assertNull(HandScheduler.nextRunAt(Hand("1", "Off", enabled = false), monday))
        assertNull(HandScheduler.nextRunAt(Hand("2", "Manual"), monday))
    }

    @Test
    fun `numeric conditions compare numbers and reject invalid operands`() {
        val variables = mapOf("score" to "8.5", "broken" to "many")

        assertEquals(true, HandCondition("score", "greater_than", "8").matches(variables))
        assertEquals(true, HandCondition("score", "less_than", "9").matches(variables))
        assertEquals(false, HandCondition("score", "greater_than", "many").matches(variables))
        assertEquals(false, HandCondition("broken", "less_than", "9").matches(variables))
    }

    @Test
    fun `empty condition aliases remain explicit`() {
        val variables = mapOf("set" to "yes", "blank" to "")

        assertEquals(true, HandCondition("set", "not_empty").matches(variables))
        assertEquals(true, HandCondition("blank", "empty").matches(variables))
    }

    @Test
    fun `condition operators are explicit and deterministic`() {
        val variables = mapOf("mode" to "work", "city" to "Baku", "empty" to "")

        assertEquals(true, HandCondition("mode", "equals", "work").matches(variables))
        assertEquals(true, HandCondition("city", "contains", "aku").matches(variables))
        assertEquals(true, HandCondition("empty", "is_empty").matches(variables))
        assertEquals(true, HandCondition("mode", "is_set").matches(variables))
        assertEquals(false, HandCondition("missing", "is_set").matches(variables))
        assertEquals(false, HandCondition("mode", "unknown", "work").matches(variables))
    }
}
