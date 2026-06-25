package com.aura.tools

import com.aura.tools.TimeParser
import org.junit.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue

/**
 * Tests for the shared time parser used by SetReminderTool + TaskManagerTool.
 */
class TimeParserTest {

    @Test
    fun `parses HH mm for later today`() {
        val cal = java.util.Calendar.getInstance()
        cal.add(java.util.Calendar.DAY_OF_YEAR, 1) // guaranteed future
        val hh = (cal.get(java.util.Calendar.HOUR_OF_DAY) + 1) % 24
        val result = TimeParser.parse(String.format("%02d:00", hh))
        assertNotNull(result)
        assertTrue(result > System.currentTimeMillis(), "result should be in the future")
    }

    @Test
    fun `parses ISO 8601 datetime`() {
        val cal = java.util.Calendar.getInstance()
        cal.add(java.util.Calendar.YEAR, 1)
        val iso = String.format(
            "%04d-%02d-%02dT15:30:00",
            cal.get(java.util.Calendar.YEAR),
            cal.get(java.util.Calendar.MONTH) + 1,
            cal.get(java.util.Calendar.DAY_OF_MONTH),
        )
        val result = TimeParser.parse(iso)
        assertNotNull(result)
        // The year should match what we put in.
        val resultCal = java.util.Calendar.getInstance().apply { timeInMillis = result }
        assertEquals(cal.get(java.util.Calendar.YEAR), resultCal.get(java.util.Calendar.YEAR))
    }

    @Test
    fun `rejects garbage input`() {
        assertNull(TimeParser.parse(""))
        assertNull(TimeParser.parse("not a time"))
        assertNull(TimeParser.parse("25:99"))  // out-of-range hours/minutes
        assertNull(TimeParser.parse("12:60"))  // minutes must be 0..59
        assertNull(TimeParser.parse("12"))     // missing minutes
    }

    @Test
    fun `format returns non-empty string`() {
        val out = TimeParser.format(System.currentTimeMillis() + 60_000L)
        assertTrue(out.isNotBlank(), "format should produce a non-blank string")
    }
}
