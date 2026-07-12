package com.aura.tools

import com.aura.agent.ToolCall
import com.aura.agent.ToolContext
import com.aura.agent.ToolResult
import com.aura.agent.ToolRisk
import com.aura.tasks.ReminderDao
import io.mockk.coEvery
import io.mockk.mockk
import org.junit.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue

/**
 * Tests for [SetReminderTool] — argument validation, risk
 * classification, and time parsing.
 */
class SetReminderToolTest {

    private val context = mockk<android.content.Context>(relaxed = true)
    private val reminderDao = mockk<ReminderDao>(relaxed = true)
    private val tool = SetReminderTool(context, reminderDao)

    private fun exec(args: Map<String, Any?>): ToolResult =
        kotlinx.coroutines.runBlocking {
            tool.tool.execute(
                ToolCall(id = "", name = "set_reminder", arguments = args),
                ToolContext(conversationId = "test"),
            )
        }

    @Test
    fun `risk is WRITE_LOCAL`() {
        assertEquals(ToolRisk.WRITE_LOCAL, tool.tool.risk)
    }

    @Test
    fun `missing when returns bad_args`() {
        val result = exec(mapOf("message" to "Take meds"))
        assertTrue(result is ToolResult.Error)
        assertEquals("bad_args", (result as ToolResult.Error).code)
    }

    @Test
    fun `missing message returns bad_args`() {
        val result = exec(mapOf("when" to "15:00"))
        assertTrue(result is ToolResult.Error)
        assertEquals("bad_args", (result as ToolResult.Error).code)
    }

    @Test
    fun `invalid time format returns bad_args`() {
        val result = exec(mapOf("when" to "not-a-time", "message" to "Test"))
        assertTrue(result is ToolResult.Error)
        assertEquals("bad_args", (result as ToolResult.Error).code)
    }

    @Test
    fun `valid HH_mm is accepted by TimeParser`() {
        // TimeParser.parse is a pure function that doesn't need
        // Android framework. Verify it accepts HH:mm format.
        val parsed = com.aura.tools.TimeParser.parse("23:59")
        assertNotNull(parsed, "23:59 should parse as a valid time")
        assertTrue(parsed > 0, "parsed timestamp should be positive")
    }

    @Test
    fun `invalid time string is rejected by TimeParser`() {
        val parsed = com.aura.tools.TimeParser.parse("not-a-time")
        assertNull(parsed, "invalid time string should return null")
    }
}