package com.aura.tools

import com.aura.agent.ToolCall
import com.aura.agent.ToolContext
import com.aura.agent.ToolResult
import com.aura.agent.ToolRisk
import io.mockk.mockk
import org.junit.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * Tests for [CalendarWriteTool] — confirmation gate (preview before
 * insert), argument validation, and risk classification.
 */
class CalendarWriteToolTest {

    private val context = mockk<android.content.Context>(relaxed = true)
    private val tool = CalendarWriteTool(context)

    private fun exec(args: Map<String, Any?>): ToolResult =
        kotlinx.coroutines.runBlocking {
            tool.tool.execute(
                ToolCall(id = "", name = "calendar_write", arguments = args),
                ToolContext(conversationId = "test"),
            )
        }

    @Test
    fun `risk is PRIVACY`() {
        assertEquals(ToolRisk.PRIVACY, tool.tool.risk)
    }

    @Test
    fun `missing title returns bad_args`() {
        val result = exec(mapOf("start" to "2026-07-15T10:00:00"))
        assertTrue(result is ToolResult.Error)
        assertEquals("bad_args", (result as ToolResult.Error).code)
    }

    @Test
    fun `missing start returns bad_args`() {
        val result = exec(mapOf("title" to "Meeting"))
        assertTrue(result is ToolResult.Error)
        assertEquals("bad_args", (result as ToolResult.Error).code)
    }

    @Test
    fun `confirmed parameter is in tool definition`() {
        val def = tool.definition()
        assertTrue(def.parameters.properties.containsKey("confirmed"),
            "tool definition should include 'confirmed' parameter")
    }

    @Test
    fun `tool description mentions confirmation flow`() {
        val def = tool.definition()
        assertTrue(def.description.contains("confirm"),
            "description should mention confirmation: ${def.description}")
    }

    private val draft = CalendarEventDraft(
        title = "Meeting",
        start = 1_000L,
        end = 2_000L,
        location = "Office",
        description = null,
    )

    @Test
    fun `model cannot confirm without a pending user-reviewed draft`() {
        val gate = CalendarApprovalGate()
        val result = gate.authorize(
            callConfirmed = true,
            ctx = ToolContext(conversationId = "c1", userMessage = "create a meeting"),
            draft = draft,
        )
        assertTrue(result?.contains("Please confirm") == true)
    }

    @Test
    fun `model cannot confirm again on the same user turn`() {
        val gate = CalendarApprovalGate()
        val sameTurn = ToolContext(conversationId = "c1", userMessage = "create a meeting")
        gate.authorize(callConfirmed = false, ctx = sameTurn, draft = draft)

        val result = gate.authorize(callConfirmed = true, ctx = sameTurn, draft = draft)

        assertTrue(result?.contains("explicit confirmation") == true)
    }

    @Test
    fun `later explicit user confirmation authorizes the exact pending draft once`() {
        val gate = CalendarApprovalGate()
        gate.authorize(
            callConfirmed = false,
            ctx = ToolContext(conversationId = "c1", userMessage = "create a meeting"),
            draft = draft,
        )

        val approved = gate.authorize(
            callConfirmed = true,
            ctx = ToolContext(conversationId = "c1", userMessage = "yes"),
            draft = draft,
        )
        val replay = gate.authorize(
            callConfirmed = true,
            ctx = ToolContext(conversationId = "c1", userMessage = "yes"),
            draft = draft,
        )

        assertEquals(null, approved)
        assertTrue(replay?.contains("Please confirm") == true)
    }

    @Test
    fun `changed draft requires a new preview even after yes`() {
        val gate = CalendarApprovalGate()
        gate.authorize(
            callConfirmed = false,
            ctx = ToolContext(conversationId = "c1", userMessage = "create a meeting"),
            draft = draft,
        )

        val changed = gate.authorize(
            callConfirmed = true,
            ctx = ToolContext(conversationId = "c1", userMessage = "yes"),
            draft = draft.copy(start = 3_000L),
        )

        assertTrue(changed?.contains("Please confirm") == true)
    }
}