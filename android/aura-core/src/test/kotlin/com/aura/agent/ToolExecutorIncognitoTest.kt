package com.aura.agent

import io.mockk.mockk
import kotlinx.coroutines.runBlocking
import org.junit.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * Tests for the privacy boundary enforced by [ToolExecutor] when
 * [ToolContext.memoryEnabled] is false (incognito mode).
 *
 * Tools whose [ToolRisk] >= WRITE_LOCAL must be refused. READ_ONLY tools
 * must still pass through to their `execute` body.
 */
class ToolExecutorIncognitoTest {

    private val registry = mockk<ToolRegistry>()
    private val context = mockk<android.content.Context>(relaxed = true)
    private val executor = ToolExecutor(registry, context)

    private fun makeTool(name: String, risk: ToolRisk) = Tool(
        name = name,
        description = "$name test tool",
        risk = risk,
        execute = { _, _ ->
            ToolResult.Ok("$name ran")
        },
    )

    @Test
    fun `write_local tool is refused when memoryEnabled=false`() = runBlocking {
        val remember = makeTool("remember", ToolRisk.WRITE_LOCAL)
        io.mockk.every { registry.get("remember") } returns remember
        val result = executor.execute(
            name = "remember",
            argumentsJson = """{"fact":"test"}""",
            ctx = ToolContext(conversationId = "c1", memoryEnabled = false),
        )
        assertTrue(result is ToolResult.Error, "expected Error, got $result")
        assertEquals("incognito_blocked", (result as ToolResult.Error).code)
    }

    @Test
    fun `write_local tool runs normally when memoryEnabled=true`() = runBlocking {
        val remember = makeTool("remember", ToolRisk.WRITE_LOCAL)
        io.mockk.every { registry.get("remember") } returns remember
        val result = executor.execute(
            name = "remember",
            argumentsJson = """{"fact":"test"}""",
            ctx = ToolContext(conversationId = "c1", memoryEnabled = true),
        )
        assertTrue(result is ToolResult.Ok, "expected Ok, got $result")
    }

    @Test
    fun `read_only tool runs even when memoryEnabled=false`() = runBlocking {
        val recall = makeTool("recall", ToolRisk.READ_ONLY)
        io.mockk.every { registry.get("recall") } returns recall
        val result = executor.execute(
            name = "recall",
            argumentsJson = """{"query":"preferences"}""",
            ctx = ToolContext(conversationId = "c1", memoryEnabled = false),
        )
        assertTrue(result is ToolResult.Ok, "recall must still work in incognito, got $result")
    }

    @Test
    fun `privacy tool (PRIVACY risk) is refused when memoryEnabled=false`() = runBlocking {
        // PRIVACY is treated as >= WRITE_LOCAL in the ordinal comparison.
        val contacts = makeTool("contacts_search", ToolRisk.PRIVACY)
        io.mockk.every { registry.get("contacts_search") } returns contacts
        val result = executor.execute(
            name = "contacts_search",
            argumentsJson = """{"query":"mom"}""",
            ctx = ToolContext(conversationId = "c1", memoryEnabled = false),
        )
        assertTrue(result is ToolResult.Error, "PRIVACY writes must be blocked in incognito, got $result")
        assertEquals("incognito_blocked", (result as ToolResult.Error).code)
    }

    @Test
    fun `default ToolContext has memoryEnabled=true`() {
        // Backwards-compat: existing call sites that don't pass memoryEnabled
        // must continue to allow WRITE_LOCAL tools.
        val ctx = ToolContext(conversationId = "c1")
        assertEquals(true, ctx.memoryEnabled)
    }
}