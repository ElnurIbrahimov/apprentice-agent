package com.aura.agent

import io.mockk.mockk
import kotlinx.coroutines.delay
import kotlinx.coroutines.test.runTest
import org.junit.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * Tests for the per-tool timeout enforced by [ToolExecutor] via
 * `withTimeout(ctx.timeout)`. Tools that exceed their budget must be
 * aborted with a typed `tool_timeout` error instead of hanging the
 * agent loop indefinitely.
 *
 * IMPORTANT: the tool's execute lambda is a `suspend fun`, so a
 * real `delay()` participates in coroutine cancellation. A
 * `runBlocking { delay() }` would block a thread and withTimeout
 * could not interrupt it.
 */
class ToolExecutorTimeoutTest {

    private val registry = mockk<ToolRegistry>()
    private val context = mockk<android.content.Context>(relaxed = true)
    private val executor = ToolExecutor(registry, context)

    private fun makeSuspendingTool(name: String, work: suspend () -> ToolResult) = Tool(
        name = name,
        description = "$name test tool",
        risk = ToolRisk.READ_ONLY,
        execute = { _, _ -> work() },
    )

    @Test
    fun `slow tool returns tool_timeout error when exceeding ctx timeout`() = runTest {
        val slow = makeSuspendingTool("slow_tool") {
            delay(5_000L)
            ToolResult.Ok("slow_tool ran")
        }
        io.mockk.every { registry.get("slow_tool") } returns slow
        val result = executor.execute(
            name = "slow_tool",
            argumentsJson = """{}""",
            ctx = ToolContext(conversationId = "c1", timeout = 100L),
        )
        assertTrue(result is ToolResult.Error, "expected Error, got $result")
        val err = result as ToolResult.Error
        assertEquals("tool_timeout", err.code)
        assertTrue(err.message.contains("0s") || err.message.contains("timed out"),
            "message should mention timeout, got: ${err.message}")
    }

    @Test
    fun `fast tool completes normally within timeout`() = runTest {
        val fast = makeSuspendingTool("fast_tool") {
            delay(10L)
            ToolResult.Ok("fast_tool ran")
        }
        io.mockk.every { registry.get("fast_tool") } returns fast
        val result = executor.execute(
            name = "fast_tool",
            argumentsJson = """{}""",
            ctx = ToolContext(conversationId = "c1", timeout = 1_000L),
        )
        assertTrue(result is ToolResult.Ok, "expected Ok, got $result")
    }
}
