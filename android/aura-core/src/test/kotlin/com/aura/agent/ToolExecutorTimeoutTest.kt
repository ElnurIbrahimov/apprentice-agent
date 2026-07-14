package com.aura.agent

import io.mockk.mockk
import java.util.concurrent.Executors
import kotlin.system.measureTimeMillis
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.delay
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withContext
import org.junit.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

/**
 * Tests for the per-tool timeout enforced by [ToolExecutor] via
 * `withTimeout(ctx.timeout)`. Tools that exceed their budget must be
 * aborted with a typed `tool_timeout` error instead of hanging the
 * agent loop indefinitely.
 *
 * Tool execution crosses an interruptible `Dispatchers.IO` boundary. These
 * tests therefore use real coroutine time: virtual `runTest` time would
 * advance the outer timeout independently of the worker thread.
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
    fun `slow tool returns tool_timeout error when exceeding ctx timeout`() = runBlocking {
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
    fun `fast tool completes normally within timeout`() = runBlocking {
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

    @Test
    fun `tool execution leaves the caller thread`() = runBlocking {
        val threadTool = makeSuspendingTool("thread_tool") {
            ToolResult.Ok(Thread.currentThread().name)
        }
        io.mockk.every { registry.get("thread_tool") } returns threadTool
        val caller = Executors.newSingleThreadExecutor { runnable ->
            Thread(runnable, "ui-caller")
        }.asCoroutineDispatcher()

        try {
            val result = withContext(caller) {
                executor.execute(
                    name = "thread_tool",
                    argumentsJson = "{}",
                    ctx = ToolContext(conversationId = "c1"),
                )
            } as ToolResult.Ok

            assertFalse(
                result.output.contains("ui-caller"),
                "tool ran on caller thread: ${result.output}",
            )
        } finally {
            caller.close()
        }
    }

    @Test
    fun `blocking tool is interrupted when timeout expires`() = runBlocking {
        val blocking = makeSuspendingTool("blocking_tool") {
            Thread.sleep(3_000L)
            ToolResult.Ok("finished")
        }
        io.mockk.every { registry.get("blocking_tool") } returns blocking
        lateinit var result: ToolResult

        val elapsedMs = measureTimeMillis {
            result = executor.execute(
                name = "blocking_tool",
                argumentsJson = "{}",
                ctx = ToolContext(conversationId = "c1", timeout = 100L),
            )
        }

        assertTrue(result is ToolResult.Error, "expected timeout error, got $result")
        assertEquals("tool_timeout", (result as ToolResult.Error).code)
        assertTrue(elapsedMs < 1_000L, "blocking timeout took ${elapsedMs}ms")
    }
}
