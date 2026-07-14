package com.aura.agent

import com.aura.providers.ToolParameters
import com.aura.usage.UsageTracker
import io.mockk.mockk
import kotlinx.coroutines.runBlocking
import kotlin.test.Test
import kotlin.test.assertEquals

class ToolExecutorUsageTest {

    @Test
    fun `successful tool output contributes to usage ledger`() = runBlocking {
        val registry = ToolRegistry().apply {
            register(
                Tool(
                    name = "test_fetch",
                    description = "test",
                    risk = ToolRisk.READ_ONLY,
                    parameters = ToolParameters(),
                    execute = { _, _ -> ToolResult.Ok("x".repeat(100)) },
                ),
            )
        }
        val tracker = UsageTracker()
        val executor = ToolExecutor(registry, mockk(relaxed = true), tracker)

        executor.execute("test_fetch", "{}", ToolContext(conversationId = "test"))

        assertEquals(100, tracker.snapshot.value.toolResultChars)
    }
}
