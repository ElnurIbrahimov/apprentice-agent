package com.aura.agent

import com.aura.providers.ToolParameters
import com.aura.providers.ToolProperty
import io.mockk.every
import io.mockk.mockk
import kotlinx.coroutines.runBlocking
import org.junit.Test
import java.util.concurrent.atomic.AtomicInteger
import kotlin.test.assertEquals
import kotlin.test.assertIs

class ToolExecutorRemoteCostApprovalTest {
    private val registry = mockk<ToolRegistry>()
    private val context = mockk<android.content.Context>(relaxed = true)
    private val executions = AtomicInteger(0)
    private val executor = ToolExecutor(registry, context)

    private fun registerCostTool() {
        every { registry.get("image_generate") } returns Tool(
            name = "image_generate",
            description = "Generate an image",
            risk = ToolRisk.REMOTE_COST,
            parameters = ToolParameters(
                properties = mapOf("prompt" to ToolProperty(type = "string")),
                required = listOf("prompt"),
            ),
            execute = { _, _ ->
                executions.incrementAndGet()
                ToolResult.Ok("generated")
            },
        )
    }

    @Test
    fun `remote cost tool requires approval before execution`() = runBlocking {
        registerCostTool()

        val result = executor.execute(
            "image_generate",
            """{"prompt":"a lighthouse"}""",
            ToolContext(conversationId = "c1", userMessage = "Generate a lighthouse"),
        )

        assertIs<ToolResult.NeedsApproval>(result)
        assertEquals(0, executions.get())
    }

    @Test
    fun `later explicit confirmation executes the exact pending request once`() = runBlocking {
        registerCostTool()
        val args = """{"prompt":"a lighthouse"}"""
        executor.execute(
            "image_generate",
            args,
            ToolContext(conversationId = "c1", userMessage = "Generate a lighthouse"),
        )

        val approved = executor.execute(
            "image_generate",
            args,
            ToolContext(conversationId = "c1", userMessage = "Yes, confirm"),
        )
        val replay = executor.execute(
            "image_generate",
            args,
            ToolContext(conversationId = "c1", userMessage = "Yes, confirm"),
        )

        assertIs<ToolResult.Ok>(approved)
        assertIs<ToolResult.NeedsApproval>(replay)
        assertEquals(1, executions.get())
    }

    @Test
    fun `changed arguments require a fresh approval`() = runBlocking {
        registerCostTool()
        executor.execute(
            "image_generate",
            """{"prompt":"a lighthouse"}""",
            ToolContext(conversationId = "c1", userMessage = "Generate a lighthouse"),
        )

        val result = executor.execute(
            "image_generate",
            """{"prompt":"a portrait"}""",
            ToolContext(conversationId = "c1", userMessage = "Yes, confirm"),
        )

        assertIs<ToolResult.NeedsApproval>(result)
        assertEquals(0, executions.get())
    }

    @Test
    fun `remote cost is not misclassified as an incognito write`() = runBlocking {
        registerCostTool()

        val result = executor.execute(
            "image_generate",
            """{"prompt":"a lighthouse"}""",
            ToolContext(
                conversationId = "c1",
                userMessage = "Generate a lighthouse",
                memoryEnabled = false,
            ),
        )

        assertIs<ToolResult.NeedsApproval>(result)
        assertEquals(0, executions.get())
    }
}
