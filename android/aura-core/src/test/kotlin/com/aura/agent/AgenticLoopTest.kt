package com.aura.agent

import com.aura.providers.ProviderChunk
import com.aura.providers.ProviderMessage
import com.aura.providers.ProviderRegistry
import com.aura.providers.FinishReason
import com.aura.providers.ProviderError
import io.mockk.coEvery
import io.mockk.every
import io.mockk.mockk
import kotlinx.coroutines.flow.flowOf
import kotlinx.coroutines.test.runTest
import org.junit.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class AgenticLoopTest {

    @Test
    fun `single text response completes and emits Done`() = runTest {
        val brain = mockk<Brain>(relaxed = true)
        val registry = mockk<ToolRegistry>(relaxed = true)
        val executor = mockk<ToolExecutor>(relaxed = true)
        every { registry.definitions() } returns emptyList()
        coEvery { brain.stream(any(), any(), any(), any()) } returns flowOf(
            BrainChunk.Text("Hello, "),
            BrainChunk.Text("world."),
            BrainChunk.Finished(FinishReason.stop.name),
        )

        val loop = AgenticLoop(brain, registry, executor)
        val events = mutableListOf<AgentEvent>()
        val conv = Conversation(systemPrompt = "you are a test bot")
        conv.addUser("hi")
        loop.run(conv, model = "ollama:test").collect { events += it }

        assertTrue(events.any { it is AgentEvent.TextDelta && it.text == "Hello, " })
        assertTrue(events.any { it is AgentEvent.Done })
        assertEquals("Hello, world.", conv.turns.last().assistant)
    }

    @Test
    fun `tool call is dispatched and result fed back`() = runTest {
        val brain = mockk<Brain>(relaxed = true)
        val registry = mockk<ToolRegistry>(relaxed = true)
        val executor = mockk<ToolExecutor>(relaxed = true)
        every { registry.definitions() } returns emptyList()
        every { registry.get("ping") } returns Tool(
            name = "ping",
            description = "ping",
            risk = ToolRisk.READ_ONLY,
            execute = { _, _ -> ToolResult.Ok("pong") },
        )
        coEvery { executor.execute(any(), any(), any()) } returns ToolResult.Ok("pong")

        // First call: model emits tool call. Second call: model emits text and stops.
        coEvery { brain.stream(any(), any(), any(), any()) } returnsMany listOf(
            flowOf(
                BrainChunk.ToolCallStart("tc1", "ping"),
                BrainChunk.ToolCallDelta("tc1", "{}"),
                BrainChunk.ToolCallEnd("tc1", "ping", "{}"),
                BrainChunk.Finished(FinishReason.tool_calls.name),
            ),
            flowOf(
                BrainChunk.Text("I pinged, got pong."),
                BrainChunk.Finished(FinishReason.stop.name),
            ),
        )

        val loop = AgenticLoop(brain, registry, executor)
        val conv = Conversation(systemPrompt = "you are a test bot")
        conv.addUser("ping")
        val events = mutableListOf<AgentEvent>()
        loop.run(conv, model = "ollama:test", maxSteps = 5).collect { events += it }

        assertTrue(events.any { it is AgentEvent.ToolExecuting && it.name == "ping" })
        assertTrue(events.any { it is AgentEvent.ToolResult && it.result == "pong" })
        assertTrue(events.any { it is AgentEvent.Done })
    }
}
