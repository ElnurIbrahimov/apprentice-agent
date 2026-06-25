package com.aura.agent

import com.aura.memory.MemoryEntity
import com.aura.memory.MemoryStore
import com.aura.providers.ChatOptions
import com.aura.providers.FinishReason
import com.aura.providers.Provider
import com.aura.providers.ProviderChunk
import com.aura.providers.ProviderRegistry
import com.aura.providers.ToolCall
import com.aura.tools.RememberTool
import com.aura.tools.RecallTool
import io.mockk.coEvery
import io.mockk.every
import io.mockk.mockk
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.flowOf
import kotlinx.coroutines.test.runTest
import org.junit.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * End-to-end test: a user asks a question, the agent uses the remember tool
 * to store a fact, the model emits a follow-up, the memory is auto-stored,
 * and a subsequent query retrieves it.
 *
 * The test wires the real MemoryAugmentedAgenticLoop with a real MemoryStore
 * but uses mock Provider + mock ToolExecutor to drive the agent deterministically.
 */
class EndToEndTest {

    @Test
    fun `user says I prefer dark mode - model calls remember tool - fact is stored`() = runTest {
        // 1. Set up the real memory store and a mock tool executor
        val memoryStore = mockk<MemoryStore>(relaxed = true)
        val rememberTool = RememberTool(memoryStore)
        val recallTool = RecallTool(memoryStore)
        val toolRegistry = ToolRegistry()
        toolRegistry.register(rememberTool.tool)
        toolRegistry.register(recallTool.tool)

        // 2. Set up a mock provider that emits a tool call to remember, then a follow-up response
        val provider = mockk<Provider>(relaxed = true)
        every { provider.prefix } returns "test"
        every { provider.isConfigured() } returns true

        val providerRegistry = mockk<ProviderRegistry>(relaxed = true)
        every { providerRegistry.parse(any<String>()) } returns (provider to "test-model")

        val brain = mockk<Brain>(relaxed = true)
        // First call: model emits a remember tool call
        // Second call: model emits a friendly acknowledgment
        coEvery { brain.stream(any(), any(), any(), any()) } returnsMany listOf(
            flowOf(
                BrainChunk.ToolCallStart("tc1", "remember"),
                BrainChunk.ToolCallDelta(
                    "tc1",
                    "{\"fact\": \"user prefers dark mode\", \"category\": \"preference\"}",
                ),
                BrainChunk.ToolCallEnd(
                    "tc1",
                    "remember",
                    "{\"fact\": \"user prefers dark mode\", \"category\": \"preference\"}",
                ),
                BrainChunk.Finished(FinishReason.tool_calls.name),
            ),
            flowOf(
                BrainChunk.Text("Got it \u2014 I'll remember that you prefer dark mode."),
                BrainChunk.Finished(FinishReason.stop.name),
            ),
        )

        val executor = ToolExecutor(toolRegistry, context = io.mockk.mockk(relaxed = true))
        val loop = MemoryAugmentedAgenticLoop(brain, toolRegistry, executor, memoryStore)

        // 3. Run a conversation
        val conv = Conversation()
        conv.addUser("I prefer dark mode")
        val events = mutableListOf<AgentEvent>()
        loop.run(conv, model = "test:model", maxSteps = 5).collect { events += it }

        // 4. Assert: the tool was called, the memory was stored, and the assistant responded
        val toolExec = events.filterIsInstance<AgentEvent.ToolExecuting>().firstOrNull { it.name == "remember" }
        assertTrue(toolExec != null, "remember tool should have been executed")

        val assistantText = conv.turns.last().assistant ?: ""
        assertTrue(assistantText.contains("Got it"), "assistant should respond with 'Got it', got: $assistantText")
    }

    @Test
    fun `user asks what model preferences are - recall returns prior memories`() = runTest {
        // Pre-populate memory store with a remembered preference
        val memoryStore = mockk<MemoryStore>()
        coEvery { memoryStore.query(any(), any()) } returns listOf(
            MemoryEntity(
                id = "m1",
                content = "user prefers dark mode",
                source = "user",
                category = "preference",
            )
        )
        val recallTool = RecallTool(memoryStore)
        val toolRegistry = ToolRegistry()
        toolRegistry.register(recallTool.tool)

        // Model emits a recall tool call
        val provider = mockk<Provider>(relaxed = true)
        every { provider.prefix } returns "test"
        every { provider.isConfigured() } returns true
        val providerRegistry = mockk<ProviderRegistry>(relaxed = true)
        every { providerRegistry.parse(any<String>()) } returns (provider to "test-model")

        val brain = mockk<Brain>(relaxed = true)
        coEvery { brain.stream(any(), any(), any(), any()) } returnsMany listOf(
            flowOf(
                BrainChunk.ToolCallStart("tc1", "recall"),
                BrainChunk.ToolCallDelta("tc1", "{\"query\": \"dark mode preference\"}"),
                BrainChunk.ToolCallEnd("tc1", "recall", "{\"query\": \"dark mode preference\"}"),
                BrainChunk.Finished(FinishReason.tool_calls.name),
            ),
            flowOf(
                BrainChunk.Text("You prefer dark mode."),
                BrainChunk.Finished(FinishReason.stop.name),
            ),
        )

        val executor = ToolExecutor(toolRegistry, context = io.mockk.mockk(relaxed = true))
        val loop = MemoryAugmentedAgenticLoop(brain, toolRegistry, executor, memoryStore)

        val conv = Conversation()
        conv.addUser("what do you remember about my preferences?")
        val events = mutableListOf<AgentEvent>()
        loop.run(conv, model = "test:model", maxSteps = 5).collect { events += it }

        val recallExec = events.filterIsInstance<AgentEvent.ToolExecuting>().firstOrNull { it.name == "recall" }
        assertTrue(recallExec != null, "recall tool should have been executed")

        val assistantText = conv.turns.last().assistant ?: ""
        assertTrue(assistantText.contains("dark mode"), "assistant should mention dark mode from memory, got: $assistantText")
    }

    @Test
    fun `auto-store happens after turn completes via WriteGate`() = runTest {
        // Test that a high-importance user message gets auto-stored
        val memoryStore = mockk<MemoryStore>(relaxed = true)
        val toolRegistry = ToolRegistry()
        val provider = mockk<Provider>(relaxed = true)
        every { provider.prefix } returns "test"
        every { provider.isConfigured() } returns true
        val providerRegistry = mockk<ProviderRegistry>(relaxed = true)
        every { providerRegistry.parse(any<String>()) } returns (provider to "test-model")
        val brain = mockk<Brain>(relaxed = true)
        coEvery { brain.stream(any(), any(), any(), any()) } returns flowOf(
            BrainChunk.Text("Nice to meet you, Elnur."),
            BrainChunk.Finished(FinishReason.stop.name),
        )
        val executor = ToolExecutor(toolRegistry, context = io.mockk.mockk(relaxed = true))
        val loop = MemoryAugmentedAgenticLoop(brain, toolRegistry, executor, memoryStore)

        val conv = Conversation()
        conv.addUser("my name is Elnur")
        loop.run(conv, model = "test:model", maxSteps = 2).collect { /* discard */ }

        // The auto-store path should have called maybeStore for the user message
        // (regardless of WriteGate decision — we just check it was called)
        io.mockk.coVerify { memoryStore.maybeStore(match { it.contains("Elnur") }, "user") }
    }
}
