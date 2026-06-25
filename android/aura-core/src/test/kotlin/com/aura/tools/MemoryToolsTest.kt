package com.aura.tools

import com.aura.memory.MemoryEntity
import com.aura.memory.MemoryStore
import io.mockk.coEvery
import io.mockk.coVerify
import io.mockk.mockk
import kotlinx.coroutines.test.runTest
import org.junit.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class MemoryToolsTest {

    @Test
    fun `remember tool calls memoryStore store`() = runTest {
        val store = mockk<MemoryStore>(relaxed = true)
        coEvery { store.store(any(), any(), any(), any(), any()) } returns "mem-1"
        val tool = RememberTool(store).tool
        val result = tool.execute(
            com.aura.agent.ToolCall(id = "tc1", name = "remember", arguments = mapOf("fact" to "User likes dark mode", "category" to "preference")),
            com.aura.agent.ToolContext(conversationId = "conv-1"),
        )
        assertTrue(result is com.aura.agent.ToolResult.Ok)
        coVerify { store.store("User likes dark mode", "user", "preference", 0.7f, emptyList()) }
    }

    @Test
    fun `recall tool queries and returns formatted hits`() = runTest {
        val store = mockk<MemoryStore>()
        coEvery { store.query("coffee", 5) } returns listOf(
            MemoryEntity(id = "1", content = "user drinks coffee", source = "user", category = "preference"),
            MemoryEntity(id = "2", content = "oat milk not dairy", source = "user", category = "preference"),
        )
        val tool = RecallTool(store).tool
        val result = tool.execute(
            com.aura.agent.ToolCall(id = "tc1", name = "recall", arguments = mapOf("query" to "coffee")),
            com.aura.agent.ToolContext(conversationId = "conv-1"),
        )
        assertTrue(result is com.aura.agent.ToolResult.Ok)
        val text = (result as com.aura.agent.ToolResult.Ok).output
        assertTrue(text.contains("coffee"))
        assertTrue(text.contains("oat milk"))
    }

    @Test
    fun `recall tool returns no results message on empty`() = runTest {
        val store = mockk<MemoryStore>()
        coEvery { store.query("xyzzy", 5) } returns emptyList()
        val tool = RecallTool(store).tool
        val result = tool.execute(
            com.aura.agent.ToolCall(id = "tc1", name = "recall", arguments = mapOf("query" to "xyzzy")),
            com.aura.agent.ToolContext(conversationId = "conv-1"),
        )
        assertTrue(result is com.aura.agent.ToolResult.Ok)
        assertTrue((result as com.aura.agent.ToolResult.Ok).output.contains("No memories"))
    }
}
