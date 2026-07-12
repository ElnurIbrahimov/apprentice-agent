package com.aura.tools

import com.aura.agent.ToolResult
import com.aura.providers.ChatOptions
import com.aura.providers.FinishReason
import com.aura.providers.ProviderChunk
import com.aura.providers.ProviderMessage
import com.aura.providers.ProviderRegistry
import io.mockk.coEvery
import io.mockk.every
import io.mockk.mockk
import kotlinx.coroutines.flow.flowOf
import kotlinx.coroutines.test.runTest
import org.junit.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * Tests for [KnowledgeGraphTool].
 *
 * Mocks ProviderRegistry so no real LLM call is made.
 */
class KnowledgeGraphToolTest {

    @Test
    fun `extracts nodes and edges from text`() = runTest {
        val llmResponse = """
            {
              "nodes": [
                {"label": "Kotlin", "type": "skill"},
                {"label": "Android", "type": "project"}
              ],
              "edges": [
                {"type": "used_for", "source_label": "Kotlin", "target_label": "Android", "weight": 0.9}
              ]
            }
        """.trimIndent()

        val mockRegistry = mockk<ProviderRegistry> {
            coEvery { chat("default", any<List<ProviderMessage>>(), any<ChatOptions>()) } returns flowOf(
                ProviderChunk(text = llmResponse),
                ProviderChunk(finishReason = FinishReason.stop),
            )
        }

        val tool = KnowledgeGraphTool(mockRegistry).tool
        val result = tool.execute(
            call("knowledge_graph_extract", "text" to "Kotlin is used for Android development."),
            ctx(),
        )

        assertTrue("expected Ok, got $result") { result is ToolResult.Ok }
        val json = (result as ToolResult.Ok).output
        assertTrue(json.contains("\"nodes\""), "should contain nodes: $json")
        assertTrue(json.contains("\"edges\""), "should contain edges: $json")
        assertTrue(json.contains("Kotlin"), "should contain node label Kotlin: $json")
        assertTrue(json.contains("Android"), "should contain node label Android: $json")
        assertTrue(json.contains("\"type\":\"skill\""), "should have skill type: $json")
        assertTrue(json.contains("\"type\":\"project\""), "should have project type: $json")
    }

    @Test
    fun `missing text returns error`() = runTest {
        val mockRegistry = mockk<ProviderRegistry>()
        val tool = KnowledgeGraphTool(mockRegistry).tool
        val result = tool.execute(call("knowledge_graph_extract"), ctx())

        assertTrue("expected Error, got $result") { result is ToolResult.Error }
        assertEquals("bad_args", (result as ToolResult.Error).code)
    }

    @Test
    fun `LLM returns empty JSON when no entities found`() = runTest {
        val llmResponse = """{"nodes":[],"edges":[]}"""

        val mockRegistry = mockk<ProviderRegistry> {
            coEvery { chat("default", any<List<ProviderMessage>>(), any<ChatOptions>()) } returns flowOf(
                ProviderChunk(text = llmResponse),
                ProviderChunk(finishReason = FinishReason.stop),
            )
        }

        val tool = KnowledgeGraphTool(mockRegistry).tool
        val result = tool.execute(
            call("knowledge_graph_extract", "text" to "Just some random text."),
            ctx(),
        )

        assertTrue("expected Ok, got $result") { result is ToolResult.Ok }
        val json = (result as ToolResult.Ok).output
        assertTrue(json.contains("\"nodes\":[]"), "should have empty nodes: $json")
        assertTrue(json.contains("\"edges\":[]"), "should have empty edges: $json")
    }

    @Test
    fun `LLM response with unknown node type maps to unknown`() = runTest {
        val llmResponse = """
            {
              "nodes": [
                {"label": "MysteryThing", "type": "alien_species"}
              ],
              "edges": []
            }
        """.trimIndent()

        val mockRegistry = mockk<ProviderRegistry> {
            coEvery { chat("default", any<List<ProviderMessage>>(), any<ChatOptions>()) } returns flowOf(
                ProviderChunk(text = llmResponse),
                ProviderChunk(finishReason = FinishReason.stop),
            )
        }

        val tool = KnowledgeGraphTool(mockRegistry).tool
        val result = tool.execute(
            call("knowledge_graph_extract", "text" to "Some alien species."),
            ctx(),
        )

        assertTrue("expected Ok, got $result") { result is ToolResult.Ok }
        val json = (result as ToolResult.Ok).output
        assertTrue(json.contains("MysteryThing"), "should contain the node label: $json")
        assertTrue(json.contains("\"type\":\"unknown\""), "unknown type should map to 'unknown': $json")
    }

    @Test
    fun `malformed LLM response falls back to empty`() = runTest {
        val llmResponse = "This is not JSON at all."

        val mockRegistry = mockk<ProviderRegistry> {
            coEvery { chat("default", any<List<ProviderMessage>>(), any<ChatOptions>()) } returns flowOf(
                ProviderChunk(text = llmResponse),
                ProviderChunk(finishReason = FinishReason.stop),
            )
        }

        val tool = KnowledgeGraphTool(mockRegistry).tool
        val result = tool.execute(
            call("knowledge_graph_extract", "text" to "Something."),
            ctx(),
        )

        assertTrue("expected Ok, got $result") { result is ToolResult.Ok }
        val json = (result as ToolResult.Ok).output
        assertEquals("""{"nodes":[],"edges":[]}""", json)
    }

    @Test
    fun `falls back to first configured provider when default parse fails`() = runTest {
        // Old test pinned the fallback to "ollama:deepseek-v4-pro" — that was
        // a 2026-07-07 bug (hardcoded model that crashed users without an
        // Ollama key). The new contract: when ProviderRegistry.parse("default")
        // throws (no configured providers, or "default" is not a valid
        // model id for any provider), fall back to the first configured
        // provider's first model. The test now mocks that path.
        val llmResponse = """{"nodes":[{"label":"Fallback","type":"concept"}],"edges":[]}"""

        val mockProvider = mockk<com.aura.providers.Provider> {
            every { prefix } returns "anthropic"
            coEvery { listModels() } returns listOf("claude-sonnet-4.6")
        }
        val mockRegistry = mockk<ProviderRegistry> {
            // First call with "default" throws — parse() can't resolve it
            coEvery { chat("default", any<List<ProviderMessage>>(), any<ChatOptions>()) } throws
                IllegalStateException("No configured providers")
            // Fallback path: configured() returns our mock provider
            every { configured() } returns listOf(mockProvider)
            // The fallback call uses the dynamically-built model id
            coEvery { chat("anthropic:claude-sonnet-4.6", any<List<ProviderMessage>>(), any<ChatOptions>()) } returns flowOf(
                ProviderChunk(text = llmResponse),
                ProviderChunk(finishReason = FinishReason.stop),
            )
        }

        val tool = KnowledgeGraphTool(mockRegistry).tool
        val result = tool.execute(
            call("knowledge_graph_extract", "text" to "Something."),
            ctx(),
        )

        assertTrue("expected Ok, got $result") { result is ToolResult.Ok }
        val json = (result as ToolResult.Ok).output
        assertTrue(json.contains("Fallback"), "should use fallback model and return nodes: $json")
    }

    @Test
    fun `no configured providers returns error instead of crashing`() = runTest {
        // When even the fallback path finds no configured providers,
        // the tool should fail with a clear error rather than crash
        // with a NullPointerException or generic IllegalStateException.
        // This pins the error-message contract callers see.
        val mockRegistry = mockk<ProviderRegistry> {
            coEvery { chat("default", any<List<ProviderMessage>>(), any<ChatOptions>()) } throws
                IllegalStateException("No configured providers")
            every { configured() } returns emptyList()
        }
        val tool = KnowledgeGraphTool(mockRegistry).tool
        val result = tool.execute(
            call("knowledge_graph_extract", "text" to "Hello."),
            ctx(),
        )
        assertTrue("expected Error, got $result") { result is ToolResult.Error }
        assertEquals("extraction_error", (result as ToolResult.Error).code)
        assertTrue(
            "error message should mention the missing-provider condition",
        ) { result.message.contains("No configured providers", ignoreCase = true) }
    }

    @Test
    fun `tool definition has correct metadata`() {
        val tool = KnowledgeGraphTool(mockk()).tool
        assertEquals("knowledge_graph_extract", tool.name)
        assertEquals(com.aura.agent.ToolRisk.READ_ONLY, tool.risk)
        assertTrue(tool.parameters.properties.containsKey("text"))
        assertTrue(tool.parameters.required.contains("text"))
    }

    // -----------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------

    private fun call(name: String, vararg pairs: Pair<String, Any?>): com.aura.agent.ToolCall =
        com.aura.agent.ToolCall(id = "tc1", name = name, arguments = mapOf(*pairs))

    private fun ctx() = com.aura.agent.ToolContext(conversationId = "conv-1")
}
