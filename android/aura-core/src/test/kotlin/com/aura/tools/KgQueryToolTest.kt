package com.aura.tools

import com.aura.agent.ToolResult
import com.aura.kg.EdgeType
import com.aura.kg.KgEdge
import com.aura.kg.KgNode
import com.aura.kg.KnowledgeGraphRepository
import com.aura.kg.NodeType
import io.mockk.coEvery
import io.mockk.mockk
import kotlinx.coroutines.test.runTest
import org.junit.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * Tests for [KgQueryTool].
 *
 * Mocks KnowledgeGraphRepository so no real database calls are made.
 */
class KgQueryToolTest {

    @Test
    fun `search by exact label returns node detail`() = runTest {
        val node = KgNode(id = "1", label = "Kotlin", type = NodeType.SKILL)
        val repo = mockk<KnowledgeGraphRepository> {
            coEvery { getNodeByLabel("Kotlin") } returns node
            coEvery { getNeighbors("1") } returns KnowledgeGraphRepository.Neighbors(
                incoming = emptyList(),
                outgoing = listOf(
                    KgEdge(id = "e1", type = EdgeType.USES, sourceId = "1", targetId = "2"),
                ),
            )
            coEvery { getNode("2") } returns KgNode(id = "2", label = "Android", type = NodeType.PROJECT)
        }

        val tool = KgQueryTool(repo).tool
        val result = tool.execute(
            call("kg_query", "query" to "Kotlin"),
            ctx(),
        )

        assertTrue("expected Ok, got $result") { result is ToolResult.Ok }
        val text = (result as ToolResult.Ok).output
        assertTrue(text.contains("Kotlin"), "should contain label: $text")
        assertTrue(text.contains("skill"), "should contain type: $text")
        assertTrue(text.contains("Android"), "should contain neighbor: $text")
    }

    @Test
    fun `search by 'what do I know about X' works`() = runTest {
        val node = KgNode(id = "3", label = "Aura", type = NodeType.PROJECT)
        val repo = mockk<KnowledgeGraphRepository> {
            coEvery { getNodeByLabel("Aura") } returns node
            coEvery { getNeighbors("3") } returns KnowledgeGraphRepository.Neighbors(
                incoming = emptyList(),
                outgoing = emptyList(),
            )
        }

        val tool = KgQueryTool(repo).tool
        val result = tool.execute(
            call("kg_query", "query" to "what do I know about Aura"),
            ctx(),
        )

        assertTrue("expected Ok, got $result") { result is ToolResult.Ok }
        val text = (result as ToolResult.Ok).output
        assertTrue(text.contains("Aura"), "should contain label: $text")
        assertTrue(text.contains("project"), "should contain type: $text")
    }

    @Test
    fun `search by 'show X' works`() = runTest {
        val node = KgNode(id = "4", label = "Android", type = NodeType.PROJECT)
        val repo = mockk<KnowledgeGraphRepository> {
            coEvery { getNodeByLabel("Android") } returns node
            coEvery { getNeighbors("4") } returns KnowledgeGraphRepository.Neighbors(
                incoming = emptyList(),
                outgoing = emptyList(),
            )
        }

        val tool = KgQueryTool(repo).tool
        val result = tool.execute(
            call("kg_query", "query" to "show Android"),
            ctx(),
        )

        assertTrue("expected Ok, got $result") { result is ToolResult.Ok }
        val text = (result as ToolResult.Ok).output
        assertTrue(text.contains("Android"), "should contain label: $text")
    }

    @Test
    fun `path between X and Y returns path`() = runTest {
        val fromNode = KgNode(id = "a", label = "Kotlin", type = NodeType.SKILL)
        val toNode = KgNode(id = "c", label = "App", type = NodeType.CONCEPT)
        val midNode = KgNode(id = "b", label = "Android", type = NodeType.PROJECT)

        val repo = mockk<KnowledgeGraphRepository> {
            coEvery { getNodeByLabel("Kotlin") } returns fromNode
            coEvery { getNodeByLabel("App") } returns toNode
            coEvery { findPath("a", "c") } returns listOf("a", "b", "c")
            coEvery { getNode("a") } returns fromNode
            coEvery { getNode("b") } returns midNode
            coEvery { getNode("c") } returns toNode
            coEvery { getNeighbors("a") } returns KnowledgeGraphRepository.Neighbors(
                incoming = emptyList(),
                outgoing = listOf(
                    KgEdge(id = "e1", type = EdgeType.USES, sourceId = "a", targetId = "b"),
                ),
            )
            coEvery { getNeighbors("b") } returns KnowledgeGraphRepository.Neighbors(
                incoming = emptyList(),
                outgoing = listOf(
                    KgEdge(id = "e2", type = EdgeType.PART_OF, sourceId = "b", targetId = "c"),
                ),
            )
        }

        val tool = KgQueryTool(repo).tool
        val result = tool.execute(
            call("kg_query", "query" to "path between Kotlin and App"),
            ctx(),
        )

        assertTrue("expected Ok, got $result") { result is ToolResult.Ok }
        val text = (result as ToolResult.Ok).output
        assertTrue(text.contains("Kotlin"), "should contain from node: $text")
        assertTrue(text.contains("Android"), "should contain mid node: $text")
        assertTrue(text.contains("App"), "should contain to node: $text")
        assertTrue(text.contains("uses"), "should contain edge type: $text")
    }

    @Test
    fun `path between with missing nodes returns helpful message`() = runTest {
        val repo = mockk<KnowledgeGraphRepository> {
            coEvery { getNodeByLabel("MISSING") } returns null
            coEvery { getNodeByLabel("Nowhere") } returns null
        }

        val tool = KgQueryTool(repo).tool
        val result = tool.execute(
            call("kg_query", "query" to "path between MISSING and Nowhere"),
            ctx(),
        )

        assertTrue("expected Ok, got $result") { result is ToolResult.Ok }
        val text = (result as ToolResult.Ok).output
        assertTrue(text.contains("couldn't find"), "should say not found: $text")
    }

    @Test
    fun `search with no results returns empty message`() = runTest {
        val repo = mockk<KnowledgeGraphRepository> {
            coEvery { getNodeByLabel("zzz_nonexistent") } returns null
            coEvery { search("zzz_nonexistent") } returns emptyList()
        }

        val tool = KgQueryTool(repo).tool
        val result = tool.execute(
            call("kg_query", "query" to "zzz_nonexistent"),
            ctx(),
        )

        assertTrue("expected Ok, got $result") { result is ToolResult.Ok }
        val text = (result as ToolResult.Ok).output
        assertTrue(text.contains("No results found"), "should say no results: $text")
    }

    @Test
    fun `missing query returns error`() = runTest {
        val repo = mockk<KnowledgeGraphRepository>()
        val tool = KgQueryTool(repo).tool
        val result = tool.execute(call("kg_query"), ctx())

        assertTrue("expected Error, got $result") { result is ToolResult.Error }
        assertEquals("bad_args", (result as ToolResult.Error).code)
    }

    @Test
    fun `tool definition has correct metadata`() {
        val tool = KgQueryTool(mockk()).tool
        assertEquals("kg_query", tool.name)
        assertEquals(com.aura.agent.ToolRisk.READ_ONLY, tool.risk)
        assertTrue(tool.parameters.properties.containsKey("query"))
        assertTrue(tool.parameters.required.contains("query"))
    }

    @Test
    fun `search with multiple results returns table`() = runTest {
        val repo = mockk<KnowledgeGraphRepository> {
            coEvery { getNodeByLabel("test") } returns null
            coEvery { search("test") } returns listOf(
                KgNode(id = "1", label = "TestNode1", type = NodeType.CONCEPT),
                KgNode(id = "2", label = "TestNode2", type = NodeType.ENTITY),
            )
        }

        val tool = KgQueryTool(repo).tool
        val result = tool.execute(
            call("kg_query", "query" to "test"),
            ctx(),
        )

        assertTrue("expected Ok, got $result") { result is ToolResult.Ok }
        val text = (result as ToolResult.Ok).output
        assertTrue(text.contains("TestNode1"), "should contain first result: $text")
        assertTrue(text.contains("TestNode2"), "should contain second result: $text")
        assertTrue(text.contains("| # |"), "should be table format: $text")
    }

    // -----------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------

    private fun call(name: String, vararg pairs: Pair<String, Any?>): com.aura.agent.ToolCall =
        com.aura.agent.ToolCall(id = "tc1", name = name, arguments = mapOf(*pairs))

    private fun ctx() = com.aura.agent.ToolContext(conversationId = "conv-1")
}
