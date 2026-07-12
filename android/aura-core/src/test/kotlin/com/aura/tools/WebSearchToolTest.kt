package com.aura.tools

import com.aura.agent.ToolCall
import com.aura.agent.ToolContext
import com.aura.agent.ToolResult
import com.aura.agent.ToolRisk
import okhttp3.OkHttpClient
import org.junit.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * Tests for [WebSearchTool] — argument validation, tool definition,
 * and risk classification. The actual DDG search is tested via
 * integration tests (requires network); here we cover the contract.
 */
class WebSearchToolTest {

    private val client = OkHttpClient.Builder().build()
    private val tool = WebSearchTool(client)

    private fun exec(args: Map<String, Any?>): ToolResult =
        kotlinx.coroutines.runBlocking {
            tool.tool.execute(
                ToolCall(id = "", name = "web_search", arguments = args),
                ToolContext(conversationId = "test"),
            )
        }

    @Test
    fun `missing query returns bad_args error`() {
        val result = exec(emptyMap())
        assertTrue(result is ToolResult.Error)
        assertEquals("bad_args", (result as ToolResult.Error).code)
    }

    @Test
    fun `query with null value returns bad_args error`() {
        val result = exec(mapOf("query" to null))
        assertTrue(result is ToolResult.Error)
    }

    @Test
    fun `tool definition has correct name and required query`() {
        val def = tool.definition()
        assertEquals("web_search", def.name)
        assertEquals("query", def.parameters.required.first())
    }

    @Test
    fun `tool risk is READ_ONLY`() {
        assertEquals(ToolRisk.READ_ONLY, tool.tool.risk)
    }
}