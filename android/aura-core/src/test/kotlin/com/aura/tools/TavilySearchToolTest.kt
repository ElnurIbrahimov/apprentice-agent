package com.aura.tools

import com.aura.agent.ToolResult
import com.aura.providers.ProviderKeys
import io.mockk.every
import io.mockk.mockk
import kotlinx.coroutines.test.runTest
import okhttp3.OkHttpClient
import okhttp3.Protocol
import okhttp3.Request
import okhttp3.Response
import okhttp3.ResponseBody.Companion.toResponseBody
import okhttp3.MediaType.Companion.toMediaTypeOrNull
import org.junit.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * Tests for [TavilySearchTool].
 *
 * Mocks the OkHttpClient to return controlled Tavily API JSON responses
 * without real network calls.
 */
class TavilySearchToolTest {

    // -----------------------------------------------------------------
    // 1. Tavily API path — key configured, valid JSON response
    // -----------------------------------------------------------------

    @Test
    fun `tavily API with valid key returns formatted results with answer`() = runTest {
        val providerKeys = mockk<ProviderKeys> {
            every { keyFor("tavily") } returns "test-key-123"
        }
        val httpClient = mockHttpClient(
            contentType = "application/json",
            body = tavilyApiResponse(includeAnswer = true),
        )
        val tool = TavilySearchTool(httpClient, providerKeys).tool
        val result = tool.execute(
            call("query" to "Kotlin programming", "max_results" to 2, "include_answer" to true),
            ctx(),
        )
        assertTrue("expected Ok, got $result") { result is ToolResult.Ok }
        val text = (result as ToolResult.Ok).output
        // Answer should be at the top
        assertTrue(text.contains("Kotlin is a modern programming language"), "answer missing: $text")
        assertTrue(text.contains("Kotlin Blog"), "citation missing: $text")
        assertTrue(text.contains("https://kotlinlang.org"), "URL missing: $text")
        assertTrue(text.startsWith("Kotlin"), "should start with answer text")
    }

    @Test
    fun `tavily API without answer skips answer section`() = runTest {
        val providerKeys = mockk<ProviderKeys> {
            every { keyFor("tavily") } returns "test-key-123"
        }
        val httpClient = mockHttpClient(
            contentType = "application/json",
            body = tavilyApiResponse(includeAnswer = false),
        )
        val tool = TavilySearchTool(httpClient, providerKeys).tool
        val result = tool.execute(
            call("query" to "Kotlin", "include_answer" to false),
            ctx(),
        )
        assertTrue("expected Ok, got $result") { result is ToolResult.Ok }
        val text = (result as ToolResult.Ok).output
        // Answer should NOT be present
        assertTrue(!text.contains("Kotlin is a modern"), "answer should not appear: $text")
        assertTrue(text.contains("Kotlin Blog"), "citation missing: $text")
        assertTrue(text.startsWith("- ["), "should start with markdown list")
    }

    @Test
    fun `tavily API returns error on non-200`() = runTest {
        val providerKeys = mockk<ProviderKeys> {
            every { keyFor("tavily") } returns "test-key-123"
        }
        val httpClient = mockHttpClient(statusCode = 401, contentType = "text/plain", body = "Unauthorized")
        val tool = TavilySearchTool(httpClient, providerKeys).tool
        val result = tool.execute(
            call("query" to "test"),
            ctx(),
        )
        assertTrue("expected Error, got $result") { result is ToolResult.Error }
        assertTrue((result as ToolResult.Error).message.contains("Tavily API HTTP"))
    }

    // -----------------------------------------------------------------
    // 2. Missing key case
    // -----------------------------------------------------------------

    @Test
    fun `missing API key returns error`() = runTest {
        val providerKeys = mockk<ProviderKeys> {
            every { keyFor("tavily") } returns null
        }
        val httpClient = mockk<OkHttpClient>()
        val tool = TavilySearchTool(httpClient, providerKeys).tool
        val result = tool.execute(
            call("query" to "test"),
            ctx(),
        )
        assertTrue("expected Error, got $result") { result is ToolResult.Error }
        assertEquals("missing_key", (result as ToolResult.Error).code)
    }

    // -----------------------------------------------------------------
    // 3. Edge cases
    // -----------------------------------------------------------------

    @Test
    fun `missing query returns error`() = runTest {
        val providerKeys = mockk<ProviderKeys> {
            every { keyFor("tavily") } returns "key"
        }
        val httpClient = mockk<OkHttpClient>()
        val tool = TavilySearchTool(httpClient, providerKeys).tool
        val result = tool.execute(call(), ctx())
        assertTrue("expected Error, got $result") { result is ToolResult.Error }
        assertEquals("bad_args", (result as ToolResult.Error).code)
    }

    @Test
    fun `invalid search_depth returns error`() = runTest {
        val providerKeys = mockk<ProviderKeys> {
            every { keyFor("tavily") } returns "key"
        }
        val httpClient = mockk<OkHttpClient>()
        val tool = TavilySearchTool(httpClient, providerKeys).tool
        val result = tool.execute(
            call("query" to "test", "search_depth" to "invalid"),
            ctx(),
        )
        assertTrue("expected Error, got $result") { result is ToolResult.Error }
        assertEquals("bad_args", (result as ToolResult.Error).code)
    }

    @Test
    fun `empty results returns no results message`() = runTest {
        val providerKeys = mockk<ProviderKeys> {
            every { keyFor("tavily") } returns "key"
        }
        val httpClient = mockHttpClient(
            contentType = "application/json",
            body = """{"results": [], "answer": null}""",
        )
        val tool = TavilySearchTool(httpClient, providerKeys).tool
        val result = tool.execute(call("query" to "nothing"), ctx())
        assertTrue("expected Ok, got $result") { result is ToolResult.Ok }
        assertEquals("No results found.", (result as ToolResult.Ok).output)
    }

    // -----------------------------------------------------------------
    // 4. Parameter defaults
    // -----------------------------------------------------------------

    @Test
    fun `default parameters are used when omitted`() = runTest {
        val providerKeys = mockk<ProviderKeys> {
            every { keyFor("tavily") } returns "key"
        }
        val httpClient = mockHttpClient(
            contentType = "application/json",
            body = tavilyApiResponse(includeAnswer = true),
        )
        val tool = TavilySearchTool(httpClient, providerKeys).tool
        // Only query provided — should use defaults: max_results=5, depth=basic, include_answer=true
        val result = tool.execute(call("query" to "test"), ctx())
        assertTrue("expected Ok, got $result") { result is ToolResult.Ok }
        val text = (result as ToolResult.Ok).output
        assertTrue(text.contains("Kotlin is a modern programming language"))
    }

    // -----------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------

    private fun call(vararg pairs: Pair<String, Any?>): com.aura.agent.ToolCall =
        com.aura.agent.ToolCall(id = "tc1", name = "tavily_search", arguments = mapOf(*pairs))

    private fun ctx() = com.aura.agent.ToolContext(conversationId = "conv-1")

    /**
     * Returns a mock OkHttpClient that returns the given body/status.
     */
    private fun mockHttpClient(
        statusCode: Int = 200,
        contentType: String = "application/json",
        body: String = "",
    ): OkHttpClient {
        val response = Response.Builder()
            .request(Request.Builder().url("https://example.com").build())
            .protocol(Protocol.HTTP_1_1)
            .code(statusCode)
            .message(if (statusCode in 200..399) "OK" else "Error")
            .body(body.toResponseBody(contentType.toMediaTypeOrNull()))
            .build()
        return mockk<OkHttpClient> {
            every { newCall(any()).execute() } returns response
        }
    }

    private fun tavilyApiResponse(includeAnswer: Boolean): String {
        val answerField = if (includeAnswer) {
            """"answer": "Kotlin is a modern programming language that runs on the JVM.", """
        } else {
            """"answer": null, """
        }
        return """{
  $answerField
  "results": [
    {
      "title": "Kotlin Blog",
      "url": "https://kotlinlang.org",
      "content": "The official blog for the Kotlin programming language."
    },
    {
      "title": "Kotlin Docs",
      "url": "https://kotlinlang.org/docs/home.html",
      "content": "Comprehensive documentation for Kotlin."
    }
  ]
}"""
    }
}
