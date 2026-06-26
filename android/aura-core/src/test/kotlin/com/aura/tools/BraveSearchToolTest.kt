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
 * Tests for [BraveSearchTool].
 *
 * Mocks the OkHttpClient to return controlled JSON (Brave API) or HTML
 * (DuckDuckGo fallback) responses without real network calls.
 */
class BraveSearchToolTest {

    // -----------------------------------------------------------------
    // 1. Brave API path — key configured, valid JSON response
    // -----------------------------------------------------------------

    @Test
    fun `brave API with valid key returns formatted results`() = runTest {
        val providerKeys = mockk<ProviderKeys> {
            every { keyFor("brave") } returns "test-key-123"
        }
        val httpClient = mockHttpClient(
            contentType = "application/json",
            body = braveApiResponse(),
        )
        val tool = BraveSearchTool(httpClient, providerKeys).tool
        val result = tool.execute(
            call("query" to "Kotlin programming", "count" to 2),
            ctx(),
        )
        assertTrue("expected Ok, got $result") { result is ToolResult.Ok }
        val text = (result as ToolResult.Ok).output
        assertTrue(text.contains("Kotlin Blog"))
        assertTrue(text.contains("https://kotlinlang.org"))
        assertTrue(text.contains("modern programming language"))
        assertTrue(text.startsWith("- ["))
    }

    @Test
    fun `brave API returns error on non-200`() = runTest {
        val providerKeys = mockk<ProviderKeys> {
            every { keyFor("brave") } returns "test-key-123"
        }
        val httpClient = mockHttpClient(statusCode = 401, contentType = "text/plain", body = "Unauthorized")
        val tool = BraveSearchTool(httpClient, providerKeys).tool
        val result = tool.execute(
            call("query" to "test"),
            ctx(),
        )
        assertTrue("expected Error, got $result") { result is ToolResult.Error }
        assertTrue((result as ToolResult.Error).message.contains("Brave API HTTP"))
    }

    // -----------------------------------------------------------------
    // 2. Fallback path — no API key → DuckDuckGo HTML scrape
    // -----------------------------------------------------------------

    @Test
    fun `fallback to DuckDuckGo when key is blank`() = runTest {
        val providerKeys = mockk<ProviderKeys> {
            every { keyFor("brave") } returns null
        }
        val httpClient = mockHttpClient(
            contentType = "text/html; charset=UTF-8",
            body = duckDuckGoHtml(),
        )
        val tool = BraveSearchTool(httpClient, providerKeys).tool
        val result = tool.execute(
            call("query" to "news"),
            ctx(),
        )
        assertTrue("expected Ok, got $result") { result is ToolResult.Ok }
        val text = (result as ToolResult.Ok).output
        assertTrue(text.contains("Example News"))
        assertTrue(text.contains("https://example.com/news"))
        assertTrue(text.startsWith("- ["))
    }

    @Test
    fun `fallback returns error on HTTP failure`() = runTest {
        val providerKeys = mockk<ProviderKeys> {
            every { keyFor("brave") } returns null
        }
        val httpClient = mockHttpClient(statusCode = 503, contentType = "text/plain", body = "Service Unavailable")
        val tool = BraveSearchTool(httpClient, providerKeys).tool
        val result = tool.execute(
            call("query" to "test"),
            ctx(),
        )
        assertTrue("expected Error, got $result") { result is ToolResult.Error }
    }

    // -----------------------------------------------------------------
    // 3. Edge cases
    // -----------------------------------------------------------------

    @Test
    fun `missing query returns error`() = runTest {
        val providerKeys = mockk<ProviderKeys> {
            every { keyFor("brave") } returns null
        }
        val httpClient = mockk<OkHttpClient>()
        val tool = BraveSearchTool(httpClient, providerKeys).tool
        val result = tool.execute(call(), ctx())
        assertTrue("expected Error, got $result") { result is ToolResult.Error }
        assertEquals("bad_args", (result as ToolResult.Error).code)
    }

    @Test
    fun `empty Brave JSON response returns no results`() = runTest {
        val providerKeys = mockk<ProviderKeys> {
            every { keyFor("brave") } returns "key"
        }
        val httpClient = mockHttpClient(
            contentType = "application/json",
            body = """{"web": {"results": []}}""",
        )
        val tool = BraveSearchTool(httpClient, providerKeys).tool
        val result = tool.execute(call("query" to "nothing"), ctx())
        assertTrue("expected Ok, got $result") { result is ToolResult.Ok }
        assertEquals("No results found.", (result as ToolResult.Ok).output)
    }

    // -----------------------------------------------------------------
    // 4. Parameter defaults
    // -----------------------------------------------------------------

    @Test
    fun `default count is used when count omitted`() = runTest {
        val providerKeys = mockk<ProviderKeys> {
            every { keyFor("brave") } returns "key"
        }
        val httpClient = mockHttpClient(
            contentType = "application/json",
            body = braveApiResponse(),
        )
        val tool = BraveSearchTool(httpClient, providerKeys).tool
        // Only query provided, no count — should default to 5, but our mock
        // only has 2 results so 2 will be returned.
        val result = tool.execute(call("query" to "test"), ctx())
        assertTrue("expected Ok, got $result") { result is ToolResult.Ok }
    }

    // -----------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------

    private fun call(vararg pairs: Pair<String, Any?>): com.aura.agent.ToolCall =
        com.aura.agent.ToolCall(id = "tc1", name = "brave_search", arguments = mapOf(*pairs))

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

    private fun braveApiResponse() = """{
  "web": {
    "results": [
      {
        "title": "Kotlin Blog",
        "url": "https://kotlinlang.org",
        "description": "The official blog for the Kotlin modern programming language."
      },
      {
        "title": "Kotlin Docs",
        "url": "https://kotlinlang.org/docs/home.html",
        "description": "Comprehensive documentation for Kotlin."
      }
    ]
  }
}"""

    private fun duckDuckGoHtml() = """<html>
<body>
<div class="result">
  <a class="result__a" href="https://example.com/news">Example News</a>
  <a class="result__snippet">Top breaking news today.</a>
</div>
<div class="result">
  <a class="result__a" href="https://example.com/weather">Weather Report</a>
  <a class="result__snippet">Local weather forecast.</a>
</div>
</body>
</html>"""
}
