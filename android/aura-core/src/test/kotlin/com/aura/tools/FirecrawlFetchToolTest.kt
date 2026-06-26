package com.aura.tools

import com.aura.agent.ToolCall
import com.aura.agent.ToolContext
import com.aura.agent.ToolResult
import com.aura.providers.ProviderKeys
import io.mockk.every
import io.mockk.mockk
import kotlinx.coroutines.test.runTest
import okhttp3.MediaType.Companion.toMediaTypeOrNull
import okhttp3.OkHttpClient
import okhttp3.Protocol
import okhttp3.Request
import okhttp3.Response
import okhttp3.ResponseBody.Companion.toResponseBody
import org.junit.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * Tests for [FirecrawlFetchTool].
 *
 * Mocks the OkHttpClient to return controlled Firecrawl API responses.
 * SSRF-guard tests verify that private/local IPs are rejected without
 * making real network calls.
 */
class FirecrawlFetchToolTest {

    // -----------------------------------------------------------------
    // 1. Happy path — valid URL, key configured, valid markdown response
    // -----------------------------------------------------------------

    @Test
    fun `fetch valid URL returns markdown content`() = runTest {
        val providerKeys = mockk<ProviderKeys> {
            every { keyFor("firecrawl") } returns "test-key-123"
        }
        val httpClient = mockHttpClient(body = firecrawlResponse("# Hello\n\nThis is markdown."))
        val tool = FirecrawlFetchTool(httpClient, providerKeys).tool
        val result = tool.execute(
            call("url" to "https://example.com/page"),
            ctx(),
        )
        assertTrue("expected Ok, got $result") { result is ToolResult.Ok }
        assertEquals("# Hello\n\nThis is markdown.", (result as ToolResult.Ok).output)
    }

    @Test
    fun `fetch returns truncated markdown when content exceeds 8000 chars`() = runTest {
        val longContent = "A".repeat(9000)
        val providerKeys = mockk<ProviderKeys> {
            every { keyFor("firecrawl") } returns "test-key-123"
        }
        val httpClient = mockHttpClient(body = firecrawlResponse(longContent))
        val tool = FirecrawlFetchTool(httpClient, providerKeys).tool
        val result = tool.execute(
            call("url" to "https://example.com/long"),
            ctx(),
        )
        assertTrue("expected Ok, got $result") { result is ToolResult.Ok }
        val output = (result as ToolResult.Ok).output
        assertTrue(output.startsWith("A".repeat(8000)))
        assertTrue(output.contains("truncated to 8000 chars"))
        assertEquals(8031, output.length)
    }

    // -----------------------------------------------------------------
    // 2. Missing / invalid arguments
    // -----------------------------------------------------------------

    @Test
    fun `missing url argument returns error`() = runTest {
        val providerKeys = mockk<ProviderKeys> {
            every { keyFor("firecrawl") } returns "key"
        }
        val httpClient = mockk<OkHttpClient>()
        val tool = FirecrawlFetchTool(httpClient, providerKeys).tool
        val result = tool.execute(call(), ctx())
        assertTrue("expected Error, got $result") { result is ToolResult.Error }
        assertEquals("bad_args", (result as ToolResult.Error).code)
    }

    @Test
    fun `missing API key returns error`() = runTest {
        val providerKeys = mockk<ProviderKeys> {
            every { keyFor("firecrawl") } returns null
        }
        val httpClient = mockk<OkHttpClient>()
        val tool = FirecrawlFetchTool(httpClient, providerKeys).tool
        val result = tool.execute(call("url" to "https://example.com"), ctx())
        assertTrue("expected Error, got $result") { result is ToolResult.Error }
        assertEquals("missing_key", (result as ToolResult.Error).code)
    }

    @Test
    fun `empty API key returns error`() = runTest {
        val providerKeys = mockk<ProviderKeys> {
            every { keyFor("firecrawl") } returns ""
        }
        val httpClient = mockk<OkHttpClient>()
        val tool = FirecrawlFetchTool(httpClient, providerKeys).tool
        val result = tool.execute(call("url" to "https://example.com"), ctx())
        assertTrue("expected Error, got $result") { result is ToolResult.Error }
        assertEquals("missing_key", (result as ToolResult.Error).code)
    }

    // -----------------------------------------------------------------
    // 3. SSRF guard — scheme and host checks (no network)
    // -----------------------------------------------------------------

    @Test
    fun `non-http scheme is rejected`() = runTest {
        val providerKeys = mockk<ProviderKeys> {
            every { keyFor("firecrawl") } returns "key"
        }
        val httpClient = mockk<OkHttpClient>()
        val tool = FirecrawlFetchTool(httpClient, providerKeys).tool
        val result = tool.execute(call("url" to "ftp://files.example.com"), ctx())
        assertTrue("expected Error, got $result") { result is ToolResult.Error }
        assertEquals("ssrf_guard", (result as ToolResult.Error).code)
    }

    @Test
    fun `file scheme is rejected`() = runTest {
        val providerKeys = mockk<ProviderKeys> {
            every { keyFor("firecrawl") } returns "key"
        }
        val httpClient = mockk<OkHttpClient>()
        val tool = FirecrawlFetchTool(httpClient, providerKeys).tool
        val result = tool.execute(call("url" to "file:///etc/passwd"), ctx())
        assertTrue("expected Error, got $result") { result is ToolResult.Error }
        assertEquals("ssrf_guard", (result as ToolResult.Error).code)
    }

    @Test
    fun `URL without host is rejected`() = runTest {
        val providerKeys = mockk<ProviderKeys> {
            every { keyFor("firecrawl") } returns "key"
        }
        val httpClient = mockk<OkHttpClient>()
        val tool = FirecrawlFetchTool(httpClient, providerKeys).tool
        val result = tool.execute(call("url" to "https:///path"), ctx())
        assertTrue("expected Error, got $result") { result is ToolResult.Error }
        assertEquals("ssrf_guard", (result as ToolResult.Error).code)
    }

    @Test
    fun `localhost hostname is rejected`() = runTest {
        val providerKeys = mockk<ProviderKeys> {
            every { keyFor("firecrawl") } returns "key"
        }
        val httpClient = mockk<OkHttpClient>()
        val tool = FirecrawlFetchTool(httpClient, providerKeys).tool
        val result = tool.execute(call("url" to "http://localhost:8080/api"), ctx())
        assertTrue("expected Error, got $result") { result is ToolResult.Error }
        assertEquals("ssrf_guard", (result as ToolResult.Error).code)
    }

    @Test
    fun `localhost dot localdomain is rejected`() = runTest {
        val providerKeys = mockk<ProviderKeys> {
            every { keyFor("firecrawl") } returns "key"
        }
        val httpClient = mockk<OkHttpClient>()
        val tool = FirecrawlFetchTool(httpClient, providerKeys).tool
        val result = tool.execute(call("url" to "http://localhost.localdomain"), ctx())
        assertTrue("expected Error, got $result") { result is ToolResult.Error }
        assertEquals("ssrf_guard", (result as ToolResult.Error).code)
    }

    @Test
    fun `loopback IP 127_0_0_1 is rejected`() = runTest {
        val providerKeys = mockk<ProviderKeys> {
            every { keyFor("firecrawl") } returns "key"
        }
        val httpClient = mockk<OkHttpClient>()
        val tool = FirecrawlFetchTool(httpClient, providerKeys).tool
        val result = tool.execute(call("url" to "http://127.0.0.1"), ctx())
        assertTrue("expected Error, got $result") { result is ToolResult.Error }
        assertEquals("ssrf_guard", (result as ToolResult.Error).code)
    }

    @Test
    fun `private IP 10_x_x_x is rejected`() = runTest {
        val providerKeys = mockk<ProviderKeys> {
            every { keyFor("firecrawl") } returns "key"
        }
        val httpClient = mockk<OkHttpClient>()
        val tool = FirecrawlFetchTool(httpClient, providerKeys).tool
        val result = tool.execute(call("url" to "http://10.0.0.1"), ctx())
        assertTrue("expected Error, got $result") { result is ToolResult.Error }
        assertEquals("ssrf_guard", (result as ToolResult.Error).code)
    }

    @Test
    fun `private IP 192_168_x_x is rejected`() = runTest {
        val providerKeys = mockk<ProviderKeys> {
            every { keyFor("firecrawl") } returns "key"
        }
        val httpClient = mockk<OkHttpClient>()
        val tool = FirecrawlFetchTool(httpClient, providerKeys).tool
        val result = tool.execute(call("url" to "http://192.168.1.100"), ctx())
        assertTrue("expected Error, got $result") { result is ToolResult.Error }
        assertEquals("ssrf_guard", (result as ToolResult.Error).code)
    }

    @Test
    fun `link-local IP 169_254_x_x is rejected`() = runTest {
        val providerKeys = mockk<ProviderKeys> {
            every { keyFor("firecrawl") } returns "key"
        }
        val httpClient = mockk<OkHttpClient>()
        val tool = FirecrawlFetchTool(httpClient, providerKeys).tool
        val result = tool.execute(call("url" to "http://169.254.1.1"), ctx())
        assertTrue("expected Error, got $result") { result is ToolResult.Error }
        assertEquals("ssrf_guard", (result as ToolResult.Error).code)
    }

    @Test
    fun `public IP is allowed past SSRF guard`() = runTest {
        val providerKeys = mockk<ProviderKeys> {
            every { keyFor("firecrawl") } returns "test-key-123"
        }
        val httpClient = mockHttpClient(body = firecrawlResponse("public content"))
        val tool = FirecrawlFetchTool(httpClient, providerKeys).tool
        val result = tool.execute(call("url" to "http://93.184.216.34"), ctx())
        assertTrue("expected Ok, got $result") { result is ToolResult.Ok }
        assertEquals("public content", (result as ToolResult.Ok).output)
    }

    // -----------------------------------------------------------------
    // 4. HTTP errors from Firecrawl API
    // -----------------------------------------------------------------

    @Test
    fun `non-200 from Firecrawl returns error`() = runTest {
        val providerKeys = mockk<ProviderKeys> {
            every { keyFor("firecrawl") } returns "test-key-123"
        }
        val httpClient = mockHttpClient(statusCode = 401, body = "Unauthorized")
        val tool = FirecrawlFetchTool(httpClient, providerKeys).tool
        val result = tool.execute(call("url" to "https://example.com"), ctx())
        assertTrue("expected Error, got $result") { result is ToolResult.Error }
        assertTrue((result as ToolResult.Error).message.contains("Firecrawl API HTTP 401"))
    }

    // -----------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------

    private fun call(vararg pairs: Pair<String, Any?>): ToolCall =
        ToolCall(id = "tc1", name = "fetch_url", arguments = mapOf(*pairs))

    private fun ctx() = ToolContext(conversationId = "conv-1")

    /**
     * Returns a mock OkHttpClient that returns the given body/status.
     */
    private fun mockHttpClient(
        statusCode: Int = 200,
        contentType: String = "application/json",
        body: String = "",
    ): OkHttpClient {
        val response = Response.Builder()
            .request(Request.Builder().url("https://api.firecrawl.dev/v1/scrape").build())
            .protocol(Protocol.HTTP_1_1)
            .code(statusCode)
            .message(if (statusCode in 200..399) "OK" else "Error")
            .body(body.toResponseBody(contentType.toMediaTypeOrNull()))
            .build()
        return mockk<OkHttpClient> {
            every { newCall(any()).execute() } returns response
        }
    }

    private fun firecrawlResponse(markdown: String): String = """
        {"data": {"markdown": ${escapeJson(markdown)}}}
    """.trimIndent()

    private fun escapeJson(s: String): String {
        // Simple JSON escaping for test strings
        val escaped = s
            .replace("\\", "\\\\")
            .replace("\"", "\\\"")
            .replace("\n", "\\n")
            .replace("\r", "\\r")
            .replace("\t", "\\t")
        return "\"$escaped\""
    }
}
