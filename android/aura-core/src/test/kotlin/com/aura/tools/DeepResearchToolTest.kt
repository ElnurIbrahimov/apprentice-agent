package com.aura.tools

import com.aura.agent.ToolResult
import com.aura.providers.ChatOptions
import com.aura.providers.ProviderKeys
import com.aura.providers.ProviderMessage
import com.aura.providers.ProviderRegistry
import com.aura.providers.FinishReason
import com.aura.providers.ProviderChunk
import io.mockk.coEvery
import io.mockk.every
import io.mockk.mockk
import io.mockk.verify
import kotlinx.coroutines.flow.flowOf
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
 * Tests for [DeepResearchTool].
 *
 * Mocks OkHttpClient for search/fetch calls and ProviderRegistry for
 * the synthesis LLM call, so no real network requests are made.
 */
class DeepResearchToolTest {

    // -----------------------------------------------------------------
    // 1. Happy path — Tavily search + Firecrawl fetch + LLM synthesis
    // -----------------------------------------------------------------

    @Test
    fun `deep research returns JSON with answer and citations`() = runTest {
        val providerKeys = mockk<ProviderKeys> {
            every { keyFor("tavily") } returns "tavily-key"
            every { keyFor("firecrawl") } returns "firecrawl-key"
            every { keyFor("brave") } returns null
        }

        val httpClient = mockHttpClient(
            contentType = "application/json",
            body = tavilySearchResponse(),
        )

        val mockRegistry = mockk<ProviderRegistry> {
            val chunks = listOf(
                ProviderChunk(text = "Based on the sources provided, "),
                ProviderChunk(text = "Kotlin is a modern programming language [1]. "),
                ProviderChunk(text = "It runs on the JVM and is fully interoperable with Java [2]."),
                ProviderChunk(finishReason = FinishReason.stop),
            )
            coEvery { chat(any(), any(), any(), any()) } returns flowOf(*chunks.toTypedArray())
        }

        val tool = DeepResearchTool(httpClient, providerKeys, mockRegistry).tool
        val result = tool.execute(
            call("query" to "What is Kotlin?", "max_sources" to 2),
            ctx(),
        )

        assertTrue("expected Ok, got $result") { result is ToolResult.Ok }
        val json = (result as ToolResult.Ok).output
        assertTrue(json.contains("\"answer\""), "should contain answer field: $json")
        assertTrue(json.contains("\"citations\""), "should contain citations field: $json")
        assertTrue(json.contains("Kotlin is a modern programming language"), "should contain synthesized answer: $json")
        assertTrue(json.contains("kotlinlang.org"), "should contain source URLs: $json")
        assertTrue(json.contains("\"index\":1"), "should have citation index 1: $json")
        assertTrue(json.contains("\"index\":2"), "should have citation index 2: $json")
    }

    // -----------------------------------------------------------------
    // 2. Missing query
    // -----------------------------------------------------------------

    @Test
    fun `missing query returns error`() = runTest {
        val providerKeys = mockk<ProviderKeys>()
        val httpClient = mockk<OkHttpClient>()
        val mockRegistry = mockk<ProviderRegistry>()

        val tool = DeepResearchTool(httpClient, providerKeys, mockRegistry).tool
        val result = tool.execute(call(), ctx())

        assertTrue("expected Error, got $result") { result is ToolResult.Error }
        assertEquals("bad_args", (result as ToolResult.Error).code)
    }

    // -----------------------------------------------------------------
    // 3. No search results
    // -----------------------------------------------------------------

    @Test
    fun `empty search results returns no results JSON`() = runTest {
        val providerKeys = mockk<ProviderKeys> {
            every { keyFor("tavily") } returns "tavily-key"
            every { keyFor("firecrawl") } returns null
            every { keyFor("brave") } returns null
        }

        val httpClient = mockHttpClient(
            contentType = "application/json",
            body = """{"results": []}""",
        )

        val mockRegistry = mockk<ProviderRegistry>()

        val tool = DeepResearchTool(httpClient, providerKeys, mockRegistry).tool
        val result = tool.execute(call("query" to "nothing"), ctx())

        assertTrue("expected Ok, got $result") { result is ToolResult.Ok }
        val json = (result as ToolResult.Ok).output
        assertTrue(json.contains("\"answer\""), "should contain answer field: $json")
        assertTrue(json.contains("\"citations\":[]"), "should have empty citations: $json")
    }

    // -----------------------------------------------------------------
    // 4. Tavily HTTP error
    // -----------------------------------------------------------------

    @Test
    fun `search HTTP error returns error`() = runTest {
        val providerKeys = mockk<ProviderKeys> {
            every { keyFor("tavily") } returns "tavily-key"
            every { keyFor("firecrawl") } returns null
            every { keyFor("brave") } returns null
        }

        val httpClient = mockHttpClient(
            statusCode = 401,
            contentType = "text/plain",
            body = "Unauthorized",
        )

        val mockRegistry = mockk<ProviderRegistry>()

        val tool = DeepResearchTool(httpClient, providerKeys, mockRegistry).tool
        val result = tool.execute(call("query" to "test"), ctx())

        assertTrue("expected Error, got $result") { result is ToolResult.Error }
        assertTrue((result as ToolResult.Error).message.contains("Tavily API HTTP"))
    }

    // -----------------------------------------------------------------
    // 5. Brave search fallback (no Tavily key)
    // -----------------------------------------------------------------

    @Test
    fun `falls back to Brave when Tavily key missing`() = runTest {
        val providerKeys = mockk<ProviderKeys> {
            every { keyFor("tavily") } returns null
            every { keyFor("brave") } returns "brave-key"
            every { keyFor("firecrawl") } returns null
        }

        val httpClient = mockHttpClient(
            contentType = "application/json",
            body = braveSearchResponse(),
        )

        val mockRegistry = mockk<ProviderRegistry> {
            val chunks = listOf(
                ProviderChunk(text = "Brave search found Kotlin [1]."),
                ProviderChunk(finishReason = FinishReason.stop),
            )
            coEvery { chat(any(), any(), any(), any()) } returns flowOf(*chunks.toTypedArray())
        }

        val tool = DeepResearchTool(httpClient, providerKeys, mockRegistry).tool
        val result = tool.execute(call("query" to "Kotlin"), ctx())

        assertTrue("expected Ok, got $result") { result is ToolResult.Ok }
        val json = (result as ToolResult.Ok).output
        assertTrue(json.contains("\"answer\""), "should contain answer field: $json")
        assertTrue(json.contains("Brave search found Kotlin"), "should contain synthesized answer: $json")
        assertTrue(json.contains("\"citations\""), "should contain citations field: $json")
    }

    // -----------------------------------------------------------------
    // 6. Default parameters (max_sources defaults to 5)
    // -----------------------------------------------------------------

    @Test
    fun `default max_sources is used when omitted`() = runTest {
        val providerKeys = mockk<ProviderKeys> {
            every { keyFor("tavily") } returns "tavily-key"
            every { keyFor("firecrawl") } returns null
            every { keyFor("brave") } returns null
        }

        val httpClient = mockHttpClient(
            contentType = "application/json",
            body = tavilySearchResponse(),
        )

        val mockRegistry = mockk<ProviderRegistry> {
            val chunks = listOf(
                ProviderChunk(text = "Result."),
                ProviderChunk(finishReason = FinishReason.stop),
            )
            coEvery { chat(any(), any(), any(), any()) } returns flowOf(*chunks.toTypedArray())
        }

        val tool = DeepResearchTool(httpClient, providerKeys, mockRegistry).tool
        // Only query, no max_sources — should default to 5
        val result = tool.execute(call("query" to "test"), ctx())

        assertTrue("expected Ok, got $result") { result is ToolResult.Ok }
        val json = (result as ToolResult.Ok).output
        assertTrue(json.contains("\"answer\""), "should contain answer field: $json")
    }

    @Test
    fun `private search result is never forwarded to Firecrawl`() = runTest {
        val providerKeys = mockk<ProviderKeys> {
            every { keyFor("tavily") } returns "tavily-key"
            every { keyFor("firecrawl") } returns "firecrawl-key"
        }
        val httpClient = mockHttpClient(
            contentType = "application/json",
            body = """{"results":[{"title":"internal","url":"http://127.0.0.1/private","content":"nope"}]}""",
        )
        val registry = mockk<ProviderRegistry> {
            coEvery { chat(any(), any(), any(), any()) } returns flowOf(
                ProviderChunk(text = "Safe answer."),
                ProviderChunk(finishReason = FinishReason.stop),
            )
        }

        val result = DeepResearchTool(httpClient, providerKeys, registry).tool.execute(call("query" to "test"), ctx())

        assertTrue(result is ToolResult.Ok)
        verify(exactly = 1) { httpClient.newCall(any()).execute() }
    }

    // -----------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------

    private fun call(vararg pairs: Pair<String, Any?>): com.aura.agent.ToolCall =
        com.aura.agent.ToolCall(id = "tc1", name = "deep_research", arguments = mapOf(*pairs))

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

    private fun tavilySearchResponse(): String = """{
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
    },
    {
      "title": "Kotlin GitHub",
      "url": "https://github.com/JetBrains/kotlin",
      "content": "Kotlin compiler and standard library source code."
    }
  ]
}"""

    private fun braveSearchResponse(): String = """{
  "web": {
    "results": [
      {
        "title": "Kotlin Programming Language",
        "url": "https://kotlinlang.org",
        "description": "Kotlin is a modern, cross-platform programming language."
      },
      {
        "title": "Kotlin - Wikipedia",
        "url": "https://en.wikipedia.org/wiki/Kotlin",
        "description": "Kotlin is a cross-platform, statically typed language."
      }
    ]
  }
}"""
}
