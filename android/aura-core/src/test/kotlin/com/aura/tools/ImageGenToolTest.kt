package com.aura.tools

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
 * Tests for [ImageGenTool].
 *
 * Mocks the OkHttpClient to return controlled JSON (OpenAI DALL-E 3) responses
 * without real network calls. Tests both the OpenAI path and the Pollinations.ai
 * free fallback.
 */
class ImageGenToolTest {

    // -----------------------------------------------------------------
    // 1. OpenAI path — key configured, valid JSON response
    // -----------------------------------------------------------------

    @Test
    fun `openai with valid key returns image url`() = runTest {
        val providerKeys = mockk<ProviderKeys> {
            every { keyFor("openai") } returns "test-openai-key"
        }
        val httpClient = mockHttpClient(
            contentType = "application/json",
            body = openAiResponse(),
        )
        val tool = ImageGenTool(httpClient, providerKeys).tool
        val result = tool.execute(
            call("prompt" to "A cat wearing a hat"),
            ctx(),
        )
        assertTrue("expected Ok, got $result") { result is ToolResult.Ok }
        val text = (result as ToolResult.Ok).output
        assertTrue(text.startsWith("https://"), "URL should start with https://, got: $text")
        assertTrue(text.contains("example.com/image"), "URL should contain example.com/image, got: $text")
    }

    // -----------------------------------------------------------------
    // 2. OpenAI fallback — key configured but API returns error
    // -----------------------------------------------------------------

    @Test
    fun `openai error falls back to pollinations`() = runTest {
        val providerKeys = mockk<ProviderKeys> {
            every { keyFor("openai") } returns "test-openai-key"
        }
        val httpClient = mockHttpClient(statusCode = 400, contentType = "text/plain", body = "Bad Request")
        val tool = ImageGenTool(httpClient, providerKeys).tool
        val result = tool.execute(
            call("prompt" to "A cat"),
            ctx(),
        )
        assertTrue("expected Ok, got $result") { result is ToolResult.Ok }
        val text = (result as ToolResult.Ok).output
        assertTrue(text.contains("image.pollinations.ai"), "Should fall back to pollinations, got: $text")
        assertTrue(text.contains("width=1024"), "Should use default width, got: $text")
        assertTrue(text.contains("height=1024"), "Should use default height, got: $text")
        assertTrue(text.contains("nologo=true"), "Should include nologo, got: $text")
    }

    // -----------------------------------------------------------------
    // 3. Pollinations fallback — no OpenAI key configured
    // -----------------------------------------------------------------

    @Test
    fun `pollinations fallback when no openai key`() = runTest {
        val providerKeys = mockk<ProviderKeys> {
            every { keyFor("openai") } returns null
        }
        val httpClient = mockk<OkHttpClient>()
        val tool = ImageGenTool(httpClient, providerKeys).tool
        val result = tool.execute(
            call("prompt" to "A beautiful landscape"),
            ctx(),
        )
        assertTrue("expected Ok, got $result") { result is ToolResult.Ok }
        val text = (result as ToolResult.Ok).output
        assertTrue(text.contains("image.pollinations.ai/prompt/A+beautiful+landscape"),
            "Should generate pollinations URL, got: $text")
    }

    // -----------------------------------------------------------------
    // 4. Custom size parameter
    // -----------------------------------------------------------------

    @Test
    fun `custom size is respected in pollinations url`() = runTest {
        val providerKeys = mockk<ProviderKeys> {
            every { keyFor("openai") } returns null
        }
        val httpClient = mockk<OkHttpClient>()
        val tool = ImageGenTool(httpClient, providerKeys).tool
        val result = tool.execute(
            call("prompt" to "Test", "size" to "1792x1024"),
            ctx(),
        )
        assertTrue("expected Ok, got $result") { result is ToolResult.Ok }
        val text = (result as ToolResult.Ok).output
        assertTrue(text.contains("width=1792"), "Should use custom width, got: $text")
        assertTrue(text.contains("height=1024"), "Should use custom height, got: $text")
    }

    // -----------------------------------------------------------------
    // 5. Missing prompt returns error
    // -----------------------------------------------------------------

    @Test
    fun `missing prompt returns error`() = runTest {
        val providerKeys = mockk<ProviderKeys> {
            every { keyFor("openai") } returns null
        }
        val httpClient = mockk<OkHttpClient>()
        val tool = ImageGenTool(httpClient, providerKeys).tool
        val result = tool.execute(call(), ctx())
        assertTrue("expected Error, got $result") { result is ToolResult.Error }
        assertEquals("bad_args", (result as ToolResult.Error).code)
    }

    // -----------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------

    private fun call(vararg pairs: Pair<String, Any?>): com.aura.agent.ToolCall =
        com.aura.agent.ToolCall(id = "tc1", name = "image_gen", arguments = mapOf(*pairs))

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

    private fun openAiResponse() = """{
  "data": [
    {
      "url": "https://example.com/image/generated-abc123.png"
    }
  ]
}"""
}
