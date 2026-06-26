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
 * Tests for [VisionTool].
 *
 * Mocks the OkHttpClient to return controlled JSON responses (Gemini, OpenAI,
 * Ollama Cloud) without real network calls.
 */
class VisionToolTest {

    // A small valid-looking base64 string (~150 bytes decoded)
    private val smallBase64 = "/9j/4AAQSkZJRgABAQEASABIAAD/2wBDAAg=="

    // A base64 string that would decode to >2MB (2,097,153 bytes)
    // 2MB = 2,097,152 bytes → need (len * 3) / 4 > 2,097,152
    // To be safe: len = 2,796,500 → (2,796,500 * 3) / 4 = 2,097,375 > 2,097,152
    private val largeBase64 = "A".repeat(2_796_500)

    // -----------------------------------------------------------------
    // 1. Gemini path — key configured, valid response
    // -----------------------------------------------------------------

    @Test
    fun `gemini with valid key returns description`() = runTest {
        val providerKeys = mockk<ProviderKeys> {
            every { keyFor("gemini") } returns "gemini-key-123"
            every { keyFor("openai") } returns null
            every { keyFor("ollama") } returns null
        }
        val httpClient = mockHttpClient(body = geminiResponse())
        val tool = VisionTool(httpClient, providerKeys).tool
        val result = tool.execute(
            call("image_base64" to smallBase64, "prompt" to "What is in this image?"),
            ctx(),
        )
        assertTrue("expected Ok, got $result") { result is ToolResult.Ok }
        val text = (result as ToolResult.Ok).output
        assertTrue(text.contains("sunset over mountains"), "missing expected text: $text")
    }

    @Test
    fun `gemini with default prompt works`() = runTest {
        val providerKeys = mockk<ProviderKeys> {
            every { keyFor("gemini") } returns "gemini-key-123"
            every { keyFor("openai") } returns null
            every { keyFor("ollama") } returns null
        }
        val httpClient = mockHttpClient(body = geminiResponse())
        val tool = VisionTool(httpClient, providerKeys).tool
        val result = tool.execute(
            call("image_base64" to smallBase64),
            ctx(),
        )
        assertTrue("expected Ok, got $result") { result is ToolResult.Ok }
        val text = (result as ToolResult.Ok).output
        assertTrue(text.contains("sunset"), "missing expected text: $text")
    }

    @Test
    fun `gemini returns error on non-200`() = runTest {
        val providerKeys = mockk<ProviderKeys> {
            every { keyFor("gemini") } returns "gemini-key-123"
        }
        val httpClient = mockHttpClient(
            statusCode = 401,
            contentType = "text/plain",
            body = "Unauthorized",
        )
        val tool = VisionTool(httpClient, providerKeys).tool
        val result = tool.execute(
            call("image_base64" to smallBase64),
            ctx(),
        )
        assertTrue("expected Error, got $result") { result is ToolResult.Error }
        assertTrue((result as ToolResult.Error).message.contains("Gemini API HTTP"))
    }

    // -----------------------------------------------------------------
    // 2. OpenAI path — Gemini not configured, OpenAI configured
    // -----------------------------------------------------------------

    @Test
    fun `openai with valid key returns description`() = runTest {
        val providerKeys = mockk<ProviderKeys> {
            every { keyFor("gemini") } returns null
            every { keyFor("openai") } returns "sk-openai-123"
            every { keyFor("ollama") } returns null
        }
        val httpClient = mockHttpClient(body = openAiResponse())
        val tool = VisionTool(httpClient, providerKeys).tool
        val result = tool.execute(
            call("image_base64" to smallBase64, "prompt" to "Describe this image"),
            ctx(),
        )
        assertTrue("expected Ok, got $result") { result is ToolResult.Ok }
        val text = (result as ToolResult.Ok).output
        assertTrue(text.contains("beach at sunset"), "missing expected text: $text")
    }

    @Test
    fun `openai returns error on non-200`() = runTest {
        val providerKeys = mockk<ProviderKeys> {
            every { keyFor("gemini") } returns null
            every { keyFor("openai") } returns "sk-openai-123"
        }
        val httpClient = mockHttpClient(
            statusCode = 429,
            contentType = "text/plain",
            body = "Rate limit exceeded",
        )
        val tool = VisionTool(httpClient, providerKeys).tool
        val result = tool.execute(
            call("image_base64" to smallBase64),
            ctx(),
        )
        assertTrue("expected Error, got $result") { result is ToolResult.Error }
        assertTrue((result as ToolResult.Error).message.contains("OpenAI API HTTP"))
    }

    // -----------------------------------------------------------------
    // 3. Ollama Cloud path — only Ollama configured
    // -----------------------------------------------------------------

    @Test
    fun `ollama with valid key returns description`() = runTest {
        val providerKeys = mockk<ProviderKeys> {
            every { keyFor("gemini") } returns null
            every { keyFor("openai") } returns null
            every { keyFor("ollama") } returns "ollama-key-123"
        }
        val httpClient = mockHttpClient(body = ollamaResponse())
        val tool = VisionTool(httpClient, providerKeys).tool
        val result = tool.execute(
            call("image_base64" to smallBase64, "prompt" to "Analyze"),
            ctx(),
        )
        assertTrue("expected Ok, got $result") { result is ToolResult.Ok }
        val text = (result as ToolResult.Ok).output
        assertTrue(text.contains("mountain landscape"), "missing expected text: $text")
    }

    // -----------------------------------------------------------------
    // 4. No vision provider configured
    // -----------------------------------------------------------------

    @Test
    fun `no provider configured returns error`() = runTest {
        val providerKeys = mockk<ProviderKeys> {
            every { keyFor("gemini") } returns null
            every { keyFor("openai") } returns null
            every { keyFor("ollama") } returns null
        }
        val httpClient = mockk<OkHttpClient>()
        val tool = VisionTool(httpClient, providerKeys).tool
        val result = tool.execute(
            call("image_base64" to smallBase64),
            ctx(),
        )
        assertTrue("expected Error, got $result") { result is ToolResult.Error }
        assertTrue((result as ToolResult.Error).message.contains("No vision-capable provider configured"))
    }

    // -----------------------------------------------------------------
    // 5. Size limit
    // -----------------------------------------------------------------

    @Test
    fun `image over 2MB returns error`() = runTest {
        val providerKeys = mockk<ProviderKeys> {
            every { keyFor("gemini") } returns "key"
        }
        val httpClient = mockk<OkHttpClient>()
        val tool = VisionTool(httpClient, providerKeys).tool
        val result = tool.execute(
            call("image_base64" to largeBase64),
            ctx(),
        )
        assertTrue("expected Error, got $result") { result is ToolResult.Error }
        val err = result as ToolResult.Error
        assertEquals("image_too_large", err.code)
        assertTrue(err.message.contains("2 MB"), "should mention size limit: ${err.message}")
    }

    @Test
    fun `image just under 2MB is accepted`() = runTest {
        // 2MB = 2,097,152 bytes → need base64 length ≤ (2,097,152 * 4) / 3 ≈ 2,796,202
        // Use 2,796,000 which gives estimatedBytes ≈ (2,796,000 * 3) / 4 = 2,097,000 < 2MB
        val justUnderBase64 = "A".repeat(2_796_000)

        val providerKeys = mockk<ProviderKeys> {
            every { keyFor("gemini") } returns "gemini-key-123"
            every { keyFor("openai") } returns null
            every { keyFor("ollama") } returns null
        }
        val httpClient = mockHttpClient(body = geminiResponse())
        val tool = VisionTool(httpClient, providerKeys).tool
        val result = tool.execute(
            call("image_base64" to justUnderBase64),
            ctx(),
        )
        // Should pass size check and attempt API call (which is mocked)
        assertTrue("expected Ok, got $result") { result is ToolResult.Ok }
    }

    // -----------------------------------------------------------------
    // 6. Edge cases
    // -----------------------------------------------------------------

    @Test
    fun `missing image_base64 returns error`() = runTest {
        val providerKeys = mockk<ProviderKeys>()
        val httpClient = mockk<OkHttpClient>()
        val tool = VisionTool(httpClient, providerKeys).tool
        val result = tool.execute(call(), ctx())
        assertTrue("expected Error, got $result") { result is ToolResult.Error }
        assertEquals("bad_args", (result as ToolResult.Error).code)
    }

    @Test
    fun `empty base64 string returns error for Gemini`() = runTest {
        val providerKeys = mockk<ProviderKeys> {
            every { keyFor("gemini") } returns "key"
        }
        val httpClient = mockHttpClient(
            statusCode = 400,
            contentType = "application/json",
            body = """{"error": {"message": "Invalid request"}}""",
        )
        val tool = VisionTool(httpClient, providerKeys).tool
        val result = tool.execute(
            call("image_base64" to ""),
            ctx(),
        )
        assertTrue("expected Error, got $result") { result is ToolResult.Error }
    }

    // -----------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------

    private fun call(vararg pairs: Pair<String, Any?>): com.aura.agent.ToolCall =
        com.aura.agent.ToolCall(id = "tc1", name = "vision", arguments = mapOf(*pairs))

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

    // ---- Mock response bodies ----

    private fun geminiResponse() = """{
  "candidates": [
    {
      "content": {
        "parts": [
          { "text": "A beautiful sunset over mountains with a lake reflecting the orange sky." }
        ]
      },
      "finishReason": "STOP"
    }
  ]
}"""

    private fun openAiResponse() = """{
  "choices": [
    {
      "message": {
        "content": "A serene beach at sunset with gentle waves lapping the shore.",
        "role": "assistant"
      }
    }
  ]
}"""

    private fun ollamaResponse() = """{
  "choices": [
    {
      "message": {
        "content": "A stunning mountain landscape covered in snow with clear blue sky.",
        "role": "assistant"
      }
    }
  ]
}"""
}
