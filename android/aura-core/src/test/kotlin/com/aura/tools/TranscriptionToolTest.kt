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
 * Tests for [TranscriptionTool].
 *
 * Mocks the OkHttpClient to return controlled JSON responses (OpenAI Whisper,
 * Groq Whisper) without real network calls.
 */
class TranscriptionToolTest {

    // A small valid base64 string (~12 bytes decoded: "Hello World!")
    private val smallBase64 = java.util.Base64.getEncoder().encodeToString("Hello World!".toByteArray())

    // A base64 string that would decode to >25MB (26,214,400 bytes)
    // 25MB = 26,214,400 bytes → need (len * 3) / 4 > 26,214,400
    // len = 34,952,535 → (34,952,535 * 3) / 4 = 26,214,401 > 26,214,400
    private val largeBase64 = "A".repeat(34_952_535)

    // -----------------------------------------------------------------
    // 1. OpenAI Whisper path — key configured, valid response
    // -----------------------------------------------------------------

    @Test
    fun `openai whisper with valid key returns transcription`() = runTest {
        val providerKeys = mockk<ProviderKeys> {
            every { keyFor("openai") } returns "openai-key-123"
            every { keyFor("groq") } returns null
        }
        val httpClient = mockHttpClient(body = openAiWhisperResponse())
        val tool = TranscriptionTool(httpClient, providerKeys).tool
        val result = tool.execute(
            call("audio_base64" to smallBase64, "language" to "en"),
            ctx(),
        )
        assertTrue("expected Ok, got $result") { result is ToolResult.Ok }
        val text = (result as ToolResult.Ok).output
        assertTrue(text.contains("testing speech recognition"), "missing expected text: $text")
    }

    @Test
    fun `openai whisper with default language works`() = runTest {
        val providerKeys = mockk<ProviderKeys> {
            every { keyFor("openai") } returns "openai-key-123"
            every { keyFor("groq") } returns null
        }
        val httpClient = mockHttpClient(body = openAiWhisperResponse())
        val tool = TranscriptionTool(httpClient, providerKeys).tool
        val result = tool.execute(
            call("audio_base64" to smallBase64),
            ctx(),
        )
        assertTrue("expected Ok, got $result") { result is ToolResult.Ok }
        val text = (result as ToolResult.Ok).output
        assertTrue(text.contains("testing"), "missing expected text: $text")
    }

    @Test
    fun `openai whisper returns error on non-200`() = runTest {
        val providerKeys = mockk<ProviderKeys> {
            every { keyFor("openai") } returns "openai-key-123"
            every { keyFor("groq") } returns null
        }
        val httpClient = mockHttpClient(
            statusCode = 401,
            contentType = "text/plain",
            body = "Unauthorized",
        )
        val tool = TranscriptionTool(httpClient, providerKeys).tool
        val result = tool.execute(
            call("audio_base64" to smallBase64),
            ctx(),
        )
        assertTrue("expected Error, got $result") { result is ToolResult.Error }
        assertTrue((result as ToolResult.Error).message.contains("OpenAI Whisper API HTTP"))
    }

    // -----------------------------------------------------------------
    // 2. Groq Whisper path — OpenAI not configured, Groq configured
    // -----------------------------------------------------------------

    @Test
    fun `groq whisper with valid key returns transcription`() = runTest {
        val providerKeys = mockk<ProviderKeys> {
            every { keyFor("openai") } returns null
            every { keyFor("groq") } returns "groq-key-456"
        }
        val httpClient = mockHttpClient(body = groqWhisperResponse())
        val tool = TranscriptionTool(httpClient, providerKeys).tool
        val result = tool.execute(
            call("audio_base64" to smallBase64, "language" to "en"),
            ctx(),
        )
        assertTrue("expected Ok, got $result") { result is ToolResult.Ok }
        val text = (result as ToolResult.Ok).output
        assertTrue(text.contains("transcribed audio"), "missing expected text: $text")
    }

    @Test
    fun `groq whisper returns error on non-200`() = runTest {
        val providerKeys = mockk<ProviderKeys> {
            every { keyFor("openai") } returns null
            every { keyFor("groq") } returns "groq-key-456"
        }
        val httpClient = mockHttpClient(
            statusCode = 429,
            contentType = "text/plain",
            body = "Rate limit exceeded",
        )
        val tool = TranscriptionTool(httpClient, providerKeys).tool
        val result = tool.execute(
            call("audio_base64" to smallBase64),
            ctx(),
        )
        assertTrue("expected Error, got $result") { result is ToolResult.Error }
        assertTrue((result as ToolResult.Error).message.contains("Groq Whisper API HTTP"))
    }

    // -----------------------------------------------------------------
    // 3. No transcription provider configured
    // -----------------------------------------------------------------

    @Test
    fun `no provider configured returns error`() = runTest {
        val providerKeys = mockk<ProviderKeys> {
            every { keyFor("openai") } returns null
            every { keyFor("groq") } returns null
        }
        val httpClient = mockk<OkHttpClient>()
        val tool = TranscriptionTool(httpClient, providerKeys).tool
        val result = tool.execute(
            call("audio_base64" to smallBase64),
            ctx(),
        )
        assertTrue("expected Error, got $result") { result is ToolResult.Error }
        assertTrue((result as ToolResult.Error).message.contains("No transcription provider configured"))
    }

    // -----------------------------------------------------------------
    // 4. Size limit
    // -----------------------------------------------------------------

    @Test
    fun `audio over 25MB returns error`() = runTest {
        val providerKeys = mockk<ProviderKeys> {
            every { keyFor("openai") } returns "key"
        }
        val httpClient = mockk<OkHttpClient>()
        val tool = TranscriptionTool(httpClient, providerKeys).tool
        val result = tool.execute(
            call("audio_base64" to largeBase64),
            ctx(),
        )
        assertTrue("expected Error, got $result") { result is ToolResult.Error }
        val err = result as ToolResult.Error
        assertEquals("audio_too_large", err.code)
        assertTrue(err.message.contains("25 MB"), "should mention 25 MB limit: ${err.message}")
    }

    @Test
    fun `audio just under 25MB is accepted`() = runTest {
        // 25MB = 26,214,400 bytes → need base64 length ≤ (26,214,400 * 4) / 3 ≈ 34,952,533
        // Use 34,952,000 which gives estimatedBytes ≈ (34,952,000 * 3) / 4 = 26,214,000 < 25MB
        val justUnderBase64 = "A".repeat(34_952_000)

        val providerKeys = mockk<ProviderKeys> {
            every { keyFor("openai") } returns "openai-key-123"
            every { keyFor("groq") } returns null
        }
        val httpClient = mockHttpClient(body = openAiWhisperResponse())
        val tool = TranscriptionTool(httpClient, providerKeys).tool
        val result = tool.execute(
            call("audio_base64" to justUnderBase64),
            ctx(),
        )
        // Should pass size check and attempt API call (which is mocked)
        assertTrue("expected Ok, got $result") { result is ToolResult.Ok }
    }

    // -----------------------------------------------------------------
    // 5. Edge cases
    // -----------------------------------------------------------------

    @Test
    fun `missing audio_base64 returns error`() = runTest {
        val providerKeys = mockk<ProviderKeys>()
        val httpClient = mockk<OkHttpClient>()
        val tool = TranscriptionTool(httpClient, providerKeys).tool
        val result = tool.execute(call(), ctx())
        assertTrue("expected Error, got $result") { result is ToolResult.Error }
        assertEquals("bad_args", (result as ToolResult.Error).code)
    }

    @Test
    fun `empty base64 string returns error for OpenAI`() = runTest {
        val providerKeys = mockk<ProviderKeys> {
            every { keyFor("openai") } returns "key"
            every { keyFor("groq") } returns null
        }
        // Empty base64 decodes to empty byte array, which succeeds at decode
        // but the HTTP call will fail (or return empty transcript)
        val httpClient = mockHttpClient(
            statusCode = 400,
            contentType = "application/json",
            body = """{"error": {"message": "Invalid file format"}}""",
        )
        val tool = TranscriptionTool(httpClient, providerKeys).tool
        val result = tool.execute(
            call("audio_base64" to ""),
            ctx(),
        )
        assertTrue("expected Error, got $result") { result is ToolResult.Error }
        assertTrue((result as ToolResult.Error).message.contains("OpenAI Whisper"))
    }

    // -----------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------

    private fun call(vararg pairs: Pair<String, Any?>): com.aura.agent.ToolCall =
        com.aura.agent.ToolCall(id = "tc1", name = "transcribe", arguments = mapOf(*pairs))

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

    private fun openAiWhisperResponse() = """{
  "text": "This is a testing speech recognition transcription from OpenAI Whisper."
}"""

    private fun groqWhisperResponse() = """{
  "text": "This is a transcribed audio sample from Groq Whisper."
}"""
}
