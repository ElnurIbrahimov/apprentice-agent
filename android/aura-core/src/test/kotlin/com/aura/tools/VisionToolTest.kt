package com.aura.tools

import com.aura.agent.ToolResult
import com.aura.data.UserPreferences
import com.aura.providers.ProviderKeys
import io.mockk.every
import io.mockk.mockk
import io.mockk.slot
import kotlinx.coroutines.flow.flowOf
import kotlinx.coroutines.test.runTest
import okhttp3.Call
import okhttp3.MediaType.Companion.toMediaTypeOrNull
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.Response
import okhttp3.ResponseBody.Companion.toResponseBody
import okhttp3.Protocol
import org.junit.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class VisionToolTest {
    private val smallBase64 = "/9j/4AAQSkZJRgABAQEASABIAAD/2wBDAAg=="
    private val largeBase64 = "A".repeat(2_796_500)

    @Test
    fun `selected Gemini model is sent in URL`() = runTest {
        val transport = transport(geminiResponse())
        val result = tool(
            transport.client,
            keys("gemini"),
            "gemini:test-vision-model",
        ).execute(call("image_base64" to smallBase64), ctx())

        assertTrue(result is ToolResult.Ok)
        assertTrue((result as ToolResult.Ok).output.contains("sunset"))
        assertTrue(transport.request.captured.url.toString().contains("test-vision-model"))
        assertEquals("test-key", transport.request.captured.header("X-Goog-Api-Key"))
    }

    @Test
    fun `selected OpenAI compatible model is sent in body`() = runTest {
        val transport = transport(openAiResponse())
        val result = tool(
            transport.client,
            keys("openai"),
            "openai:test-vision-model",
        ).execute(call("image_base64" to smallBase64), ctx())

        assertTrue(result is ToolResult.Ok)
        assertTrue((result as ToolResult.Ok).output.contains("beach"))
        assertTrue(transport.request.captured.url.toString().endsWith("/chat/completions"))
        assertTrue(transport.requestBody().contains("test-vision-model"))
    }

    @Test
    fun `selected Ollama model uses OpenAI compatible transport`() = runTest {
        val transport = transport(openAiResponse("mountain landscape"))
        val result = tool(
            transport.client,
            keys("ollama"),
            "ollama:test-vision-model",
        ).execute(call("image_base64" to smallBase64), ctx())

        assertTrue(result is ToolResult.Ok)
        assertTrue(transport.request.captured.url.host.contains("ollama"))
        assertTrue(transport.requestBody().contains("test-vision-model"))
    }

    @Test
    fun `selected Anthropic model uses messages transport`() = runTest {
        val transport = transport(anthropicResponse())
        val result = tool(
            transport.client,
            keys("anthropic"),
            "anthropic:test-vision-model",
        ).execute(call("image_base64" to smallBase64), ctx())

        assertTrue(result is ToolResult.Ok)
        assertTrue((result as ToolResult.Ok).output.contains("city"))
        assertTrue(transport.requestBody().contains("test-vision-model"))
        assertEquals("test-key", transport.request.captured.header("x-api-key"))
    }

    @Test
    fun `missing selected vision role returns recovery error`() = runTest {
        val result = tool(
            mockk(),
            keys("gemini"),
            null,
        ).execute(call("image_base64" to smallBase64), ctx())

        assertTrue(result is ToolResult.Error)
        assertTrue((result as ToolResult.Error).message.contains("Choose a vision model"))
    }

    @Test
    fun `selected provider without key returns recovery error`() = runTest {
        val result = tool(
            mockk(),
            keys(null),
            "gemini:test-vision-model",
        ).execute(call("image_base64" to smallBase64), ctx())

        assertTrue(result is ToolResult.Error)
        assertTrue((result as ToolResult.Error).message.contains("not configured"))
    }

    @Test
    fun `unsupported selected provider returns clear error`() = runTest {
        val result = tool(
            mockk(),
            keys("unsupported"),
            "unsupported:test-vision-model",
        ).execute(call("image_base64" to smallBase64), ctx())

        assertTrue(result is ToolResult.Error)
        assertTrue((result as ToolResult.Error).message.contains("does not support"))
    }

    @Test
    fun `image over 2MB is rejected before model lookup`() = runTest {
        val result = tool(mockk(), keys(null), null)
            .execute(call("image_base64" to largeBase64), ctx())

        assertTrue(result is ToolResult.Error)
        assertEquals("image_too_large", (result as ToolResult.Error).code)
    }

    @Test
    fun `missing image argument is rejected`() = runTest {
        val result = tool(mockk(), keys(null), null).execute(call(), ctx())

        assertTrue(result is ToolResult.Error)
        assertEquals("bad_args", (result as ToolResult.Error).code)
    }

    @Test
    fun `HTTP failure does not include response body`() = runTest {
        val transport = transport(
            body = "sensitive upstream details",
            statusCode = 500,
        )
        val result = tool(
            transport.client,
            keys("gemini"),
            "gemini:test-vision-model",
        ).execute(call("image_base64" to smallBase64), ctx())

        assertTrue(result is ToolResult.Error)
        val message = (result as ToolResult.Error).message
        assertTrue(message.contains("HTTP 500"))
        assertTrue(!message.contains("sensitive upstream details"))
    }

    private fun tool(
        client: OkHttpClient,
        providerKeys: ProviderKeys,
        modelId: String?,
    ) = VisionTool(
        httpClient = client,
        providerKeys = providerKeys,
        userPreferences = mockk<UserPreferences> {
            every { visionModel } returns flowOf(modelId)
        },
    ).tool

    private fun keys(configuredPrefix: String?): ProviderKeys = mockk {
        every { keyFor(any()) } answers {
            if (args[0] as kotlin.String == configuredPrefix) "test-key" else null
        }
    }

    private data class MockTransport(
        val client: OkHttpClient,
        val request: io.mockk.CapturingSlot<Request>,
    ) {
        fun requestBody(): String {
            val buffer = okio.Buffer()
            request.captured.body?.writeTo(buffer)
            return buffer.readUtf8()
        }
    }

    private fun transport(
        body: String,
        statusCode: Int = 200,
    ): MockTransport {
        val request = slot<Request>()
        val call = mockk<Call>()
        every { call.execute() } answers {
            Response.Builder()
                .request(request.captured)
                .protocol(Protocol.HTTP_1_1)
                .code(statusCode)
                .message(if (statusCode in 200..299) "OK" else "Error")
                .body(body.toResponseBody("application/json".toMediaTypeOrNull()))
                .build()
        }
        val client = mockk<OkHttpClient> {
            every { newCall(capture(request)) } returns call
        }
        return MockTransport(client, request)
    }

    private fun call(vararg pairs: Pair<String, Any?>): com.aura.agent.ToolCall =
        com.aura.agent.ToolCall("tc1", "vision", mapOf(*pairs))

    private fun ctx() = com.aura.agent.ToolContext(conversationId = "conv-1")

    private fun geminiResponse() = """{
      "candidates": [{"content":{"parts":[{"text":"A sunset over mountains."}]}}]
    }"""

    private fun openAiResponse(text: String = "A beach at sunset.") = """{
      "choices": [{"message":{"content":"$text","role":"assistant"}}]
    }"""

    private fun anthropicResponse() = """{
      "content": [{"type":"text","text":"A city skyline at night."}]
    }"""
}
