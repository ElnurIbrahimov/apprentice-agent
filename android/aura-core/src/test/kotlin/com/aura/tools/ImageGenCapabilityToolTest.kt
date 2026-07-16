package com.aura.tools

import com.aura.agent.ToolCall
import com.aura.agent.ToolContext
import com.aura.agent.ToolResult
import com.aura.capabilities.CapabilityKind
import com.aura.capabilities.CapabilityProvider
import com.aura.capabilities.CapabilityRouter
import com.aura.capabilities.ImageProvider
import com.aura.capabilities.ImageRequest
import com.aura.capabilities.ImageResult
import io.mockk.coEvery
import io.mockk.every
import io.mockk.mockk
import kotlinx.coroutines.test.runTest
import org.junit.Assert.assertTrue
import org.junit.Test

class ImageGenCapabilityToolTest {

    private val capabilityRouter = mockk<CapabilityRouter>(relaxed = true)
    private val tool = ImageGenCapabilityTool(capabilityRouter).tool

    private val ctx = ToolContext(conversationId = "test", approvedRemoteCostTools = setOf("image_generate"))

    @Test
    fun returns_error_when_no_provider_configured() = runTest {
        every { capabilityRouter.resolve(CapabilityKind.ImageGeneration, "generate") } returns null

        val result = tool.execute(
            ToolCall(id = "1", name = "image_generate", arguments = mapOf("prompt" to "a cat")),
            ctx,
        )
        assertTrue("Expected error, got $result", result is ToolResult.Error)
        assertTrue(
            "Expected no_provider, got ${(result as ToolResult.Error).code}",
            (result as ToolResult.Error).code == "no_provider",
        )
    }

    @Test
    fun returns_error_when_provider_not_image_provider() = runTest {
        val nonImageProvider = mockk<CapabilityProvider>()
        every { nonImageProvider.displayName } returns "Some Provider"
        every { capabilityRouter.resolve(CapabilityKind.ImageGeneration, "generate") } returns nonImageProvider

        val result = tool.execute(
            ToolCall(id = "1", name = "image_generate", arguments = mapOf("prompt" to "a cat")),
            ctx,
        )
        assertTrue("Expected error, got $result", result is ToolResult.Error)
        assertTrue(
            "Expected provider_mismatch, got ${(result as ToolResult.Error).code}",
            (result as ToolResult.Error).code == "provider_mismatch",
        )
    }

    @Test
    fun returns_ok_when_generation_succeeds() = runTest {
        val imageProvider = mockk<ImageProvider>()
        every { imageProvider.displayName } returns "Stability AI"
        every { capabilityRouter.resolve(CapabilityKind.ImageGeneration, "generate") } returns imageProvider
        coEvery { imageProvider.generate(any()) } returns ImageResult(
            url = "https://example.com/image.png",
            mimeType = "image/png",
        )

        val result = tool.execute(
            ToolCall(id = "1", name = "image_generate", arguments = mapOf("prompt" to "a cat")),
            ctx,
        )
        assertTrue("Expected Ok, got $result", result is ToolResult.Ok)
        val output = (result as ToolResult.Ok).output
        assertTrue("Output should mention provider", output.contains("Stability AI"))
        assertTrue("Output should include URL", output.contains("https://example.com/image.png"))
    }

    @Test
    fun returns_error_when_generation_throws() = runTest {
        val imageProvider = mockk<ImageProvider>()
        every { imageProvider.displayName } returns "Stability AI"
        every { capabilityRouter.resolve(CapabilityKind.ImageGeneration, "generate") } returns imageProvider
        coEvery { imageProvider.generate(any()) } throws RuntimeException("API timeout")

        val result = tool.execute(
            ToolCall(id = "1", name = "image_generate", arguments = mapOf("prompt" to "a cat")),
            ctx,
        )
        assertTrue("Expected error, got $result", result is ToolResult.Error)
        assertTrue(
            "Expected generation_error, got ${(result as ToolResult.Error).code}",
            (result as ToolResult.Error).code == "generation_error",
        )
    }

    @Test
    fun returns_error_when_prompt_missing() = runTest {
        val result = tool.execute(
            ToolCall(id = "1", name = "image_generate", arguments = emptyMap()),
            ctx,
        )
        assertTrue("Expected error, got $result", result is ToolResult.Error)
        assertTrue(
            "Expected bad_args, got ${(result as ToolResult.Error).code}",
            (result as ToolResult.Error).code == "bad_args",
        )
    }
}