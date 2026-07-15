package com.aura.tools

import com.aura.agent.ToolCall
import com.aura.agent.ToolContext
import com.aura.agent.ToolResult
import com.aura.capabilities.CapabilityRegistry
import com.aura.capabilities.TtsRequest
import com.aura.capabilities.TtsResult
import com.aura.capabilities.TextToSpeechProvider
import io.mockk.coEvery
import io.mockk.every
import io.mockk.mockk
import kotlinx.coroutines.test.runTest
import org.junit.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class TtsSpeakToolTest {

    @Test
    fun `tts_speak uses configured provider and returns success`() = runTest {
        val provider = mockk<TextToSpeechProvider>(relaxed = true)
        every { provider.prefix } returns "elevenlabs"
        every { provider.displayName } returns "ElevenLabs"
        every { provider.isConfigured() } returns true
        coEvery { provider.speak(any()) } returns TtsResult(
            audio = "audio".toByteArray(),
            mimeType = "audio/mpeg",
            extension = "mp3",
        )
        val registry = mockk<CapabilityRegistry>(relaxed = true)
        every { registry.configuredForKind(com.aura.capabilities.CapabilityKind.TextToSpeech) } returns listOf(provider)

        val tool = TtsSpeakTool(mockk(relaxed = true), registry).tool
        val result = tool.execute(ToolCall("id-1", tool.name, mapOf("text" to "hello", "play" to false)),
            ToolContext(conversationId = "test"))
        assertTrue(result is ToolResult.Ok)
        assertTrue((result as ToolResult.Ok).output.contains("ElevenLabs"))
    }

    @Test
    fun `tts_speak returns error when no provider configured`() = runTest {
        val registry = mockk<CapabilityRegistry>(relaxed = true)
        every { registry.configuredForKind(com.aura.capabilities.CapabilityKind.TextToSpeech) } returns emptyList()
        val tool = TtsSpeakTool(mockk(relaxed = true), registry).tool
        val result = tool.execute(ToolCall("id-2", tool.name, mapOf("text" to "hello")),
            ToolContext(conversationId = "test"))
        assertTrue(result is ToolResult.Error)
        assertEquals("No TTS provider configured. Add an ElevenLabs key in Settings or enable platform TTS.", (result as ToolResult.Error).message)
    }
}
