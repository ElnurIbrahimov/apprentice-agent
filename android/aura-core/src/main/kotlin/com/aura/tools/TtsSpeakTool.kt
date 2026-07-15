package com.aura.tools

import android.content.Context
import android.media.MediaPlayer
import androidx.core.net.toFile
import com.aura.agent.Tool
import com.aura.agent.ToolResult
import com.aura.agent.ToolRisk
import com.aura.capabilities.CapabilityRegistry
import com.aura.capabilities.TextToSpeechProvider
import com.aura.providers.ToolDefinition
import com.aura.providers.ToolParameters
import com.aura.providers.ToolProperty
import dagger.hilt.android.qualifiers.ApplicationContext
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import java.util.Base64
import javax.inject.Inject
import javax.inject.Singleton

/**
 * Text-to-speech tool. Uses the first configured TTS capability provider
 * (ElevenLabs) if available; otherwise falls back to platform TTS.
 * The platform path speaks immediately; the ElevenLabs path returns
 * base64 audio and optionally plays it.
 * Risk: REMOTE_COST when using a cloud provider.
 */
@Singleton
class TtsSpeakTool @Inject constructor(
    @ApplicationContext private val context: Context,
    private val capabilityRegistry: CapabilityRegistry,
) {
    fun definition() = ToolDefinition(
        name = "tts_speak",
        description = "Convert text to speech. Uses ElevenLabs if configured, otherwise platform TTS. Set play=true to speak immediately; returns base64 audio when a cloud provider is used.",
        parameters = ToolParameters(
            properties = mapOf(
                "text" to ToolProperty(type = "string", description = "Text to speak"),
                "voice" to ToolProperty(type = "string", description = "Voice ID (cloud providers only, default provider default)"),
                "play" to ToolProperty(type = "boolean", description = "Play audio immediately (default true)"),
            ),
            required = listOf("text"),
        ),
    )

    val tool = Tool(
        name = "tts_speak",
        description = definition().description,
        risk = ToolRisk.REMOTE_COST,
        parameters = definition().parameters,
        execute = { call, _ ->
            val text = call.arguments["text"] as? String
                ?: return@Tool ToolResult.Error("missing 'text'", "bad_args")
            val voice = call.arguments["voice"] as? String ?: ""
            val play = call.arguments["play"] as? Boolean ?: true

            val provider = capabilityRegistry.configuredForKind(com.aura.capabilities.CapabilityKind.TextToSpeech)
                .filterIsInstance<TextToSpeechProvider>()
                .firstOrNull()

            if (provider == null) {
                return@Tool ToolResult.Error(
                    "No TTS provider configured. Add an ElevenLabs key in Settings or enable platform TTS.",
                    "missing_provider",
                )
            }

            try {
                val result = provider.speak(
                    com.aura.capabilities.TtsRequest(text = text, voice = voice)
                )
                if (play) {
                    playAudio(result.audio, result.mimeType)
                }
                ToolResult.Ok(
                    "Spoke ${text.length} chars via ${provider.displayName}. " +
                        "Audio: ${result.audio.size} bytes (${result.mimeType}). " +
                        if (play) "Playing now." else "Base64 omitted."
                )
            } catch (e: Exception) {
                ToolResult.Error("TTS failed: ${e.message}", "tts_error")
            }
        },
        category = "media",
    )

    private suspend fun playAudio(audio: ByteArray, mimeType: String) = withContext(Dispatchers.IO) {
        val ext = when (mimeType) {
            "audio/mpeg" -> "mp3"
            "audio/wav" -> "wav"
            "audio/ogg" -> "ogg"
            "audio/mp4" -> "m4a"
            else -> "tmp"
        }
        val file = File(context.cacheDir, "aura_tts_${System.currentTimeMillis()}.$ext").apply {
            writeBytes(audio)
            deleteOnExit()
        }
        MediaPlayer().apply {
            setDataSource(file.absolutePath)
            prepare()
            start()
            setOnCompletionListener { release() }
        }
    }
}
