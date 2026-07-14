package com.aura.tools

import com.aura.agent.Tool
import com.aura.agent.ToolResult
import com.aura.agent.ToolRisk
import com.aura.providers.ProviderKeys
import com.aura.providers.ToolParameters
import com.aura.providers.ToolProperty
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.MultipartBody
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import java.util.Base64
import javax.inject.Inject
import javax.inject.Singleton

/** Maximum decoded audio accepted by transcription providers: 25 MB. */
const val MAX_TRANSCRIPTION_AUDIO_BYTES: Int = 25 * 1024 * 1024

/**
 * Transcription tool: sends base64-encoded audio to a cloud Whisper API
 * and returns the transcribed text.
 *
 * Provider selection order:
 * 1. OpenAI Whisper (whisper-1) — if an OpenAI API key is configured
 * 2. Groq (whisper-large-v3) — if a Groq API key is configured
 *
 * Risk: READ_ONLY (network egress only, no phone permissions).
 */
@Singleton
class TranscriptionTool @Inject constructor(
    private val httpClient: OkHttpClient,
    private val providerKeys: ProviderKeys,
) {
    private val json = Json { ignoreUnknownKeys = true }
    private val mediaTypeOctet = "application/octet-stream".toMediaType()

    fun definition() = ToolParameters(
        properties = mapOf(
            "audio_base64" to ToolProperty(
                type = "string",
                description = "Base64-encoded audio data (WAV, MP3, M4A, etc.)",
            ),
            "language" to ToolProperty(
                type = "string",
                description = "Language code (default: 'en'). Optional.",
            ),
        ),
        required = listOf("audio_base64"),
    )

    val tool = Tool(
        name = "transcribe",
        description = "Transcribe audio to text using a cloud Whisper API. " +
            "Provide audio as base64-encoded data and an optional language code. " +
            "Audio must be 25 MB or smaller.",
        risk = ToolRisk.REMOTE_COST,
        parameters = definition(),
        execute = { call, _ ->
            val audioBase64 = call.arguments["audio_base64"] as? String
                ?: return@Tool ToolResult.Error("missing 'audio_base64' argument", "bad_args")
            val language = call.arguments["language"] as? String ?: "en"

            // Size check: estimate decoded bytes from base64 length
            // base64 encodes 3 bytes into 4 chars, so length * 3/4 ≈ decoded size
            val estimatedBytes = (audioBase64.length * 3L) / 4
            if (estimatedBytes > MAX_TRANSCRIPTION_AUDIO_BYTES) {
                return@Tool ToolResult.Error(
                    "Audio too large (${estimatedBytes / (1024 * 1024)} MB). Maximum is 25 MB.",
                    "audio_too_large",
                )
            }

            try {
                val result = transcribeAudio(audioBase64, language)
                ToolResult.Ok(result)
            } catch (e: Exception) {
                ToolResult.Error("transcription failed: ${e.message}", "http_error")
            }
        },
    category = "media")
    /**
     * Decodes a base64 string to a byte array.
     */
    private fun decodeBase64(base64: String): ByteArray {
        return Base64.getDecoder().decode(base64)
    }

    /**
     * Routes to the first configured transcription provider.
     *
     * Order: OpenAI Whisper -> Groq.
     *
     * @throws RuntimeException if no transcription provider is configured.
     */
    private fun transcribeAudio(audioBase64: String, language: String): String {
        val audioBytes = decodeBase64(audioBase64)

        val openaiKey = providerKeys.keyFor("openai")
        if (!openaiKey.isNullOrBlank()) {
            return transcribeWithOpenAi(audioBytes, language, openaiKey)
        }

        val groqKey = providerKeys.keyFor("groq")
        if (!groqKey.isNullOrBlank()) {
            return transcribeWithGroq(audioBytes, language, groqKey)
        }

        throw RuntimeException(
            "No transcription provider configured. Please configure OpenAI or Groq in Settings.",
        )
    }

    // ------------------------------------------------------------------
    // OpenAI Whisper API
    // ------------------------------------------------------------------
    // POST https://api.openai.com/v1/audio/transcriptions
    // Multipart form: file=<audio bytes> model=whisper-1 language=<code>
    // Header: Authorization: Bearer <OPENAI_API_KEY>
    // Response: { "text": "..." }
    // ------------------------------------------------------------------

    private fun transcribeWithOpenAi(
        audioBytes: ByteArray,
        language: String,
        apiKey: String,
    ): String {
        val audioBody = audioBytes.toRequestBody(mediaTypeOctet)
        val requestBody = MultipartBody.Builder()
            .setType(MultipartBody.FORM)
            .addFormDataPart("file", "audio.mp3", audioBody)
            .addFormDataPart("model", "whisper-1")
            .addFormDataPart("language", language)
            .build()

        val req = Request.Builder()
            .url("https://api.openai.com/v1/audio/transcriptions")
            .header("Authorization", "Bearer $apiKey")
            .post(requestBody)
            .build()

        return httpClient.newCall(req).execute().use { response ->
            if (!response.isSuccessful) {
                val errorBody = response.body?.string() ?: ""
                throw RuntimeException("OpenAI Whisper API HTTP ${response.code}: $errorBody")
            }
            val respBody = response.body?.string()
                ?: throw RuntimeException("Empty response from OpenAI Whisper")

            val root = json.parseToJsonElement(respBody).jsonObject
            root["text"]?.jsonPrimitive?.content
                ?: throw RuntimeException("OpenAI Whisper response missing 'text' field")
        }
    }

    // ------------------------------------------------------------------
    // Groq Whisper API (whisper-large-v3)
    // ------------------------------------------------------------------
    // POST https://api.groq.com/openai/v1/audio/transcriptions
    // Multipart form: file=<audio bytes> model=whisper-large-v3 language=<code>
    // Header: Authorization: Bearer <GROQ_API_KEY>
    // Response: { "text": "..." }
    // ------------------------------------------------------------------

    private fun transcribeWithGroq(
        audioBytes: ByteArray,
        language: String,
        apiKey: String,
    ): String {
        val audioBody = audioBytes.toRequestBody(mediaTypeOctet)
        val requestBody = MultipartBody.Builder()
            .setType(MultipartBody.FORM)
            .addFormDataPart("file", "audio.mp3", audioBody)
            .addFormDataPart("model", "whisper-large-v3")
            .addFormDataPart("language", language)
            .build()

        val req = Request.Builder()
            .url("https://api.groq.com/openai/v1/audio/transcriptions")
            .header("Authorization", "Bearer $apiKey")
            .post(requestBody)
            .build()

        return httpClient.newCall(req).execute().use { response ->
            if (!response.isSuccessful) {
                val errorBody = response.body?.string() ?: ""
                throw RuntimeException("Groq Whisper API HTTP ${response.code}: $errorBody")
            }
            val respBody = response.body?.string()
                ?: throw RuntimeException("Empty response from Groq Whisper")

            val root = json.parseToJsonElement(respBody).jsonObject
            root["text"]?.jsonPrimitive?.content
                ?: throw RuntimeException("Groq Whisper response missing 'text' field")
        }
    }
}
