package com.aura.capabilities.elevenlabs

import com.aura.capabilities.TextToSpeechProvider
import com.aura.capabilities.TtsRequest
import com.aura.capabilities.TtsResult
import com.aura.capabilities.http.CapabilityHttp
import com.aura.providers.ProviderKeys
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.RequestBody.Companion.toRequestBody
import javax.inject.Inject
import javax.inject.Singleton

/**
 * ElevenLabs TTS. POST https://api.elevenlabs.io/v1/text-to-speech/{voice_id}
 * Auth: xi-api-key header.
 * https://elevenlabs.io/docs/api-reference/text-to-speech/convert
 */
@Singleton
class ElevenLabsTtsProvider @Inject constructor(
    private val client: OkHttpClient,
    private val providerKeys: ProviderKeys,
) : TextToSpeechProvider {
    override val prefix = "elevenlabs"
    override val displayName = "ElevenLabs"
    private val apiKey: String get() = providerKeys.keyFor(prefix).orEmpty()
    override fun isConfigured(): Boolean = apiKey.isNotBlank()

    override suspend fun speak(req: TtsRequest): TtsResult = withContext(Dispatchers.IO) {
        val voice = req.voice.ifBlank { "EXAVITQu4vr4xnSDxMaL" } // Bella default voice
        val url = "https://api.elevenlabs.io/v1/text-to-speech/$voice"
        val body = CapabilityHttp.buildJsonBody(
            "text" to req.text,
            "model_id" to (req.model.ifBlank { "eleven_multilingual_v2" }),
        )
        val request = okhttp3.Request.Builder()
            .url(url)
            .header("xi-api-key", apiKey)
            .header("Content-Type", "application/json")
            .header("Accept", "audio/mpeg")
            .post(body.toRequestBody("application/json".toMediaType()))
            .build()
        val response = client.newCall(request).execute()
        val bytes = response.use { it.body?.bytes() ?: ByteArray(0) }
        if (!response.isSuccessful) {
            val msg = String(bytes, Charsets.UTF_8)
            throw com.aura.capabilities.CapabilityCatalogException.NetworkException(
                message = "ElevenLabs returned ${response.code}: $msg",
                statusCode = response.code,
            )
        }
        if (bytes.isEmpty()) {
            throw com.aura.capabilities.CapabilityCatalogException.MalformedResponseException(
                "ElevenLabs returned empty audio body",
            )
        }
        TtsResult(audio = bytes, mimeType = "audio/mpeg", extension = "mp3")
    }
}
