package com.aura.capabilities

/**
 * Text-to-speech request. [model] may be empty to use the provider's default voice model.
 * [voice] is a provider-specific voice id (ElevenLabs voice id, OpenAI alloy/echo/etc., Stability voice preset).
 */
data class TtsRequest(
    val text: String,
    val voice: String,
    val model: String = "",
    val format: String = "mp3",
)

/**
 * TTS response is always raw audio bytes plus a suggested filename extension.
 * Callers write to a file or stream to a player.
 */
data class TtsResult(
    val audio: ByteArray,
    val mimeType: String,
    val extension: String,
) {
    override fun equals(other: Any?): Boolean = other is TtsResult &&
        audio.contentEquals(other.audio) &&
        mimeType == other.mimeType &&
        extension == other.extension
    override fun hashCode(): Int {
        var result = audio.contentHashCode()
        result = 31 * result + mimeType.hashCode()
        result = 31 * result + extension.hashCode()
        return result
    }
}

interface TextToSpeechProvider : CapabilityProvider {
    override val kind: CapabilityKind get() = CapabilityKind.TextToSpeech
    suspend fun speak(req: TtsRequest): TtsResult
}
