package com.aura.capabilities

/**
 * Categorizes non-chat capability providers (TTS, image, video, search, 3D, embeddings).
 * Distinct from [com.aura.providers.Provider] which is the chat-completions interface.
 */
enum class CapabilityKind {
    TextToSpeech,
    ImageGeneration,
    VideoGeneration,
    World3DGeneration,
    WebSearch,
    Transcription,
}

/**
 * Base interface every non-chat capability provider implements.
 *
 * Chat-style providers implement [com.aura.providers.Provider] (ProviderRegistry,
 * OpenAiCompatProvider, etc.). Capability providers live in their own registry
 * so that a single [CapabilityKind] can have multiple interchangeable backends
 * (e.g. ElevenLabs vs Stability for TTS-vs-image, Exa vs Jina for web search).
 */
interface CapabilityProvider {
    val prefix: String
    val displayName: String
    val kind: CapabilityKind
    fun isConfigured(): Boolean
}
