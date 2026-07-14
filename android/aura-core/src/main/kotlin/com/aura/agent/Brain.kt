package com.aura.agent

import com.aura.providers.ChatOptions
import com.aura.providers.ProviderMessage
import com.aura.providers.ProviderRegistry
import com.aura.providers.ToolDefinition
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import javax.inject.Inject
import javax.inject.Singleton

/**
 * The brain. Wraps the provider registry, assembles system prompts,
 * owns the conversation context, hands the work to the agentic loop.
 *
 * ## Identity resolution
 *
 * The user override is stored in DataStore through [IdentityStore], which is
 * also what Settings and backup/restore use. When no custom text exists,
 * [IdentityStore] reads the bundled `assets/SOUL.md`; the hardcoded fallback
 * is only used if the asset is missing.
 */
@Singleton
class Brain @Inject constructor(
    private val providerRegistry: ProviderRegistry,
    private val identityStore: IdentityStore,
) {
    /** Current identity as a cold flow; each subscription resolves DataStore anew. */
    val identity: Flow<String> = flow {
        emit(identityStore.readCurrent())
    }

    /**
     * Resolved system prompt. Reads the DataStore override, then bundled
     * asset, then hardcoded fallback through [IdentityStore]. This is the
     * "persona" layer only — the "about the user" layer (name, traits, facts)
     * is handled separately by [com.aura.profile.UserProfileStore]
     * and concatenated at the call site.
     */
    suspend fun resolvedIdentity(): String = identityStore.readCurrent()

    /**
     * Stream chat tokens from the configured provider. The function is
     * `suspend` (not `fun`) so the underlying `providerRegistry.chat`
     * call can dispatch without `runBlocking` — the call site is
     * already inside a coroutine context, and the previous
     * thread-blocking call was freezing the main thread when the
     * call came from a Compose context.
     */
    suspend fun stream(
        model: String,
        messages: List<ProviderMessage>,
        tools: List<ToolDefinition> = emptyList(),
        options: ChatOptions = ChatOptions(),
    ): Flow<BrainChunk> = flow {
        // nameById accumulates tool-call ids to names across the stream so
        // providers that send argument deltas without re-sending the name
        // (e.g. Anthropic input_json_delta) can still be routed to the
        // correct tool. Reset per stream call.
        val nameById = mutableMapOf<String, String>()
        providerRegistry.chat(model, messages, options, tools).collect { providerChunk ->
            emit(BrainChunk.fromProvider(providerChunk, nameById))
        }
    }

    companion object {
        /** Legacy override filename retained for one-time migration. */
        const val IDENTITY_OVERRIDE_FILENAME = "identity.md"

        /** Path of the bundled identity asset (shipped with the APK). */
        const val IDENTITY_ASSET_FILENAME = "SOUL.md"

        /**
         * Hardcoded fallback identity. The compiled artifact of the
         * original Michaela Osbourne persona, kept here as a safety
         * net in case both the asset and the user override are
         * missing (e.g. someone manually deleted SOUL.md from the
         * assets folder at build time).
         *
         * Not user-facing. Resetting identity clears the DataStore override
         * and resolves this only if the bundled asset is unavailable.
         */
        val IDENTITY_FALLBACK = """
            You are Aura, a personal AI assistant.
        """.trimIndent()
    }
}

sealed class BrainChunk {
    data class Text(val text: String) : BrainChunk()
    data class ToolCallStart(val id: String, val name: String) : BrainChunk()
    data class ToolCallDelta(val id: String, val argumentsDelta: String) : BrainChunk()
    data class ToolCallEnd(val id: String, val name: String, val arguments: String) : BrainChunk()
    data class Finished(val reason: String) : BrainChunk()
    data class Error(
        val code: String,
        val message: String,
        val retryable: Boolean,
        val error: com.aura.providers.ProviderError? = null,
    ) : BrainChunk()

    companion object {
        /**
         * Map a ProviderChunk into the higher-level BrainChunk stream the
         * agentic loop consumes. For tool calls we emit:
         *   - ToolCallStart the first time we see a given id with a name
         *   - ToolCallDelta for every subsequent argument chunk
         *   - ToolCallEnd once (typically on the finish_reason=tool_calls event,
         *     but providers are inconsistent so we also accept ProviderChunks
         *     that carry a full arguments string).
         *
         * The previous version collapsed all three into a single ToolCallDelta
         * and never emitted ToolCallStart, which broke the loop's tool-name
         * lookup for OpenAI/Anthropic (the loop reads the name from
         * `toolCallStarts[id]`).
         */
        fun fromProvider(p: com.aura.providers.ProviderChunk, nameById: MutableMap<String, String> = mutableMapOf()): BrainChunk {
            p.error?.let { return Error(it.code, it.message, it.retryable, error = it) }
            p.finishReason?.let { return Finished(it.name) }
            val tc = p.toolCall
            if (tc != null) {
                if (tc.id.isNotEmpty() && tc.name.isNotEmpty()) {
                    if (nameById.put(tc.id, tc.name) == null) {
                        return ToolCallStart(tc.id, tc.name)
                    }
                    if (tc.arguments.isNotEmpty()) {
                        return ToolCallEnd(tc.id, tc.name, tc.arguments)
                    }
                    return ToolCallDelta(tc.id, "")
                }
                // delta-style chunk (Anthropic input_json_delta): no id, no name.
                // Look up the most-recent id we saw and append to its args.
                val id = nameById.keys.lastOrNull() ?: return Text("")
                return ToolCallDelta(id, tc.arguments)
            }
            p.text?.let { return Text(it) }
            return Text("")
        }
    }
}
