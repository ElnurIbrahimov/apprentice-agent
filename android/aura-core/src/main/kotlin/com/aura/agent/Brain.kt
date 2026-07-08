package com.aura.agent

import android.content.Context
import com.aura.data.UserPreferences
import com.aura.providers.ChatOptions
import com.aura.providers.ProviderMessage
import com.aura.providers.ProviderRegistry
import com.aura.providers.ToolDefinition
import dagger.hilt.android.qualifiers.ApplicationContext
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.flow
import java.io.File
import javax.inject.Inject
import javax.inject.Singleton

/**
 * The brain. Wraps the provider registry, assembles system prompts,
 * owns the conversation context, hands the work to the agentic loop.
 *
 * ## Identity resolution
 *
 * The system prompt comes from one of three sources, checked in order:
 *
 *   1. **User override** at `filesDir/identity.md` (editable in
 *      Settings → Persona → Identity). Most recent change wins.
 *   2. **Bundled asset** at `assets/SOUL.md` (this repo ships with the
 *      Michaela Osbourne persona as the default).
 *   3. **Hardcoded fallback** — the historical `IDENTITY` constant, in
 *      case both files are missing. Never used in practice; exists so
 *      the brain always has *something* to send.
 *
 * `resolvedIdentity()` is called once per chat send (not on every
 * token), so reading a 5KB markdown file from disk is negligible.
 */
@Singleton
class Brain @Inject constructor(
    @ApplicationContext private val context: Context,
    private val providerRegistry: ProviderRegistry,
    private val userPreferences: UserPreferences,
) {
    /**
     * Identity as a `Flow<String>` so the chat loop and Settings
     * can both observe changes. Re-emits when the user edits the
     * file in Settings (the file's lastModified is the source of
     * truth — DataStore doesn't need to know).
     */
    val identity: Flow<String> = flow {
        // Emit once on subscribe. We don't poll the file for changes
        // because the only writer is the same process (Settings), and
        // the chat loop calls resolvedIdentity() on every send — the
        // file is read fresh there, not from this flow.
        emit(loadIdentity())
    }

    /**
     * Resolved system prompt. Reads the user override file →
     * the bundled asset → the hardcoded fallback, and returns
     * the first one that exists. This is the "persona" layer
     * only — the "about the user" layer (name, traits, facts)
     * is handled separately by [com.aura.profile.UserProfileStore]
     * and concatenated at the call site.
     */
    suspend fun resolvedIdentity(): String = loadIdentity()

    /**
     * Read the identity file in priority order: user override →
     * bundled asset → hardcoded fallback. Synchronous on purpose —
     * it's called once per chat send, not on a hot path.
     */
    private fun loadIdentity(): String {
        // 1. User override
        val override = File(context.filesDir, IDENTITY_OVERRIDE_FILENAME)
        if (override.exists() && override.length() > 0) {
            return override.readText().trim()
        }
        // 2. Bundled asset
        return try {
            context.assets.open(IDENTITY_ASSET_FILENAME)
                .bufferedReader()
                .use { it.readText() }
                .trim()
        } catch (e: Exception) {
            // 3. Hardcoded fallback — should never trigger in shipped
            // builds because the asset is bundled, but if someone
            // strips the asset we still want the brain to work.
            IDENTITY_FALLBACK.trim()
        }
    }

    fun stream(
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
        /** Path of the user-editable identity file in app-private storage. */
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
         * Not user-facing. Settings → Persona → Identity → "Reset
         * to default" restores from the bundled asset, not from
         * this constant.
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
