package com.aura.providers

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.flowOn
import kotlinx.coroutines.withContext
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.put
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import okhttp3.sse.EventSource
import okhttp3.sse.EventSourceListener
import okhttp3.sse.EventSources
import javax.inject.Inject
import javax.inject.Singleton

/**
 * ChatGPT subscription auth via OAuth bearer token.
 *
 * The user gets a ChatGPT Plus/Pro/Go subscription token by running
 * `codex login` on a machine with a browser, then exporting the resulting
 * token (or CODEX_AUTH_JSON) to Aura via the Settings UI. The token is
 * then routed through OpenAI's ChatGPT Responses API as a Bearer token
 * (Authorization: Bearer <access_token>), which is what the Codex CLI does.
 *
 * Model ids for this provider are derived from the openai catalog (the
 * user can also use a model catalog cached by the openai provider) but
 * routed through the subscription token. We treat the live token from
 * the user as the source of truth; if the user wants to use their
 * OpenAI API key instead, they can use the regular `openai` provider.
 */
class ChatGptSubscriptionProvider(
    private val providerKeys: ProviderKeys,
    private val httpClient: OkHttpClient,
    /**
     * Base URL for the OpenAI ChatGPT backend. Override in tests; default
     * is the production Codex/ChatGPT endpoint.
     */
    private val baseUrl: String = "https://chatgpt.com/backend-api/codex",
) : Provider {
    override val prefix = "chatgpt"
    override val displayName = "ChatGPT Subscription"
    private val apiKey: String get() = providerKeys.keyFor(prefix).orEmpty()
    @Volatile private var activeEventSource: EventSource? = null

    override fun isConfigured(): Boolean = apiKey.isNotBlank()

    override fun chat(
        model: String,
        messages: List<ProviderMessage>,
        options: ChatOptions,
        tools: List<ToolDefinition>,
    ): Flow<ProviderChunk> = flow {
        val body = buildJsonObject {
            put("model", model)
            put("stream", true)
            put("temperature", options.temperature)
            put("top_p", options.topP)
            options.maxTokens?.let { put("max_tokens", it) }
            put("input", JsonArray(messages.map { msg ->
                buildJsonObject {
                    put("role", msg.role.name)
                    put("content", msg.content)
                }
            }))
        }
        val request = Request.Builder()
            .url("$baseUrl/responses")
            .header("Authorization", "Bearer $apiKey")
            .header("Content-Type", "application/json")
            .header("OpenAI-Beta", "responses=experimental")
            .post(body.toString().toRequestBody("application/json".toMediaType()))
            .build()
        val channel = kotlinx.coroutines.channels.Channel<ProviderChunk>(capacity = kotlinx.coroutines.channels.Channel.BUFFERED)
        val src = EventSources.createFactory(httpClient).newEventSource(request, object : EventSourceListener() {
            override fun onEvent(eventSource: EventSource, id: String?, type: String?, data: String) {
                if (data == "[DONE]") { channel.trySend(ProviderChunk(finishReason = FinishReason.stop)); channel.close(); return }
                val obj = try { Json.parseToJsonElement(data).jsonObject } catch (e: Exception) { return }
                // The responses API streams "output_text.delta" events. Fall back to chat-completions style for forward compat.
                val text = (obj["delta"] as? JsonObject)?.get("text").let { (it as? JsonPrimitive)?.content }
                    ?: (obj["output_text"] as? JsonPrimitive)?.content
                if (text != null) channel.trySend(ProviderChunk(text = text))
                val done = (obj["type"] as? JsonPrimitive)?.content in setOf("response.completed", "response.done")
                if (done) channel.trySend(ProviderChunk(finishReason = FinishReason.stop))
            }
            override fun onFailure(eventSource: EventSource, t: Throwable?, response: okhttp3.Response?) {
                channel.trySend(ProviderChunk(error = ProviderError("http_error", t?.message ?: "unknown", retryable = true)))
                channel.close()
            }
            override fun onClosed(eventSource: EventSource) { channel.close() }
        })
        activeEventSource = src
        try {
            for (chunk in channel) emit(chunk)
        } finally {
            activeEventSource = null
        }
    }.flowOn(Dispatchers.IO)

    override suspend fun listModels(): List<String> = withContext(Dispatchers.IO) {
        // Use the same model list as openai (live catalog), so the user
        // doesn't see a parallel/different set in the picker.
        val request = Request.Builder()
            .url("https://api.openai.com/v1/models")
            .header("Authorization", "Bearer $apiKey")
            .build()
        val response = httpClient.newCall(request).execute()
        val raw = response.use { it.body?.string().orEmpty() }
        when (response.code) {
            401 -> throw ProviderCatalogException.AuthenticationException()
            429 -> throw ProviderCatalogException.RateLimitedException(
                retryAfterMs = response.header("Retry-After")?.toLongOrNull()?.times(1_000L),
            )
            in 200..299 -> Unit
            else -> throw ProviderCatalogException.NetworkException(
                message = "ChatGPT subscription catalog failed with HTTP ${response.code}.",
                statusCode = response.code,
            )
        }
        if (raw.isBlank()) return@withContext emptyList()
        val parsed = try {
            Json.parseToJsonElement(raw).jsonObject["data"] as? JsonArray
        } catch (e: Exception) {
            throw ProviderCatalogException.MalformedResponseException(
                "ChatGPT subscription catalog returned malformed JSON.", e,
            )
        } ?: return@withContext emptyList()
        // OpenAI's catalog has dozens of models; for the chatgpt subscription
        // picker we only surface the ones that ship with ChatGPT Plus/Pro
        // (gpt-4o, gpt-4.1, gpt-5, gpt-5-mini, o3, o4-mini). Users can pick
        // any model id from the openai catalog if they want.
        val preferred = setOf("gpt-5", "gpt-5-mini", "gpt-5-nano", "gpt-4.1", "gpt-4.1-mini", "gpt-4o", "gpt-4o-mini", "o3", "o4-mini")
        parsed.mapNotNull { (it as? JsonObject)?.get("id")?.let { it as? JsonPrimitive }?.content }
            .filter(String::isNotBlank)
            .filter { it in preferred }
            .ifEmpty {
                // Fallback: at least surface gpt-4o so the picker isn't empty.
                listOf("gpt-4o")
            }
    }

    override suspend fun cancel() {
        activeEventSource?.cancel()
        activeEventSource = null
    }
}
