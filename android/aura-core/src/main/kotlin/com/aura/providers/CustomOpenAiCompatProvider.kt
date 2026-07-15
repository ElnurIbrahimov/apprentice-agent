package com.aura.providers

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.channels.Channel
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
 * Per-process mutable state for the user's "Custom Endpoint" provider.
 * Distinct from [ProviderKeys] because we need to store two coupled
 * values (base URL + API key) and a list of static models, and we want
 * the user to set them as one operation.
 */
@Singleton
class CustomEndpointState @Inject constructor() {
    @Volatile var baseUrl: String = ""
    @Volatile var apiKey: String = ""
    @Volatile var modelOverride: List<String> = emptyList()

    fun setEndpoint(baseUrl: String, apiKey: String, modelOverride: List<String> = emptyList()) {
        this.baseUrl = baseUrl.trim().trimEnd('/')
        this.apiKey = apiKey.trim()
        this.modelOverride = modelOverride
    }

    fun isConfigured(): Boolean = baseUrl.isNotBlank() && apiKey.isNotBlank()

    /** Read the current values atomically. */
    fun snapshot(): Triple<String, String, List<String>> = Triple(baseUrl, apiKey, modelOverride)
}

/**
 * User-defined OpenAI-compatible chat-completions endpoint.
 *
 * Acts as a first-class provider with prefix `custom`. The user supplies a
 * base URL and API key at runtime via the Settings UI. The provider fetches
 * models from the live `/models` endpoint (or uses [CustomEndpointState.modelOverride]
 * if the user supplied static models).
 *
 * Implementation is intentionally self-contained (not a subclass of
 * [OpenAiCompatProvider]) because the parent's baseUrl is captured in its
 * constructor — to make a dynamic URL we'd have to rebuild the provider on
 * every set, which would invalidate the Hilt graph.
 */
class CustomOpenAiCompatProvider(
    private val state: CustomEndpointState,
    private val httpClient: OkHttpClient,
) : Provider {
    override val prefix = "custom"
    override val displayName = "Custom Endpoint"

    @Volatile private var activeEventSource: EventSource? = null

    override fun isConfigured(): Boolean = state.isConfigured()

    override fun chat(
        model: String,
        messages: List<ProviderMessage>,
        options: ChatOptions,
        tools: List<ToolDefinition>,
    ): Flow<ProviderChunk> = flow {
        val (baseUrl, apiKey, _) = state.snapshot()
        if (baseUrl.isBlank() || apiKey.isBlank()) {
            emit(ProviderChunk(error = ProviderError("not_configured", "Custom endpoint not configured.", retryable = false)))
            return@flow
        }
        val body = buildJsonObject {
            put("model", model)
            put("stream", true)
            put("temperature", options.temperature)
            put("top_p", options.topP)
            options.maxTokens?.let { put("max_tokens", it) }
            put("messages", JsonArray(messages.map { msg ->
                buildJsonObject {
                    put("role", msg.role.name)
                    put("content", msg.content)
                }
            }))
            if (tools.isNotEmpty()) {
                put("tools", JsonArray(tools.map { tool ->
                    buildJsonObject {
                        put("type", "function")
                        put("function", buildJsonObject {
                            put("name", tool.name)
                            put("description", tool.description)
                            put("parameters", Json.parseToJsonElement(Json.encodeToString(ToolParameters.serializer(), tool.parameters)))
                        })
                    }
                }))
            }
        }
        val request = Request.Builder()
            .url("$baseUrl/chat/completions")
            .header("Authorization", "Bearer $apiKey")
            .header("Content-Type", "application/json")
            .post(body.toString().toRequestBody("application/json".toMediaType()))
            .build()
        val channel = Channel<ProviderChunk>(capacity = Channel.BUFFERED)
        val src = EventSources.createFactory(httpClient).newEventSource(request, object : EventSourceListener() {
            override fun onEvent(eventSource: EventSource, id: String?, type: String?, data: String) {
                if (data == "[DONE]") { channel.trySend(ProviderChunk(finishReason = FinishReason.stop)); channel.close(); return }
                val obj = try { Json.parseToJsonElement(data).jsonObject } catch (e: Exception) { return }
                val choice = (obj["choices"] as? JsonArray)?.firstOrNull()?.jsonObject ?: return
                val delta = (choice["delta"] as? JsonObject) ?: return
                val text = (delta["content"] as? JsonPrimitive)?.content
                if (text != null) channel.trySend(ProviderChunk(text = text))
                val toolCalls = (delta["tool_calls"] as? JsonArray)
                toolCalls?.forEach { tc ->
                    val tco = tc.jsonObject
                    val fn = tco["function"]?.jsonObject
                    if (fn != null) {
                        val tcId = (tco["id"] as? JsonPrimitive)?.content ?: ""
                        val name = (fn["name"] as? JsonPrimitive)?.content ?: ""
                        val args = (fn["arguments"] as? JsonPrimitive)?.content ?: ""
                        channel.trySend(ProviderChunk(toolCall = ToolCall(id = tcId, name = name, arguments = args)))
                    }
                }
                val finish = (choice["finish_reason"] as? JsonPrimitive)?.content
                if (finish == "stop" || finish == "length" || finish == "tool_calls") {
                    val reason = when (finish) { "stop" -> FinishReason.stop; "length" -> FinishReason.length; "tool_calls" -> FinishReason.tool_calls; else -> FinishReason.stop }
                    channel.trySend(ProviderChunk(finishReason = reason))
                }
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
        val (baseUrl, apiKey, modelOverride) = state.snapshot()
        if (baseUrl.isBlank() || apiKey.isBlank()) return@withContext emptyList()
        if (modelOverride.isNotEmpty()) return@withContext modelOverride
        val request = Request.Builder()
            .url("$baseUrl/models")
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
                message = "Custom endpoint catalog returned HTTP ${response.code}.",
                statusCode = response.code,
            )
        }
        if (raw.isBlank()) return@withContext emptyList()
        val parsed = try {
            Json.parseToJsonElement(raw).jsonObject["data"] as? JsonArray
        } catch (e: Exception) {
            throw ProviderCatalogException.MalformedResponseException(
                "Custom endpoint returned malformed model catalog JSON.", e,
            )
        } ?: return@withContext emptyList()
        parsed.mapNotNull { (it as? JsonObject)?.get("id")?.let { it as? JsonPrimitive }?.content }
            .filter(String::isNotBlank)
            .ifEmpty { throw ProviderCatalogException.EmptyCatalogException() }
    }

    override suspend fun cancel() {
        activeEventSource?.cancel()
        activeEventSource = null
    }
}
