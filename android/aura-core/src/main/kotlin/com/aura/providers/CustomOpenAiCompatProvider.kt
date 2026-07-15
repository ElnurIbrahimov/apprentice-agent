package com.aura.providers

import com.aura.security.SecureDataStore
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.flowOn
import kotlinx.coroutines.launch
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
 *
 * Holds the base URL, API key, and optional static-model override.
 * Distinct from [ProviderKeys] because (a) we need three coupled values
 * and (b) the user sets them as one operation in the Settings UI.
 *
 * Persisted to [SecureDataStore] under three keys (`custom_base_url`,
 * `custom_api_key`, `custom_model_override`) so the choice survives
 * process death. The [CustomOpenAiCompatProvider] reads from this
 * singleton on every chat call, so updates take effect on the next
 * request — no app restart required.
 */
@Singleton
class CustomEndpointState @Inject constructor(
    private val secureDataStore: SecureDataStore,
) {
    @Volatile private var _baseUrl: kotlin.String = ""
    @Volatile private var _apiKey: kotlin.String = ""
    @Volatile private var _modelOverride: List<kotlin.String> = emptyList()

    /** Reactive view of (baseUrl, apiKey, modelOverride) — emits on every set. */
    private val _state = MutableStateFlow(Triple("", "", emptyList<kotlin.String>()))
    val state: StateFlow<Triple<kotlin.String, kotlin.String, List<kotlin.String>>> = _state.asStateFlow()

    /** Reactive view of just the base URL, for UI binding. */
    val baseUrlFlow: StateFlow<kotlin.String> get() = baseUrlInternal
    private val baseUrlInternal = MutableStateFlow("")

    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)

    /** Async initial load. Sets fields once the DataStore read completes. */
    init {
        scope.launch { reload() }
    }

    /** Re-read from DataStore. Called on init and after the user saves. */
    suspend fun reload() {
        val url = secureDataStore.getString(KEY_BASE_URL).orEmpty()
        val key = secureDataStore.getString(KEY_API_KEY).orEmpty()
        val overrideRaw = secureDataStore.getString(KEY_MODEL_OVERRIDE).orEmpty()
        val override = overrideRaw.split('\n').map { it.trim() }.filter { it.isNotEmpty() }
        _baseUrl = url
        _apiKey = key
        _modelOverride = override
        baseUrlInternal.value = url
        _state.value = Triple(url, key, override)
    }

    val baseUrl: kotlin.String get() = _baseUrl
    val apiKey: kotlin.String get() = _apiKey
    val modelOverride: List<kotlin.String> get() = _modelOverride

    fun setEndpoint(
        baseUrl: kotlin.String,
        apiKey: kotlin.String,
        modelOverride: List<kotlin.String> = emptyList(),
    ) {
        val cleanUrl = baseUrl.trim().trimEnd('/')
        val cleanKey = apiKey.trim()
        _baseUrl = cleanUrl
        _apiKey = cleanKey
        _modelOverride = modelOverride
        baseUrlInternal.value = cleanUrl
        _state.value = Triple(cleanUrl, cleanKey, modelOverride)
        scope.launch {
            if (cleanUrl.isBlank()) secureDataStore.removeString(KEY_BASE_URL)
            else secureDataStore.putString(KEY_BASE_URL, cleanUrl)
            if (cleanKey.isBlank()) secureDataStore.removeString(KEY_API_KEY)
            else secureDataStore.putString(KEY_API_KEY, cleanKey)
            val overrideSerialized = modelOverride.joinToString("\n")
            if (overrideSerialized.isBlank()) secureDataStore.removeString(KEY_MODEL_OVERRIDE)
            else secureDataStore.putString(KEY_MODEL_OVERRIDE, overrideSerialized)
        }
    }

    fun isConfigured(): Boolean = _baseUrl.isNotBlank() && _apiKey.isNotBlank()

    /** Read the current values atomically. */
    fun snapshot(): Triple<kotlin.String, kotlin.String, List<kotlin.String>> =
        Triple(_baseUrl, _apiKey, _modelOverride)

    companion object {
        const val KEY_BASE_URL = "custom_base_url"
        const val KEY_API_KEY = "custom_api_key"
        const val KEY_MODEL_OVERRIDE = "custom_model_override"
    }
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
    private val httpClient: okhttp3.OkHttpClient,
) : Provider {
    override val prefix = "custom"
    override val displayName = "Custom Endpoint"

    @Volatile private var activeEventSource: okhttp3.sse.EventSource? = null

    override fun isConfigured(): Boolean = state.isConfigured()

    override fun chat(
        model: kotlin.String,
        messages: List<ProviderMessage>,
        options: ChatOptions,
        tools: List<ToolDefinition>,
    ): Flow<ProviderChunk> = flow {
        val (baseUrl, apiKey, _) = state.snapshot()
        if (baseUrl.isBlank() || apiKey.isBlank()) {
            emit(ProviderChunk(error = ProviderError("not_configured", "Custom endpoint not configured.", retryable = false)))
            return@flow
        }
        val body = kotlinx.serialization.json.buildJsonObject {
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
        val request = okhttp3.Request.Builder()
            .url("$baseUrl/chat/completions")
            .header("Authorization", "Bearer $apiKey")
            .header("Content-Type", "application/json")
            .post(body.toString().toRequestBody("application/json".toMediaType()))
            .build()
        val channel = kotlinx.coroutines.channels.Channel<ProviderChunk>(capacity = Channel.BUFFERED)
        val src = okhttp3.sse.EventSources.createFactory(httpClient).newEventSource(request, object : okhttp3.sse.EventSourceListener() {
            override fun onEvent(eventSource: okhttp3.sse.EventSource, id: kotlin.String?, type: kotlin.String?, data: kotlin.String) {
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
            override fun onFailure(eventSource: okhttp3.sse.EventSource, t: Throwable?, response: okhttp3.Response?) {
                channel.trySend(ProviderChunk(error = ProviderError("http_error", t?.message ?: "unknown", retryable = true)))
                channel.close()
            }
            override fun onClosed(eventSource: okhttp3.sse.EventSource) { channel.close() }
        })
        activeEventSource = src
        try {
            for (chunk in channel) emit(chunk)
        } finally {
            activeEventSource = null
        }
    }.flowOn(Dispatchers.IO)

    override suspend fun listModels(): List<kotlin.String> = withContext(Dispatchers.IO) {
        val (baseUrl, apiKey, modelOverride) = state.snapshot()
        if (baseUrl.isBlank() || apiKey.isBlank()) return@withContext emptyList()
        if (modelOverride.isNotEmpty()) return@withContext modelOverride
        val request = okhttp3.Request.Builder()
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
            .filter(kotlin.String::isNotBlank)
            .ifEmpty { throw ProviderCatalogException.EmptyCatalogException() }
    }

    override suspend fun cancel() {
        activeEventSource?.cancel()
        activeEventSource = null
    }
}
