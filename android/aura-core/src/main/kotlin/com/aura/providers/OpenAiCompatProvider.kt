package com.aura.providers

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.withContext
import kotlinx.serialization.encodeToString
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

/**
 * Generic provider for any OpenAI-compatible chat completions API.
 *
 * Works with OpenAI, DeepSeek, Ollama Cloud, Groq, NVIDIA, vLLM, Together, Fireworks,
 * and any other service that exposes `/v1/chat/completions` and `/v1/models`.
 *
 * The API key is read from [ProviderKeys] on every [chat] call, not at
 * construction time, so changes the user makes in the Settings UI take
 * effect immediately without restarting the app.
 *
 * Subclasses can override [defaultModels] to provide a hardcoded list instead
 * of fetching from the `/models` endpoint.
 */
open class OpenAiCompatProvider(
    override val prefix: String,
    override val displayName: String,
    private val baseUrl: String,
    private val providerKeys: ProviderKeys,
    private val httpClient: OkHttpClient,
    protected val defaultModels: List<String> = emptyList(),
) : Provider {

    @Volatile private var activeEventSource: EventSource? = null

    /**
     * The current API key, looked up at call time. Returns blank if the user
     * hasn't set a key for this provider; [isConfigured] will return false in
     * that case and the chat request will fail with a clear 401.
     */
    private val apiKey: String get() = providerKeys.keyFor(prefix) ?: ""

    override fun isConfigured(): Boolean = providerKeys.isConfigured(prefix)

    override fun chat(
        model: String,
        messages: List<ProviderMessage>,
        options: ChatOptions,
        tools: List<ToolDefinition>,
    ): Flow<ProviderChunk> = flow {
        val request = buildRequest(model, messages, options, tools, stream = true)
        // We'll consume the SSE via callback to keep simple; switch to channel-based emission
        val channel = kotlinx.coroutines.channels.Channel<ProviderChunk>(capacity = kotlinx.coroutines.channels.Channel.BUFFERED)
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
        // Bound the per-request read so a misbehaving server that never sends
        // [DONE] and never closes the stream can't hang the agent forever. The
        // OkHttp read timeout (configured in ProviderModule) is the primary
        // backstop; this is a defensive upper bound.
        try {
            kotlinx.coroutines.withTimeout(STREAM_READ_TIMEOUT_MS) {
                for (chunk in channel) emit(chunk)
            }
        } catch (_: kotlinx.coroutines.TimeoutCancellationException) {
            // Synthesize a finish so downstream loops terminate cleanly.
            emit(ProviderChunk(finishReason = FinishReason.stop))
        } finally {
            activeEventSource = null
        }
    }

    override suspend fun listModels(): List<String> = withContext(Dispatchers.IO) {
        if (defaultModels.isNotEmpty()) return@withContext defaultModels
        val req = Request.Builder().url("$baseUrl/models")
            .addHeader("Authorization", "Bearer $apiKey")
            .build()
        httpClient.newCall(req).execute().use { resp ->
            if (!resp.isSuccessful) {
                val detail = resp.body?.string()?.take(200)?.trim().orEmpty()
                val suffix = if (detail.isNotBlank()) " — $detail" else ""
                throw IllegalStateException("HTTP ${resp.code} ${resp.message}$suffix".trim())
            }
            val body = resp.body?.string() ?: throw IllegalStateException("Empty /models response")
            val obj = Json.parseToJsonElement(body).jsonObject
            val data = (obj["data"] as? JsonArray)
                ?: throw IllegalStateException("Missing data[] in /models response")
            data.mapNotNull { (it as? JsonObject)?.get("id")?.let { id -> (id as? JsonPrimitive)?.content } }
                .ifEmpty { throw IllegalStateException("Provider returned zero models") }
        }
    }

    override suspend fun cancel() {
        activeEventSource?.cancel()
        activeEventSource = null
    }

    private fun buildRequest(
        model: String,
        messages: List<ProviderMessage>,
        options: ChatOptions,
        tools: List<ToolDefinition>,
        stream: Boolean,
    ): Request {
        val body = buildJsonObject {
            put("model", model)
            put("stream", stream)
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
        return Request.Builder()
            .url("$baseUrl/chat/completions")
            .addHeader("Authorization", "Bearer $apiKey")
            .addHeader("Content-Type", "application/json")
            .post(body.toString().toRequestBody("application/json".toMediaType()))
            .build()
    }

    companion object {
        // 5 minutes: long enough for a slow model + max_tokens worth of tokens,
        // short enough that a misbehaving server doesn't hang the agent loop.
        const val STREAM_READ_TIMEOUT_MS = 5L * 60L * 1000L
    }
}
