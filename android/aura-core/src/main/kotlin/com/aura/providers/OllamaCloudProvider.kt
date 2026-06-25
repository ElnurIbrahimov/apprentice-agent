package com.aura.providers

import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonObject
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
 * OpenAI-compatible chat completions client. Works with Ollama Cloud, DeepSeek, OpenAI, NVIDIA, vLLM, etc.
 * Mirrors aura/providers/openai_compat.py.
 */
class OllamaCloudProvider(
    override val prefix: String,
    override val displayName: String,
    private val baseUrl: String,
    private val apiKey: String,
    private val httpClient: OkHttpClient,
) : Provider {

    @Volatile private var activeEventSource: EventSource? = null

    override fun isConfigured(): Boolean = apiKey.isNotBlank()

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
                val choice = (obj["choices"] as? kotlinx.serialization.json.JsonArray)?.firstOrNull()?.jsonObject ?: return
                val delta = (choice["delta"] as? JsonObject) ?: return
                val text = (delta["content"] as? kotlinx.serialization.json.JsonPrimitive)?.content
                if (text != null) channel.trySend(ProviderChunk(text = text))
                val toolCalls = (delta["tool_calls"] as? kotlinx.serialization.json.JsonArray)
                toolCalls?.forEach { tc ->
                    val tco = tc.jsonObject
                    val fn = tco["function"]?.jsonObject
                    if (fn != null) {
                        val id = (tco["id"] as? kotlinx.serialization.json.JsonPrimitive)?.content ?: ""
                        val name = (fn["name"] as? kotlinx.serialization.json.JsonPrimitive)?.content ?: ""
                        val args = (fn["arguments"] as? kotlinx.serialization.json.JsonPrimitive)?.content ?: ""
                        channel.trySend(ProviderChunk(toolCall = ToolCall(id = id, name = name, arguments = args)))
                    }
                }
                val finish = (choice["finish_reason"] as? kotlinx.serialization.json.JsonPrimitive)?.content
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
        for (chunk in channel) emit(chunk)
    }

    override suspend fun listModels(): List<String> {
        val req = Request.Builder().url("$baseUrl/models")
            .addHeader("Authorization", "Bearer $apiKey")
            .build()
        httpClient.newCall(req).execute().use { resp ->
            if (!resp.isSuccessful) return emptyList()
            val body = resp.body?.string() ?: return emptyList()
            val obj = Json.parseToJsonElement(body).jsonObject
            val data = (obj["data"] as? kotlinx.serialization.json.JsonArray) ?: return emptyList()
            return data.mapNotNull { (it as? JsonObject)?.get("id")?.let { id -> (id as? kotlinx.serialization.json.JsonPrimitive)?.content } }
        }
    }

    override suspend fun cancel() {
        activeEventSource?.cancel()
        activeEventSource = null
    }

    private fun buildRequest(model: String, messages: List<ProviderMessage>, options: ChatOptions, tools: List<ToolDefinition>, stream: Boolean): Request {
        val body = buildJsonObject {
            put("model", model)
            put("stream", stream)
            put("temperature", options.temperature)
            put("top_p", options.topP)
            options.maxTokens?.let { put("max_tokens", it) }
            put("messages", kotlinx.serialization.json.JsonArray(messages.map { msg ->
                buildJsonObject {
                    put("role", msg.role.name)
                    put("content", msg.content)
                }
            }))
            if (tools.isNotEmpty()) {
                put("tools", kotlinx.serialization.json.JsonArray(tools.map { tool ->
                    buildJsonObject {
                        put("type", "function")
                        put("function", buildJsonObject {
                            put("name", tool.name)
                            put("description", tool.description)
                            put("parameters", kotlinx.serialization.json.Json.parseToJsonElement(Json.encodeToString(ToolParameters.serializer(), tool.parameters)))
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
}
