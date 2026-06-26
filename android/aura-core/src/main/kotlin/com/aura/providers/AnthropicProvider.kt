package com.aura.providers

import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import kotlinx.serialization.json.put
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import okhttp3.sse.EventSources

/**
 * Anthropic Messages API. Tool use blocks are converted to ToolCall chunks.
 *
 * Like [OllamaCloudProvider], the API key is read from [ProviderKeys] on
 * every [chat] call so user changes in Settings take effect immediately.
 */
class AnthropicProvider(
    private val providerKeys: ProviderKeys,
    private val httpClient: OkHttpClient,
) : Provider {
    override val prefix = "anthropic"
    override val displayName = "Anthropic"

    /** Live API key, looked up at call time. */
    private val apiKey: String get() = providerKeys.keyFor(prefix) ?: ""

    @Volatile private var activeCall: okhttp3.Call? = null

    override fun isConfigured(): Boolean = providerKeys.isConfigured(prefix)

    override fun chat(model: String, messages: List<ProviderMessage>, options: ChatOptions, tools: List<ToolDefinition>): Flow<ProviderChunk> = flow {
        val (systemPrompt, anthropicMessages) = splitSystem(messages)
        val body = buildJsonObject {
            put("model", model)
            put("stream", true)
            put("max_tokens", options.maxTokens ?: 4096)
            put("temperature", options.temperature)
            systemPrompt?.let { put("system", it) }
            put("messages", kotlinx.serialization.json.JsonArray(anthropicMessages.map { msg ->
                buildJsonObject {
                    put("role", msg.role.name)
                    put("content", kotlinx.serialization.json.JsonArray(listOf(buildJsonObject {
                        put("type", "text")
                        put("text", msg.content)
                    })))
                }
            }))
            if (tools.isNotEmpty()) {
                put("tools", kotlinx.serialization.json.JsonArray(tools.map { tool ->
                    buildJsonObject {
                        put("name", tool.name)
                        put("description", tool.description)
                        put("input_schema", kotlinx.serialization.json.Json.parseToJsonElement(Json.encodeToString(ToolParameters.serializer(), tool.parameters)))
                    }
                }))
            }
        }
        val request = Request.Builder()
            .url("https://api.anthropic.com/v1/messages")
            .addHeader("x-api-key", apiKey)
            .addHeader("anthropic-version", "2023-06-01")
            .addHeader("Content-Type", "application/json")
            .post(body.toString().toRequestBody("application/json".toMediaType()))
            .build()
        val call = httpClient.newCall(request)
        activeCall = call
        try {
            call.execute().use { resp ->
                if (!resp.isSuccessful) {
                    emit(ProviderChunk(error = ProviderError("http_${resp.code}", resp.message, retryable = resp.code in 500..599)))
                    return@flow
                }
                val source = resp.body?.source() ?: return@flow
                // SSE stream. readUtf8Line() blocks until a line arrives (or returns
                // null at EOF). We process each 'data: ...' line. The two terminal
                // events we care about are 'message_stop' (Anthropic's normal end)
                // and finish_reason=tool_calls in 'message_delta' (which we also
                // map to FinishReason.tool_calls so the loop dispatches tools).
                while (true) {
                    val line = source.readUtf8Line() ?: break
                    if (line.isEmpty()) continue
                    if (!line.startsWith("data: ")) continue
                    val data = line.removePrefix("data: ").trim()
                    if (data.isEmpty()) continue
                    val obj = try { Json.parseToJsonElement(data).jsonObject } catch (e: Exception) { continue }
                    when ((obj["type"] as? JsonPrimitive)?.content) {
                        "content_block_start" -> {
                            val block = (obj["content_block"] as? JsonObject)
                            if (block?.get("type")?.let { (it as JsonPrimitive).content } == "tool_use") {
                                val id = (block["id"] as? JsonPrimitive)?.content ?: ""
                                val name = (block["name"] as? JsonPrimitive)?.content ?: ""
                                // Emit with empty arguments; downstream BrainChunk
                                // emits ToolCallStart (id+name, no args) which the
                                // agentic loop will associate with subsequent
                                // input_json_delta events keyed by id.
                                emit(ProviderChunk(toolCall = ToolCall(id, name, "")))
                            }
                        }
                        "content_block_delta" -> {
                            val delta = (obj["delta"] as? JsonObject) ?: continue
                            when ((delta["type"] as? JsonPrimitive)?.content) {
                                "text_delta" -> {
                                    val text = (delta["text"] as? JsonPrimitive)?.content
                                    if (text != null) emit(ProviderChunk(text = text))
                                }
                                "input_json_delta" -> {
                                    val partial = (delta["partial_json"] as? JsonPrimitive)?.content
                                    if (partial != null) emit(ProviderChunk(toolCall = ToolCall("", "", partial)))
                                }
                            }
                        }
                        "message_stop" -> emit(ProviderChunk(finishReason = FinishReason.stop))
                        "message_delta" -> {
                            val stop = (obj["delta"] as? JsonObject)?.get("stop_reason")
                            val reason = when ((stop as? JsonPrimitive)?.content) {
                                "end_turn" -> FinishReason.stop
                                "max_tokens" -> FinishReason.length
                                "tool_use" -> FinishReason.tool_calls
                                else -> null
                            }
                            if (reason != null) emit(ProviderChunk(finishReason = reason))
                        }
                    }
                }
            }
        } finally {
            activeCall = null
        }
    }

    override suspend fun listModels(): List<String> = listOf(
        "claude-sonnet-4-5", "claude-opus-4-1", "claude-haiku-4-5"
    )

    override suspend fun cancel() {
        activeCall?.cancel()
        activeCall = null
    }

    private fun splitSystem(messages: List<ProviderMessage>): Pair<String?, List<ProviderMessage>> {
        val sys = messages.filter { it.role == ProviderMessage.Role.system }.joinToString("\n\n") { it.content }
        val rest = messages.filter { it.role != ProviderMessage.Role.system }
        return sys.ifBlank { null } to rest
    }
}
