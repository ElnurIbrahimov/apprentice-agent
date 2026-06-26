package com.aura.providers

import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import kotlinx.serialization.json.put
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody

/**
 * Google Gemini provider (Text-only).
 *
 * Uses the [streamGenerateContent] endpoint which returns newline-delimited
 * JSON objects (not SSE). The API key is passed as a query parameter.
 *
 * API: POST https://generativelanguage.googleapis.com/v1beta/models/{model}:streamGenerateContent?key={apiKey}
 *
 * Text-only (no image support in this pass). Image support will be added in
 * Task 4.1 via a [ProviderMessage.imageData] field.
 *
 * The API key is read from [ProviderKeys] on every [chat] call, not at
 * construction time, so changes the user makes in the Settings UI take
 * effect immediately without restarting the app.
 */
class GeminiProvider(
    private val providerKeys: ProviderKeys,
    private val httpClient: OkHttpClient,
) : Provider {

    override val prefix = "gemini"
    override val displayName = "Google Gemini"

    /** Live API key, looked up at call time. */
    private val apiKey: String get() = providerKeys.keyFor(prefix) ?: ""

    @Volatile private var activeCall: okhttp3.Call? = null

    override fun isConfigured(): Boolean = providerKeys.isConfigured(prefix)

    override fun chat(
        model: String,
        messages: List<ProviderMessage>,
        options: ChatOptions,
        tools: List<ToolDefinition>,
    ): Flow<ProviderChunk> = flow {
        val body = buildRequestBody(messages, options)
        val request = Request.Builder()
            .url("https://generativelanguage.googleapis.com/v1beta/models/$model:streamGenerateContent?key=$apiKey")
            .addHeader("Content-Type", "application/json")
            .post(body.toString().toRequestBody("application/json".toMediaType()))
            .build()
        val call = httpClient.newCall(request)
        activeCall = call
        try {
            call.execute().use { resp ->
                if (!resp.isSuccessful) {
                    // Try to read error detail from body
                    val errorDetail = try {
                        val errBody = resp.body?.string() ?: ""
                        val errObj = Json.parseToJsonElement(errBody).jsonObject
                        val error = errObj["error"]?.jsonObject
                        val msg = error?.get("message")?.jsonPrimitive?.content
                        msg ?: resp.message
                    } catch (_: Exception) {
                        resp.message
                    }
                    emit(ProviderChunk(
                        error = ProviderError("http_${resp.code}", errorDetail, retryable = resp.code in 500..599)
                    ))
                    return@flow
                }
                val source = resp.body?.source() ?: return@flow
                var sawFinish = false
                while (true) {
                    val line = source.readUtf8Line() ?: break
                    if (line.isBlank()) continue
                    val obj = try { Json.parseToJsonElement(line).jsonObject } catch (_: Exception) { continue }

                    // Parse candidates[0].content.parts[0].text
                    val candidates = obj["candidates"]?.jsonArray
                    val candidate = candidates?.firstOrNull()?.jsonObject
                    val content = candidate?.get("content")?.jsonObject
                    val parts = content?.get("parts")?.jsonArray
                    val text = parts?.firstOrNull()?.jsonObject?.get("text")?.jsonPrimitive?.content
                    if (text != null) emit(ProviderChunk(text = text))

                    // Check finish reason on the candidate
                    val finishReason = candidate?.get("finishReason")?.jsonPrimitive?.content
                    if (finishReason != null) {
                        sawFinish = true
                        val reason = when (finishReason) {
                            "STOP" -> FinishReason.stop
                            "MAX_TOKENS" -> FinishReason.length
                            "SAFETY", "RECITATION", "PROHIBITED" -> FinishReason.stop
                            else -> FinishReason.stop
                        }
                        // Also check usage metadata if present on the final chunk
                        val usage = parseUsage(obj)
                        emit(ProviderChunk(finishReason = reason, usage = usage))
                    }
                }
                // If the stream ended without a finishReason, emit a terminal stop
                if (!sawFinish) {
                    emit(ProviderChunk(finishReason = FinishReason.stop))
                }
            }
        } catch (e: kotlinx.coroutines.CancellationException) {
            throw e
        } catch (e: Exception) {
            emit(ProviderChunk(error = ProviderError("stream_error", e.message ?: "unknown", retryable = true)))
        } finally {
            activeCall = null
        }
    }

    override suspend fun listModels(): List<String> = listOf(
        "gemini-1.5-flash",
        "gemini-1.5-pro",
    )

    override suspend fun cancel() {
        activeCall?.cancel()
        activeCall = null
    }

    // ---- internal helpers ----

    private fun buildRequestBody(
        messages: List<ProviderMessage>,
        options: ChatOptions,
    ): JsonObject = buildJsonObject {
        // Gemini supports system_instruction at the top level
        val systemMessages = messages.filter { it.role == ProviderMessage.Role.system }
        if (systemMessages.isNotEmpty()) {
            val systemText = systemMessages.joinToString("\n\n") { it.content }
            put("system_instruction", buildJsonObject {
                put("parts", JsonArray(listOf(buildJsonObject {
                    put("text", systemText)
                })))
            })
        }

        // Filter out system messages (already handled above) and map roles
        val chatMessages = messages.filter { it.role != ProviderMessage.Role.system }
        put("contents", JsonArray(chatMessages.map { msg ->
            buildJsonObject {
                val geminiRole = when (msg.role) {
                    ProviderMessage.Role.assistant -> "model"
                    else -> "user"
                }
                put("role", geminiRole)
                put("parts", JsonArray(listOf(buildJsonObject {
                    put("text", msg.content)
                })))
            }
        }))

        put("generationConfig", buildJsonObject {
            put("temperature", options.temperature)
            put("topP", options.topP)
            options.maxTokens?.let { put("maxOutputTokens", it) }
            if (options.stop.isNotEmpty()) {
                put("stopSequences", JsonArray(options.stop.map { JsonPrimitive(it) }))
            }
        })
    }

    /**
     * Parses usage metadata from the final Gemini response chunk.
     *
     * Gemini returns usageMetadata at the top level:
     * { "usageMetadata": { "promptTokenCount": N, "candidatesTokenCount": M, "totalTokenCount": T } }
     */
    private fun parseUsage(obj: JsonObject): Usage? {
        return try {
            val meta = obj["usageMetadata"]?.jsonObject ?: return null
            Usage(
                promptTokens = meta["promptTokenCount"]?.jsonPrimitive?.content?.toIntOrNull() ?: 0,
                completionTokens = meta["candidatesTokenCount"]?.jsonPrimitive?.content?.toIntOrNull() ?: 0,
                totalTokens = meta["totalTokenCount"]?.jsonPrimitive?.content?.toIntOrNull() ?: 0,
            )
        } catch (_: Exception) {
            null
        }
    }
}
