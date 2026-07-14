package com.aura.providers

import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineStart
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.awaitCancellation
import kotlinx.coroutines.cancelAndJoin
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.flowOn
import kotlinx.coroutines.launch
import kotlinx.coroutines.runInterruptible
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
 * JSON objects (not SSE). The API key is passed via the `X-Goog-Api-Key`
 * header.
 *
 * API: POST https://generativelanguage.googleapis.com/v1beta/models/{model}:streamGenerateContent
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
    private val baseUrl: String = "https://generativelanguage.googleapis.com/v1beta",
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
        val key = apiKey
        if (key.isBlank()) {
            emit(ProviderChunk(error = ProviderError("missing_api_key", "Gemini API key not configured", retryable = false)))
            return@flow
        }
        val request = Request.Builder()
            .url("$baseUrl/models/$model:streamGenerateContent")
            .addHeader("Content-Type", "application/json")
            .addHeader("X-Goog-Api-Key", key)
            .post(body.toString().toRequestBody("application/json".toMediaType()))
            .build()
        val call = httpClient.newCall(request)
        activeCall = call
        coroutineScope {
            val cancellationGuard = launch(start = CoroutineStart.UNDISPATCHED) {
                try {
                    awaitCancellation()
                } finally {
                    call.cancel()
                }
            }
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
                    return@use
                }
                val source = resp.body?.source() ?: return@use
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
            } catch (e: CancellationException) {
                throw e
            } catch (e: Exception) {
                currentCoroutineContext().ensureActive()
                emit(ProviderChunk(error = ProviderError("stream_error", e.message ?: "unknown", retryable = true)))
            } finally {
                cancellationGuard.cancelAndJoin()
                if (activeCall === call) activeCall = null
            }
        }
    }.flowOn(Dispatchers.IO)

    override suspend fun listModels(): List<String> {
        return try {
            runInterruptible(Dispatchers.IO) {
                val request = Request.Builder()
                .url("$baseUrl/models")
                .addHeader("Content-Type", "application/json")
                .addHeader("X-Goog-Api-Key", apiKey)
                .get()
                .build()
                httpClient.newCall(request).execute().use { response ->
                    when (response.code) {
                        401 -> throw ProviderCatalogException.AuthenticationException()
                        429 -> throw ProviderCatalogException.RateLimitedException(
                            retryAfterMs = response.header("Retry-After")?.toLongOrNull()?.times(1_000L),
                        )
                        in 200..299 -> Unit
                        else -> throw ProviderCatalogException.NetworkException(
                            message = "Gemini catalog request failed with HTTP ${response.code}.",
                            statusCode = response.code,
                        )
                    }
                    val body = response.body?.string()?.takeIf(String::isNotBlank)
                        ?: throw ProviderCatalogException.MalformedResponseException(
                            "Gemini returned an empty model catalog response.",
                        )
                    val models = try {
                        Json.parseToJsonElement(body).jsonObject["models"] as? JsonArray
                            ?: throw ProviderCatalogException.MalformedResponseException(
                                "Missing models[] in Gemini response.",
                            )
                    } catch (e: ProviderCatalogException) {
                        throw e
                    } catch (e: Exception) {
                        throw ProviderCatalogException.MalformedResponseException(
                            "Gemini returned malformed model catalog JSON.",
                            e,
                        )
                    }
                    models.mapNotNull { (it as? JsonObject)?.get("name")?.let { n -> (n as? JsonPrimitive)?.content } }
                        .mapNotNull { fullName ->
                            fullName.removePrefix("models/").takeIf { it.isNotBlank() }
                        }.ifEmpty { throw ProviderCatalogException.EmptyCatalogException() }
                }
            }
        } catch (e: kotlinx.coroutines.CancellationException) {
            throw e
        } catch (e: ProviderCatalogException) {
            throw e
        } catch (e: java.io.IOException) {
            currentCoroutineContext().ensureActive()
            throw ProviderCatalogException.NetworkException(cause = e)
        } catch (e: Exception) {
            throw ProviderCatalogException.MalformedResponseException(
                "Gemini model catalog could not be read.",
                e,
            )
        }
    }

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
