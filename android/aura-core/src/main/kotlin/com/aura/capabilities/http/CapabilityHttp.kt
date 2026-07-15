package com.aura.capabilities.http

import com.aura.capabilities.CapabilityCatalogException
import com.aura.providers.ProviderCatalogException
import com.aura.providers.ProviderError
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.runInterruptible
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
import kotlin.coroutines.coroutineContext

/**
 * Common HTTP primitives shared by every non-chat capability provider.
 * Keeps the per-provider implementation files small and ensures
 * consistent error handling (401 -> Authentication, 429 -> RateLimit, ...).
 */
object CapabilityHttp {
    val json: Json = Json { ignoreUnknownKeys = true; isLenient = true }

    fun applyAuth(builder: Request.Builder, apiKey: String, scheme: String = "Bearer") {
        if (apiKey.isBlank()) return
        builder.header("Authorization", "$scheme $apiKey")
    }

    fun buildJsonBody(vararg pairs: Pair<String, Any?>): String = buildJsonObject {
        for ((k, v) in pairs) {
            when (v) {
                null -> Unit
                is Number -> put(k, v)
                is Boolean -> put(k, v)
                is String -> put(k, v)
                is JsonObject -> put(k, v)
                is JsonArray -> put(k, v)
                is List<*> -> {
                    val arr = JsonArray(v.map { item ->
                        when (item) {
                            is String -> JsonPrimitive(item)
                            is Number -> JsonPrimitive(item)
                            is Boolean -> JsonPrimitive(item)
                            is JsonObject -> item
                            else -> JsonPrimitive(item.toString())
                        }
                    })
                    put(k, arr)
                }
                else -> put(k, v.toString())
            }
        }
    }.toString()

    fun postJson(
        client: OkHttpClient,
        url: String,
        apiKey: String,
        body: String,
        extraHeaders: Map<String, String> = emptyMap(),
    ): okhttp3.Response {
        val builder = Request.Builder().url(url)
        if (apiKey.isNotBlank()) builder.header("Authorization", "Bearer $apiKey")
        builder.header("Content-Type", "application/json")
        for ((k, v) in extraHeaders) builder.header(k, v)
        val req = builder.post(body.toRequestBody("application/json".toMediaType())).build()
        return client.newCall(req).execute()
    }

    fun classify(response: okhttp3.Response, rawBody: String) {
        when (response.code) {
            401, 403 -> throw CapabilityCatalogException.AuthenticationException()
            429 -> throw CapabilityCatalogException.RateLimitedException(
                retryAfterMs = response.header("Retry-After")?.toLongOrNull()?.times(1_000L)
            )
            in 500..599 -> throw CapabilityCatalogException.NetworkException(
                message = "Provider returned HTTP ${response.code}.",
                statusCode = response.code,
            )
            in 200..299 -> return
            else -> throw CapabilityCatalogException.NetworkException(
                message = "Provider returned HTTP ${response.code}.",
                statusCode = response.code,
            )
        }
        if (rawBody.isBlank()) throw CapabilityCatalogException.MalformedResponseException("Empty response body")
    }

    /**
     * Run [block] on Dispatchers.IO with cooperative cancellation. Callers
     * inside coroutineScope will throw CancellationException if the parent
     * Job is cancelled.
     */
    suspend fun <T> withIoCancellation(block: () -> T): T {
        return runInterruptible(Dispatchers.IO) { block() }.also {
            currentCoroutineContext().ensureActive()
        }
    }
}
