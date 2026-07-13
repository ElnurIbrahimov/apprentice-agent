package com.aura.memory

import com.aura.providers.ProviderKeys
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import kotlinx.serialization.json.put
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import java.security.MessageDigest
import java.util.LinkedHashMap
import javax.inject.Inject
import javax.inject.Singleton

/**
 * Cloud semantic embedder with local fallback.
 *
 * Tries to call the Ollama Cloud embeddings API:
 *   POST https://api.ollama.com/api/embeddings
 *   Content-Type: application/json
 *   Authorization: Bearer <ollama-key>
 *
 * Falls back to [LocalEmbedder] if:
 *   - No Ollama Cloud API key is configured
 *   - The network request fails (timeout / HTTP error)
 *   - A parsing error occurs
 *
 * Results are cached in an in-memory LRU keyed by SHA-256 of the input text,
 * so re-embedding the same string (e.g. repeated recall queries) is instant.
 */
@Singleton
class CloudEmbedder @Inject constructor(
    private val localEmbedder: LocalEmbedder,
    private val providerKeys: ProviderKeys,
    private val httpClient: OkHttpClient,
) : Embedder {

    private val json = Json { ignoreUnknownKeys = true }
    private val mediaType = "application/json".toMediaType()

    /** In-memory LRU cache: SHA-256(hex) → FloatArray. */
    private val cache = object : LinkedHashMap<String, FloatArray>(16, 0.75f, true) {
        override fun removeEldestEntry(eldest: MutableMap.MutableEntry<String, FloatArray>): Boolean =
            size > MAX_CACHE_ENTRIES
    }

    override suspend fun embed(text: String): FloatArray = withContext(Dispatchers.IO) {
        val cacheKey = sha256Hex(text)

        // 1. Check cache
        synchronized(cache) {
            cache[cacheKey]?.let { return@withContext it }
        }

        // 2. Try cloud only for a selected Ollama catalog model.
        val apiKey = providerKeys.keyFor("ollama")
        val selected = providerKeys.embeddingModel
        val parts = selected.split(":", limit = 2)
        val model = parts.getOrNull(1)?.takeIf {
            parts.firstOrNull() == "ollama" && it.isNotBlank()
        }
        if (!apiKey.isNullOrBlank() && model != null) {
            try {
                val vec = cloudEmbed(text, apiKey, model)
                synchronized(cache) { cache[cacheKey] = vec }
                return@withContext vec
            } catch (_: Exception) {
                // Fall through to local fallback
            }
        }

        // 3. Fallback to local
        val vec = localEmbedder.embed(text)
        synchronized(cache) { cache[cacheKey] = vec }
        vec
    }

    /**
     * Makes the HTTP call to the Ollama Cloud embeddings API.
     * Throws on any failure (network, HTTP error, bad response).
     */
    private fun cloudEmbed(text: String, apiKey: String, model: String): FloatArray {
        val requestBody = buildJsonObject {
            put("model", model)
            put("prompt", text)
        }.toString().toRequestBody(mediaType)

        val request = Request.Builder()
            .url(EMBEDDINGS_URL)
            .addHeader("Authorization", "Bearer $apiKey")
            .post(requestBody)
            .build()

        val response = httpClient.newCall(request).execute()
        val body = response.body?.string() ?: throw RuntimeException("Empty response body")

        if (!response.isSuccessful) {
            throw RuntimeException(
                "Ollama Cloud embeddings returned ${response.code}: $body"
            )
        }

        val obj = json.parseToJsonElement(body).jsonObject
        val embeddingArray = obj["embedding"]?.jsonArray
            ?: // Some Ollama endpoints return { data: [{ embedding: [...] }] }
            obj["data"]?.jsonArray?.firstOrNull()?.jsonObject?.get("embedding")?.jsonArray
            ?: throw RuntimeException("No embedding in response: $body")

        return FloatArray(embeddingArray.size) { i ->
            embeddingArray[i].jsonPrimitive.content.toFloat()
        }
    }

    /** Full SHA-256 hex string for cache keys. */
    private fun sha256Hex(text: String): String {
        val md = MessageDigest.getInstance("SHA-256")
        val bytes = md.digest(text.toByteArray())
        return bytes.joinToString("") { "%02x".format(it) }
    }

    companion object {
        private const val EMBEDDINGS_URL = "https://api.ollama.com/api/embeddings"
        private const val MAX_CACHE_ENTRIES = 1000
    }
}
