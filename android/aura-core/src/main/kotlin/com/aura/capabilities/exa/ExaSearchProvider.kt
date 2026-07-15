package com.aura.capabilities.exa

import com.aura.capabilities.WebSearchProvider
import com.aura.capabilities.WebSearchRequest
import com.aura.capabilities.WebSearchResult
import com.aura.capabilities.http.CapabilityHttp
import com.aura.capabilities.http.asJsonObjectOrNull
import com.aura.capabilities.http.stringOrNull
import com.aura.providers.ProviderKeys
import kotlinx.coroutines.withContext
import okhttp3.OkHttpClient
import javax.inject.Inject
import javax.inject.Singleton

/**
 * Exa AI search. POST /search with x-api-key header.
 * https://docs.exa.ai/reference/search
 */
@Singleton
class ExaSearchProvider @Inject constructor(
    private val client: OkHttpClient,
    private val providerKeys: ProviderKeys,
) : WebSearchProvider {
    override val prefix = "exa"
    override val displayName = "Exa"
    private val apiKey: String get() = providerKeys.keyFor(prefix).orEmpty()
    override fun isConfigured(): Boolean = apiKey.isNotBlank()

    override suspend fun search(req: WebSearchRequest): List<WebSearchResult> = withContext(kotlinx.coroutines.Dispatchers.IO) {
        val body = CapabilityHttp.buildJsonBody(
            "query" to req.query,
            "numResults" to req.numResults,
            "contents" to buildMap<String, Any> {
                put("text", req.includeText)
                put("highlights", true)
            },
        )
        val response = CapabilityHttp.postJson(
            client = client,
            url = "https://api.exa.ai/search",
            apiKey = apiKey,
            body = body,
            extraHeaders = mapOf("x-api-key" to apiKey),
        )
        val raw = response.use { it.body?.string().orEmpty() }
        CapabilityHttp.classify(response, raw)
        val obj = CapabilityHttp.json.parseToJsonElement(raw).asJsonObjectOrNull() ?: return@withContext emptyList()
        val results = (obj["results"] as? kotlinx.serialization.json.JsonArray) ?: return@withContext emptyList()
        results.mapNotNull { item ->
            val j = item as? kotlinx.serialization.json.JsonObject ?: return@mapNotNull null
            val title = j["title"].stringOrNull().orEmpty()
            val url = j["url"].stringOrNull().orEmpty()
            if (title.isBlank() || url.isBlank()) return@mapNotNull null
            WebSearchResult(
                title = title,
                url = url,
                snippet = (j["highlights"] as? kotlinx.serialization.json.JsonArray)
                    ?.firstOrNull()
                    ?.asJsonObjectOrNull()
                    ?.get("text").stringOrNull()
                    .orEmpty(),
                text = if (req.includeText) j["text"].stringOrNull() else null,
                score = j["score"].stringOrNull()?.toDoubleOrNull(),
            )
        }
    }
}
