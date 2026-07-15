package com.aura.capabilities.jina

import com.aura.capabilities.WebSearchProvider
import com.aura.capabilities.WebSearchRequest
import com.aura.capabilities.WebSearchResult
import com.aura.capabilities.http.CapabilityHttp
import com.aura.capabilities.http.asJsonObjectOrNull
import com.aura.capabilities.http.stringOrNull
import com.aura.providers.ProviderKeys
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.OkHttpClient
import javax.inject.Inject
import javax.inject.Singleton

/**
 * Jina Reader. GET https://r.jina.ai/{url} (URL-to-text) or GET https://s.jina.ai/?q= (search-as-SERP).
 * Authenticated via Authorization: Bearer $JINA_API_KEY.
 * https://jina.ai/reader/
 */
@Singleton
class JinaReaderProvider @Inject constructor(
    private val client: OkHttpClient,
    private val providerKeys: ProviderKeys,
) : WebSearchProvider {
    override val prefix = "jina"
    override val displayName = "Jina Reader"
    private val apiKey: String get() = providerKeys.keyFor(prefix).orEmpty()
    override fun isConfigured(): Boolean = apiKey.isNotBlank()

    override suspend fun search(req: WebSearchRequest): List<WebSearchResult> = withContext(Dispatchers.IO) {
        val url = "https://s.jina.ai/?q=" + java.net.URLEncoder.encode(req.query, "UTF-8") +
            "&num=" + req.numResults.coerceAtLeast(1)
        val request = okhttp3.Request.Builder()
            .url(url)
            .header("Authorization", "Bearer $apiKey")
            .header("X-Return-Format", "json")
            .build()
        val response = client.newCall(request).execute()
        val raw = response.use { it.body?.string().orEmpty() }
        CapabilityHttp.classify(response, raw)
        val obj = CapabilityHttp.json.parseToJsonElement(raw).asJsonObjectOrNull()
            ?: return@withContext emptyList()
        val results = (obj["data"] as? kotlinx.serialization.json.JsonArray) ?: return@withContext emptyList()
        results.mapNotNull { item ->
            val j = item as? kotlinx.serialization.json.JsonObject ?: return@mapNotNull null
            val title = j["title"].stringOrNull().orEmpty()
            val u = j["url"].stringOrNull().orEmpty()
            if (title.isBlank() || u.isBlank()) return@mapNotNull null
            WebSearchResult(
                title = title,
                url = u,
                snippet = j["description"].stringOrNull().orEmpty(),
                text = if (req.includeText) j["content"].stringOrNull() else null,
            )
        }
    }
}
