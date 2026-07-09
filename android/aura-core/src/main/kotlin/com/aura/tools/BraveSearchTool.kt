package com.aura.tools

import com.aura.agent.Tool
import com.aura.agent.ToolContext
import com.aura.agent.ToolResult
import com.aura.agent.ToolRisk
import com.aura.providers.ProviderKeys
import com.aura.providers.ToolDefinition
import com.aura.providers.ToolParameters
import com.aura.providers.ToolProperty
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import kotlinx.serialization.json.contentOrNull
import okhttp3.OkHttpClient
import okhttp3.Request
import javax.inject.Inject
import javax.inject.Singleton

/**
 * Web search via Brave Search API, with automatic fallback to DuckDuckGo
 * HTML scrape when no API key is configured.
 *
 * Port of aura/tools/brave_search.py.
 * Risk: READ_ONLY (network egress only, no phone permissions).
 */
@Singleton
class BraveSearchTool @Inject constructor(
    private val httpClient: OkHttpClient,
    private val providerKeys: ProviderKeys,
) {
    private val json = Json { ignoreUnknownKeys = true }

    fun definition() = ToolDefinition(
        name = "brave_search",
        description = "Search the web using Brave Search and return results as a markdown list.",
        parameters = ToolParameters(
            properties = mapOf(
                "query" to ToolProperty(type = "string", description = "Search query"),
                "count" to ToolProperty(type = "integer", description = "Number of results (default 5, max 10)"),
            ),
            required = listOf("query"),
        ),
    )

    val tool = Tool(
        name = "brave_search",
        description = definition().description,
        risk = ToolRisk.READ_ONLY,
        parameters = definition().parameters,
        execute = { call, ctx ->
            val query = call.arguments["query"] as? String
                ?: return@Tool ToolResult.Error("missing 'query' argument", "bad_args")
            val count = (call.arguments["count"] as? Int ?: 5).coerceIn(1, 10)
            try {
                val results = search(query, count)
                ToolResult.Ok(formatResults(results))
            } catch (e: Exception) {
                ToolResult.Error("search failed: ${e.message}", "http_error")
            }
        },
    category = "web")
    private data class Result(val title: String, val url: String, val snippet: String)

    /**
     * Try Brave API first; fall back to DuckDuckGo HTML scrape if no key.
     */
    private fun search(query: String, maxResults: Int): List<Result> {
        val braveKey = providerKeys.keyFor("brave")
        return if (!braveKey.isNullOrBlank()) {
            searchBraveApi(query, maxResults, braveKey)
        } else {
            searchDuckDuckGo(query, maxResults)
        }
    }

    // ------------------------------------------------------------------
    // Brave Web Search API
    // ------------------------------------------------------------------

    private fun searchBraveApi(query: String, maxResults: Int, apiKey: String): List<Result> {
        val encoded = java.net.URLEncoder.encode(query, "UTF-8")
        val url = "https://api.search.brave.com/res/v1/web/search?q=$encoded&count=$maxResults"
        val req = Request.Builder()
            .url(url)
            .header("X-Subscription-Token", apiKey)
            .header("User-Agent", "Mozilla/5.0 Aura/1.0")
            .header("Accept", "application/json")
            .build()
        httpClient.newCall(req).execute().use { resp ->
            if (!resp.isSuccessful) throw RuntimeException("Brave API HTTP ${resp.code}")
            val body = resp.body?.string() ?: return emptyList()
            return parseBraveJson(body, maxResults)
        }
    }

    private fun parseBraveJson(body: String, maxResults: Int): List<Result> {
        val root = json.parseToJsonElement(body).jsonObject
        val web = root["web"]?.jsonObject ?: return emptyList()
        val results = web["results"]?.jsonArray ?: return emptyList()
        return results.take(maxResults).mapNotNull { el ->
            val obj = el.jsonObject
            val title = obj["title"]?.jsonPrimitive?.contentOrNull ?: return@mapNotNull null
            val url = obj["url"]?.jsonPrimitive?.contentOrNull ?: return@mapNotNull null
            val snippet = obj["description"]?.jsonPrimitive?.contentOrNull ?: ""
            Result(title = title, url = url, snippet = snippet)
        }
    }

    // ------------------------------------------------------------------
    // DuckDuckGo HTML fallback (same approach as WebSearchTool)
    // ------------------------------------------------------------------

    private fun searchDuckDuckGo(query: String, maxResults: Int): List<Result> {
        val encoded = java.net.URLEncoder.encode(query, "UTF-8")
        val url = "https://html.duckduckgo.com/html/?q=$encoded"
        val req = Request.Builder()
            .url(url)
            .header("User-Agent", "Mozilla/5.0 Aura/1.0")
            .build()
        httpClient.newCall(req).execute().use { resp ->
            if (!resp.isSuccessful) throw RuntimeException("DuckDuckGo HTTP ${resp.code}")
            val html = resp.body?.string() ?: return emptyList()
            return parseDuckDuckGoHtml(html, maxResults)
        }
    }

    private fun parseDuckDuckGoHtml(html: String, maxResults: Int): List<Result> {
        val out = mutableListOf<Result>()
        val linkPattern = Regex("""class="result__a"[^>]*href="([^"]+)"[^>]*>([^<]+)</a>""")
        val snippetPattern = Regex("""class="result__snippet"[^>]*>([^<]+)</a>""")
        val links = linkPattern.findAll(html).take(maxResults).toList()
        val snippets = snippetPattern.findAll(html).take(maxResults).toList()
        for (i in links.indices) {
            val m = links[i]
            val snippet = snippets.getOrNull(i)?.groupValues?.get(1)?.trim() ?: ""
            out += Result(
                title = m.groupValues[2].trim(),
                url = m.groupValues[1].trim(),
                snippet = snippet,
            )
        }
        return out
    }

    // ------------------------------------------------------------------
    // Formatting
    // ------------------------------------------------------------------

    private fun formatResults(results: List<Result>): String =
        if (results.isEmpty()) "No results found."
        else results.joinToString("\n") { "- [${it.title}](${it.url}): ${it.snippet}" }
}
