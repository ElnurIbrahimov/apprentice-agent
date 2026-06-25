package com.aura.tools

import com.aura.agent.Tool
import com.aura.agent.ToolContext
import com.aura.agent.ToolResult
import com.aura.agent.ToolRisk
import com.aura.providers.ToolDefinition
import com.aura.providers.ToolParameters
import com.aura.providers.ToolProperty
import okhttp3.OkHttpClient
import okhttp3.Request
import org.json.JSONObject
import javax.inject.Inject
import javax.inject.Singleton

/**
 * Web search via DuckDuckGo HTML (free, no API key).
 * Port of aura/tools/web_search.py + brave_search.py.
 * Risk: READ_ONLY (network egress only, no phone permissions).
 */
@Singleton
class WebSearchTool @Inject constructor(
    private val httpClient: OkHttpClient,
) {
    fun definition() = ToolDefinition(
        name = "web_search",
        description = "Search the web and return top results as title/URL/snippet.",
        parameters = ToolParameters(
            properties = mapOf(
                "query" to ToolProperty(type = "string", description = "Search query"),
                "max_results" to ToolProperty(type = "integer", description = "Number of results (default 5, max 10)"),
            ),
            required = listOf("query"),
        ),
    )

    val tool = Tool(
        name = "web_search",
        description = definition().description,
        risk = ToolRisk.READ_ONLY,
        parameters = definition().parameters,
        execute = { call, ctx ->
            val query = call.arguments["query"] as? String
                ?: return@Tool ToolResult.Error("missing 'query' argument", "bad_args")
            val maxResults = (call.arguments["max_results"] as? Int ?: 5).coerceIn(1, 10)
            try {
                val results = search(query, maxResults)
                ToolResult.Ok(formatResults(results))
            } catch (e: Exception) {
                ToolResult.Error("search failed: ${e.message}", "http_error")
            }
        },
    )

    private data class Result(val title: String, val url: String, val snippet: String)

    private fun search(query: String, maxResults: Int): List<Result> {
        val encoded = java.net.URLEncoder.encode(query, "UTF-8")
        val url = "https://html.duckduckgo.com/html/?q=$encoded"
        val req = Request.Builder()
            .url(url)
            .header("User-Agent", "Mozilla/5.0 Aura/1.0")
            .build()
        httpClient.newCall(req).execute().use { resp ->
            if (!resp.isSuccessful) throw RuntimeException("HTTP ${resp.code}")
            val html = resp.body?.string() ?: return emptyList()
            return parseResults(html, maxResults)
        }
    }

    private fun parseResults(html: String, maxResults: Int): List<Result> {
        // DuckDuckGo HTML format: <a class="result__a" href="URL">Title</a>
        // followed by <a class="result__snippet">Snippet</a>
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

    private fun formatResults(results: List<Result>): String =
        if (results.isEmpty()) "No results found."
        else results.mapIndexed { i, r -> "${i + 1}. ${r.title}\n   ${r.url}\n   ${r.snippet}" }.joinToString("\n\n")
}
