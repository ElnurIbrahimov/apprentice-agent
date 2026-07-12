package com.aura.tools

import okhttp3.OkHttpClient
import okhttp3.Request

/**
 * Free web search via DuckDuckGo HTML. Used as a fallback by [WebSearchTool],
 * [BraveSearchTool], and [DeepResearchTool]. Centralizing the parser means
 * one DDG layout change is fixed in a single place.
 */
class DuckDuckGoSearch(
    private val httpClient: OkHttpClient,
) {
    data class Result(val title: String, val url: String, val snippet: String)

    fun search(query: String, maxResults: Int): List<Result> {
        val encoded = java.net.URLEncoder.encode(query, "UTF-8")
        val url = "https://html.duckduckgo.com/html/?q=$encoded"
        val req = Request.Builder()
            .url(url)
            .header("User-Agent", "Mozilla/5.0 Aura/1.0")
            .build()
        httpClient.newCall(req).execute().use { resp ->
            if (!resp.isSuccessful) throw RuntimeException("DuckDuckGo HTTP ${resp.code}")
            val html = resp.body?.string() ?: return emptyList()
            return parseHtml(html, maxResults)
        }
    }

    fun parseHtml(html: String, maxResults: Int): List<Result> {
        val out = mutableListOf<Result>()
        val linkPattern = Regex("""class="result__a"[^\u003e]*href="([^"]+)"[^\u003e]*\u003e([^\u003c]+)\u003c/a\u003e""")
        val snippetPattern = Regex("""class="result__snippet"[^\u003e]*\u003e([^\u003c]+)\u003c/a\u003e""")
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
}
