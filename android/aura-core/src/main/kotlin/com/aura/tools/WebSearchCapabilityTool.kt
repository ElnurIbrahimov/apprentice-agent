package com.aura.tools

import com.aura.agent.Tool
import com.aura.agent.ToolResult
import com.aura.agent.ToolRisk
import com.aura.capabilities.CapabilityKind
import com.aura.capabilities.CapabilityRouter
import com.aura.capabilities.WebSearchProvider
import com.aura.capabilities.WebSearchRequest
import com.aura.capabilities.WebSearchResult
import com.aura.providers.ToolParameters
import com.aura.providers.ToolProperty
import javax.inject.Inject
import javax.inject.Singleton

/**
 * Unified web search tool that routes through [CapabilityRouter].
 * Prefers a configured [WebSearchProvider] (Exa) when available;
 * falls back to DuckDuckGo (free, no API key) when no provider is
 * configured. Always reports which source was used.
 *
 * Risk: READ_ONLY — search does not mutate state.
 */
@Singleton
class WebSearchCapabilityTool @Inject constructor(
    private val capabilityRouter: CapabilityRouter,
    private val httpClient: okhttp3.OkHttpClient,
) {
    private val ddg by lazy { DuckDuckGoSearch(httpClient) }

    val tool = Tool(
        name = "web_search_capability",
        description = "Search the web using the configured search provider (Exa) or free DuckDuckGo fallback. Returns title/URL/snippet results.",
        risk = ToolRisk.READ_ONLY,
        parameters = ToolParameters(
            properties = mapOf(
                "query" to ToolProperty(type = "string", description = "Search query"),
                "max_results" to ToolProperty(type = "integer", description = "Number of results (default 5, max 10)"),
            ),
            required = listOf("query"),
        ),
        execute = { call, _ ->
            val query = call.arguments["query"] as? String
                ?: return@Tool ToolResult.Error("missing 'query' argument", "bad_args")
            val maxResults = (call.arguments["max_results"] as? Int ?: 5).coerceIn(1, 10)

            val provider = capabilityRouter.resolve(CapabilityKind.WebSearch)

            if (provider != null && provider is WebSearchProvider) {
                try {
                    val results = kotlinx.coroutines.runBlocking {
                        provider.search(WebSearchRequest(query = query, numResults = maxResults))
                    }
                    ToolResult.Ok(formatResults(results, provider.displayName))
                } catch (e: Exception) {
                    // Fall back to DuckDuckGo on provider error
                    val ddgResults = ddg.search(query, maxResults)
                    ToolResult.Ok(formatResultsDdg(ddgResults, "DuckDuckGo (fallback: ${e.message})"))
                }
            } else {
                // No provider configured — use free DuckDuckGo
                val ddgResults = ddg.search(query, maxResults)
                ToolResult.Ok(formatResultsDdg(ddgResults, "DuckDuckGo (free)"))
            }
        },
        category = "web",
    )

    private data class DdgResult(val title: String, val url: String, val snippet: String)

    private fun formatResults(results: List<WebSearchResult>, source: String): String {
        if (results.isEmpty()) return "No results from $source."
        return buildString {
            appendLine("Results from $source:")
            appendLine()
            results.forEachIndexed { i, r ->
                appendLine("${i + 1}. ${r.title}")
                appendLine("   ${r.url}")
                if (r.snippet.isNotBlank()) appendLine("   ${r.snippet}")
                appendLine()
            }
        }
    }

    private fun formatResultsDdg(results: List<DuckDuckGoSearch.Result>, source: String): String {
        if (results.isEmpty()) return "No results from $source."
        return buildString {
            appendLine("Results from $source:")
            appendLine()
            results.forEachIndexed { i, r ->
                appendLine("${i + 1}. ${r.title}")
                appendLine("   ${r.url}")
                if (r.snippet.isNotBlank()) appendLine("   ${r.snippet}")
                appendLine()
            }
        }
    }
}