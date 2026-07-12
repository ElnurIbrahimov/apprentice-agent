package com.aura.tools

import com.aura.agent.Tool
import com.aura.agent.ToolContext
import com.aura.agent.ToolResult
import com.aura.agent.ToolRisk
import com.aura.core.url.SsrfGuard
import com.aura.core.util.buildFirecrawlBody
import com.aura.providers.ChatOptions
import com.aura.providers.ProviderKeys
import com.aura.providers.ProviderMessage
import com.aura.providers.ProviderRegistry
import com.aura.providers.ToolParameters
import com.aura.providers.ToolProperty
import com.aura.tools.DuckDuckGoSearch
import kotlinx.coroutines.TimeoutCancellationException
import kotlinx.coroutines.flow.toList
import kotlinx.coroutines.withTimeout
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.buildJsonArray
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.contentOrNull
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import kotlinx.serialization.json.put
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import javax.inject.Inject
import javax.inject.Singleton

/**
 * Deep research tool with citations.
 *
 * Pipeline:
 * 1. Search the web (Tavily if configured, else Brave/DuckDuckGo).
 * 2. Fetch the top [max_sources] URLs via Firecrawl (or direct HTTP).
 * 3. Concatenate content truncated to fit model context (~6000 chars).
 * 4. Call a provider model to synthesize an answer citing sources as [1], [2], etc.
 * 5. Return a JSON object with "answer" and "citations" fields.
 *
 * Risk: READ_ONLY (network egress only, no phone permissions).
 * Timeout: 60 seconds for the entire pipeline.
 */
@Singleton
class DeepResearchTool @Inject constructor(
    private val httpClient: OkHttpClient,
    private val providerKeys: ProviderKeys,
    private val providerRegistry: ProviderRegistry,
) {
    private val json = Json { ignoreUnknownKeys = true }
    private val mediaType = "application/json".toMediaType()
    private val ddg by lazy { DuckDuckGoSearch(httpClient) }

    data class Citation(val index: Int, val title: String, val url: String)

    fun definition() = ToolParameters(
        properties = mapOf(
            "query" to ToolProperty(
                type = "string",
                description = "The research question or topic to investigate",
            ),
            "max_sources" to ToolProperty(
                type = "integer",
                description = "Maximum number of sources to fetch and cite (default 5, max 10)",
            ),
            "model" to ToolProperty(
                type = "string",
                description = "Model ID for synthesis, e.g. \"ollama:deepseek-v4-pro\" or \"openai:gpt-4o\" (default: first configured provider)",
            ),
        ),
        required = listOf("query"),
    )

    val tool = Tool(
        name = "deep_research",
        description = "Perform deep research: search the web, fetch content from top sources, and synthesize an answer with numbered citations.",
        risk = ToolRisk.READ_ONLY,
        parameters = definition(),
        execute = { call, _ ->
            val query = call.arguments["query"] as? String
                ?: return@Tool ToolResult.Error("missing 'query' argument", "bad_args")
            val maxSources = (call.arguments["max_sources"] as? Int ?: 5).coerceIn(1, 10)
            val modelArg = call.arguments["model"] as? String

            try {
                val result = runResearch(query, maxSources, modelArg)
                ToolResult.Ok(result)
            } catch (e: TimeoutCancellationException) {
                ToolResult.Error("Research timed out after 60s", "timeout")
            } catch (e: Exception) {
                ToolResult.Error("Research failed: ${e.message}", "research_error")
            }
        },
    category = "web")
    // ------------------------------------------------------------------
    // Pipeline
    // ------------------------------------------------------------------

    private suspend fun runResearch(
        query: String,
        maxSources: Int,
        modelArg: String?,
    ): String = withTimeout(RESEARCH_TIMEOUT_MS) {
        // Step 1 — Search
        val searchResults = performSearch(query, maxSources * 2)
        if (searchResults.isEmpty()) {
            return@withTimeout """{"answer":"No search results found for the query.","citations":[]}"""
        }

        // Step 2 — Build citations (numbered)
        val citations = searchResults.take(maxSources).mapIndexed { idx, r ->
            Citation(index = idx + 1, title = r.title, url = r.url)
        }

        // Step 3 — Fetch content for each source
        val contents = mutableMapOf<String, String>()
        for (citation in citations) {
            val content = fetchUrlContent(citation.url)
            if (!content.isNullOrBlank()) {
                contents[citation.url] = content
            }
        }

        // Step 4 — Build context block (truncated to ~6000 chars total)
        val contextBlock = buildContextBlock(citations, contents)

        // Step 5 — Call LLM to synthesize.
        // If the caller specified a model, use it. Otherwise pick the
        // first configured provider's first model — passing a literal
        // "default" here would hit ProviderRegistry.parse("default")
        // which resolves to "first configured provider" with model name
        // "default", which no provider recognizes (404 / model_not_found).
        val modelId = modelArg ?: runCatching {
            val p = providerRegistry.configured().firstOrNull()
                ?: return@runCatching "default"
            val m = p.listModels().firstOrNull()
            if (m != null) "${p.prefix}:$m" else "${p.prefix}:default"
        }.getOrDefault("default")
        val answer = synthesizeAnswer(query, contextBlock, modelId)

        // Step 6 — Build & return JSON output
        buildJsonOutput(answer, citations)
    }

    // ------------------------------------------------------------------
    // Search
    // ------------------------------------------------------------------

    private data class SearchResult(val title: String, val url: String, val snippet: String)

    private fun performSearch(query: String, maxResults: Int): List<SearchResult> {
        val tavilyKey = providerKeys.keyFor("tavily")
        return if (!tavilyKey.isNullOrBlank()) {
            searchTavily(query, maxResults, tavilyKey)
        } else {
            searchBrave(query, maxResults)
        }
    }

    private fun searchTavily(query: String, maxResults: Int, apiKey: String): List<SearchResult> {
        val requestBody = buildJsonObject {
            put("api_key", apiKey)
            put("query", query)
            put("max_results", maxResults.coerceIn(5, 20))
            put("search_depth", "basic")
        }.toString()

        val req = Request.Builder()
            .url("https://api.tavily.com/search")
            .header("Content-Type", "application/json")
            .header("User-Agent", "Mozilla/5.0 Aura/1.0")
            .post(requestBody.toRequestBody(mediaType))
            .build()

        httpClient.newCall(req).execute().use { resp ->
            if (!resp.isSuccessful) throw RuntimeException("Tavily API HTTP ${resp.code}")
            val bodyStr = resp.body?.string() ?: return emptyList()
            val root = json.parseToJsonElement(bodyStr).jsonObject
            val results = root["results"]?.jsonArray ?: return emptyList()
            return results.mapNotNull { el ->
                val obj = el.jsonObject
                val title = obj["title"]?.jsonPrimitive?.contentOrNull ?: return@mapNotNull null
                val resultUrl = obj["url"]?.jsonPrimitive?.contentOrNull ?: return@mapNotNull null
                val snippet = obj["content"]?.jsonPrimitive?.contentOrNull ?: ""
                SearchResult(title = title, url = resultUrl, snippet = snippet)
            }
        }
    }

    private fun searchBrave(query: String, maxResults: Int): List<SearchResult> {
        val braveKey = providerKeys.keyFor("brave")
        val encoded = java.net.URLEncoder.encode(query, "UTF-8")

        return if (!braveKey.isNullOrBlank()) {
            searchBraveApi(maxResults, braveKey, encoded)
        } else {
            searchDuckDuckGo(encoded, maxResults)
        }
    }

    private fun searchBraveApi(
        maxResults: Int,
        apiKey: String,
        encoded: String,
    ): List<SearchResult> {
        val url = "https://api.search.brave.com/res/v1/web/search?q=$encoded&count=$maxResults"
        val req = Request.Builder()
            .url(url)
            .header("X-Subscription-Token", apiKey)
            .header("User-Agent", "Mozilla/5.0 Aura/1.0")
            .header("Accept", "application/json")
            .build()

        httpClient.newCall(req).execute().use { resp ->
            if (!resp.isSuccessful) throw RuntimeException("Brave API HTTP ${resp.code}")
            val bodyStr = resp.body?.string() ?: return emptyList()
            val root = json.parseToJsonElement(bodyStr).jsonObject
            val web = root["web"]?.jsonObject ?: return emptyList()
            val results = web["results"]?.jsonArray ?: return emptyList()
            return results.mapNotNull { el ->
                val obj = el.jsonObject
                val title = obj["title"]?.jsonPrimitive?.contentOrNull ?: return@mapNotNull null
                val resultUrl = obj["url"]?.jsonPrimitive?.contentOrNull ?: return@mapNotNull null
                val snippet = obj["description"]?.jsonPrimitive?.contentOrNull ?: ""
                SearchResult(title = title, url = resultUrl, snippet = snippet)
            }
        }
    }

    private fun searchDuckDuckGo(encoded: String, maxResults: Int): List<SearchResult> {
        val query = java.net.URLDecoder.decode(encoded, "UTF-8")
        return ddg.search(query, maxResults).map { SearchResult(it.title, it.url, it.snippet) }
    }

    // ------------------------------------------------------------------
    // Fetch
    // ------------------------------------------------------------------

    private fun fetchUrlContent(url: String): String? {
        // Search-result URLs are untrusted. Apply the guard before choosing
        // either outbound fetch backend—Firecrawl must not become an SSRF
        // bypass merely because it is configured.
        if (SsrfGuard.validate(url) != null) return null

        val firecrawlKey = providerKeys.keyFor("firecrawl")
        if (!firecrawlKey.isNullOrBlank()) {
            return fetchViaFirecrawl(url, firecrawlKey)
        }
        return fetchDirect(url)
    }

    private fun fetchViaFirecrawl(url: kotlin.String, apiKey: kotlin.String): kotlin.String? {
        try {
            val body = buildFirecrawlBody(url)

            val req = Request.Builder()
                .url("https://api.firecrawl.dev/v1/scrape")
                .header("Authorization", "Bearer $apiKey")
                .header("Content-Type", "application/json")
                .header("User-Agent", "Mozilla/5.0 Aura/1.0")
                .post(body.toRequestBody(mediaType))
                .build()

            httpClient.newCall(req).execute().use { resp ->
                if (!resp.isSuccessful) return null
                val bodyStr = resp.body?.string() ?: return null
                val root = json.parseToJsonElement(bodyStr).jsonObject
                val data = root["data"]?.jsonObject ?: return null
                return data["markdown"]?.jsonPrimitive?.contentOrNull
            }
        } catch (_: Exception) {
            return null
        }
    }

    private fun fetchDirect(url: String): String? {
        // SSRF guard: reject non-http(s) schemes, localhost, and private IPs.
        // This is the same check FirecrawlFetchTool applies; without it, a
        // search result URL pointing at an internal address would be fetched
        // directly, leaking internal network state.
        val ssrfError = SsrfGuard.validate(url)
        if (ssrfError != null) return null

        try {
            val req = Request.Builder()
                .url(url)
                .header("User-Agent", "Mozilla/5.0 Aura/1.0")
                .header("Accept", "text/html,text/plain,*/*")
                .build()

            httpClient.newCall(req).execute().use { resp ->
                if (!resp.isSuccessful) return null
                val body = resp.body?.string() ?: return null
                // Strip HTML tags and collapse whitespace
                return body
                    .replace(Regex("<[^>]+>"), " ")
                    .replace(Regex("\\s+"), " ")
                    .trim()
                    .take(3000)
            }
        } catch (_: Exception) {
            return null
        }
    }

    // ------------------------------------------------------------------
    // Context assembly, synthesis, output formatting
    // ------------------------------------------------------------------

    private fun buildContextBlock(
        citations: List<Citation>,
        contents: Map<String, String>,
    ): String {
        val sb = StringBuilder()
        sb.appendLine("You have the following sources to answer the research question.\n")
        for (c in citations) {
            val content = contents[c.url] ?: continue
            sb.appendLine("[$c.index] ${c.title}")
            sb.appendLine("URL: ${c.url}")
            sb.appendLine("```")
            sb.appendLine(content.take(1500))
            sb.appendLine("```")
            sb.appendLine()
        }
        val raw = sb.toString()
        return if (raw.length > CONTEXT_LIMIT) raw.take(CONTEXT_LIMIT) + "\n\n[...truncated]"
        else raw
    }

    private suspend fun synthesizeAnswer(
        query: String,
        context: String,
        modelId: String,
    ): String {
        val systemPrompt = "You are a research assistant. Synthesize an answer using ONLY the provided sources. " +
            "Cite using [1], [2], etc. Be concise but thorough."

        val userPrompt = "Research question: $query\n\n" +
            "---\n$context\n---\n" +
            "Provide a well-structured answer based only on the sources above. " +
            "Use citation markers like [1], [2] after each claim."

        val messages = listOf(
            ProviderMessage(role = ProviderMessage.Role.system, content = systemPrompt),
            ProviderMessage(role = ProviderMessage.Role.user, content = userPrompt),
        )

        val options = ChatOptions(temperature = 0.5, maxTokens = 2048)
        val flow = providerRegistry.chat(modelId, messages, options)
        val chunks = flow.toList()

        val answer = chunks.filter { it.text != null }.joinToString("") { it.text!! }
        return answer.ifBlank { "No answer could be synthesized from the available sources." }
    }

    private fun buildJsonOutput(answer: String, citations: List<Citation>): String {
        val citationsJson = buildJsonArray {
            for (c in citations) {
                add(buildJsonObject {
                    put("index", c.index)
                    put("title", c.title)
                    put("url", c.url)
                })
            }
        }
        return buildJsonObject {
            put("answer", answer)
            put("citations", citationsJson)
        }.toString()
    }

    companion object {
        const val RESEARCH_TIMEOUT_MS = 60_000L
        const val CONTEXT_LIMIT = 6000
    }
}
