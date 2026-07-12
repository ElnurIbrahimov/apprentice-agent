package com.aura.tools

import com.aura.agent.Tool
import com.aura.agent.ToolContext
import com.aura.agent.ToolResult
import com.aura.agent.ToolRisk
import com.aura.providers.ProviderKeys
import com.aura.providers.ToolParameters
import com.aura.providers.ToolProperty
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import kotlinx.serialization.json.contentOrNull
import kotlinx.serialization.json.put
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import javax.inject.Inject
import javax.inject.Singleton

/**
 * Web search via Tavily Search API.
 *
 * Port of aura/tools/tavily_search.py.
 * Risk: READ_ONLY (network egress only, no phone permissions).
 */
@Singleton
class TavilySearchTool @Inject constructor(
    private val httpClient: OkHttpClient,
    private val providerKeys: ProviderKeys,
) {
    private val json = Json { ignoreUnknownKeys = true }
    private val mediaType = "application/json".toMediaType()

    fun definition() = com.aura.providers.ToolDefinition(
        name = "tavily_search",
        description = "Search the web using Tavily and return an answer with sources.",
        parameters = ToolParameters(
            properties = mapOf(
                "query" to ToolProperty(type = "string", description = "Search query"),
                "max_results" to ToolProperty(type = "integer", description = "Number of results (default 5, max 20)"),
                "search_depth" to ToolProperty(
                    type = "string",
                    description = "Search depth: 'basic' or 'advanced' (default: 'basic')",
                    enum = listOf("basic", "advanced"),
                ),
                "include_answer" to ToolProperty(
                    type = "boolean",
                    description = "Include an AI-generated answer summary (default: true)",
                ),
            ),
            required = listOf("query"),
        ),
    )

    val tool = Tool(
        name = "tavily_search",
        description = definition().description,
        risk = ToolRisk.READ_ONLY,
        parameters = definition().parameters,
        execute = { call, _ ->
            val query = call.arguments["query"] as? String
                ?: return@Tool ToolResult.Error("missing 'query' argument", "bad_args")
            val maxResults = (call.arguments["max_results"] as? Int ?: 5).coerceIn(1, 20)
            val searchDepth = call.arguments["search_depth"] as? String ?: "basic"
            val includeAnswer = call.arguments["include_answer"] as? Boolean ?: true

            // Validate search_depth
            if (searchDepth !in listOf("basic", "advanced")) {
                return@Tool ToolResult.Error(
                    "invalid search_depth '$searchDepth'; must be 'basic' or 'advanced'",
                    "bad_args",
                )
            }

            val apiKey = providerKeys.keyFor("tavily")
            if (apiKey.isNullOrBlank()) {
                return@Tool ToolResult.Error(
                    "Tavily API key is not configured. Set it in Settings.",
                    "missing_key",
                )
            }

            try {
                val result = search(query, maxResults, searchDepth, includeAnswer, apiKey)
                ToolResult.Ok(result)
            } catch (e: Exception) {
                ToolResult.Error("search failed: ${e.message}", "http_error")
            }
        },
    category = "web")
    private fun search(
        query: String,
        maxResults: Int,
        searchDepth: String,
        includeAnswer: Boolean,
        apiKey: String,
    ): String {
        val requestBody = buildJsonBody(query, maxResults, searchDepth, includeAnswer)
        val req = Request.Builder()
            .url("https://api.tavily.com/search")
            .header("Authorization", "Bearer $apiKey")
            .header("User-Agent", "Mozilla/5.0 Aura/1.0")
            .header("Accept", "application/json")
            .header("Content-Type", "application/json")
            .post(requestBody.toRequestBody(mediaType))
            .build()

        val response = httpClient.newCall(req).execute()
        if (!response.isSuccessful) {
            val errorBody = response.body?.string() ?: ""
            throw RuntimeException("Tavily API HTTP ${response.code}: $errorBody")
        }
        val body = response.body?.string() ?: throw RuntimeException("Empty response body")
        return parseResponse(body, includeAnswer)
    }

    private fun buildJsonBody(
        query: String,
        maxResults: Int,
        searchDepth: String,
        includeAnswer: Boolean,
    ): String {
        val obj = kotlinx.serialization.json.buildJsonObject {
            put("query", query)
            put("max_results", maxResults)
            put("search_depth", searchDepth)
            put("include_answer", includeAnswer)
        }
        return obj.toString()
    }

    private fun parseResponse(body: String, includeAnswer: Boolean): String {
        val root = json.parseToJsonElement(body).jsonObject
        val sb = StringBuilder()

        // Include answer at the top if requested and present
        if (includeAnswer) {
            root["answer"]?.jsonPrimitive?.contentOrNull?.let { answer ->
                if (answer.isNotBlank()) {
                    sb.appendLine(answer)
                    sb.appendLine()
                }
            }
        }

        // Build sources list
        val results = root["results"]?.jsonArray ?: return sb.toString().trimEnd()
        val formattedResults = results.mapNotNull { el ->
            val obj = el.jsonObject
            val title = obj["title"]?.jsonPrimitive?.contentOrNull ?: return@mapNotNull null
            val url = obj["url"]?.jsonPrimitive?.contentOrNull ?: return@mapNotNull null
            val content = obj["content"]?.jsonPrimitive?.contentOrNull ?: ""
            "- [$title]($url): $content"
        }

        if (formattedResults.isEmpty()) {
            if (sb.isEmpty()) return "No results found."
            return sb.toString().trimEnd()
        }

        sb.appendLine(formattedResults.joinToString("\n"))
        return sb.toString().trimEnd()
    }
}
