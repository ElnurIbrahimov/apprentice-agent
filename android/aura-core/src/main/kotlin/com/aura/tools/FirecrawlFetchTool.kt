package com.aura.tools

import com.aura.agent.Tool
import com.aura.agent.ToolResult
import com.aura.agent.ToolRisk
import com.aura.core.url.SsrfGuard
import com.aura.core.util.buildFirecrawlBody
import com.aura.providers.ProviderKeys
import com.aura.providers.ToolParameters
import com.aura.providers.ToolProperty
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.contentOrNull
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import javax.inject.Inject
import javax.inject.Singleton

/**
 * Fetches a URL and returns its content as markdown via the Firecrawl API.
 *
 * Port of aura/tools/fetch_url.py.
 * Risk: READ_ONLY (network egress only, no phone permissions).
 *
 * SSRF guard: only http/https schemes, no private IPs, no localhost.
 */
@Singleton
class FirecrawlFetchTool @Inject constructor(
    private val httpClient: OkHttpClient,
    private val providerKeys: ProviderKeys,
) {
    private val json = Json { ignoreUnknownKeys = true }
    private val mediaType = "application/json".toMediaType()

    fun definition() = ToolParameters(
        properties = mapOf(
            "url" to ToolProperty(type = "string", description = "The URL to fetch"),
        ),
        required = listOf("url"),
    )

    val tool = Tool(
        name = "fetch_url",
        description = "Fetch a URL and return its content as markdown (truncated to ~8000 chars).",
        risk = ToolRisk.READ_ONLY,
        parameters = definition(),
        execute = { call, _ ->
            val url = call.arguments["url"] as? String
                ?: return@Tool ToolResult.Error("missing 'url' argument", "bad_args")

            // --- SSRF guard ---
            val ssrfError = SsrfGuard.validate(url)
            if (ssrfError != null) {
                return@Tool ToolResult.Error(ssrfError, "ssrf_guard")
            }

            // --- API key check ---
            val apiKey = providerKeys.keyFor("firecrawl")
            if (apiKey.isNullOrBlank()) {
                return@Tool ToolResult.Error(
                    "Firecrawl API key is not configured. Set it in Settings.",
                    "missing_key",
                )
            }

            // --- Fetch via Firecrawl ---
            try {
                val markdown = fetchUrl(url, apiKey)
                ToolResult.Ok(markdown)
            } catch (e: Exception) {
                ToolResult.Error("fetch failed: ${e.message}", "http_error")
            }
        },
    category = "web")
    // ------------------------------------------------------------------
    // Firecrawl API call
    // ------------------------------------------------------------------

    private fun fetchUrl(url: String, apiKey: String): String {
        val requestBody = buildFirecrawlBody(url)
        val req = Request.Builder()
            .url("https://api.firecrawl.dev/v1/scrape")
            .header("Authorization", "Bearer $apiKey")
            .header("Content-Type", "application/json")
            .header("User-Agent", "Mozilla/5.0 Aura/1.0")
            .post(requestBody.toRequestBody(mediaType))
            .build()

        val response = httpClient.newCall(req).execute()
        if (!response.isSuccessful) {
            val errorBody = response.body?.string() ?: ""
            throw RuntimeException("Firecrawl API HTTP ${response.code}: $errorBody")
        }
        val body = response.body?.string()
            ?: throw RuntimeException("Empty response body from Firecrawl")
        return parseResponse(body)
    }

    private fun parseResponse(body: String): String {
        val root = json.parseToJsonElement(body).jsonObject
        val data = root["data"]?.jsonObject
            ?: throw RuntimeException("Firecrawl response missing 'data' field")
        val markdown = data["markdown"]?.jsonPrimitive?.contentOrNull
            ?: throw RuntimeException("Firecrawl response missing 'data.markdown' field")

        return if (markdown.length > MAX_OUTPUT_LENGTH) {
            markdown.take(MAX_OUTPUT_LENGTH) + "\n\n[... truncated to $MAX_OUTPUT_LENGTH chars]"
        } else {
            markdown
        }
    }

    companion object {
        const val MAX_OUTPUT_LENGTH = 8000
    }
}
