package com.aura.tools

import com.aura.agent.Tool
import com.aura.agent.ToolResult
import com.aura.agent.ToolRisk
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
import java.net.InetAddress
import java.net.URI
import java.net.UnknownHostException
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

            // --- SSRF guard: parse URL ---
            val uri = try {
                URI(url)
            } catch (e: Exception) {
                return@Tool ToolResult.Error("invalid URL: ${e.message}", "bad_args")
            }

            val scheme = uri.scheme ?: ""
            if (scheme != "http" && scheme != "https") {
                return@Tool ToolResult.Error("only http/https URLs are allowed", "ssrf_guard")
            }

            val host = uri.host ?: return@Tool ToolResult.Error("URL has no host", "ssrf_guard")

            // Reject bare localhost hostnames
            if (host == "localhost" || host == "localhost.localdomain") {
                return@Tool ToolResult.Error("access to localhost is not allowed", "ssrf_guard")
            }

            // Resolve and check for private / loopback / link-local IPs
            try {
                val addr = InetAddress.getByName(host)
                if (isPrivateAddress(addr)) {
                    return@Tool ToolResult.Error("access to private IP is not allowed", "ssrf_guard")
                }
            } catch (_: UnknownHostException) {
                return@Tool ToolResult.Error("could not resolve host", "dns_error")
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
    )

    // ------------------------------------------------------------------
    // Firecrawl API call
    // ------------------------------------------------------------------

    private fun fetchUrl(url: String, apiKey: String): String {
        val requestBody = buildJsonBody(url)
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

    private fun buildJsonBody(url: String): String {
        return """{"url":"${url.replace("\\", "\\\\").replace("\"", "\\\"")}","formats":["markdown"]}"""
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

    // ------------------------------------------------------------------
    // SSRF helpers
    // ------------------------------------------------------------------

    /**
     * Returns true if [addr] is a private, loopback, link-local, or
     * unique-local IPv6 address — i.e. something that should never be
     * reachable from a cloud service.
     */
    private fun isPrivateAddress(addr: InetAddress): Boolean {
        // Covers 127.x.x.x, ::1
        if (addr.isLoopbackAddress) return true
        // Covers 169.254.x.x, fe80::/10
        if (addr.isLinkLocalAddress) return true
        // Covers 10.x.x.x, 172.16-31.x.x, 192.168.x.x, fec0::/10
        if (addr.isSiteLocalAddress) return true

        // IPv6 unique local address range fc00::/7
        val raw = addr.address ?: return false
        if (raw.size == 16 && (raw[0].toInt() and 0xfe) == 0xfc) return true

        return false
    }

    companion object {
        const val MAX_OUTPUT_LENGTH = 8000
    }
}
