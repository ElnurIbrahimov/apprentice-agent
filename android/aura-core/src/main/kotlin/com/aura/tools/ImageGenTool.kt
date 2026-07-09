package com.aura.tools

import com.aura.agent.Tool
import com.aura.agent.ToolContext
import com.aura.agent.ToolResult
import com.aura.agent.ToolRisk
import com.aura.providers.ProviderKeys
import com.aura.providers.ToolParameters
import com.aura.providers.ToolProperty
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.buildJsonObject
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
 * Image generation tool: generates images using OpenAI DALL-E 3 or
 * Pollinations.ai (free, no API key required).
 *
 * Provider selection order:
 * 1. OpenAI DALL-E 3 — if an OpenAI API key is configured
 * 2. Pollinations.ai — free fallback, no API key needed
 *
 * If OpenAI returns an error, automatically falls back to Pollinations.ai.
 *
 * Risk: READ_ONLY (network egress only, no phone permissions).
 */
@Singleton
class ImageGenTool @Inject constructor(
    private val httpClient: OkHttpClient,
    private val providerKeys: ProviderKeys,
) {
    private val json = Json { ignoreUnknownKeys = true }
    private val mediaTypeJson = "application/json".toMediaType()

    fun definition() = ToolParameters(
        properties = mapOf(
            "prompt" to ToolProperty(
                type = "string",
                description = "Text description of the desired image",
            ),
            "provider_hint" to ToolProperty(
                type = "string",
                description = "Preferred provider (default: \"openai\")",
            ),
            "size" to ToolProperty(
                type = "string",
                description = "Image size in format WxH (default: \"1024x1024\")",
            ),
        ),
        required = listOf("prompt"),
    )

    val tool = Tool(
        name = "image_gen",
        description = "Generate an image from a text prompt using AI. Supports OpenAI DALL-E 3 and Pollinations.ai.",
        risk = ToolRisk.READ_ONLY,
        parameters = definition(),
        execute = { call, _ ->
            val prompt = call.arguments["prompt"] as? String
                ?: return@Tool ToolResult.Error("missing 'prompt' argument", "bad_args")
            val size = call.arguments["size"] as? String ?: "1024x1024"

            try {
                val result = generateImage(prompt, size)
                ToolResult.Ok(result)
            } catch (e: Exception) {
                ToolResult.Error("image generation failed: ${e.message}", "http_error")
            }
        },
    category = "media")
    /**
     * Routes to the first available image generation provider.
     *
     * Order: OpenAI DALL-E 3 → Pollinations.ai (free fallback).
     * If OpenAI fails, automatically falls back to Pollinations.
     */
    private fun generateImage(prompt: String, size: String): String {
        val openaiKey = providerKeys.keyFor("openai")
        if (!openaiKey.isNullOrBlank()) {
            try {
                return generateWithOpenAi(prompt, size, openaiKey)
            } catch (_: Exception) {
                // Fall through to Pollinations.ai
            }
        }
        return generateWithPollinations(prompt, size)
    }

    // ------------------------------------------------------------------
    // OpenAI DALL-E 3 API
    // ------------------------------------------------------------------
    // POST https://api.openai.com/v1/images/generations
    // Body: { model: "dall-e-3", prompt, n: 1, size }
    // Response: data[0].url
    // ------------------------------------------------------------------

    private fun generateWithOpenAi(prompt: String, size: String, apiKey: String): String {
        val body = buildJsonObject {
            put("model", "dall-e-3")
            put("prompt", prompt)
            put("n", 1)
            put("size", size)
        }

        val requestBody = body.toString().toRequestBody(mediaTypeJson)
        val req = Request.Builder()
            .url("https://api.openai.com/v1/images/generations")
            .header("Authorization", "Bearer $apiKey")
            .header("Content-Type", "application/json")
            .post(requestBody)
            .build()

        val response = httpClient.newCall(req).execute()
        if (!response.isSuccessful) {
            val errorBody = response.body?.string() ?: ""
            throw RuntimeException("OpenAI API HTTP ${response.code}: $errorBody")
        }
        val respBody = response.body?.string()
            ?: throw RuntimeException("Empty response from OpenAI")

        val root = json.parseToJsonElement(respBody).jsonObject
        val data = root["data"]?.jsonArray
            ?: throw RuntimeException("OpenAI response missing data array")
        val first = data.firstOrNull()?.jsonObject
            ?: throw RuntimeException("OpenAI response has empty data array")
        val url = first["url"]?.jsonPrimitive?.content
            ?: throw RuntimeException("OpenAI response missing url in data[0]")
        return url
    }

    // ------------------------------------------------------------------
    // Pollinations.ai (free, no API key)
    // ------------------------------------------------------------------
    // GET https://image.pollinations.ai/prompt/{urlencoded_prompt}?width=...&height=...&nologo=true
    // Returns the image URL directly — no HTTP call needed.
    // ------------------------------------------------------------------

    private data class Size(val width: Int, val height: Int)

    private fun parseSize(size: String): Size {
        val parts = size.lowercase().split("x")
        if (parts.size != 2) return Size(1024, 1024)
        val w = parts[0].toIntOrNull()
        val h = parts[1].toIntOrNull()
        return if (w != null && h != null) Size(w, h) else Size(1024, 1024)
    }

    private fun generateWithPollinations(prompt: String, size: String): String {
        val encodedPrompt = java.net.URLEncoder.encode(prompt, "UTF-8")
        val parsedSize = parseSize(size)
        return "https://image.pollinations.ai/prompt/$encodedPrompt?width=${parsedSize.width}&height=${parsedSize.height}&nologo=true"
    }
}
