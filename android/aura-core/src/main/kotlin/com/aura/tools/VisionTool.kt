package com.aura.tools

import com.aura.agent.Tool
import com.aura.agent.ToolResult
import com.aura.agent.ToolRisk
import com.aura.providers.ProviderKeys
import com.aura.providers.ToolParameters
import com.aura.providers.ToolProperty
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
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
 * Vision tool: sends a base64-encoded image to a vision-capable cloud LLM
 * (Gemini, GPT-4o, or Ollama Cloud gemma3) and returns the generated text.
 *
 * Provider selection order:
 * 1. Gemini (gemini-1.5-flash) — if a Gemini API key is configured
 * 2. OpenAI (gpt-4o) — if an OpenAI API key is configured
 * 3. Ollama Cloud (gemma3:12b:cloud) — if an Ollama Cloud API key is configured
 *
 * Risk: READ_ONLY (network egress only, no phone permissions).
 */
@Singleton
class VisionTool @Inject constructor(
    private val httpClient: OkHttpClient,
    private val providerKeys: ProviderKeys,
) {
    private val json = Json { ignoreUnknownKeys = true }
    private val mediaTypeJson = "application/json".toMediaType()

    companion object {
        /** Maximum allowed decoded image size: 2 MB. */
        private const val MAX_IMAGE_BYTES = 2 * 1024 * 1024
    }

    fun definition() = ToolParameters(
        properties = mapOf(
            "image_base64" to ToolProperty(
                type = "string",
                description = "Base64-encoded JPEG image data",
            ),
            "prompt" to ToolProperty(
                type = "string",
                description = "What to ask about the image (default: 'Describe this image in detail')",
            ),
        ),
        required = listOf("image_base64"),
    )

    val tool = Tool(
        name = "vision",
        description = "Analyze an image using a vision-capable AI model. Provide the image as base64-encoded JPEG data and an optional prompt. The image must be 2 MB or smaller.",
        risk = ToolRisk.READ_ONLY,
        parameters = definition(),
        execute = { call, _ ->
            val imageBase64 = call.arguments["image_base64"] as? String
                ?: return@Tool ToolResult.Error("missing 'image_base64' argument", "bad_args")
            val prompt = call.arguments["prompt"] as? String ?: "Describe this image in detail"

            // Image size check: estimate decoded bytes from base64 length
            // base64 encodes 3 bytes into 4 chars, so length * 3/4 ≈ decoded size
            val estimatedBytes = (imageBase64.length * 3L) / 4
            if (estimatedBytes > MAX_IMAGE_BYTES) {
                return@Tool ToolResult.Error(
                    "Image too large (${estimatedBytes / 1024} KB). Maximum is 2 MB. Please take a smaller photo.",
                    "image_too_large",
                )
            }

            try {
                val result = analyzeImage(imageBase64, prompt)
                ToolResult.Ok(result)
            } catch (e: Exception) {
                ToolResult.Error("vision analysis failed: ${e.message}", "http_error")
            }
        },
    )

    /**
     * Routes to the first configured vision-capable provider.
     *
     * Order: Gemini → OpenAI → Ollama Cloud.
     *
     * @throws RuntimeException if no vision provider is configured.
     */
    private fun analyzeImage(imageBase64: String, prompt: String): String {
        val geminiKey = providerKeys.keyFor("gemini")
        if (!geminiKey.isNullOrBlank()) {
            return analyzeWithGemini(imageBase64, prompt, geminiKey)
        }

        val openaiKey = providerKeys.keyFor("openai")
        if (!openaiKey.isNullOrBlank()) {
            return analyzeWithOpenAi(imageBase64, prompt, openaiKey)
        }

        val ollamaKey = providerKeys.keyFor("ollama")
        if (!ollamaKey.isNullOrBlank()) {
            return analyzeWithOllama(imageBase64, prompt, ollamaKey)
        }

        throw RuntimeException(
            "No vision-capable provider configured. Please configure Gemini, OpenAI, or Ollama Cloud in Settings."
        )
    }

    // ------------------------------------------------------------------
    // Gemini API (gemini-1.5-flash)
    // ------------------------------------------------------------------
    // POST https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key=...
    // Body: { contents: [{ parts: [{ text: prompt }, { inlineData: { mimeType: "image/jpeg", data: image_base64 } }] }] }
    // ------------------------------------------------------------------

    private fun analyzeWithGemini(imageBase64: String, prompt: String, apiKey: kotlin.String): String {
        val body = buildJsonObject {
            put("contents", JsonArray(listOf(buildJsonObject {
                put("parts", JsonArray(listOf(
                    buildJsonObject { put("text", prompt) },
                    buildJsonObject {
                        put("inlineData", buildJsonObject {
                            put("mimeType", "image/jpeg")
                            put("data", imageBase64)
                        })
                    },
                )))
            })))
        }

        val requestBody = body.toString().toRequestBody(mediaTypeJson)
        // API key in header, not URL query — the URL is logged in HTTP
        // traces, proxy captures, and crash reports. The X-Goog-Api-Key
        // header is the documented mechanism and keeps the key out of logs.
        val req = Request.Builder()
            .url("https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent")
            .header("Content-Type", "application/json")
            .header("X-Goog-Api-Key", apiKey)
            .post(requestBody)
            .build()

        val response = httpClient.newCall(req).execute()
        if (!response.isSuccessful) {
            val errorBody = response.body?.string() ?: ""
            throw RuntimeException("Gemini API HTTP ${response.code}: $errorBody")
        }
        val respBody = response.body?.string()
            ?: throw RuntimeException("Empty response from Gemini")

        val root = json.parseToJsonElement(respBody).jsonObject
        val candidates = root["candidates"]?.jsonArray
        val candidate = candidates?.firstOrNull()?.jsonObject
        val content = candidate?.get("content")?.jsonObject
        val parts = content?.get("parts")?.jsonArray
        val text = parts?.firstOrNull()?.jsonObject?.get("text")?.jsonPrimitive?.content
            ?: throw RuntimeException("Gemini response missing text")
        return text
    }

    // ------------------------------------------------------------------
    // OpenAI / GPT-4o API
    // ------------------------------------------------------------------
    // POST https://api.openai.com/v1/chat/completions
    // Body: { model: "gpt-4o", messages: [{ role:"user", content:[{type:"text",text:prompt},{type:"image_url",image_url:{url:"data:image/jpeg;base64,..."}}] }] }
    // ------------------------------------------------------------------

    private fun analyzeWithOpenAi(imageBase64: String, prompt: String, apiKey: String): String {
        val body = buildJsonObject {
            put("model", "gpt-4o")
            put("messages", JsonArray(listOf(buildJsonObject {
                put("role", "user")
                put("content", JsonArray(listOf(
                    buildJsonObject { put("type", "text"); put("text", prompt) },
                    buildJsonObject {
                        put("type", "image_url")
                        put("image_url", buildJsonObject {
                            put("url", "data:image/jpeg;base64,$imageBase64")
                        })
                    },
                )))
            })))
        }

        val requestBody = body.toString().toRequestBody(mediaTypeJson)
        val req = Request.Builder()
            .url("https://api.openai.com/v1/chat/completions")
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
        val choices = root["choices"]?.jsonArray
        val choice = choices?.firstOrNull()?.jsonObject
        val message = choice?.get("message")?.jsonObject
        val text = message?.get("content")?.jsonPrimitive?.content
            ?: throw RuntimeException("OpenAI response missing content")
        return text
    }

    // ------------------------------------------------------------------
    // Ollama Cloud (gemma3:12b:cloud) — OpenAI-compatible API
    // ------------------------------------------------------------------
    // POST https://ollama.com/v1/chat/completions
    // Body: { model: "gemma3:12b:cloud", messages: [{ role:"user", content:[...] }] }
    // ------------------------------------------------------------------

    private fun analyzeWithOllama(imageBase64: String, prompt: String, apiKey: String): String {
        val body = buildJsonObject {
            put("model", "gemma3:12b:cloud")
            put("messages", JsonArray(listOf(buildJsonObject {
                put("role", "user")
                put("content", JsonArray(listOf(
                    buildJsonObject { put("type", "text"); put("text", prompt) },
                    buildJsonObject {
                        put("type", "image_url")
                        put("image_url", buildJsonObject {
                            put("url", "data:image/jpeg;base64,$imageBase64")
                        })
                    },
                )))
            })))
        }

        val requestBody = body.toString().toRequestBody(mediaTypeJson)
        val req = Request.Builder()
            .url("https://ollama.com/v1/chat/completions")
            .header("Authorization", "Bearer $apiKey")
            .header("Content-Type", "application/json")
            .post(requestBody)
            .build()

        val response = httpClient.newCall(req).execute()
        if (!response.isSuccessful) {
            val errorBody = response.body?.string() ?: ""
            throw RuntimeException("Ollama Cloud API HTTP ${response.code}: $errorBody")
        }
        val respBody = response.body?.string()
            ?: throw RuntimeException("Empty response from Ollama Cloud")

        val root = json.parseToJsonElement(respBody).jsonObject
        val choices = root["choices"]?.jsonArray
        val choice = choices?.firstOrNull()?.jsonObject
        val message = choice?.get("message")?.jsonObject
        val text = message?.get("content")?.jsonPrimitive?.content
            ?: throw RuntimeException("Ollama Cloud response missing content")
        return text
    }
}
