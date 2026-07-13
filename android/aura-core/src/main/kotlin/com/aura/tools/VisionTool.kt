package com.aura.tools

import com.aura.agent.Tool
import com.aura.agent.ToolResult
import com.aura.agent.ToolRisk
import com.aura.data.UserPreferences
import com.aura.providers.ProviderKeys
import com.aura.providers.ToolParameters
import com.aura.providers.ToolProperty
import kotlinx.coroutines.flow.first
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

/** Image analysis routed through the catalog model selected for the vision role. */
@Singleton
class VisionTool @Inject constructor(
    private val httpClient: OkHttpClient,
    private val providerKeys: ProviderKeys,
    private val userPreferences: UserPreferences? = null,
) {
    private val json = Json { ignoreUnknownKeys = true }
    private val mediaTypeJson = "application/json".toMediaType()

    fun definition() = ToolParameters(
        properties = mapOf(
            "image_base64" to ToolProperty(
                type = "string",
                description = "Base64-encoded JPEG image data",
            ),
            "prompt" to ToolProperty(
                type = "string",
                description = "What to ask about the image",
            ),
        ),
        required = listOf("image_base64"),
    )

    val tool = Tool(
        name = "vision",
        description = "Analyze an image using the vision model selected in Settings.",
        risk = ToolRisk.READ_ONLY,
        parameters = definition(),
        execute = { call, _ ->
            val imageBase64 = call.arguments["image_base64"] as? kotlin.String
                ?: return@Tool ToolResult.Error("missing 'image_base64' argument", "bad_args")
            val prompt = call.arguments["prompt"] as? kotlin.String
                ?: "Describe this image in detail"
            val estimatedBytes = (imageBase64.length * 3L) / 4L
            if (estimatedBytes > MAX_IMAGE_BYTES) {
                return@Tool ToolResult.Error(
                    "Image too large (${estimatedBytes / 1024L} KB). Maximum is 2 MB.",
                    "image_too_large",
                )
            }
            try {
                ToolResult.Ok(analyzeImage(imageBase64, prompt))
            } catch (e: Exception) {
                ToolResult.Error("vision analysis failed: ${e.message}", "http_error")
            }
        },
        category = "vision",
    )

    private suspend fun analyzeImage(
        imageBase64: kotlin.String,
        prompt: kotlin.String,
    ): kotlin.String {
        val selected = userPreferences?.visionModel?.first()
            ?: throw RuntimeException("Choose a vision model in Settings first.")
        val parts = selected.split(":", limit = 2)
        require(parts.size == 2 && parts.all(kotlin.String::isNotBlank)) {
            "Vision model must be fully qualified as provider:model."
        }
        val providerPrefix = parts[0]
        val model = parts[1]
        val key = providerKeys.keyFor(providerPrefix)
            ?: throw RuntimeException("The selected vision provider is not configured.")

        return when (providerPrefix) {
            "gemini" -> analyzeWithGemini(imageBase64, prompt, key, model)
            "anthropic" -> analyzeWithAnthropic(imageBase64, prompt, key, model)
            "openai", "deepseek", "groq", "openrouter", "ollama" ->
                analyzeWithOpenAiCompatible(imageBase64, prompt, key, providerPrefix, model)
            else -> throw RuntimeException(
                "The selected provider does not support Aura's vision transport.",
            )
        }
    }

    private fun analyzeWithGemini(
        imageBase64: kotlin.String,
        prompt: kotlin.String,
        key: kotlin.String,
        model: kotlin.String,
    ): kotlin.String {
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
        val request = Request.Builder()
            .url("https://generativelanguage.googleapis.com/v1beta/models/$model:generateContent")
            .header("Content-Type", "application/json")
            .header("X-Goog-Api-Key", key)
            .post(body.toString().toRequestBody(mediaTypeJson))
            .build()
        val responseBody = execute(request, "Gemini")
        return json.parseToJsonElement(responseBody).jsonObject["candidates"]
            ?.jsonArray
            ?.firstOrNull()
            ?.jsonObject
            ?.get("content")
            ?.jsonObject
            ?.get("parts")
            ?.jsonArray
            ?.firstOrNull()
            ?.jsonObject
            ?.get("text")
            ?.jsonPrimitive
            ?.content
            ?: throw RuntimeException("Gemini response missing text")
    }

    private fun analyzeWithOpenAiCompatible(
        imageBase64: kotlin.String,
        prompt: kotlin.String,
        key: kotlin.String,
        providerPrefix: kotlin.String,
        model: kotlin.String,
    ): kotlin.String {
        val baseUrl = when (providerPrefix) {
            "openai" -> "https://api.openai.com/v1"
            "deepseek" -> "https://api.deepseek.com/v1"
            "groq" -> "https://api.groq.com/openai/v1"
            "openrouter" -> "https://openrouter.ai/api/v1"
            "ollama" -> "https://ollama.com/v1"
            else -> error("Unsupported OpenAI-compatible provider")
        }
        val body = buildJsonObject {
            put("model", model)
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
        val builder = Request.Builder()
            .url("$baseUrl/chat/completions")
            .header("Authorization", "Bearer $key")
            .header("Content-Type", "application/json")
            .post(body.toString().toRequestBody(mediaTypeJson))
        if (providerPrefix == "openrouter") {
            builder.header("HTTP-Referer", "https://aura-android")
                .header("X-Title", "Aura Android")
        }
        val responseBody = execute(builder.build(), "Vision provider")
        return json.parseToJsonElement(responseBody).jsonObject["choices"]
            ?.jsonArray
            ?.firstOrNull()
            ?.jsonObject
            ?.get("message")
            ?.jsonObject
            ?.get("content")
            ?.jsonPrimitive
            ?.content
            ?: throw RuntimeException("Vision provider response missing content")
    }

    private fun analyzeWithAnthropic(
        imageBase64: kotlin.String,
        prompt: kotlin.String,
        key: kotlin.String,
        model: kotlin.String,
    ): kotlin.String {
        val body = buildJsonObject {
            put("model", model)
            put("max_tokens", 1024)
            put("messages", JsonArray(listOf(buildJsonObject {
                put("role", "user")
                put("content", JsonArray(listOf(
                    buildJsonObject {
                        put("type", "image")
                        put("source", buildJsonObject {
                            put("type", "base64")
                            put("media_type", "image/jpeg")
                            put("data", imageBase64)
                        })
                    },
                    buildJsonObject { put("type", "text"); put("text", prompt) },
                )))
            })))
        }
        val request = Request.Builder()
            .url("https://api.anthropic.com/v1/messages")
            .header("x-api-key", key)
            .header("anthropic-version", "2023-06-01")
            .header("Content-Type", "application/json")
            .post(body.toString().toRequestBody(mediaTypeJson))
            .build()
        val responseBody = execute(request, "Anthropic")
        return json.parseToJsonElement(responseBody).jsonObject["content"]
            ?.jsonArray
            ?.firstOrNull { element ->
                element.jsonObject["type"]?.jsonPrimitive?.content == "text"
            }
            ?.jsonObject
            ?.get("text")
            ?.jsonPrimitive
            ?.content
            ?: throw RuntimeException("Anthropic response missing content")
    }

    private fun execute(request: Request, providerName: kotlin.String): kotlin.String =
        httpClient.newCall(request).execute().use { response ->
            if (!response.isSuccessful) {
                throw RuntimeException("$providerName HTTP ${response.code}")
            }
            response.body?.string()
                ?: throw RuntimeException("Empty response from $providerName")
        }

    private companion object {
        const val MAX_IMAGE_BYTES = 2 * 1024 * 1024
    }
}
