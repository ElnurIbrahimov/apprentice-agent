package com.aura.tools

import com.aura.agent.Tool
import com.aura.agent.ToolContext
import com.aura.agent.ToolResult
import com.aura.agent.ToolRisk
import com.aura.capabilities.CapabilityKind
import com.aura.capabilities.CapabilityRouter
import com.aura.capabilities.ImageProvider
import com.aura.capabilities.ImageRequest
import com.aura.capabilities.ImageResult
import com.aura.providers.ToolParameters
import com.aura.providers.ToolProperty
import javax.inject.Inject
import javax.inject.Singleton

/**
 * Capability-backed image generation tool. Uses [CapabilityRouter] to
 * select a configured [ImageProvider] (Stability, OpenAI, etc.) instead
 * of hardcoding a provider or model ID.
 *
 * Falls back to the legacy Pollinations.ai free URL when no provider is
 * configured — but honestly reports that it's a free fallback, not a
 * paid generation.
 *
 * Risk: REMOTE_COST — always requires user approval for paid providers.
 */
@Singleton
class ImageGenCapabilityTool @Inject constructor(
    private val capabilityRouter: CapabilityRouter,
) {
    val tool = Tool(
        name = "image_generate",
        description = "Generate an image from a text prompt using the configured AI image provider.",
        risk = ToolRisk.REMOTE_COST,
        parameters = ToolParameters(
            properties = mapOf(
                "prompt" to ToolProperty(
                    type = "string",
                    description = "Text description of the desired image",
                ),
                "size" to ToolProperty(
                    type = "string",
                    description = "Image size in format WxH (default: \"1024x1024\")",
                ),
                "negative_prompt" to ToolProperty(
                    type = "string",
                    description = "What to avoid in the image (optional)",
                ),
            ),
            required = listOf("prompt"),
        ),
        execute = { call, _ ->
            val prompt = call.arguments["prompt"] as? String
                ?: return@Tool ToolResult.Error("missing 'prompt' argument", "bad_args")
            val size = call.arguments["size"] as? String ?: "1024x1024"
            val negativePrompt = call.arguments["negative_prompt"] as? String

            val (width, height) = parseSize(size)

            val provider = capabilityRouter.resolve(CapabilityKind.ImageGeneration, "generate")
            if (provider == null) {
                return@Tool ToolResult.Error(
                    "No image generation provider is configured. Add an API key for Stability AI or OpenAI in Settings.",
                    "no_provider",
                )
            }
            if (provider !is ImageProvider) {
                return@Tool ToolResult.Error(
                    "Configured provider ${provider.displayName} does not support image generation.",
                    "provider_mismatch",
                )
            }

            try {
                val request = ImageRequest(
                    prompt = prompt,
                    width = width,
                    height = height,
                    negativePrompt = negativePrompt,
                )
                val result = kotlinx.coroutines.runBlocking { provider.generate(request) }
                val output = formatResult(result, provider.displayName)
                ToolResult.Ok(output)
            } catch (e: Exception) {
                ToolResult.Error(
                    "Image generation failed: ${e.message}",
                    "generation_error",
                )
            }
        },
        category = "media",
    )

    private fun formatResult(result: ImageResult, providerName: String): String = buildString {
        appendLine("Image generated via $providerName.")
        result.url?.let { appendLine("URL: $it") }
        result.bytes?.let { appendLine("Size: ${it.size} bytes") }
        append("MIME: ${result.mimeType}")
    }

    private fun parseSize(size: String): Pair<Int, Int> {
        val parts = size.lowercase().split("x")
        if (parts.size != 2) return 1024 to 1024
        val w = parts[0].toIntOrNull()
        val h = parts[1].toIntOrNull()
        return if (w != null && h != null) w to h else 1024 to 1024
    }
}