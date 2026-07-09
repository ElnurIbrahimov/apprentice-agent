package com.aura.tools

import com.aura.agent.Tool
import com.aura.agent.ToolResult
import com.aura.agent.ToolRisk
import com.aura.providers.ChatOptions
import com.aura.providers.ProviderMessage
import com.aura.providers.ProviderRegistry
import com.aura.providers.ToolDefinition
import com.aura.providers.ToolParameters
import com.aura.providers.ToolProperty
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.flow.toList
import javax.inject.Inject
import javax.inject.Singleton

/**
 * Translates text to a target language using the configured LLM.
 * Uses the first configured provider's first model.
 *
 * Risk: READ_ONLY (network egress only, no local mutations).
 */
@Singleton
class TranslateTool @Inject constructor(
    private val providerRegistry: ProviderRegistry,
) {
    fun definition() = ToolDefinition(
        name = "translate",
        description = "Translate text to a specified target language using the AI model.",
        parameters = ToolParameters(
            properties = mapOf(
                "text" to ToolProperty(
                    type = "string",
                    description = "The text to translate",
                ),
                "target_language" to ToolProperty(
                    type = "string",
                    description = "The target language to translate into (e.g. Spanish, French, Japanese)",
                ),
            ),
            required = listOf("text", "target_language"),
        ),
    )

    val tool = Tool(
        name = "translate",
        description = definition().description,
        risk = ToolRisk.READ_ONLY,
        parameters = definition().parameters,
        execute = { call, _ ->
            val text = call.arguments["text"] as? String
                ?: return@Tool ToolResult.Error("missing 'text' argument", "bad_args")
            val targetLanguage = call.arguments["target_language"] as? String
                ?: return@Tool ToolResult.Error("missing 'target_language' argument", "bad_args")

            try {
                val result = performTranslation(text, targetLanguage)
                ToolResult.Ok(result)
            } catch (e: CancellationException) {
                throw e
            } catch (e: Exception) {
                ToolResult.Error("Translation failed: ${e.message}", "translation_error")
            }
        },
    category = "media")
    private suspend fun performTranslation(text: String, targetLanguage: String): String {
        // Resolve the first configured provider's first model
        val providers = providerRegistry.configured()
        if (providers.isEmpty()) {
            throw IllegalStateException("No LLM providers are configured")
        }
        val provider = providers.first()
        val models = provider.listModels()
        val modelName = models.firstOrNull()
            ?: throw IllegalStateException("Provider '${provider.displayName}' has no available models")
        val modelId = "${provider.prefix}:$modelName"

        val systemPrompt = "Translate the following text to $targetLanguage. Return only the translation."
        val messages = listOf(
            ProviderMessage(role = ProviderMessage.Role.system, content = systemPrompt),
            ProviderMessage(role = ProviderMessage.Role.user, content = text),
        )

        val options = ChatOptions(temperature = 0.3, maxTokens = 2048)
        val flow = providerRegistry.chat(modelId, messages, options)
        val chunks = flow.toList()

        val translation = chunks.filter { it.text != null }.joinToString("") { it.text!! }
        return translation.ifBlank { throw IllegalStateException("Translation produced empty output") }
    }
}
