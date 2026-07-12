package com.aura.providers

import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.runBlocking
import javax.inject.Inject
import javax.inject.Singleton

/**
 * Holds all providers, routes by `provider:model` prefix.
 * Mirrors aura/providers/registry.py + aura/core/router.py task-aware routing.
 */
@Singleton
class ProviderRegistry @Inject constructor(
    private val providers: Map<String, @JvmSuppressWildcards Provider>,
) {
    private val byPrefix: Map<String, Provider> = providers.mapKeys { (k, _) -> "$k:" }

    /**
     * Resolve a `provider:model` id. The special model id "default" is
     * routed to the first configured provider's first available model.
     */
    suspend fun parse(modelId: String): Pair<Provider, String> {
        val parts = modelId.split(":", limit = 2)
        return if (parts.size == 2) {
            val provider = byPrefix[parts[0] + ":"]
                ?: throw IllegalArgumentException("Unknown provider prefix: ${parts[0]}")
            provider to parts[1]
        } else {
            // Default: first configured provider
            val default = providers.values.firstOrNull { it.isConfigured() }
                ?: throw IllegalStateException("No configured providers")
            val firstModel = runCatching { default.listModels().firstOrNull() }.getOrNull()
            val resolvedModel = firstModel ?: modelId
            default to resolvedModel
        }
    }

    fun chat(modelId: String, messages: List<ProviderMessage>, options: ChatOptions = ChatOptions(), tools: List<ToolDefinition> = emptyList()): Flow<ProviderChunk> {
        val (provider, model) = runBlocking { parse(modelId) }
        return provider.chat(model, messages, options, tools)
    }

    fun configured(): List<Provider> = providers.values.filter { it.isConfigured() }
    fun all(): List<Provider> = providers.values.toList()
    fun get(prefix: String): Provider? = byPrefix["$prefix:"]
}
