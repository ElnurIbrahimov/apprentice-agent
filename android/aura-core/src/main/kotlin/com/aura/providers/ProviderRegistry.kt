package com.aura.providers

import kotlinx.coroutines.flow.Flow
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

    fun parse(modelId: String): Pair<Provider, String> {
        val parts = modelId.split(":", limit = 2)
        return if (parts.size == 2) {
            val provider = byPrefix[parts[0] + ":"]
                ?: throw IllegalArgumentException("Unknown provider prefix: ${parts[0]}")
            provider to parts[1]
        } else {
            // Default: first configured provider
            val default = providers.values.firstOrNull { it.isConfigured() }
                ?: throw IllegalStateException("No configured providers")
            default to modelId
        }
    }

    fun chat(modelId: String, messages: List<ProviderMessage>, options: ChatOptions = ChatOptions(), tools: List<ToolDefinition> = emptyList()): Flow<ProviderChunk> {
        val (provider, model) = parse(modelId)
        return provider.chat(model, messages, options, tools)
    }

    fun configured(): List<Provider> = providers.values.filter { it.isConfigured() }
    fun all(): List<Provider> = providers.values.toList()
    fun get(prefix: String): Provider? = byPrefix["$prefix:"]
}
