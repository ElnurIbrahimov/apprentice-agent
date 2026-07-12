package com.aura.providers

import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import com.aura.usage.UsageTracker
import javax.inject.Inject
import javax.inject.Singleton

/**
 * Holds all providers, routes by `provider:model` prefix.
 * Mirrors aura/providers/registry.py + aura/core/router.py task-aware routing.
 */
@Singleton
class ProviderRegistry @Inject constructor(
    private val providers: Map<String, @JvmSuppressWildcards Provider>,
    private val usageTracker: UsageTracker = UsageTracker(),
) {
    private val byPrefix: Map<String, Provider> = providers.mapKeys { (key, _) -> "$key:" }

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
            val resolvedId = firstConfiguredModelId()
                ?: throw IllegalStateException("No configured provider with an available model")
            val (prefix, model) = resolvedId.split(":", limit = 2)
            providers.getValue(prefix) to model
        }
    }

    /**
     * Resolve the first real model exposed by a configured provider. Providers
     * with failed or empty catalogs are skipped; callers must never synthesize
     * a `provider:default` id that the provider does not advertise.
     */
    suspend fun firstConfiguredModelId(
        excludedPrefixes: Set<String> = emptySet(),
    ): String? {
        for (provider in providers.values) {
            if (provider.prefix in excludedPrefixes || !provider.isConfigured()) continue
            val model = runCatching { provider.listModels().firstOrNull() }.getOrNull()
            if (!model.isNullOrBlank()) return "${provider.prefix}:$model"
        }
        return null
    }

    /**
     * Resolve a model id and dispatch a chat request to the matching
     * provider. Suspending so the dispatch participates in structured
     * concurrency — all 7 callers (Brain.stream, LlmWriteGate,
     * DeepResearchTool, KnowledgeGraphTool, TranslateTool,
     * QuickAskActivity) are already inside coroutine contexts and
     * the previous `runBlocking { parse }` was blocking the calling
     * thread (main thread when invoked from a Compose context).
     */
    suspend fun chat(
        modelId: String,
        messages: List<ProviderMessage>,
        options: ChatOptions = ChatOptions(),
        tools: List<ToolDefinition> = emptyList(),
    ): Flow<ProviderChunk> {
        val (provider, model) = parse(modelId)
        val upstream = provider.chat(model, messages, options, tools)
        // MoA dispatches its reference and aggregator calls back through this
        // registry. Track those concrete calls and skip the synthetic outer
        // flow so a single MoA answer is not double-counted.
        if (provider.prefix == "moa") return upstream
        return flow {
            var outputChars = 0
            var exactUsage: Usage? = null
            var billableChunkSeen = false
            try {
                upstream.collect { chunk ->
                    outputChars += chunk.text?.length ?: 0
                    if (chunk.usage != null) exactUsage = chunk.usage
                    if (chunk.text != null || chunk.usage != null || chunk.finishReason != null) {
                        billableChunkSeen = true
                    }
                    emit(chunk)
                }
            } finally {
                if (billableChunkSeen) {
                    usageTracker.recordLlmCall(
                        modelId = modelId,
                        inputChars = messages.sumOf { it.content.length },
                        outputChars = outputChars,
                        reportedUsage = exactUsage,
                    )
                }
            }
        }
    }

    fun configured(): List<Provider> = providers.values.filter { it.isConfigured() }
    fun all(): List<Provider> = providers.values.toList()
    fun get(prefix: String): Provider? = byPrefix["$prefix:"]
}

