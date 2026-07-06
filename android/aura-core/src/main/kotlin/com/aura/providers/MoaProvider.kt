package com.aura.providers

import dagger.Lazy
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Job
import kotlinx.coroutines.async
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.channelFlow
import kotlinx.coroutines.isActive

/**
 * Mixture of Agents virtual provider.
 *
 * Runs multiple reference models in parallel without tool schemas,
 * then calls an aggregator model with the full tool set to synthesize
 * the final response. Reference outputs are injected as private
 * context appended to the last user message.
 *
 * Mirrors the Hermes MoA runtime pattern (tools/mixture_of_agents_tool.py).
 */
class MoaProvider(
    override val prefix: String = "moa",
    override val displayName: String = "Mixture of Agents",
    private val registry: Lazy<ProviderRegistry>,
    presets: Map<String, Preset> = emptyMap(),
) : Provider {

    /** Reference model descriptor — provider prefix + model name. */
    data class ModelRef(val providerPrefix: String, val modelName: String)

    /** Preset: named MoA configuration with reference models + aggregator. */
    data class Preset(
        val name: String,
        val referenceModels: List<ModelRef>,
        val aggregator: ModelRef,
        val referenceTemperature: Double = 0.6,
        val aggregatorTemperature: Double = 0.4,
        val enabled: Boolean = true,
    )

    private val defaultPreset = presets.values.firstOrNull { it.enabled } ?: Preset(
        name = "default",
        referenceModels = listOf(
            ModelRef("ollama", "glm-5.2"),
            ModelRef("ollama", "kimi-k2.7-code"),
        ),
        aggregator = ModelRef("deepseek", "deepseek-v4-pro"),
    )

    private val loadedPresets: Map<String, Preset> = presets.ifEmpty { mapOf("default" to defaultPreset) }

    // Parent job for all in-flight MoA work. Cancelling it tears down the
    // aggregator stream and every reference-model coroutine launched in it.
    private var activeJob: Job? = null

    // ── Provider contract ──

    override fun isConfigured(): Boolean {
        // MoA is configured when the aggregator provider is configured.
        val aggId = "${defaultPreset.aggregator.providerPrefix}:${defaultPreset.aggregator.modelName}"
        return runCatching { registry.get().parse(aggId) }.isSuccess
    }

    override suspend fun listModels(): List<String> = loadedPresets.keys.toList()

    override suspend fun cancel() {
        activeJob?.cancel()
        activeJob = null
    }

    /**
     * Run MoA chat:
     * 1. Resolve the preset by model name (falls back to "default").
     * 2. Run reference models in parallel (no tools).
     * 3. Inject reference outputs into the last user message.
     * 4. Call the aggregator with full tools.
     *
     * Cancellation is cooperative: the caller coroutine owns the [scope];
     * cancelling it stops both the reference model collection and the
     * aggregator stream.
     */
    override fun chat(
        model: String,          // preset name: "default"
        messages: List<ProviderMessage>,
        options: ChatOptions,
        tools: List<ToolDefinition>,
    ): Flow<ProviderChunk> = channelFlow {
        val preset = loadedPresets[model] ?: defaultPreset
        val scope = this
        activeJob = scope.coroutineContext[Job]
        ensureActive()

        if (!preset.enabled) {
            // Disabled preset → aggregator acts alone.
            val aggId = "${preset.aggregator.providerPrefix}:${preset.aggregator.modelName}"
            registry.get().chat(aggId, messages, options, tools).collect { send(it) }
            return@channelFlow
        }

        // 1) Run reference models in parallel — each gets the full message
        //    list but WITHOUT tool schemas, saving tokens and avoiding
        //    strict-provider rejections on tool definitions they don't support.
        val referenceOutputs = runReferenceModels(scope, preset, messages, options)
        ensureActive()

        // 2) Build the aggregator message list: inject reference outputs
        //    into the last user message as private context.
        val aggregatorMessages = buildAggregatorMessages(messages, preset, referenceOutputs)

        // 3) Call the aggregator with full tools.
        val aggId = "${preset.aggregator.providerPrefix}:${preset.aggregator.modelName}"
        val aggOptions = options.copy(temperature = preset.aggregatorTemperature)
        registry.get().chat(aggId, aggregatorMessages, aggOptions, tools).collect { send(it) }
    }

    // ── Reference model execution ──

    private suspend fun runReferenceModels(
        scope: CoroutineScope,
        preset: Preset,
        messages: List<ProviderMessage>,
        options: ChatOptions,
    ): List<ReferenceOutput> = scope.run {
        preset.referenceModels.map { ref ->
            async {
                runReference(ref, messages, options.copy(temperature = preset.referenceTemperature))
            }
        }.map { deferred ->
            runCatching { deferred.await() }.getOrElse { e ->
                if (e is CancellationException) throw e
                ReferenceOutput(
                    providerPrefix = "error",
                    modelName = "error",
                    text = "[Reference model failed: ${e.message ?: "unknown"}]",
                    isError = true,
                )
            }
        }
    }

    private suspend fun runReference(
        ref: ModelRef,
        messages: List<ProviderMessage>,
        options: ChatOptions,
    ): ReferenceOutput {
        val modelId = "${ref.providerPrefix}:${ref.modelName}"
        val text = StringBuilder()
        try {
            registry.get().chat(modelId, messages, options, emptyList()).collect { chunk ->
                if (!currentCoroutineContext().isActive) return@collect
                chunk.text?.let { text.append(it) }
                if (chunk.error != null) {
                    text.append("\n[Error: ${chunk.error.message}]")
                }
            }
        } catch (e: CancellationException) {
            throw e
        } catch (e: Exception) {
            text.append("\n[Exception: ${e.message}]")
        }
        return ReferenceOutput(
            providerPrefix = ref.providerPrefix,
            modelName = ref.modelName,
            text = text.toString().trim(),
        )
    }

    data class ReferenceOutput(
        val providerPrefix: String,
        val modelName: String,
        val text: String,
        val isError: Boolean = false,
    )

    // ── Aggregator message assembly ──

    private fun buildAggregatorMessages(
        messages: List<ProviderMessage>,
        preset: Preset,
        referenceOutputs: List<ReferenceOutput>,
    ): List<ProviderMessage> {
        if (messages.isEmpty()) return messages

        val referenceBlock = buildString {
            appendLine("[MoA Reference Analysis — Private Context]")
            appendLine()
            referenceOutputs.forEach { output ->
                val label = if (output.isError) "ERROR" else "${output.providerPrefix}:${output.modelName}"
                appendLine("## $label")
                appendLine(output.text)
                appendLine()
            }
            appendLine("[End MoA Reference Analysis]")
        }

        // Append reference block to the LAST user message (the one the
        // aggregator is meant to respond to). This is a deterministic,
        // cache-friendly position — the system prompt + prior history
        // (everything before the last user message) stays byte-stable.
        val lastIndex = messages.indexOfLast { it.role == ProviderMessage.Role.user }
        if (lastIndex < 0) return messages

        val amendedMessages = messages.toMutableList()
        val lastUser = amendedMessages[lastIndex]
        amendedMessages[lastIndex] = lastUser.copy(
            content = lastUser.content + "\n\n" + referenceBlock,
        )
        return amendedMessages
    }
}
