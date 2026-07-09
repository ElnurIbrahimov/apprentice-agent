package com.aura.providers

import android.content.Context
import dagger.hilt.android.qualifiers.ApplicationContext
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import javax.inject.Inject
import javax.inject.Singleton

/**
 * Loads MoA presets from an application asset, falling back to a compiled-in
 * default so the provider always has at least one usable preset.
 */
@Singleton
class MoaPresetRepository @Inject constructor(
    @ApplicationContext private val context: Context,
) {
    private val json = Json { ignoreUnknownKeys = true; isLenient = true }

    fun loadPresets(): Map<String, MoaProvider.Preset> = try {
        context.assets.open("moa_presets.json").bufferedReader().use { reader ->
            json.decodeFromString<List<MoaPresetDto>>(reader.readText())
        }
            .map { it.toPreset() }
            .associateBy { it.name }
            .ifEmpty { defaultMap() }
    } catch (e: Exception) {
        defaultMap()
    }

    private fun defaultMap(): Map<String, MoaProvider.Preset> = mapOf(
        // Verified against https://ollama.com/v1/models on 2026-07-09.
        // Every Ollama Cloud model has the `:cloud` suffix in its real
        // id — a missing or stale suffix makes the next MoA turn
        // return "model not found" before any tokens are streamed.
        "default" to MoaProvider.Preset(
            name = "default",
            referenceModels = listOf(
                MoaProvider.ModelRef("ollama", "glm-5.1:cloud"),
                MoaProvider.ModelRef("ollama", "kimi-k2.6:cloud"),
            ),
            aggregator = MoaProvider.ModelRef("deepseek", "deepseek-v4-pro"),
        ),
    )

    @Serializable
    private data class MoaPresetDto(
        val name: String,
        val referenceModels: List<ModelRefDto>,
        val aggregator: ModelRefDto,
        val referenceTemperature: Double = 0.6,
        val aggregatorTemperature: Double = 0.4,
        val enabled: Boolean = true,
    ) {
        fun toPreset(): MoaProvider.Preset = MoaProvider.Preset(
            name = name,
            referenceModels = referenceModels.map { MoaProvider.ModelRef(it.providerPrefix, it.modelName) },
            aggregator = MoaProvider.ModelRef(aggregator.providerPrefix, aggregator.modelName),
            referenceTemperature = referenceTemperature,
            aggregatorTemperature = aggregatorTemperature,
            enabled = enabled,
        )
    }

    @Serializable
    private data class ModelRefDto(
        val providerPrefix: String,
        val modelName: String,
    )
}
