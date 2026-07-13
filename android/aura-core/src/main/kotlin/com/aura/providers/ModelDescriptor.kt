package com.aura.providers

import kotlinx.serialization.Serializable

/**
 * Describes a single model discovered from a provider's model catalog.
 *
 * The [id] is **namespaced exactly once** with the provider prefix,
 * e.g. `"ollama:deepseek-v4-pro:cloud"`. The [name] is the raw model
 * identifier the provider advertises, without any prefix. Callers that
 * dispatch requests use the [id] as the canonical model reference.
 *
 * @property id Fully-qualified model id: `"$providerPrefix:$name"`.
 * @property name Model name as returned by the provider (e.g. `"gpt-4o"`).
 * @property providerPrefix Provider that owns this model (e.g. `"openai"`).
 */
@Serializable
data class ModelDescriptor(
    val id: String,
    val name: String,
    val providerPrefix: String,
)
