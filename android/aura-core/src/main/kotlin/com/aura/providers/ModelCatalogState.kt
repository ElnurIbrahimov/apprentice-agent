package com.aura.providers

/**
 * Status of a single provider's model catalog after the most recent
 * [ModelCatalogRepository.refresh] attempt.
 */
enum class ProviderStatus {
    /** The provider is not configured (no API key, missing dependencies). */
    NotConfigured,

    /** A catalog request is in-flight. */
    Loading,

    /** The provider returned a valid model list (may be stale models
     * from cache if the live query failed — check [ProviderModelList.errorMessage]). */
    Ready,

    /** 401/403 — the API key was rejected. */
    Unauthorized,

    /** 429 — rate-limited by the upstream provider. */
    RateLimit,

    /** Network-level failure (DNS, connection reset, unreachable). */
    Network,

    /** The request exceeded the per-provider timeout. */
    Timeout,

    /** The provider returned an unparsable or unexpected response. */
    Malformed,

    /** The provider responded successfully but returned zero models. */
    Empty,
}

/**
 * Snapshot of one provider's model-list state at a point in time.
 *
 * @property providerPrefix The provider key (e.g. `"ollama"`, `"moa"`).
 * @property status Current status of this provider's catalog.
 * @property models Models currently visible for this provider. When [status]
 *   is [ProviderStatus.Ready] these are the latest results (possibly stale
 *   from cache). For other statuses this may be empty or hold cached entries.
 * @property errorMessage Human-readable error description (null when healthy).
 *   Never includes raw API keys or full response bodies.
 */
data class ProviderModelList(
    val providerPrefix: String,
    val status: ProviderStatus,
    val models: List<ModelDescriptor> = emptyList(),
    val errorMessage: String? = null,
)

/**
 * Immutable snapshot of the full model catalog across all configured
 * providers.
 *
 * @property providers Per-provider states, keyed by [ProviderModelList.providerPrefix].
 * @property allModels Flat list of every model from every [ProviderStatus.Ready]
 *   provider, in provider-registration order. Each model id is namespaced
 *   exactly once.
 */
data class ModelCatalog(
    val providers: Map<String, ProviderModelList>,
    val allModels: List<ModelDescriptor>,
)
