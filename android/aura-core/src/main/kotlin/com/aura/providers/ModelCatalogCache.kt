package com.aura.providers

import java.util.concurrent.ConcurrentHashMap
import kotlinx.serialization.Serializable

/**
 * A cached model list for a single provider.
 *
 * @property models The model descriptors that were successfully retrieved.
 * @property cachedAt Timestamp ([System.currentTimeMillis]) when these models
 *   were last successfully fetched.
 * @property isStale `true` when a subsequent refresh attempt failed and the
 *   cache is being used as a fallback. A stale cache has [cachedAt] set to
 *   the *original* successful fetch time, not the failure time.
 */
@Serializable
data class CachedProviderModels(
    val models: List<ModelDescriptor>,
    val cachedAt: Long,
    val isStale: Boolean,
)

/**
 * Persistent cache for provider model lists.
 *
 * Implementations must be **thread-safe** and **repository-lifecycle-safe**:
 * cached entries survive [ModelCatalogRepository] recreation so that a failed
 * refresh on a new instance can still fall back to previously successful
 * results.
 *
 * The cache **does not validate keys or models** — it is a pure
 * storage-and-retrieval layer.
 */
interface ModelCatalogCache {

    /**
     * Retrieve previously cached models for a provider, or `null` if no
     * cache entry exists.
     */
    suspend fun getCachedModels(providerPrefix: String): CachedProviderModels?

    /**
     * Store a successfully fetched model list. Sets [CachedProviderModels.isStale]
     * to `false` and [CachedProviderModels.cachedAt] to the current time.
     */
    suspend fun cacheModels(providerPrefix: String, models: List<ModelDescriptor>)

    /**
     * Mark an existing cache entry as stale without modifying its models
     * or timestamp. No-op when no entry exists for the provider.
     */
    suspend fun markStale(providerPrefix: String)

    /**
     * Remove all cached provider model lists.
     */
    suspend fun clear()
}

/**
 * In-memory implementation of [ModelCatalogCache] backed by a
 * [ConcurrentHashMap]. Entries are lost when the object is garbage-collected;
 * for a persistent (DataStore-backed) cache, create a separate implementation.
 *
 * This is the default cache used by [ModelCatalogRepository] when no other
 * implementation is provided. It is suitable for unit tests and for the
 * common case where the repository is a singleton.
 *
 * **Production binding note:** Because DataStore-based persistence adds
 * DataStore and Context dependencies, the production binding is left to
 * the controller (e.g. ProviderModule / Hilt module). If persistent cache
 * across app restarts is desired, create a DataStoreModelCatalogCache and
 * bind it via Hilt.
 */
class InMemoryModelCatalogCache : ModelCatalogCache {

    private val cache = ConcurrentHashMap<String, CachedProviderModels>()

    override suspend fun getCachedModels(providerPrefix: String): CachedProviderModels? =
        cache[providerPrefix]

    override suspend fun cacheModels(providerPrefix: String, models: List<ModelDescriptor>) {
        cache[providerPrefix] = CachedProviderModels(
            models = models,
            cachedAt = System.currentTimeMillis(),
            isStale = false,
        )
    }

    override suspend fun markStale(providerPrefix: String) {
        cache.computeIfPresent(providerPrefix) { _, entry ->
            entry.copy(isStale = true)
        }
    }

    override suspend fun clear() {
        cache.clear()
    }
}
