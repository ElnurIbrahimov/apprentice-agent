package com.aura.providers

import com.aura.security.SecureDataStore
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import javax.inject.Inject
import javax.inject.Singleton

@Serializable
private data class PersistedModelCatalog(
    val providers: Map<kotlin.String, CachedProviderModels> = emptyMap(),
)

/**
 * Encrypted process-persistent model catalog cache.
 *
 * The complete cache is written as one document so provider updates are
 * atomic at the DataStore layer. Corrupt or undecryptable cache data is a
 * cache miss; live discovery remains the source of truth.
 */
@Singleton
class SecureModelCatalogCache @Inject constructor(
    private val secureDataStore: SecureDataStore,
) : ModelCatalogCache {

    private val mutex = Mutex()
    private val json = Json {
        ignoreUnknownKeys = true
        encodeDefaults = true
    }

    override suspend fun getCachedModels(providerPrefix: kotlin.String): CachedProviderModels? =
        mutex.withLock { readUnsafe()[providerPrefix] }

    override suspend fun cacheModels(
        providerPrefix: kotlin.String,
        models: List<ModelDescriptor>,
    ) {
        mutate { providers ->
            providers[providerPrefix] = CachedProviderModels(
                models = models,
                cachedAt = System.currentTimeMillis(),
                isStale = false,
            )
        }
    }

    override suspend fun markStale(providerPrefix: kotlin.String) {
        mutate { providers ->
            providers[providerPrefix]?.let { cached ->
                providers[providerPrefix] = cached.copy(isStale = true)
            }
        }
    }

    override suspend fun clear() {
        mutex.withLock {
            secureDataStore.removeString(CACHE_KEY)
        }
    }

    private suspend fun mutate(
        operation: (MutableMap<kotlin.String, CachedProviderModels>) -> Unit,
    ) {
        mutex.withLock {
            val providers = readUnsafe().toMutableMap()
            operation(providers)
            if (providers.isEmpty()) {
                secureDataStore.removeString(CACHE_KEY)
            } else {
                secureDataStore.putString(
                    CACHE_KEY,
                    json.encodeToString(
                        PersistedModelCatalog.serializer(),
                        PersistedModelCatalog(providers),
                    ),
                )
            }
        }
    }

    private suspend fun readUnsafe(): Map<kotlin.String, CachedProviderModels> {
        val raw = runCatching { secureDataStore.getString(CACHE_KEY) }.getOrNull()
            ?: return emptyMap()
        return runCatching {
            json.decodeFromString(PersistedModelCatalog.serializer(), raw).providers
        }.getOrDefault(emptyMap())
    }

    private companion object {
        const val CACHE_KEY: kotlin.String = "model_catalog_cache_v1"
    }
}
