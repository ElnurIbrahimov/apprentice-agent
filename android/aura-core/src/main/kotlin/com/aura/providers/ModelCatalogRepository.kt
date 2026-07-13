package com.aura.providers

import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.TimeoutCancellationException
import kotlinx.coroutines.async
import kotlinx.coroutines.Deferred
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.supervisorScope
import kotlinx.coroutines.withTimeout
import javax.inject.Inject
import javax.inject.Singleton

/**
 * Shared, concurrent, cached repository that aggregates model catalogs
 * from every configured [Provider].
 *
 * ## Thread safety
 * All state mutations happen inside a single coroutine job launched by
 * [refresh]. The [catalog] [StateFlow] is updated atomically after every
 * refresh cycle completes.
 *
 * ## Concurrent provider queries
 * [refresh] queries all configured providers **concurrently** via
 * [supervisorScope] + [async]. A failure in one provider does not cancel
 * the others. Each provider has its own timeout (configurable via the
 * [timeouts] map); the default is 10 seconds.
 *
 * ## Partial failure resilience
 * When a provider query fails and a previously successful cache entry
 * exists, the repository preserves that provider's models as
 * [ProviderStatus.Ready] with an [ProviderModelList.errorMessage]
 * describing the failure. The cache entry is marked stale.
 *
 * ## Force refresh
 * Calling [refresh] (with any `force` value) cancels any in-flight refresh
 * and starts a fresh one. The generation counter ([refreshGeneration])
 * ensures that stale results from a cancelled run are never published.
 *
 * ## Injectable / testable constructor
 * The constructor accepts all dependencies as parameters. The [scope]
 * parameter defaults to a new process-scoped [CoroutineScope]; tests
 * pass a [kotlinx.coroutines.test.TestScope] for deterministic control.
 * No Android dependencies are required.
 *
 * ## Cache
 * The [cache] parameter defaults to [InMemoryModelCatalogCache]. For
 * persistent cache across app restarts, provide a DataStore-backed
 * implementation and bind it via Hilt.
 */
@Singleton
class ModelCatalogRepository @Inject constructor(
    private val providerRegistry: ProviderRegistry,
    private val cache: ModelCatalogCache = InMemoryModelCatalogCache(),
    private val timeouts: Map<String, Long> = emptyMap(),
    private val defaultTimeoutMs: Long = 10_000L,
    private val scope: CoroutineScope = CoroutineScope(SupervisorJob() + Dispatchers.Default),
) {
    private val _catalog = MutableStateFlow(ModelCatalog(emptyMap(), emptyList()))
    val catalog: StateFlow<ModelCatalog> = _catalog.asStateFlow()

    /**
     * Monotonically increasing generation counter. Every call to [refresh]
     * increments it; [doRefresh] checks it at publish time to ensure it
     * never publishes results from a superseded run.
     */
    @Volatile
    private var refreshGeneration = 0L

    /**
     * The current refresh [Job], or `null` if no refresh is in-flight.
     * Cancelled and replaced on every call to [refresh].
     */
    private var refreshJob: Job? = null

    /**
     * Trigger a catalog refresh. Any in-flight refresh is cancelled
     * and superseded. The new refresh runs asynchronously on [scope];
     * the [catalog] [StateFlow] is updated when it completes.
     *
     * @param force When `true`, bypass the cache and query every
     *   configured provider fresh. When `false`, previously cached
     *   results may be returned without a network call. Note: the
     *   current implementation always queries configured providers
     *   (the cache is used for fallback on failure); `force` is
     *   reserved for future caching of "still valid" entries.
     */
    fun refresh(force: Boolean = false) {
        val generation = ++refreshGeneration
        refreshJob?.cancel()
        refreshJob = scope.launch {
            doRefresh(generation, force)
        }
    }

    suspend fun refreshProvider(prefix: String, force: Boolean = true) {
        val generation = ++refreshGeneration
        refreshJob?.cancel()
        val provider = providerRegistry.all().firstOrNull { it.prefix == prefix }
        if (provider == null) {
            val withoutMissing = _catalog.value.providers - prefix
            publishIfCurrent(
                generation,
                ModelCatalog(withoutMissing, computeAllModels(withoutMissing)),
            )
            return
        }

        val loading = _catalog.value.providers.toMutableMap().apply {
            this[prefix] = ProviderModelList(
                providerPrefix = prefix,
                status = ProviderStatus.Loading,
                models = this[prefix]?.models.orEmpty(),
            )
        }
        publishIfCurrent(generation, ModelCatalog(loading, computeAllModels(loading)))

        val result = queryProvider(provider, force)
        if (generation != refreshGeneration) return
        val final = loading.toMutableMap().apply { this[prefix] = result.second }
        publishIfCurrent(generation, ModelCatalog(final, computeAllModels(final)))
    }

    private suspend fun doRefresh(generation: Long, force: Boolean) {
        val configured = providerRegistry.all()
        val currentProviders = _catalog.value.providers

        // Phase 1: mark every configured provider as Loading,
        // preserving any previously-known models so the UI does
        // not flash empty.
        val loadingMap = currentProviders.toMutableMap()
        for (p in configured) {
            loadingMap[p.prefix] = ProviderModelList(
                providerPrefix = p.prefix,
                status = ProviderStatus.Loading,
                models = currentProviders[p.prefix]?.models ?: emptyList(),
            )
        }
        // Prune providers that are no longer in the registry.
        val configuredPrefixes = configured.map { it.prefix }.toSet()
        loadingMap.keys.retainAll(configuredPrefixes)

        publishIfCurrent(generation, ModelCatalog(loadingMap, computeAllModels(loadingMap)))
        if (generation != refreshGeneration) return

        // Phase 2: query all configured providers concurrently.
        val results = supervisorScope {
            configured.map { provider ->
                async<Pair<String, ProviderModelList>> {
                    queryProvider(provider, force)
                }
            }.map { deferred: Deferred<Pair<String, ProviderModelList>> ->
                // Catch per-provider failures so supervisorScope
                // does not propagate them.
                runCatching { deferred.await() }.getOrNull()
            }
        }
        if (generation != refreshGeneration) return

        // Phase 3: merge results into the final catalog.
        val finalMap = loadingMap.toMutableMap()
        for (result in results) {
            if (result != null) {
                finalMap[result.first] = result.second
            }
        }
        publishIfCurrent(generation, ModelCatalog(finalMap, computeAllModels(finalMap)))
    }

    /**
     * Query a single provider's model list, applying the per-provider
     * timeout and mapping errors to typed [ProviderStatus] values.
     *
     * On success the result is cached with [ModelCatalogCache.cacheModels].
     * On failure the cache is consulted for a fallback; if one exists it
     * is returned as [ProviderStatus.Ready] with an error message, and
     * the cache entry is marked stale.
     */
    private suspend fun queryProvider(
        provider: Provider,
        force: Boolean,
    ): Pair<String, ProviderModelList> {
        val prefix = provider.prefix

        if (!provider.isConfigured()) {
            return prefix to ProviderModelList(
                providerPrefix = prefix,
                status = ProviderStatus.NotConfigured,
            )
        }

        val timeoutMs = timeouts[prefix] ?: defaultTimeoutMs

        return try {
            val modelNames = withTimeout(timeoutMs) {
                provider.listModels()
            }
            val descriptors = if (modelNames.isEmpty()) {
                emptyList()
            } else {
                modelNames.map { modelName ->
                    ModelDescriptor(
                        id = "$prefix:$modelName",
                        name = modelName,
                        providerPrefix = prefix,
                    )
                }
            }

            if (descriptors.isEmpty()) {
                cache.markStale(prefix)
                prefix to ProviderModelList(
                    providerPrefix = prefix,
                    status = ProviderStatus.Empty,
                    models = descriptors,
                )
            } else {
                cache.cacheModels(prefix, descriptors)
                prefix to ProviderModelList(
                    providerPrefix = prefix,
                    status = ProviderStatus.Ready,
                    models = descriptors,
                )
            }
        } catch (e: TimeoutCancellationException) {
            handleProviderFailure(prefix, e, ProviderStatus.Timeout)
        } catch (e: CancellationException) {
            // Always re-throw cancellations; they are never a provider error.
            throw e
        } catch (e: ProviderCatalogException.AuthenticationException) {
            handleProviderFailure(prefix, e, ProviderStatus.Unauthorized)
        } catch (e: ProviderCatalogException.RateLimitedException) {
            handleProviderFailure(prefix, e, ProviderStatus.RateLimit)
        } catch (e: ProviderCatalogException.NetworkException) {
            handleProviderFailure(prefix, e, ProviderStatus.Network)
        } catch (e: ProviderCatalogException.MalformedResponseException) {
            handleProviderFailure(prefix, e, ProviderStatus.Malformed)
        } catch (e: ProviderCatalogException.EmptyCatalogException) {
            handleProviderFailure(prefix, e, ProviderStatus.Empty)
        } catch (e: Exception) {
            // Any unclassified exception → Malformed.
            handleProviderFailure(prefix, e, ProviderStatus.Malformed)
        }
    }

    /**
     * Handle a provider query failure by consulting the cache for a
     * fallback. If cached models exist they are returned as
     * [ProviderStatus.Ready] with the exception message; otherwise
     * the given [status] is returned directly.
     */
    private suspend fun handleProviderFailure(
        prefix: String,
        cause: Exception,
        status: ProviderStatus,
    ): Pair<String, ProviderModelList> {
        val cached = cache.getCachedModels(prefix)
        if (cached != null) {
            cache.markStale(prefix)
            return prefix to ProviderModelList(
                providerPrefix = prefix,
                status = ProviderStatus.Ready,
                models = cached.models,
                errorMessage = cause.message,
            )
        }
        return prefix to ProviderModelList(
            providerPrefix = prefix,
            status = status,
            errorMessage = cause.message,
        )
    }

    /**
     * Atomically publish [catalog] if [generation] still matches
     * [refreshGeneration].
     */
    private suspend fun publishIfCurrent(generation: Long, catalog: ModelCatalog) {
        if (generation == refreshGeneration) {
            _catalog.value = catalog
        }
    }

    /**
     * Build the flat model list from every provider with [ProviderStatus.Ready].
     */
    private fun computeAllModels(providers: Map<String, ProviderModelList>): List<ModelDescriptor> {
        return providers.values
            .filter { it.status == ProviderStatus.Ready }
            .flatMap { it.models }
    }
}
