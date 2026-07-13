package com.aura.providers

import io.mockk.coEvery
import io.mockk.every
import io.mockk.mockk
import io.mockk.verify
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.test.StandardTestDispatcher
import kotlinx.coroutines.test.TestScope
import kotlinx.coroutines.test.advanceTimeBy
import kotlinx.coroutines.test.advanceUntilIdle
import kotlinx.coroutines.test.runCurrent
import kotlinx.coroutines.test.runTest
import org.junit.After
import org.junit.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

class ModelCatalogRepositoryTest {

    private val testDispatcher = StandardTestDispatcher()
    private val testScope = TestScope(testDispatcher + SupervisorJob())

    @After
    fun tearDown() {
        // Ensure no leftover jobs
    }

    // ── Initial state ──

    @Test
    fun `initial catalog has no providers and no models`() = runTest {
        val registry = ProviderRegistry(emptyMap())
        val repo = ModelCatalogRepository(
            providerRegistry = registry,
            cache = InMemoryModelCatalogCache(),
            scope = testScope,
        )
        val catalog = repo.catalog.value
        assertEquals(emptyMap(), catalog.providers)
        assertEquals(emptyList(), catalog.allModels)
    }

    // ── No providers ──

    @Test
    fun `refresh with empty registry keeps catalog empty`() = runTest {
        val registry = ProviderRegistry(emptyMap())
        val repo = ModelCatalogRepository(
            providerRegistry = registry,
            cache = InMemoryModelCatalogCache(),
            scope = testScope,
        )
        repo.refresh()
        testScope.advanceUntilIdle()
        val catalog = repo.catalog.value
        assertEquals(emptyMap(), catalog.providers)
        assertEquals(emptyList(), catalog.allModels)
    }

    // ── Single provider success ──

    @Test
    fun `single configured provider returns Ready status with namespaced models`() = runTest {
        val provider = mockk<Provider>(relaxed = true) {
            every { prefix } returns "ollama"
            every { isConfigured() } returns true
            coEvery { listModels() } returns listOf("model-a", "model-b")
        }
        val registry = ProviderRegistry(mapOf("ollama" to provider))
        val repo = ModelCatalogRepository(
            providerRegistry = registry,
            cache = InMemoryModelCatalogCache(),
            scope = testScope,
        )

        repo.refresh()
        testScope.advanceUntilIdle()

        val catalog = repo.catalog.value
        val pml = catalog.providers["ollama"]
        assertNotNull(pml)
        assertEquals(ProviderStatus.Ready, pml.status)
        assertEquals(2, pml.models.size)
        assertEquals("ollama:model-a", pml.models[0].id)
        assertEquals("ollama:model-b", pml.models[1].id)
        assertEquals("model-a", pml.models[0].name)
        assertEquals("ollama", pml.models[0].providerPrefix)

        // allModels reflects successful results
        assertEquals(2, catalog.allModels.size)
        assertEquals("ollama:model-a", catalog.allModels[0].id)
    }

    // ── Unconfigured provider ──

    @Test
    fun `unconfigured provider shows NotConfigured status`() = runTest {
        val provider = mockk<Provider>(relaxed = true) {
            every { prefix } returns "ollama"
            every { isConfigured() } returns false
        }
        val registry = ProviderRegistry(mapOf("ollama" to provider))
        val repo = ModelCatalogRepository(
            providerRegistry = registry,
            cache = InMemoryModelCatalogCache(),
            scope = testScope,
        )

        repo.refresh()
        testScope.advanceUntilIdle()

        val pml = repo.catalog.value.providers["ollama"]
        assertNotNull(pml)
        assertEquals(ProviderStatus.NotConfigured, pml.status)
        assertEquals(emptyList(), pml.models)
    }

    // ── Multiple concurrent providers ──

    @Test
    fun `multiple configured providers queried concurrently`() = runTest {
        val providerA = mockk<Provider>(relaxed = true) {
            every { prefix } returns "prov-a"
            every { isConfigured() } returns true
            coEvery { listModels() } returns listOf("m1")
        }
        val providerB = mockk<Provider>(relaxed = true) {
            every { prefix } returns "prov-b"
            every { isConfigured() } returns true
            coEvery { listModels() } returns listOf("m2")
        }
        val registry = ProviderRegistry(
            linkedMapOf("prov-a" to providerA, "prov-b" to providerB),
        )
        val repo = ModelCatalogRepository(
            providerRegistry = registry,
            cache = InMemoryModelCatalogCache(),
            scope = testScope,
        )

        repo.refresh()
        testScope.advanceUntilIdle()

        val catalog = repo.catalog.value
        assertEquals(2, catalog.providers.size)
        assertEquals(ProviderStatus.Ready, catalog.providers["prov-a"]?.status)
        assertEquals(ProviderStatus.Ready, catalog.providers["prov-b"]?.status)
        assertEquals(2, catalog.allModels.size)
    }

    // ── Partial failure preserves previous success ──

    @Test
    fun `partial failure preserves previous successful results from cache`() = runTest {
        val cache = InMemoryModelCatalogCache()

        // First refresh: both succeed
        val providerA = mockk<Provider>(relaxed = true) {
            every { prefix } returns "prov-a"
            every { isConfigured() } returns true
            coEvery { listModels() } returns listOf("model-a")
        }
        val providerB = mockk<Provider>(relaxed = true) {
            every { prefix } returns "prov-b"
            every { isConfigured() } returns true
            coEvery { listModels() } returns listOf("model-b")
        }
        val registry1 = ProviderRegistry(
            linkedMapOf("prov-a" to providerA, "prov-b" to providerB),
        )
        val repo1 = ModelCatalogRepository(
            providerRegistry = registry1,
            cache = cache,
            scope = testScope,
        )

        repo1.refresh()
        testScope.advanceUntilIdle()

        assertEquals(2, repo1.catalog.value.allModels.size)

        // Second refresh: provider B fails
        val providerBFail = mockk<Provider>(relaxed = true) {
            every { prefix } returns "prov-b"
            every { isConfigured() } returns true
            coEvery { listModels() } throws ProviderCatalogException.NetworkException()
        }
        val registry2 = ProviderRegistry(
            linkedMapOf("prov-a" to providerA, "prov-b" to providerBFail),
        )
        val repo2 = ModelCatalogRepository(
            providerRegistry = registry2,
            cache = cache,
            scope = testScope,
        )

        repo2.refresh()
        testScope.advanceUntilIdle()

        val catalog = repo2.catalog.value
        // Provider B should still show Ready with cached models (stale)
        val pmlB = catalog.providers["prov-b"]
        assertNotNull(pmlB)
        assertEquals(ProviderStatus.Ready, pmlB.status)
        assertEquals(1, pmlB.models.size)
        assertEquals("prov-b:model-b", pmlB.models[0].id)
        assertNotNull(pmlB.errorMessage)

        // Provider A should still be Ready with fresh models
        assertEquals(ProviderStatus.Ready, catalog.providers["prov-a"]?.status)

        // allModels should include both
        assertEquals(2, catalog.allModels.size)
    }

    // ── Force refresh supersedes ──

    @Test
    fun `force refresh supersedes older refresh`() = runTest {
        val provider = mockk<Provider>(relaxed = true) {
            every { prefix } returns "slow"
            every { isConfigured() } returns true
            coEvery { listModels() } coAnswers {
                delay(10_000) // slow provider
                listOf("slow-model")
            }
        }
        val registry = ProviderRegistry(mapOf("slow" to provider))
        val repo = ModelCatalogRepository(
            providerRegistry = registry,
            cache = InMemoryModelCatalogCache(),
            scope = testScope,
            defaultTimeoutMs = 30_000L, // longer than the delay so it doesn't timeout
        )

        // Start first refresh
        repo.refresh()

        // Immediately start second force refresh
        repo.refresh(force = true)

        testScope.advanceUntilIdle()

        val catalog = repo.catalog.value
        // Should have the result from the second refresh
        val pml = catalog.providers["slow"]
        assertNotNull(pml)
        assertEquals(ProviderStatus.Ready, pml.status)
        assertEquals("slow:slow-model", pml.models[0].id)
    }

    // ── Timeout ──

    @Test
    fun `timeout results in Timeout status and cached fallback`() = runTest {
        val cache = InMemoryModelCatalogCache()

        // Cache a previous successful result
        cache.cacheModels("timeout-prov", listOf(
            ModelDescriptor(id = "timeout-prov:cached", name = "cached", providerPrefix = "timeout-prov"),
        ))

        val provider = mockk<Provider>(relaxed = true) {
            every { prefix } returns "timeout-prov"
            every { isConfigured() } returns true
            coEvery { listModels() } coAnswers {
                delay(30_000) // never responds within our short timeout
                listOf("model")
            }
        }
        val registry = ProviderRegistry(mapOf("timeout-prov" to provider))
        val repo = ModelCatalogRepository(
            providerRegistry = registry,
            cache = cache,
            scope = testScope,
            defaultTimeoutMs = 1_000L, // 1 second timeout
        )

        repo.refresh()
        testScope.advanceTimeBy(1_000) // advance just past timeout
        testScope.runCurrent() // run the timeout handler
        testScope.advanceUntilIdle() // finish remaining coroutines

        val pml = repo.catalog.value.providers["timeout-prov"]
        assertNotNull(pml)
        // Should still show Ready with cached models (stale fallback)
        assertEquals(ProviderStatus.Ready, pml.status)
        assertEquals(1, pml.models.size)
        assertEquals("timeout-prov:cached", pml.models[0].id)
        assertNotNull(pml.errorMessage)
    }

    @Test
    fun `timeout without cache returns Timeout status`() = runTest {
        val provider = mockk<Provider>(relaxed = true) {
            every { prefix } returns "timeout-prov"
            every { isConfigured() } returns true
            coEvery { listModels() } coAnswers {
                delay(30_000)
                listOf("model")
            }
        }
        val registry = ProviderRegistry(mapOf("timeout-prov" to provider))
        val repo = ModelCatalogRepository(
            providerRegistry = registry,
            cache = InMemoryModelCatalogCache(),
            scope = testScope,
            defaultTimeoutMs = 1_000L,
        )

        repo.refresh()
        testScope.advanceTimeBy(1_000)
        testScope.runCurrent()
        testScope.advanceUntilIdle()

        val pml = repo.catalog.value.providers["timeout-prov"]
        assertNotNull(pml)
        assertEquals(ProviderStatus.Timeout, pml.status)
        assertNotNull(pml.errorMessage)
    }

    // ── Network error ──

    @Test
    fun `network error returns Network status without cache`() = runTest {
        val provider = mockk<Provider>(relaxed = true) {
            every { prefix } returns "net-prov"
            every { isConfigured() } returns true
            coEvery { listModels() } throws ProviderCatalogException.NetworkException(
                message = "Unable to resolve host: api.example.com",
            )
        }
        val registry = ProviderRegistry(mapOf("net-prov" to provider))
        val repo = ModelCatalogRepository(
            providerRegistry = registry,
            cache = InMemoryModelCatalogCache(),
            scope = testScope,
        )

        repo.refresh()
        testScope.advanceUntilIdle()

        val pml = repo.catalog.value.providers["net-prov"]
        assertNotNull(pml)
        assertEquals(ProviderStatus.Network, pml.status)
        assertNotNull(pml.errorMessage)
    }

    // ── Unauthorized ──

    @Test
    fun `AuthenticationException returns Unauthorized status`() = runTest {
        val provider = mockk<Provider>(relaxed = true) {
            every { prefix } returns "auth-prov"
            every { isConfigured() } returns true
            coEvery { listModels() } throws ProviderCatalogException.AuthenticationException()
        }
        val registry = ProviderRegistry(mapOf("auth-prov" to provider))
        val repo = ModelCatalogRepository(
            providerRegistry = registry,
            cache = InMemoryModelCatalogCache(),
            scope = testScope,
        )

        repo.refresh()
        testScope.advanceUntilIdle()

        val pml = repo.catalog.value.providers["auth-prov"]
        assertNotNull(pml)
        assertEquals(ProviderStatus.Unauthorized, pml.status)
    }

    // ── Rate limit ──

    @Test
    fun `RateLimitedException returns RateLimit status`() = runTest {
        val provider = mockk<Provider>(relaxed = true) {
            every { prefix } returns "rate-prov"
            every { isConfigured() } returns true
            coEvery { listModels() } throws ProviderCatalogException.RateLimitedException()
        }
        val registry = ProviderRegistry(mapOf("rate-prov" to provider))
        val repo = ModelCatalogRepository(
            providerRegistry = registry,
            cache = InMemoryModelCatalogCache(),
            scope = testScope,
        )

        repo.refresh()
        testScope.advanceUntilIdle()

        val pml = repo.catalog.value.providers["rate-prov"]
        assertNotNull(pml)
        assertEquals(ProviderStatus.RateLimit, pml.status)
    }

    // ── Empty provider ──

    @Test
    fun `provider returning empty model list shows Empty status`() = runTest {
        val provider = mockk<Provider>(relaxed = true) {
            every { prefix } returns "empty-prov"
            every { isConfigured() } returns true
            coEvery { listModels() } returns emptyList()
        }
        val registry = ProviderRegistry(mapOf("empty-prov" to provider))
        val repo = ModelCatalogRepository(
            providerRegistry = registry,
            cache = InMemoryModelCatalogCache(),
            scope = testScope,
        )

        repo.refresh()
        testScope.advanceUntilIdle()

        val pml = repo.catalog.value.providers["empty-prov"]
        assertNotNull(pml)
        assertEquals(ProviderStatus.Empty, pml.status)
        assertEquals(emptyList(), pml.models)
    }

    // ── Cache survives repository recreation ──

    @Test
    fun `cache survives repository recreation`() = runTest {
        val cache = InMemoryModelCatalogCache()

        // First repo: cache some models
        val provider = mockk<Provider>(relaxed = true) {
            every { prefix } returns "prov"
            every { isConfigured() } returns true
            coEvery { listModels() } returns listOf("m1")
        }
        val registry1 = ProviderRegistry(mapOf("prov" to provider))
        val repo1 = ModelCatalogRepository(
            providerRegistry = registry1,
            cache = cache,
            scope = testScope,
        )

        repo1.refresh()
        testScope.advanceUntilIdle()
        assertEquals(1, repo1.catalog.value.allModels.size)

        // Second repo with same cache, but provider now fails
        val providerFail = mockk<Provider>(relaxed = true) {
            every { prefix } returns "prov"
            every { isConfigured() } returns true
            coEvery { listModels() } throws ProviderCatalogException.NetworkException()
        }
        val registry2 = ProviderRegistry(mapOf("prov" to providerFail))
        val repo2 = ModelCatalogRepository(
            providerRegistry = registry2,
            cache = cache,
            scope = testScope,
        )

        repo2.refresh()
        testScope.advanceUntilIdle()

        // Should fall back to cached models
        val pml = repo2.catalog.value.providers["prov"]
        assertNotNull(pml)
        assertEquals(ProviderStatus.Ready, pml.status)
        assertEquals(1, pml.models.size)
        assertEquals("prov:m1", pml.models[0].id)
        assertNotNull(pml.errorMessage)
    }

    // ── Model IDs are namespaced exactly once ──

    @Test
    fun `model IDs namespaced exactly once with provider prefix`() = runTest {
        val provider = mockk<Provider>(relaxed = true) {
            every { prefix } returns "ollama"
            every { isConfigured() } returns true
            coEvery { listModels() } returns listOf("model-x")
        }
        val registry = ProviderRegistry(mapOf("ollama" to provider))
        val repo = ModelCatalogRepository(
            providerRegistry = registry,
            cache = InMemoryModelCatalogCache(),
            scope = testScope,
        )

        repo.refresh()
        testScope.advanceUntilIdle()

        val model = repo.catalog.value.allModels.single()
        assertEquals("ollama:model-x", model.id)
        // Verify no double-namespacing
        assertEquals(1, model.id.count { it == ':' })
    }

    // ── Loading state during refresh ──

    @Test
    fun `catalog shows Loading status while provider is being queried`() = runTest {
        val provider = mockk<Provider>(relaxed = true) {
            every { prefix } returns "slow"
            every { isConfigured() } returns true
            coEvery { listModels() } coAnswers {
                delay(5_000)
                listOf("m1")
            }
        }
        val registry = ProviderRegistry(mapOf("slow" to provider))
        val repo = ModelCatalogRepository(
            providerRegistry = registry,
            cache = InMemoryModelCatalogCache(),
            scope = testScope,
            defaultTimeoutMs = 30_000L,
        )

        // Start refresh but don't advance past the delay
        repo.refresh()
        testScope.advanceTimeBy(100) // just enough to start query
        testScope.runCurrent()

        val pml = repo.catalog.value.providers["slow"]
        assertNotNull(pml)
        assertEquals(ProviderStatus.Loading, pml.status)
    }

    // ── MoA included only when configured ──

    @Test
    fun `MoA provider is queried only when configured`() = runTest {
        // MoA that is configured (has keys for dependencies)
        val moaProvider = mockk<Provider>(relaxed = true) {
            every { prefix } returns "moa"
            every { isConfigured() } returns true
            coEvery { listModels() } returns listOf("default")
        }

        val registry = ProviderRegistry(mapOf("moa" to moaProvider))
        val repo = ModelCatalogRepository(
            providerRegistry = registry,
            cache = InMemoryModelCatalogCache(),
            scope = testScope,
        )

        repo.refresh()
        testScope.advanceUntilIdle()

        val pml = repo.catalog.value.providers["moa"]
        assertNotNull(pml)
        assertEquals(ProviderStatus.Ready, pml.status)
        assertEquals("moa:default", pml.models[0].id)
    }

    @Test
    fun `MoA provider shows NotConfigured when dependencies not available`() = runTest {
        val moaProvider = mockk<Provider>(relaxed = true) {
            every { prefix } returns "moa"
            every { isConfigured() } returns false
        }
        val registry = ProviderRegistry(mapOf("moa" to moaProvider))
        val repo = ModelCatalogRepository(
            providerRegistry = registry,
            cache = InMemoryModelCatalogCache(),
            scope = testScope,
        )

        repo.refresh()
        testScope.advanceUntilIdle()

        val pml = repo.catalog.value.providers["moa"]
        assertNotNull(pml)
        assertEquals(ProviderStatus.NotConfigured, pml.status)
    }

    // ── Cached models include timestamp ──

    @Test
    fun `cached provider models include timestamp`() = runTest {
        val cache = InMemoryModelCatalogCache()
        cache.cacheModels("prov", listOf(
            ModelDescriptor(id = "prov:m", name = "m", providerPrefix = "prov"),
        ))

        val cached = cache.getCachedModels("prov")
        assertNotNull(cached)
        assertTrue(cached.cachedAt > 0)
        assertEquals(false, cached.isStale)
    }

    // ── Malformed error ──

    @Test
    fun `unknown error returns Malformed status`() = runTest {
        val provider = mockk<Provider>(relaxed = true) {
            every { prefix } returns "bad-prov"
            every { isConfigured() } returns true
            coEvery { listModels() } throws ProviderCatalogException.MalformedResponseException(
                "Unexpected response format",
            )
        }
        val registry = ProviderRegistry(mapOf("bad-prov" to provider))
        val repo = ModelCatalogRepository(
            providerRegistry = registry,
            cache = InMemoryModelCatalogCache(),
            scope = testScope,
        )

        repo.refresh()
        testScope.advanceUntilIdle()

        val pml = repo.catalog.value.providers["bad-prov"]
        assertNotNull(pml)
        assertEquals(ProviderStatus.Malformed, pml.status)
    }

    // ── Per-provider timeout override ──

    @Test
    fun `per-provider timeout override is respected`() = runTest {
        val shortTimeoutMs = 100L

        val fastProvider = mockk<Provider>(relaxed = true) {
            every { prefix } returns "fast"
            every { isConfigured() } returns true
            coEvery { listModels() } returns listOf("fast-model")
        }
        val slowProvider = mockk<Provider>(relaxed = true) {
            every { prefix } returns "slow"
            every { isConfigured() } returns true
            coEvery { listModels() } coAnswers {
                delay(500)
                listOf("slow-model")
            }
        }
        val registry = ProviderRegistry(
            linkedMapOf("fast" to fastProvider, "slow" to slowProvider),
        )
        val repo = ModelCatalogRepository(
            providerRegistry = registry,
            cache = InMemoryModelCatalogCache(),
            scope = testScope,
            defaultTimeoutMs = 10_000L,
            timeouts = mapOf("slow" to shortTimeoutMs),
        )

        repo.refresh()
        testScope.advanceTimeBy(shortTimeoutMs)
        testScope.runCurrent()
        testScope.advanceUntilIdle()

        // Slow provider should have timed out
        val slowPml = repo.catalog.value.providers["slow"]
        assertNotNull(slowPml)
        assertEquals(ProviderStatus.Timeout, slowPml.status)

        // Fast provider should have succeeded
        val fastPml = repo.catalog.value.providers["fast"]
        assertNotNull(fastPml)
        assertEquals(ProviderStatus.Ready, fastPml.status)
    }

    // ── Cached models marked stale after failure ──

    @Test
    fun `cached models become stale after provider failure`() = runTest {
        val cache = InMemoryModelCatalogCache()

        // First: successful query
        val providerOk = mockk<Provider>(relaxed = true) {
            every { prefix } returns "prov"
            every { isConfigured() } returns true
            coEvery { listModels() } returns listOf("m1")
        }
        val registry1 = ProviderRegistry(mapOf("prov" to providerOk))
        val repo1 = ModelCatalogRepository(
            providerRegistry = registry1,
            cache = cache,
            scope = testScope,
        )
        repo1.refresh()
        testScope.advanceUntilIdle()

        val cachedBefore = cache.getCachedModels("prov")
        assertNotNull(cachedBefore)
        assertEquals(false, cachedBefore.isStale)

        // Second: failure query
        val providerFail = mockk<Provider>(relaxed = true) {
            every { prefix } returns "prov"
            every { isConfigured() } returns true
            coEvery { listModels() } throws ProviderCatalogException.NetworkException()
        }
        val registry2 = ProviderRegistry(mapOf("prov" to providerFail))
        val repo2 = ModelCatalogRepository(
            providerRegistry = registry2,
            cache = cache,
            scope = testScope,
        )
        repo2.refresh()
        testScope.advanceUntilIdle()

        val cachedAfter = cache.getCachedModels("prov")
        assertNotNull(cachedAfter)
        assertEquals(true, cachedAfter.isStale)
    }
}
