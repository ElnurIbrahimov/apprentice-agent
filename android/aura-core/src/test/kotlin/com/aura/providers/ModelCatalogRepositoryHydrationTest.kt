package com.aura.providers

import io.mockk.every
import io.mockk.mockk
import kotlinx.coroutines.test.advanceUntilIdle
import kotlinx.coroutines.test.runTest
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class ModelCatalogRepositoryHydrationTest {

    @Test
    fun `repository hydrates configured provider models from cache without network`() = runTest {
        val provider = mockk<Provider>(relaxed = true)
        every { provider.prefix } returns "ollama"
        every { provider.isConfigured() } returns true

        val registry = mockk<ProviderRegistry>(relaxed = true)
        every { registry.all() } returns listOf(provider)

        val cache = InMemoryModelCatalogCache()
        cache.cacheModels(
            "ollama",
            listOf(ModelDescriptor("ollama:model-a", "model-a", "ollama")),
        )

        val repository = ModelCatalogRepository(
            providerRegistry = registry,
            cache = cache,
            scope = this,
        )
        advanceUntilIdle()

        assertEquals(listOf("ollama:model-a"), repository.catalog.value.allModels.map { it.id })
        assertEquals(ProviderStatus.Ready, repository.catalog.value.providers["ollama"]?.status)
        assertTrue(repository.catalog.value.providers["ollama"]?.errorMessage == null)
    }
}
