package com.aura.providers

import com.aura.security.SecureDataStore
import io.mockk.coEvery
import io.mockk.mockk
import kotlinx.coroutines.test.runTest
import org.junit.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue

class SecureModelCatalogCacheTest {

    @Test
    fun `cached models survive cache instance recreation`() = runTest {
        val backing = mutableMapOf<kotlin.String, kotlin.String>()
        val store = fakeStore(backing)
        val descriptor = ModelDescriptor(
            id = "test:model-a",
            name = "model-a",
            providerPrefix = "test",
        )

        SecureModelCatalogCache(store).cacheModels("test", listOf(descriptor))
        val restored = SecureModelCatalogCache(store).getCachedModels("test")

        assertNotNull(restored)
        assertEquals(listOf(descriptor), restored.models)
        assertFalse(restored.isStale)
        assertTrue(restored.cachedAt > 0L)
    }

    @Test
    fun `stale state and original timestamp survive recreation`() = runTest {
        val backing = mutableMapOf<kotlin.String, kotlin.String>()
        val store = fakeStore(backing)
        val first = SecureModelCatalogCache(store)
        first.cacheModels(
            "test",
            listOf(ModelDescriptor("test:model-a", "model-a", "test")),
        )
        val originalTimestamp = first.getCachedModels("test")?.cachedAt
        first.markStale("test")

        val restored = SecureModelCatalogCache(store).getCachedModels("test")

        assertNotNull(restored)
        assertTrue(restored.isStale)
        assertEquals(originalTimestamp, restored.cachedAt)
    }

    @Test
    fun `corrupt persisted cache is treated as cache miss`() = runTest {
        val backing = mutableMapOf(
            "model_catalog_cache_v1" to "not-json",
        )

        val restored = SecureModelCatalogCache(fakeStore(backing))
            .getCachedModels("test")

        assertNull(restored)
    }

    @Test
    fun `clear removes persisted catalog`() = runTest {
        val backing = mutableMapOf<kotlin.String, kotlin.String>()
        val store = fakeStore(backing)
        val cache = SecureModelCatalogCache(store)
        cache.cacheModels(
            "test",
            listOf(ModelDescriptor("test:model-a", "model-a", "test")),
        )

        cache.clear()

        assertTrue(backing.isEmpty())
        assertNull(SecureModelCatalogCache(store).getCachedModels("test"))
    }

    private fun fakeStore(
        backing: MutableMap<kotlin.String, kotlin.String>,
    ): SecureDataStore = mockk {
        coEvery { getString(any()) } coAnswers { backing[args[0] as kotlin.String] }
        coEvery { putString(any(), any()) } coAnswers {
            backing[args[0] as kotlin.String] = args[1] as kotlin.String
        }
        coEvery { removeString(any()) } coAnswers {
            backing.remove(args[0] as kotlin.String)
            Unit
        }
    }
}
