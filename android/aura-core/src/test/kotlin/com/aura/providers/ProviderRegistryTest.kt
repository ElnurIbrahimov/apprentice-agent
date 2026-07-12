package com.aura.providers

import io.mockk.coEvery
import io.mockk.every
import io.mockk.mockk
import kotlinx.coroutines.test.runTest
import org.junit.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith

class ProviderRegistryTest {

    @Test
    fun `parse splits on colon`() = runTest {
        val p = mockk<Provider>(relaxed = true)
        val registry = ProviderRegistry(mapOf("foo" to p))
        val (prov, model) = registry.parse("foo:bar")
        assertEquals(p, prov)
        assertEquals("bar", model)
    }

    @Test
    fun `parse default resolves to first configured provider model`() = runTest {
        val p = mockk<Provider>(relaxed = true) {
            every { isConfigured() } returns true
            every { prefix } returns "foo"
            coEvery { listModels() } returns listOf("foo-model")
        }
        val registry = ProviderRegistry(mapOf("foo" to p))
        val (prov, model) = registry.parse("default")
        assertEquals(p, prov)
        assertEquals("foo-model", model)
    }

    @Test
    fun `parse default falls back to raw model when listModels fails`() = runTest {
        val p = mockk<Provider>(relaxed = true) {
            every { isConfigured() } returns true
            every { prefix } returns "foo"
            coEvery { listModels() } throws RuntimeException("network")
        }
        val registry = ProviderRegistry(mapOf("foo" to p))
        val (prov, model) = registry.parse("default")
        assertEquals(p, prov)
        assertEquals("default", model)
    }

    @Test
    fun `parse throws on unknown prefix`() = runTest {
        val p = mockk<Provider>(relaxed = true)
        val registry = ProviderRegistry(mapOf("foo" to p))
        assertFailsWith<IllegalArgumentException> { registry.parse("bar:baz") }
    }
}
