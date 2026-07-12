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
    fun `parse default rejects providers whose model catalog fails`() = runTest {
        val p = mockk<Provider>(relaxed = true) {
            every { isConfigured() } returns true
            every { prefix } returns "foo"
            coEvery { listModels() } throws RuntimeException("network")
        }
        val registry = ProviderRegistry(mapOf("foo" to p))
        assertFailsWith<IllegalStateException> { registry.parse("default") }
    }

    @Test
    fun `firstConfiguredModelId skips failed and empty provider catalogs`() = runTest {
        val failed = mockk<Provider>(relaxed = true) {
            every { isConfigured() } returns true
            every { prefix } returns "failed"
            coEvery { listModels() } throws RuntimeException("network")
        }
        val empty = mockk<Provider>(relaxed = true) {
            every { isConfigured() } returns true
            every { prefix } returns "empty"
            coEvery { listModels() } returns emptyList()
        }
        val valid = mockk<Provider>(relaxed = true) {
            every { isConfigured() } returns true
            every { prefix } returns "valid"
            coEvery { listModels() } returns listOf("real-model")
        }
        val registry = ProviderRegistry(
            linkedMapOf("failed" to failed, "empty" to empty, "valid" to valid),
        )

        assertEquals("valid:real-model", registry.firstConfiguredModelId())
        assertEquals(null, registry.firstConfiguredModelId(setOf("valid")))
    }

    @Test
    fun `parse throws on unknown prefix`() = runTest {
        val p = mockk<Provider>(relaxed = true)
        val registry = ProviderRegistry(mapOf("foo" to p))
        assertFailsWith<IllegalArgumentException> { registry.parse("bar:baz") }
    }
}
