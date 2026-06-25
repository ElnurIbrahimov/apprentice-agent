package com.aura.providers

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
    fun `parse throws on unknown prefix`() {
        val p = mockk<Provider>(relaxed = true)
        val registry = ProviderRegistry(mapOf("foo" to p))
        assertFailsWith<IllegalArgumentException> { registry.parse("bar:baz") }
    }
}
