package com.aura.providers

import io.mockk.every
import io.mockk.mockk
import org.junit.Test
import kotlin.test.assertEquals

class ProviderRouterTest {

    @Test
    fun `route picks first preferred with configured provider`() {
        val p1 = mockk<Provider>(relaxed = true).also { every { it.prefix } returns "anthropic"; every { it.isConfigured() } returns true }
        val p2 = mockk<Provider>(relaxed = true).also { every { it.prefix } returns "ollama"; every { it.isConfigured() } returns false }
        val registry = ProviderRegistry(mapOf("anthropic" to p1, "ollama" to p2))
        val router = ProviderRouter(registry)
        val choice = router.route(ProviderRouter.Task.CHAT)
        // First preferred model in CHAT rule whose provider is configured
        assertEquals(true, choice.startsWith("anthropic:"))
    }
}
