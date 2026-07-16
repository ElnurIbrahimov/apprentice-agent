package com.aura.capabilities

import io.mockk.every
import io.mockk.mockk
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class CapabilityRouterTest {

    private fun makeProvider(
        kind: CapabilityKind,
        configured: Boolean = true,
        prefix: kotlin.String = "test",
        operations: Set<kotlin.String> = emptySet(),
    ): CapabilityProvider {
        val provider = mockk<CapabilityProvider>()
        every { provider.kind } returns kind
        every { provider.isConfigured() } returns configured
        every { provider.prefix } returns prefix
        every { provider.displayName } returns "Test $prefix"
        return if (operations.isNotEmpty()) {
            val aware = mockk<CapabilityProvider>()
            every { aware.kind } returns kind
            every { aware.isConfigured() } returns configured
            every { aware.prefix } returns prefix
            every { aware.displayName } returns "Test $prefix"
            // OperationAwareProvider support
            every { aware } returns aware
            // Make it implement OperationAwareProvider via a wrapper
            object : CapabilityProvider, OperationAwareProvider {
                override val prefix = prefix
                override val displayName = "Test $prefix"
                override val kind = kind
                override fun isConfigured() = configured
                override fun supportsOperation(operation: kotlin.String) = operation in operations
            }
        } else {
            provider
        }
    }

    @Test
    fun resolve_returns_null_when_no_providers() {
        val registry = CapabilityRegistry(
            providers = emptyMap(),
            providerKeys = mockk(relaxed = true),
        )
        val router = CapabilityRouter(registry, mockk(relaxed = true))
        assertNull(router.resolve(CapabilityKind.WebSearch))
    }

    @Test
    fun resolve_returns_null_when_provider_not_configured() {
        val unconfigured = mockk<CapabilityProvider>()
        every { unconfigured.kind } returns CapabilityKind.WebSearch
        every { unconfigured.isConfigured() } returns false
        every { unconfigured.prefix } returns "exa"
        every { unconfigured.displayName } returns "Exa"
        val registry = CapabilityRegistry(
            providers = mapOf("exa" to unconfigured),
            providerKeys = mockk(relaxed = true),
        )
        val router = CapabilityRouter(registry, mockk(relaxed = true))
        assertNull(router.resolve(CapabilityKind.WebSearch))
    }

    @Test
    fun resolve_returns_first_configured_provider() {
        val configured = mockk<CapabilityProvider>()
        every { configured.kind } returns CapabilityKind.ImageGeneration
        every { configured.isConfigured() } returns true
        every { configured.prefix } returns "stability"
        every { configured.displayName } returns "Stability"
        val registry = CapabilityRegistry(
            providers = mapOf("stability" to configured),
            providerKeys = mockk(relaxed = true),
        )
        val router = CapabilityRouter(registry, mockk(relaxed = true))
        val result = router.resolve(CapabilityKind.ImageGeneration)
        assertNotNull(result)
        assertEquals("stability", result?.prefix)
    }

    @Test
    fun resolve_with_operation_prefers_operation_aware_provider() {
        val withGenerate = object : CapabilityProvider, OperationAwareProvider {
            override val prefix = "stability"
            override val displayName = "Stability"
            override val kind = CapabilityKind.ImageGeneration
            override fun isConfigured() = true
            override fun supportsOperation(operation: kotlin.String) = operation == "generate"
        }
        val withEdit = object : CapabilityProvider, OperationAwareProvider {
            override val prefix = "dalle"
            override val displayName = "DALL-E"
            override val kind = CapabilityKind.ImageGeneration
            override fun isConfigured() = true
            override fun supportsOperation(operation: kotlin.String) = operation == "edit"
        }
        val registry = CapabilityRegistry(
            providers = mapOf("stability" to withGenerate, "dalle" to withEdit),
            providerKeys = mockk(relaxed = true),
        )
        val router = CapabilityRouter(registry, mockk(relaxed = true))
        val result = router.resolve(CapabilityKind.ImageGeneration, "edit")
        assertNotNull(result)
        assertEquals("dalle", result?.prefix)
    }

    @Test
    fun resolve_with_operation_falls_back_to_first_when_no_match() {
        val configured = mockk<CapabilityProvider>()
        every { configured.kind } returns CapabilityKind.WebSearch
        every { configured.isConfigured() } returns true
        every { configured.prefix } returns "exa"
        every { configured.displayName } returns "Exa"
        val registry = CapabilityRegistry(
            providers = mapOf("exa" to configured),
            providerKeys = mockk(relaxed = true),
        )
        val router = CapabilityRouter(registry, mockk(relaxed = true))
        val result = router.resolve(CapabilityKind.WebSearch, "unrealistic_operation")
        // Falls back to first configured since the provider doesn't implement OperationAwareProvider
        assertNotNull(result)
        assertEquals("exa", result?.prefix)
    }

    @Test
    fun is_available_returns_true_when_configured() {
        val configured = mockk<CapabilityProvider>()
        every { configured.kind } returns CapabilityKind.TextToSpeech
        every { configured.isConfigured() } returns true
        every { configured.prefix } returns "elevenlabs"
        every { configured.displayName } returns "ElevenLabs"
        val registry = CapabilityRegistry(
            providers = mapOf("elevenlabs" to configured),
            providerKeys = mockk(relaxed = true),
        )
        val router = CapabilityRouter(registry, mockk(relaxed = true))
        assertTrue(router.isAvailable(CapabilityKind.TextToSpeech))
        assertTrue(!router.isAvailable(CapabilityKind.VideoGeneration))
    }

    private fun assertNull(value: Any?) {
        assertTrue("Expected null but was $value", value == null)
    }

    private fun assertNotNull(value: Any?) {
        assertTrue("Expected non-null", value != null)
    }
}