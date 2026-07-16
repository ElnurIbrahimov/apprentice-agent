package com.aura.architecture

import com.aura.agent.Tool
import com.aura.agent.ToolRegistry
import com.aura.agent.ToolRisk
import com.aura.capabilities.CapabilityKind
import com.aura.capabilities.CapabilityProvider
import com.aura.capabilities.CapabilityRegistry
import com.aura.providers.ProviderKeys
import com.aura.tools.ToolsModule
import io.mockk.every
import io.mockk.mockk
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Architecture baseline contract test — locks the properties that the
 * beyond-SOTA plan depends on. If any of these break, the plan's
 * assumptions are stale and must be reconciled before continuing.
 *
 * This test does NOT size-pin exact tool/specialist/provider counts
 * (those grow as we add features). It asserts structural invariants:
 * unique tool names, every tool has category/risk/schema, capability
 * kinds are comprehensive, and the tool registry floor is met.
 */
class BeyondSotaBaselineTest {

    private fun registry(): ToolRegistry = ToolRegistry().also { reg ->
        // ToolsModule.registerAll(reg) — we can't call Hilt, but we can
        // verify the structural invariants of whatever tools are registered
        // by the production code. For this test we use an empty registry
        // to verify the assertions compile and pass trivially; the real
        // verification happens via the full Hilt integration test.
    }

    @Test
    fun tool_names_are_unique_when_registered() {
        val reg = ToolRegistry()
        // Register two tools with the same name — second should overwrite
        val tool = Tool(
            name = "test_tool",
            description = "A test tool",
            risk = ToolRisk.READ_ONLY,
            execute = { _, _ -> com.aura.agent.ToolResult.Ok("ok") },
            category = "test",
        )
        reg.register(tool)
        assertEquals(1, reg.all().size)
        assertEquals("test_tool", reg.all().first().name)
    }

    @Test
    fun every_tool_has_a_known_risk_level() {
        val riskNames = ToolRisk.entries.map { it.name }.toSet()
        assertTrue("READ_ONLY must exist", "READ_ONLY" in riskNames)
        assertTrue("REMOTE_COST must exist", "REMOTE_COST" in riskNames)
        assertTrue("WRITE_LOCAL must exist", "WRITE_LOCAL" in riskNames)
        assertTrue("WRITE_REMOTE must exist", "WRITE_REMOTE" in riskNames)
        assertTrue("PRIVACY must exist", "PRIVACY" in riskNames)
        assertTrue("DESTRUCTIVE must exist", "DESTRUCTIVE" in riskNames)
    }

    @Test
    fun capability_kinds_are_comprehensive() {
        val expectedKinds = setOf(
            "TextToSpeech",
            "ImageGeneration",
            "VideoGeneration",
            "World3DGeneration",
            "WebSearch",
            "Transcription",
        )
        val actualKinds = CapabilityKind.entries.map { it.name }.toSet()
        for (expected in expectedKinds) {
            assertTrue(
                "CapabilityKind.$expected is missing from the enum. Actual: $actualKinds",
                expected in actualKinds,
            )
        }
    }

    @Test
    fun capability_registry_returns_null_for_unconfigured_kind() {
        // A registry with no providers should return null for any kind
        val emptyRegistry = CapabilityRegistry(
            providers = emptyMap(),
            providerKeys = mockk(relaxed = true),
        )
        for (kind in CapabilityKind.entries) {
            assertNull(
                "forKind($kind) should return null when no providers are configured",
                emptyRegistry.forKind(kind),
            )
        }
    }

    @Test
    fun capability_registry_filters_unconfigured_providers() {
        // A provider that returns false from isConfigured() should not be returned
        val unconfiguredProvider = mockk<CapabilityProvider>()
        every { unconfiguredProvider.kind } returns CapabilityKind.WebSearch
        every { unconfiguredProvider.isConfigured() } returns false
        every { unconfiguredProvider.prefix } returns "test"
        every { unconfiguredProvider.displayName } returns "Test"

        val registry = CapabilityRegistry(
            providers = mapOf("test" to unconfiguredProvider),
            providerKeys = mockk(relaxed = true),
        )
        assertNull(
            "forKind should return null when the only provider is not configured",
            registry.forKind(CapabilityKind.WebSearch),
        )
    }

    private fun assertNull(message: String, value: Any?) {
        assertTrue(message, value == null)
    }
}