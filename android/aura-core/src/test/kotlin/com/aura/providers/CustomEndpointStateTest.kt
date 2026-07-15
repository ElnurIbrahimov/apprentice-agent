package com.aura.providers

import io.mockk.mockk
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Tests the user's "Custom Endpoint" path: the user can point Aura at any
 * OpenAI-compatible URL/key/model and the state object tracks it.
 *
 * The actual HTTP path is exercised by [CustomOpenAiCompatProvider] integration
 * tests; here we verify the state contract that the Settings UI depends on.
 */
class CustomEndpointStateTest {

    @Test
    fun `fresh state is unconfigured`() {
        val state = CustomEndpointState()
        assertFalse(state.isConfigured())
    }

    @Test
    fun `setEndpoint makes state configured`() {
        val state = CustomEndpointState()
        state.setEndpoint("https://api.example.com/v1", "sk-test-123")
        assertTrue(state.isConfigured())
        assertEquals("https://api.example.com/v1", state.baseUrl)
        assertEquals("sk-test-123", state.apiKey)
    }

    @Test
    fun `setEndpoint trims trailing slash from baseUrl`() {
        val state = CustomEndpointState()
        state.setEndpoint("https://api.example.com/v1/", "sk-test-123")
        assertEquals("https://api.example.com/v1", state.baseUrl)
    }

    @Test
    fun `setEndpoint overrides models`() {
        val state = CustomEndpointState()
        state.setEndpoint(
            baseUrl = "https://api.example.com/v1",
            apiKey = "sk-test-123",
            modelOverride = listOf("custom-1", "custom-2"),
        )
        assertEquals(listOf("custom-1", "custom-2"), state.modelOverride)
    }

    @Test
    fun `snapshot is atomic`() {
        val state = CustomEndpointState()
        state.setEndpoint("https://api.example.com/v1", "sk-1")
        val (url, key, models) = state.snapshot()
        assertEquals("https://api.example.com/v1", url)
        assertEquals("sk-1", key)
        assertTrue(models.isEmpty())
    }

    @Test
    fun `setEndpoint with blank key leaves state unconfigured`() {
        val state = CustomEndpointState()
        state.setEndpoint("https://api.example.com/v1", "  ")
        assertFalse(state.isConfigured())
    }
}
