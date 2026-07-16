package com.aura.providers

import com.aura.data.UserPreferences
import io.mockk.coEvery
import io.mockk.every
import io.mockk.mockk
import kotlinx.coroutines.flow.flowOf
import kotlinx.coroutines.test.runTest
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class ModelRoleRouterTest {

    private val userPreferences = mockk<UserPreferences>(relaxed = true)
    private val providerRegistry = mockk<ProviderRegistry>(relaxed = true)
    private val router = ModelRoleRouter(userPreferences, providerRegistry)

    @Test
    fun resolve_returns_role_specific_model_when_set() = runTest {
        every { userPreferences.forRole(ModelRole.CREATIVE_DRAFT) } returns flowOf("ollama:deepseek-v4-pro:cloud")
        coEvery { userPreferences.defaultModel } returns flowOf("ollama:gemma4:e4b")
        // Need to mock the Flow returned by forRole
        val result = router.resolve(ModelRole.CREATIVE_DRAFT)
        assertEquals("ollama:deepseek-v4-pro:cloud", result)
    }

    @Test
    fun resolve_falls_back_to_default_when_role_not_set() = runTest {
        every { userPreferences.forRole(ModelRole.PLANNER) } returns flowOf(null)
        every { userPreferences.defaultModel } returns flowOf("ollama:deepseek-v4-pro:cloud")
        val result = router.resolve(ModelRole.PLANNER)
        assertEquals("ollama:deepseek-v4-pro:cloud", result)
    }

    @Test
    fun resolve_returns_null_when_nothing_configured() = runTest {
        every { userPreferences.forRole(ModelRole.VERIFIER) } returns flowOf(null)
        every { userPreferences.defaultModel } returns flowOf(null)
        val result = router.resolve(ModelRole.VERIFIER)
        assertNull(result)
    }

    @Test
    fun resolve_conversation_uses_default_model_directly() = runTest {
        every { userPreferences.forRole(ModelRole.CONVERSATION) } returns flowOf("ollama:deepseek-v4-pro:cloud")
        val result = router.resolve(ModelRole.CONVERSATION)
        assertEquals("ollama:deepseek-v4-pro:cloud", result)
    }

    @Test
    fun configurable_roles_exclude_embedding() {
        val configurable = ModelRole.configurable
        assertTrue(ModelRole.EMBEDDING !in configurable)
        assertTrue(ModelRole.CONVERSATION in configurable)
        assertTrue(ModelRole.CREATIVE_DRAFT in configurable)
        assertTrue(ModelRole.CREATIVE_CRITIC in configurable)
        assertTrue(ModelRole.PLANNER in configurable)
        assertTrue(ModelRole.VERIFIER in configurable)
    }

    private fun assertNull(value: Any?) {
        assertTrue("Expected null but was $value", value == null)
    }
}