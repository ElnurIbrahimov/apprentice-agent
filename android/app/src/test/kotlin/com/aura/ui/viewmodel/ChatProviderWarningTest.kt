package com.aura.ui.viewmodel

import com.aura.agent.AgentEvent
import kotlin.test.Test
import kotlin.test.assertEquals

class ChatProviderWarningTest {
    @Test
    fun `failover warning adopts the fallback model for the session`() {
        val state = ChatUiState(activeModel = "openai:primary")
        val event = AgentEvent.Warning(
            message = "Primary failed",
            fromModel = "openai:primary",
            toModel = "mistral:fallback",
        )

        val updated = applyProviderWarning(state, event)

        assertEquals("mistral:fallback", updated.activeModel)
        assertEquals("mistral:fallback", updated.sessionModelOverride)
        assertEquals("Primary failed", updated.providerWarning)
    }

    @Test
    fun `generic warning does not rewrite the selected model`() {
        val state = ChatUiState(activeModel = "openai:primary")

        val updated = applyProviderWarning(state, AgentEvent.Warning("Heads up"))

        assertEquals("openai:primary", updated.activeModel)
        assertEquals(null, updated.sessionModelOverride)
        assertEquals("Heads up", updated.providerWarning)
    }
}
