package com.aura.ui.screens.chat

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertNull
import kotlin.test.assertTrue

class ChatPermissionStateTest {

    @Test
    fun `first microphone grant persists and launches the pending mode`() {
        val pending = MicPermissionState(granted = false).request(ChatVoiceMode.Hold)

        val outcome = pending.resolve(granted = true)

        assertTrue(outcome.state.granted)
        assertNull(outcome.state.pendingMode)
        assertEquals(ChatVoiceMode.Hold, outcome.launchMode)
    }

    @Test
    fun `denial clears pending mode without granting permission`() {
        val pending = MicPermissionState(granted = false).request(ChatVoiceMode.Continuous)

        val outcome = pending.resolve(granted = false)

        assertFalse(outcome.state.granted)
        assertNull(outcome.state.pendingMode)
        assertNull(outcome.launchMode)
    }
}
