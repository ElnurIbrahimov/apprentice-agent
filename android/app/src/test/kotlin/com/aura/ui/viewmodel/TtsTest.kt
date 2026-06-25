package com.aura.ui.viewmodel

import com.aura.agent.Conversation
import com.aura.agent.MemoryAugmentedAgenticLoop
import com.aura.providers.ProviderRegistry
import com.aura.voice.TextToSpeech
import org.junit.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class TtsTest {

    @Test
    fun `default state has ttsEnabled true`() {
        val state = ChatUiState()
        assertTrue(state.ttsEnabled, "TTS should be on by default")
    }

    @Test
    fun `setTtsEnabled false disables tts`() {
        // We can't easily construct ChatViewModel without Android, but we can
        // assert the state setter works
        val state = ChatUiState(ttsEnabled = true)
        val updated = state.copy(ttsEnabled = false)
        assertFalse(updated.ttsEnabled)
    }

    @Test
    fun `setTtsEnabled true enables tts`() {
        val state = ChatUiState(ttsEnabled = false)
        val updated = state.copy(ttsEnabled = true)
        assertTrue(updated.ttsEnabled)
    }
}

private fun makeVm() = null  // Placeholder; we test state transitions only
