package com.aura.voice

import io.mockk.mockk
import org.junit.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class TextToSpeechTest {

    @Test
    fun `initial state is Idle`() {
        val tts = TextToSpeech(mockk(relaxed = true))
        assertTrue(tts.state.value is TextToSpeech.State.Idle)
    }

    @Test
    fun `speak before initialize records Failed state`() {
        val tts = TextToSpeech(mockk(relaxed = true))
        tts.speak("hello world")
        val state = tts.state.value
        assertTrue(state is TextToSpeech.State.Failed, "expected Failed, got $state")
        assertEquals("TTS not initialized; call initialize() first", (state as TextToSpeech.State.Failed).message)
    }
}
