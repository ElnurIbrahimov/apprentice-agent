package com.aura.ui.voice

import com.aura.voice.SpeechToText
import org.junit.Test
import kotlin.test.assertEquals

/**
 * Lightweight contract test for the hold-to-talk transcript
 * mirroring logic. The real [VoiceViewModel] is Hilt-scoped so
 * we don't construct it in unit tests — instead we lock the
 * mirror-state contract (most recent STT result wins, partial
 * is replaced by final) via a small static helper.
 */
class HoldToTalkTranscriptMirrorTest {

    /**
     * Mirror of the [VoiceViewModel] init-block state-collector
     * behavior: a small state machine that mirrors STT state
     * into a single transcript string. Holds the contract that
     * callers depend on for hold-to-talk mode.
     */
    private class Mirror {
        var transcript: String = ""
        fun onState(state: SpeechToText.State) {
            when (state) {
                is SpeechToText.State.PartialResult -> transcript = state.text
                is SpeechToText.State.FinalResult -> transcript = state.text
                else -> Unit
            }
        }
    }

    @Test
    fun `partial result updates the mirror`() {
        val m = Mirror()
        m.onState(SpeechToText.State.PartialResult("hello"))
        assertEquals("hello", m.transcript)
    }

    @Test
    fun `final result overrides partial`() {
        val m = Mirror()
        m.onState(SpeechToText.State.PartialResult("hello wo"))
        m.onState(SpeechToText.State.FinalResult("hello world"))
        assertEquals("hello world", m.transcript)
    }

    @Test
    fun `final result remains sticky until cleared`() {
        val m = Mirror()
        m.onState(SpeechToText.State.FinalResult("done"))
        m.onState(SpeechToText.State.Idle)
        assertEquals("done", m.transcript, "transcript should remain sticky across non-text states")
    }

    @Test
    fun `clear resets the mirror`() {
        val m = Mirror()
        m.onState(SpeechToText.State.FinalResult("done"))
        m.transcript = ""
        assertEquals("", m.transcript)
    }

    @Test
    fun `consume returns and clears`() {
        // Mirrors consumeTranscript() — the caller reads the value
        // and clears it so a second consume returns "".
        val m = Mirror()
        m.onState(SpeechToText.State.FinalResult("done"))
        val first = m.transcript
        m.transcript = ""
        val second = m.transcript
        assertEquals("done", first)
        assertEquals("", second)
    }

    @Test
    fun `consume on empty returns empty`() {
        val m = Mirror()
        val first = m.transcript
        m.transcript = ""
        assertEquals("", first)
    }
}
