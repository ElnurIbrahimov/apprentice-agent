package com.aura.ui.voice

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.aura.voice.SpeechToText
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import javax.inject.Inject

/**
 * ViewModel for the voice overlay. Owns the SpeechToText engine and
 * exposes its state to Compose.
 *
 * Modes:
 *  - **tap-to-speak** (default): overlay auto-dismisses when STT
 *    emits a FinalResult, sending the transcript.
 *  - **hold-to-talk** (`holdToTalk = true`): STT runs while the
 *    overlay is shown. The overlay does NOT auto-dismiss on
 *    FinalResult — the user releases the hold (taps the
 *    dismiss button) and the most recent final OR partial
 *    transcript is sent. This is the model used by the
 *    continuous voice mode button in the input bar.
 */
@HiltViewModel
class VoiceViewModel @Inject constructor(
    application: Application,
    private val speechToText: SpeechToText,
) : AndroidViewModel(application) {

    val sttState: StateFlow<SpeechToText.State> = speechToText.state

    /**
     * The most recent transcript (final or partial). Surfaced for
     * hold-to-talk mode so the caller can send the latest text
     * when the user releases the hold.
     */
    private val _lastTranscript = MutableStateFlow("")
    val lastTranscript: StateFlow<String> = _lastTranscript.asStateFlow()

    init {
        // Mirror STT state into a single transcript string so the
        // caller (hold-to-talk) can read it without subscribing to
        // the typed STT state. Partial and final both flow through
        // here; final is sticky until reset.
        viewModelScope.launch {
            speechToText.state.collect { state ->
                when (state) {
                    is SpeechToText.State.PartialResult -> {
                        _lastTranscript.value = state.text
                    }
                    is SpeechToText.State.FinalResult -> {
                        _lastTranscript.value = state.text
                    }
                    else -> Unit
                }
            }
        }
    }

    fun start() {
        if (!speechToText.hasPermission()) return
        _lastTranscript.value = ""
        speechToText.start()
    }

    fun stop() {
        speechToText.stop()
    }

    fun cancel() {
        speechToText.cancel()
        _lastTranscript.value = ""
    }

    fun reset() {
        speechToText.cancel()
        _lastTranscript.value = ""
    }

    /**
     * Read and clear the most recent transcript. Used by
     * hold-to-talk mode on release — returns "" when nothing was
     * said.
     */
    fun consumeTranscript(): String {
        val t = _lastTranscript.value
        _lastTranscript.value = ""
        return t
    }
}
