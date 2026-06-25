package com.aura.ui.voice

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import com.aura.voice.SpeechToText
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.flow.StateFlow
import javax.inject.Inject

/**
 * ViewModel for the voice overlay. Owns the SpeechToText engine and
 * exposes its state to Compose.
 */
@HiltViewModel
class VoiceViewModel @Inject constructor(
    application: Application,
    private val speechToText: SpeechToText,
) : AndroidViewModel(application) {

    val sttState: StateFlow<SpeechToText.State> = speechToText.state

    fun start() {
        if (!speechToText.hasPermission()) return
        speechToText.start()
    }

    fun stop() {
        speechToText.stop()
    }

    fun cancel() {
        speechToText.cancel()
    }

    fun reset() {
        speechToText.cancel()
    }
}
