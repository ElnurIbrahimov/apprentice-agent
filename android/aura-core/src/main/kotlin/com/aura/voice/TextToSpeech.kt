package com.aura.voice

import android.content.Context
import android.speech.tts.TextToSpeech
import android.speech.tts.UtteranceProgressListener
import dagger.hilt.android.qualifiers.ApplicationContext
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import java.util.Locale
import javax.inject.Inject
import javax.inject.Singleton

/**
 * Text-to-speech using the platform TTS engine. v1.5 swaps in Piper ONNX
 * for higher-quality offline voices.
 *
 * Lifecycle:
 *   initialize(locale) → state goes Initializing → Ready (or Failed)
 *   speak(text, id) → state goes Speaking → Idle when done
 *   stop() → interrupts current utterance
 */
@Singleton
class TextToSpeech @Inject constructor(
    @ApplicationContext private val context: Context,
) {
    sealed class State {
        data object Idle : State()
        data object Initializing : State()
        data object Ready : State()
        data class Speaking(val utteranceId: String) : State()
        data class Failed(val message: String) : State()
    }

    private val _state = MutableStateFlow<State>(State.Idle)
    val state: StateFlow<State> = _state.asStateFlow()

    private var tts: android.speech.tts.TextToSpeech? = null

    /**
     * Initialize the TTS engine. Idempotent — calling twice is a no-op.
     * Must be called before speak(). The first call is async; the result
     * is reflected in [state].
     */
    fun initialize(locale: Locale = Locale.getDefault(), onReady: () -> Unit = {}) {
        if (tts != null) {
            onReady()
            return
        }
        _state.value = State.Initializing
        tts = android.speech.tts.TextToSpeech(context) { status ->
            if (status == TextToSpeech.SUCCESS) {
                tts?.language = locale
                tts?.setPitch(1.0f)
                tts?.setSpeechRate(1.0f)
                tts?.setOnUtteranceProgressListener(object : UtteranceProgressListener() {
                    override fun onStart(utteranceId: String?) {
                        if (utteranceId != null) _state.value = State.Speaking(utteranceId)
                    }
                    override fun onDone(utteranceId: String?) {
                        _state.value = State.Ready
                    }
                    @Deprecated("Deprecated in API 21+")
                    override fun onError(utteranceId: String?) {
                        _state.value = State.Failed("TTS error on $utteranceId")
                    }
                    override fun onError(utteranceId: String?, errorCode: Int) {
                        _state.value = State.Failed("TTS error $errorCode on $utteranceId")
                    }
                })
                _state.value = State.Ready
                onReady()
            } else {
                _state.value = State.Failed("TTS init failed: $status")
            }
        }
    }

    fun isReady(): Boolean = _state.value is State.Ready || _state.value is State.Speaking

    /**
     * Speak the given text. If currently speaking, queue the new utterance
     * (Android TTS API queues by default with QUEUE_ADD).
     */
    fun speak(text: String, utteranceId: String = "u-${System.currentTimeMillis()}", flush: Boolean = false) {
        val engine = tts ?: run {
            _state.value = State.Failed("TTS not initialized; call initialize() first")
            return
        }
        if (_state.value !is State.Ready && _state.value !is State.Speaking) {
            // Don't try to speak before ready
            return
        }
        val mode = if (flush) TextToSpeech.QUEUE_FLUSH else TextToSpeech.QUEUE_ADD
        engine.speak(text, mode, null, utteranceId)
    }

    fun stop() {
        tts?.stop()
        _state.value = State.Ready
    }

    fun shutdown() {
        tts?.stop()
        tts?.shutdown()
        tts = null
        _state.value = State.Idle
    }
}
