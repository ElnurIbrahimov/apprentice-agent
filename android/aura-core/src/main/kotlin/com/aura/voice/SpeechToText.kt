package com.aura.voice

import android.Manifest
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import android.speech.RecognitionListener
import android.speech.RecognizerIntent
import android.speech.SpeechRecognizer
import androidx.core.content.ContextCompat
import dagger.hilt.android.qualifiers.ApplicationContext
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharedFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asSharedFlow
import kotlinx.coroutines.flow.asStateFlow
import javax.inject.Inject
import javax.inject.Singleton

@Singleton
class SpeechToText @Inject constructor(
    @ApplicationContext private val context: Context,
) {
    sealed class State {
        data object Idle : State()
        data object Ready : State()
        data object Listening : State()
        data class PartialResult(val text: String) : State()
        data class FinalResult(val text: String) : State()
        data class Error(val code: Int, val message: String) : State()
    }

    private val _state = MutableStateFlow<State>(State.Idle)
    val state: StateFlow<State> = _state.asStateFlow()

    /**
     * Convenience: just the latest partial transcript text (or empty if not in a partial state).
     * Mirrors what the UI binds to the "what we heard so far" line.
     */
    val partialTranscript: StateFlow<String> = _state
        .let { src ->
            MutableStateFlow("").also { derived ->
                kotlinx.coroutines.flow.combine(src, MutableStateFlow(Unit)) { s, _ ->
                    (s as? State.PartialResult)?.text ?: (s as? State.FinalResult)?.text ?: ""
                }
            }
        }

    private val _results = MutableSharedFlow<String>(extraBufferCapacity = 16)
    val results: SharedFlow<String> = _results.asSharedFlow()

    private var recognizer: SpeechRecognizer? = null

    fun isAvailable(): Boolean = SpeechRecognizer.isRecognitionAvailable(context)

    fun hasPermission(): Boolean = ContextCompat.checkSelfPermission(
        context, Manifest.permission.RECORD_AUDIO
    ) == PackageManager.PERMISSION_GRANTED

    fun start(languageTag: String = "en-US") {
        if (!hasPermission()) {
            _state.value = State.Error(-1, "RECORD_AUDIO permission not granted")
            return
        }
        if (!isAvailable()) {
            _state.value = State.Error(-2, "No speech recognizer available on this device")
            return
        }
        if (recognizer != null) return
        val r = SpeechRecognizer.createSpeechRecognizer(context)
        val intent = Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
            putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
            putExtra(RecognizerIntent.EXTRA_LANGUAGE, languageTag)
            putExtra(RecognizerIntent.EXTRA_PARTIAL_RESULTS, true)
            putExtra(RecognizerIntent.EXTRA_MAX_RESULTS, 1)
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
                putExtra(RecognizerIntent.EXTRA_PREFER_OFFLINE, true)
            }
        }
        r.setRecognitionListener(object : RecognitionListener {
            override fun onReadyForSpeech(params: Bundle?) { _state.value = State.Ready }
            override fun onBeginningOfSpeech() { _state.value = State.Listening }
            override fun onRmsChanged(rmsdB: Float) {}
            override fun onBufferReceived(buffer: ByteArray?) {}
            override fun onEndOfSpeech() {}
            override fun onError(error: Int) {
                _state.value = State.Error(error, errorString(error))
                cleanup()
            }
            override fun onResults(results: Bundle?) {
                val text = results?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)?.firstOrNull() ?: ""
                _state.value = State.FinalResult(text)
                _results.tryEmit(text)
                cleanup()
            }
            override fun onPartialResults(partial: Bundle?) {
                val text = partial?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)?.firstOrNull() ?: return
                _state.value = State.PartialResult(text)
            }
            override fun onEvent(eventType: Int, params: Bundle?) {}
        })
        r.startListening(intent)
        recognizer = r
        _state.value = State.Idle
    }

    fun stop() {
        recognizer?.stopListening()
    }

    fun cancel() {
        recognizer?.cancel()
        _state.value = State.Idle
        cleanup()
    }

    private fun cleanup() {
        recognizer?.destroy()
        recognizer = null
    }

    private fun errorString(code: Int): String = when (code) {
        SpeechRecognizer.ERROR_AUDIO -> "Audio recording error"
        SpeechRecognizer.ERROR_CLIENT -> "Client side error"
        SpeechRecognizer.ERROR_INSUFFICIENT_PERMISSIONS -> "Insufficient permissions"
        SpeechRecognizer.ERROR_NETWORK -> "Network error"
        SpeechRecognizer.ERROR_NETWORK_TIMEOUT -> "Network timeout"
        SpeechRecognizer.ERROR_NO_MATCH -> "No match"
        SpeechRecognizer.ERROR_RECOGNIZER_BUSY -> "Recognition service busy"
        SpeechRecognizer.ERROR_SERVER -> "Server error"
        SpeechRecognizer.ERROR_SPEECH_TIMEOUT -> "No speech detected"
        else -> "Unknown error ($code)"
    }
}
