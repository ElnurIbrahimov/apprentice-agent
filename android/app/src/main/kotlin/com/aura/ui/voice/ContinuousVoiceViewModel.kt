package com.aura.ui.voice

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.aura.voice.SpeechToText
import com.aura.voice.TextToSpeech
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import javax.inject.Inject

/**
 * State machine for continuous (hands-free) voice mode.
 *
 * LISTENING → user speaks → STT FinalResult → send() → THINKING
 * THINKING → agent Done → TTS speaks response → SPEAKING
 * SPEAKING → TTS done → LISTENING (loop)
 *
 * Exit conditions:
 * - User taps "Stop"
 * - STT timeout (no speech for 10s in LISTENING state)
 * - User says "stop listening"
 */
enum class VoiceModeState { IDLE, LISTENING, THINKING, SPEAKING }

data class ContinuousVoiceState(
    val active: Boolean = false,
    val phase: VoiceModeState = VoiceModeState.IDLE,
    val partialTranscript: String = "",
    val lastResponse: String = "",
    val error: String? = null,
)

@HiltViewModel
class ContinuousVoiceViewModel @Inject constructor(
    application: Application,
    private val speechToText: SpeechToText,
    private val textToSpeech: TextToSpeech,
) : AndroidViewModel(application) {

    private val _state = MutableStateFlow(ContinuousVoiceState())
    val state: StateFlow<ContinuousVoiceState> = _state.asStateFlow()

    private var silenceTimer: Job? = null

    // ── Single-owner collector jobs ──────────────────────────
    // Only one STT collector and one TTS collector may be active at
    // a time. Each cycle cancels the previous job before starting a
    // new one so stale collectors cannot fire duplicate callbacks.
    private var sttCollectorJob: Job? = null
    private var ttsCollectorJob: Job? = null
    private var responseWaitJob: Job? = null

    /**
     * Start the voice loop. The caller (ChatScreen) provides callbacks
     * for sending a message and listening for completion. The loop
     * runs in viewModelScope and exits when the user taps Stop or
     * the silence timer fires.
     */
    fun startLoop(
        onSend: (String) -> Unit,
        onStreamingDone: () -> Boolean,
    ) {
        _state.value = ContinuousVoiceState(active = true, phase = VoiceModeState.LISTENING)
        startListening(onSend, onStreamingDone)
    }

    fun stopLoop() {
        silenceTimer?.cancel()
        sttCollectorJob?.cancel()
        ttsCollectorJob?.cancel()
        responseWaitJob?.cancel()
        speechToText.cancel()
        textToSpeech.stop()
        _state.update { it.copy(active = false, phase = VoiceModeState.IDLE) }
    }

    private fun startListening(
        onSend: (String) -> Unit,
        onStreamingDone: () -> Boolean,
    ) {
        // Cancel any previous STT collector so only one exists at a time.
        sttCollectorJob?.cancel()
        responseWaitJob?.cancel()

        _state.update { it.copy(phase = VoiceModeState.LISTENING, partialTranscript = "") }
        speechToText.start()

        // Silence timeout: if no speech detected in 10s, stop the loop.
        silenceTimer?.cancel()
        silenceTimer = viewModelScope.launch {
            delay(SILENCE_TIMEOUT_MS)
            if (_state.value.phase == VoiceModeState.LISTENING) {
                stopLoop()
            }
        }

        // Watch STT state for final results — single collector.
        sttCollectorJob = viewModelScope.launch {
            speechToText.state.collect { sttState ->
                when (sttState) {
                    is SpeechToText.State.PartialResult -> {
                        silenceTimer?.cancel()
                        _state.update { it.copy(partialTranscript = sttState.text) }
                    }
                    is SpeechToText.State.FinalResult -> {
                        silenceTimer?.cancel()
                        val text = sttState.text.trim()
                        if (text.isBlank()) {
                            // Empty result — restart listening
                            if (_state.value.active) startListening(onSend, onStreamingDone)
                            return@collect
                        }
                        // Check for stop command
                        if (text.lowercase() in STOP_PHRASES) {
                            stopLoop()
                            return@collect
                        }
                        // Send the message
                        _state.update { it.copy(phase = VoiceModeState.THINKING, partialTranscript = text) }
                        onSend(text)
                        // Wait for streaming to complete, then speak
                        responseWaitJob = viewModelScope.launch {
                            // Poll until streaming is done
                            while (_state.value.active && !onStreamingDone()) {
                                delay(200)
                            }
                            if (!_state.value.active) return@launch
                            speakResponse(onSend, onStreamingDone)
                        }
                    }
                    is SpeechToText.State.Error -> {
                        silenceTimer?.cancel()
                        if (_state.value.active) {
                            _state.update { it.copy(error = sttState.message) }
                            // Retry listening after a brief pause
                            viewModelScope.launch {
                                delay(1000)
                                if (_state.value.active) startListening(onSend, onStreamingDone)
                            }
                        }
                    }
                    else -> Unit
                }
            }
        }
    }

    private fun speakResponse(
        onSend: (String) -> Unit,
        onStreamingDone: () -> Boolean,
    ) {
        // Cancel any previous TTS collector so only one exists at a time.
        ttsCollectorJob?.cancel()

        _state.update { it.copy(phase = VoiceModeState.SPEAKING) }
        val response = _state.value.lastResponse
        if (response.isBlank()) {
            startListening(onSend, onStreamingDone)
            return
        }
        textToSpeech.speak(
            text = response,
            utteranceId = "voice-mode-${System.currentTimeMillis()}",
            flush = true,
        )
        // Watch TTS state — when it goes back to Ready, the utterance
        // is done and we can resume listening. Single collector.
        ttsCollectorJob = viewModelScope.launch {
            textToSpeech.state.collect { ttsState ->
                if (ttsState is TextToSpeech.State.Ready && _state.value.phase == VoiceModeState.SPEAKING) {
                    if (_state.value.active) {
                        startListening(onSend, onStreamingDone)
                    }
                    return@collect
                }
            }
        }
    }

    /**
     * Called by the ChatScreen when the streaming response is complete
     * and the full assistant text is available for TTS.
     */
    fun setLastResponse(text: String) {
        _state.update { it.copy(lastResponse = text) }
    }

    override fun onCleared() {
        stopLoop()
        super.onCleared()
    }

    companion object {
        private const val SILENCE_TIMEOUT_MS = 10_000L
        private val STOP_PHRASES = setOf("stop listening", "stop", "exit voice mode", "that's all")
    }
}