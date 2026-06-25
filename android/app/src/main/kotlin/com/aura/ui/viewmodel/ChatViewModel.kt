package com.aura.ui.viewmodel

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.aura.agent.AgentEvent
import com.aura.agent.MemoryAugmentedAgenticLoop
import com.aura.providers.ProviderRegistry
import com.aura.voice.TextToSpeech
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import javax.inject.Inject

data class ChatUiState(
    val conversation: com.aura.agent.Conversation = com.aura.agent.Conversation(),
    val streaming: Boolean = false,
    val draft: String = "",
    val error: String? = null,
    val activeModel: String = "ollama:deepseek-v3.2:cloud",
    val availableModels: List<String> = emptyList(),
    val ttsEnabled: Boolean = true,
)

@HiltViewModel
class ChatViewModel @Inject constructor(
    application: Application,
    private val loop: MemoryAugmentedAgenticLoop,
    private val providerRegistry: ProviderRegistry,
    private val textToSpeech: TextToSpeech,
) : AndroidViewModel(application) {

    private val _state = MutableStateFlow(ChatUiState())
    val state: StateFlow<ChatUiState> = _state.asStateFlow()

    private var runJob: Job? = null
    private var currentResponseBuffer = StringBuilder()

    init {
        refreshModels()
        textToSpeech.initialize()
    }

    override fun onCleared() {
        textToSpeech.shutdown()
        super.onCleared()
    }

    fun refreshModels() {
        viewModelScope.launch {
            val all = providerRegistry.all().flatMap { p ->
                runCatching {
                    p.listModels()
                }.getOrDefault(emptyList()).map { "${p.prefix}:$it" }
            }
            val defaults = listOf(
                "ollama:deepseek-v3.2:cloud",
                "ollama:kimi-k2.6:cloud",
                "anthropic:claude-sonnet-4-5",
                "ollama:minimax-m2.7:cloud",
                "ollama:qwen3-coder:480b-cloud",
            )
            val merged = (defaults + all).distinct()
            _state.update { it.copy(availableModels = merged) }
        }
    }

    fun setDraft(text: String) {
        _state.update { it.copy(draft = text) }
    }

    fun setModel(model: String) {
        _state.update { it.copy(activeModel = model) }
    }

    fun setTtsEnabled(enabled: Boolean) {
        _state.update { it.copy(ttsEnabled = enabled) }
        if (!enabled) textToSpeech.stop()
    }

    fun toggleTts() {
        setTtsEnabled(!_state.value.ttsEnabled)
    }

    fun cancel() {
        runJob?.cancel()
        runJob = null
        _state.update { it.copy(streaming = false) }
    }

    fun send() {
        val current = _state.value
        val text = current.draft.trim()
        if (text.isEmpty() || current.streaming) return

        current.conversation.addUser(text)
        currentResponseBuffer = StringBuilder()
        _state.update { it.copy(draft = "", streaming = true, error = null) }

        runJob = viewModelScope.launch {
            try {
                loop.run(current.conversation, model = current.activeModel).collect { event ->
                    when (event) {
                        is AgentEvent.TextDelta -> {
                            currentResponseBuffer.append(event.text)
                            current.conversation.turns.lastOrNull()?.let { last ->
                                val updated = last.copy(assistant = (last.assistant ?: "") + event.text)
                                current.conversation.turns[current.conversation.turns.lastIndex] = updated
                            }
                            _state.update { it.copy(conversation = current.conversation) }
                        }
                        is AgentEvent.ToolExecuting, is AgentEvent.ToolResult -> {
                            _state.update { it.copy(conversation = current.conversation) }
                        }
                        is AgentEvent.Error -> {
                            _state.update { it.copy(error = "${event.code}: ${event.message}") }
                        }
                        is AgentEvent.Done -> {
                            // Auto-TTS: speak the final assistant turn if enabled
                            if (_state.value.ttsEnabled && currentResponseBuffer.isNotBlank()) {
                                textToSpeech.speak(
                                    text = currentResponseBuffer.toString(),
                                    utteranceId = "turn-${System.currentTimeMillis()}",
                                    flush = true,
                                )
                            }
                        }
                        else -> Unit
                    }
                }
            } catch (e: kotlinx.coroutines.CancellationException) {
                // user cancelled; no-op
            } catch (e: Exception) {
                _state.update { it.copy(error = e.message ?: "unknown error") }
            } finally {
                _state.update { it.copy(streaming = false) }
                currentResponseBuffer = StringBuilder()
            }
        }
    }
}
