package com.aura.ui.viewmodel

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.aura.agent.AgentEvent
import com.aura.agent.AgenticLoop
import com.aura.agent.Brain
import com.aura.agent.Conversation
import com.aura.providers.ProviderRegistry
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import javax.inject.Inject

data class ChatUiState(
    val conversation: Conversation = Conversation(),
    val streaming: Boolean = false,
    val draft: String = "",
    val error: String? = null,
    val activeModel: String = "ollama:deepseek-v3.2:cloud",
    val availableModels: List<String> = emptyList(),
)

@HiltViewModel
class ChatViewModel @Inject constructor(
    private val brain: Brain,
    private val loop: AgenticLoop,
    private val providerRegistry: ProviderRegistry,
) : ViewModel() {

    private val _state = MutableStateFlow(ChatUiState())
    val state: StateFlow<ChatUiState> = _state.asStateFlow()

    private var runJob: Job? = null

    init {
        refreshModels()
    }

    fun refreshModels() {
        viewModelScope.launch {
            val all = providerRegistry.all().flatMap { p ->
                runCatching {
                    p.listModels()
                }.getOrDefault(emptyList()).map { "${p.prefix}:$it" }
            }
            _state.update { it.copy(availableModels = all) }
        }
    }

    fun setDraft(text: String) {
        _state.update { it.copy(draft = text) }
    }

    fun setModel(model: String) {
        _state.update { it.copy(activeModel = model) }
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
        _state.update { it.copy(draft = "", streaming = true, error = null) }

        runJob = viewModelScope.launch {
            try {
                loop.run(current.conversation, model = current.activeModel).collect { event ->
                    when (event) {
                        is AgentEvent.TextDelta -> {
                            current.conversation.turns.lastOrNull()?.let { last ->
                                val updated = last.copy(assistant = (last.assistant ?: "") + event.text)
                                current.conversation.turns[current.conversation.turns.lastIndex] = updated
                            }
                            _state.update { it.copy(conversation = current.conversation) }
                        }
                        is AgentEvent.ToolExecuting -> {
                            _state.update { it.copy(conversation = current.conversation) }
                        }
                        is AgentEvent.ToolResult -> {
                            _state.update { it.copy(conversation = current.conversation) }
                        }
                        is AgentEvent.Error -> {
                            _state.update { it.copy(error = "${event.code}: ${event.message}") }
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
            }
        }
    }
}
