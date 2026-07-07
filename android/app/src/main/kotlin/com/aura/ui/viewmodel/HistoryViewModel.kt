package com.aura.ui.viewmodel

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.aura.agent.Conversation
import com.aura.agent.ConversationStore
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import javax.inject.Inject

data class HistoryUiState(
    val conversations: List<Conversation> = emptyList(),
    val loading: Boolean = true,
    val query: String = "",
    val searching: Boolean = false,
)

/**
 * Owns the conversation-history list plus per-conversation actions
 * (open, delete, export-as-markdown). The search state mirrors the
 * UI in [HistoryScreen] — debouncing and SQL are in
 * [com.aura.agent.ConversationStore.search].
 */
@HiltViewModel
class HistoryViewModel @Inject constructor(
    application: Application,
    private val store: ConversationStore,
) : AndroidViewModel(application) {

    private val _state = MutableStateFlow(HistoryUiState())
    val state: StateFlow<HistoryUiState> = _state.asStateFlow()

    private var searchJob: kotlinx.coroutines.Job? = null

    init { load() }

    fun load() {
        viewModelScope.launch {
            _state.update { it.copy(query = "", loading = true, searching = false) }
            val convos = store.recent(50)
            _state.update { HistoryUiState(conversations = convos, loading = false) }
            searchJob?.cancel()
        }
    }

    /**
     * Debounced full-text search. 250ms keeps Room from being hammered
     * on every keystroke.
     */
    fun onQueryChanged(query: String) {
        _state.update { it.copy(query = query) }
        searchJob?.cancel()
        val trimmed = query.trim()
        if (trimmed.isEmpty()) {
            viewModelScope.launch {
                _state.update { it.copy(conversations = store.recent(50), searching = false) }
            }
            return
        }
        searchJob = viewModelScope.launch {
            _state.update { it.copy(searching = true) }
            kotlinx.coroutines.delay(250)
            val results = store.search(trimmed, 50)
            _state.update { it.copy(conversations = results, searching = false) }
        }
    }

    fun delete(id: String) {
        viewModelScope.launch {
            store.delete(id)
            val q = _state.value.query
            if (q.isBlank()) {
                _state.update { it.copy(conversations = store.recent(50)) }
            } else {
                _state.update { it.copy(conversations = store.search(q, 50)) }
            }
        }
    }

    /**
     * Render [conversation] as a Markdown document. Used by the
     * "Share" action on each row in [com.aura.ui.screens.HistoryScreen].
     *
     * Format:
     *   # Title
     *   _Created ... · Updated ... · model: ..._
     *   ---
     *   ## User
     *   text
     *   ## Assistant
     *   text
     *   ## Tool: name
     *   args
     *   > result
     */
    fun exportMarkdown(conversation: Conversation): String = buildString {
        append("# ").append(conversation.title.ifBlank { "Conversation" }).append('\n')
        append("_Created ").append(formatTimestamp(conversation.createdAt))
        append(" · Updated ").append(formatTimestamp(conversation.updatedAt)).append("_\n")
        conversation.model?.let { append("_Model: `").append(it).append("`_\n") }
        append("---\n\n")
        for (turn in conversation.turns) {
            turn.imageUri?.let { uri ->
                append("![image](").append(uri).append(")\n\n")
            }
            turn.user?.takeIf { it.isNotBlank() }?.let {
                append("## User\n\n").append(it).append("\n\n")
            }
            turn.assistant?.takeIf { it.isNotBlank() }?.let {
                append("## Assistant\n\n").append(it).append("\n\n")
            }
            for (toolTurn in turn.toolTurns) {
                append("## Tool: `").append(toolTurn.name).append("`\n\n")
                append("```json\n").append(toolTurn.args).append("\n```\n\n")
                append("> ").append(toolTurn.result).append("\n\n")
            }
        }
    }

    private fun formatTimestamp(millis: Long): String =
        java.text.SimpleDateFormat("yyyy-MM-dd HH:mm", java.util.Locale.US).format(java.util.Date(millis))
}
