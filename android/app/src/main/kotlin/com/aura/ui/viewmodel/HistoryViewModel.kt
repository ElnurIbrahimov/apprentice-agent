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
    /**
     * Multi-select mode is off by default. Toggled on by long-press
     * on a row, or by the top-bar "Select" action. When true, each
     * row renders a checkbox and the top bar swaps in bulk actions
     * (delete N, share N).
     */
    val selectMode: Boolean = false,
    /**
     * IDs of conversations currently selected. Empty when not in
     * select mode. Survives within the session so partial selection
     * is preserved across scroll.
     */
    val selectedIds: Set<String> = emptySet(),
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
            // Pin-aware order: pinned items surface at the top of
            // the list, the rest fall through to the recent
            // updatedAt desc order.
            val convos = store.recentPinnedFirst(50)
            _state.update { HistoryUiState(conversations = convos, loading = false) }
            searchJob?.cancel()
        }
    }

    /**
     * Rename a conversation. No-op when the new title is blank
     * or the conversation doesn't exist. The new title is
     * reflected in the local list immediately (no DB round-trip
     * for the UI) and the DB is updated asynchronously.
     */
    fun setTitle(id: String, newTitle: String) {
        if (newTitle.isBlank()) return
        // Optimistic update so the row reflects the new title
        // before the DB write completes.
        _state.update { s ->
            s.copy(conversations = s.conversations.map { c ->
                if (c.id == id) c.copy(title = newTitle.trim().take(120)) else c
            })
        }
        viewModelScope.launch {
            val ok = store.setTitle(id, newTitle)
            if (!ok) {
                // Roll back the optimistic update on failure.
                _state.update { s ->
                    s.copy(conversations = s.conversations.map { c ->
                        if (c.id == id) store.load(id) ?: c else c
                    })
                }
            }
        }
    }

    /**
     * Toggle the pinned flag for a conversation. Pin state lives
     * in the conversation's metadata JSON. Pinned items sort to
     * the top of the History list so they're always one tap
     * away — the equivalent of "starred" in chat apps.
     */
    fun togglePinned(id: String) {
        val current = _state.value.conversations.firstOrNull { it.id == id } ?: return
        val target = !store.isPinned(current)
        // Optimistic update.
        val updatedMeta = if (target) current.metadata + ("pinned" to "true")
            else current.metadata - "pinned"
        _state.update { s ->
            s.copy(
                conversations = s.conversations.map { c ->
                    if (c.id == id) c.copy(metadata = updatedMeta) else c
                }
            )
        }
        viewModelScope.launch {
            val ok = store.setPinned(id, target)
            if (!ok) {
                // Roll back.
                _state.update { s ->
                    s.copy(conversations = s.conversations.map { c ->
                        if (c.id == id) store.load(id) ?: c else c
                    })
                }
            }
        }
    }

    /**
     * Whether a conversation is pinned. The pin flag is stored
     * in the conversation's metadata JSON under "pinned".
     */
    fun isPinned(conv: Conversation): Boolean = store.isPinned(conv)

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
                _state.update { it.copy(conversations = store.recentPinnedFirst(50), searching = false) }
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
                _state.update { it.copy(conversations = store.recentPinnedFirst(50)) }
            } else {
                _state.update { it.copy(conversations = store.search(q, 50)) }
            }
        }
    }

    // ── Multi-select ──────────────────────────────────────────────
    //
    // Long-press on a row toggles select mode and pre-selects the
    // tapped row. Tapping another row while in select mode adds it
    // to the selection. Tapping the top-bar "Select" action enters
    // select mode with empty selection. Tapping "Cancel" exits and
    // clears the selection.

    /**
     * Enter or exit select mode. Entering resets the selection to
     * the single tapped [id] (when provided) so the long-press
     * path feels natural. The top-bar "Select" button passes null
     * to enter with empty selection.
     */
    fun toggleSelectMode(id: String? = null) {
        val current = _state.value
        if (current.selectMode) {
            _state.update { it.copy(selectMode = false, selectedIds = emptySet()) }
        } else {
            _state.update {
                it.copy(
                    selectMode = true,
                    selectedIds = if (id != null) setOf(id) else emptySet(),
                )
            }
        }
    }

    /**
     * Toggle a single row's selection. No-op outside select mode.
     * Exit select mode automatically when the selection drops to
     * empty.
     */
    fun toggleSelected(id: String) {
        if (!_state.value.selectMode) return
        val current = _state.value.selectedIds
        val next = if (id in current) current - id else current + id
        if (next.isEmpty()) {
            _state.update { it.copy(selectMode = false, selectedIds = emptySet()) }
        } else {
            _state.update { it.copy(selectedIds = next) }
        }
    }

    /**
     * Select every visible conversation. Used by the "Select all"
     * top-bar action. Replaces the current selection.
     */
    fun selectAll() {
        _state.update { it.copy(selectedIds = it.conversations.map { c -> c.id }.toSet()) }
    }

    /**
     * Delete every selected conversation in a single batch. After
     * deletion the selection is cleared and select mode exits. The
     * delete is sequential (not parallel) to keep the DB on one
     * thread — Room's @Query is single-thread-safe but mixing it
     * with parallel coroutines can produce P2002/P2003 races.
     */
    fun deleteSelected() {
        val ids = _state.value.selectedIds.toList()
        if (ids.isEmpty()) return
        _state.update { it.copy(selectMode = false, selectedIds = emptySet()) }
        viewModelScope.launch {
            for (id in ids) {
                store.delete(id)
            }
            refreshList()
        }
    }

    /**
     * Build a single Markdown document containing every selected
     * conversation. Returns "" if the selection is empty so the
     * caller can skip the share intent.
     */
    fun exportSelectedMarkdown(): String {
        val ids = _state.value.selectedIds
        if (ids.isEmpty()) return ""
        val byId = _state.value.conversations.associateBy { it.id }
        return buildString {
            var first = true
            for (id in ids) {
                val conv = byId[id] ?: continue
                if (!first) append("\n\n---\n\n")
                first = false
                append(exportMarkdown(conv))
            }
        }
    }

    private suspend fun refreshList() {
        val q = _state.value.query
        val convos = if (q.isBlank()) store.recentPinnedFirst(50) else store.search(q, 50)
        _state.update { it.copy(conversations = convos) }
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

    /**
     * Compute stats for a conversation: turn count, tool call count,
     * and duration (updated - created). Used by the History screen
     * to show "12 turns · 3 tools · 2d" per row.
     */
    fun getStats(conversation: Conversation): ConversationStats {
        val turnCount = conversation.turns.count { it.user != null || it.assistant != null }
        val toolCallCount = conversation.turns.sumOf { it.toolTurns.size }
        val durationMs = conversation.updatedAt - conversation.createdAt
        return ConversationStats(
            turns = turnCount,
            toolCalls = toolCallCount,
            durationMs = durationMs,
        )
    }

    data class ConversationStats(
        val turns: Int,
        val toolCalls: Int,
        val durationMs: Long,
    )

    /**
     * Export all conversations as a single Markdown document.
     * Conversations are separated by horizontal rules. Used by
     * the "Export all" button in HistoryScreen.
     */
    fun exportAllMarkdown(): String = buildString {
        val convos = _state.value.conversations
        for ((i, conv) in convos.withIndex()) {
            if (i > 0) append("\n\n---\n\n")
            append(exportMarkdown(conv))
        }
    }
}
