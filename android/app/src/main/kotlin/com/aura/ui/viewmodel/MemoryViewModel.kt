package com.aura.ui.viewmodel

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.aura.memory.MemoryEditEntity
import com.aura.memory.MemoryEntity
import com.aura.memory.MemoryStore
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import javax.inject.Inject

data class MemoryUiState(
    val memories: List<MemoryEntity> = emptyList(),
    val query: String = "",
    /** When non-null, restricts the list to memories in this category. */
    val categoryFilter: String? = null,
    val loading: Boolean = true,
    /**
     * True while [rebuildEmbeddings] is in progress. Disables the
     * button in the UI. After the rebuild completes, [rebuildResult]
     * holds the count of rows re-embedded (or null if not run).
     */
    val rebuildInFlight: Boolean = false,
    /**
     * Most recent rebuild summary, e.g. "Rebuilt 142 of 142
     * embeddings." Null until the first rebuild completes. The UI
     * renders this as a snackbar-style banner; the user can dismiss
     * it with [clearRebuildResult].
     */
    val rebuildResult: String? = null,
    val undoMessage: String? = null,
    val editHistoryMemoryId: String? = null,
    val editHistory: List<MemoryEditEntity> = emptyList(),
    val editHistoryLoading: Boolean = false,
)

@HiltViewModel
class MemoryViewModel @Inject constructor(
    private val memoryStore: MemoryStore,
) : ViewModel() {

    private val _state = MutableStateFlow(MemoryUiState())
    val state: StateFlow<MemoryUiState> = _state.asStateFlow()

    private data class DeletedMemory(
        val memory: MemoryEntity,
        val edits: List<MemoryEditEntity>,
    )

    private var searchJob: kotlinx.coroutines.Job? = null
    private var lastDeleted: DeletedMemory? = null

    init {
        refresh()
        // Auto-refresh whenever the memory count changes (new memory stored or deleted).
        viewModelScope.launch {
            memoryStore.observeCount().collect { refresh() }
        }
    }

    fun refresh() {
        viewModelScope.launch {
            _state.update { it.copy(loading = true) }
            val current = _state.value
            val results = when {
                current.categoryFilter != null -> memoryStore.listByCategory(current.categoryFilter, 100)
                current.query.isNotBlank() -> memoryStore.searchByText(current.query, 50)
                else -> memoryStore.recent(100)
            }
            _state.update { it.copy(memories = results, loading = false) }
        }
    }

    /**
     * Debounced text search. 250ms keeps Room from being hammered
     * on every keystroke. When the query is cleared, immediately
     * shows the recent list. When a category filter is active, the
     * text search is ignored (the two are mutually exclusive in v1).
     */
    fun setQuery(q: String) {
        _state.update { it.copy(query = q) }
        searchJob?.cancel()
        val trimmed = q.trim()
        if (trimmed.isEmpty()) {
            viewModelScope.launch {
                val results = if (_state.value.categoryFilter != null) {
                    memoryStore.listByCategory(_state.value.categoryFilter!!, 100)
                } else {
                    memoryStore.recent(100)
                }
                _state.update { it.copy(memories = results, loading = false) }
            }
            return
        }
        searchJob = viewModelScope.launch {
            _state.update { it.copy(loading = true) }
            kotlinx.coroutines.delay(250)
            val results = memoryStore.searchByText(trimmed, 50)
            _state.update { it.copy(memories = results, loading = false) }
        }
    }

    /**
     * Set a category filter. Pass null to clear (show all). Tapping a category
     * chip in the UI calls this with the category name; tapping "All" passes
     * null. The category filter takes precedence over the text query — the
     * two are mutually exclusive in the v1 UI.
     */
    fun setCategory(category: String?) {
        _state.update { it.copy(categoryFilter = category) }
        refresh()
    }

    fun search() {
        // Clear any category filter when a text search is explicitly triggered.
        if (_state.value.categoryFilter != null) {
            _state.update { it.copy(categoryFilter = null) }
        }
        refresh()
    }

    fun forget(id: String) {
        viewModelScope.launch {
            val memory = memoryStore.recent(200).find { it.id == id } ?: return@launch
            val edits = memoryStore.getEditHistory(id)
            lastDeleted = DeletedMemory(memory, edits)
            memoryStore.forget(id)
            _state.update { it.copy(undoMessage = "Memory deleted") }
            refresh()
        }
    }

    fun undoDelete() {
        val deleted = lastDeleted ?: return
        viewModelScope.launch {
            memoryStore.restore(deleted.memory, deleted.edits)
            lastDeleted = null
            _state.update { it.copy(undoMessage = null) }
            refresh()
        }
    }

    fun clearUndo() {
        _state.update { it.copy(undoMessage = null) }
        lastDeleted = null
    }

    /**
     * Delete all memories. Irreversible — the UI shows a confirm
     * dialog before calling this.
     */
    fun forgetAll() {
        viewModelScope.launch {
            memoryStore.forgetAll()
            _state.update { it.copy(categoryFilter = null, query = "") }
            refresh()
        }
    }

    /**
     * Delete all memories in [category]. Irreversible.
     */
    fun forgetByCategory(category: String) {
        viewModelScope.launch {
            memoryStore.forgetByCategory(category)
            refresh()
        }
    }

    /**
     * Edit a memory's content + category + importance + tags. Embedding
     * is invalidated by the store; the next recall will re-embed on
     * demand, or the user can run the "Rebuild embeddings" action.
     */
    fun update(id: String, content: String, category: String, importance: Float, tags: String) {
        viewModelScope.launch {
            memoryStore.update(id, content, category, importance, tags)
            refresh()
        }
    }

    fun loadEditHistory(memoryId: String) {
        _state.update {
            it.copy(
                editHistoryMemoryId = memoryId,
                editHistory = emptyList(),
                editHistoryLoading = true,
            )
        }
        viewModelScope.launch {
            val entries = memoryStore.getEditHistory(memoryId)
            _state.update { current ->
                if (current.editHistoryMemoryId != memoryId) current
                else current.copy(editHistory = entries, editHistoryLoading = false)
            }
        }
    }

    fun clearEditHistory() {
        _state.update {
            it.copy(
                editHistoryMemoryId = null,
                editHistory = emptyList(),
                editHistoryLoading = false,
            )
        }
    }

    /**
     * Re-embed every memory whose embedding is currently null.
     * After a backup import, every row has embedding=null and the
     * next recall would re-embed one at a time on demand — slow for
     * a fresh restore with hundreds of rows. This sweeps them all in
     * one pass.
     *
     * Idempotent: memories with an existing embedding are left
     * alone, so running it twice is a no-op the second time.
     *
     * Safe to re-enter: the second call waits for the first to
     * finish because [rebuildInFlight] is set synchronously at the
     * start of the function and the button in the UI is disabled
     * while true.
     */
    fun rebuildEmbeddings() {
        if (_state.value.rebuildInFlight) return
        _state.update { it.copy(rebuildInFlight = true) }
        viewModelScope.launch {
            val total = runCatching { memoryStore.count() }.getOrDefault(0)
            val rebuilt = runCatching { memoryStore.rebuildEmbeddings() }
                .getOrDefault(0)
            val msg = when {
                total == 0 -> "No memories to rebuild."
                rebuilt == 0 -> "All $total memories already have embeddings."
                rebuilt == total -> "Rebuilt $rebuilt embedding${if (rebuilt == 1) "" else "s"}."
                else -> "Rebuilt $rebuilt of $total embeddings (some failed)."
            }
            _state.update {
                it.copy(
                    rebuildInFlight = false,
                    rebuildResult = msg,
                )
            }
            refresh()
        }
    }

    fun clearRebuildResult() {
        _state.update { it.copy(rebuildResult = null) }
    }

    /**
     * Manually create a memory — bypasses the write gate and dedup
     * check. The user explicitly wants this stored, so it goes in
     * directly via [memoryStore.store]. Tagged with source="manual"
     * so the UI can distinguish agent-stored from user-stored notes.
     */
    fun createNote(content: String, category: String, importance: Float) {
        if (content.isBlank()) return
        viewModelScope.launch {
            runCatching {
                memoryStore.store(
                    content = content.trim(),
                    source = "manual",
                    category = category,
                    importance = importance,
                )
            }
            refresh()
        }
    }

    fun renameCategory(oldCategory: String, newCategory: String) {
        viewModelScope.launch {
            runCatching { memoryStore.renameCategory(oldCategory, newCategory) }
            refresh()
        }
    }

    fun mergeCategories(source: String, target: String) {
        viewModelScope.launch {
            runCatching { memoryStore.mergeCategories(source, target) }
            refresh()
        }
    }
}
