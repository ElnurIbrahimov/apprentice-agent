package com.aura.ui.viewmodel

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
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
)

@HiltViewModel
class MemoryViewModel @Inject constructor(
    private val memoryStore: MemoryStore,
) : ViewModel() {

    private val _state = MutableStateFlow(MemoryUiState())
    val state: StateFlow<MemoryUiState> = _state.asStateFlow()

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
                current.query.isNotBlank() -> memoryStore.query(current.query, 50)
                else -> memoryStore.recent(100)
            }
            _state.update { it.copy(memories = results, loading = false) }
        }
    }

    fun setQuery(q: String) {
        _state.update { it.copy(query = q) }
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
            memoryStore.forget(id)
            refresh()
        }
    }
}
