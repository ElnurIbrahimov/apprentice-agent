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
    val loading: Boolean = true,
)

@HiltViewModel
class MemoryViewModel @Inject constructor(
    private val memoryStore: MemoryStore,
) : ViewModel() {

    private val _state = MutableStateFlow(MemoryUiState())
    val state: StateFlow<MemoryUiState> = _state.asStateFlow()

    init { refresh() }

    fun refresh() {
        viewModelScope.launch {
            _state.update { it.copy(loading = true) }
            val recent = memoryStore.recent(100)
            _state.update { it.copy(memories = recent, loading = false) }
        }
    }

    fun setQuery(q: String) {
        _state.update { it.copy(query = q) }
    }

    fun search() {
        val q = _state.value.query.trim()
        viewModelScope.launch {
            _state.update { it.copy(loading = true) }
            val results = if (q.isBlank()) memoryStore.recent(100) else memoryStore.query(q, 50)
            _state.update { it.copy(memories = results, loading = false) }
        }
    }

    fun forget(id: String) {
        viewModelScope.launch {
            memoryStore.forget(id)
            refresh()
        }
    }
}
