package com.aura.ui.viewmodel

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.aura.kg.KgNode
import com.aura.kg.KnowledgeGraphRepository
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import javax.inject.Inject

data class GraphUiState(
    val query: String = "",
    val nodes: List<KgNode> = emptyList(),
    val selectedNode: KgNode? = null,
    val neighbors: KnowledgeGraphRepository.Neighbors? = null,
    val path: List<KgNode>? = null,
    val loading: Boolean = false,
    val error: String? = null,
)

@HiltViewModel
class GraphViewModel @Inject constructor(
    private val repository: KnowledgeGraphRepository,
) : ViewModel() {

    private val _state = MutableStateFlow(GraphUiState())
    val state: StateFlow<GraphUiState> = _state.asStateFlow()

    init {
        loadRecent()
    }

    private fun loadRecent() {
        viewModelScope.launch {
            _state.update { it.copy(loading = true, error = null) }
            try {
                val nodes = repository.recent(50)
                _state.update { it.copy(nodes = nodes, loading = false) }
            } catch (e: Exception) {
                _state.update { it.copy(error = e.message, loading = false) }
            }
        }
    }

    fun onQueryChange(q: String) {
        _state.update { it.copy(query = q) }
    }

    fun search() {
        val q = _state.value.query.trim()
        if (q.isBlank()) {
            loadRecent()
            return
        }
        viewModelScope.launch {
            _state.update { it.copy(loading = true, error = null) }
            try {
                val nodes = repository.search(q)
                _state.update { it.copy(nodes = nodes, loading = false) }
            } catch (e: Exception) {
                _state.update { it.copy(error = e.message, loading = false) }
            }
        }
    }

    fun selectNode(id: String) {
        viewModelScope.launch {
            try {
                val node = repository.getNode(id)
                val neighbors = repository.getNeighbors(id)
                _state.update { it.copy(selectedNode = node, neighbors = neighbors, path = null) }
            } catch (e: Exception) {
                _state.update { it.copy(error = e.message) }
            }
        }
    }

    fun findPath(fromId: String, toId: String) {
        viewModelScope.launch {
            _state.update { it.copy(loading = true, error = null) }
            try {
                val ids = repository.findPath(fromId, toId)
                val nodes = ids.mapNotNull { id -> runCatching { repository.getNode(id) }.getOrNull() }
                _state.update { it.copy(path = nodes, loading = false) }
            } catch (e: Exception) {
                _state.update { it.copy(error = e.message, loading = false) }
            }
        }
    }

    fun clearSelection() {
        _state.update { it.copy(selectedNode = null, neighbors = null, path = null) }
    }
}
