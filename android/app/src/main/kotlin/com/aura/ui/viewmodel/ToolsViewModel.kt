package com.aura.ui.viewmodel

import androidx.lifecycle.ViewModel
import com.aura.agent.ToolRegistry
import com.aura.providers.ToolDefinition
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import javax.inject.Inject

/**
 * State for the Tools browser screen. Holds the raw tool list and
 * the user's search query; `grouped` is the search-filtered result
 * bucketed by [com.aura.tools.ToolCategories].
 */
data class ToolsUiState(
    val tools: List<ToolDefinition> = emptyList(),
    val query: String = "",
    val grouped: List<Pair<String, List<ToolDefinition>>> = emptyList(),
)

/**
 * ViewModel for the Tools browser. The agent's [ToolRegistry] is
 * the single source of truth — the screen doesn't maintain a
 * separate list, it derives everything (search, grouping) from
 * `toolRegistry.definitions()`.
 */
@HiltViewModel
class ToolsViewModel @Inject constructor(
    private val toolRegistry: ToolRegistry,
) : ViewModel() {

    private val _state = MutableStateFlow(loadInitial())
    val state: StateFlow<ToolsUiState> = _state.asStateFlow()

    fun setQuery(query: String) {
        _state.value = _state.value.copy(
            query = query,
            grouped = group(filter(_state.value.tools, query)),
        )
    }

    private fun loadInitial(): ToolsUiState {
        val tools = toolRegistry.definitions()
            .sortedBy { it.name }
        return ToolsUiState(
            tools = tools,
            query = "",
            grouped = group(tools),
        )
    }

    private fun filter(tools: List<ToolDefinition>, query: String): List<ToolDefinition> {
        if (query.isBlank()) return tools
        val q = query.trim().lowercase()
        return tools.filter {
            it.name.lowercase().contains(q) ||
                it.description.lowercase().contains(q)
        }
    }

    /**
     * Group by category, then within each group sort by tool name.
     * Categories appear in the same order as [com.aura.tools.ToolCategories.ALL],
     * with any unknown category (or empty) at the end under "Other".
     */
    private fun group(tools: List<ToolDefinition>): List<Pair<String, List<ToolDefinition>>> {
        val order = com.aura.tools.ToolCategories.ALL
        val bucketed = tools.groupBy { it.category.ifBlank { com.aura.tools.ToolCategories.OTHER } }
        val ordered = mutableListOf<Pair<String, List<ToolDefinition>>>()
        for (cat in order) {
            bucketed[cat]?.let { ordered.add(cat to it.sortedBy { t -> t.name }) }
        }
        // anything not in the order list goes at the end
        bucketed.keys.filter { it !in order }.forEach {
            ordered.add(it to bucketed[it]!!.sortedBy { t -> t.name })
        }
        return ordered
    }
}
