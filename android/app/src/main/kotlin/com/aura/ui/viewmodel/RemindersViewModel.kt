package com.aura.ui.viewmodel

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.aura.tasks.ReminderEntity
import com.aura.tasks.ReminderStore
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.combine
import kotlinx.coroutines.launch
import javax.inject.Inject

data class RemindersUiState(
    val upcoming: List<ReminderEntity> = emptyList(),
    val history: List<ReminderEntity> = emptyList(),
    val loading: Boolean = true,
)

@HiltViewModel
class RemindersViewModel @Inject constructor(
    private val store: ReminderStore,
) : ViewModel() {
    private val _state = MutableStateFlow(RemindersUiState())
    val state: StateFlow<RemindersUiState> = _state.asStateFlow()

    init {
        viewModelScope.launch {
            combine(store.observeUpcoming(), store.observeHistory()) { upcoming, history ->
                RemindersUiState(upcoming = upcoming, history = history, loading = false)
            }.collect { snapshot -> _state.value = snapshot }
        }
    }

    fun create(message: String, triggerAt: Long, recurrence: String) {
        if (message.isBlank()) return
        viewModelScope.launch { store.create(message, triggerAt, recurrence) }
    }

    fun update(id: String, message: String, triggerAt: Long, recurrence: String) {
        if (message.isBlank()) return
        viewModelScope.launch { store.update(id, message, triggerAt, recurrence) }
    }

    fun cancel(id: String) {
        viewModelScope.launch { store.cancel(id) }
    }

    fun delete(id: String) {
        viewModelScope.launch { store.delete(id) }
    }

    fun clearHistory() {
        viewModelScope.launch { store.clearHistory() }
    }
}
