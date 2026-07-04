package com.aura.ui.viewmodel

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.aura.proactive.ProactiveEvents
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.map
import kotlinx.coroutines.flow.stateIn
import javax.inject.Inject

@HiltViewModel
class ProactiveHistoryViewModel @Inject constructor(
    proactiveEvents: ProactiveEvents,
) : ViewModel() {

    val state: StateFlow<ProactiveHistoryUiState> = proactiveEvents.history
        .map { ProactiveHistoryUiState(it) }
        .stateIn(
            viewModelScope,
            SharingStarted.WhileSubscribed(5000),
            ProactiveHistoryUiState(emptyList()),
        )

    init {
        proactiveEvents.markSeen()
    }
}

data class ProactiveHistoryUiState(
    val events: List<com.aura.proactive.ProactiveEventBus.Event> = emptyList(),
)
