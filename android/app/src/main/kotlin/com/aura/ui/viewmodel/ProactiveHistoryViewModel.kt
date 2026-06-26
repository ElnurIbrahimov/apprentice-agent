package com.aura.ui.viewmodel

import android.app.Application; import androidx.lifecycle.AndroidViewModel; import androidx.lifecycle.viewModelScope
import com.aura.proactive.ProactiveEventBus; import com.aura.proactive.ProactiveEvents
import dagger.hilt.EntryPoint; import dagger.hilt.InstallIn; import dagger.hilt.android.EntryPointAccessors
import dagger.hilt.android.lifecycle.HiltViewModel; import dagger.hilt.components.SingletonComponent
import kotlinx.coroutines.flow.MutableStateFlow; import kotlinx.coroutines.flow.StateFlow; import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch; import javax.inject.Inject

data class ProactiveHistoryUiState(val events: List<ProactiveEventBus.Event> = emptyList())
@EntryPoint @InstallIn(SingletonComponent::class) interface ProactiveEntry { fun proactiveEvents(): ProactiveEvents }
@HiltViewModel class ProactiveHistoryViewModel @Inject constructor(app: Application) : AndroidViewModel(app) {
    private val _state = MutableStateFlow(ProactiveHistoryUiState()); val state: StateFlow<ProactiveHistoryUiState> = _state.asStateFlow()
    init { viewModelScope.launch { EntryPointAccessors.fromApplication(getApplication(), ProactiveEntry::class.java).proactiveEvents().history.collect { _state.value = ProactiveHistoryUiState(events = it.reversed()) } } }
}
