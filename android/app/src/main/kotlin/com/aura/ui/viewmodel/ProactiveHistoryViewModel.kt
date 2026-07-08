package com.aura.ui.viewmodel

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.aura.proactive.ProactiveEvents
import com.aura.proactive.ProactiveRunner
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.map
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.launch
import javax.inject.Inject

@HiltViewModel
class ProactiveHistoryViewModel @Inject constructor(
    proactiveEvents: ProactiveEvents,
    private val runner: ProactiveRunner,
) : ViewModel() {

    val state: StateFlow<ProactiveHistoryUiState> = proactiveEvents.history
        .map { ProactiveHistoryUiState(it) }
        .stateIn(
            viewModelScope,
            SharingStarted.WhileSubscribed(5000),
            ProactiveHistoryUiState(emptyList()),
        )

    /**
     * Status of the most recent "fire now" tap. Renders as a
     * snackbar / inline message on the screen. Cleared on
     * the next [clearStatus] call or after a few seconds.
     */
    private val _status = MutableStateFlow<String?>(null)
    val status: StateFlow<String?> = _status.asStateFlow()

    init {
        proactiveEvents.markSeen()
    }

    fun fireMorningBrief() = run { _status.value = "Firing morning brief…" }
        .also { viewModelScope.launch {
            val r = runner.fireMorningBrief()
            _status.value = r.toMessage()
        } }

    fun fireDecayPass() = run { _status.value = "Firing decay pass…" }
        .also { viewModelScope.launch {
            val r = runner.fireDecayPass()
            _status.value = r.toMessage()
        } }

    fun fireCalendarCheck() = run { _status.value = "Firing calendar check…" }
        .also { viewModelScope.launch {
            val r = runner.fireCalendarCheck()
            _status.value = r.toMessage()
        } }

    fun clearStatus() { _status.value = null }

    private fun ProactiveRunner.RunResult.toMessage(): String = when (this) {
        is ProactiveRunner.RunResult.Ok -> "✅ $message"
        is ProactiveRunner.RunResult.Error -> "❌ $message"
    }
}

data class ProactiveHistoryUiState(
    val events: List<com.aura.proactive.ProactiveEventBus.Event> = emptyList(),
)
