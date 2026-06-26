package com.aura.ui.viewmodel

import android.app.Application; import androidx.lifecycle.AndroidViewModel; import androidx.lifecycle.viewModelScope
import com.aura.hands.Hand; import com.aura.hands.HandDao
import dagger.hilt.EntryPoint; import dagger.hilt.InstallIn; import dagger.hilt.android.EntryPointAccessors
import dagger.hilt.android.lifecycle.HiltViewModel; import dagger.hilt.components.SingletonComponent
import kotlinx.coroutines.flow.MutableStateFlow; import kotlinx.coroutines.flow.StateFlow; import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch; import javax.inject.Inject

data class HandsUiState(val hands: List<Hand> = emptyList(), val loading: Boolean = true, val running: String? = null, val lastResult: String? = null)

@EntryPoint @InstallIn(SingletonComponent::class) interface HandsEntry { fun handDao(): HandDao }
@HiltViewModel class HandsViewModel @Inject constructor(app: Application) : AndroidViewModel(app) {
    private val _state = MutableStateFlow(HandsUiState()); val state: StateFlow<HandsUiState> = _state.asStateFlow()
    private fun dao() = EntryPointAccessors.fromApplication(getApplication(), HandsEntry::class.java).handDao()
    init { load() }
    fun load() { viewModelScope.launch { _state.value = _state.value.copy(hands = dao().getAll(), loading = false) } }

    fun add(name: String, triggerPhrase: String, stepsJson: String) {
        viewModelScope.launch {
            dao().insert(Hand(
                id = java.util.UUID.randomUUID().toString(),
                name = name,
                triggerPhrase = triggerPhrase,
                steps = stepsJson,
            ))
            load()
        }
    }

    fun toggle(hand: Hand) {
        viewModelScope.launch {
            dao().update(hand.copy(enabled = !hand.enabled))
            load()
        }
    }

    fun delete(name: String) {
        viewModelScope.launch { dao().deleteByName(name); load() }
    }

    fun runHand(hand: Hand) {
        _state.value = _state.value.copy(running = hand.name)
        viewModelScope.launch {
            val entry = EntryPointAccessors.fromApplication(
                getApplication(),
                com.aura.ui.viewmodel.ToolExecutorEntryPoint::class.java,
            )
            val ctx = com.aura.agent.ToolContext(conversationId = "")
            val result = entry.toolExecutor().execute(
                "run_hand",
                "{\"name\":\"${hand.name}\"}",
                ctx,
            )
            val msg = when (result) {
                is com.aura.agent.ToolResult.Ok -> result.output
                is com.aura.agent.ToolResult.Error -> "Error: ${result.message}"
                is com.aura.agent.ToolResult.NeedsPermission -> "Permission needed: ${result.permission}"
                is com.aura.agent.ToolResult.NeedsApproval -> "Approval needed: ${result.rationale}"
            }
            _state.value = _state.value.copy(running = null, lastResult = msg)
        }
    }
}
