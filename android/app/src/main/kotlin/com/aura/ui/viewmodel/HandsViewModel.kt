package com.aura.ui.viewmodel

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.aura.agent.ToolContext
import com.aura.agent.ToolExecutor
import com.aura.agent.ToolRegistry
import com.aura.agent.ToolResult
import com.aura.hands.Hand
import com.aura.hands.HandDao
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import java.util.UUID
import javax.inject.Inject

data class HandsUiState(
    val hands: List<Hand> = emptyList(),
    val loading: Boolean = true,
    val running: String? = null,
    val lastResult: String? = null,
)

@HiltViewModel
class HandsViewModel @Inject constructor(
    app: Application,
    private val handDao: HandDao,
    private val toolExecutor: ToolExecutor,
    val toolRegistry: ToolRegistry,
) : AndroidViewModel(app) {

    private val _state = MutableStateFlow(HandsUiState())
    val state: StateFlow<HandsUiState> = _state.asStateFlow()

    init { load() }

    fun load() {
        viewModelScope.launch {
            _state.value = _state.value.copy(hands = handDao.getAll(), loading = false)
        }
    }

    fun add(name: String, triggerPhrase: String, stepsJson: String) {
        viewModelScope.launch {
            handDao.insert(Hand(
                id = UUID.randomUUID().toString(),
                name = name,
                triggerPhrase = triggerPhrase,
                steps = stepsJson,
            ))
            load()
        }
    }

    fun toggle(hand: Hand) {
        viewModelScope.launch {
            handDao.update(hand.copy(enabled = !hand.enabled))
            load()
        }
    }

    fun delete(name: String) {
        viewModelScope.launch {
            handDao.deleteByName(name)
            load()
        }
    }

    fun runHand(hand: Hand) {
        _state.value = _state.value.copy(running = hand.name)
        viewModelScope.launch {
            val ctx = ToolContext(conversationId = "")
            val result = toolExecutor.execute(
                "run_hand",
                """{"name":"${hand.name}"}""",
                ctx,
            )
            val msg = when (result) {
                is ToolResult.Ok -> result.output
                is ToolResult.Error -> "Error: ${result.message}"
                is ToolResult.NeedsPermission -> "Permission needed: ${result.permission}"
                is ToolResult.NeedsApproval -> "Approval needed: ${result.rationale}"
            }
            _state.value = _state.value.copy(running = null, lastResult = msg)
        }
    }
}
