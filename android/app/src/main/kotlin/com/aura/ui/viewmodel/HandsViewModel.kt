package com.aura.ui.viewmodel

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.aura.agent.ToolContext
import com.aura.agent.ToolExecutor
import com.aura.agent.ToolRegistry
import com.aura.agent.ToolResult
import com.aura.hands.Hand
import com.aura.hands.HandCondition
import com.aura.hands.HandRepository
import com.aura.hands.HandRun
import com.aura.hands.HandRunTrigger
import com.aura.hands.HandScheduler
import dagger.hilt.android.lifecycle.HiltViewModel
import java.util.UUID
import javax.inject.Inject
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.catch
import kotlinx.coroutines.launch

data class HandDraft(
    val name: String,
    val triggerPhrase: String,
    val stepsJson: String,
    val variables: Map<String, String>,
    val conditions: List<HandCondition>,
    val scheduleType: String,
    val scheduleHour: Int,
    val scheduleMinute: Int,
    val scheduleDayOfWeek: Int,
)

data class HandsUiState(
    val hands: List<Hand> = emptyList(),
    val runs: List<HandRun> = emptyList(),
    val loading: Boolean = true,
    val running: String? = null,
    val lastResult: String? = null,
    val error: String? = null,
)

@HiltViewModel
class HandsViewModel @Inject constructor(
    private val repository: HandRepository,
    private val toolExecutor: ToolExecutor,
    val toolRegistry: ToolRegistry,
    private val scheduler: HandScheduler,
) : ViewModel() {

    private val _state = MutableStateFlow(HandsUiState())
    val state: StateFlow<HandsUiState> = _state.asStateFlow()

    init {
        load()
        viewModelScope.launch {
            repository.observeRecentRuns(100)
                .catch { error -> _state.value = _state.value.copy(error = error.message) }
                .collect { runs -> _state.value = _state.value.copy(runs = runs) }
        }
    }

    fun load() {
        viewModelScope.launch {
            runCatching { repository.getAll() }
                .onSuccess { hands -> _state.value = _state.value.copy(hands = hands, loading = false) }
                .onFailure { error ->
                    _state.value = _state.value.copy(loading = false, error = error.message ?: "Could not load hands")
                }
        }
    }

    fun save(existing: Hand?, draft: HandDraft) {
        if (draft.name.isBlank()) {
            _state.value = _state.value.copy(error = "Hand name is required")
            return
        }
        viewModelScope.launch {
            val now = System.currentTimeMillis()
            val hand = Hand(
                id = existing?.id ?: UUID.randomUUID().toString(),
                name = draft.name.trim(),
                triggerPhrase = draft.triggerPhrase.trim(),
                steps = draft.stepsJson,
                variables = repository.variablesToJson(draft.variables),
                conditions = repository.conditionsToJson(draft.conditions),
                scheduleType = draft.scheduleType,
                scheduleHour = draft.scheduleHour.coerceIn(0, 23),
                scheduleMinute = draft.scheduleMinute.coerceIn(0, 59),
                scheduleDayOfWeek = draft.scheduleDayOfWeek.coerceIn(1, 7),
                enabled = existing?.enabled ?: true,
                createdAt = existing?.createdAt ?: now,
                updatedAt = now,
            )
            runCatching {
                if (existing == null) repository.insert(hand) else repository.update(hand)
                scheduler.schedule(hand)
            }.onSuccess {
                _state.value = _state.value.copy(error = null)
                load()
            }.onFailure { error ->
                _state.value = _state.value.copy(error = error.message ?: "Could not save hand")
            }
        }
    }

    fun toggle(hand: Hand) {
        viewModelScope.launch {
            val updated = hand.copy(enabled = !hand.enabled, updatedAt = System.currentTimeMillis())
            runCatching {
                repository.update(updated)
                if (updated.enabled) scheduler.schedule(updated) else scheduler.cancel(updated.id)
            }.onSuccess { load() }
                .onFailure { error -> _state.value = _state.value.copy(error = error.message) }
        }
    }

    fun delete(hand: Hand) {
        viewModelScope.launch {
            runCatching {
                repository.deleteById(hand.id)
                scheduler.cancel(hand.id)
            }.onSuccess { load() }
                .onFailure { error -> _state.value = _state.value.copy(error = error.message) }
        }
    }

    fun runHand(hand: Hand, variables: Map<String, String> = emptyMap()) {
        _state.value = _state.value.copy(running = hand.name, lastResult = null)
        viewModelScope.launch {
            val result = runCatching {
                repository.run(
                    hand = hand,
                    executor = toolExecutor,
                    ctx = ToolContext(conversationId = "hand:${hand.id}"),
                    variables = variables,
                    trigger = HandRunTrigger.MANUAL.value,
                )
            }.getOrElse { ToolResult.Error(it.message ?: "Hand failed", "hand_runtime_error") }
            val message = when (result) {
                is ToolResult.Ok -> result.output
                is ToolResult.Error -> "Error: ${result.message}"
                is ToolResult.NeedsPermission -> "Permission needed: ${result.permission}"
                is ToolResult.NeedsApproval -> "Approval needed: ${result.rationale}"
            }
            _state.value = _state.value.copy(running = null, lastResult = message)
        }
    }

    fun clearHistory() {
        viewModelScope.launch {
            runCatching { repository.deleteRunHistory() }
                .onSuccess { _state.value = _state.value.copy(runs = emptyList()) }
                .onFailure { error -> _state.value = _state.value.copy(error = error.message) }
        }
    }

    fun clearResult() {
        _state.value = _state.value.copy(lastResult = null)
    }

    fun clearError() {
        _state.value = _state.value.copy(error = null)
    }
}
