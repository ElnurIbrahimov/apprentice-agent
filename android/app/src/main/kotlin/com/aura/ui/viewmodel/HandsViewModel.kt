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
    /**
     * Free-text search query. Applied client-side by the screen
     * (no Room round-trip) — case-insensitive substring match
     * against hand name and trigger phrase.
     */
    val searchQuery: String = "",
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
            _state.value = _state.value.copy(running = null, lastResult = resultMessage(result))
        }
    }

    /** Permission needed by the stopped step, used by the ActivityResult launcher. */
    fun pendingPermission(run: HandRun): String? {
        val hand = _state.value.hands.firstOrNull { it.id == run.handId } ?: return null
        val stepIndex = run.failedStep?.minus(1) ?: return null
        val step = repository.parseSteps(hand.steps).getOrNull(stepIndex) ?: return null
        return toolRegistry.get(step.tool)?.requiredPermissions?.firstOrNull()
    }

    /**
     * Resume exactly at the stopped step. Approval is scoped to that step's
     * tool name; edited automations and redacted runtime secrets fail closed.
     */
    fun resumeRun(run: HandRun) {
        _state.value = _state.value.copy(running = run.handName, lastResult = null)
        viewModelScope.launch {
            val result = runCatching {
                val hand = repository.getById(run.handId)
                    ?: return@runCatching ToolResult.Error("This hand no longer exists", "hand_missing")
                if (hand.updatedAt > run.startedAt) {
                    return@runCatching ToolResult.Error(
                        "This hand changed after the stopped run. Start a fresh run to review the new steps.",
                        "hand_changed_since_run",
                    )
                }
                val stepIndex = run.failedStep?.minus(1)
                    ?: return@runCatching ToolResult.Error("No stopped step was recorded", "resume_step_missing")
                val step = repository.parseSteps(hand.steps).getOrNull(stepIndex)
                    ?: return@runCatching ToolResult.Error("The stopped step no longer exists", "resume_step_missing")
                val variables = repository.parseVariables(run.variablesJson)
                if (variables.values.any { it == "[redacted]" }) {
                    return@runCatching ToolResult.Error(
                        "This run used a secret input that was not stored. Start a fresh run and enter it again.",
                        "resume_secret_required",
                    )
                }
                val approvedTools = if (run.status == com.aura.hands.HandRunStatus.NEEDS_APPROVAL.value) {
                    setOf(step.tool)
                } else {
                    emptySet()
                }
                repository.run(
                    hand = hand,
                    executor = toolExecutor,
                    ctx = ToolContext(
                        conversationId = "hand:${hand.id}:resume:${run.id}",
                        approvedRemoteCostTools = approvedTools,
                    ),
                    variables = variables,
                    trigger = HandRunTrigger.RESUME.value,
                    startStepIndex = stepIndex,
                )
            }.getOrElse { ToolResult.Error(it.message ?: "Hand resume failed", "hand_resume_error") }
            _state.value = _state.value.copy(running = null, lastResult = resultMessage(result))
        }
    }

    private fun resultMessage(result: ToolResult): String = when (result) {
        is ToolResult.Ok -> result.output
        is ToolResult.Error -> "Error: ${result.message}"
        is ToolResult.NeedsPermission -> "Permission needed: ${result.permission}"
        is ToolResult.NeedsApproval -> "Approval needed: ${result.rationale}"
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

    fun setSearchQuery(query: String) {
        _state.value = _state.value.copy(searchQuery = query)
    }
}
