package com.aura.ui.viewmodel

import android.app.Application
import android.graphics.Bitmap
import android.net.Uri
import android.util.Base64
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope

import com.aura.data.UserPreferences
import com.aura.agent.AgentEvent
import com.aura.agent.ConversationStore
import com.aura.agent.MemoryAugmentedAgenticLoop
import com.aura.agent.Reaction
import com.aura.agent.Specialist
import com.aura.agent.SpecialistRouter
import com.aura.agent.ToolCall
import com.aura.agent.ToolContext
import com.aura.agent.ToolExecutor
import com.aura.agent.ToolRegistry
import com.aura.agent.ToolResult
import com.aura.agent.toAuraError
import com.aura.core.error.AuraError
import com.aura.kg.KnowledgeGraphRepository
import com.aura.providers.ProviderKeys
import com.aura.providers.ProviderRegistry
import com.aura.providers.ModelCatalog
import com.aura.providers.ModelCatalogRepository
import com.aura.providers.ProviderStatus
import com.aura.tools.Citation
import com.aura.tools.MAX_TRANSCRIPTION_AUDIO_BYTES
import com.aura.voice.TextToSpeech
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import java.io.ByteArrayOutputStream
import java.util.UUID
import javax.inject.Inject

private val json = Json { ignoreUnknownKeys = true; isLenient = true }

private fun formatToolResult(result: ToolResult): String = when (result) {
    is ToolResult.Ok -> result.output
    is ToolResult.Error -> "Error: ${result.message}"
    is ToolResult.NeedsPermission -> "Still needs permission: ${result.permission}"
    is ToolResult.NeedsApproval -> "Approval needed: ${result.rationale}"
}

private fun extractCitations(toolName: String, result: String): List<Citation> {
    return when (toolName) {
        "deep_research" -> {
            runCatching {
                val obj = json.parseToJsonElement(result).jsonObject
                val arr = obj["citations"]?.jsonArray ?: return@runCatching emptyList<Citation>()
                arr.mapIndexed { idx, el ->
                    val map = el.jsonObject
                    Citation(
                        index = idx + 1,
                        title = map["title"]?.jsonPrimitive?.content ?: "Source",
                        url = map["url"]?.jsonPrimitive?.content ?: "",
                    )
                }
            }.getOrDefault(emptyList())
        }
        "brave_search", "tavily_search" -> {
            // Parse markdown lines: "- [title](url): snippet"
            val regex = """- \[([^\]]+)\]\(([^)]+)\):""".toRegex()
            regex.findAll(result).mapIndexed { idx, match ->
                Citation(index = idx + 1, title = match.groupValues[1], url = match.groupValues[2])
            }.toList()
        }
        else -> emptyList()
    }
}

sealed interface ModelSelectionState {
    data object Missing : ModelSelectionState
    data class Loading(val activeModel: String?, val models: List<String>) : ModelSelectionState
    data class Ready(
        val activeModel: String,
        val models: List<String>,
        val staleProviders: Set<String> = emptySet(),
    ) : ModelSelectionState
    data class Failed(
        val activeModel: String?,
        val models: List<String>,
        val message: String,
    ) : ModelSelectionState
}

internal fun resolveModelSelection(
    activeModel: String,
    catalog: ModelCatalog,
): ModelSelectionState {
    val active = activeModel.takeIf(String::isNotBlank) ?: return ModelSelectionState.Missing
    val models = catalog.allModels.map { it.id }.distinct().sorted()
    if (catalog.providers.values.any { it.status == ProviderStatus.Loading }) {
        return ModelSelectionState.Loading(active, models)
    }
    val staleProviders = catalog.providers.values
        .filter { it.status == ProviderStatus.Ready && it.errorMessage != null }
        .mapTo(mutableSetOf()) { it.providerPrefix }
    if (active in models) return ModelSelectionState.Ready(active, models, staleProviders)

    val failure = catalog.providers.values.firstOrNull {
        it.status !in setOf(
            ProviderStatus.NotConfigured,
            ProviderStatus.Loading,
            ProviderStatus.Ready,
        )
    }
    val message = when {
        models.isNotEmpty() -> "The selected model is no longer available. Choose another model."
        failure != null -> failure.errorMessage ?: failure.status.name
        else -> "No verified models are available. Save & Test a provider in Settings."
    }
    return ModelSelectionState.Failed(active, models, message)
}

data class ChatUiState(
    val conversation: com.aura.agent.Conversation = com.aura.agent.Conversation(),
    val conversationLoading: Boolean = false,
    val streaming: Boolean = false,
    val draft: String = "",
    val error: String? = null,
    val errorTyped: AuraError? = null,
    /** Model used by this chat; Settings supplies it until the user picks a session override. */
    val activeModel: String = "",
    /** Null means follow the global Settings default; non-null is chat-only. */
    val sessionModelOverride: String? = null,
    val availableModels: List<String> = emptyList(),
    /**
     * True while [refreshModels] is in flight. Drives a small
     * "Loading models…" indicator in the picker.
     */
    val modelsLoading: Boolean = false,
    /**
     * Last error from [refreshModels], if any. Picker shows
     * "Couldn't load models — tap to retry". Null on success.
     */
    val modelsError: String? = null,
    val modelSelection: ModelSelectionState = ModelSelectionState.Missing,
    val ttsEnabled: Boolean = true,
    val selectedSpecialist: Specialist? = null,
    val suggestedSpecialist: Specialist? = null,
    val pendingPermission: String? = null,
    val permissionRationale: String? = null,
    /** Tool name + args that the permission was requested for. Used to retry after grant. */
    val pendingToolRetry: Pair<String, String>? = null,
    val errorRetryable: Boolean = false,
    val kgNodeCount: Int = 0,
    /** True when user has toggled Deep Mode for the next turn. */
    val deepModeEnabled: Boolean = false,
    /** True when the current turn is running through MoA. */
    val deepModeActive: Boolean = false,
    /** Explicit model role used while Deep Mode is enabled. */
    val deepModeModel: String = "",
    /**
     * True when incognito mode is on. The agent still runs and
     * reads memory (so the user can ask questions about their
     * existing memories), but does not auto-store the user's
     * message or extract profile facts from the assistant's
     * response. Session-scoped — not persisted across app restarts.
     */
    val incognitoMode: Boolean = false,
    /**
     * Tool calls that have started (ToolCallStart) but whose
     * ToolResult has not yet been emitted by the agentic loop.
     * In-memory only; cleared when the next user message sends or
     * when the matching ToolResult lands. The completed entry moves
     * into `conversation.turns.last().toolTurns` once the loop
     * returns a result, so this list is just the "in-flight" view.
     */
    val inFlightToolCalls: List<InFlightToolCall> = emptyList(),
    /**
     * A bitmap that was just captured or picked but not yet sent
     * to vision. Cleared when the user picks a vision prompt chip
     * (Describe / Read text / Translate) or dismisses the row.
     * Lives in UI state — not persisted.
     */
    val pendingVisionBitmap: Bitmap? = null,
    /**
     * Non-blocking warning shown when a conversation save fails.
     * De-duplicated: only the first failure per session is shown so
     * repeated autosave failures don't spam the UI.
     */
    val saveWarning: String? = null,
)

/**
 * A tool call that the agentic loop has started but not yet
 * finished. The companion [com.aura.agent.ToolTurn] (persisted on
 * the turn) carries the completed form once the loop emits a
 * `ToolResult` event.
 */
data class InFlightToolCall(
    val id: String,
    val name: String,
    val args: String,
    val startedAtMs: Long = System.currentTimeMillis(),
)

@HiltViewModel
class ChatViewModel @Inject constructor(
    application: Application,
    private val loop: MemoryAugmentedAgenticLoop,
    private val providerKeys: ProviderKeys,
    private val providerRegistry: ProviderRegistry,
    private val toolRegistry: ToolRegistry,
    private val toolExecutor: ToolExecutor,
    private val textToSpeech: TextToSpeech,
    private val userPreferences: UserPreferences,
    private val memoryStore: com.aura.memory.MemoryStore,
    private val conversationStore: ConversationStore,
    private val knowledgeGraphRepository: KnowledgeGraphRepository,
    private val crashLogger: com.aura.core.error.CrashLogger,
    private val modelCatalogRepository: ModelCatalogRepository? = null,
) : AndroidViewModel(application) {

    private val _state = MutableStateFlow(ChatUiState())
    val state: StateFlow<ChatUiState> = _state.asStateFlow()
    @Volatile private var isolatedSessionRequested = false

    /**
     * Send pipeline controller. Owns the streaming runJob, the
     * consecutive-failures counter (for adaptive MoA escalation),
     * and the assistant-text buffer used for TTS on completion.
     * Extracted from this VM so the send logic lives in one place
     * and the VM is just a state holder + lifecycle owner.
     */
    private val sendController: ChatSendController by lazy {
        ChatSendController(
            application = application,
            state = _state,
            loop = loop,
            userPreferences = userPreferences,
            textToSpeech = textToSpeech,
            knowledgeGraphRepository = knowledgeGraphRepository,
            onSaveConversation = { saveConversation() },
            onKgNodeCountChanged = { refreshKgNodeCount() },
            onFirstConversationComplete = { onFirstConversationComplete() },
            extractCitations = ::extractCitations,
            setErrorWithAutoDismiss = ::setErrorWithAutoDismiss,
            generateTitle = ::generateTitle,
            onError = { msg -> _state.update { it.copy(error = msg) } },
        )
    }

    private fun saveConversation() {
        if (_state.value.incognitoMode) return
        viewModelScope.launch {
            runCatching {
                conversationStore.save(_state.value.conversation)
            }.onFailure { e ->
                // Surface the first save failure per session as a
                // non-blocking warning. Subsequent failures don't
                // re-set the warning so the UI doesn't spam.
                if (_state.value.saveWarning == null) {
                    _state.update {
                        it.copy(saveWarning = "Conversation could not be saved: ${e.message ?: e.javaClass.simpleName}")
                    }
                }
            }
        }
    }

    init {
        textToSpeech.initialize()
        viewModelScope.launch {
            val recent = conversationStore.mostRecent()
            if (isolatedSessionRequested) return@launch
            if (recent != null) {
                _state.update { it.copy(conversation = recent) }
            } else {
                // First conversation ever — set a welcoming system prompt
                _state.update { it.copy(conversation = it.conversation.copy(
                    systemPrompt = "You are Aura, a personal AI assistant. This is your first conversation with a new user. Introduce yourself warmly — tell them what you can do: remember things, set reminders, search the web, manage tasks, and work through delegated hands. Be concise and direct. Ask what they need help with.",
                    title = "Welcome",
                )) }
            }
        }
        viewModelScope.launch {
            userPreferences.defaultModel.collect { model ->
                _state.update { current ->
                    if (current.sessionModelOverride == null) {
                        current.copy(activeModel = model.orEmpty())
                    } else current
                }
                applyModelCatalog(modelCatalogRepository?.catalog?.value)
            }
        }
        modelCatalogRepository?.let { repository ->
            viewModelScope.launch {
                repository.catalog.collect(::applyModelCatalog)
            }
        }
        viewModelScope.launch {
            userPreferences.deepModeModel.collect { model ->
                _state.update { it.copy(deepModeModel = model.orEmpty()) }
            }
        }
        // Restore persisted TTS preference so the user's mute/unmute
        // choice survives app restarts. Defaults to true (on).
        viewModelScope.launch {
            userPreferences.ttsEnabled.collect { enabled ->
                _state.update { it.copy(ttsEnabled = enabled) }
            }
        }
        // Restore persisted incognito default. The session toggle still
        // works on top of this — it's just the starting value.
        viewModelScope.launch {
            userPreferences.incognitoDefault.collect { default ->
                _state.update { it.copy(incognitoMode = default) }
            }
        }
    }

    /**
     * Persist the onboarding marker exactly once. This callback runs after
     * every successful reply, so conversation-count timing is not a valid
     * deduplication boundary; MemoryStore owns the durable exact-content gate.
     */
    private suspend fun onFirstConversationComplete() {
        if (_state.value.incognitoMode) return
        runCatching {
            memoryStore.storeIfAbsent(
                content = "This user started using Aura. They went through the onboarding.",
                source = "system",
                category = "episode",
                importance = 0.8f,
            )
        }
    }

    override fun onCleared() {
        textToSpeech.shutdown()
        super.onCleared()
    }

    /**
     * Re-fetch the model list from every configured provider. Safe
     * to call repeatedly — concurrent calls are no-ops (we check
     * [ChatUiState.modelsLoading] first). Sets [ChatUiState.modelsLoading]
     * while in flight, surfaces the last error on failure so the
     * picker can show "tap to retry" instead of a silent empty list.
     *
     * This is the fix for the "I added an Ollama Cloud key and the
     * picker still shows nothing" bug. Triggered on:
     *   - init (when the chat first opens)
     *   - the picker opening (so a key added in Settings is picked up)
     *   - a manual retry tap from the picker
     */
    fun refreshModels() {
        val repository = modelCatalogRepository
        if (repository == null) {
            _state.update {
                it.copy(
                    modelSelection = ModelSelectionState.Failed(
                        activeModel = it.activeModel.takeIf(String::isNotBlank),
                        models = it.availableModels,
                        message = "Model catalog is unavailable.",
                    ),
                    modelsError = "Model catalog is unavailable.",
                )
            }
            return
        }
        val current = _state.value
        _state.update {
            it.copy(
                modelsLoading = true,
                modelsError = null,
                modelSelection = ModelSelectionState.Loading(
                    current.activeModel.takeIf(String::isNotBlank),
                    current.availableModels,
                ),
            )
        }
        repository.refresh(force = true)
    }

    private fun applyModelCatalog(catalog: ModelCatalog?) {
        val current = _state.value
        if (catalog == null) {
            val selection = if (current.activeModel.isBlank()) {
                ModelSelectionState.Missing
            } else {
                ModelSelectionState.Failed(
                    current.activeModel,
                    current.availableModels,
                    "Model catalog is unavailable.",
                )
            }
            _state.update { it.copy(modelSelection = selection) }
            return
        }

        val models = catalog.allModels.map { it.id }.distinct().sorted()
        val selection = resolveModelSelection(current.activeModel, catalog)
        _state.update {
            it.copy(
                availableModels = models,
                modelsLoading = selection is ModelSelectionState.Loading,
                modelsError = (selection as? ModelSelectionState.Failed)?.message,
                modelSelection = selection,
            )
        }
    }

    fun setDraft(text: String) {
        _state.update { old ->
            val suggested = if (text.isBlank()) null
                else SpecialistRouter.pickSpecialist(text)
            old.copy(draft = text, suggestedSpecialist = suggested)
        }
    }

    /** Pick a model for this chat only; global default belongs to Settings. */
    fun setModel(model: String) {
        _state.update {
            it.copy(
                activeModel = model,
                sessionModelOverride = model,
                modelSelection = ModelSelectionState.Ready(model, it.availableModels),
            )
        }
    }

    /** Explicitly promote the current chat model to the default for future chats. */
    fun makeActiveModelDefault() {
        val model = _state.value.activeModel
        if (model.isBlank()) return
        viewModelScope.launch {
            userPreferences.setDefaultModel(model)
            _state.update { it.copy(sessionModelOverride = null) }
        }
    }

    fun setSpecialist(specialist: Specialist?) {
        _state.update { old ->
            val newModel = specialist?.suggestedModel ?: old.activeModel
            old.copy(selectedSpecialist = specialist, activeModel = newModel)
        }
    }

    fun setTtsEnabled(enabled: Boolean) {
        _state.update { it.copy(ttsEnabled = enabled) }
        if (!enabled) textToSpeech.stop()
        viewModelScope.launch { userPreferences.setTtsEnabled(enabled) }
    }

    fun toggleTts() {
        setTtsEnabled(!_state.value.ttsEnabled)
    }

    fun toggleDeepMode() {
        _state.update { it.copy(deepModeEnabled = !it.deepModeEnabled) }
    }

    /**
     * Flip incognito mode. When on, the next send does not write
     * to memory or extract profile facts. Toggling it mid-conversation
     * takes effect on the next user message — the in-flight turn
     * (if any) keeps the value it was started with.
     */
    fun toggleIncognito() {
        _state.update { it.copy(incognitoMode = !it.incognitoMode) }
    }

    /**
     * Compute the most recent assistant text from the current
     * conversation. Returns an empty string if the last turn is a
     * user message (or the conversation is empty). Used by the
     * 'Copy last response' button in the chat header.
     */
    fun lastAssistantText(): String {
        val conv = _state.value.conversation
        // Walk turns in reverse to find the most recent assistant
        // text. We don't just take the last turn because the model
        // may have returned an empty assistant text + a tool call
        // — we want the last turn that actually had assistant
        // content.
        for (i in conv.turns.indices.reversed()) {
            val assistant = conv.turns[i].assistant
            if (!assistant.isNullOrBlank()) return assistant
        }
        return ""
    }

    fun loadConversation(id: String) {
        viewModelScope.launch {
            _state.update { it.copy(conversationLoading = true) }
            try {
                conversationStore.load(id)?.let { conv ->
                    _state.update { it.copy(conversation = conv) }
                }
            } finally {
                _state.update { it.copy(conversationLoading = false) }
            }
        }
    }

    /**
     * Fork the current conversation from a specific turn index.
     * Creates a new conversation with turns up to [fromTurnIndex],
     * loads it, and saves. The original conversation is untouched.
     */
    fun forkConversation(fromTurnIndex: Int) {
        val convId = _state.value.conversation.id
        viewModelScope.launch {
            val forkId = conversationStore.fork(convId, fromTurnIndex)
            if (forkId != null) {
                conversationStore.load(forkId)?.let { conv ->
                    _state.update { it.copy(conversation = conv) }
                }
            }
        }
    }

    /**
     * Start a fresh session owned by a compact surface (widget, share sheet,
     * future quick actions) without allowing the asynchronous recent-chat load
     * to overwrite it. Execution still uses the full send controller and agent loop.
     */
    fun startIsolatedSession(
        systemPrompt: String,
        model: String? = null,
        title: String = "Quick Ask",
    ) {
        isolatedSessionRequested = true
        cancel()
        _state.update { old ->
            val selectedModel = model?.takeIf(String::isNotBlank) ?: old.activeModel
            old.copy(
                conversation = com.aura.agent.Conversation(
                    systemPrompt = systemPrompt,
                    title = title,
                ),
                streaming = false,
                draft = "",
                error = null,
                errorRetryable = false,
                errorTyped = null,
                activeModel = selectedModel,
                sessionModelOverride = model?.takeIf(String::isNotBlank),
                modelSelection = if (selectedModel.isNotBlank()) {
                    ModelSelectionState.Ready(selectedModel, old.availableModels)
                } else {
                    ModelSelectionState.Missing
                },
                ttsEnabled = false,
                selectedSpecialist = null,
                suggestedSpecialist = null,
                inFlightToolCalls = emptyList(),
            )
        }
    }

    /**
     * Start a fresh conversation. Clears the current conversation,
     * resets the draft, and creates a new empty one with the default
     * system prompt. The old conversation is NOT deleted — it stays
     * in History. Cancels any in-flight streaming first.
     */
    fun newConversation() {
        cancel()
        _state.update {
            it.copy(
                conversation = com.aura.agent.Conversation(
                    systemPrompt = "You are Aura, a personal AI assistant. Be concise and direct. Ask what they need help with.",
                    title = "New conversation",
                ),
                draft = "",
                error = null,
                errorRetryable = false,
                errorTyped = null,
                deepModeEnabled = false,
                deepModeActive = false,
                selectedSpecialist = null,
                suggestedSpecialist = null,
                inFlightToolCalls = emptyList(),
            )
        }
    }

    /**
     * Delete the current conversation from the store and start a
     * fresh one. The deleted conversation is gone for good — no
     * undo. In incognito mode the conversation was never persisted
     * so we just reset the UI state.
     */
    fun deleteCurrentConversation() {
        cancel()
        val convId = _state.value.conversation.id
        viewModelScope.launch {
            if (!_state.value.incognitoMode) {
                runCatching { conversationStore.delete(convId) }
            }
            _state.update {
                it.copy(
                    conversation = com.aura.agent.Conversation(
                        systemPrompt = "You are Aura, a personal AI assistant. Be concise and direct. Ask what they need help with.",
                        title = "New conversation",
                    ),
                    draft = "",
                    error = null,
                    errorRetryable = false,
                    errorTyped = null,
                    deepModeEnabled = false,
                    deepModeActive = false,
                    selectedSpecialist = null,
                    suggestedSpecialist = null,
                    inFlightToolCalls = emptyList(),
                )
            }
        }
    }

    fun dismissPermission() {
        _state.update { it.copy(pendingPermission = null, permissionRationale = null) }
    }

    /**
     * Set or clear the user's thumbs-up/thumbs-down reaction on a
     * single assistant turn. Tapping the same reaction a second time
     * clears it (null → Up → null, or null → Down → null). Tapping the
     * opposite reaction switches it. No-op if [turnTimestamp] doesn't
     * match any current turn. The new conversation is persisted to
     * Room so the reaction survives reloads.
     */
    fun reactToTurn(turnTimestamp: Long, reaction: Reaction?) {
        val current = _state.value.conversation
        val index = current.turns.indexOfFirst { it.timestamp == turnTimestamp }
        if (index < 0) return
        val turn = current.turns[index]
        // Toggle: tapping the same reaction clears it
        val newReaction = if (reaction == turn.reaction) null else reaction
        val newTurns = current.turns.toMutableList().apply {
            this[index] = turn.copy(reaction = newReaction)
        }
        _state.update { it.copy(conversation = current.copy(turns = newTurns)) }
        saveConversation()
    }

    fun dismissError() {
        _state.update { it.copy(error = null, errorRetryable = false, errorTyped = null) }
    }

    fun dismissSaveWarning() {
        _state.update { it.copy(saveWarning = null) }
    }

    /**
     * Generate a short conversation title from the first user message.
     * Smarter than just first-6-words:
     * - strips conversational starters ("can you", "please", "i want to")
     * - uses up to the first question mark
     * - falls back to first 6 content words
     * - truncates to 50 chars
     * - capitalizes the first letter
     * No LLM call — deterministic and instant.
     */
    private fun generateTitle(text: String): String {
        val raw = text.trim()
        if (raw.isEmpty()) return "New conversation"

        // Cut at the first sentence terminator if it exists and is short.
        val firstSentence = raw.split(Regex("[.!?\\n]")).firstOrNull()?.trim().orEmpty()

        // Strip conversational starters so "Can you help me plan my day" →
        // "Plan my day" rather than "Can you help".
        val starterPatterns = listOf(
            "can you", "could you", "would you", "will you",
            "please", "hey", "hi ", "hello", "yo ",
            "i want to", "i need to", "i'd like to", "i would like to",
            "help me", "i have a question", "i was wondering",
        )
        val lowered = firstSentence.lowercase()
        var cleaned = firstSentence
        for (starter in starterPatterns) {
            if (lowered.startsWith(starter)) {
                cleaned = firstSentence.substring(starter.length).trimStart(' ', ',', '.')
                break
            }
        }

        // Take first 6 content words (filter very short ones).
        val words = cleaned
            .split(Regex("\\s+"))
            .filter { it.length > 1 || it.all(Char::isLetter) }
            .take(6)
        var title = words.joinToString(" ").trim().ifBlank { raw.take(50) }

        if (title.length > 50) {
            title = title.take(47) + "…"
        }
        if (title.isNotEmpty() && title[0].isLowerCase()) {
            title = title[0].uppercaseChar() + title.substring(1)
        }
        return title.ifBlank { "New conversation" }
    }

    private fun refreshKgNodeCount() {
        viewModelScope.launch {
            runCatching {
                val count = knowledgeGraphRepository.stats().nodeCount
                if (count > _state.value.kgNodeCount) {
                    _state.update { it.copy(kgNodeCount = count) }
                }
            }
        }
    }

    private fun setErrorWithAutoDismiss(error: String, retryable: Boolean = false, typed: AuraError? = null) {
        _state.update { it.copy(error = error, errorRetryable = retryable, errorTyped = typed) }
        crashLogger.log(
            code = typed?.code ?: "error",
            message = error,
        )
        // Only auto-dismiss non-retryable errors. Retryable errors
        // must stay visible until the user retries or sends a new
        // message — auto-dismissing them takes the retry button away.
        if (!retryable) {
            viewModelScope.launch {
                kotlinx.coroutines.delay(5_000L)
                if (_state.value.error == error) {
                    _state.update { it.copy(error = null, errorRetryable = false, errorTyped = null) }
                }
            }
        }
    }

    /** Retry the last user turn without duplicating it in history. */
    fun retryLast() {
        if (_state.value.streaming) return
        val retry = prepareConversationForRetry(_state.value.conversation) ?: return
        _state.update {
            it.copy(
                conversation = retry.conversation,
                draft = "",
                error = null,
                errorRetryable = false,
                errorTyped = null,
            )
        }
        sendController.runSend(viewModelScope, retryUserText = retry.userText)
    }

    fun retryAfterPermission(@Suppress("UNUSED_PARAMETER") permission: String) {
        // Re-execute the failed tool directly with the original args, no model round-trip.
        val (toolName, args) = _state.value.pendingToolRetry ?: ("" to "")
        _state.update { it.copy(
            pendingPermission = null,
            permissionRationale = null,
            pendingToolRetry = null,
            streaming = true,
        ) }
        if (toolName.isBlank()) {
            _state.update { it.copy(streaming = false, error = "Lost tool call context") }
            return
        }
        // Cancel any in-flight agent stream so this retry isn't
        // racing with the loop. The original code stored both
        // the send loop and the retry in the same runJob field;
        // now they're separate but cancel() still does the right
        // thing via the controller.
        sendController.cancel()
        viewModelScope.launch {
            try {
                val ctx = ToolContext(conversationId = _state.value.conversation.id)
                val result = toolExecutor.execute(toolName, args, ctx)
                val resultText = formatToolResult(result)
                // Add the result as a tool turn so the model sees it in the right format
                _state.update { old ->
                    val updated = old.conversation
                        .addToolCall("retry-$toolName", toolName, args)
                        .setToolResult("retry-$toolName", resultText)
                    old.copy(conversation = updated)
                }
                saveConversation()
            } catch (e: kotlinx.coroutines.CancellationException) { /* cancelled */ }
            catch (e: Exception) { _state.update { it.copy(error = e.message ?: "unknown error") } }
            finally { _state.update { it.copy(streaming = false) } }
        }
    }

    fun cancel() {
        sendController.cancel()
    }

    fun onUserMessage(text: String) {
        _state.update { it.copy(draft = text) }
        send()
    }

    fun send() {
        sendController.runSend(viewModelScope)
    }

    /**
     * Capture/pick an image and stage it for vision. The user is
     * shown a small row of 3 chips (Describe / Read text / Translate)
     * above the input bar and picks the prompt they want. The
     * bitmap is held in [ChatUiState.pendingVisionBitmap] until a
     * chip is tapped, at which point [runVisionPrompt] is called
     * with the picked prompt and the bitmap is cleared.
     *
     * If no chips are visible (the user can still see the photo
     * in their gallery / camera roll), the default behavior — a
     * direct vision call — is preserved.
     */
    fun onImageCaptured(bitmap: Bitmap, question: String = "Describe this image in detail") {
        // Stage the bitmap. The UI will show the quick-question
        // chips if they haven't been answered yet. If the user
        // dismisses the row (or if a chip fires immediately), the
        // bitmap is cleared by runVisionPrompt / dismissPendingVision.
        _state.update { it.copy(pendingVisionBitmap = bitmap) }
        // If the caller passed a non-default question, fire
        // immediately. This keeps the gallery / camera flow
        // backward-compatible — gallery "describe" still works.
        if (question != "Describe this image in detail") {
            runVisionPrompt(bitmap, question)
        }
    }

    /**
     * Send the staged bitmap to vision with the chosen prompt.
     * Clears the staged bitmap from state once the call starts.
     */
    fun runVisionPrompt(bitmap: Bitmap, question: String) {
        _state.update { it.copy(pendingVisionBitmap = null) }
        viewModelScope.launch(Dispatchers.IO) {
            val base64 = bitmap.toBase64Jpeg()
            val tool = toolRegistry.get("vision") ?: run {
                _state.update { it.copy(error = "vision tool not available") }
                return@launch
            }
            // Add the user's question as a real user turn so the
            // agentic loop's memory recall, profile extraction, and
            // KG extraction all see it. Previously this bypassed the
            // loop entirely — the vision query was invisible to future
            // memory recall.
            _state.update { old ->
                old.copy(
                    conversation = old.conversation.addUser(question),
                    streaming = true,
                )
            }
            val toolCallId = UUID.randomUUID().toString()
            val result = tool.execute(
                ToolCall(
                    id = toolCallId,
                    name = "vision",
                    arguments = mapOf("image_base64" to base64, "prompt" to question),
                ),
                ToolContext(
                    conversationId = _state.value.conversation.id,
                    memoryEnabled = !_state.value.incognitoMode,
                ),
            )
            val text = when (result) {
                is ToolResult.Ok -> result.output
                is ToolResult.Error -> "Vision error: ${result.message}"
                is ToolResult.NeedsPermission -> "Permission needed: ${result.permission}"
                is ToolResult.NeedsApproval -> "Approval needed: ${result.rationale}"
            }
            // Add the tool result as a tool turn and an assistant turn
            // so the conversation history is coherent. The agentic loop
            // post-processing (KG extraction, profile extraction, memory
            // write gate) runs via saveConversation + the next send().
            _state.update { old ->
                val conv = old.conversation
                    .attachCompletedToolTurn(toolCallId, "vision", "{}", text)
                    .addAssistant(text)
                old.copy(
                    conversation = conv,
                    streaming = false,
                )
            }
            saveConversation()
        }
    }

    /**
     * Clear a staged vision bitmap without sending it. The user
     * can hit a small X on the chip row to dismiss the staged
     * image (e.g. they captured the wrong thing).
     */
    fun dismissPendingVision() {
        _state.update { it.copy(pendingVisionBitmap = null) }
    }

    /**
     * Transcribe an audio file picked by the user and insert the text as a user
     * message so the agent can act on it.
     */
    fun onAudioPicked(uri: Uri) {
        viewModelScope.launch(Dispatchers.IO) {
            val resolver = getApplication<Application>().contentResolver
            val knownSize = runCatching {
                resolver.query(
                    uri,
                    arrayOf(android.provider.OpenableColumns.SIZE),
                    null,
                    null,
                    null,
                )?.use { cursor ->
                    if (cursor.moveToFirst() && !cursor.isNull(0)) cursor.getLong(0) else null
                }
            }.getOrNull()
            if (knownSize != null && knownSize > MAX_TRANSCRIPTION_AUDIO_BYTES) {
                _state.update { it.copy(error = "Audio is larger than the 25 MB limit.") }
                return@launch
            }

            val bytes = try {
                val stream = resolver.openInputStream(uri) ?: run {
                    _state.update { it.copy(error = "failed to open audio") }
                    return@launch
                }
                stream.use { readStreamWithinLimit(it, MAX_TRANSCRIPTION_AUDIO_BYTES) }
            } catch (e: Exception) {
                _state.update { it.copy(error = "failed to read audio: ${e.message}") }
                return@launch
            }
            if (bytes == null) {
                _state.update { it.copy(error = "Audio is larger than the 25 MB limit.") }
                return@launch
            }
            val base64 = Base64.encodeToString(bytes, Base64.NO_WRAP)
            val tool = toolRegistry.get("transcribe") ?: run {
                _state.update { it.copy(error = "transcribe tool not available") }
                return@launch
            }
            _state.update { it.copy(streaming = true) }
            val result = tool.execute(
                ToolCall(
                    id = UUID.randomUUID().toString(),
                    name = "transcribe",
                    arguments = mapOf("audio_base64" to base64, "language" to "en"),
                ),
                ToolContext(conversationId = _state.value.conversation.id),
            )
            val text = when (result) {
                is ToolResult.Ok -> result.output
                is ToolResult.Error -> "Transcription error: ${result.message}"
                is ToolResult.NeedsPermission -> "Permission needed: ${result.permission}"
                is ToolResult.NeedsApproval -> "Approval needed: ${result.rationale}"
            }
            _state.update { old ->
                old.copy(
                    conversation = old.conversation.addUser("🎤 $text"),
                    streaming = false,
                )
            }
            saveConversation()
        }
    }

    private fun Bitmap.toBase64Jpeg(quality: Int = 85): String {
        val stream = ByteArrayOutputStream()
        compress(Bitmap.CompressFormat.JPEG, quality, stream)
        return Base64.encodeToString(stream.toByteArray(), Base64.NO_WRAP)
    }
}
