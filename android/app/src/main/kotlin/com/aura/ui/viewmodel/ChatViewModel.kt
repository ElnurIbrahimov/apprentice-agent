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
import com.aura.providers.ProviderRegistry
import com.aura.tools.Citation
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

data class ChatUiState(
    val conversation: com.aura.agent.Conversation = com.aura.agent.Conversation(),
    val streaming: Boolean = false,
    val draft: String = "",
    val error: String? = null,
    val errorTyped: AuraError? = null,
    val activeModel: String = "ollama:deepseek-v4-pro:cloud",
    val availableModels: List<String> = emptyList(),
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
    private val providerRegistry: ProviderRegistry,
    private val toolRegistry: ToolRegistry,
    private val toolExecutor: ToolExecutor,
    private val textToSpeech: TextToSpeech,
    private val userPreferences: UserPreferences,
    private val memoryStore: com.aura.memory.MemoryStore,
    private val conversationStore: ConversationStore,
    private val knowledgeGraphRepository: KnowledgeGraphRepository,
    private val crashLogger: com.aura.core.error.CrashLogger,
) : AndroidViewModel(application) {

    private val _state = MutableStateFlow(ChatUiState())
    val state: StateFlow<ChatUiState> = _state.asStateFlow()

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
            }
        }
    }

    init {
        refreshModels()
        textToSpeech.initialize()
        viewModelScope.launch {
            val recent = conversationStore.mostRecent()
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
                _state.update { it.copy(activeModel = model) }
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
     * After the first conversation, create a memory that this user has
     * started using Aura, so future conversations can reference it.
     * Skipped in incognito mode so a private first conversation doesn't
     * leave a permanent "this user started using Aura" fact.
     */
    private suspend fun onFirstConversationComplete() {
        if (_state.value.incognitoMode) return
        val count = runCatching {
            conversationStore.recent(2).size
        }.getOrDefault(0)
        if (count <= 1) {
            runCatching {
                memoryStore.store(
                    content = "This user started using Aura. They went through the onboarding.",
                    source = "system",
                    category = "episode",
                    importance = 0.8f,
                )
            }
        }
    }

    override fun onCleared() {
        textToSpeech.shutdown()
        super.onCleared()
    }

    fun refreshModels() {
        viewModelScope.launch {
            // Only show models from configured providers (where an API
            // key is present). Showing models for unconfigured providers
            // leads to a 401 on first send — confusing UX.
            val configuredProviders = providerRegistry.configured()
            val all = configuredProviders.flatMap { p ->
                runCatching {
                    p.listModels()
                }.getOrDefault(emptyList()).map { "${p.prefix}:$it" }
            }
            // Keep MoA in the list if its aggregator is configured.
            val moaModels = providerRegistry.get("moa")?.let { moa ->
                if (moa.isConfigured()) {
                    runCatching { moa.listModels() }.getOrDefault(emptyList()).map { "moa:$it" }
                } else emptyList()
            } ?: emptyList()
            val merged = (all + moaModels).distinct()
            _state.update { it.copy(availableModels = merged) }
        }
    }

    fun setDraft(text: String) {
        _state.update { old ->
            val toolNames = toolRegistry.definitions().map { it.name }
            val suggested = if (text.isBlank()) null
                else SpecialistRouter.pickSpecialist(text, toolNames)
            old.copy(draft = text, suggestedSpecialist = suggested)
        }
    }

    fun setModel(model: String) {
        _state.update { it.copy(activeModel = model) }
        viewModelScope.launch { userPreferences.setDefaultModel(model) }
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
            conversationStore.load(id)?.let { conv ->
                _state.update { it.copy(conversation = conv) }
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

    fun dismissError() {
        _state.update { it.copy(error = null, errorRetryable = false, errorTyped = null) }
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
        // Persist the error so it survives the 5-second auto-dismiss.
        crashLogger.log(
            code = typed?.code ?: "error",
            message = error,
        )
        // Auto-dismiss after 5 seconds
        viewModelScope.launch {
            kotlinx.coroutines.delay(5_000L)
            if (_state.value.error == error) {
                _state.update { it.copy(error = null, errorRetryable = false, errorTyped = null) }
            }
        }
    }

    /** Retry the last user message by re-running the agent. */
    fun retryLast() {
        val conv = _state.value.conversation
        val lastUser = conv.turns.lastOrNull { it.user != null } ?: return
        if (lastUser.user.isNullOrBlank()) return
        _state.update { it.copy(error = null, errorRetryable = false) }
        send()
    }

    fun retryAfterPermission(permission: String) {
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
            _state.update { old ->
                old.copy(
                    conversation = old.conversation.addUser(question),
                    streaming = true,
                )
            }
            val result = tool.execute(
                ToolCall(
                    id = UUID.randomUUID().toString(),
                    name = "vision",
                    arguments = mapOf("image_base64" to base64, "prompt" to question),
                ),
                ToolContext(conversationId = _state.value.conversation.id),
            )
            val text = when (result) {
                is ToolResult.Ok -> result.output
                is ToolResult.Error -> "Vision error: ${result.message}"
                is ToolResult.NeedsPermission -> "Permission needed: ${result.permission}"
                is ToolResult.NeedsApproval -> "Approval needed: ${result.rationale}"
            }
            _state.update { old ->
                old.copy(
                    conversation = old.conversation.addAssistant(text),
                    streaming = false,
                )
            }
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
            val bytes = try {
                getApplication<Application>().contentResolver.openInputStream(uri)?.use { it.readBytes() }
                    ?: return@launch
            } catch (e: Exception) {
                _state.update { it.copy(error = "failed to read audio: ${e.message}") }
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
        }
    }

    private fun Bitmap.toBase64Jpeg(quality: Int = 85): String {
        val stream = ByteArrayOutputStream()
        compress(Bitmap.CompressFormat.JPEG, quality, stream)
        return Base64.encodeToString(stream.toByteArray(), Base64.NO_WRAP)
    }
}
