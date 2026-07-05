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
) : AndroidViewModel(application) {

    private val _state = MutableStateFlow(ChatUiState())
    val state: StateFlow<ChatUiState> = _state.asStateFlow()

    private var runJob: Job? = null

    /** Track consecutive tool failures for adaptive MoA escalation. */
    private var consecutiveFailures: Int = 0
    private var lastUserMessage: String = ""

    /** Known user correction patterns that suggest the model is struggling. */
    private val correctionPatterns = listOf(
        Regex("""\b(?:no|wrong|incorrect|not right|nope|that's wrong|bad answer)\b""", RegexOption.IGNORE_CASE),
        Regex("""\b(?:try again|redo|retry|do it again|another way)\b""", RegexOption.IGNORE_CASE),
    )

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
            val all = providerRegistry.all().flatMap { p ->
                runCatching {
                    p.listModels()
                }.getOrDefault(emptyList()).map { "${p.prefix}:$it" }
            }
            val defaults = listOf(
                "ollama:deepseek-v4-pro:cloud",
                "ollama:kimi-k2.7-code:cloud",
                "anthropic:claude-sonnet-4-5",
                "ollama:minimax-m2.7:cloud",
                "ollama:qwen3.5:cloud",
            )
            val merged = (defaults + all).distinct()
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

    fun dismissPermission() {
        _state.update { it.copy(pendingPermission = null, permissionRationale = null) }
    }

    fun dismissError() {
        _state.update { it.copy(error = null, errorRetryable = false, errorTyped = null) }
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
        runJob = viewModelScope.launch {
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
        runJob?.cancel()
        runJob = null
        _state.update { it.copy(streaming = false) }
    }

    fun onUserMessage(text: String) {
        _state.update { it.copy(draft = text) }
        send()
    }

    fun send() {
        val current = _state.value
        val text = current.draft.trim()
        if (text.isEmpty() || current.streaming) return

        // Adaptive MoA escalation: if the user corrected the last response
        // or the model has been struggling, auto-enable Deep Mode.
        val userIsCorrecting = correctionPatterns.any { it.containsMatchIn(text) }
        val shouldEscalate = !current.deepModeEnabled && (
            userIsCorrecting || consecutiveFailures >= 3
        )
        if (shouldEscalate) {
            _state.update { it.copy(deepModeEnabled = true) }
        }
        lastUserMessage = text

        // If Deep Mode is enabled, use the MoA provider for this turn.
        val useMoa = _state.value.deepModeEnabled
        val model = if (useMoa) "moa:default" else current.activeModel

        _state.update { it.copy(
            conversation = it.conversation.addUser(text),
            draft = "",
            streaming = true,
            error = null,
            deepModeActive = useMoa,
        ) }
        val specialist = current.selectedSpecialist
        var responseBuffer = StringBuilder()

        saveConversation()  // Save user message immediately

        runJob = viewModelScope.launch {
            try {
                val conversation = _state.value.conversation
                loop.run(
                    conversation = conversation,
                    model = model,
                    specialist = specialist,
                    memoryEnabled = !_state.value.incognitoMode,
                ).collect { event ->
                    when (event) {
                        is AgentEvent.TextDelta -> {
                            responseBuffer.append(event.text)
                            _state.update { old ->
                                val turns = old.conversation.turns
                                val last = turns.lastOrNull()
                                val updatedConversation = if (last != null) {
                                    old.conversation.replaceLastTurn(
                                        last.copy(assistant = (last.assistant ?: "") + event.text)
                                    )
                                } else old.conversation
                                old.copy(conversation = updatedConversation)
                            }
                        }
                        is AgentEvent.ToolResult -> {
                            val citations = extractCitations(event.name, event.result)
                            if (citations.isNotEmpty()) {
                                _state.update { old ->
                                    old.copy(conversation = old.conversation.setCitations(citations))
                                }
                            }
                            if (event.needsPermission != null) {
                                _state.update { old ->
                                    old.copy(
                                        pendingPermission = event.needsPermission,
                                        permissionRationale = event.permissionRationale,
                                        pendingToolRetry = event.name to event.arguments,
                                    )
                                }
                            } else if (event.result.startsWith("Still needs permission")) {
                                // Fallback: parse permission back out of formatted tool result
                                val perm = event.result.substringAfter("Still needs permission: ").trim()
                                _state.update { old ->
                                    old.copy(
                                        pendingPermission = perm,
                                        permissionRationale = "Permission is still required for ${event.name}.",
                                        pendingToolRetry = event.name to event.arguments,
                                    )
                                }
                            }
                        }
                        is AgentEvent.ToolExecuting -> {
                            // not used in this collector
                        }
                        is AgentEvent.Error -> {
                            consecutiveFailures++
                            val typed = event.typedError
                            val display = typed?.formatUserMessage() ?: "${event.code}: ${event.message}"
                            setErrorWithAutoDismiss(display, retryable = event.retryable, typed = typed)
                        }
                        is AgentEvent.Result -> {
                            // Replace the conversation snapshot with the loop's final state
                            // which includes all tool calls, tool results, and assistant text.
                            _state.update { old ->
                                old.copy(conversation = event.conversation)
                            }
                        }
                        is AgentEvent.Done -> {
                            // Reset failure counter on successful completion.
                            consecutiveFailures = 0
                            if (_state.value.ttsEnabled && responseBuffer.isNotBlank()) {
                                textToSpeech.speak(
                                    text = responseBuffer.toString(),
                                    utteranceId = "turn-${System.currentTimeMillis()}",
                                    flush = true,
                                )
                            }
                            saveConversation()
                            // Check if KG learned new nodes
                            refreshKgNodeCount()
                            onFirstConversationComplete()
                            // Reset Deep Mode after a successful MoA turn.
                            _state.update { it.copy(deepModeEnabled = false, deepModeActive = false) }
                        }
                        else -> Unit
                    }
                }
            } catch (e: kotlinx.coroutines.CancellationException) {
                // user cancelled; no-op
            } catch (e: Exception) {
                _state.update { it.copy(error = e.message ?: "unknown error") }
            } finally {
                _state.update { it.copy(streaming = false, deepModeActive = false) }
            }
        }
    }

    /**
     * Capture/pick an image and ask the cloud vision model directly. The result is
     * inserted as an assistant turn so the user can continue the conversation.
     */
    fun onImageCaptured(bitmap: Bitmap, question: String = "Describe this image in detail") {
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
