package com.aura.ui.viewmodel

import android.app.Application
import android.graphics.Bitmap
import android.net.Uri
import android.util.Base64
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.aura.agent.AgentEvent
import com.aura.agent.MemoryAugmentedAgenticLoop
import com.aura.agent.Specialist
import com.aura.agent.SpecialistRouter
import com.aura.agent.ToolCall
import com.aura.agent.ToolContext
import com.aura.agent.ToolRegistry
import com.aura.agent.ToolResult
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
    val activeModel: String = "ollama:deepseek-v3.2:cloud",
    val availableModels: List<String> = emptyList(),
    val ttsEnabled: Boolean = true,
    val selectedSpecialist: Specialist? = null,
    val suggestedSpecialist: Specialist? = null,
)

@HiltViewModel
class ChatViewModel @Inject constructor(
    application: Application,
    private val loop: MemoryAugmentedAgenticLoop,
    private val providerRegistry: ProviderRegistry,
    private val toolRegistry: ToolRegistry,
    private val textToSpeech: TextToSpeech,
) : AndroidViewModel(application) {

    private val _state = MutableStateFlow(ChatUiState())
    val state: StateFlow<ChatUiState> = _state.asStateFlow()

    private var runJob: Job? = null
    private var currentResponseBuffer = StringBuilder()

    init {
        refreshModels()
        textToSpeech.initialize()
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
                "ollama:deepseek-v3.2:cloud",
                "ollama:kimi-k2.6:cloud",
                "anthropic:claude-sonnet-4-5",
                "ollama:minimax-m2.7:cloud",
                "ollama:qwen3-coder:480b-cloud",
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
    }

    fun setSpecialist(specialist: Specialist?) {
        _state.update { it.copy(selectedSpecialist = specialist) }
    }

    fun setTtsEnabled(enabled: Boolean) {
        _state.update { it.copy(ttsEnabled = enabled) }
        if (!enabled) textToSpeech.stop()
    }

    fun toggleTts() {
        setTtsEnabled(!_state.value.ttsEnabled)
    }

    fun cancel() {
        runJob?.cancel()
        runJob = null
        _state.update { it.copy(streaming = false) }
    }

    fun send() {
        val current = _state.value
        val text = current.draft.trim()
        if (text.isEmpty() || current.streaming) return

        current.conversation.addUser(text)
        currentResponseBuffer = StringBuilder()
        _state.update { it.copy(draft = "", streaming = true, error = null) }

        runJob = viewModelScope.launch {
            val specialist = current.selectedSpecialist
            try {
                loop.run(current.conversation, model = current.activeModel, specialist = specialist).collect { event ->
                    when (event) {
                        is AgentEvent.TextDelta -> {
                            currentResponseBuffer.append(event.text)
                            current.conversation.turns.lastOrNull()?.let { last ->
                                val updated = last.copy(assistant = (last.assistant ?: "") + event.text)
                                current.conversation.turns[current.conversation.turns.lastIndex] = updated
                            }
                            _state.update { it.copy(conversation = current.conversation) }
                        }
                        is AgentEvent.ToolExecuting, is AgentEvent.ToolResult -> {
                            if (event is AgentEvent.ToolResult) {
                                val citations = extractCitations(event.name, event.result)
                                if (citations.isNotEmpty()) {
                                    current.conversation.setCitations(citations)
                                }
                            }
                            _state.update { it.copy(conversation = current.conversation) }
                        }
                        is AgentEvent.Error -> {
                            _state.update { it.copy(error = "${event.code}: ${event.message}") }
                        }
                        is AgentEvent.Done -> {
                            if (_state.value.ttsEnabled && currentResponseBuffer.isNotBlank()) {
                                textToSpeech.speak(
                                    text = currentResponseBuffer.toString(),
                                    utteranceId = "turn-${System.currentTimeMillis()}",
                                    flush = true,
                                )
                            }
                        }
                        else -> Unit
                    }
                }
            } catch (e: kotlinx.coroutines.CancellationException) {
                // user cancelled; no-op
            } catch (e: Exception) {
                _state.update { it.copy(error = e.message ?: "unknown error") }
            } finally {
                _state.update { it.copy(streaming = false) }
                currentResponseBuffer = StringBuilder()
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
            _state.update {
                it.conversation.addUser(question)
                it.copy(conversation = it.conversation, streaming = true)
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
            _state.value.conversation.addAssistant(text)
            _state.update { it.copy(streaming = false) }
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
            _state.value.conversation.addUser("🎤 $text")
            _state.update { it.copy(streaming = false) }
        }
    }

    private fun Bitmap.toBase64Jpeg(quality: Int = 85): String {
        val stream = ByteArrayOutputStream()
        compress(Bitmap.CompressFormat.JPEG, quality, stream)
        return Base64.encodeToString(stream.toByteArray(), Base64.NO_WRAP)
    }
}
