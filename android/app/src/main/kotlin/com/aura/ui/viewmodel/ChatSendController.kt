package com.aura.ui.viewmodel

import android.app.Application
import com.aura.agent.AgentEvent
import com.aura.agent.MemoryAugmentedAgenticLoop
import com.aura.agent.Specialist
import com.aura.agent.ToolResult
import com.aura.core.error.AuraError
import com.aura.data.UserPreferences
import com.aura.kg.KnowledgeGraphRepository
import com.aura.tools.Citation
import com.aura.voice.TextToSpeech
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch

/**
 * Owns the "send a message" pipeline that used to live in
 * [ChatViewModel]. The VM exposes the user-facing methods (send,
 * cancel, retry, onUserMessage) but the body of [runSend] is the
 * actual control flow: text buffering, model selection (MoA
 * escalation), specialist prompt overrides, tool result handling,
 * TTS on completion, and conversation persistence.
 *
 * Why extracted: ChatViewModel had grown to 774 lines with
 * multiple controllers mixed together (state, send, lifecycle).
 * The send pipeline alone was ~150 lines. Pulling it into its
 * own class makes the VM a thin state holder, makes the pipeline
 * independently testable, and gives the send loop a single owner
 * of the runJob field (was previously mixed with VM lifecycle).
 *
 * Public surface mirrors the VM's send-side methods exactly so
 * callers don't have to know about the split.
 */
class ChatSendController(
    private val application: Application,
    private val state: MutableStateFlow<ChatUiState>,
    private val loop: MemoryAugmentedAgenticLoop,
    private val userPreferences: UserPreferences,
    private val textToSpeech: TextToSpeech,
    private val knowledgeGraphRepository: KnowledgeGraphRepository,
    private val onSaveConversation: () -> Unit,
    private val onKgNodeCountChanged: () -> Unit,
    private val onFirstConversationComplete: suspend () -> Unit,
    private val extractCitations: (String, String) -> List<Citation>,
    private val setErrorWithAutoDismiss: (String, Boolean, AuraError?) -> Unit,
    private val generateTitle: (String) -> String,
    private val onError: (String) -> Unit,
) {
    /** Active streaming coroutine, if any. */
    var runJob: Job? = null
        private set

    /** Last text the user sent. Used by retry-last. */
    var lastUserMessage: String = ""
        private set

    /** Buffer for the assistant's streamed text, used by TTS at end. */
    private var responseBuffer: StringBuilder = StringBuilder()

    /** Count of consecutive streaming failures; triggers MoA escalation at 3. */
    var consecutiveFailures: Int = 0

    /** User-correction regexes used to detect "try again" / "no, I meant" turns. */
    private val correctionPatterns: List<Regex> = listOf(
        Regex("\\b(no|wrong|incorrect|that's not|that isn't|try again|nope)\\b", RegexOption.IGNORE_CASE),
        Regex("\\b(do it again|redo|one more time|again please)\\b", RegexOption.IGNORE_CASE),
    )

    /**
     * Send the current draft through the agent. The VM is responsible
     * for the user-facing state updates (clearing the draft, marking
     * streaming, etc.) — this method assumes the VM has already
     * updated state with `addUser`, `streaming = true`, and a
     * non-blank draft before the previous turn was appended.
     */
    fun runSend(scope: CoroutineScope) {
        val current = state.value
        val text = current.draft.trim()
        if (text.isEmpty() || current.streaming) return

        // Adaptive MoA escalation: if the user corrected the last response
        // or the model has been struggling, auto-enable Deep Mode.
        val userIsCorrecting = correctionPatterns.any { it.containsMatchIn(text) }
        val shouldEscalate = current.deepModeModel.isNotBlank() && !current.deepModeEnabled && (
            userIsCorrecting || consecutiveFailures >= 3
        )
        if (shouldEscalate) {
            state.update { it.copy(deepModeEnabled = true) }
        }
        lastUserMessage = text

        // Deep Mode uses the explicit role selected by the user.
        val useMoa = state.value.deepModeEnabled
        val model = if (useMoa) current.deepModeModel else current.activeModel
        if (model.isBlank()) {
            state.update {
                it.copy(
                    error = if (useMoa) {
                        "Choose a Deep Mode model in Settings before sending."
                    } else {
                        "Choose and verify a chat model before sending."
                    },
                )
            }
            return
        }

        state.update { it.copy(
            conversation = it.conversation.addUser(text),
            draft = "",
            streaming = true,
            error = null,
            deepModeActive = useMoa,
        ) }

        // Auto-title: if this is the first user message and the title
        // is still a default placeholder, generate a short title from
        // the first 6 words. No LLM call — fast, deterministic, good
        // enough for History browsing. The title is part of the
        // conversation object so it persists on the next
        // saveConversation() call.
        val isDefaultTitle = state.value.conversation.title == "New conversation" ||
            state.value.conversation.title == "Welcome"
        if (isDefaultTitle &&
            state.value.conversation.turns.count { it.user != null } == 1
        ) {
            val title = generateTitle(text)
            state.update { it.copy(conversation = it.conversation.copy(title = title)) }
        }

        val specialist = current.selectedSpecialist
        responseBuffer = StringBuilder()

        onSaveConversation()  // Save user message immediately

        // Clear any stale in-flight tool calls from a previous turn.
        // The new turn starts fresh; we'll repopulate as the loop
        // emits ToolCallStart events.
        state.update { it.copy(inFlightToolCalls = emptyList()) }

        runJob = scope.launch {
            try {
                val conversation = state.value.conversation
                // Apply user-defined specialist prompt overrides
                val resolvedSpecialist = specialist?.let { s ->
                    val overridesJson = userPreferences.specialistOverrides.first()
                    val toolOverridesJson = userPreferences.specialistToolOverrides.first()
                    try {
                        val promptOverrides = kotlinx.serialization.json.Json
                            .decodeFromString<Map<String, String>>(overridesJson)
                        val toolOverrides = if (toolOverridesJson.isNotBlank() && toolOverridesJson != "{}") {
                            kotlinx.serialization.json.Json
                                .decodeFromString<Map<String, List<String>>>(toolOverridesJson)
                                .mapValues { it.value.toSet() }
                        } else emptyMap()
                        val customPrompt = promptOverrides[s.name]
                        val customTools = toolOverrides[s.name]
                        val withPrompt = if (customPrompt.isNullOrBlank()) s else s.copy(systemPrompt = customPrompt)
                        if (customTools != null && customTools.isNotEmpty()) withPrompt.copy(toolsAllowed = customTools) else withPrompt
                    } catch (e: Exception) { s }
                }
                loop.run(
                    conversation = conversation,
                    model = model,
                    specialist = resolvedSpecialist,
                    memoryEnabled = !state.value.incognitoMode,
                ).collect { event ->
                    when (event) {
                        is AgentEvent.TextDelta -> {
                            responseBuffer.append(event.text)
                            state.update { old ->
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
                        is AgentEvent.ToolCallStart -> {
                            // The loop announced a tool call. Push it
                            // onto the in-flight list so the chat UI
                            // can show a running badge. The matching
                            // ToolResult will remove it.
                            state.update { old ->
                                val inFlight = InFlightToolCall(
                                    id = event.id,
                                    name = event.name,
                                    args = "",  // args come via ToolCallDelta + ToolCallEnd
                                )
                                old.copy(
                                    inFlightToolCalls = old.inFlightToolCalls + inFlight,
                                )
                            }
                        }
                        is AgentEvent.ToolCallEnd -> {
                            // The args for the tool call are now
                            // complete. Update the in-flight entry
                            // with the full args so the UI can show
                            // them. ToolResult will close it out.
                            state.update { old ->
                                val updated = old.inFlightToolCalls.map { inFlight ->
                                    if (inFlight.id == event.id) inFlight.copy(args = event.arguments)
                                    else inFlight
                                }
                                old.copy(inFlightToolCalls = updated)
                            }
                        }
                        is AgentEvent.ToolResult -> {
                            // The tool finished. Remove the in-flight
                            // entry — the completed form is now on
                            // conversation.turns.last().toolTurns.
                            state.update { old ->
                                old.copy(
                                    inFlightToolCalls = old.inFlightToolCalls.filterNot { it.id == event.id },
                                )
                            }
                            val citations = extractCitations(event.name, event.result)
                            if (citations.isNotEmpty()) {
                                state.update { old ->
                                    old.copy(conversation = old.conversation.setCitations(citations))
                                }
                            }
                            if (event.needsPermission != null) {
                                state.update { old ->
                                    old.copy(
                                        pendingPermission = event.needsPermission,
                                        permissionRationale = event.permissionRationale,
                                        pendingToolRetry = event.name to event.arguments,
                                    )
                                }
                            } else if (event.result.startsWith("Still needs permission")) {
                                // Fallback: parse permission back out of formatted tool result
                                val perm = event.result.substringAfter("Still needs permission: ").trim()
                                state.update { old ->
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
                            setErrorWithAutoDismiss(display, event.retryable, typed)
                        }
                        is AgentEvent.Result -> {
                            // Replace the conversation snapshot with the loop's final state
                            // which includes all tool calls, tool results, and assistant text.
                            state.update { old ->
                                old.copy(conversation = event.conversation)
                            }
                        }
                        is AgentEvent.Done -> {
                            // Reset failure counter on successful completion.
                            consecutiveFailures = 0
                            if (state.value.ttsEnabled && responseBuffer.isNotBlank()) {
                                textToSpeech.speak(
                                    text = responseBuffer.toString(),
                                    utteranceId = "turn-${System.currentTimeMillis()}",
                                    flush = true,
                                )
                            }
                            onSaveConversation()
                            // Check if KG learned new nodes
                            onKgNodeCountChanged()
                            onFirstConversationComplete()
                            // Reset Deep Mode after a successful MoA turn.
                            // Clear in-flight tool calls — anything still
                            // running at Done is orphaned (the loop finished
                            // without emitting a matching ToolResult). The
                            // visible conversation turns already have the
                            // complete tool history.
                            state.update { it.copy(
                                deepModeEnabled = false,
                                deepModeActive = false,
                                inFlightToolCalls = emptyList(),
                            ) }
                        }
                        else -> Unit
                    }
                }
            } catch (e: CancellationException) {
                // user cancelled; no-op
            } catch (e: Exception) {
                onError(e.message ?: "unknown error")
            } finally {
                state.update { it.copy(streaming = false, deepModeActive = false) }
            }
        }
    }

    /**
     * Cancel the active streaming job, if any. Saves the partial
     * response rather than discarding it — the streaming text was
     * already appended to the conversation's last turn via
     * replaceLastTurn, so the partial assistant text is in
     * `state.value.conversation`. We just need to persist it.
     */
    fun cancel() {
        runJob?.cancel()
        runJob = null
        if (!state.value.incognitoMode && state.value.conversation.turns.isNotEmpty()) {
            onSaveConversation()
        }
        // Clear in-flight badges — the stream is being torn down.
        state.update { it.copy(streaming = false, inFlightToolCalls = emptyList()) }
    }
}
