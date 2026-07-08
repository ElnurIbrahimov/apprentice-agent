package com.aura.ui.screens

import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.imePadding
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.AddAPhoto
import androidx.compose.material.icons.filled.Add
import androidx.compose.material.icons.filled.DeleteOutline
import androidx.compose.material.icons.filled.GraphicEq
import androidx.compose.material.icons.filled.ArrowDropDown
import androidx.compose.material.icons.filled.AudioFile
import androidx.compose.material.icons.filled.ContentCopy
import androidx.compose.material.icons.filled.History
import androidx.compose.material.icons.filled.Mic
import androidx.compose.material.icons.filled.PhotoLibrary
import androidx.compose.material.icons.filled.Send
import androidx.compose.material.icons.filled.Stop
import androidx.compose.material.icons.filled.VolumeOff
import androidx.compose.material.icons.filled.VolumeUp
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.ModalBottomSheet
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.rememberModalBottomSheetState
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.hilt.navigation.compose.hiltViewModel
import com.aura.IncomingShareStore
import com.aura.agent.Specialist
import com.aura.ui.components.MarkdownText
import com.aura.ui.components.MoaThinkingIndicator
import com.aura.ui.components.ModelPickerSheet
import com.aura.ui.components.SpecialistChips
import com.aura.ui.viewmodel.ChatViewModel
import com.aura.ui.voice.VoiceOverlay
import com.aura.ui.voice.ContinuousVoiceOverlay
import com.aura.ui.voice.ContinuousVoiceViewModel
import dagger.hilt.EntryPoint
import dagger.hilt.InstallIn
import dagger.hilt.android.EntryPointAccessors
import dagger.hilt.components.SingletonComponent
import androidx.compose.ui.draw.clip
import androidx.compose.foundation.lazy.itemsIndexed
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.fadeIn
import androidx.compose.animation.slideInVertically
import androidx.compose.animation.scaleIn
import androidx.compose.animation.scaleOut
import androidx.compose.runtime.derivedStateOf
import androidx.compose.material.icons.filled.KeyboardArrowDown
import androidx.compose.runtime.rememberCoroutineScope
import kotlinx.coroutines.launch
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.animation.core.animateFloat

@EntryPoint
@InstallIn(SingletonComponent::class)
interface HiltEntryPoint {
    fun incomingShareStore(): IncomingShareStore
}

@Composable
fun ChatScreen(
    viewModel: ChatViewModel = hiltViewModel(),
    resumeConversationId: String? = null,
    morningBriefSummary: String? = null,
    initialDraft: String? = null,
    onNavigateHistory: () -> Unit = {},
) {
    LaunchedEffect(resumeConversationId) {
        if (resumeConversationId != null) viewModel.loadConversation(resumeConversationId)
    }
    LaunchedEffect(morningBriefSummary) {
        if (!morningBriefSummary.isNullOrBlank()) viewModel.onUserMessage(morningBriefSummary)
    }
    LaunchedEffect(initialDraft) {
        if (!initialDraft.isNullOrBlank()) viewModel.setDraft(initialDraft)
    }

    val state by viewModel.state.collectAsState()
    val listState = rememberLazyListState()
    val coroutineScope = rememberCoroutineScope()
    val context = LocalContext.current

    var showModelPicker by remember { mutableStateOf(false) }
    var showSources by remember { mutableStateOf(false) }
    var showVoiceOverlay by remember { mutableStateOf(false) }
    var showContinuousVoice by remember { mutableStateOf(false) }
    val continuousVoiceViewModel: ContinuousVoiceViewModel = hiltViewModel()
    var showStopStreamConfirm by remember { mutableStateOf(false) }

    // Intercept back press during streaming — the user gets a chance to
    // stop and save the partial response instead of navigating away
    // while the stream continues in the background.
    if (state.streaming) {
        androidx.activity.compose.BackHandler(enabled = true) {
            showStopStreamConfirm = true
        }
    }
    if (showStopStreamConfirm) {
        androidx.compose.material3.AlertDialog(
            onDismissRequest = { showStopStreamConfirm = false },
            title = { androidx.compose.material3.Text("Stop streaming?") },
            text = { androidx.compose.material3.Text("The response will be saved with what's been generated so far.") },
            confirmButton = {
                androidx.compose.material3.TextButton(onClick = {
                    viewModel.cancel()
                    showStopStreamConfirm = false
                }) { androidx.compose.material3.Text("Stop and save") }
            },
            dismissButton = {
                androidx.compose.material3.TextButton(onClick = { showStopStreamConfirm = false }) {
                    androidx.compose.material3.Text("Keep listening")
                }
            },
        )
    }
    var showAttachmentSheet by remember { mutableStateOf(false) }
    var showDeleteConfirm by remember { mutableStateOf(false) }

    val hasMicPermission = rememberMicPermission()
    val micPermissionLauncher = rememberLauncherForActivityResult(
        contract = androidx.activity.result.contract.ActivityResultContracts.RequestPermission()
    ) { granted -> if (granted) showVoiceOverlay = true }

    val cameraLauncher = rememberLauncherForActivityResult(
        contract = androidx.activity.result.contract.ActivityResultContracts.TakePicturePreview()
    ) { bitmap -> bitmap?.let { viewModel.onImageCaptured(it) } }

    val galleryLauncher = rememberLauncherForActivityResult(
        contract = androidx.activity.result.contract.ActivityResultContracts.PickVisualMedia()
    ) { uri ->
        uri?.let {
            val bmp = decodeBitmap(context, it)
            bmp?.let(viewModel::onImageCaptured)
        }
    }

    val audioLauncher = rememberLauncherForActivityResult(
        contract = androidx.activity.result.contract.ActivityResultContracts.GetContent()
    ) { uri -> uri?.let(viewModel::onAudioPicked) }

    val hasCameraPermission = rememberCameraPermission()
    val cameraPermissionLauncher = rememberLauncherForActivityResult(
        contract = androidx.activity.result.contract.ActivityResultContracts.RequestPermission()
    ) { granted ->
        if (granted) cameraLauncher.launch(null)
    }

    ConsumeIncomingShare(context, viewModel)

    // Auto-scroll on:
    // 1. New turn added (size change)
    // 2. Streaming token arrives on the last turn (assistant length change)
    // 3. First message after send
    val lastTurn = state.conversation.turns.lastOrNull()
    val assistantLen = lastTurn?.assistant?.length ?: 0
    LaunchedEffect(state.conversation.turns.size, assistantLen) {
        if (state.conversation.turns.isNotEmpty()) {
            val target = state.conversation.turns.size - 1
            // Only auto-scroll if user is near the bottom. If they've
            // scrolled up to read old messages, don't yank them back.
            val visible = listState.layoutInfo.visibleItemsInfo
            val lastVisible = visible.lastOrNull()?.index ?: 0
            if (lastVisible >= target - 1) {
                listState.animateScrollToItem(target)
            }
        }
    }

    // Haptic feedback — when streaming finishes, the phone gives a
    // short vibration so the user can feel the response arrived.
    // Uses a counter to avoid firing on every recomposition.
    val hapticView = androidx.compose.ui.platform.LocalView.current
    var lastStreamingSnapshot by androidx.compose.runtime.remember { androidx.compose.runtime.mutableStateOf(state.streaming) }
    androidx.compose.runtime.LaunchedEffect(state.streaming) {
        if (lastStreamingSnapshot && !state.streaming) {
            com.aura.ui.util.Haptics.receive(hapticView)
        }
        lastStreamingSnapshot = state.streaming
    }

    Box(
        modifier = Modifier
            .fillMaxSize()
            .imePadding()
            .background(MaterialTheme.colorScheme.background),
    ) {
    Column(
        modifier = Modifier.fillMaxSize(),
    ) {
        ChatHeader(
            activeModel = state.activeModel,
            conversationModel = state.conversation.model,
            ttsEnabled = state.ttsEnabled,
            deepModeEnabled = state.deepModeEnabled,
            deepModeActive = state.deepModeActive,
            incognitoMode = state.incognitoMode,
            onToggleTts = viewModel::toggleTts,
            onCopyLast = { copyToClipboard(context, viewModel.lastAssistantText()) },
            onHistory = onNavigateHistory,
            onNewConversation = viewModel::newConversation,
            onDeleteConversation = { showDeleteConfirm = true },
            onToggleDeepMode = viewModel::toggleDeepMode,
            onToggleIncognito = viewModel::toggleIncognito,
            onShowModelPicker = { showModelPicker = true },
            onVoiceMode = {
                if (hasMicPermission) showContinuousVoice = true
                else micPermissionLauncher.launch(android.Manifest.permission.RECORD_AUDIO)
            },
        )

        if (state.deepModeActive) {
            MoaThinkingIndicator(
                modifier = Modifier.padding(horizontal = 16.dp, vertical = 6.dp),
            )
        }

        if (state.incognitoMode) {
            IncognitoBanner()
        }

        ChatMessageList(
            state = state,
            listState = listState,
            onShowSources = { showSources = true },
            modifier = Modifier.weight(1f),
        )

        state.error?.let { err ->
            ErrorBanner(
                error = err,
                retryable = state.errorRetryable,
                typedError = state.errorTyped,
                onRetry = viewModel::retryLast,
                onSwitchModel = { showModelPicker = true },
                onDismiss = viewModel::dismissError,
            )
        }

        if (state.draft.isNotBlank() || state.selectedSpecialist != null) {
            SpecialistChips(
                selected = state.selectedSpecialist,
                suggested = state.suggestedSpecialist,
                onSelect = viewModel::setSpecialist,
                modifier = Modifier.padding(horizontal = 12.dp, vertical = 4.dp),
            )
        }

        ChatInputBar(
            hapticView = hapticView,
            draft = state.draft,
            streaming = state.streaming,
            onDraftChange = viewModel::setDraft,
            onSend = viewModel::send,
            onCancel = viewModel::cancel,
            onMicClick = {
                if (hasMicPermission) showVoiceOverlay = true
                else micPermissionLauncher.launch(android.Manifest.permission.RECORD_AUDIO)
            },
            onCameraClick = {
                if (hasCameraPermission) cameraLauncher.launch(null)
                else cameraPermissionLauncher.launch(android.Manifest.permission.CAMERA)
            },
            onGalleryClick = { galleryLauncher.launch(androidx.activity.result.PickVisualMediaRequest(androidx.activity.result.contract.ActivityResultContracts.PickVisualMedia.ImageOnly)) },
            onAudioClick = { audioLauncher.launch("audio/*") },
            showAttachmentSheet = showAttachmentSheet,
            onAttachmentSheetChange = { showAttachmentSheet = it },
        )
    }

    if (showModelPicker) {
        ModelPickerSheet(
            currentModel = state.activeModel,
            models = state.availableModels,
            onPick = viewModel::setModel,
            onDismiss = { showModelPicker = false },
        )
    }

    val sources = state.conversation.turns.lastOrNull()?.citations ?: emptyList()
    if (showSources && sources.isNotEmpty()) {
        SourcesSheet(
            citations = sources,
            onDismiss = { showSources = false },
        )
    }

    if (showVoiceOverlay && hasMicPermission) {
        VoiceOverlay(
            onTranscript = { transcript ->
                viewModel.setDraft(transcript)
                viewModel.send()
            },
            onDismiss = { showVoiceOverlay = false },
        )
    } else if (showVoiceOverlay) {
        LaunchedEffect(Unit) { showVoiceOverlay = false }
    }

    // Continuous voice mode — hands-free conversation loop
    if (showContinuousVoice && hasMicPermission) {
        val cvState by continuousVoiceViewModel.state.collectAsState()
        ContinuousVoiceOverlay(
            state = cvState,
            onStop = {
                continuousVoiceViewModel.stopLoop()
                showContinuousVoice = false
            },
        )
        LaunchedEffect(Unit) {
            continuousVoiceViewModel.startLoop(
                onSend = { text ->
                    viewModel.setDraft(text)
                    viewModel.send()
                },
                onStreamingDone = { !viewModel.state.value.streaming },
            )
        }
        // Update last response when streaming completes
        LaunchedEffect(viewModel.state.value.streaming) {
            if (!viewModel.state.value.streaming && viewModel.state.value.conversation.turns.isNotEmpty()) {
                continuousVoiceViewModel.setLastResponse(viewModel.lastAssistantText())
            }
        }
    } else if (showContinuousVoice && !hasMicPermission) {
        LaunchedEffect(Unit) { showContinuousVoice = false }
        micPermissionLauncher.launch(android.Manifest.permission.RECORD_AUDIO)
    }

    PermissionDialog(
        permission = state.pendingPermission,
        rationale = state.permissionRationale,
        onGrant = viewModel::retryAfterPermission,
        onDismiss = viewModel::dismissPermission,
    )

    if (showDeleteConfirm) {
        AlertDialog(
            onDismissRequest = { showDeleteConfirm = false },
            title = { Text("Delete conversation?") },
            text = { Text("This conversation will be permanently deleted. This cannot be undone.") },
            confirmButton = {
                TextButton(onClick = {
                    showDeleteConfirm = false
                    viewModel.deleteCurrentConversation()
                }) {
                    Text("Delete", color = MaterialTheme.colorScheme.error)
                }
            },
            dismissButton = {
                TextButton(onClick = { showDeleteConfirm = false }) {
                    Text("Cancel")
                }
            },
        )
    }

    // Scroll-to-bottom FAB — appears when the user has scrolled
    // away from the bottom of the conversation. Tapping snaps
    // back to the latest message with a smooth animation.
    val isNearBottom by remember {
        derivedStateOf {
            val total = listState.layoutInfo.totalItemsCount
            val last = listState.layoutInfo.visibleItemsInfo.lastOrNull()?.index ?: 0
            total == 0 || last >= total - 2
        }
    }
    AnimatedVisibility(
        visible = !isNearBottom && state.conversation.turns.isNotEmpty(),
        enter = androidx.compose.animation.fadeIn() + androidx.compose.animation.scaleIn(initialScale = 0.8f),
        exit = androidx.compose.animation.fadeOut() + androidx.compose.animation.scaleOut(targetScale = 0.8f),
        modifier = Modifier
            .align(androidx.compose.ui.Alignment.BottomEnd)
            .padding(end = 16.dp, bottom = 88.dp),
    ) {
        Surface(
            onClick = {
                coroutineScope.launch {
                    if (state.conversation.turns.isNotEmpty()) {
                        listState.animateScrollToItem(state.conversation.turns.size - 1)
                    }
                }
            },
            color = MaterialTheme.colorScheme.primary,
            shape = androidx.compose.foundation.shape.CircleShape,
            tonalElevation = 4.dp,
            shadowElevation = 6.dp,
            modifier = Modifier.size(48.dp),
        ) {
            Box(contentAlignment = androidx.compose.ui.Alignment.Center) {
                Icon(
                    imageVector = androidx.compose.material.icons.Icons.Filled.KeyboardArrowDown,
                    contentDescription = "Scroll to bottom",
                    tint = MaterialTheme.colorScheme.onPrimary,
                    modifier = Modifier.size(24.dp),
                )
            }
        }
    }
    }
}

@Composable
private fun ChatHeader(
    activeModel: String,
    conversationModel: String?,
    ttsEnabled: Boolean,
    deepModeEnabled: Boolean,
    deepModeActive: Boolean,
    incognitoMode: Boolean,
    onToggleTts: () -> Unit,
    onCopyLast: () -> Unit,
    onHistory: () -> Unit,
    onNewConversation: () -> Unit,
    onDeleteConversation: () -> Unit,
    onToggleDeepMode: () -> Unit,
    onToggleIncognito: () -> Unit,
    onShowModelPicker: () -> Unit,
    onVoiceMode: () -> Unit = {},
) {
    val displayModel = conversationModel ?: activeModel
    val modelMismatch = conversationModel != null && conversationModel != activeModel

    Surface(
        color = MaterialTheme.colorScheme.surface,
        tonalElevation = 2.dp,
        modifier = Modifier.fillMaxWidth(),
    ) {
        Row(
            modifier = Modifier
                .clickable { onShowModelPicker() }
                .padding(horizontal = 16.dp, vertical = 12.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Column(modifier = Modifier.weight(1f)) {
                Text(
                    text = "Aura",
                    style = MaterialTheme.typography.titleLarge,
                    fontWeight = FontWeight.Bold,
                    color = MaterialTheme.colorScheme.onSurface,
                )
                Text(
                    text = com.aura.ui.util.modelDisplayName(displayModel),
                    style = MaterialTheme.typography.labelMedium,
                    color = if (modelMismatch) MaterialTheme.colorScheme.tertiary
                    else MaterialTheme.colorScheme.onSurface.copy(alpha = 0.5f),
                )
                if (modelMismatch && conversationModel != null) {
                    Text(
                        text = "was ${com.aura.ui.util.modelDisplayName(conversationModel)}",
                        style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.4f),
                    )
                }
            }
            IconButton(onClick = onToggleTts) {
                Icon(
                    imageVector = if (ttsEnabled) Icons.Filled.VolumeUp else Icons.Filled.VolumeOff,
                    contentDescription = if (ttsEnabled) "TTS on, tap to mute" else "TTS off, tap to enable",
                    tint = if (ttsEnabled) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.onSurface.copy(alpha = 0.4f),
                )
            }
            IconButton(onClick = onCopyLast) {
                Icon(
                    imageVector = Icons.Filled.ContentCopy,
                    contentDescription = "Copy last response",
                    tint = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.6f),
                )
            }
            IconButton(onClick = onNewConversation) {
                Icon(
                    imageVector = Icons.Filled.Add,
                    contentDescription = "New conversation",
                    tint = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.6f),
                )
            }
            IconButton(onClick = onDeleteConversation) {
                Icon(
                    imageVector = Icons.Filled.DeleteOutline,
                    contentDescription = "Delete conversation",
                    tint = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.6f),
                )
            }
            IconButton(onClick = onVoiceMode) {
                Icon(
                    imageVector = Icons.Filled.GraphicEq,
                    contentDescription = "Continuous voice mode",
                    tint = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.6f),
                )
            }
            IconButton(onClick = onHistory) {
                Icon(Icons.Filled.History, contentDescription = "History", tint = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.6f))
            }
            FilterChip(
                selected = deepModeEnabled,
                label = if (deepModeActive) "🪩 Thinking..." else "🚀 Deep",
                onClick = onToggleDeepMode,
            )
            FilterChip(
                selected = incognitoMode,
                label = if (incognitoMode) "🕵️ Incognito" else "👤",
                onClick = onToggleIncognito,
            )
            Icon(
                imageVector = Icons.Filled.ArrowDropDown,
                contentDescription = "Change model",
                tint = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.6f),
            )
        }
    }
}

@Composable
private fun FilterChip(selected: Boolean, label: String, onClick: () -> Unit) {
    androidx.compose.material3.FilterChip(
        selected = selected,
        onClick = onClick,
        label = { Text(text = label, style = MaterialTheme.typography.labelSmall) },
        colors = androidx.compose.material3.FilterChipDefaults.filterChipColors(
            selectedContainerColor = if (label.contains("Deep")) {
                MaterialTheme.colorScheme.primaryContainer.copy(alpha = 0.8f)
            } else {
                MaterialTheme.colorScheme.tertiaryContainer.copy(alpha = 0.8f)
            },
        ),
        modifier = Modifier.padding(end = 4.dp),
    )
}

@Composable
private fun IncognitoBanner() {
    Surface(
        color = MaterialTheme.colorScheme.tertiaryContainer.copy(alpha = 0.4f),
        shape = RoundedCornerShape(8.dp),
        modifier = Modifier
            .fillMaxWidth()
            .padding(horizontal = 16.dp, vertical = 4.dp),
    ) {
        Text(
            text = "🕵️ Incognito — nothing from this conversation is saved. " +
                "No memory, no profile facts, no knowledge-graph nodes, " +
                "and no conversation history. Read-only tools still work.",
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onTertiaryContainer,
            modifier = Modifier.padding(horizontal = 12.dp, vertical = 8.dp),
        )
    }
}

@Composable
private fun ChatMessageList(
    state: com.aura.ui.viewmodel.ChatUiState,
    listState: androidx.compose.foundation.lazy.LazyListState,
    onShowSources: () -> Unit,
    modifier: Modifier = Modifier,
) {
    LazyColumn(
        state = listState,
        modifier = modifier
            .fillMaxWidth(),
        contentPadding = PaddingValues(horizontal = 16.dp, vertical = 12.dp),
        verticalArrangement = Arrangement.spacedBy(10.dp),
    ) {
        itemsIndexed(state.conversation.turns) { index, turn ->
            // Subtle entry animation — each new turn fades + slides in.
            // We track "has this turn been shown" by remembering the
            // index it first appeared at. The first 4 turns appear
            // immediately; subsequent turns animate in.
            val alreadyShown = remember(index, state.conversation.turns.size) {
                index < state.conversation.turns.size - 1
            }
            val visible = remember { mutableStateOf(alreadyShown) }
            LaunchedEffect(turn, state.streaming) {
                if (!visible.value) {
                    kotlinx.coroutines.delay(20L)
                    visible.value = true
                }
            }
            AnimatedVisibility(
                visible = visible.value,
                enter = fadeIn() + slideInVertically { it / 4 },
            ) {
                Column {
                    turn.user?.let { MessageBubble(text = it, isUser = true, timestamp = turn.timestamp) }
                    turn.assistant?.let {
                        val isLast = turn === state.conversation.turns.lastOrNull()
                        val isStreaming = state.streaming && isLast
                        MessageBubble(
                            text = it,
                            isUser = false,
                            citations = turn.citations,
                            onShowSources = onShowSources,
                            isStreaming = isStreaming,
                            timestamp = turn.timestamp,
                        )
                        // Memory recall chip: shown below the assistant
                        // bubble when the agentic loop actually performed
                        // recall. Hidden for incognito turns and for turns
                        // where the loop errored before the recall step.
                        // Use a local val because `turn.recall` is in a
                        // different module (aura-core) and Kotlin won't
                        // smart-cast cross-module public properties.
                        val recall = turn.recall
                        if (recall != null) {
                            com.aura.ui.components.MemoryRecallChip(
                                recall = recall,
                            )
                        }
                    }
                    for (toolTurn in turn.toolTurns) {
                        if (toolTurn.result.isNotEmpty()) {
                            // A tool call with a result is in the "done"
                            // state. The result may be a plain string OR
                            // a "Tool errored: ..." payload — surface the
                            // distinction so the user sees failures.
                            val state = if (toolTurn.result.startsWith("Tool errored:")) {
                                com.aura.ui.components.ToolCallState.Failed(
                                    name = toolTurn.name,
                                    args = toolTurn.args,
                                    error = toolTurn.result.removePrefix("Tool errored:").trim(),
                                )
                            } else {
                                com.aura.ui.components.ToolCallState.Done(
                                    name = toolTurn.name,
                                    args = toolTurn.args,
                                    result = toolTurn.result,
                                )
                            }
                            com.aura.ui.components.ToolCallBadge(state = state)
                        }
                    }
                    // In-flight tool calls: tools the agent announced but
                    // whose result hasn't returned yet. Only render these
                    // for the LAST turn (the one currently being built
                    // by the streaming loop) and only while the loop is
                    // actively running.
                    val isCurrentTurn = turn === state.conversation.turns.lastOrNull()
                    if (isCurrentTurn && state.streaming) {
                        for (inFlight in state.inFlightToolCalls) {
                            com.aura.ui.components.ToolCallBadge(
                                state = com.aura.ui.components.ToolCallState.Running(inFlight),
                            )
                        }
                    }
                }
            }
        }
        if (state.streaming && state.conversation.turns.lastOrNull()?.assistant.isNullOrBlank()) {
            item { TypingIndicator() }
        }
    }
}

@Composable
private fun ErrorBanner(
    error: String,
    retryable: Boolean,
    typedError: com.aura.core.error.AuraError?,
    onRetry: () -> Unit,
    onSwitchModel: () -> Unit,
    onDismiss: () -> Unit,
) {
    val display = typedError?.formatUserMessage() ?: error
    Surface(
        color = MaterialTheme.colorScheme.errorContainer,
        modifier = Modifier.fillMaxWidth().padding(horizontal = 16.dp, vertical = 6.dp),
        shape = RoundedCornerShape(8.dp),
    ) {
        Row(Modifier.padding(start = 12.dp, end = 4.dp, top = 8.dp, bottom = 8.dp), verticalAlignment = Alignment.CenterVertically) {
            Text(
                text = display,
                modifier = Modifier.weight(1f),
                color = MaterialTheme.colorScheme.onErrorContainer,
                style = MaterialTheme.typography.bodySmall,
            )
            if (retryable || typedError?.retryable == true) {
                TextButton(onClick = onRetry) {
                    Text("Retry", color = MaterialTheme.colorScheme.onErrorContainer)
                }
                TextButton(onClick = onSwitchModel) {
                    Text("Switch model", color = MaterialTheme.colorScheme.onErrorContainer)
                }
            }
            TextButton(onClick = onDismiss) {
                Text("✕", color = MaterialTheme.colorScheme.onErrorContainer)
            }
        }
    }
}

@Composable
private fun MessageBubble(
    text: String,
    isUser: Boolean,
    citations: List<com.aura.tools.Citation> = emptyList(),
    onShowSources: () -> Unit = {},
    isStreaming: Boolean = false,
    timestamp: Long = 0L,
) {
    val context = androidx.compose.ui.platform.LocalContext.current
    val bubbleColor = if (isUser) MaterialTheme.colorScheme.primary
        else MaterialTheme.colorScheme.surfaceVariant
    val textColor = if (isUser) MaterialTheme.colorScheme.onPrimary
        else MaterialTheme.colorScheme.onSurfaceVariant
    val alignment = if (isUser) Alignment.End else Alignment.Start
    var copied by androidx.compose.runtime.remember { androidx.compose.runtime.mutableStateOf(false) }
    var showFullTime by androidx.compose.runtime.remember { androidx.compose.runtime.mutableStateOf(false) }
    val timeText = if (timestamp > 0) {
        if (showFullTime) com.aura.ui.util.formatClockTime(timestamp)
        else com.aura.ui.util.formatRelativeTime(timestamp)
    } else ""
    Column(
        modifier = Modifier.fillMaxWidth().padding(vertical = 2.dp),
        horizontalAlignment = alignment,
    ) {
        Surface(
            color = bubbleColor,
            shape = RoundedCornerShape(
                topStart = 20.dp,
                topEnd = 20.dp,
                bottomStart = if (isUser) 20.dp else 4.dp,
                bottomEnd = if (isUser) 4.dp else 20.dp,
            ),
            modifier = Modifier.widthIn(max = 340.dp),
        ) {
            Column(modifier = Modifier.padding(horizontal = 16.dp, vertical = 12.dp)) {
                if (isUser) {
                    Text(
                        text = text.ifBlank { "…" },
                        color = textColor,
                        style = MaterialTheme.typography.bodyLarge,
                    )
                } else {
                    com.aura.ui.components.StreamingText(
                        text = text.ifBlank { "…" },
                        isStreaming = isStreaming,
                        style = MaterialTheme.typography.bodyLarge.copy(color = textColor),
                    )
                }
                if (!isUser && citations.isNotEmpty()) {
                    Spacer(modifier = Modifier.height(8.dp))
                    Surface(
                        color = MaterialTheme.colorScheme.primary.copy(alpha = 0.12f),
                        shape = RoundedCornerShape(8.dp),
                        modifier = Modifier.clickable { onShowSources() },
                    ) {
                        Text(
                            text = "${citations.size} sources",
                            modifier = Modifier.padding(horizontal = 10.dp, vertical = 4.dp),
                            style = MaterialTheme.typography.labelSmall,
                            color = MaterialTheme.colorScheme.primary,
                        )
                    }
                }
            }
        }
        // Action row under the bubble — copy button for assistant
        // messages. Only shown when not streaming.
        if (!isUser && !isStreaming && text.isNotBlank()) {
            Row(
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(4.dp),
                modifier = Modifier.padding(top = 4.dp, start = 4.dp),
            ) {
                Surface(
                    onClick = {
                        val clipboard = context.getSystemService(android.content.Context.CLIPBOARD_SERVICE) as android.content.ClipboardManager
                        clipboard.setPrimaryClip(android.content.ClipData.newPlainText("Aura", text))
                        copied = true
                    },
                    color = androidx.compose.ui.graphics.Color.Transparent,
                    shape = RoundedCornerShape(8.dp),
                    modifier = Modifier.size(32.dp),
                ) {
                    Row(
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.Center,
                        modifier = Modifier.fillMaxSize(),
                    ) {
                        Icon(
                            imageVector = androidx.compose.material.icons.Icons.Filled.ContentCopy,
                            contentDescription = "Copy",
                            tint = MaterialTheme.colorScheme.onBackground.copy(alpha = 0.5f),
                            modifier = Modifier.size(16.dp),
                        )
                    }
                }
                androidx.compose.runtime.LaunchedEffect(copied) {
                    if (copied) {
                        kotlinx.coroutines.delay(1500)
                        copied = false
                    }
                }
                if (copied) {
                    Text(
                        text = "Copied",
                        style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.onBackground.copy(alpha = 0.5f),
                    )
                }
                if (timestamp > 0 && !isStreaming) {
                    // Tap the timestamp to toggle between relative
                    // ("5m") and full clock time ("3:42 PM").
                    Surface(
                        onClick = { showFullTime = !showFullTime },
                        color = androidx.compose.ui.graphics.Color.Transparent,
                        shape = RoundedCornerShape(6.dp),
                    ) {
                        Text(
                            text = timeText,
                            style = MaterialTheme.typography.labelSmall,
                            color = MaterialTheme.colorScheme.onBackground.copy(alpha = 0.4f),
                            modifier = Modifier.padding(horizontal = 4.dp, vertical = 2.dp),
                        )
                    }
                }
            }
        }
    }
}

@Composable
private fun TypingIndicator() {
    // Three animated dots in a bubble — like iMessage / WhatsApp.
    // Each dot pulses with a 200ms stagger so the eye sees a wave.
    Row(
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(4.dp),
        modifier = Modifier
            .padding(start = 4.dp, top = 4.dp, bottom = 4.dp)
            .background(
                color = MaterialTheme.colorScheme.surfaceVariant,
                shape = RoundedCornerShape(16.dp, 16.dp, 16.dp, 4.dp),
            )
            .padding(horizontal = 14.dp, vertical = 10.dp),
    ) {
        for (i in 0 until 3) {
            val transition = androidx.compose.animation.core.rememberInfiniteTransition(label = "dot-$i")
            val offset by transition.animateFloat(
                initialValue = 0f,
                targetValue = -6f,
                animationSpec = androidx.compose.animation.core.infiniteRepeatable(
                    animation = androidx.compose.animation.core.tween(500),
                    repeatMode = androidx.compose.animation.core.RepeatMode.Reverse,
                    initialStartOffset = androidx.compose.animation.core.StartOffset(i * 150),
                ),
                label = "y-$i",
            )
            val alpha by transition.animateFloat(
                initialValue = 0.4f,
                targetValue = 1f,
                animationSpec = androidx.compose.animation.core.infiniteRepeatable(
                    animation = androidx.compose.animation.core.tween(500),
                    repeatMode = androidx.compose.animation.core.RepeatMode.Reverse,
                    initialStartOffset = androidx.compose.animation.core.StartOffset(i * 150),
                ),
                label = "alpha-$i",
            )
            Box(
                modifier = Modifier
                    .graphicsLayer { translationY = offset }
                    .size(7.dp)
                    .background(
                        color = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = alpha),
                        shape = androidx.compose.foundation.shape.CircleShape,
                    ),
            )
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun ChatInputBar(
    hapticView: android.view.View,
    draft: String,
    streaming: Boolean,
    onDraftChange: (String) -> Unit,
    onSend: () -> Unit,
    onCancel: () -> Unit,
    onMicClick: () -> Unit,
    onCameraClick: () -> Unit,
    onGalleryClick: () -> Unit,
    onAudioClick: () -> Unit,
    showAttachmentSheet: Boolean,
    onAttachmentSheetChange: (Boolean) -> Unit,
) {
    val sheetState = rememberModalBottomSheetState(skipPartiallyExpanded = true)
    Surface(
        color = MaterialTheme.colorScheme.surface,
        tonalElevation = 3.dp,
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .navigationBarsPadding()
                .padding(horizontal = 12.dp, vertical = 10.dp),
            verticalAlignment = Alignment.Bottom,
        ) {
            IconButton(
                onClick = { onAttachmentSheetChange(true) },
                modifier = Modifier.size(44.dp),
            ) {
                Icon(
                    Icons.Filled.AddAPhoto,
                    contentDescription = "Attach",
                    tint = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
            OutlinedTextField(
                value = draft,
                onValueChange = onDraftChange,
                placeholder = { Text("Ask Aura…", style = MaterialTheme.typography.bodyLarge) },
                modifier = Modifier.weight(1f).padding(horizontal = 6.dp),
                maxLines = 5,
                shape = RoundedCornerShape(24.dp),
                keyboardOptions = androidx.compose.foundation.text.KeyboardOptions(
                    imeAction = androidx.compose.ui.text.input.ImeAction.Send
                ),
                keyboardActions = androidx.compose.foundation.text.KeyboardActions(onSend = { onSend() }),
                colors = androidx.compose.material3.TextFieldDefaults.colors(
                    focusedContainerColor = MaterialTheme.colorScheme.surfaceVariant,
                    unfocusedContainerColor = MaterialTheme.colorScheme.surfaceVariant,
                    focusedIndicatorColor = androidx.compose.ui.graphics.Color.Transparent,
                    unfocusedIndicatorColor = androidx.compose.ui.graphics.Color.Transparent,
                    disabledIndicatorColor = androidx.compose.ui.graphics.Color.Transparent,
                ),
                textStyle = MaterialTheme.typography.bodyLarge,
            )
            if (streaming) {
                IconButton(
                    onClick = onCancel,
                    modifier = Modifier
                        .size(44.dp)
                        .clip(RoundedCornerShape(22.dp))
                        .background(MaterialTheme.colorScheme.error),
                ) {
                    Icon(
                        Icons.Filled.Stop,
                        contentDescription = "Stop",
                        tint = MaterialTheme.colorScheme.onError,
                    )
                }
            } else {
                IconButton(
                    onClick = onMicClick,
                    modifier = Modifier.size(44.dp),
                ) {
                    Icon(
                        Icons.Filled.Mic,
                        contentDescription = "Voice input",
                        tint = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
                IconButton(
                    onClick = {
                        com.aura.ui.util.Haptics.send(hapticView)
                        onSend()
                    },
                    enabled = draft.isNotBlank(),
                    modifier = Modifier
                        .size(44.dp)
                        .clip(RoundedCornerShape(22.dp))
                        .background(
                            if (draft.isNotBlank()) MaterialTheme.colorScheme.primary
                            else MaterialTheme.colorScheme.surfaceVariant
                        ),
                ) {
                    Icon(
                        Icons.Filled.Send,
                        contentDescription = "Send",
                        tint = if (draft.isNotBlank()) MaterialTheme.colorScheme.onPrimary
                               else MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.4f),
                    )
                }
            }
        }
    }

    if (showAttachmentSheet) {
        ModalBottomSheet(
            onDismissRequest = { onAttachmentSheetChange(false) },
            sheetState = sheetState,
        ) {
            Column(
                modifier = Modifier.padding(20.dp),
                verticalArrangement = Arrangement.spacedBy(12.dp),
            ) {
                Text(
                    text = "Attach",
                    style = MaterialTheme.typography.titleMedium,
                    fontWeight = FontWeight.SemiBold,
                )
                AttachmentOption(
                    icon = Icons.Filled.PhotoLibrary,
                    label = "Gallery",
                    onClick = {
                        onAttachmentSheetChange(false)
                        onGalleryClick()
                    }
                )
                AttachmentOption(
                    icon = Icons.Filled.AddAPhoto,
                    label = "Camera",
                    onClick = {
                        onAttachmentSheetChange(false)
                        onCameraClick()
                    }
                )
                AttachmentOption(
                    icon = Icons.Filled.AudioFile,
                    label = "Audio",
                    onClick = {
                        onAttachmentSheetChange(false)
                        onAudioClick()
                    }
                )
            }
        }
    }
}

@Composable
private fun AttachmentOption(
    icon: androidx.compose.ui.graphics.vector.ImageVector,
    label: String,
    onClick: () -> Unit,
) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .clickable { onClick() }
            .padding(vertical = 12.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Icon(icon, contentDescription = null, tint = MaterialTheme.colorScheme.primary)
        Spacer(modifier = Modifier.width(16.dp))
        Text(text = label, style = MaterialTheme.typography.bodyLarge)
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun SourcesSheet(citations: List<com.aura.tools.Citation>, onDismiss: () -> Unit) {
    val sheetState = rememberModalBottomSheetState(skipPartiallyExpanded = true)
    ModalBottomSheet(
        onDismissRequest = onDismiss,
        sheetState = sheetState,
    ) {
        Column(
            modifier = Modifier.padding(20.dp),
        ) {
            Text(
                text = "Sources",
                style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.SemiBold,
            )
            Spacer(modifier = Modifier.height(12.dp))
            LazyColumn {
                items(citations) { citation ->
                    Surface(
                        color = MaterialTheme.colorScheme.surfaceVariant,
                        shape = RoundedCornerShape(8.dp),
                        modifier = Modifier.fillMaxWidth().padding(vertical = 4.dp),
                    ) {
                        Column(modifier = Modifier.padding(12.dp)) {
                            Text(
                                text = citation.title ?: citation.url ?: "Source",
                                style = MaterialTheme.typography.bodyMedium,
                                fontWeight = FontWeight.SemiBold,
                            )
                            citation.url?.let { url ->
                                Text(
                                    text = url,
                                    style = MaterialTheme.typography.bodySmall,
                                    color = MaterialTheme.colorScheme.primary,
                                )
                            }
                        }
                    }
                }
            }
        }
    }
}

@Composable
private fun ConsumeIncomingShare(context: android.content.Context, viewModel: ChatViewModel) {
    LaunchedEffect(Unit) {
        val store = EntryPointAccessors.fromApplication<HiltEntryPoint>(context.applicationContext as android.app.Application)
            .incomingShareStore()
        // Text share → set as draft
        store.consume()?.let(viewModel::setDraft)
        // Image share → decode Bitmap and route through onImageCaptured
        // so the vision tool analyzes it instead of dumping base64
        // text into the chat input.
        store.consumeImageUri()?.let { uri ->
            val bitmap = decodeSharedImage(context, uri)
            bitmap?.let { viewModel.onImageCaptured(it) }
        }
    }
}

@Composable
private fun rememberMicPermission(): Boolean {
    val context = LocalContext.current
    return remember {
        androidx.core.content.ContextCompat.checkSelfPermission(
            context,
            android.Manifest.permission.RECORD_AUDIO,
        ) == android.content.pm.PackageManager.PERMISSION_GRANTED
    }
}

@Composable
private fun rememberCameraPermission(): Boolean {
    val context = LocalContext.current
    return remember {
        androidx.core.content.ContextCompat.checkSelfPermission(
            context,
            android.Manifest.permission.CAMERA,
        ) == android.content.pm.PackageManager.PERMISSION_GRANTED
    }
}

private fun decodeSharedImage(context: android.content.Context, uri: android.net.Uri): android.graphics.Bitmap? {
    return if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.P) {
        runCatching {
            android.graphics.ImageDecoder.decodeBitmap(
                android.graphics.ImageDecoder.createSource(context.contentResolver, uri),
            ) { decoder, _, _ ->
                // Cap at 1024px to avoid OOM on large photos
                decoder.setTargetSize(1024, 1024)
            }
        }.getOrNull()
    } else {
        // API 26–27 fallback: decode through the content resolver and downsample.
        runCatching {
            @Suppress("DEPRECATION")
            android.graphics.BitmapFactory.decodeStream(
                context.contentResolver.openInputStream(uri),
                null,
                android.graphics.BitmapFactory.Options().apply {
                    inJustDecodeBounds = true
                },
            )
            android.graphics.BitmapFactory.Options().apply {
                val target = 1024
                var dim = 1.coerceAtLeast(outWidth.coerceAtLeast(outHeight))
                while (dim > target * 2) { dim /= 2; inSampleSize *= 2 }
                inJustDecodeBounds = false
            }.let { opts ->
                @Suppress("DEPRECATION")
                android.graphics.BitmapFactory.decodeStream(
                    context.contentResolver.openInputStream(uri),
                    null,
                    opts,
                )
            }
        }.getOrNull()
    }
}

private fun decodeBitmap(context: android.content.Context, uri: android.net.Uri): android.graphics.Bitmap? {
    return if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.P) {
        runCatching {
            android.graphics.ImageDecoder.decodeBitmap(
                android.graphics.ImageDecoder.createSource(context.contentResolver, uri)
            )
        }.getOrNull()
    } else {
        runCatching {
            @Suppress("DEPRECATION")
            android.provider.MediaStore.Images.Media.getBitmap(context.contentResolver, uri)
        }.getOrNull()
    }
}

private fun copyToClipboard(context: android.content.Context, text: String) {
    if (text.isBlank()) return
    val clip = context.getSystemService(android.content.Context.CLIPBOARD_SERVICE) as android.content.ClipboardManager
    clip.setPrimaryClip(android.content.ClipData.newPlainText("Aura response", text))
}

@Composable
private fun PermissionDialog(
    permission: String?,
    rationale: String?,
    onGrant: (String) -> Unit,
    onDismiss: () -> Unit,
) {
    if (permission == null) return
    val context = androidx.compose.ui.platform.LocalContext.current
    val permLauncher = rememberLauncherForActivityResult(
        contract = androidx.activity.result.contract.ActivityResultContracts.RequestPermission()
    ) { granted ->
        if (granted) onGrant(permission) else onDismiss()
    }
    // Some permissions can become "permanently denied" if the user
    // selected "Don't ask again" or pressed deny twice in a row.
    // The system prompt then silently no-ops and grants=false. The
    // only path forward is the system app-info page where the user
    // can re-enable the permission manually. Detect this and offer
    // the open-settings button.
    val activity = context as? android.app.Activity
    val packageName = context.packageName
    val shouldShowRationale = activity?.shouldShowRequestPermissionRationale(permission) ?: false
    val isPermanentlyDenied = !shouldShowRationale &&
        androidx.core.content.ContextCompat.checkSelfPermission(
            context, permission,
        ) != android.content.pm.PackageManager.PERMISSION_GRANTED
    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text("Permission needed") },
        text = {
            Text(
                buildString {
                    append(rationale ?: "Aura needs access to continue.")
                    append("\n\nPermission: $permission")
                    if (isPermanentlyDenied) {
                        append(
                            "\n\nThe system won't show the prompt again. " +
                                "Open Settings to enable it manually.",
                        )
                    }
                }
            )
        },
        confirmButton = {
            if (isPermanentlyDenied) {
                // "Don't ask again" was selected (or Android considers
                // the request denied enough times). The runtime
                // permission API is locked out — the user has to go
                // through the system Settings app.
                TextButton(onClick = {
                    val intent = android.content.Intent(
                        android.provider.Settings.ACTION_APPLICATION_DETAILS_SETTINGS,
                    ).apply {
                        data = android.net.Uri.fromParts("package", packageName, null)
                        flags = android.content.Intent.FLAG_ACTIVITY_NEW_TASK
                    }
                    context.startActivity(intent)
                    onDismiss()
                }) {
                    Text("Open Settings")
                }
            } else {
                TextButton(onClick = { permLauncher.launch(permission) }) {
                    Text("Grant")
                }
            }
        },
        dismissButton = {
            TextButton(onClick = onDismiss) {
                Text("Deny")
            }
        },
    )
}
