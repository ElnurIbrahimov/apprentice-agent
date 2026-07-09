package com.aura.ui.screens

import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.text.BasicTextField
import androidx.compose.foundation.text.KeyboardActions
import androidx.compose.foundation.text.KeyboardOptions
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
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.animation.core.animateDpAsState
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.animation.core.spring
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.Send
import androidx.compose.material.icons.filled.AddAPhoto
import androidx.compose.material.icons.filled.Add
import androidx.compose.material.icons.filled.DeleteOutline
import androidx.compose.material.icons.filled.GraphicEq
import androidx.compose.material.icons.filled.ArrowDownward
import androidx.compose.material.icons.filled.ArrowDropDown
import androidx.compose.material.icons.filled.AudioFile
import androidx.compose.material.icons.filled.ContentCopy
import androidx.compose.material.icons.filled.History
import androidx.compose.material.icons.filled.Mic
import androidx.compose.material.icons.filled.PhotoLibrary
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
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.SolidColor
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.hilt.navigation.compose.hiltViewModel
import com.aura.IncomingShareStore
import com.aura.agent.Specialist
import com.aura.ui.components.MarkdownText
import com.aura.ui.components.MoaThinkingIndicator
import com.aura.ui.components.ModelPickerSheet
import com.aura.ui.theme.AuraTokens
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
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
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
    // Show the "jump to bottom" pill when the user has scrolled up
    // by more than 5 messages. The web has this as a small floating
    // button bottom-right; on Android a small pill above the input
    // bar is more discoverable.
    val showJumpToBottom by remember(listState) {
        derivedStateOf {
            val total = listState.layoutInfo.totalItemsCount
            val visible = listState.firstVisibleItemIndex
            total > 0 && visible > 5
        }
    }
    val coroutineScope = rememberCoroutineScope()
    val context = LocalContext.current

    var showModelPicker by remember { mutableStateOf(false) }
    var showSources by remember { mutableStateOf(false) }
    var showVoiceOverlay by remember { mutableStateOf(false) }
    var showHoldToTalk by remember { mutableStateOf(false) }
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
            val bmp = decodeSharedImage(context, it)
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
            onHistory = onNavigateHistory,
            onNewConversation = viewModel::newConversation,
            onDeleteConversation = { showDeleteConfirm = true },
            onToggleDeepMode = viewModel::toggleDeepMode,
            onToggleIncognito = viewModel::toggleIncognito,
            onShowModelPicker = { showModelPicker = true },
        )

        if (state.deepModeActive) {
            MoaThinkingIndicator(
                modifier = Modifier.padding(horizontal = 16.dp, vertical = 6.dp),
            )
        }

        if (state.incognitoMode) {
            IncognitoBanner()
        }

        // Empty state: logomark + welcome text in the upper
        // portion of the chat area, with quick chips rendered
        // BELOW the input bar (web puts them there too, not
        // floating in the middle of the screen).
        if (state.conversation.turns.isEmpty() && !state.streaming) {
            com.aura.ui.components.EmptyChatState(
                modifier = Modifier.weight(1f),
            )
        } else {
            ChatMessageList(
                state = state,
                listState = listState,
                onShowSources = { showSources = true },
                onShowSourcesForLastTurn = { showSources = true },
                onSendSuggestion = { pick ->
                    viewModel.setDraft(pick)
                    viewModel.send()
                },
                modifier = Modifier.weight(1f),
            )
        }

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

        // Vision quick-prompt chips: shown above the input bar
        // right after the user captures or picks an image. The
        // bitmap is held in state.pendingVisionBitmap until the
        // user picks a prompt (which fires runVisionPrompt) or
        // dismisses the row.
        state.pendingVisionBitmap?.let { bitmap ->
            com.aura.ui.components.VisionPromptChips(
                onPick = { prompt -> viewModel.runVisionPrompt(bitmap, prompt) },
                onDismiss = { viewModel.dismissPendingVision() },
            )
        }

        if (state.conversation.turns.isEmpty() && !state.streaming) {
            com.aura.ui.components.QuickChipRow(
                onPick = { prompt ->
                    viewModel.setDraft(prompt)
                    viewModel.send()
                },
            )
        } else if (showJumpToBottom) {
            // Pill that shows above the input bar when the user has
            // scrolled up. Tap to jump back to the latest message.
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(bottom = 4.dp),
                horizontalArrangement = Arrangement.Center,
            ) {
                Surface(
                    onClick = {
                        coroutineScope.launch {
                            listState.animateScrollToItem(state.conversation.turns.size - 1)
                        }
                    },
                    color = AuraTokens.Dark.surface2,
                    shape = RoundedCornerShape(16.dp),
                    shadowElevation = 4.dp,
                ) {
                    Row(
                        verticalAlignment = Alignment.CenterVertically,
                        modifier = Modifier.padding(horizontal = 12.dp, vertical = 6.dp),
                    ) {
                        Icon(
                            imageVector = Icons.Filled.ArrowDownward,
                            contentDescription = "Jump to latest",
                            modifier = Modifier.size(14.dp),
                            tint = AuraTokens.Dark.textSecondary,
                        )
                        Spacer(Modifier.width(4.dp))
                        Text(
                            text = "Jump to latest",
                            fontFamily = com.aura.ui.theme.InterDisplay,
                            fontSize = 12.sp,
                            color = AuraTokens.Dark.textSecondary,
                        )
                    }
                }
            }
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
            onMicLongPress = {
                // Long-press the mic button = push-to-talk. The
                // overlay shows "Hold to talk" and the user controls
                // when to send via the stop button. Distinct from
                // tap (auto-send on first final result).
                if (hasMicPermission) showHoldToTalk = true
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

    // Re-fetch the model list every time the picker opens, so a key
    // added in Settings shows up without a manual app restart. The
    // refresh is a no-op if a fetch is already in flight.
    LaunchedEffect(showModelPicker) {
        if (showModelPicker) viewModel.refreshModels()
    }
    if (showModelPicker) {
        ModelPickerSheet(
            currentModel = state.activeModel,
            models = state.availableModels,
            isLoading = state.modelsLoading,
            errorMessage = state.modelsError,
            onPick = viewModel::setModel,
            onRefresh = { viewModel.refreshModels() },
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
        // Two voice modes are reachable from the mic button:
        // - Tap: tap-to-speak (auto-send on first final result)
        // - Long-press: hold-to-talk (the user controls when to send)
        // The mode flag is on the overlay state below; for now we
        // expose a single entry point. Hold-to-talk is wired in
        // commit 6 — see the holdToTalk flag in VoiceOverlay.
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

    // Hold-to-talk overlay: the user is in control. STT runs while
    // the overlay is shown. The user dismisses explicitly via the
    // stop button; the most recent transcript (final OR partial) is
    // sent on dismiss. Distinct from continuous voice mode (which
    // loops LISTENING → THINKING → SPEAKING until explicitly stopped)
    // and from tap-to-speak (which auto-sends on the first final
    // result). The hold-to-talk mode is for users who want to
    // speak, review what was transcribed, then send — not auto-send.
    if (showHoldToTalk && hasMicPermission) {
        VoiceOverlay(
            holdToTalk = true,
            onTranscript = { transcript ->
                viewModel.setDraft(transcript)
                viewModel.send()
            },
            onDismiss = { showHoldToTalk = false },
        )
    } else if (showHoldToTalk) {
        LaunchedEffect(Unit) { showHoldToTalk = false }
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
    onHistory: () -> Unit,
    onNewConversation: () -> Unit,
    onDeleteConversation: () -> Unit,
    onToggleDeepMode: () -> Unit,
    onToggleIncognito: () -> Unit,
    onShowModelPicker: () -> Unit,
) {
    val displayModel = conversationModel ?: activeModel
    val modelMismatch = conversationModel != null && conversationModel != activeModel
    // Web-style compact top bar: model picker pill on the
    // left, a row of 32dp icon buttons on the right, with
    // status pills (Deep / Incognito) floating above when
    // active. No surface background — the screen background
    // bleeds through. The previous version had a flat
    // Material 3 Surface with 8 action buttons and 2 filter
    // chips visible at once.
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .statusBarsPadding()
            .padding(horizontal = 12.dp, vertical = 8.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        // Model picker pill — 32dp tall, surface-2 background,
        // 16dp radius. Shows the active model in 13sp Inter
        // Medium + a small chevron. Matches the web's mode
        // pill exactly.
        Row(
            modifier = Modifier
                .clip(RoundedCornerShape(16.dp))
                .background(AuraTokens.Dark.surface2)
                .clickable { onShowModelPicker() }
                .padding(horizontal = 12.dp, vertical = 6.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            // Tiny breathing accent dot for the active model
            // — a brand touch, not a status indicator.
            Box(
                modifier = Modifier
                    .size(6.dp)
                    .clip(androidx.compose.foundation.shape.CircleShape)
                    .background(AuraTokens.Dark.accentPurple),
            )
            Spacer(modifier = Modifier.width(6.dp))
            Text(
                text = com.aura.ui.util.modelDisplayName(displayModel),
                fontSize = 13.sp,
                fontWeight = FontWeight.Medium,
                fontFamily = com.aura.ui.theme.InterDisplay,
                color = if (modelMismatch) AuraTokens.Dark.glowOrange
                       else AuraTokens.Dark.textPrimary,
            )
            Spacer(modifier = Modifier.width(4.dp))
            Icon(
                imageVector = Icons.Filled.ArrowDropDown,
                contentDescription = "Change model",
                tint = AuraTokens.Dark.textTertiary,
                modifier = Modifier.size(18.dp),
            )
        }
        // Active mode pills — show only when enabled. These
        // float as a thin row between the model picker and
        // the action icons.
        if (deepModeEnabled) {
            Spacer(modifier = Modifier.width(8.dp))
            ModeChip(
                label = if (deepModeActive) "Thinking…" else "Deep",
                accent = AuraTokens.Dark.glowBlue,
                onClick = onToggleDeepMode,
            )
        }
        if (incognitoMode) {
            Spacer(modifier = Modifier.width(8.dp))
            ModeChip(
                label = "Incognito",
                accent = AuraTokens.Dark.glowOrange,
                onClick = onToggleIncognito,
            )
        }
        Spacer(modifier = Modifier.weight(1f))
        // Action icons — 32dp round buttons, no background
        // by default. TTS uses accent color when on.
        HeaderIconButton(
            icon = if (ttsEnabled) Icons.Filled.VolumeUp else Icons.Filled.VolumeOff,
            contentDescription = if (ttsEnabled) "Mute TTS" else "Enable TTS",
            tint = if (ttsEnabled) AuraTokens.Dark.accentPurple
                   else AuraTokens.Dark.textSecondary,
            onClick = onToggleTts,
        )
        // The web has 4 small icons in the top-right
        // (TTS, share, save, more). The Copy + Voice
        // actions live in the message-bubble footer, not
        // the header — that's where users look for them
        // after reading a response, not before sending.
        HeaderIconButton(
            icon = Icons.Filled.Add,
            contentDescription = "New conversation",
            onClick = onNewConversation,
        )
        HeaderIconButton(
            icon = Icons.Filled.History,
            contentDescription = "History",
            onClick = onHistory,
        )
        HeaderIconButton(
            icon = Icons.Filled.DeleteOutline,
            contentDescription = "Delete conversation",
            onClick = onDeleteConversation,
        )
    }
}

@Composable
private fun HeaderIconButton(
    icon: androidx.compose.ui.graphics.vector.ImageVector,
    contentDescription: String,
    onClick: () -> Unit,
    tint: androidx.compose.ui.graphics.Color = AuraTokens.Dark.textSecondary,
) {
    Box(
        modifier = Modifier
            .size(36.dp)
            .clip(androidx.compose.foundation.shape.CircleShape)
            .clickable(onClick = onClick),
        contentAlignment = Alignment.Center,
    ) {
        Icon(
            imageVector = icon,
            contentDescription = contentDescription,
            tint = tint,
            modifier = Modifier.size(18.dp),
        )
    }
}

@Composable
private fun ModeChip(
    label: String,
    accent: androidx.compose.ui.graphics.Color,
    onClick: () -> Unit,
) {
    Row(
        modifier = Modifier
            .clip(RoundedCornerShape(12.dp))
            .background(accent.copy(alpha = 0.15f))
            .clickable(onClick = onClick)
            .padding(horizontal = 8.dp, vertical = 4.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Box(
            modifier = Modifier
                .size(6.dp)
                .clip(androidx.compose.foundation.shape.CircleShape)
                .background(accent),
        )
        Spacer(modifier = Modifier.width(4.dp))
        Text(
            text = label,
            fontSize = 11.sp,
            fontWeight = FontWeight.Medium,
            fontFamily = com.aura.ui.theme.InterDisplay,
            color = accent,
        )
    }
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
    onSendSuggestion: (String) -> Unit = {},
    onShowSourcesForLastTurn: () -> Unit = {},
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
                    turn.user?.let {
                        com.aura.ui.components.MessageBubble(
                            text = it,
                            isUser = true,
                            timestamp = turn.timestamp,
                        )
                    }
                    turn.assistant?.let {
                        val isLast = turn === state.conversation.turns.lastOrNull()
                        val isStreaming = state.streaming && isLast
                        com.aura.ui.components.MessageBubble(
                            text = it,
                            isUser = false,
                            citations = turn.citations,
                            isStreaming = isStreaming,
                            timestamp = turn.timestamp,
                            modelLabel = state.conversation.model,
                            onShowSources = onShowSourcesForLastTurn,
                        )
                        // Smart follow-up suggestions: shown for
                        // the LAST assistant turn (only) and only
                        // when not streaming. The user can tap a
                        // chip to send it as the next user
                        // message, or long-press to fill the
                        // draft. Heuristic-only — no LLM call.
                        if (isLast && !isStreaming) {
                            val isCodey = it.contains("```") || it.contains("`")
                            val suggestions = com.aura.ui.components.FollowUpSuggestions.suggest(
                                assistantText = it,
                                isCodey = isCodey,
                            )
                            com.aura.ui.components.FollowUpSuggestionChips(
                                suggestions = suggestions,
                                onPick = { pick ->
                                    // Send the picked suggestion
                                    // directly as the next user
                                    // message — the assistant's
                                    // turn just completed, so the
                                    // conversation is ready for
                                    // the next turn.
                                    onSendSuggestion(pick)
                                },
                            )
                        }
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
private fun TypingIndicator() {
    // Three animated dots — same idea as iMessage / WhatsApp, but
    // now rendered with Aura brand typography. The shimmer
    // [ThinkingShimmer] in the message list component handles the
    // in-list shimmer; this stub remains for the case where the
    // screen wants a standalone indicator above the input bar.
    com.aura.ui.components.ThinkingShimmer()
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
    onMicLongPress: () -> Unit = {},
    onCameraClick: () -> Unit,
    onGalleryClick: () -> Unit,
    onAudioClick: () -> Unit,
    showAttachmentSheet: Boolean,
    onAttachmentSheetChange: (Boolean) -> Unit,
) {
    val sheetState = rememberModalBottomSheetState(skipPartiallyExpanded = true)
    // The input bar has no surface of its own — the screen
    // background bleeds through so the glass text-field looks
    // like it's floating on the chat, not sitting in a separate
    // panel. The Web does the same with no border on the bottom
    // row.
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .navigationBarsPadding()
            .padding(horizontal = 10.dp, vertical = 8.dp),
        verticalAlignment = Alignment.Bottom,
        horizontalArrangement = Arrangement.spacedBy(8.dp),
    ) {
        IconButton(
            onClick = { onAttachmentSheetChange(true) },
            modifier = Modifier
                .size(40.dp)
                .clip(RoundedCornerShape(10.dp))
                .background(AuraTokens.Dark.surface2),
        ) {
            Icon(
                Icons.Filled.AddAPhoto,
                contentDescription = "Attach",
                tint = AuraTokens.Dark.textSecondary,
            )
        }
        // Glass input — surface-1 + border-subtle + 24dp radius.
        // The web does the same but with backdrop-blur 24dp; on
        // Android Compose there's no equivalent for arbitrary
        // composables (only for whole surfaces), so we use a
        // slightly-opaque surface-1 instead.
        Box(
            modifier = Modifier
                .weight(1f)
                .clip(RoundedCornerShape(24.dp))
                .background(AuraTokens.Dark.surface1)
                .border(1.dp, AuraTokens.Dark.borderSubtle, RoundedCornerShape(24.dp)),
        ) {
            BasicTextField(
                value = draft,
                onValueChange = onDraftChange,
                textStyle = MaterialTheme.typography.bodyLarge.copy(
                    color = AuraTokens.Dark.textPrimary,
                ),
                cursorBrush = SolidColor(AuraTokens.Dark.accentPurple),
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(horizontal = 16.dp, vertical = 14.dp),
                decorationBox = { inner ->
                    if (draft.isEmpty()) {
                        Text(
                            text = "Message AURA…",
                            fontFamily = com.aura.ui.theme.InterDisplay,
                            fontSize = 16.sp,
                            // Brighter than textTertiary (0xFF6B6B6B) which
                            // was barely visible against surface-1
                            // (0xFF202022) — placeholder now reads.
                            color = AuraTokens.Dark.textSecondary,
                        )
                    }
                    inner()
                },
                keyboardOptions = androidx.compose.foundation.text.KeyboardOptions(
                    imeAction = androidx.compose.ui.text.input.ImeAction.Send,
                ),
                keyboardActions = androidx.compose.foundation.text.KeyboardActions(onSend = { onSend() }),
            )
        }
        if (streaming) {
            // Stop button — square (12dp) red surface.
            Box(
                modifier = Modifier
                    .size(40.dp)
                    .clip(RoundedCornerShape(12.dp))
                    .background(AuraTokens.Dark.glowRed.copy(alpha = 0.15f))
                    .border(1.dp, AuraTokens.Dark.glowRed.copy(alpha = 0.3f), RoundedCornerShape(12.dp))
                    .clickable { onCancel() },
                contentAlignment = Alignment.Center,
            ) {
                Icon(
                    Icons.Filled.Stop,
                    contentDescription = "Stop",
                    tint = Color(0xFFF87171),
                    modifier = Modifier.size(16.dp),
                )
            }
        } else {
            // Mic button — 40dp circle, no background. Long-press
            // opens push-to-talk. Tap opens the tap-to-speak
            // overlay.
            Box(
                modifier = Modifier
                    .size(40.dp)
                    .clip(androidx.compose.foundation.shape.CircleShape)
                    .clickable { onMicClick() }
                    .pointerInput(Unit) {
                        detectTapGestures(onLongPress = { onMicLongPress() })
                    },
                contentAlignment = Alignment.Center,
            ) {
                Icon(
                    Icons.Filled.Mic,
                    contentDescription = "Voice input",
                    tint = AuraTokens.Dark.textSecondary,
                    modifier = Modifier.size(20.dp),
                )
            }
            // Send button — morphs between square (12dp, empty)
            // and pill (20dp, ready). The animation is spring-eased
            // matching the web's
            // `transition: all 0.25s cubic-bezier(0.34, 1.56, 0.64, 1)`.
            val canSend = draft.isNotBlank()
            val targetRadius = if (canSend) 20.dp else 12.dp
            val targetScale = if (canSend) 1f else 0.9f
            val animatedRadius by animateDpAsState(
                targetValue = targetRadius,
                animationSpec = spring(
                    dampingRatio = 0.65f,
                    stiffness = androidx.compose.animation.core.Spring.StiffnessMedium,
                ),
                label = "send-radius",
            )
            val animatedScale by animateFloatAsState(
                targetValue = targetScale,
                animationSpec = spring(
                    dampingRatio = 0.65f,
                    stiffness = androidx.compose.animation.core.Spring.StiffnessMedium,
                ),
                label = "send-scale",
            )
            Box(
                modifier = Modifier
                    .size(40.dp)
                    .graphicsLayer { scaleX = animatedScale; scaleY = animatedScale }
                    .clip(RoundedCornerShape(animatedRadius))
                    .background(
                        if (canSend) AuraTokens.Dark.sendReady
                        else AuraTokens.Dark.surface2,
                    )
                    .border(
                        width = if (canSend) 0.dp else 1.dp,
                        color = AuraTokens.Dark.borderSubtle,
                        shape = RoundedCornerShape(animatedRadius),
                    )
                    .clickable(enabled = canSend) {
                        com.aura.ui.util.Haptics.send(hapticView)
                        onSend()
                    },
                contentAlignment = Alignment.Center,
            ) {
                Icon(
                    imageVector = androidx.compose.material.icons.Icons.AutoMirrored.Filled.Send,
                    contentDescription = "Send",
                    tint = if (canSend) Color.White
                           else AuraTokens.Dark.textTertiary,
                    modifier = Modifier.size(18.dp),
                )
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
    val store = remember {
        EntryPointAccessors.fromApplication<HiltEntryPoint>(context.applicationContext as android.app.Application)
            .incomingShareStore()
    }
    // Collect the pending share as state so repeated shares — including
    // identical text shared twice while Chat is already visible — are
    // delivered. The previous LaunchedEffect(Unit) ran exactly once and
    // silently ignored any subsequent share intents.
    val pendingShare by store.pending.collectAsState()
    LaunchedEffect(pendingShare?.seq) {
        val payload = pendingShare ?: return@LaunchedEffect
        // Text share → set as draft
        payload.text?.let(viewModel::setDraft)
        // Image share → decode Bitmap and route through onImageCaptured
        // so the vision tool analyzes it instead of dumping base64
        // text into the chat input.
        payload.imageUri?.let { uri ->
            val bitmap = withContext(Dispatchers.IO) { decodeSharedImage(context, uri) }
            bitmap?.let { viewModel.onImageCaptured(it) }
        }
        // Clear the store so the same seq doesn't re-fire on recomposition.
        store.consume()
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
