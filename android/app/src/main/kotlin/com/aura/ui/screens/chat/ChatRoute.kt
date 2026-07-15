package com.aura.ui.screens.chat

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

import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size

import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.interaction.collectIsDraggedAsState
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
import androidx.compose.runtime.snapshotFlow
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.SolidColor
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.hilt.navigation.compose.hiltViewModel
import com.aura.IncomingShareStore
import com.aura.agent.Specialist
import com.aura.ui.components.MarkdownText
import com.aura.ui.components.MoaThinkingIndicator
import com.aura.ui.components.ModelPickerSheet
import com.aura.ui.theme.AuraThemeTokens
import com.aura.ui.components.SpecialistChips
import com.aura.ui.viewmodel.ChatViewModel
import com.aura.ui.viewmodel.calculateImageSampleSize
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

internal enum class ChatVoiceMode { Tap, Hold, Continuous }

internal data class MicPermissionState(
    val granted: Boolean,
    val pendingMode: ChatVoiceMode? = null,
) {
    fun request(mode: ChatVoiceMode): MicPermissionState = copy(pendingMode = mode)

    fun resolve(granted: Boolean): MicPermissionOutcome = MicPermissionOutcome(
        state = copy(granted = granted, pendingMode = null),
        launchMode = pendingMode?.takeIf { granted },
    )
}

internal data class MicPermissionOutcome(
    val state: MicPermissionState,
    val launchMode: ChatVoiceMode?,
)

internal fun findSourceTurnIndex(turnTimestamps: List<Long>, targetTimestamp: Long): Int =
    turnTimestamps.indexOf(targetTimestamp)

@Composable
fun ChatRoute(
    viewModel: ChatViewModel = hiltViewModel(),
    resumeConversationId: String? = null,
    morningBriefSummary: String? = null,
    initialDraft: String? = null,
    focusTurnTimestamp: Long? = null,
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
    val skills by viewModel.skills.collectAsState()
    val listState = rememberLazyListState()
    val isUserDragging by listState.interactionSource.collectIsDraggedAsState()
    var followLiveEdge by remember { mutableStateOf(true) }
    val physicallyAtLiveEdge by remember(listState) {
        derivedStateOf {
            val total = listState.layoutInfo.totalItemsCount
            val lastVisible = listState.layoutInfo.visibleItemsInfo.lastOrNull()?.index ?: -1
            total == 0 || lastVisible >= total - 1
        }
    }
    LaunchedEffect(isUserDragging) {
        if (isUserDragging) {
            snapshotFlow { physicallyAtLiveEdge }.collect { atEdge ->
                if (shouldDetachFromLiveEdge(isUserDragging, atEdge)) followLiveEdge = false
            }
        }
    }
    LaunchedEffect(state.conversation.id) {
        followLiveEdge = focusTurnTimestamp == null
    }
    LaunchedEffect(
        state.conversation.id,
        state.conversation.turns.size,
        focusTurnTimestamp,
    ) {
        val targetTimestamp = focusTurnTimestamp ?: return@LaunchedEffect
        if (state.conversation.id != resumeConversationId) return@LaunchedEffect
        val targetIndex = findSourceTurnIndex(
            state.conversation.turns.map { it.timestamp },
            targetTimestamp,
        )
        if (targetIndex >= 0) {
            followLiveEdge = false
            listState.scrollToItem(targetIndex)
        }
    }
    val showJumpToBottom = shouldShowJumpToLatest(state.conversation.turns.size, followLiveEdge)
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
    StopStreamingDialog(
        visible = showStopStreamConfirm,
        onStop = {
            viewModel.cancel()
            showStopStreamConfirm = false
        },
        onKeepStreaming = { showStopStreamConfirm = false },
    )
    var showDeleteConfirm by remember { mutableStateOf(false) }

    var micPermissionState by remember(context) {
        mutableStateOf(MicPermissionState(granted = checkMicPermission(context)))
    }
    val hasMicPermission = micPermissionState.granted
    val micPermissionLauncher = rememberLauncherForActivityResult(
        contract = androidx.activity.result.contract.ActivityResultContracts.RequestPermission()
    ) { granted ->
        val outcome = micPermissionState.resolve(granted)
        micPermissionState = outcome.state
        when (outcome.launchMode) {
            ChatVoiceMode.Tap -> showVoiceOverlay = true
            ChatVoiceMode.Hold -> showHoldToTalk = true
            ChatVoiceMode.Continuous -> showContinuousVoice = true
            null -> Unit
        }
    }

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

    var hasCameraPermission by remember(context) {
        mutableStateOf(checkCameraPermission(context))
    }
    val cameraPermissionLauncher = rememberLauncherForActivityResult(
        contract = androidx.activity.result.contract.ActivityResultContracts.RequestPermission()
    ) { granted ->
        hasCameraPermission = granted
        if (granted) cameraLauncher.launch(null)
    }

    ConsumeIncomingShare(context, viewModel)

    val lastTurn = state.conversation.turns.lastOrNull()
    val assistantLen = lastTurn?.assistant?.length ?: 0
    LaunchedEffect(state.conversation.turns.size, assistantLen) {
        if (shouldAutoFollow(state.conversation.turns.size, followLiveEdge)) {
            listState.scrollToItem(state.conversation.turns.size - 1)
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

    ChatContent(
        state = state,
        listState = listState,
        showJumpToLatest = showJumpToBottom,
        onJumpToLatest = {
            followLiveEdge = true
            coroutineScope.launch {
                if (state.conversation.turns.isNotEmpty()) {
                    listState.animateScrollToItem(state.conversation.turns.size - 1)
                }
            }
        },
        onShowModelPicker = { showModelPicker = true },
        onToggleTts = viewModel::toggleTts,
        onHistory = onNavigateHistory,
        onNewConversation = {
            followLiveEdge = true
            viewModel.newConversation()
        },
        onDeleteConversation = { showDeleteConfirm = true },
        onToggleDeepMode = viewModel::toggleDeepMode,
        onToggleIncognito = viewModel::toggleIncognito,
        onSendSuggestion = { prompt ->
            followLiveEdge = true
            viewModel.setDraft(prompt)
            viewModel.send()
        },
        onRetry = viewModel::retryLast,
        onDismissError = viewModel::dismissError,
        onDismissProviderWarning = viewModel::dismissProviderWarning,
        onDismissSaveWarning = viewModel::dismissSaveWarning,
        onSelectSpecialist = viewModel::setSpecialist,
        onRunVisionPrompt = viewModel::runVisionPrompt,
        onDismissVision = viewModel::dismissPendingVision,
        onShowSources = { showSources = true },
        onReact = { timestamp, reaction -> viewModel.reactToTurn(timestamp, reaction) },
        composer = {
            ChatComposer(
                draft = state.draft,
                streaming = state.streaming,
                sendEnabled = state.modelSelection is com.aura.ui.viewmodel.ModelSelectionState.Ready,
                onDraftChange = viewModel::setDraft,
                onSend = {
                    followLiveEdge = true
                    viewModel.send()
                },
                onCancel = viewModel::cancel,
                onTapToSpeak = {
                    if (hasMicPermission) showVoiceOverlay = true
                    else {
                        micPermissionState = micPermissionState.request(ChatVoiceMode.Tap)
                        micPermissionLauncher.launch(android.Manifest.permission.RECORD_AUDIO)
                    }
                },
                onHoldToTalk = {
                    if (hasMicPermission) showHoldToTalk = true
                    else {
                        micPermissionState = micPermissionState.request(ChatVoiceMode.Hold)
                        micPermissionLauncher.launch(android.Manifest.permission.RECORD_AUDIO)
                    }
                },
                onContinuousVoice = {
                    if (hasMicPermission) showContinuousVoice = true
                    else {
                        micPermissionState = micPermissionState.request(ChatVoiceMode.Continuous)
                        micPermissionLauncher.launch(android.Manifest.permission.RECORD_AUDIO)
                    }
                },
                onCameraClick = {
                    if (hasCameraPermission) cameraLauncher.launch(null)
                    else cameraPermissionLauncher.launch(android.Manifest.permission.CAMERA)
                },
                onGalleryClick = {
                    galleryLauncher.launch(
                        androidx.activity.result.PickVisualMediaRequest(
                            androidx.activity.result.contract.ActivityResultContracts.PickVisualMedia.ImageOnly,
                        ),
                    )
                },
                onAudioClick = { audioLauncher.launch("audio/*") },
                skills = skills,
                onUseSkill = { skill ->
                    val directive = "/use_skill ${skill.name}\n"
                    viewModel.setDraft(directive + state.draft)
                },
            )
        },
    )

    if (showModelPicker) {
        ModelPickerSheet(
            currentModel = state.activeModel,
            models = state.availableModels,
            isLoading = state.modelsLoading,
            errorMessage = state.modelsError,
            staleProviderPrefixes = (state.modelSelection as? com.aura.ui.viewmodel.ModelSelectionState.Ready)
                ?.staleProviders.orEmpty(),
            selectionScopeLabel = "Applies to this chat only",
            onMakeDefault = {
                viewModel.makeActiveModelDefault()
                showModelPicker = false
            },
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

    DeleteConversationDialog(
        visible = showDeleteConfirm,
        onDelete = {
            showDeleteConfirm = false
            viewModel.deleteCurrentConversation()
        },
        onDismiss = { showDeleteConfirm = false },
    )

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

private fun checkMicPermission(context: android.content.Context): Boolean =
    androidx.core.content.ContextCompat.checkSelfPermission(
        context,
        android.Manifest.permission.RECORD_AUDIO,
    ) == android.content.pm.PackageManager.PERMISSION_GRANTED

private fun checkCameraPermission(context: android.content.Context): Boolean =
    androidx.core.content.ContextCompat.checkSelfPermission(
        context,
        android.Manifest.permission.CAMERA,
    ) == android.content.pm.PackageManager.PERMISSION_GRANTED

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
            val resolver = context.contentResolver
            val bounds = android.graphics.BitmapFactory.Options().apply {
                inJustDecodeBounds = true
            }
            resolver.openInputStream(uri)?.use { stream ->
                @Suppress("DEPRECATION")
                android.graphics.BitmapFactory.decodeStream(stream, null, bounds)
            }
            if (bounds.outWidth <= 0 || bounds.outHeight <= 0) {
                return@runCatching null
            }
            val opts = android.graphics.BitmapFactory.Options().apply {
                inSampleSize = calculateImageSampleSize(
                    width = bounds.outWidth,
                    height = bounds.outHeight,
                    target = 1024,
                )
            }
            resolver.openInputStream(uri)?.use { stream ->
                @Suppress("DEPRECATION")
                android.graphics.BitmapFactory.decodeStream(stream, null, opts)
            }
        }.getOrNull()
    }
}
