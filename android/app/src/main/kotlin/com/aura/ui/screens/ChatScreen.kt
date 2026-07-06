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
import com.aura.ui.components.MoaThinkingIndicator
import com.aura.ui.components.ModelPickerSheet
import com.aura.ui.components.SpecialistChips
import com.aura.ui.viewmodel.ChatViewModel
import com.aura.ui.voice.VoiceOverlay
import dagger.hilt.EntryPoint
import dagger.hilt.InstallIn
import dagger.hilt.android.EntryPointAccessors
import dagger.hilt.components.SingletonComponent

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
    onNavigateHistory: () -> Unit = {},
) {
    LaunchedEffect(resumeConversationId) {
        if (resumeConversationId != null) viewModel.loadConversation(resumeConversationId)
    }
    LaunchedEffect(morningBriefSummary) {
        if (!morningBriefSummary.isNullOrBlank()) viewModel.onUserMessage(morningBriefSummary)
    }

    val state by viewModel.state.collectAsState()
    val listState = rememberLazyListState()
    val context = LocalContext.current

    var showModelPicker by remember { mutableStateOf(false) }
    var showSources by remember { mutableStateOf(false) }
    var showVoiceOverlay by remember { mutableStateOf(false) }
    var showAttachmentSheet by remember { mutableStateOf(false) }

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

    LaunchedEffect(state.conversation.turns.size) {
        if (state.conversation.turns.isNotEmpty()) {
            listState.animateScrollToItem(state.conversation.turns.size - 1)
        }
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .imePadding()
            .background(MaterialTheme.colorScheme.background),
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

    PermissionDialog(
        permission = state.pendingPermission,
        rationale = state.permissionRationale,
        onGrant = viewModel::retryAfterPermission,
        onDismiss = viewModel::dismissPermission,
    )
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
    onToggleDeepMode: () -> Unit,
    onToggleIncognito: () -> Unit,
    onShowModelPicker: () -> Unit,
) {
    Surface(
        color = MaterialTheme.colorScheme.surface,
        tonalElevation = 1.dp,
        modifier = Modifier.fillMaxWidth(),
    ) {
        Row(
            modifier = Modifier
                .clickable { onShowModelPicker() }
                .padding(horizontal = 16.dp, vertical = 14.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Column(modifier = Modifier.weight(1f)) {
                Text(
                    text = "Aura",
                    style = MaterialTheme.typography.titleLarge,
                    fontWeight = FontWeight.SemiBold,
                    color = MaterialTheme.colorScheme.onSurface,
                )
                // Show the conversation's model if it differs from the
                // active selection — the user reopened an old conversation
                // that used a different model and needs to know which model
                // produced the responses they're reading.
                val displayModel = conversationModel ?: activeModel
                val modelMismatch = conversationModel != null && conversationModel != activeModel
                Text(
                    text = com.aura.ui.util.modelDisplayName(displayModel),
                    style = MaterialTheme.typography.labelMedium,
                    color = if (modelMismatch) MaterialTheme.colorScheme.tertiary
                    else MaterialTheme.colorScheme.onSurface.copy(alpha = 0.6f),
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
        items(state.conversation.turns) { turn ->
            turn.user?.let { MessageBubble(text = it, isUser = true) }
            turn.assistant?.let {
                MessageBubble(
                    text = it,
                    isUser = false,
                    citations = turn.citations,
                    onShowSources = onShowSources,
                )
            }
            for (toolTurn in turn.toolTurns) {
                if (toolTurn.result.isNotEmpty()) {
                    ToolCallBubble(name = toolTurn.name, result = toolTurn.result)
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
) {
    val bubbleColor = if (isUser) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.surfaceVariant
    val textColor = if (isUser) MaterialTheme.colorScheme.onPrimary else MaterialTheme.colorScheme.onSurfaceVariant
    val alignment = if (isUser) Alignment.End else Alignment.Start
    Column(
        modifier = Modifier.fillMaxWidth(),
        horizontalAlignment = alignment,
    ) {
        Surface(
            color = bubbleColor,
            shape = RoundedCornerShape(
                topStart = 16.dp,
                topEnd = 16.dp,
                bottomStart = if (isUser) 16.dp else 4.dp,
                bottomEnd = if (isUser) 4.dp else 16.dp,
            ),
            modifier = Modifier.widthIn(max = 320.dp),
        ) {
            Column(modifier = Modifier.padding(horizontal = 14.dp, vertical = 10.dp)) {
                Text(
                    text = text.ifBlank { "…" },
                    color = textColor,
                    style = MaterialTheme.typography.bodyMedium,
                )
                if (!isUser && citations.isNotEmpty()) {
                    Spacer(modifier = Modifier.height(6.dp))
                    Surface(
                        color = MaterialTheme.colorScheme.primary.copy(alpha = 0.12f),
                        shape = RoundedCornerShape(8.dp),
                        modifier = Modifier.clickable { onShowSources() },
                    ) {
                        Text(
                            text = "${citations.size} sources",
                            modifier = Modifier.padding(horizontal = 10.dp, vertical = 4.dp),
                            style = MaterialTheme.typography.labelSmall,
                            color = MaterialTheme.colorScheme.onSurface,
                        )
                    }
                }
            }
        }
    }
}

@Composable
private fun ToolCallBubble(name: String, result: String) {
    Surface(
        color = MaterialTheme.colorScheme.secondaryContainer.copy(alpha = 0.7f),
        shape = RoundedCornerShape(10.dp),
        modifier = Modifier.fillMaxWidth(0.85f).padding(start = 32.dp),
    ) {
        Column(modifier = Modifier.padding(horizontal = 12.dp, vertical = 8.dp)) {
            Text(
                text = name,
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.onSecondaryContainer,
                fontWeight = FontWeight.SemiBold,
            )
            Spacer(modifier = Modifier.height(2.dp))
            Text(
                text = result.take(280),
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSecondaryContainer.copy(alpha = 0.85f),
            )
        }
    }
}

@Composable
private fun TypingIndicator() {
    Row(
        modifier = Modifier.padding(start = 12.dp, top = 4.dp, bottom = 4.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Text(
            text = "Aura is typing",
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onBackground.copy(alpha = 0.6f),
        )
        Spacer(modifier = Modifier.width(6.dp))
        Text(
            text = "●●●",
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.primary.copy(alpha = 0.7f),
        )
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun ChatInputBar(
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
        tonalElevation = 2.dp,
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .navigationBarsPadding()
                .padding(horizontal = 12.dp, vertical = 10.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            IconButton(onClick = { onAttachmentSheetChange(true) }) {
                Icon(Icons.Filled.AddAPhoto, contentDescription = "Attach")
            }
            OutlinedTextField(
                value = draft,
                onValueChange = onDraftChange,
                placeholder = { Text("Ask Aura…") },
                modifier = Modifier.weight(1f).padding(horizontal = 8.dp),
                maxLines = 5,
                keyboardOptions = androidx.compose.foundation.text.KeyboardOptions(
                    imeAction = androidx.compose.ui.text.input.ImeAction.Send
                ),
                keyboardActions = androidx.compose.foundation.text.KeyboardActions(onSend = { onSend() }),
            )
            if (streaming) {
                IconButton(onClick = onCancel) {
                    Icon(Icons.Filled.Stop, contentDescription = "Stop", tint = MaterialTheme.colorScheme.error)
                }
            } else {
                IconButton(onClick = onMicClick) {
                    Icon(Icons.Filled.Mic, contentDescription = "Voice input", tint = MaterialTheme.colorScheme.primary)
                }
                IconButton(
                    onClick = onSend,
                    enabled = draft.isNotBlank(),
                ) {
                    Icon(Icons.Filled.Send, contentDescription = "Send", tint = MaterialTheme.colorScheme.primary)
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
    val permLauncher = rememberLauncherForActivityResult(
        contract = androidx.activity.result.contract.ActivityResultContracts.RequestPermission()
    ) { granted ->
        if (granted) onGrant(permission) else onDismiss()
    }
    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text("Permission needed") },
        text = {
            Text(
                buildString {
                    append(rationale ?: "Aura needs access to continue.")
                    append("\n\nPermission: $permission")
                }
            )
        },
        confirmButton = {
            TextButton(onClick = { permLauncher.launch(permission) }) {
                Text("Grant")
            }
        },
        dismissButton = {
            TextButton(onClick = onDismiss) {
                Text("Deny")
            }
        },
    )
}
