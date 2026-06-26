package com.aura.ui.screens

import android.Manifest
import android.app.Activity
import android.content.pm.PackageManager
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
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
import androidx.compose.material.icons.filled.Mic
import androidx.compose.material.icons.filled.PhotoLibrary
import androidx.compose.material.icons.filled.Send
import androidx.compose.material.icons.filled.Stop
import androidx.compose.material.icons.filled.VolumeOff
import androidx.compose.material.icons.filled.VolumeUp
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.ModalBottomSheet
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
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
import androidx.core.content.ContextCompat
import androidx.hilt.navigation.compose.hiltViewModel
import com.aura.IncomingShareStore
import com.aura.agent.Specialist
import com.aura.ui.components.ModelPickerSheet
import com.aura.ui.components.SpecialistChips
import com.aura.ui.viewmodel.ChatViewModel
import com.aura.ui.voice.VoiceOverlay
import dagger.hilt.android.EntryPointAccessors

@Composable
fun ChatScreen(viewModel: ChatViewModel = hiltViewModel()) {
    val state by viewModel.state.collectAsState()
    val listState = rememberLazyListState()
    val context = LocalContext.current
    var showModelPicker by remember { mutableStateOf(false) }
    var showSources by remember { mutableStateOf(false) }
    var showVoiceOverlay by remember { mutableStateOf(false) }
    var hasMicPermission by remember {
        mutableStateOf(
            ContextCompat.checkSelfPermission(context, Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED
        )
    }
    val micPermissionLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.RequestPermission()
    ) { granted ->
        hasMicPermission = granted
        if (granted) showVoiceOverlay = true
    }

    val cameraLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.TakePicturePreview()
    ) { bitmap ->
        bitmap?.let { viewModel.onImageCaptured(it) }
    }

    val galleryLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.PickVisualMedia()
    ) { uri ->
        uri?.let {
            val bitmap = try {
                android.graphics.ImageDecoder.decodeBitmap(
                    android.graphics.ImageDecoder.createSource(context.contentResolver, it)
                )
            } catch (e: Exception) { null }
            bitmap?.let { bmp -> viewModel.onImageCaptured(bmp) }
        }
    }

    val audioLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.GetContent()
    ) { uri ->
        uri?.let { viewModel.onAudioPicked(it) }
    }

    var hasCameraPermission by remember {
        mutableStateOf(
            ContextCompat.checkSelfPermission(context, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED
        )
    }
    val cameraPermissionLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.RequestPermission()
    ) { granted ->
        hasCameraPermission = granted
        if (granted) cameraLauncher.launch(null)
    }

    // On first composition, consume any pending shared text from the share sheet
    // and pre-fill the chat draft. IncomingShareStore is a Hilt singleton.
    LaunchedEffect(Unit) {
        val store = EntryPointAccessors.fromApplication<HiltEntryPoint>(context.applicationContext as android.app.Application)
            .incomingShareStore()
        val pending = store.consume()
        if (!pending.isNullOrBlank()) {
            viewModel.setDraft(pending)
        }
    }

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
        // Header
        Surface(
            color = MaterialTheme.colorScheme.surface,
            tonalElevation = 1.dp,
            modifier = Modifier.fillMaxWidth(),
        ) {
            Row(
                modifier = Modifier
                    .clickable { showModelPicker = true }
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
                    Text(
                        text = humanModelName(state.activeModel),
                        style = MaterialTheme.typography.labelMedium,
                        color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.6f),
                    )
                }
                IconButton(onClick = { viewModel.toggleTts() }) {
                    Icon(
                        imageVector = if (state.ttsEnabled) Icons.Filled.VolumeUp else Icons.Filled.VolumeOff,
                        contentDescription = if (state.ttsEnabled) "TTS on, tap to mute" else "TTS off, tap to enable",
                        tint = if (state.ttsEnabled) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.onSurface.copy(alpha = 0.4f),
                    )
                }
                Icon(
                    imageVector = Icons.Filled.ArrowDropDown,
                    contentDescription = "Change model",
                    tint = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.6f),
                )
            }
        }

        // Messages
        LazyColumn(
            state = listState,
            modifier = Modifier
                .weight(1f)
                .fillMaxWidth(),
            contentPadding = PaddingValues(horizontal = 16.dp, vertical = 12.dp),
            verticalArrangement = Arrangement.spacedBy(10.dp),
        ) {
            items(state.conversation.turns) { turn ->
                turn.user?.let { userMsg ->
                    MessageBubble(text = userMsg, isUser = true)
                }
                turn.assistant?.let { assistantMsg ->
                    MessageBubble(
                        text = assistantMsg,
                        isUser = false,
                        citations = turn.citations,
                        onShowSources = { showSources = true },
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

        // Error banner
        state.error?.let { err ->
            Surface(
                color = MaterialTheme.colorScheme.errorContainer,
                modifier = Modifier.fillMaxWidth().padding(horizontal = 16.dp, vertical = 6.dp),
                shape = RoundedCornerShape(8.dp),
            ) {
                Text(
                    text = err,
                    modifier = Modifier.padding(12.dp),
                    color = MaterialTheme.colorScheme.onErrorContainer,
                    style = MaterialTheme.typography.bodySmall,
                )
            }
        }

        // Specialist chip row — shown when draft has content or a specialist is selected
        if (state.draft.isNotBlank() || state.selectedSpecialist != null) {
            SpecialistChips(
                selected = state.selectedSpecialist,
                suggested = state.suggestedSpecialist,
                onSelect = { specialist ->
                    viewModel.setSpecialist(specialist)
                },
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
                if (hasMicPermission) {
                    showVoiceOverlay = true
                } else {
                    micPermissionLauncher.launch(Manifest.permission.RECORD_AUDIO)
                }
            },
            onCameraClick = {
                if (hasCameraPermission) {
                    cameraLauncher.launch(null)
                } else {
                    cameraPermissionLauncher.launch(Manifest.permission.CAMERA)
                }
            },
            onGalleryClick = { galleryLauncher.launch(androidx.activity.result.PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly)) },
            onAudioClick = { audioLauncher.launch("audio/*") },
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

    if (showVoiceOverlay) {
        if (hasMicPermission) {
            VoiceOverlay(
                onTranscript = { transcript ->
                    viewModel.setDraft(transcript)
                    viewModel.send()
                },
                onDismiss = { showVoiceOverlay = false },
            )
        } else {
            LaunchedEffect(Unit) {
                showVoiceOverlay = false
            }
        }
    }
}

private fun humanModelName(id: String): String {
    val parts = id.split(":", limit = 2)
    val provider = parts.getOrNull(0) ?: "?"
    val model = parts.getOrNull(1) ?: id
    return when (id) {
        "ollama:deepseek-v3.2:cloud" -> "DeepSeek V3.2 · $provider"
        "ollama:kimi-k2.6:cloud" -> "Kimi K2.6 · $provider"
        "anthropic:claude-sonnet-4-5" -> "Claude Sonnet 4.5 · $provider"
        else -> "$model · $provider"
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
                            modifier = Modifier.padding(horizontal = 8.dp, vertical = 4.dp),
                            color = MaterialTheme.colorScheme.primary,
                            style = MaterialTheme.typography.labelSmall,
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
        color = MaterialTheme.colorScheme.tertiaryContainer,
        shape = RoundedCornerShape(10.dp),
        modifier = Modifier.fillMaxWidth().padding(horizontal = 32.dp),
    ) {
        Column(modifier = Modifier.padding(10.dp)) {
            Text(
                text = "🔧 $name",
                style = MaterialTheme.typography.labelMedium,
                color = MaterialTheme.colorScheme.onTertiaryContainer,
                fontWeight = FontWeight.SemiBold,
            )
            Spacer(modifier = Modifier.height(2.dp))
            Text(
                text = result.take(280) + if (result.length > 280) "…" else "",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onTertiaryContainer,
            )
        }
    }
}

@Composable
private fun TypingIndicator() {
    Row(
        modifier = Modifier.fillMaxWidth().padding(start = 4.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Surface(
            color = MaterialTheme.colorScheme.surfaceVariant,
            shape = RoundedCornerShape(16.dp),
        ) {
            Text(
                text = "● ● ●",
                modifier = Modifier.padding(horizontal = 14.dp, vertical = 10.dp),
                color = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.6f),
                style = MaterialTheme.typography.bodyLarge,
            )
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun SourcesSheet(
    citations: List<com.aura.tools.Citation>,
    onDismiss: () -> Unit,
) {
    val sheetState = rememberModalBottomSheetState(skipPartiallyExpanded = true)
    ModalBottomSheet(
        onDismissRequest = onDismiss,
        sheetState = sheetState,
        containerColor = MaterialTheme.colorScheme.surface,
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 20.dp, vertical = 12.dp),
        ) {
            Text(
                text = "Sources",
                style = MaterialTheme.typography.titleLarge,
                fontWeight = FontWeight.SemiBold,
            )
            Spacer(modifier = Modifier.height(12.dp))
            LazyColumn(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                items(citations) { citation ->
                    CitationRow(citation = citation)
                }
            }
            Spacer(modifier = Modifier.height(8.dp))
        }
    }
}

@Composable
private fun CitationRow(citation: com.aura.tools.Citation) {
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 4.dp),
    ) {
        Text(
            text = "[${citation.index}] ${citation.title}",
            style = MaterialTheme.typography.bodyMedium,
            fontWeight = FontWeight.SemiBold,
            color = MaterialTheme.colorScheme.onSurface,
        )
        Text(
            text = citation.url,
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.primary,
        )
        HorizontalDivider(modifier = Modifier.padding(top = 8.dp))
    }
}

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
) {
    Surface(
        color = MaterialTheme.colorScheme.surface,
        tonalElevation = 2.dp,
        modifier = Modifier
            .fillMaxWidth()
            .navigationBarsPadding(),
    ) {
        Row(
            modifier = Modifier.padding(horizontal = 12.dp, vertical = 10.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            if (!streaming) {
                IconButton(onClick = onCameraClick) {
                    Icon(Icons.Filled.AddAPhoto, contentDescription = "Camera", tint = MaterialTheme.colorScheme.onSurface)
                }
                IconButton(onClick = onGalleryClick) {
                    Icon(Icons.Filled.PhotoLibrary, contentDescription = "Gallery", tint = MaterialTheme.colorScheme.onSurface)
                }
                IconButton(onClick = onAudioClick) {
                    Icon(Icons.Filled.AudioFile, contentDescription = "Audio", tint = MaterialTheme.colorScheme.onSurface)
                }
            }
            OutlinedTextField(
                value = draft,
                onValueChange = onDraftChange,
                modifier = Modifier.weight(1f),
                placeholder = { Text("Ask Aura…") },
                maxLines = 4,
                shape = RoundedCornerShape(24.dp),
            )
            Spacer(modifier = Modifier.width(4.dp))
            IconButton(onClick = onMicClick) {
                Icon(
                    imageVector = Icons.Filled.Mic,
                    contentDescription = "Voice input",
                    tint = MaterialTheme.colorScheme.onSurface,
                )
            }
            Spacer(modifier = Modifier.width(4.dp))
            if (streaming) {
                IconButton(
                    onClick = onCancel,
                    modifier = Modifier.background(MaterialTheme.colorScheme.error, RoundedCornerShape(50)),
                ) {
                    Icon(Icons.Filled.Stop, contentDescription = "Stop", tint = MaterialTheme.colorScheme.onError)
                }
            } else {
                IconButton(
                    onClick = onSend,
                    enabled = draft.isNotBlank(),
                    modifier = Modifier.background(
                        if (draft.isNotBlank()) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.surfaceVariant,
                        RoundedCornerShape(50),
                    ),
                ) {
                    Icon(Icons.Filled.Send, contentDescription = "Send", tint = MaterialTheme.colorScheme.onPrimary)
                }
            }
        }
    }
}
