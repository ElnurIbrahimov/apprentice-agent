package com.aura.ui.screens

import android.content.Context
import android.content.Intent
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.tween
import androidx.compose.foundation.background
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Chat
import androidx.compose.material.icons.filled.Close
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material.icons.filled.PushPin
import androidx.compose.material.icons.filled.Search
import androidx.compose.material.icons.filled.Share
import androidx.compose.material3.Checkbox
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Surface
import androidx.compose.material3.TextButton
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.hilt.navigation.compose.hiltViewModel
import androidx.core.content.FileProvider
import com.aura.agent.Conversation
import com.aura.ui.components.AuraScreenHeader
import com.aura.ui.viewmodel.HistoryViewModel
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

@Composable
fun HistoryScreen(
    viewModel: HistoryViewModel = hiltViewModel(),
    onSelect: (String) -> Unit,
) {
    val state by viewModel.state.collectAsState()
    val context = LocalContext.current
    val coroutineScope = rememberCoroutineScope()

    val subtitle = if (state.selectMode) {
        val n = state.selectedIds.size
        "$n selected"
    } else if (state.query.isBlank()) "${state.conversations.size} saved conversations"
        else "${state.conversations.size} match${if (state.conversations.size == 1) "" else "es"} for \"${state.query}\""
    Column(modifier = Modifier.fillMaxSize().padding(horizontal = 20.dp)) {
        val headerAction: (@Composable () -> Unit)? = when {
            // Bulk action bar — replaces the normal "Export all" action
            // when in select mode. Shows the count and Delete/Share/Cancel.
            state.selectMode -> ({
                Row(verticalAlignment = Alignment.CenterVertically) {
                    TextButton(onClick = { viewModel.selectAll() }) {
                        Text("All", color = MaterialTheme.colorScheme.primary)
                    }
                    TextButton(
                        onClick = {
                            val md = viewModel.exportSelectedMarkdown()
                            if (md.isNotEmpty()) {
                                coroutineScope.launch {
                                    shareMarkdown(
                                        context,
                                        md,
                                        "aura-${state.selectedIds.size}-conversations",
                                    )
                                }
                            }
                        },
                        enabled = state.selectedIds.isNotEmpty(),
                    ) {
                        Icon(Icons.Filled.Share, contentDescription = null, modifier = Modifier.size(18.dp))
                        Spacer(Modifier.width(4.dp))
                        Text("Share")
                    }
                    TextButton(
                        onClick = { viewModel.deleteSelected() },
                        enabled = state.selectedIds.isNotEmpty(),
                    ) {
                        Icon(
                            Icons.Filled.Delete,
                            contentDescription = null,
                            tint = MaterialTheme.colorScheme.error,
                            modifier = Modifier.size(18.dp),
                        )
                        Spacer(Modifier.width(4.dp))
                        Text("Delete", color = MaterialTheme.colorScheme.error)
                    }
                    TextButton(onClick = { viewModel.toggleSelectMode() }) {
                        Text("Cancel")
                    }
                }
            })
            // Normal mode: "Select" enters select mode; "Export all"
            // shares the entire list as one Markdown doc.
            state.conversations.isNotEmpty() && state.query.isBlank() -> ({
                Row(verticalAlignment = Alignment.CenterVertically) {
                    TextButton(onClick = { viewModel.toggleSelectMode() }) {
                        Text("Select")
                    }
                    TextButton(onClick = {
                        coroutineScope.launch {
                            shareMarkdown(context, viewModel.exportAllMarkdown(), "aura-conversations")
                        }
                    }) {
                        Icon(Icons.Filled.Share, contentDescription = null, modifier = Modifier.size(18.dp))
                        Spacer(Modifier.width(4.dp))
                        Text("Export all")
                    }
                }
            })
            else -> null
        }
        AuraScreenHeader(
            title = "History",
            subtitle = subtitle,
            action = headerAction,
        )

        // Search bar.
        OutlinedTextField(
            value = state.query,
            onValueChange = viewModel::onQueryChanged,
            placeholder = { Text("Search by meaning, title, or message") },
            leadingIcon = { Icon(Icons.Filled.Search, contentDescription = null) },
            trailingIcon = {
                if (state.query.isNotEmpty()) {
                    IconButton(onClick = { viewModel.onQueryChanged("") }) {
                        Icon(Icons.Filled.Close, contentDescription = "Clear search")
                    }
                }
            },
            singleLine = true,
            modifier = Modifier.fillMaxWidth(),
        )

        if (state.searching) {
            Spacer(Modifier.height(2.dp))
            LinearProgressIndicator(
                modifier = Modifier.fillMaxWidth(),
            )
        }

        Spacer(modifier = Modifier.height(8.dp))
        HorizontalDivider()
        Spacer(modifier = Modifier.height(8.dp))

        if (state.loading) {
            HistorySkeletonLoading()
        } else if (state.conversations.isEmpty()) {
            Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                Column(horizontalAlignment = Alignment.CenterHorizontally) {
                    val emptyMessage = if (state.query.isBlank()) "No conversations yet"
                    else "No matches for \"${state.query}\""
                    Icon(
                        imageVector = if (state.query.isBlank()) Icons.Filled.Chat else Icons.Filled.Search,
                        contentDescription = null,
                        modifier = Modifier.size(64.dp),
                        tint = MaterialTheme.colorScheme.onBackground.copy(alpha = 0.25f),
                    )
                    Spacer(Modifier.height(8.dp))
                    Text(emptyMessage, style = MaterialTheme.typography.titleMedium)
                    if (state.query.isNotBlank()) {
                        Spacer(Modifier.height(4.dp))
                        Text(
                            "Aura searches exact words and related meaning across titles and messages.",
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onBackground.copy(alpha = 0.6f),
                        )
                    }
                }
            }
        } else {
            LazyColumn(contentPadding = PaddingValues(vertical = 8.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                items(state.conversations, key = { it.id }) { conv ->
                    HistoryRow(
                        conv = conv,
                        stats = viewModel.getStats(conv),
                        isPinned = viewModel.isPinned(conv),
                        isSelected = conv.id in state.selectedIds,
                        selectMode = state.selectMode,
                        onClick = {
                            if (state.selectMode) {
                                viewModel.toggleSelected(conv.id)
                            } else {
                                onSelect(conv.id)
                            }
                        },
                        onLongPress = {
                            if (state.selectMode) viewModel.toggleSelected(conv.id)
                        },
                        onDelete = { viewModel.delete(conv.id) },
                        onShare = {
                            coroutineScope.launch {
                                shareMarkdown(context, viewModel.exportMarkdown(conv), conv.title)
                            }
                        },
                        onTogglePin = { viewModel.togglePinned(conv.id) },
                        onRename = { newTitle -> viewModel.setTitle(conv.id, newTitle) },
                    )
                }
            }
        }
    }
}

@Composable
private fun HistoryRow(
    conv: Conversation,
    stats: HistoryViewModel.ConversationStats,
    isPinned: Boolean,
    isSelected: Boolean,
    selectMode: Boolean,
    onClick: () -> Unit,
    onLongPress: () -> Unit,
    onDelete: () -> Unit,
    onShare: () -> Unit,
    onTogglePin: () -> Unit,
    onRename: (String) -> Unit,
) {
    val fmt = SimpleDateFormat("MMM d, HH:mm", Locale.US)
    val lastTurn = conv.turns.lastOrNull()
    val preview = lastTurn?.user ?: lastTurn?.assistant ?: "Empty"

    // Long-press the row → rename dialog. Tap = open conversation.
    // Pin icon toggles the pinned flag.
    var showRenameDialog by androidx.compose.runtime.remember { mutableStateOf(false) }

    if (showRenameDialog) {
        var text by androidx.compose.runtime.remember { mutableStateOf(conv.title) }
        androidx.compose.material3.AlertDialog(
            onDismissRequest = { showRenameDialog = false },
            title = { androidx.compose.material3.Text("Rename conversation") },
            text = {
                androidx.compose.material3.OutlinedTextField(
                    value = text,
                    onValueChange = { text = it.take(120) },
                    singleLine = true,
                    label = { androidx.compose.material3.Text("Title") },
                )
            },
            confirmButton = {
                androidx.compose.material3.TextButton(onClick = {
                    onRename(text)
                    showRenameDialog = false
                }) {
                    androidx.compose.material3.Text("Save")
                }
            },
            dismissButton = {
                androidx.compose.material3.TextButton(onClick = { showRenameDialog = false }) {
                    androidx.compose.material3.Text("Cancel")
                }
            },
        )
    }

    Surface(
        color = if (isSelected) MaterialTheme.colorScheme.primary.copy(alpha = 0.12f)
                else MaterialTheme.colorScheme.surfaceVariant,
        shape = RoundedCornerShape(10.dp),
        modifier = Modifier
            .fillMaxWidth()
            .pointerInput(conv.id) {
                detectTapGestures(
                    onTap = { onClick() },
                    onLongPress = {
                        // Normal mode: long-press opens the rename
                        // dialog (preserves the existing UX).
                        // Select mode: long-press toggles the row's
                        // selection — no rename dialog, since the
                        // user is in bulk-action mode.
                        if (selectMode) onLongPress() else showRenameDialog = true
                    },
                )
            },
    ) {
        Row(modifier = Modifier.padding(12.dp), verticalAlignment = Alignment.CenterVertically) {
            if (selectMode) {
                Checkbox(
                    checked = isSelected,
                    onCheckedChange = { onClick() },
                )
                Spacer(Modifier.width(4.dp))
            } else {
                Icon(
                    imageVector = if (isPinned) Icons.Filled.PushPin else Icons.Filled.Chat,
                    contentDescription = if (isPinned) "Pinned" else null,
                    tint = if (isPinned) MaterialTheme.colorScheme.primary
                           else MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.5f),
                    modifier = Modifier.size(24.dp),
                )
                Spacer(Modifier.width(12.dp))
            }
            Column(modifier = Modifier.weight(1f)) {
                Text(conv.title, style = MaterialTheme.typography.bodyMedium, fontWeight = FontWeight.SemiBold, maxLines = 1, overflow = TextOverflow.Ellipsis)
                Spacer(Modifier.height(2.dp))
                Text(preview.take(80), style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.7f), maxLines = 1, overflow = TextOverflow.Ellipsis)
                Spacer(Modifier.height(3.dp))
                Text(
                    text = formatConversationStats(stats),
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.primary.copy(alpha = 0.82f),
                )
                Spacer(Modifier.height(2.dp))
                Text(
                    text = conv.model?.let { com.aura.ui.util.modelDisplayName(it) } ?: "Unknown model",
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.6f),
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis,
                )
                Spacer(Modifier.height(2.dp))
                Text(fmt.format(Date(conv.updatedAt)), style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.4f))
            }
            IconButton(
                onClick = onTogglePin,
                modifier = Modifier.size(32.dp),
            ) {
                Icon(
                    imageVector = Icons.Filled.PushPin,
                    contentDescription = if (isPinned) "Unpin" else "Pin",
                    tint = if (isPinned) MaterialTheme.colorScheme.primary
                           else MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.4f),
                    modifier = Modifier.size(18.dp),
                )
            }
            IconButton(onClick = onShare, modifier = Modifier.size(48.dp)) {
                Icon(
                    imageVector = Icons.Filled.Share,
                    contentDescription = "Share as Markdown",
                    tint = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.6f),
                    modifier = Modifier.size(18.dp),
                )
            }
            IconButton(onClick = onDelete, modifier = Modifier.size(48.dp)) {
                Icon(Icons.Filled.Delete, "Delete", tint = MaterialTheme.colorScheme.error.copy(alpha = 0.6f), modifier = Modifier.size(18.dp))
            }
        }
    }
}

internal fun formatConversationStats(stats: HistoryViewModel.ConversationStats): String {
    val parts = mutableListOf(
        "${stats.turns} ${if (stats.turns == 1) "turn" else "turns"}",
    )
    if (stats.toolCalls > 0) {
        parts += "${stats.toolCalls} ${if (stats.toolCalls == 1) "tool" else "tools"}"
    }
    if (stats.durationMs >= 60_000L) {
        val minutes = stats.durationMs / 60_000L
        parts += when {
            minutes >= 1_440L -> "${minutes / 1_440L}d span"
            minutes >= 60L -> "${minutes / 60L}h span"
            else -> "${minutes}m span"
        }
    }
    return parts.joinToString(" · ")
}

// ── Skeleton Loading ─────────────────────────────────────────────────────────

/**
 * Skeleton loading placeholder for the history list. Shows 5 card-shaped
 * placeholders that mirror the HistoryRow layout — icon, title, preview,
 * meta lines, and action button slots — pulsing subtly so the transition
 * to real content feels smooth.
 */
@Composable
private fun HistorySkeletonLoading() {
    val infiniteTransition = rememberInfiniteTransition(label = "skeleton-pulse")
    val pulseAlpha by infiniteTransition.animateFloat(
        initialValue = 0.3f,
        targetValue = 0.7f,
        animationSpec = infiniteRepeatable(animation = tween(durationMillis = 900)),
        label = "pulse-alpha",
    )

    LazyColumn(
        contentPadding = PaddingValues(vertical = 8.dp),
        verticalArrangement = Arrangement.spacedBy(8.dp),
    ) {
        items(5) {
            SkeletonHistoryCard(alpha = pulseAlpha)
        }
    }
}

@Composable
private fun SkeletonHistoryCard(alpha: Float) {
    Surface(
        color = MaterialTheme.colorScheme.surfaceVariant,
        shape = RoundedCornerShape(10.dp),
        modifier = Modifier.fillMaxWidth(),
    ) {
        Row(
            modifier = Modifier.padding(12.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            // Icon placeholder (matches HistoryRow 24dp Chat/PushPin icon)
            Box(
                modifier = Modifier
                    .size(24.dp)
                    .alpha(alpha * 0.5f)
                    .background(
                        color = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = alpha * 0.3f),
                        shape = CircleShape,
                    ),
            )
            Spacer(Modifier.width(12.dp))
            Column(modifier = Modifier.weight(1f)) {
                // Title line
                Box(
                    modifier = Modifier
                        .fillMaxWidth(0.55f)
                        .height(16.dp)
                        .background(
                            color = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = alpha * 0.4f),
                            shape = RoundedCornerShape(4.dp),
                        ),
                )
                Spacer(Modifier.height(6.dp))
                // Preview line
                Box(
                    modifier = Modifier
                        .fillMaxWidth(0.85f)
                        .height(12.dp)
                        .background(
                            color = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = alpha * 0.25f),
                            shape = RoundedCornerShape(4.dp),
                        ),
                )
                Spacer(Modifier.height(6.dp))
                // Meta row: model name + date
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Box(
                        modifier = Modifier
                            .width(120.dp)
                            .height(10.dp)
                            .background(
                                color = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = alpha * 0.2f),
                                shape = RoundedCornerShape(4.dp),
                            ),
                    )
                    Spacer(Modifier.width(12.dp))
                    Box(
                        modifier = Modifier
                            .width(80.dp)
                            .height(10.dp)
                            .background(
                                color = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = alpha * 0.2f),
                                shape = RoundedCornerShape(4.dp),
                            ),
                    )
                }
            }
            Spacer(Modifier.width(4.dp))
            // Action button placeholders (pin, share, delete)
            Box(
                modifier = Modifier
                    .size(32.dp)
                    .background(
                        color = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = alpha * 0.2f),
                        shape = RoundedCornerShape(6.dp),
                    ),
            )
            Spacer(Modifier.width(4.dp))
            Box(
                modifier = Modifier
                    .size(32.dp)
                    .background(
                        color = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = alpha * 0.2f),
                        shape = RoundedCornerShape(6.dp),
                    ),
            )
            Spacer(Modifier.width(4.dp))
            Box(
                modifier = Modifier
                    .size(32.dp)
                    .background(
                        color = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = alpha * 0.2f),
                        shape = RoundedCornerShape(6.dp),
                    ),
            )
        }
    }
}

/**
 * Build a file in the app cache, hand it to the system share
 * chooser, and clean up after. The file is named after the
 * conversation title (slugified) plus a timestamp so multiple
 * shares in a row don't collide.
 */
private suspend fun shareMarkdown(context: Context, markdown: String, title: String) {
    withContext(Dispatchers.IO) {
        val safeTitle = title.lowercase(Locale.US)
            .replace(Regex("[^a-z0-9]+"), "-")
            .trim('-')
            .take(40)
            .ifBlank { "conversation" }
        val ts = SimpleDateFormat("yyyyMMdd-HHmmss", Locale.US).format(Date())
        val file = File(context.cacheDir, "aura-$safeTitle-$ts.md")
        file.writeText(markdown)
        val uri = FileProvider.getUriForFile(
            context,
            "${context.packageName}.fileprovider",
            file,
        )
        val share = Intent(Intent.ACTION_SEND).apply {
            type = "text/markdown"
            putExtra(Intent.EXTRA_STREAM, uri)
            putExtra(Intent.EXTRA_SUBJECT, title)
            addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
        }
        context.startActivity(
            Intent.createChooser(share, "Share conversation")
                .addFlags(Intent.FLAG_ACTIVITY_NEW_TASK),
        )
    }
}
