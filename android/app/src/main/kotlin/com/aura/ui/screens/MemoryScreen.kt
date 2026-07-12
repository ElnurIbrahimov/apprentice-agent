package com.aura.ui.screens

import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.tween
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ExperimentalLayoutApi
import androidx.compose.foundation.layout.FlowRow
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
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Add
import androidx.compose.material.icons.filled.Build
import androidx.compose.material.icons.filled.Close
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material.icons.filled.Edit
import androidx.compose.material.icons.filled.Search
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.AssistChip
import androidx.compose.material3.AssistChipDefaults
import androidx.compose.material3.Button
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Slider
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.unit.dp
import androidx.hilt.navigation.compose.hiltViewModel
import com.aura.memory.MemoryEntity
import com.aura.ui.components.AuraScreenHeader
import com.aura.ui.viewmodel.MemoryViewModel
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

private val MEMORY_CATEGORIES = listOf("fact", "preference", "episode", "person", "project", "idea", "task")

@OptIn(ExperimentalLayoutApi::class)
@Composable
fun MemoryScreen(viewModel: MemoryViewModel = hiltViewModel()) {
    val state by viewModel.state.collectAsState()
    var editingMemory by remember { mutableStateOf<MemoryEntity?>(null) }
    var showRebuildConfirm by remember { mutableStateOf(false) }
    var showClearAllConfirm by remember { mutableStateOf(false) }
    var showClearCategoryConfirm by remember { mutableStateOf(false) }
    var showAddNote by remember { mutableStateOf(false) }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(horizontal = 20.dp)
    ) {
        val filterText = state.categoryFilter?.let { " · $it" } ?: ""
        val subtitle = when {
            state.memories.isEmpty() && state.query.isBlank() -> "No memories yet$filterText"
            state.memories.isEmpty() -> "No memories match your filters$filterText"
            state.memories.size == 1 -> "1 memory$filterText"
            else -> "${state.memories.size} memories$filterText"
        }
        AuraScreenHeader(title = "Memory", subtitle = subtitle)

        // Rebuild-embeddings banner — shown after the user has run the
        // action. Dismissible. The text comes from the VM so the
        // success / no-op / failure cases all render the same way.
        state.rebuildResult?.let { result ->
            Surface(
                color = MaterialTheme.colorScheme.tertiaryContainer,
                shape = RoundedCornerShape(10.dp),
                modifier = Modifier.fillMaxWidth().padding(bottom = 8.dp),
            ) {
                Row(
                    modifier = Modifier.padding(horizontal = 12.dp, vertical = 8.dp),
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    Text(
                        text = result,
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onTertiaryContainer,
                        modifier = Modifier.weight(1f),
                    )
                    IconButton(
                        onClick = { viewModel.clearRebuildResult() },
                        modifier = Modifier.size(24.dp),
                    ) {
                        Icon(
                            imageVector = Icons.Filled.Close,
                            contentDescription = "Dismiss",
                            tint = MaterialTheme.colorScheme.onTertiaryContainer,
                            modifier = Modifier.size(18.dp),
                        )
                    }
                }
            }
        }

        // Search bar
        OutlinedTextField(
            value = state.query,
            onValueChange = { viewModel.setQuery(it) },
            placeholder = { Text("Search memories") },
            leadingIcon = { Icon(Icons.Filled.Search, contentDescription = null) },
            singleLine = true,
            modifier = Modifier.fillMaxWidth(),
            keyboardOptions = KeyboardOptions(imeAction = ImeAction.Search),
        )
        Spacer(modifier = Modifier.height(8.dp))

        // Category filter chips
        FlowRow(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(6.dp),
            verticalArrangement = Arrangement.spacedBy(6.dp),
        ) {
            AssistChip(
                onClick = { viewModel.setCategory(null) },
                label = { Text("All") },
                colors = if (state.categoryFilter == null)
                    AssistChipDefaults.assistChipColors(containerColor = MaterialTheme.colorScheme.primary, labelColor = MaterialTheme.colorScheme.onPrimary)
                else AssistChipDefaults.assistChipColors(),
            )
            for (cat in MEMORY_CATEGORIES) {
                AssistChip(
                    onClick = { viewModel.setCategory(cat) },
                    label = { Text(cat) },
                    colors = if (state.categoryFilter == cat)
                        AssistChipDefaults.assistChipColors(containerColor = MaterialTheme.colorScheme.primary, labelColor = MaterialTheme.colorScheme.onPrimary)
                    else AssistChipDefaults.assistChipColors(),
                )
            }
        }


        // Manual note creation — bypasses the write gate so the user
        // can explicitly store anything they want without going through
        // the agent.
        Button(
            onClick = { showAddNote = true },
            modifier = Modifier.fillMaxWidth(),
            shape = RoundedCornerShape(18.dp),
        ) {
            Icon(Icons.Filled.Add, contentDescription = null, modifier = Modifier.size(18.dp))
            Spacer(Modifier.width(6.dp))
            Text("Add note")
        }

        if (state.memories.isNotEmpty()) {
            Spacer(modifier = Modifier.height(8.dp))

            // Rebuild embeddings action. Visible only once there is something
            // to rebuild.
            OutlinedButton(
                onClick = { showRebuildConfirm = true },
                enabled = !state.rebuildInFlight,
                modifier = Modifier.fillMaxWidth(),
            ) {
                if (state.rebuildInFlight) {
                    CircularProgressIndicator(
                        modifier = Modifier.size(16.dp),
                        strokeWidth = 2.dp,
                        color = MaterialTheme.colorScheme.primary,
                    )
                    Spacer(modifier = Modifier.size(8.dp))
                    Text("Rebuilding…")
                } else {
                    Icon(
                        imageVector = Icons.Filled.Build,
                        contentDescription = null,
                        modifier = Modifier.size(18.dp),
                    )
                    Spacer(modifier = Modifier.size(8.dp))
                    Text("Rebuild embeddings")
                }
            }
            Spacer(modifier = Modifier.height(8.dp))

            // Bulk delete actions. "Clear category" only shows when a
            // category filter is active. Both show a confirm dialog.
            if (state.categoryFilter != null) {
                OutlinedButton(
                    onClick = { showClearCategoryConfirm = true },
                    modifier = Modifier.fillMaxWidth(),
                ) {
                    Icon(
                        imageVector = Icons.Filled.Delete,
                        contentDescription = null,
                        modifier = Modifier.size(18.dp),
                        tint = MaterialTheme.colorScheme.error,
                    )
                    Spacer(modifier = Modifier.size(8.dp))
                    Text("Clear ${state.categoryFilter} memories", color = MaterialTheme.colorScheme.error)
                }
                Spacer(modifier = Modifier.height(8.dp))
            }
            OutlinedButton(
                onClick = { showClearAllConfirm = true },
                modifier = Modifier.fillMaxWidth(),
            ) {
                Icon(
                    imageVector = Icons.Filled.Delete,
                    contentDescription = null,
                    modifier = Modifier.size(18.dp),
                    tint = MaterialTheme.colorScheme.error,
                )
                Spacer(modifier = Modifier.size(8.dp))
                Text("Clear all memories", color = MaterialTheme.colorScheme.error)
            }
            Spacer(modifier = Modifier.height(8.dp))
        }

        Spacer(modifier = Modifier.height(8.dp))

        if (state.loading) {
            MemorySkeletonLoading()
        } else if (state.memories.isEmpty()) {
            Surface(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(top = 12.dp),
                shape = RoundedCornerShape(20.dp),
                color = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.28f),
            ) {
                Column(
                    modifier = Modifier.padding(horizontal = 18.dp, vertical = 18.dp),
                    verticalArrangement = Arrangement.spacedBy(6.dp),
                ) {
                    Text(
                        text = if (state.query.isBlank()) "No memories yet" else "Nothing matches that search",
                        style = MaterialTheme.typography.titleMedium,
                        fontWeight = FontWeight.SemiBold,
                    )
                    Text(
                        text = if (state.query.isBlank()) {
                            "Add a note, or keep chatting and Aura will save facts, preferences, and episodes here."
                        } else {
                            "Try a different keyword or clear the category filter."
                        },
                        style = MaterialTheme.typography.bodyMedium,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
            }
        } else {
            LazyColumn(
                contentPadding = PaddingValues(vertical = 8.dp),
                verticalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                items(state.memories, key = { it.id }) { mem ->
                    MemoryRow(
                        mem = mem,
                        onEdit = { editingMemory = mem },
                        onForget = { viewModel.forget(mem.id) },
                    )
                }
            }
        }
    }

    // Edit dialog (rendered outside the main Column so it overlays everything)
    editingMemory?.let { mem ->
        EditMemoryDialog(
            memory = mem,
            onDismiss = { editingMemory = null },
            onSave = { newContent, newCategory, newImportance, newTags ->
                viewModel.update(mem.id, newContent, newCategory, newImportance, newTags)
                editingMemory = null
            },
        )
    }

    // Rebuild confirmation dialog. The action is reversible (re-running
    // it on a clean table is a no-op) but for a 500-row install it
    // takes long enough that an accidental tap is annoying.
    if (showRebuildConfirm) {
        AlertDialog(
            onDismissRequest = { showRebuildConfirm = false },
            title = { Text("Rebuild embeddings?") },
            text = {
                Text(
                    "Re-embed every memory that currently has a null embedding. " +
                        "Use this after restoring a backup — Aura intentionally " +
                        "drops embeddings on export because they're model-specific. " +
                        "Existing embeddings are left alone.",
                )
            },
            confirmButton = {
                TextButton(onClick = {
                    showRebuildConfirm = false
                    viewModel.rebuildEmbeddings()
                }) { Text("Rebuild") }
            },
            dismissButton = {
                TextButton(onClick = { showRebuildConfirm = false }) { Text("Cancel") }
            },
        )
    }

    // Clear all memories confirmation. Irreversible.
    if (showClearAllConfirm) {
        AlertDialog(
            onDismissRequest = { showClearAllConfirm = false },
            title = { Text("Clear all memories?") },
            text = {
                Text(
                    "This will permanently delete all ${state.memories.size} memories. " +
                        "This cannot be undone. Consider exporting a backup first.",
                )
            },
            confirmButton = {
                TextButton(onClick = {
                    showClearAllConfirm = false
                    viewModel.forgetAll()
                }) { Text("Delete all", color = MaterialTheme.colorScheme.error) }
            },
            dismissButton = {
                TextButton(onClick = { showClearAllConfirm = false }) { Text("Cancel") }
            },
        )
    }

    // Clear category confirmation. Irreversible.
    if (showClearCategoryConfirm) {
        val cat = state.categoryFilter ?: ""
        val count = state.memories.size
        AlertDialog(
            onDismissRequest = { showClearCategoryConfirm = false },
            title = { Text("Clear $cat memories?") },
            text = {
                Text(
                    "This will permanently delete all $count memories in the " +
                        "\"$cat\" category. This cannot be undone.",
                )
            },
            confirmButton = {
                TextButton(onClick = {
                    showClearCategoryConfirm = false
                    viewModel.forgetByCategory(cat)
                }) { Text("Delete $count", color = MaterialTheme.colorScheme.error) }
            },
            dismissButton = {
                TextButton(onClick = { showClearCategoryConfirm = false }) { Text("Cancel") }
            },
        )
    }

    if (showAddNote) {
        AddNoteDialog(
            onDismiss = { showAddNote = false },
            onAdd = { content, category, importance ->
                viewModel.createNote(content, category, importance)
                showAddNote = false
            },
        )
    }
}

@Composable
private fun MemoryRow(
    mem: MemoryEntity,
    onEdit: () -> Unit,
    onForget: () -> Unit,
) {
    val age = (System.currentTimeMillis() - mem.createdAt) / 1000
    val ageDisplay = when {
        age < 60 -> "just now"
        age < 3600 -> "${age / 60}m ago"
        age < 86400 -> "${age / 3600}h ago"
        else -> "${age / 86400}d ago"
    }

    Surface(
        color = MaterialTheme.colorScheme.surfaceVariant,
        shape = RoundedCornerShape(10.dp),
        modifier = Modifier.fillMaxWidth(),
    ) {
        Row(
            modifier = Modifier.padding(12.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            CategoryDot(mem.category)
            Spacer(modifier = Modifier.size(10.dp))
            Column(modifier = Modifier.weight(1f)) {
                Text(
                    text = mem.content,
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
                Spacer(modifier = Modifier.height(2.dp))
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Text(
                        text = mem.category,
                        style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.7f),
                    )
                    Text(
                        text = "  \u00B7  ${mem.source}",
                        style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.4f),
                    )
                    Text(
                        text = "  \u00B7  $ageDisplay",
                        style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.7f),
                    )
                    if (mem.decayScore < 0.5f) {
                        Text(
                            text = "  \u00B7  fading",
                            style = MaterialTheme.typography.labelSmall,
                            color = MaterialTheme.colorScheme.error.copy(alpha = 0.7f),
                            fontWeight = FontWeight.Medium,
                        )
                    }
                }
                // Second metadata line: importance + recall count + tags.
                // These fields are used by the RRF retrieval engine but
                // were previously invisible to the user.
                val metaParts = mutableListOf<String>()
                if (mem.importance != 0.5f) {
                    metaParts += "${"%.0f".format(mem.importance * 100)}% important"
                }
                if (mem.accessCount > 0) {
                    metaParts += if (mem.accessCount == 1) "recalled 1\u00D7" else "recalled ${mem.accessCount}\u00D7"
                }
                if (mem.tags.isNotBlank()) {
                    metaParts += mem.tags
                }
                if (metaParts.isNotEmpty()) {
                    Spacer(modifier = Modifier.height(2.dp))
                    Text(
                        text = metaParts.joinToString("  \u00B7  "),
                        style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.5f),
                    )
                }
            }
            IconButton(onClick = onEdit) {
                Icon(
                    imageVector = Icons.Filled.Edit,
                    contentDescription = "Edit",
                    tint = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.6f),
                )
            }
            IconButton(onClick = onForget) {
                Icon(
                    imageVector = Icons.Filled.Delete,
                    contentDescription = "Forget",
                    tint = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.6f),
                )
            }
        }
    }
}

@Composable
private fun EditMemoryDialog(
    memory: MemoryEntity,
    onDismiss: () -> Unit,
    onSave: (content: String, category: String, importance: Float, tags: String) -> Unit,
) {
    var content by remember(memory.id) { mutableStateOf(memory.content) }
    var category by remember(memory.id) { mutableStateOf(memory.category) }
    var importance by remember(memory.id) { mutableStateOf(memory.importance) }
    var tags by remember(memory.id) { mutableStateOf(memory.tags) }
    var categoryMenuOpen by remember { mutableStateOf(false) }

    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text("Edit memory") },
        text = {
            Column {
                OutlinedTextField(
                    value = content,
                    onValueChange = { content = it },
                    label = { Text("Content") },
                    minLines = 2,
                    maxLines = 6,
                    modifier = Modifier.fillMaxWidth(),
                )
                Spacer(Modifier.height(8.dp))
                Box {
                    OutlinedTextField(
                        value = category,
                        onValueChange = {},
                        label = { Text("Category") },
                        readOnly = true,
                        trailingIcon = {
                            TextButton(onClick = { categoryMenuOpen = true }) {
                                Text("\u25BE")
                            }
                        },
                        modifier = Modifier.fillMaxWidth(),
                    )
                    DropdownMenu(
                        expanded = categoryMenuOpen,
                        onDismissRequest = { categoryMenuOpen = false },
                    ) {
                        for (cat in MEMORY_CATEGORIES) {
                            DropdownMenuItem(
                                text = { Text(cat) },
                                onClick = {
                                    category = cat
                                    categoryMenuOpen = false
                                },
                            )
                        }
                    }
                }
                Spacer(Modifier.height(8.dp))
                OutlinedTextField(
                    value = tags,
                    onValueChange = { tags = it },
                    label = { Text("Tags (comma-separated)") },
                    singleLine = true,
                    modifier = Modifier.fillMaxWidth(),
                )
                Spacer(Modifier.height(8.dp))
                Text(
                    text = "Importance: ${"%.0f".format(importance * 100)}%",
                    style = MaterialTheme.typography.labelMedium,
                    color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.7f),
                )
                Slider(
                    value = importance,
                    onValueChange = { importance = it },
                    valueRange = 0f..1f,
                    steps = 9,
                )
                Spacer(Modifier.height(4.dp))
                Text(
                    text = "Higher importance = ranks higher in recall. Embedding is re-computed on next recall.",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.6f),
                )
            }
        },
        confirmButton = {
            TextButton(
                onClick = { onSave(content.trim(), category, importance, tags.trim()) },
                enabled = content.trim().isNotBlank() && (
                    content.trim() != memory.content ||
                    category != memory.category ||
                    importance != memory.importance ||
                    tags.trim() != memory.tags
                ),
            ) {
                Text("Save")
            }
        },
        dismissButton = {
            TextButton(onClick = onDismiss) { Text("Cancel") }
        },
    )
}

@Composable
private fun CategoryDot(category: String) {
    val color = when (category) {
        "preference" -> MaterialTheme.colorScheme.primary
        "person" -> MaterialTheme.colorScheme.secondary
        "task" -> MaterialTheme.colorScheme.tertiary
        "idea" -> MaterialTheme.colorScheme.primary.copy(alpha = 0.6f)
        else -> MaterialTheme.colorScheme.outline
    }
    Box(
        modifier = Modifier
            .size(10.dp)
            .background(color, CircleShape),
    )
}

// ── Skeleton Loading ─────────────────────────────────────────────────────────

/**
 * Skeleton loading placeholder for the memories list. Shows 5 card-shaped
 * placeholders with a pulse animation that mirrors the MemoryRow layout —
 * category dot, content line, metadata rows, and action icon placeholders.
 * The header, search bar, and filter chips remain visible throughout so the
 * transition to full content feels natural.
 */
@Composable
private fun MemorySkeletonLoading() {
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
            SkeletonMemoryCard(alpha = pulseAlpha)
        }
    }
}

@Composable
private fun SkeletonMemoryCard(alpha: Float) {
    Surface(
        color = MaterialTheme.colorScheme.surfaceVariant,
        shape = RoundedCornerShape(10.dp),
        modifier = Modifier.fillMaxWidth(),
    ) {
        Row(
            modifier = Modifier.padding(12.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            // Category dot placeholder
            Box(
                modifier = Modifier
                    .size(10.dp)
                    .background(
                        color = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = alpha * 0.4f),
                        shape = CircleShape,
                    ),
            )
            Spacer(modifier = Modifier.size(10.dp))
            Column(modifier = Modifier.weight(1f)) {
                // Content line placeholder
                Box(
                    modifier = Modifier
                        .fillMaxWidth(0.8f)
                        .height(14.dp)
                        .background(
                            color = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = alpha * 0.35f),
                            shape = RoundedCornerShape(4.dp),
                        ),
                )
                Spacer(modifier = Modifier.height(4.dp))
                // First metadata line: category · source · age
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Box(
                        modifier = Modifier
                            .width(40.dp)
                            .height(10.dp)
                            .background(
                                color = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = alpha * 0.2f),
                                shape = RoundedCornerShape(4.dp),
                            ),
                    )
                    Spacer(modifier = Modifier.width(6.dp))
                    Box(
                        modifier = Modifier
                            .width(60.dp)
                            .height(10.dp)
                            .background(
                                color = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = alpha * 0.2f),
                                shape = RoundedCornerShape(4.dp),
                            ),
                    )
                    Spacer(modifier = Modifier.width(6.dp))
                    Box(
                        modifier = Modifier
                            .width(36.dp)
                            .height(10.dp)
                            .background(
                                color = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = alpha * 0.2f),
                                shape = RoundedCornerShape(4.dp),
                            ),
                    )
                }
                // Second metadata line: importance · recall count · tags
                Spacer(modifier = Modifier.height(4.dp))
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Box(
                        modifier = Modifier
                            .width(72.dp)
                            .height(10.dp)
                            .background(
                                color = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = alpha * 0.15f),
                                shape = RoundedCornerShape(4.dp),
                            ),
                    )
                    Spacer(modifier = Modifier.width(6.dp))
                    Box(
                        modifier = Modifier
                            .width(52.dp)
                            .height(10.dp)
                            .background(
                                color = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = alpha * 0.15f),
                                shape = RoundedCornerShape(4.dp),
                            ),
                    )
                }
            }
            // Edit icon placeholder
            Box(
                modifier = Modifier
                    .size(24.dp)
                    .background(
                        color = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = alpha * 0.15f),
                        shape = CircleShape,
                    ),
            )
            Spacer(modifier = Modifier.width(4.dp))
            // Delete icon placeholder
            Box(
                modifier = Modifier
                    .size(24.dp)
                    .background(
                        color = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = alpha * 0.15f),
                        shape = CircleShape,
                    ),
            )
        }
    }
}


@Composable
private fun AddNoteDialog(
    onDismiss: () -> Unit,
    onAdd: (content: String, category: String, importance: Float) -> Unit,
) {
    var content by remember { mutableStateOf("") }
    var category by remember { mutableStateOf("note") }
    var importance by remember { mutableStateOf(0.5f) }
    val categories = listOf("note", "fact", "preference", "person", "episode", "idea", "task")

    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text("Add note") },
        text = {
            Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
                OutlinedTextField(
                    value = content,
                    onValueChange = { content = it },
                    label = { Text("Note content") },
                    modifier = Modifier.fillMaxWidth(),
                    minLines = 2,
                )
                Row(
                    horizontalArrangement = Arrangement.spacedBy(8.dp),
                    modifier = Modifier.fillMaxWidth(),
                ) {
                    categories.forEach { cat ->
                        AssistChip(
                            onClick = { category = cat },
                            label = { Text(cat.replaceFirstChar { it.uppercase() }) },
                            colors = if (category == cat)
                                AssistChipDefaults.assistChipColors()
                            else
                                AssistChipDefaults.assistChipColors(
                                    containerColor = MaterialTheme.colorScheme.surface,
                                ),
                        )
                    }
                }
                Text("Importance: ${"%.1f".format(importance)}", style = MaterialTheme.typography.labelMedium)
                Slider(
                    value = importance,
                    onValueChange = { importance = it },
                    valueRange = 0f..1f,
                )
            }
        },
        confirmButton = {
            TextButton(
                onClick = { onAdd(content, category, importance) },
                enabled = content.isNotBlank(),
            ) { Text("Add") }
        },
        dismissButton = {
            TextButton(onClick = onDismiss) { Text("Cancel") }
        },
    )
}
