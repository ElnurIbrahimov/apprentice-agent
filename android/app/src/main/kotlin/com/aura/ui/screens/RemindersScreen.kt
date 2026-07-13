package com.aura.ui.screens

import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding

import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Add
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material.icons.filled.Edit
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.AssistChip
import androidx.compose.material3.AssistChipDefaults
import androidx.compose.material3.FloatingActionButton
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
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
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.hilt.navigation.compose.hiltViewModel
import com.aura.tasks.ReminderEntity
import com.aura.ui.components.AuraScreenHeader
import com.aura.ui.viewmodel.RemindersViewModel
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

@Composable
fun RemindersScreen(
    onBack: () -> Unit = {},
    viewModel: RemindersViewModel = hiltViewModel(),
) {
    androidx.activity.compose.BackHandler(onBack = onBack)
    val state by viewModel.state.collectAsState()
    var showHistory by remember { mutableStateOf(false) }
    var showAdd by remember { mutableStateOf(false) }
    var editing by remember { mutableStateOf<ReminderEntity?>(null) }
    var cancelling by remember { mutableStateOf<ReminderEntity?>(null) }
    var confirmClearHistory by remember { mutableStateOf(false) }

    val rows = if (showHistory) state.history else state.upcoming
    Scaffold(
        contentWindowInsets = WindowInsets(0),
        floatingActionButton = {
            if (!showHistory) {
                FloatingActionButton(onClick = { showAdd = true }) {
                    Icon(Icons.Filled.Add, contentDescription = "Add reminder")
                }
            }
        },
    ) { padding ->
        Column(
            modifier = Modifier.fillMaxSize().padding(padding).padding(horizontal = 16.dp),
        ) {
            AuraScreenHeader(
                title = "Reminders",
                subtitle = if (showHistory) "${state.history.size} completed or cancelled" else "${state.upcoming.size} upcoming",
                action = if (showHistory && state.history.isNotEmpty()) ({
                    TextButton(onClick = { confirmClearHistory = true }) { Text("Clear history") }
                }) else null,
            )
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                AssistChip(
                    onClick = { showHistory = false },
                    label = { Text("Upcoming") },
                    colors = if (!showHistory) AssistChipDefaults.assistChipColors(
                        containerColor = MaterialTheme.colorScheme.primaryContainer,
                    ) else AssistChipDefaults.assistChipColors(),
                )
                AssistChip(
                    onClick = { showHistory = true },
                    label = { Text("History") },
                    colors = if (showHistory) AssistChipDefaults.assistChipColors(
                        containerColor = MaterialTheme.colorScheme.primaryContainer,
                    ) else AssistChipDefaults.assistChipColors(),
                )
            }
            Spacer(Modifier.height(8.dp))
            if (!state.loading && rows.isEmpty()) {
                Column(
                    modifier = Modifier.fillMaxWidth().padding(vertical = 56.dp),
                    horizontalAlignment = Alignment.CenterHorizontally,
                ) {
                    Text(
                        if (showHistory) "No reminder history" else "Nothing scheduled",
                        style = MaterialTheme.typography.titleMedium,
                        color = MaterialTheme.colorScheme.onBackground.copy(alpha = 0.65f),
                    )
                    if (!showHistory) {
                        TextButton(onClick = { showAdd = true }) { Text("Add a reminder") }
                    }
                }
            } else {
                LazyColumn(
                    contentPadding = PaddingValues(bottom = 96.dp),
                    verticalArrangement = Arrangement.spacedBy(8.dp),
                ) {
                    items(rows, key = { it.id }) { reminder ->
                        ReminderLifecycleRow(
                            reminder = reminder,
                            isHistory = showHistory,
                            onEdit = { editing = reminder },
                            onCancel = { cancelling = reminder },
                            onDelete = { viewModel.delete(reminder.id) },
                        )
                    }
                }
            }
        }
    }

    if (showAdd) {
        ReminderEditorDialog(
            title = "Add reminder",
            confirmLabel = "Add",
            onDismiss = { showAdd = false },
            onConfirm = { message, triggerAt, recurrence ->
                viewModel.create(message, triggerAt, recurrence)
                showAdd = false
            },
        )
    }
    editing?.let { reminder ->
        ReminderEditorDialog(
            title = "Edit reminder",
            confirmLabel = "Save",
            initialMessage = reminder.message,
            initialTriggerAt = reminder.triggerAt,
            initialRecurrence = reminder.recurrence,
            onDismiss = { editing = null },
            onConfirm = { message, triggerAt, recurrence ->
                viewModel.update(reminder.id, message, triggerAt, recurrence)
                editing = null
            },
        )
    }
    cancelling?.let { reminder ->
        AlertDialog(
            onDismissRequest = { cancelling = null },
            title = { Text("Cancel reminder?") },
            text = { Text("${reminder.message}\n\nThe reminder stays in History.") },
            confirmButton = {
                TextButton(onClick = {
                    viewModel.cancel(reminder.id)
                    cancelling = null
                }) { Text("Cancel reminder", color = MaterialTheme.colorScheme.error) }
            },
            dismissButton = { TextButton(onClick = { cancelling = null }) { Text("Keep") } },
        )
    }
    if (confirmClearHistory) {
        AlertDialog(
            onDismissRequest = { confirmClearHistory = false },
            title = { Text("Clear reminder history?") },
            text = { Text("This permanently removes fired and cancelled reminder records.") },
            confirmButton = {
                TextButton(onClick = {
                    viewModel.clearHistory()
                    confirmClearHistory = false
                }) { Text("Clear", color = MaterialTheme.colorScheme.error) }
            },
            dismissButton = { TextButton(onClick = { confirmClearHistory = false }) { Text("Keep") } },
        )
    }
}

@Composable
private fun ReminderLifecycleRow(
    reminder: ReminderEntity,
    isHistory: Boolean,
    onEdit: () -> Unit,
    onCancel: () -> Unit,
    onDelete: () -> Unit,
) {
    val format = SimpleDateFormat("MMM d, yyyy · HH:mm", Locale.US)
    Surface(
        color = MaterialTheme.colorScheme.surfaceVariant,
        shape = RoundedCornerShape(14.dp),
        modifier = Modifier.fillMaxWidth(),
    ) {
        Row(
            modifier = Modifier.padding(14.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Column(Modifier.weight(1f)) {
                Text(
                    reminder.message,
                    style = MaterialTheme.typography.bodyLarge,
                    fontWeight = FontWeight.Medium,
                    maxLines = 2,
                    overflow = TextOverflow.Ellipsis,
                )
                Spacer(Modifier.height(3.dp))
                Text(
                    buildString {
                        append(if (isHistory) reminder.status.replaceFirstChar { it.uppercase() } else format.format(Date(reminder.triggerAt)))
                        if (reminder.recurrence != "none") append(" · ${reminder.recurrence}")
                    },
                    style = MaterialTheme.typography.labelMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.65f),
                )
            }
            if (isHistory) {
                IconButton(onClick = onDelete) {
                    Icon(Icons.Filled.Delete, contentDescription = "Delete history record")
                }
            } else {
                IconButton(onClick = onEdit) {
                    Icon(Icons.Filled.Edit, contentDescription = "Edit reminder")
                }
                IconButton(onClick = onCancel) {
                    Icon(Icons.Filled.Delete, contentDescription = "Cancel reminder")
                }
            }
        }
    }
}
