package com.aura.ui.screens

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
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Add
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material.icons.filled.Edit
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.AssistChip
import androidx.compose.material3.AssistChipDefaults
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.DatePicker
import androidx.compose.material3.DatePickerDialog
import androidx.compose.material3.DisplayMode
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.FloatingActionButton
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.InputChip
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TimePicker
import androidx.compose.material3.rememberDatePickerState
import androidx.compose.material3.rememberTimePickerState
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
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
import com.aura.tasks.TaskEntity
import com.aura.ui.viewmodel.TasksViewModel
import java.text.SimpleDateFormat
import java.util.Calendar
import java.util.Date
import java.util.Locale

@Composable
fun TasksScreen(viewModel: TasksViewModel = hiltViewModel()) {
    val state by viewModel.state.collectAsState()
    var showAdd by remember { mutableStateOf(false) }
    var editingTask by remember { mutableStateOf<TaskEntity?>(null) }
    var showClearConfirm by remember { mutableStateOf(false) }

    Scaffold(
        floatingActionButton = {
            FloatingActionButton(onClick = { showAdd = true }) {
                Icon(Icons.Filled.Add, contentDescription = "Add task")
            }
        },
    ) { padding ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(16.dp)
                .padding(padding),
        ) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Text("Tasks", style = MaterialTheme.typography.headlineMedium, fontWeight = FontWeight.SemiBold)
                Spacer(Modifier.weight(1f))
                val doneCount = state.tasks.count { it.status == "done" }
                if (doneCount > 0 && state.statusFilter != "pending") {
                    TextButton(onClick = { showClearConfirm = true }) {
                        Text("Clear $doneCount done", color = MaterialTheme.colorScheme.error)
                    }
                }
            }
            Text(
                "${state.tasks.size} task${if (state.tasks.size == 1) "" else "s"}",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onBackground.copy(alpha = 0.6f),
            )
            Spacer(modifier = Modifier.height(8.dp))

            // Status filter chips
            Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                AssistChip(
                    onClick = { viewModel.setStatusFilter("all") },
                    label = { Text("All") },
                    colors = if (state.statusFilter == "all")
                        AssistChipDefaults.assistChipColors(containerColor = MaterialTheme.colorScheme.primary, labelColor = MaterialTheme.colorScheme.onPrimary)
                    else AssistChipDefaults.assistChipColors(),
                )
                AssistChip(
                    onClick = { viewModel.setStatusFilter("pending") },
                    label = { Text("Pending") },
                    colors = if (state.statusFilter == "pending")
                        AssistChipDefaults.assistChipColors(containerColor = MaterialTheme.colorScheme.primary, labelColor = MaterialTheme.colorScheme.onPrimary)
                    else AssistChipDefaults.assistChipColors(),
                )
                AssistChip(
                    onClick = { viewModel.setStatusFilter("done") },
                    label = { Text("Done") },
                    colors = if (state.statusFilter == "done")
                        AssistChipDefaults.assistChipColors(containerColor = MaterialTheme.colorScheme.primary, labelColor = MaterialTheme.colorScheme.onPrimary)
                    else AssistChipDefaults.assistChipColors(),
                )
            }
            Spacer(modifier = Modifier.height(8.dp))

            if (state.loading) {
                Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) { CircularProgressIndicator() }
            } else {
                LazyColumn(
                    contentPadding = PaddingValues(vertical = 8.dp),
                    verticalArrangement = Arrangement.spacedBy(8.dp),
                ) {
                    if (state.reminders.isNotEmpty() && state.statusFilter != "done") {
                        item { RemindersHeader() }
                        items(state.reminders, key = { "reminder-${it.id}" }) { reminder ->
                            ReminderRow(
                                reminder = reminder,
                                onCancel = { viewModel.cancelReminder(reminder.id) },
                            )
                        }
                        item { Spacer(Modifier.height(8.dp)); HorizontalDivider(); Spacer(Modifier.height(8.dp)) }
                    }

                    if (state.tasks.isEmpty()) {
                        item {
                            Box(Modifier.fillMaxWidth().padding(vertical = 32.dp), contentAlignment = Alignment.Center) {
                                Text(
                                    if (state.statusFilter == "done") "No completed tasks"
                                    else if (state.statusFilter == "pending") "No pending tasks"
                                    else "No tasks yet",
                                    style = MaterialTheme.typography.bodyLarge,
                                    color = MaterialTheme.colorScheme.onBackground.copy(alpha = 0.5f),
                                )
                            }
                        }
                    } else {
                        items(state.tasks, key = { it.id }) { task ->
                            TaskRow(
                                task = task,
                                onDelete = { viewModel.deleteTask(task.id) },
                                onDone = { viewModel.markDone(task.id) },
                                onReopen = { viewModel.reopenTask(task.id) },
                                onEdit = { editingTask = task },
                            )
                        }
                    }
                }
            }
        }
    }

    if (showAdd) {
        AddTaskDialog(
            onDismiss = { showAdd = false },
            onAdd = { title, description, dueAt, priority, tags ->
                viewModel.addTask(title, description, dueAt, priority, tags)
                showAdd = false
            },
        )
    }

    editingTask?.let { task ->
        EditTaskDialog(
            task = task,
            onDismiss = { editingTask = null },
            onSave = { title, description, dueAt, priority, tags ->
                viewModel.updateTask(task.id, title, description, dueAt, priority, tags)
                editingTask = null
            },
        )
    }

    if (showClearConfirm) {
        AlertDialog(
            onDismissRequest = { showClearConfirm = false },
            title = { Text("Clear completed tasks?") },
            text = { Text("This will permanently delete all completed tasks. This cannot be undone.") },
            confirmButton = {
                TextButton(onClick = {
                    showClearConfirm = false
                    viewModel.clearCompleted()
                }) { Text("Delete", color = MaterialTheme.colorScheme.error) }
            },
            dismissButton = {
                TextButton(onClick = { showClearConfirm = false }) { Text("Cancel") }
            },
        )
    }
}

@Composable
private fun RemindersHeader() {
    Text(
        "Upcoming reminders",
        style = MaterialTheme.typography.titleSmall,
        fontWeight = FontWeight.SemiBold,
        color = MaterialTheme.colorScheme.primary,
    )
}

@Composable
private fun ReminderRow(
    reminder: ReminderEntity,
    onCancel: () -> Unit,
) {
    val fmt = SimpleDateFormat("MMM d, HH:mm", Locale.US)
    Surface(
        color = MaterialTheme.colorScheme.secondaryContainer,
        shape = RoundedCornerShape(10.dp),
        modifier = Modifier.fillMaxWidth(),
    ) {
        Row(
            modifier = Modifier.padding(12.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Column(Modifier.weight(1f)) {
                Text(
                    reminder.message,
                    style = MaterialTheme.typography.bodyMedium,
                    maxLines = 2,
                    overflow = TextOverflow.Ellipsis,
                )
                Spacer(Modifier.height(2.dp))
                Text(
                    fmt.format(Date(reminder.triggerAt)),
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.7f),
                )
            }
            IconButton(onClick = onCancel) {
                Icon(
                    imageVector = Icons.Filled.Delete,
                    contentDescription = "Cancel reminder",
                    tint = MaterialTheme.colorScheme.error.copy(alpha = 0.7f),
                )
            }
        }
    }
}

@Composable
private fun TaskRow(
    task: TaskEntity,
    onDelete: () -> Unit,
    onDone: () -> Unit,
    onReopen: () -> Unit,
    onEdit: () -> Unit,
) {
    val fmt = SimpleDateFormat("MMM d, HH:mm", Locale.US)
    Surface(
        color = MaterialTheme.colorScheme.surfaceVariant,
        shape = RoundedCornerShape(10.dp),
        modifier = Modifier.fillMaxWidth(),
    ) {
        Row(
            modifier = Modifier.padding(12.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Column(Modifier.weight(1f)) {
                Text(
                    task.title,
                    style = MaterialTheme.typography.bodyMedium,
                    fontWeight = FontWeight.SemiBold,
                    textDecoration = if (task.status == "done") androidx.compose.ui.text.style.TextDecoration.LineThrough else null,
                )
                if (task.description.isNotBlank()) {
                    Spacer(Modifier.height(2.dp))
                    Text(
                        task.description,
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.7f),
                        maxLines = 2,
                        overflow = TextOverflow.Ellipsis,
                    )
                }
                Spacer(Modifier.height(4.dp))
                Row(verticalAlignment = Alignment.CenterVertically) {
                    if (task.dueAt != null) {
                        val due = task.dueAt!!
                        Text(
                            "Due ${fmt.format(Date(due))}",
                            style = MaterialTheme.typography.labelSmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.7f),
                        )
                        Spacer(Modifier.width(8.dp))
                    }
                    PriorityChip(task.priority)
                    if (task.tags.isNotBlank()) {
                        Spacer(Modifier.width(8.dp))
                        Text(
                            task.tags,
                            style = MaterialTheme.typography.labelSmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.6f),
                            maxLines = 1,
                            overflow = TextOverflow.Ellipsis,
                        )
                    }
                }
            }
            if (task.status == "done") {
                TextButton(onClick = onReopen) { Text("Reopen") }
            } else {
                TextButton(onClick = onDone) { Text("Done") }
            }
            IconButton(onClick = onEdit) {
                Icon(
                    imageVector = Icons.Filled.Edit,
                    contentDescription = "Edit",
                    tint = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.6f),
                )
            }
            IconButton(onClick = onDelete) {
                Icon(
                    imageVector = Icons.Filled.Delete,
                    contentDescription = "Delete",
                    tint = MaterialTheme.colorScheme.error.copy(alpha = 0.6f),
                )
            }
        }
    }
}

@Composable
private fun PriorityChip(priority: Int) {
    val label = when (priority) {
        3 -> "High"
        2 -> "Medium"
        1 -> "Low"
        else -> "None"
    }
    InputChip(
        selected = false,
        onClick = { },
        label = { Text(label, style = MaterialTheme.typography.labelSmall) },
        modifier = Modifier.height(24.dp),
    )
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun AddTaskDialog(
    onDismiss: () -> Unit,
    onAdd: (title: String, description: String, dueAt: Long?, priority: Int, tags: String) -> Unit,
) {
    TaskEditFields(
        title = "",
        description = "",
        dueAt = null,
        priority = 0,
        tags = "",
        dialogTitle = "Add task",
        confirmLabel = "Add",
        onDismiss = onDismiss,
        onConfirm = onAdd,
    )
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun EditTaskDialog(
    task: TaskEntity,
    onDismiss: () -> Unit,
    onSave: (title: String, description: String, dueAt: Long?, priority: Int, tags: String) -> Unit,
) {
    TaskEditFields(
        title = task.title,
        description = task.description,
        dueAt = task.dueAt,
        priority = task.priority,
        tags = task.tags,
        dialogTitle = "Edit task",
        confirmLabel = "Save",
        onDismiss = onDismiss,
        onConfirm = onSave,
    )
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun TaskEditFields(
    title: String,
    description: String,
    dueAt: Long?,
    priority: Int,
    tags: String,
    dialogTitle: String,
    confirmLabel: String,
    onDismiss: () -> Unit,
    onConfirm: (title: String, description: String, dueAt: Long?, priority: Int, tags: String) -> Unit,
) {
    var titleState by remember { mutableStateOf(title) }
    var descriptionState by remember { mutableStateOf(description) }
    var priorityState by remember { mutableIntStateOf(priority) }
    var tagsState by remember { mutableStateOf(tags) }
    var showDatePicker by remember { mutableStateOf(false) }
    var showTimePicker by remember { mutableStateOf(false) }
    var selectedDateMs by remember { mutableStateOf(dueAt) }

    val calendar = remember { Calendar.getInstance() }
    if (selectedDateMs != null) calendar.timeInMillis = selectedDateMs!!

    fun dueAtMs(): Long? {
        if (selectedDateMs == null) return null
        val c = Calendar.getInstance().apply { timeInMillis = selectedDateMs!! }
        c.set(Calendar.HOUR_OF_DAY, calendar.get(Calendar.HOUR_OF_DAY))
        c.set(Calendar.MINUTE, calendar.get(Calendar.MINUTE))
        c.set(Calendar.SECOND, 0)
        c.set(Calendar.MILLISECOND, 0)
        return c.timeInMillis
    }

    val dateText = selectedDateMs?.let { ms ->
        SimpleDateFormat("MMM d, HH:mm", Locale.US).format(Date(dueAtMs() ?: ms))
    } ?: "No due date"

    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text(dialogTitle) },
        text = {
            Column {
                OutlinedTextField(
                    value = titleState,
                    onValueChange = { titleState = it },
                    label = { Text("Title") },
                    singleLine = true,
                    modifier = Modifier.fillMaxWidth(),
                )
                Spacer(Modifier.height(8.dp))
                OutlinedTextField(
                    value = descriptionState,
                    onValueChange = { descriptionState = it },
                    label = { Text("Description") },
                    minLines = 2,
                    maxLines = 4,
                    modifier = Modifier.fillMaxWidth(),
                )
                Spacer(Modifier.height(8.dp))
                OutlinedButton(
                    onClick = { showDatePicker = true },
                    modifier = Modifier.fillMaxWidth(),
                ) {
                    Text(dateText)
                }
                Spacer(Modifier.height(8.dp))
                Text("Priority", style = MaterialTheme.typography.labelMedium)
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    PriorityOption(label = "None", selected = priorityState == 0) { priorityState = 0 }
                    PriorityOption(label = "Low", selected = priorityState == 1) { priorityState = 1 }
                    PriorityOption(label = "Medium", selected = priorityState == 2) { priorityState = 2 }
                    PriorityOption(label = "High", selected = priorityState == 3) { priorityState = 3 }
                }
                Spacer(Modifier.height(8.dp))
                OutlinedTextField(
                    value = tagsState,
                    onValueChange = { tagsState = it },
                    label = { Text("Tags (comma separated)") },
                    singleLine = true,
                    modifier = Modifier.fillMaxWidth(),
                )
            }
        },
        confirmButton = {
            TextButton(
                onClick = {
                    onConfirm(titleState, descriptionState, dueAtMs(), priorityState, tagsState)
                },
                enabled = titleState.isNotBlank(),
            ) { Text(confirmLabel) }
        },
        dismissButton = {
            TextButton(onClick = onDismiss) { Text("Cancel") }
        },
    )

    if (showDatePicker) {
        val dateState = rememberDatePickerState(initialDisplayMode = DisplayMode.Picker)
        DatePickerDialog(
            onDismissRequest = { showDatePicker = false },
            confirmButton = {
                TextButton(onClick = {
                    dateState.selectedDateMillis?.let { selectedDateMs = it }
                    showDatePicker = false
                    showTimePicker = true
                }) { Text("OK") }
            },
            dismissButton = {
                TextButton(onClick = { showDatePicker = false }) { Text("Cancel") }
            },
        ) {
            DatePicker(state = dateState)
        }
    }

    if (showTimePicker) {
        val timeState = rememberTimePickerState()
        AlertDialog(
            onDismissRequest = { showTimePicker = false },
            title = { Text("Due time") },
            text = { TimePicker(state = timeState) },
            confirmButton = {
                TextButton(onClick = {
                    calendar.set(Calendar.HOUR_OF_DAY, timeState.hour)
                    calendar.set(Calendar.MINUTE, timeState.minute)
                    showTimePicker = false
                }) { Text("OK") }
            },
            dismissButton = {
                TextButton(onClick = { showTimePicker = false }) { Text("Cancel") }
            },
        )
    }
}

@Composable
private fun PriorityOption(label: String, selected: Boolean, onClick: () -> Unit) {
    Card(
        onClick = onClick,
        colors = CardDefaults.cardColors(
            containerColor = if (selected) MaterialTheme.colorScheme.primaryContainer else MaterialTheme.colorScheme.surfaceVariant,
        ),
    ) {
        Box(Modifier.padding(horizontal = 12.dp, vertical = 6.dp)) {
            Text(label, style = MaterialTheme.typography.labelMedium)
        }
    }
}