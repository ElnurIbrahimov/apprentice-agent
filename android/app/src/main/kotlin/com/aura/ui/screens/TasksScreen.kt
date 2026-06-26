package com.aura.ui.screens

import androidx.compose.foundation.layout.*; import androidx.compose.foundation.lazy.LazyColumn; import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape; import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Add; import androidx.compose.material.icons.filled.CheckCircle; import androidx.compose.material.icons.filled.Delete; import androidx.compose.material.icons.filled.RadioButtonUnchecked
import androidx.compose.material3.*; import androidx.compose.runtime.*; import androidx.compose.ui.Alignment; import androidx.compose.ui.Modifier; import androidx.compose.ui.text.font.FontWeight; import androidx.compose.ui.text.style.TextDecoration; import androidx.compose.ui.unit.dp
import androidx.hilt.navigation.compose.hiltViewModel; import com.aura.tasks.TaskEntity; import com.aura.ui.viewmodel.TasksViewModel
import java.text.SimpleDateFormat; import java.util.*

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun TasksScreen(viewModel: TasksViewModel = hiltViewModel()) {
    val state by viewModel.state.collectAsState()
    var showAddDialog by remember { mutableStateOf(false) }

    Scaffold(
        floatingActionButton = {
            ExtendedFloatingActionButton(
                onClick = { showAddDialog = true },
                icon = { Icon(Icons.Filled.Add, null) },
                text = { Text("Add task") },
            )
        },
    ) { padding ->
        Column(Modifier.fillMaxSize().padding(padding).padding(16.dp)) {
            Text("Tasks & Reminders", style = MaterialTheme.typography.headlineMedium, fontWeight = FontWeight.SemiBold)
            val pending = state.tasks.count { it.status == "pending" }
            val done = state.tasks.size - pending
            Text("$pending pending · $done done", style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onBackground.copy(alpha = 0.6f))
            Spacer(Modifier.height(12.dp)); HorizontalDivider(); Spacer(Modifier.height(8.dp))
            if (state.loading) Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) { CircularProgressIndicator() }
            else if (state.tasks.isEmpty()) Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) { Column(horizontalAlignment = Alignment.CenterHorizontally) { Text("📋", style = MaterialTheme.typography.displayLarge); Spacer(Modifier.height(8.dp)); Text("No tasks yet", style = MaterialTheme.typography.titleMedium); Text("Tap + to add one, or ask Aura to set a reminder.", style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onBackground.copy(alpha = 0.6f)) } }
            else LazyColumn(contentPadding = PaddingValues(vertical = 8.dp).let { PaddingValues(top = 8.dp, bottom = 80.dp) }, verticalArrangement = Arrangement.spacedBy(8.dp)) {
                items(state.tasks, key = { it.id }) { t -> TaskRow(t, onMarkDone = { viewModel.markDone(t.id) }, onDelete = { viewModel.delete(t.id) }) }
            }
        }
    }

    if (showAddDialog) {
        AddTaskDialog(onDismiss = { showAddDialog = false }, onSave = { title -> viewModel.add(title); showAddDialog = false })
    }
}

@Composable
private fun TaskRow(task: TaskEntity, onMarkDone: () -> Unit, onDelete: () -> Unit) {
    val fmt = SimpleDateFormat("MMM d, HH:mm", Locale.US)
    val isDone = task.status == "done"
    Surface(color = MaterialTheme.colorScheme.surfaceVariant, shape = RoundedCornerShape(10.dp), modifier = Modifier.fillMaxWidth()) {
        Row(Modifier.padding(12.dp), verticalAlignment = Alignment.CenterVertically) {
            IconButton(onClick = onMarkDone, enabled = !isDone) {
                Icon(if (isDone) Icons.Filled.CheckCircle else Icons.Filled.RadioButtonUnchecked, null, tint = if (isDone) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.6f))
            }
            Column(Modifier.weight(1f)) {
                Text(task.title, style = MaterialTheme.typography.bodyMedium, fontWeight = FontWeight.SemiBold, textDecoration = if (isDone) TextDecoration.LineThrough else TextDecoration.None, color = if (isDone) MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.5f) else MaterialTheme.colorScheme.onSurfaceVariant)
                Spacer(Modifier.height(2.dp))
                Text(fmt.format(Date(task.createdAt)), style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.4f))
            }
            IconButton(onClick = onDelete, modifier = Modifier.size(32.dp)) { Icon(Icons.Filled.Delete, "Delete", tint = MaterialTheme.colorScheme.error.copy(alpha = 0.5f), modifier = Modifier.size(18.dp)) }
        }
    }
}

@Composable
private fun AddTaskDialog(onDismiss: () -> Unit, onSave: (String) -> Unit) {
    var title by remember { mutableStateOf("") }
    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text("New task") },
        text = { OutlinedTextField(value = title, onValueChange = { title = it }, label = { Text("Title") }, singleLine = true) },
        confirmButton = { TextButton(enabled = title.isNotBlank(), onClick = { onSave(title) }) { Text("Save") } },
        dismissButton = { TextButton(onClick = onDismiss) { Text("Cancel") } },
    )
}
