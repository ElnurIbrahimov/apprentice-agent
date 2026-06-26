package com.aura.ui.screens

import androidx.compose.foundation.layout.*; import androidx.compose.foundation.lazy.LazyColumn; import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape; import androidx.compose.material.icons.Icons; import androidx.compose.material.icons.filled.Delete; import androidx.compose.material.icons.filled.TaskAlt
import androidx.compose.material3.*; import androidx.compose.runtime.*; import androidx.compose.ui.Alignment; import androidx.compose.ui.Modifier; import androidx.compose.ui.text.font.FontWeight; import androidx.compose.ui.unit.dp
import androidx.hilt.navigation.compose.hiltViewModel; import com.aura.tasks.TaskEntity; import com.aura.ui.viewmodel.TasksViewModel
import java.text.SimpleDateFormat; import java.util.*

@Composable
fun TasksScreen(viewModel: TasksViewModel = hiltViewModel()) {
    val state by viewModel.state.collectAsState()
    Column(Modifier.fillMaxSize().padding(16.dp)) {
        Text("Tasks & Reminders", style = MaterialTheme.typography.headlineMedium, fontWeight = FontWeight.SemiBold)
        Text("${state.tasks.size} total", style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onBackground.copy(alpha = 0.6f))
        Spacer(Modifier.height(12.dp)); HorizontalDivider(); Spacer(Modifier.height(8.dp))
        if (state.loading) Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) { CircularProgressIndicator() }
        else if (state.tasks.isEmpty()) Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) { Column(horizontalAlignment = Alignment.CenterHorizontally) { Text("📋", style = MaterialTheme.typography.displayLarge); Spacer(Modifier.height(8.dp)); Text("No tasks yet", style = MaterialTheme.typography.titleMedium); Text("Ask Aura to set a reminder.", style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onBackground.copy(alpha = 0.6f)) } }
        else LazyColumn(contentPadding = PaddingValues(vertical = 8.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) { items(state.tasks, key = { it.id }) { t -> TaskRow(t, onDelete = { viewModel.delete(t.id) }) } }
    }
}

@Composable
private fun TaskRow(task: TaskEntity, onDelete: () -> Unit) {
    val fmt = SimpleDateFormat("MMM d, HH:mm", Locale.US)
    Surface(color = MaterialTheme.colorScheme.surfaceVariant, shape = RoundedCornerShape(10.dp), modifier = Modifier.fillMaxWidth()) {
        Row(Modifier.padding(12.dp), verticalAlignment = Alignment.CenterVertically) {
            Icon(Icons.Filled.TaskAlt, null, tint = if (task.status == "done") MaterialTheme.colorScheme.primary.copy(alpha = 0.5f) else MaterialTheme.colorScheme.primary, modifier = Modifier.size(24.dp))
            Spacer(Modifier.width(12.dp))
            Column(Modifier.weight(1f)) { Text(task.title, style = MaterialTheme.typography.bodyMedium, fontWeight = FontWeight.SemiBold); Text(task.status, style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.6f)); Spacer(Modifier.height(2.dp)); Text(fmt.format(Date(task.createdAt)), style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.4f)) }
            IconButton(onClick = onDelete, modifier = Modifier.size(32.dp)) { Icon(Icons.Filled.Delete, "Delete", tint = MaterialTheme.colorScheme.error.copy(alpha = 0.5f), modifier = Modifier.size(18.dp)) }
        }
    }
}
