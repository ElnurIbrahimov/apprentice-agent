package com.aura.ui.screens

import androidx.compose.foundation.layout.*; import androidx.compose.foundation.lazy.LazyColumn; import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape; import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Add; import androidx.compose.material.icons.filled.Build; import androidx.compose.material.icons.filled.Delete; import androidx.compose.material.icons.filled.PlayArrow
import androidx.compose.material3.*; import androidx.compose.runtime.*; import androidx.compose.ui.Alignment; import androidx.compose.ui.Modifier; import androidx.compose.ui.text.font.FontWeight; import androidx.compose.ui.unit.dp
import androidx.hilt.navigation.compose.hiltViewModel; import com.aura.hands.Hand; import com.aura.ui.viewmodel.HandsViewModel
import java.text.SimpleDateFormat; import java.util.*

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun HandsScreen(viewModel: HandsViewModel = hiltViewModel()) {
    val state by viewModel.state.collectAsState()
    var showAddDialog by remember { mutableStateOf(false) }

    Scaffold(
        floatingActionButton = {
            ExtendedFloatingActionButton(
                onClick = { showAddDialog = true },
                icon = { Icon(Icons.Filled.Add, null) },
                text = { Text("Add hand") },
            )
        },
    ) { padding ->
        Column(Modifier.fillMaxSize().padding(padding).padding(16.dp)) {
            Text("Hands", style = MaterialTheme.typography.headlineMedium, fontWeight = FontWeight.SemiBold)
            Text("Automation macros. ${state.hands.size} configured.", style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onBackground.copy(alpha = 0.6f))
            Spacer(Modifier.height(12.dp)); HorizontalDivider(); Spacer(Modifier.height(8.dp))

            // Last result banner
            state.lastResult?.let { msg ->
                Surface(color = MaterialTheme.colorScheme.secondaryContainer, shape = RoundedCornerShape(8.dp), modifier = Modifier.fillMaxWidth().padding(bottom = 8.dp)) {
                    Text(msg, Modifier.padding(12.dp), style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSecondaryContainer)
                }
            }

            if (state.loading) Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) { CircularProgressIndicator() }
            else if (state.hands.isEmpty()) Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) { Column(horizontalAlignment = Alignment.CenterHorizontally) { Text("🤖", style = MaterialTheme.typography.displayLarge); Spacer(Modifier.height(8.dp)); Text("No hands yet", style = MaterialTheme.typography.titleMedium); Text("Tap + to add one, or ask Aura to set one up.", style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onBackground.copy(alpha = 0.6f)) } }
            else LazyColumn(contentPadding = PaddingValues(vertical = 8.dp).let { PaddingValues(top = 8.dp, bottom = 80.dp) }, verticalArrangement = Arrangement.spacedBy(8.dp)) {
                items(state.hands, key = { it.id }) { h -> HandRow(h, isRunning = state.running == h.name, onRun = { viewModel.runHand(h) }, onToggle = { viewModel.toggle(h) }, onDelete = { viewModel.delete(h.name) }) }
            }
        }
    }

    if (showAddDialog) {
        AddHandDialog(
            onDismiss = { showAddDialog = false },
            onSave = { name, trigger, steps ->
                viewModel.add(name, trigger, steps)
                showAddDialog = false
            },
        )
    }
}

@Composable
private fun HandRow(hand: Hand, isRunning: Boolean, onRun: () -> Unit, onToggle: () -> Unit, onDelete: () -> Unit) {
    val fmt = SimpleDateFormat("MMM d", Locale.US)
    Surface(color = MaterialTheme.colorScheme.surfaceVariant, shape = RoundedCornerShape(10.dp), modifier = Modifier.fillMaxWidth()) {
        Row(Modifier.padding(12.dp), verticalAlignment = Alignment.CenterVertically) {
            Icon(Icons.Filled.Build, null, tint = if (hand.enabled) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.4f), modifier = Modifier.size(24.dp))
            Spacer(Modifier.width(12.dp))
            Column(Modifier.weight(1f)) { Text(hand.name, style = MaterialTheme.typography.bodyMedium, fontWeight = FontWeight.SemiBold); Text(hand.triggerPhrase.ifBlank { "no trigger phrase" }, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.7f)) }
            Switch(checked = hand.enabled, onCheckedChange = { onToggle() })
            Spacer(Modifier.width(4.dp))
            FilledIconButton(onClick = onRun, enabled = hand.enabled && !isRunning) { Icon(Icons.Filled.PlayArrow, "Run", tint = MaterialTheme.colorScheme.onPrimary) }
            Spacer(Modifier.width(4.dp))
            IconButton(onClick = onDelete, modifier = Modifier.size(32.dp)) { Icon(Icons.Filled.Delete, "Delete", tint = MaterialTheme.colorScheme.error.copy(alpha = 0.6f), modifier = Modifier.size(18.dp)) }
        }
    }
}

@Composable
private fun AddHandDialog(onDismiss: () -> Unit, onSave: (String, String, String) -> Unit) {
    var name by remember { mutableStateOf("") }
    var trigger by remember { mutableStateOf("") }
    var steps by remember { mutableStateOf("[]") }

    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text("New hand") },
        text = {
            Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                OutlinedTextField(value = name, onValueChange = { name = it }, label = { Text("Name") }, singleLine = true)
                OutlinedTextField(value = trigger, onValueChange = { trigger = it }, label = { Text("Trigger phrase (optional)") }, singleLine = true, supportingText = { Text("What the user says to trigger it") })
                OutlinedTextField(value = steps, onValueChange = { steps = it }, label = { Text("Steps (JSON)") }, supportingText = { Text("""e.g. [{"tool":"get_current_time","args":{}}]""") }, minLines = 3)
            }
        },
        confirmButton = {
            TextButton(
                enabled = name.isNotBlank() && steps.isNotBlank(),
                onClick = { onSave(name.trim(), trigger.trim(), steps) },
            ) { Text("Save") }
        },
        dismissButton = { TextButton(onClick = onDismiss) { Text("Cancel") } },
    )
}
